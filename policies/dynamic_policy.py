from policies.aqua_policy import aqua_policy, tensor_home
from datastructures.responsive_tensor import responsive_tensor, tensor_device
from typing import Dict, List
from controller.mmc import MemoryManagerClient
from core.logger import init_logger
import utils.virtual_gid
import time
import traceback

logger = init_logger(__name__)

RESPONSIVENESS_SECONDS = 4

class dynamic_policy(aqua_policy):
    def __init__(self, host: str, port: int):
        self.rtensors: List[responsive_tensor] = []  
        self.allocation_ids_to_rt_tensors: Dict[str, List[responsive_tensor]] = {}
        self.mmc = MemoryManagerClient(host, port)
        self.prev_checked_responsive = 0
        self.previous_purged_allocations = None
    
    def time_in_seconds(self):
        return int(time.time())

    def _real_to_virtual_gid(self, device_address: str):
        try:
            return f"cuda:{utils.virtual_gid._real_to_virtual_gid(device_address)}"
        except:
            return device_address

    def add_rtensor(self, rt: responsive_tensor) -> tensor_home:
        self.rtensors.append(rt)
        rt_size = rt.get_size_in_bytes()

        # {
        #   "size": 100,
        #   "address": "cuda:1",
        #   "store_id": 1,
        #   "allocation_id": "b652a412-4742-49c4-a4f0-4da9cfd0d617"
        # }
        try:
            allocation = self.mmc.malloc_nv_memory(rt_size)
            allocated_size = allocation['size']
            device_address: str = allocation['address']
            allocation_id: str = allocation['allocation_id']
            device_address = self._real_to_virtual_gid(device_address)
            self.allocation_ids_to_rt_tensors[allocation_id] = [rt]
        except Exception as e:
            logger.error("Received an exception while trying to allocate nv memory on mmc, {}".format(e))
            device_address: str = 'cpu'
            allocation_id = 'local'
            allocated_size = -1

        logger.info("MMC responded with allocation of size {}/{} on location {} with id {}".format(allocated_size, rt_size, device_address, allocation_id))

        mapped_device: tensor_device = tensor_device.DRAM if device_address == 'cpu' else tensor_device.GPU
        
        return tensor_home(device_type=mapped_device, device_address=device_address)

    def add_rtensors(self, rts: List[responsive_tensor]) -> List[tensor_home]:
        total_size = 0
        for rt in rts:
            total_size += rt.get_size_in_bytes()
            self.rtensors.append(rt)
        try:
            allocation = self.mmc.malloc_nv_memory(total_size)
            allocated_size = allocation['size']
            assert allocated_size == total_size
            allocation_id: str = allocation['allocation_id']
            device_address: str = allocation['address']
            device_address = self._real_to_virtual_gid(device_address)
            self.allocation_ids_to_rt_tensors[allocation_id] = rts
        except Exception as e:
            logger.error("Received an error while allocating batched tensors, in add_rtensors: {}".format(traceback.format_exc()))
            device_address: str = 'cpu'
            allocation_id = 'local'
        
        tensor_homes: List[tensor_home] = []
        for rt in rts:
            mapped_device: tensor_device = tensor_device.DRAM if device_address == 'cpu' else tensor_device.GPU
            tensor_homes.append(tensor_home(device_type=mapped_device, device_address=device_address))
        return tensor_homes

    def _check_for_purging(self) -> Dict[responsive_tensor, tensor_home]:
        allocation_ids = list(self.allocation_ids_to_rt_tensors.keys())
        allocations_to_purge = self.mmc.responsive_reclaim(allocation_ids)
        # Get tensors to move to CPU
        # logger.info("Purging : {}".format(allocations_to_purge))
        rt_mapping: Dict[responsive_tensor, tensor_home] = {}
        for allocation_id in allocations_to_purge:
            for rt in self.allocation_ids_to_rt_tensors[allocation_id]:
                rt_mapping[rt] = tensor_home(device_type=tensor_device.DRAM, device_address='cpu')
        self.previous_purged_allocations = allocations_to_purge
        return rt_mapping

    def _check_for_expansion(self) -> Dict[responsive_tensor, tensor_home]:
        total_size = 0
        for rt in self.rtensors:
            total_size += rt.get_size_in_bytes()
        try:
            allocation = self.mmc.malloc_nv_memory(total_size)
            if allocation == None:
                return {}
            allocated_size = allocation['size']
            assert allocated_size == total_size
            allocation_id: str = allocation['allocation_id']
            device_address: str = allocation['address']
            device_address = self._real_to_virtual_gid(device_address)
            self.allocation_ids_to_rt_tensors[allocation_id] = [rt for rt in self.rtensors]
            
            rt_mapping: Dict[responsive_tensor, tensor_home] = {}
            for rt in self.rtensors:
                rt_mapping[rt] = tensor_home(device_type=tensor_device.GPU, device_address=device_address)
            return rt_mapping
        except Exception as e:
            logger.error("Allocation response: {}".format(allocation))
            logger.error("Received an error while allocating batched tensors while checking for expansion: {}".format(traceback.format_exc()))
            return {}
    
    def get_rtensors_to_move(self) -> Dict[responsive_tensor, tensor_home]:
        curr_time = self.time_in_seconds()
        if curr_time - self.prev_checked_responsive < RESPONSIVENESS_SECONDS:
            return {}

        rt_mapping = self._check_for_purging()
        if len(rt_mapping) == 0 and len(self.allocation_ids_to_rt_tensors) == 0:
            rt_mapping = self._check_for_expansion()
        
        self.prev_checked_responsive = self.time_in_seconds()
        return rt_mapping
    
    def done_moving_rtensors(self, mapping: Dict[responsive_tensor, tensor_home]):
        # logger.info("Resetting local allocations, length of allocation ids before:  {}".format(len(self.allocation_ids_to_rt_tensors.keys())))
        
        if self.previous_purged_allocations == None:
            return

        for allocation_id in self.previous_purged_allocations:
            self.allocation_ids_to_rt_tensors.pop(allocation_id, None)
    
        # logger.info("Resetting local allocations, length of allocation ids after:  {}".format(len(self.allocation_ids_to_rt_tensors.keys())))

        for allocation_id in self.previous_purged_allocations:
            self.mmc.free_nv_memory(allocation_id)
        self.previous_purged_allocations = None