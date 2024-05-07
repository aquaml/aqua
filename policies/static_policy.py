from policies.aqua_policy import aqua_policy, tensor_home
from datastructures.responsive_tensor import responsive_tensor, tensor_device
from typing import Dict, List
from core.logger import init_logger

logger = init_logger(__name__)

class static_policy(aqua_policy):
    def __init__(self, storage_in_gb: int = 20, gpu_device: str = 'cuda:1'):
        self.rtensors = []  
        self.storage_remaining = storage_in_gb * (1024 ** 3)
        self.gpu_device = gpu_device

    def add_rtensor(self, rt: responsive_tensor) -> tensor_home:
        self.rtensors.append(rt)
        rt_size = rt.get_size_in_bytes()

        mapped_device: tensor_device = tensor_device.DRAM
        device_address: str = 'cpu'

        if rt_size <= self.storage_remaining:
            mapped_device = tensor_device.GPU
            device_address = self.gpu_device
            self.storage_remaining -= rt_size
            # logger.info("Storage left after storing rt of size: {} is {}".format((rt_size / (1024 ** 2)), (self.storage_remaining / (1024 ** 2))))
        
        return tensor_home(device_type=mapped_device, device_address=device_address)

    def add_rtensors(self, rts: List[responsive_tensor]) -> List[tensor_home]:
        homes: List[tensor_home] = []
        for rt in rts:
            homes.append(self.add_rtensor(rt))
        return homes

    def get_rtensors_to_move(self) -> Dict[responsive_tensor, tensor_home]:
        rt_mapping: Dict[responsive_tensor, tensor_home] = {}
        for rt in self.rtensors:
            pass
        return rt_mapping

    def done_moving_rtensors(self, mapping: Dict[responsive_tensor, tensor_home]):
        pass