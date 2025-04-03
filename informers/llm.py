from controller.mmc import MemoryManagerClient
from core.logger import init_logger
from typing import List
import time
import os
from informers.base import abstract_informer
from utils import virtual_gid

logger = init_logger(__name__)

class llm_informer(abstract_informer):

    def get_device_address(self, real_gpu_id: int) -> str:
        return "cuda:{}".format(real_gpu_id)
    
    def __init__(self, host: str, port: int, min_memory_to_retian_GB: int, max_memory_GB: int, world_size: int) -> None:
        super().__init__()
        self.mmc = MemoryManagerClient(host, port)
        self.under_reclamation = False
        self.visible_devices = virtual_gid._get_visible_devices()[: world_size]

        for visible_device in self.visible_devices:
            real_gpu_id = int(visible_device)
            logger.info('Creating store for: {}, visible devices: {}, world_size: {}'.format(real_gpu_id, virtual_gid._get_visible_devices(), world_size))
            self.mmc.offer_memory(real_gpu_id, self.get_device_address(real_gpu_id), 0)

        self.min_memory_to_retain = min_memory_to_retian_GB * (1024 ** 3)
        self.max_memory = max_memory_GB * (1024 ** 3)
        self.current_memory = self.max_memory
        self.offering_memory = False
        self.previous_memory_offered = 0
        

    def get_time_in_seconds(self) -> int:
        return int(time.time())

    def handle_reclamation(self) -> int:
        #         {
        # "capacity": 21373334323,
        # "available": 11307004723,
        # "can_reclaim": false
        # }
        reclamation_status = self.mmc.reclaim_status(self._virtual_to_real_gid())
        logger.info("Reclamation status: {}".format(reclamation_status))
        if reclamation_status['can_reclaim'] == True:
            how_much_reclaim = reclamation_status['capacity']
            self.under_reclamation = False
            for visible_device in self.visible_devices:
                real_gpu_id = int(visible_device)
                self.mmc.add_memory(real_gpu_id, self.get_device_address(real_gpu_id), -1 * how_much_reclaim)
                self.mmc.remove_reclaim_request(real_gpu_id)
            return how_much_reclaim
        return 0

    def done_making_space(self):
        assert self.offering_memory
        self.offering_memory = False
        logger.info("Done making space, adding memory now to MMC")
        for visible_device in self.visible_devices:
            real_gpu_id = int(visible_device)
            self.mmc.add_memory(real_gpu_id, self.get_device_address(real_gpu_id), self.previous_memory_offered)

    def maybe_inform_stats(self, pending_queue: int, gpu_cache_used: int) -> int:
        """
        Returns how much to shrink/grow the key value cache by
        """

        if self.under_reclamation:
            return self.handle_reclamation()
        
        elif self.offering_memory:
            return 0
        
        if pending_queue >= 10 and self.current_memory == self.min_memory_to_retain:
            for visible_device in self.visible_devices:
                real_gpu_id = int(visible_device)
                self.mmc.reclaim_request(real_gpu_id)
            logger.info("Issued reclaim request")
            self.under_reclamation = True
            self.current_memory = self.max_memory
            return 0

        elif pending_queue <= 2 and self.current_memory == self.max_memory:
            if gpu_cache_used >= self.min_memory_to_retain:
                logger.info("Cannot reclaim yet because the GPU cache is fully utilized, let's wait. Used: {}, to Retain: {}".format(gpu_cache_used, self.min_memory_to_retain))
                return 0 
            
            memory_to_offer = self.max_memory - self.min_memory_to_retain
            
            self.previous_memory_offered = memory_to_offer
            self.offering_memory = True
            
            logger.info("Offering memory {}".format(memory_to_offer))
            self.current_memory = self.min_memory_to_retain
            return -1 * memory_to_offer
        
            
        return 0
            