from controller.mmc import MemoryManagerClient
from core.logger import init_logger
from typing import List
import time
import os
from informers.base import abstract_informer

logger = init_logger(__name__)

class llm_informer(abstract_informer):
    def __init__(self, host: str, port: int, window_size_seconds: int, gpu_id: int) -> None:
        self.mmc = MemoryManagerClient(host, port)
        self.window_start_time = 0
        self.seen_free_memory: List[int] = []
        self.seen_pending_queue: List[int] = []
        self.window_size_seconds = window_size_seconds
        self.gpu_id = gpu_id
        self.offering_threshold = 5 * (1024 ** 3) # 1GB at least
        self.reclaim_when = 800 * 1024 * 1024 # When less than 300 MB is remaining
        self.under_reclamation = False
        self.mmc.offer_memory(self._virtual_to_real_gid(), self.get_address(), 0)
        self.min_memory_to_retain =  5 * (1024 ** 3)

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
            self.mmc.add_memory(self._virtual_to_real_gid(), self.get_address(), -1 * how_much_reclaim)
            self.mmc.remove_reclaim_request(self._virtual_to_real_gid())
            return how_much_reclaim
        return 0


    def maybe_inform_stats(self, free_memory: int, pending_queue: int) -> int:
        """
        Returns how much to shrink/grow the key value cache by
        """

        if self.under_reclamation:
            return self.handle_reclamation()

        self.seen_free_memory.append(free_memory)
        self.seen_pending_queue.append(pending_queue)
        
        if self.window_start_time == 0:
            self.window_start_time = self.get_time_in_seconds()
            return 0
        
        if self.get_time_in_seconds() - self.window_start_time >= self.window_size_seconds:
            self.window_start_time = 0
            max_free_memory = max(self.seen_free_memory)
            min_free_memory = min(self.seen_free_memory)
            diff_in_memory = max_free_memory - min_free_memory
            diff_in_memory /= (1024 * 1024) # converting to MB

            # check if the available free memory is constant
            average_pending_size = sum(self.seen_pending_queue) * 1.0 / len(self.seen_pending_queue)
            request_rate = (self.seen_pending_queue[-1] - self.seen_pending_queue[0]) * 1.0 / self.window_size_seconds
            logger.info("Average pending size is: {}, diff in memory is {} MB, min free memory: {}, request rate: {}".format(average_pending_size, diff_in_memory, min_free_memory, request_rate))

            self.seen_free_memory.clear()
            self.seen_pending_queue.clear() 
            
            memory_to_offer = int(min_free_memory)
            memory_to_offer -= self.min_memory_to_retain

            if memory_to_offer > self.offering_threshold and diff_in_memory < 300 and request_rate < 1:    
                if memory_to_offer < 0:
                    return 0
                self.mmc.add_memory(self._virtual_to_real_gid(), self.get_address(), memory_to_offer)
                logger.info("Offering memory {}".format(memory_to_offer))
                return -1 * memory_to_offer
            
            if min_free_memory < self.reclaim_when and average_pending_size > 2 and request_rate > 1.5:
                self.mmc.reclaim_request(self._virtual_to_real_gid())
                logger.info("Issued reclaim request")
                self.under_reclamation = True
                return 0
            
        return 0
            