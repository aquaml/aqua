from controller.mmc import MemoryManagerClient
from core.logger import init_logger
from informers.base import abstract_informer

logger = init_logger(__name__)

class static_informer(abstract_informer):

    def __init__(self, host: str, port: int, gpu_id: int) -> None:
        self.mmc = MemoryManagerClient(host, port)
        self.gpu_id = gpu_id
        self.buffer = 5 * (1024 ** 3)
        self.min_offering = 10 * (1024 * 3)
    
    def offer_memory(self, memory_in_bytes: int):
        if memory_in_bytes - self.buffer > self.min_offering:
            memory_in_bytes -= self.buffer
            self.mmc.offer_memory(self._virtual_to_real_gid(), self.get_address(), memory_in_bytes)