import os
from abc import ABC
from utils import virtual_gid

class abstract_informer(ABC):

    def __init__(self) -> None:
        self.gpu_id = 0

    def _virtual_to_real_gid(self):
        return virtual_gid._virtual_to_real_gid(self.gpu_id)
    
    def get_address(self):
        # virtual gid to real
        return "cuda:{}".format(self._virtual_to_real_gid())