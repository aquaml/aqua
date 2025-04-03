import torch
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class tensor_device(Enum):
    DRAM = "DRAM"
    GPU = "GPU"

class responsive_tensor:

    def __init__(self, data: torch.Tensor, id: int = -1):
        self.dram_tensor: torch.Tensor = torch.empty(data.shape, dtype=data.dtype, memory_format=torch.contiguous_format, pin_memory=True)
        self.dram_tensor = self.dram_tensor.copy_(data)
        self.gpu_tensor: torch.Tensor = None
        self.active_device: tensor_device = tensor_device.DRAM
    
    def get_size_in_bytes(self) -> int:
        return self.dram_tensor.numel() * self.dram_tensor.element_size()

    def _move_to_gpu(self, cuda_device: str) -> None:
        try:
            assert self.gpu_tensor == None
            assert self.active_device == tensor_device.DRAM
        except AssertionError as e:
            logger.error("This should never happen, please report assertion error. The tensor was requested to move to GPU when either the gpu tensor was not empty or it was already on GPU. Active device: {}".format(self.active_device))
            raise e
        self.gpu_tensor = self.dram_tensor.to(cuda_device)
        self.active_device = tensor_device.GPU

    def _move_to_dram(self) -> None:
        try:
            assert self.gpu_tensor != None
            assert self.active_device == tensor_device.GPU
        except AssertionError as e:
            logger.error("This should never happen, an empty gpu tensor was asked to move to DRAM or the current active device was not GPU. Active device: {}.".format(self.active_device))
            raise e
        # TODO: DO dirty tracking
        self.dram_tensor.copy_(self.gpu_tensor)
        del self.gpu_tensor            
        self.gpu_tensor = None
        self.active_device = tensor_device.DRAM

    def to_torch_tensor(self) -> torch.Tensor:
        return self.dram_tensor if self.active_device == tensor_device.DRAM else self.gpu_tensor
    