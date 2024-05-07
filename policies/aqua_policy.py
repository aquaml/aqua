from abc import ABC, abstractmethod
from datastructures.responsive_tensor import responsive_tensor, tensor_device
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class tensor_home:
    device_type: tensor_device
    device_address: str

class aqua_policy(ABC):

    @abstractmethod
    def add_rtensor(self, rt: responsive_tensor) -> tensor_home:
        """
        Abstract method to add a responsive tensor (rt) to the policy.

        Args:
            rt: The responsive tensor to be added.

        Returns:
            tensor_device: if this is GPU, call move_to_gpu on the tensor
        """
        pass

    @abstractmethod
    def add_rtensors(self, rt: List[responsive_tensor]) -> List[tensor_home]:
        """
        Abstract method to add a list of responsive tensors (rt) to the policy.

        Args:
           rt: The responsive tensors to be added.

        Returns:
            tensor_devices: if this is GPU, call move_to_gpu on the tensor
        """
        pass

    @abstractmethod
    def get_rtensors_to_move(self) -> Dict[responsive_tensor, tensor_home]:
        """
        Get a dictionary of responsive tensors and their associated tensor homes.

        Returns:
            Dict[responsive_tensor, tensor_home]: A mapping of responsive tensors to their corresponding tensor homes.
        """
        pass

    @abstractmethod
    def done_moving_rtensors(self, mapping: Dict[responsive_tensor, tensor_home]):
        """
        Inform the policy that tesnsors were moved to DRAM.
        """
        pass