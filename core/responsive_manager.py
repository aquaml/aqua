from datastructures.responsive_tensor import responsive_tensor, tensor_device
from policies.aqua_policy import aqua_policy, tensor_home
from policies.static_policy import static_policy
from typing import List
import torch

class responsive_manager:
    _policy: aqua_policy = None

    @classmethod
    def to_responsive_tensor(cls, torch_tensor: torch.Tensor, id: int = -1) -> responsive_tensor:
        r_tensor = responsive_tensor(torch_tensor, id=id)
        
        if cls._policy == None:
            cls._policy = static_policy(80)
        
        rt_home = cls._policy.add_rtensor(r_tensor)

        if rt_home.device_type == tensor_device.GPU:
            r_tensor._move_to_gpu(rt_home.device_address)
        torch.cuda.empty_cache()
        return r_tensor

    @classmethod
    def to_responsive_tensors(cls, torch_tensors: List[torch.Tensor], ids: List[int] = None) -> List[responsive_tensor]:
        rts: List[responsive_tensor] = []
        if cls._policy == None:
            cls._policy = static_policy(80)
        
        for idx, torch_tensor in enumerate(torch_tensors):
            id = -1
            if ids != None:
                id = ids[idx]
            rts.append(responsive_tensor(torch_tensor, id=id))
        
        tensor_homes: List[tensor_home] = cls._policy.add_rtensors(rts)
        assert len(tensor_homes) == len(rts)

        for rt, rt_home in zip(rts, tensor_homes):
            if rt_home.device_type == tensor_device.GPU:
                rt._move_to_gpu(rt_home.device_address)
        torch.cuda.empty_cache()
        return rts
        

    @classmethod
    def respond(cls):
        rt_mapping = cls._policy.get_rtensors_to_move()
        for rt in rt_mapping:
            rt_home = rt_mapping[rt]
            if rt_home.device_type == tensor_device.DRAM:
                rt._move_to_dram()
            elif rt_home.device_type == tensor_device.GPU:
                rt._move_to_gpu(rt_home.device_address)
        if len(rt_mapping.keys()) > 0:
            torch.cuda.empty_cache()
        cls._policy.done_moving_rtensors(rt_mapping)
    
    @classmethod
    def set_policy(cls, policy: aqua_policy):
        assert cls._policy == None
        cls._policy = policy