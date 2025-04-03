import os
from typing import List

def _get_visible_devices() -> List[str]:
    visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
    visible_devices = visible_devices.split(',')
    return visible_devices

def _virtual_to_real_gid(gpu_id: int):
    try:
        visible_devices = _get_visible_devices()
        visible_devices = [int(v) for v in visible_devices]
        return visible_devices[gpu_id]
    except:
        if gpu_id == -1:
            raise Exception("Could not convert a negative virtual gpu id")
        return gpu_id

def _real_to_virtual_gid(device_address: str):
    g_id = device_address.split(':')[1]
    g_id = int(g_id)
    visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
    visible_devices = visible_devices.split(',')
    visible_devices = [int(v) for v in visible_devices]
    for idx in range(len(visible_devices)):
        if visible_devices[idx] == g_id:
            return idx