import requests
import json
from typing import List
import utils.virtual_gid
import torch

class MemoryManagerClient:
    def __init__(self, host: str, port: int):
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()

    def _make_request(self, method: str, endpoint: str, payload: dict) -> dict:
        url = f"{self.base_url}/{endpoint}"
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
            'source-gpu': str(utils.virtual_gid._virtual_to_real_gid(torch.cuda.current_device()))
        }
        
        try:
            response = self.session.request(method, url, headers=headers, data=json.dumps(payload))
            response_data = json.loads(response.content.decode())
            return response_data
        except Exception as e:
            return {"error while making request": "{}".format(e)}

    def offer_memory(self, id_val, address_val, size_val):
        endpoint = "lease"
        payload = {"id": id_val, "address": address_val, "size": size_val}
        return self._make_request("POST", endpoint, payload)
    
    def add_memory(self, id_val, address_val, size_val):
        endpoint = "lease"
        payload = {"id": id_val, "address": address_val, "size": size_val}
        return self._make_request("PUT", endpoint, payload)

    def take_back_memory(self, store_id: int) -> None:
        endpoint = "unlease"
        payload = {"id": store_id}
        return self._make_request("DELETE", endpoint, payload)

    def malloc_nv_memory(self, memory: int) -> None:
        endpoint = "nv_allocate"
        payload = {"memory": memory}
        return self._make_request("POST", endpoint, payload)

    def free_nv_memory(self, allocation_id: str) -> None:
        endpoint = "nv_free"
        payload = {"allocation_id": allocation_id}
        return self._make_request("DELETE", endpoint, payload)

    def reclaim_request(self, store_id: int) -> None:
        endpoint = "reclaim_request"
        payload = {"id": store_id}
        return self._make_request("POST", endpoint, payload)
    
    def remove_reclaim_request(self, store_id: int) -> None:
        endpoint = "reclaim_request"
        payload = {"id": store_id}
        return self._make_request("DELETE", endpoint, payload)

    def reclaim_status(self, id_value: int) -> None:
        endpoint = "reclaim_status"
        payload = {"id": id_value}
        return self._make_request("POST", endpoint, payload)

    def responsive_reclaim(self, allocation_ids: list) -> List[str]:
        endpoint = "responsive_reclaim"
        payload = {"allocations": allocation_ids}
        return self._make_request("POST", endpoint, payload)