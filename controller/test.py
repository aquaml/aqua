from controller.mmc import MemoryManagerClient

mmc = MemoryManagerClient('localhost', 8080)

# mmc.reclaim_request(1)
print(mmc.offer_memory(0, 'cuda:0', 0))
print(mmc.add_memory(0, 'cuda:0', 1 * 54240870400))
# data = mmc.reclaim_status(1)
# print(data)
# if data['can_reclaim']:
#     print(mmc.add_memory(1, 'cuda:1', -1 * data['capacity'])) 
#     mmc.remove_reclaim_request(1)
# print(data)
