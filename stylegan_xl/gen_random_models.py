import dnnlib
import torch
import copy
import pickle

# common_kwargs = dict(c_dim=100, img_resolution=32, img_channels=3, )
# G_kwargs = dict(class_name = "training.networks_stylegan3.Generator", z_dim= 64, w_dim= 512, num_layers=10)
# G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False)
#
# snapshot_data = dict(G=G)
# for key, value in snapshot_data.items():
#     if isinstance(value, torch.nn.Module):
#         value = copy.deepcopy(value).eval().requires_grad_(False)
#         snapshot_data[key] = value.cpu()
#     del value # conserve memory
# snapshot_pkl = "random_conditional_256.pkl"
# with open(snapshot_pkl, 'wb') as f:
#     pickle.dump(snapshot_data, f)



common_kwargs = dict(c_dim=0, img_resolution=256, img_channels=3, )
G_kwargs = dict(class_name = "training.networks_stylegan3.Generator", z_dim= 64, w_dim= 512, num_layers=17)
G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False)

snapshot_data = dict(G=G)
for key, value in snapshot_data.items():
    if isinstance(value, torch.nn.Module):
        value = copy.deepcopy(value).eval().requires_grad_(False)
        snapshot_data[key] = value.cpu()
    del value # conserve memory
snapshot_pkl = "random_unconditional_256.pkl"
with open(snapshot_pkl, 'wb') as f:
    pickle.dump(snapshot_data, f)