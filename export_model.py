import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_engine.common.recipe import Format, DelayedScaling, MXFP8BlockScaling, NVFP4BlockScaling
import transformer_engine.pytorch as te
from model import GPT

device = torch.device('cuda')

# model = GPT()
# model.load_state_dict(torch.load('log/model_00000.pt', weights_only=False)['model'])
# model.to(device)

# fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
# fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")

# dummy_input = torch.zeros(1, 1024, dtype=torch.long, device=device)
# with te.autocast(enabled=True, recipe=fp8_recipe):
#     model(dummy_input)
#     for name, module in model.named_modules():
#         if isinstance(module, te.Linear):
#             # print(module._get_weight_tensors())
#             torch.save(module.quantizers['scaling_fwd'][1](module._get_weight_tensors()[0])._data, f"{os.path.join('model', name)}.pt")

model = torch.load('log/morning-pine-3520.pt', map_location=torch.device('cpu'), weights_only=False)['model']
layer = model['transformer.h.6.attn.c_proj.z']
print(torch.mean(layer.round()).item())
# plt.hist(layer.flatten(), 100)
plt.imshow(layer.round(), interpolation='none')
plt.colorbar()
plt.show()

# np.save("w.npy", layer.detach().numpy().astype(np.float16))

# plt.hist(model['transformer.wte.weight'].flatten(), 2**8)
# plt.show()
# for i in model.keys():
#     # if "weight" in i:
#     print(model[i].dtype, i)
#     # bit_mask = model[f"{i[:-6]}z"]
#     # layer = model[i] * ((bit_mask.round() - bit_mask).detach() + bit_mask)
#     # torch.save(layer, f"{os.path.join('model', i)}.pt")
#     torch.save(model[i], f"{os.path.join('model', i)}.pt")