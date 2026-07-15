import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import GPT
import lzma

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

model = torch.load('log/model_22000.pt', map_location=torch.device('cpu'), weights_only=False)['model']
weight = model['transformer.h.6.mlp.c_fc.weight']
bits = model['transformer.h.6.mlp.c_fc.b']

b = bits.clip(0, 16)
s = (3*weight.std())/(2**(b-1))
s = s.clamp_min(1e-8)
q = weight / s
qmax = 2**(b-1) - 1
qmin = -(2**(b-1))
q_clamped = q.round().clamp(qmin, qmax)
# w = (q_clamped - q).detach() + q   # STE, now with clamped forward value
# w = w * s

print(torch.mean(b))

print(q_clamped.shape)
a = lzma.compress(q_clamped.numpy())
print(len(a))

plt.imshow(q_clamped)
plt.show()

# plt.hist(q_clamped.flatten(), 100)
# plt.show()

# plt.hist(w.flatten(), int(w.max()-w.min()))
# plt.show()

# counts, bin_edges = torch.histogram(w, bins=int(w.max()-w.min()))
# probs = (counts/sum(counts)).detach()
# bpw = 0

# for p in probs:
#     if p != 0:
#         bpw += -1*p.item()*np.log2(p.item())
# print(bpw)
# print(bits[i].item())
# plt.hist(sub_array.flatten(), max(int(sub_array.max()-sub_array.min()), 1))
# plt.show()

# plt.hist(w.flatten(), bins=int(w.max()-w.min()))
# plt.show()

# counts, bin_edges = torch.histogram(w, bins=int(w.max()-w.min()))

# print(int(w.max()-w.min()))
# probs = (counts/sum(counts)).detach()
# bpw = 0

# for p in probs:
#     if p != 0:
#         bpw += -1*p.item()*np.log2(p.item())
# print(bpw)

# plt.hist(weight.flatten(), 100000)
# plt.show()

# print(torch.mean(weight).item())

# plt.imshow(weight, interpolation='none')
# plt.colorbar()
# plt.show()

# print(torch.mean(weight**2))

# plt.hist(weight.flatten(), 100)
# plt.show()

# print(mask.flatten().nonzero().shape[0])
# print(mask.flatten().nonzero().shape)
# z = (mask < 0.0056157955)
# mask[z] = 0.0
# print(mask.flatten().nonzero().shape)
# plt.imshow(mask, interpolation='none')
# plt.show()

# bit_mask = mask.clip(0, 1)
# _, idx = torch.topk(bit_mask.flatten(), int(bit_mask.shape.numel()*0.9), dim=-1, largest=False)
# mask = torch.ones_like(bit_mask.flatten())
# mask.scatter_(-1, idx, 0.0)
# mask = mask.view_as(bit_mask)
# w = weight * bit_mask * mask
# a = lzma.compress(w.numpy())
# b = lzma.compress(weight.numpy())
# print(weight.shape)
# print(len(a))
# print(len(b))

# a = []
# for m in [i for i in list(model.keys()) if i[-1:] == "z"]:
#     a.append(model[m].flatten())
# a = torch.cat(a)
# print(np.quantile(a.numpy(), q=0.5))
# plt.hist(a, 200, log=True)
# plt.show()
# plt.imshow(mask, interpolation='none')
# plt.colorbar()
# plt.show()

# print(mask.shape)
# print(torch.masked_select(layer.flatten(), mask.round().flatten().bool()).shape)

# np.save("w.npy", layer.detach().numpy().astype(np.float16))

# plt.hist(model['transformer.wte.weight'].flatten(), 2**8)
# plt.show()
# for i in model.keys():
#     print(model[i].dtype, i)
#     torch.save(model[i], f"{os.path.join('model', i)}.pt")