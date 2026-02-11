import torch
from model import GPT
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import lzma
import bz2

device = "cuda"

model = GPT()

# fig, ax = plt.subplots(figsize=(6.4, 3))

# results = []

# cals = list(range(0, 17500, 500))

# for i in range(0, 17500, 500):
# model.load_state_dict(torch.load(f'log/model_{str(i).zfill(5)}.pt', weights_only=False)['model'])
model.load_state_dict(torch.load('log/lemon-durian-3768.pt', weights_only=False)['model'])

# layer = model.transformer.h[1].mlp.c_proj
layer = model.transformer.wte

# qw = layer.qweight()
# w = (qw.round() - qw).detach() + qw  # straight through estimator

# w = 2**layer.e.clip(-8, 8) * w
# bit_mask = layer.z.clip(0, 1)
# w = w * ((bit_mask.round() - bit_mask).detach() + bit_mask)  # straight through estimator
# plt.imshow(w.detach())
# plt.show()
# w = w[w != 0]

# a = lzma.compress(w.detach().numpy())
# print(len(a))

u, s, v = torch.linalg.svd(layer.z.round().detach())

k = 512  # number of singular values to keep

U_k = u[:, :k]          # (m, k)
S_k = s[:k]             # (k,)
Vh_k = v[:k, :]        # (k, n)

A_k = U_k @ torch.diag(S_k) @ Vh_k

plt.imshow(layer.z.round().detach() - A_k.round().clip(0, 1), interpolation='none')
plt.show()


# print(mw.flatten().shape)
# print(torch.nonzero(mw.flatten()).shape)

# plt.imshow((w*layer.z.round()).detach(), interpolation='none')
# plt.show()

# i=0
# print(layer.z[i:i+1].round().int().tolist())
# print(layer.b.flatten()[i])
# print(len(set((w[i:i+1]*layer.z[i:i+1].round()).flatten().tolist())))

# plt.imshow(w.detach())
# plt.colorbar()
# plt.show()

# for i in range(512):
# 	print(2**layer.b1[i].item(), len(set(dw[i, :].flatten().tolist())))

# plt.hist(layer.weight[0, :].detach(), 100, alpha=0.5)
# plt.hist(dw[0, :].detach(), 100, alpha=0.5)
# plt.show()

# plt.hist(w.flatten().detach(), 100)
# plt.show()

# print(torch.mean(layer.b1).item())
# print(w.shape)
# print(w.dtype)
# print(torch.allclose(w, w.to(torch.int8).float()))

# plt.imshow(layer.weight.detach())
# plt.show()

# print(layer.weight.shape)
# a = lzma.compress(layer.weight.data.numpy())
# print(len(a))

# plt.imshow(w.detach())
# plt.hist(w.flatten().int().detach(), 100)
# plt.show()

# print(w.flatten().shape)
# print(torch.bincount(w.int().flatten().abs()).tolist())

# print(np.allclose(w.detach().numpy(), w.detach().numpy().astype(np.int8)))
# np.save("w.npy", w.detach().numpy())

# u, s, v = torch.linalg.svd(dw)

# plt.hist(u.detach().cpu().flatten().clip(-0.1, 0.1), 100)
# plt.show()

# plt.imshow(model.transformer.h[0].mlp.c_proj.weight.detach().cpu())
# plt.colorbar()
# plt.show()

# plt.imshow(u.detach().cpu().clip(-0.1, 0.1))
# plt.show()

# singular_values = s.detach().cpu()

# results.append(torch.stack([torch.arange(512), s.detach().cpu()]).permute(1, 0))
# print(i)

# ax.set_xlim(0, 512)
# ax.set_ylim(0, 80)
# line_collection = LineCollection(results, array=cals, cmap="viridis")
# ax.add_collection(line_collection)
# fig.colorbar(line_collection)

# plt.show()

# 	plt.plot(s.detach().cpu())
# plt.show()

	# plt.imshow(dw.detach().cpu())
	# plt.show()