import torch
from model import GPT
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

device = "cuda"

model = GPT()

# fig, ax = plt.subplots(figsize=(6.4, 3))

# results = []

# cals = list(range(0, 17500, 500))

# for i in range(0, 17500, 500):
# model.load_state_dict(torch.load(f'log/model_{str(i).zfill(5)}.pt', weights_only=False)['model'])
model.load_state_dict(torch.load('log/model_00000.pt', weights_only=False)['model'])

layer = model.transformer.h[7].mlp.c_proj

qw = layer.qweight()
w = (qw.round() - qw).detach() + qw

dw = 2**layer.e * w

# print(w.shape)

# plt.imshow(w.detach())
# plt.hist(w.flatten().int().detach(), 100)
# plt.show()

# print(torch.mean(layer.b))
# print(w.flatten().shape)
# print(torch.bincount(w.int().flatten().abs()).tolist())

# print(np.allclose(w.detach().numpy(), w.detach().numpy().astype(np.int8)))
# np.save("w.npy", w.detach().numpy().astype(np.int16))

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