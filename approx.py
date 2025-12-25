import torch
from model import GPT
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

device = "cuda"

model = GPT()

fig, ax = plt.subplots(figsize=(6.4, 3))

results = []

cals = list(range(0, 30500, 500))

# for i in range(0, 30500, 500):
model.load_state_dict(torch.load('log/wise-feather-2874.pt', weights_only=False)['model'])

# qw = model.transformer.h[7].attn.c_proj.qweight()
# w = (qw.round() - qw).detach() + qw

# dw = 2**model.transformer.h[7].attn.c_proj.e * w

# u, s, v = torch.linalg.svd(model.transformer.h[0].attn.c_attn.weight)

# plt.hist(u.detach().cpu().flatten().clip(-0.1, 0.1), 100)
# plt.show()

plt.imshow(model.transformer.h[0].mlp.c_proj.weight.detach().cpu())
plt.colorbar()
plt.show()

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