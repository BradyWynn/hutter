import torch
import numpy as np
from model import GPT, GPTConfig, Quantized_Linear, Quantized_Embedding
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

def norm(x):
	return F.rms_norm(x, (x.size(-1),))

model = GPT()
model.load_state_dict(torch.load('log/deep-music-4746.pt', weights_only=False)['model'])

total_bits = 0
# layer = model.transformer.wte
for layer in model.modules():
	if isinstance(layer, (Quantized_Linear, Quantized_Embedding)):
		s = (3*layer.weight.std())/(2**(layer.b.clip(0, 16)-1))
		w = layer.weight/s
		w = (w.round() - w).detach() + w  # straight through estimator
		counts, bin_edges = torch.histogram(w, bins=max(1, int(w.max()-w.min())))
		probs = (counts/sum(counts)).detach()
		bpw = 0

		for p in probs:
			if p != 0:
				bpw += -1*p.item()*np.log2(p.item())
		total_bits += (bpw*layer.weight.shape.numel())
print(total_bits/8)

# with open('tokenized_enwik9', 'rb') as f:
#     data = f.read()
# tokens = np.frombuffer(data, dtype=np.uint16)
# tokens = torch.tensor(tokens, dtype=torch.uint16)

# i = np.random.randint(0, len(tokens)-1024, size=(1,)).item()
# x = tokens[i:i+1024].unsqueeze(0).long()
# print(x.flatten().tolist())

# x = model.transformer.wte(x)

# mask = None
# input_pos = None

# for layer in model.transformer.h[:6]:
#     x = layer(x, input_pos, mask)

# x = model.lm_head(x)

# plt.imshow(x.squeeze().detach().numpy())
# plt.colorbar()
# plt.hist(x.flatten().detach().numpy(), 100)
# plt.show()