import torch
import numpy as np
from model import GPT
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

model = GPT()
model.load_state_dict(torch.load('log/model_08000.pt', weights_only=False)['model'])

with open('tokenized_enwik9', 'rb') as f:
    data = f.read()
tokens = np.frombuffer(data, dtype=np.uint16)
tokens = torch.tensor(tokens, dtype=torch.uint16)

i = np.random.randint(0, len(tokens)-1024, size=(1,)).item()
x = tokens[i:i+1024].unsqueeze(0).long()
# print(x.flatten().tolist())

x = model.transformer.wte(x)

mask = None
input_pos = None

for layer in model.transformer.h[:6]:
    x = layer(x, input_pos, mask)

# x = model.lm_head(x)

# plt.imshow(x.squeeze().detach().numpy())
# plt.colorbar()
plt.hist(x.flatten().detach().numpy(), 100)
plt.show()