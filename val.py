import torch
import numpy as np
from model import GPT
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device('cuda')

model = GPT()
model.load_state_dict(torch.load('log/model_55500.pt', weights_only=False)['model'])
model.to(device)

with open('30k_enwik9', 'rb') as f:
    data = f.read()
tokens = np.frombuffer(data, dtype=np.uint16)
tokens = torch.tensor(tokens, dtype=torch.long, device=device)

scores = []
k = 0

with torch.no_grad():
    while True:
        i = torch.randint(0, len(tokens)-898, (1,)).item()
        y = tokens[i+898]
        logits, act_norm = model(tokens[i:i+898].unsqueeze(0).long())
        probs = torch.softmax(logits[:, -1], dim=-1)
        scores.append(-torch.log(probs[0, y]).item())
        print(sum(scores)/len(scores), k)
        k = k + 1

# seq_len = 1024
# batch_size = 16

# scores = []

# with torch.no_grad():
#     for _ in tqdm(range(20000)):
#         # sample start indices
#         i = torch.randint(0, len(tokens) - seq_len - 1, (batch_size,))

#         # build index matrix
#         idx = i[:, None] + torch.arange(seq_len)

#         # gather token windows
#         x = tokens[idx].long()
#         y = tokens[idx + 1].long()

#         # forward
#         logits, act_norm = model(x)

#         # compute per-token CE
#         loss = F.cross_entropy(
#             logits.reshape(-1, logits.size(-1)),
#             y.reshape(-1),
#             reduction="none"
#         ).view(batch_size, seq_len)

#         scores.append(loss)

# scores = torch.cat(scores)
# print(scores.shape)

# plt.plot(scores.mean(dim=0).cpu())
# plt.show()