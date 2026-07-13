import torch
import numpy as np
from model import GPT
import torch.nn.functional as F
from tqdm import tqdm

device_type = "cuda"
device = torch.device('cuda')

def load(path):
    weights = torch.load(path, weights_only=False)['model']
    model = GPT()
    model.load_state_dict(weights)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model = torch.compile(model)
    return model.to(device)

names = [
    'comfy-vortex-5164'
    ]

models = [load(f"log/{n}.pt") for n in names]

with open('5k_enwik9', 'rb') as f:
    data = f.read()
tokens = np.frombuffer(data, dtype=np.uint16)
tokens = torch.tensor(tokens, dtype=torch.uint16, device=device)

scores = []
ctx_len = 1024

for _ in tqdm(range(10000)):
    i = torch.randint(0, len(tokens)-ctx_len, (1,)).item()
    y = tokens[i+ctx_len].long()
    x = tokens[i:i+ctx_len].unsqueeze(0).long()
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        probs = sum(F.softmax(m(x)[:, -1], dim=-1) for m in models) / len(models)
    scores.append(-torch.log(probs[0, y]).item())

t = np.asarray(scores)
mean = t.mean()
se = t.std(ddof=1) / np.sqrt(len(t))
lo, hi = mean - 1.96*se, mean + 1.96*se   # 95%
print(f"{mean:.3f} nats  [{lo:.3f}, {hi:.3f}]")