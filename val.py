import torch
import numpy as np
from model import GPT
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device('cuda')

model = GPT()
model.load_state_dict(torch.load('log/sparkling-serenity-3529.pt', weights_only=False)['model'])
model.to(device)

with open('tokenized_enwik9', 'rb') as f:
    data = f.read()
tokens = np.frombuffer(data, dtype=np.uint16)
tokens = torch.tensor(tokens, dtype=torch.uint16, device=device)

scores = []

with torch.no_grad():
    while True:
        i = torch.randint(0, len(tokens)-1024, (1,)).item()
        y = tokens[i+1024]
        logits, act_norm = model(tokens[i:i+1024].unsqueeze(0).long())
        probs = torch.softmax(logits[:, -1], dim=-1)
        scores.append(-torch.log(probs[0, y]).item())
        print(sum(scores)/len(scores))

