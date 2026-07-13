import torch
import numpy as np
from model import GPT
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device('cuda')

weights = torch.load('log/model_32500.pt', weights_only=False)['model']

model = GPT()
model.load_state_dict(weights)
model.to(device)

with open('enwik9', 'rb') as f:
    data = f.read()
tokens = np.frombuffer(data, dtype=np.uint8)
tokens = torch.tensor(tokens, dtype=torch.long, device=device)

bit_count = 0
byte_list = []

with torch.no_grad():
    for i in range(len(tokens)-1024):
        y = tokens[i+1024]
        logits = model(tokens[i:i+1024].unsqueeze(0).long())
        probs = torch.softmax(logits[:, -1], dim=-1)
        loss = -torch.log(probs[0, y]).item()
        if bit_count > 100:
            # print(''.join(chr(i) for i in byte_list))
            print(len(byte_list))
            bit_count = 0
            byte_list.clear()
        else:
            bit_count += loss * np.log2(np.e)
            byte_list.append(y)
