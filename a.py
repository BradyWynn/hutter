import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import GPT, GPTConfig
from tqdm import tqdm

device_type = "cpu"
device = torch.device('cpu')

def load(path):
    weights = torch.load(path, weights_only=False, map_location=device)['model']
    model = GPT(GPTConfig(n_layer=2, n_head=4, n_embd=256))
    model.load_state_dict(weights)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model = torch.compile(model)
    return model.to(device)

names = [
    'model_10000',
    ]

models = [load(f"log/{n}.pt") for n in names]

plt.imshow(np.arange(256).reshape(1, -1).repeat(256, axis=1) @ models[0].transformer.h[0].attn.c_proj.weight.data.numpy())
plt.colorbar()
plt.show()
