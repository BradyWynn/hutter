import torch
import matplotlib.pyplot as plt
from model import GPT, GPTConfig

device_type = "cpu"
device = torch.device('cpu')

def load(path):
    weights = torch.load(path, weights_only=False, map_location=device)['model']
    model = GPT(GPTConfig(n_layer=1, n_head=4, n_embd=256))
    model.load_state_dict(weights)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model = torch.compile(model)
    return model.to(device)

model = load("log/model_35000.pt")

plt.imshow(model.transformer.h[0].attn.c_attn.weight)
plt.show()