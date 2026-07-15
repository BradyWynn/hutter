import torch
import matplotlib.pyplot as plt
from model import GPT, GPTConfig

device_type = "cpu"
device = torch.device('cpu')

def load(path):
    weights = torch.load(path, weights_only=False, map_location=device)['model']
    model = GPT(GPTConfig(n_layer=12, n_head=4, n_embd=256))
    model.load_state_dict(weights)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model = torch.compile(model)
    return model.to(device)

model1 = load("log/model_00000.pt")
model2 = load("log/model_03000.pt")

print(torch.linalg.norm(model2.transformer.h[0].attn.c_proj.weight).item())
print(torch.linalg.norm(model2.transformer.h[0].attn.c_proj.weight - model1.transformer.h[0].attn.c_proj.weight).item())

# plt.imshow(model2.transformer.h[0].attn.c_proj.weight)
# plt.show()

# plt.imshow(model2.transformer.h[0].attn.c_proj.weight - model1.transformer.h[0].attn.c_proj.weight)
# plt.show()