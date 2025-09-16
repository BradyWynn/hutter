import os
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

t = 1024
n_embd = 512
head_embd = 64
n_heads = n_embd // head_embd

c_attn = np.load(os.path.join('model','transformer.h.0.attn.c_attn.weight.npy')).T
x = np.ones((t, n_embd)) * 0.01

qkv = x @ c_attn
q = qkv[:, n_embd*0:n_embd*1].reshape(t, n_heads, head_embd)
k = qkv[:, n_embd*1:n_embd*2].reshape(t, n_heads, head_embd)
v = qkv[:, n_embd*2:n_embd*3].reshape(t, n_heads, head_embd)

q = torch.from_numpy(q)#.permute(1, 0, 2)
k = torch.from_numpy(k)#.permute(1, 0, 2)
v = torch.from_numpy(v)#.permute(1, 0, 2)

# eps = 1e-8
# rms = torch.zeros_like(q[..., 0])
# for i in range(q.size(-1)):
# 	rms += q[..., i]**2
# rms = torch.sqrt(rms/q.size(-1) + eps).unsqueeze(-1)
# print(rms.flatten().tolist()[:10])
# q = q / rms

q, k = norm(q), norm(k)

print(q.flatten()[:10].tolist())

# result = q @ k.permute(0, 2, 1)
# result = result / 8.0
# # print(torch.sum(result).item())

# for h in range(result.size(0)):
#     for j in range(result.size(1)):
#         max_val = -10000
#         exp_sum = 0
#         for i in range(j+1):
#             if result[h][j][i] > max_val:
#                 max_val = result[h][j][i]
#         for i in range(j+1):
#             result[h][j][i] = torch.exp(result[h][j][i] - max_val)
#             exp_sum += result[h][j][i]
#         for i in range(j+1):
#             result[h][j][i] = result[h][j][i] / exp_sum
#         for i in range(t):
#             if i > j:
#                 result[h][j][i] = 0
# # print(result.flatten()[873].item())
# manual_out = result @ v
# print(torch.sum(manual_out).item())

# out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
# print(torch.sum(out).item())