import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

# a = torch.zeros(16, 16)
# for i in range(16):
#     for j in range(16):
#         if i > j:
#             a[i][j] = 1
# plt.imshow(a)
# plt.show()

t = 1024
n_embd = 512
head_embd = 64
n_heads = n_embd // head_embd

# c_attn = np.random.randn(n_embd, 3*n_embd) * 0.05
# x = np.random.randn(t, n_embd) * 0.05
c_attn = np.arange(n_embd*3*n_embd).reshape(n_embd, 3*n_embd)
x = np.arange(t*n_embd).reshape(t, n_embd)
c_attn = ((c_attn % 64) - 32) * 0.0015625
x = ((x % 64) - 32) * 0.0015625

qkv = x @ c_attn
q = qkv[:, n_embd*0:n_embd*1].reshape(t, n_heads, head_embd)
k = qkv[:, n_embd*1:n_embd*2].reshape(t, n_heads, head_embd)
v = qkv[:, n_embd*2:n_embd*3].reshape(t, n_heads, head_embd)

q = torch.from_numpy(q).permute(1, 0, 2)
k = torch.from_numpy(k).permute(1, 0, 2)
v = torch.from_numpy(v).permute(1, 0, 2)

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

out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print(torch.sum(out).item())