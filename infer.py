import numpy as np

t = 1024
n_embd = 512
n_heads = 8
head_embd = 64

c_attn = (np.arange(n_embd*n_embd*3) % 256).reshape(n_embd, 3*n_embd)
x = (np.arange(t*n_embd) % 256).reshape(t, n_embd)

qkv = x @ c_attn
q = qkv[:, n_embd*0:n_embd*1].reshape(t, n_heads, head_embd)
k = qkv[:, n_embd*1:n_embd*2].reshape(t, n_heads, head_embd)
v = qkv[:, n_embd*2:n_embd*3].reshape(t, n_heads, head_embd)

q = np.transpose(q, axes=(1, 0, 2))
k = np.transpose(k, axes=(1, 0, 2))
v = np.transpose(v, axes=(1, 0, 2))

result = q @ np.transpose(k, axes=(0, 2, 1))
print(result.flatten()[8432])