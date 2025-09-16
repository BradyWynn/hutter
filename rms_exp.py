import torch
import torch.nn.functional as F

def norm(x):
	return F.rms_norm(x, (x.size(-1),))

torch.manual_seed(1)
a = torch.randn((16, 32, 32))

# (1) Use F.rms_norm directly (normalized_shape can be int or tuple)
out_f = norm(a)   # or F.rms_norm(a, (a.size(-1),))

# (2) Manual, vectorized equivalent (with eps to avoid div-by-zero)
# eps = 1e-8
# rms = torch.zeros_like(a[..., 0])
# for i in range(a.size(-1)):
# 	rms += a[..., i]**2
# rms = torch.sqrt(rms/a.size(-1) + eps).unsqueeze(-1)
# print(rms.shape)
# out_manual = a / rms
print(out_f.flatten()[:10])

# print(torch.allclose(out_manual, out_f))
# print(out_f.flatten()[:10])
# print(out_manual.flatten()[:10])