import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        # unconstrained parameters
        self.H_hat = nn.Parameter(torch.randn(out_dim, in_dim) * 0.05)
        self.b = nn.Parameter(torch.zeros(out_dim))
        self.a_hat = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        # enforce constraints
        H = F.softplus(self.H_hat)          # H >= 0
        a = torch.tanh(self.a_hat)          # a in (-1,1)

        x = F.linear(x, H, self.b)          # Hx + b
        return x + a * torch.tanh(x)        # g_k(x)
    

class CDF(nn.Module):
    def __init__(self, r=(3,3,3)):
        super().__init__()

        dims = [1] + list(r) + [1]
        K = len(dims) - 1

        self.layers = nn.ModuleList()

        for i in range(K):
            in_dim = dims[i]
            out_dim = dims[i+1]

            if i < K - 1:
                self.layers.append(Layer(in_dim, out_dim))
            else:
                # final layer (sigmoid only)
                layer = nn.Linear(in_dim, out_dim)
                self.layers.append(layer)

        self.K = K

    def forward(self, x):
        x = x.view(-1, 1)

        for i, layer in enumerate(self.layers):
            if i < self.K - 1:
                x = layer(x)
            else:
                # final layer with positivity constraint
                H = F.softplus(layer.weight)
                b = layer.bias
                x = torch.sigmoid(F.linear(x, H, b))

        return x.view(-1)

    def log_prob_quantized(self, x):
        upper = self.forward(x + 0.5)
        lower = self.forward(x - 0.5)
        mass = torch.clamp(upper - lower, min=1e-12)
        return torch.log(mass)

# import torch.optim as optim

# model = CDF(r=(3,3,3))
# print(list(model.parameters()))
# optimizer = optim.Adam(model.parameters(), lr=1e-3)

# for step in range(8000):
#     n = 512

#     bern = torch.bernoulli(torch.full((n,), 0.3))
#     y = bern * torch.randn(n) + (1-bern) * (torch.randn(n) + 5)

#     loss = -model.log_prob_quantized(y*256).mean()

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     if step % 1000 == 0:
#         print(step, loss.item())

#         grid = torch.linspace(-100, 100, 500)

#         with torch.no_grad():
#             upper = model(grid + 0.1)
#             lower = model(grid - 0.1)
#             density = upper - lower

#         plt.plot(grid.numpy(), density.numpy(), label=f"{step}")

# plt.legend()
# plt.show()
