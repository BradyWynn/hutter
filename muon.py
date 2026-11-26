import os
import torch

# 5-step Polar Express minimax orthogonalization coefficients (Amsel et al. 2025 / modded-nanogpt).
# Maps singular values in [1e-3, 1] tightly to [0.85, 1.14] in bfloat16 (vs [0.68, 1.14] for fixed ns5).
POLAR_EXPRESS_COEFFS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


def polar_express(X, steps=5):
    X = X.bfloat16()
    t = X.size(-2) > X.size(-1)
    if t:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    coeffs = POLAR_EXPRESS_COEFFS[:steps]
    for a, b, c in coeffs:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.mT if t else X


def bucket(params, grads, bufs, row_v, lr, beta, beta2, ns_steps, wd):
    one_m_b = 1.0 - beta
    for b, g in zip(bufs, grads):
        b.lerp_(g, one_m_b)
    upd = [g.lerp(m, beta) for g, m in zip(grads, bufs)]
    shape = params[0].shape
    is_3d = params[0].ndim == 3
    if is_3d:
        # 3D MoE expert stacks [E, M, N]: orthogonalize every expert [M, N] independently
        u_in = torch.cat(upd, dim=0)
        rv = torch.cat(row_v, dim=0)
    else:
        u_in = torch.stack(upd)
        rv = torch.stack(row_v)
    u = polar_express(u_in, ns_steps).float() * max(1, shape[-2] / shape[-1]) ** 0.5
    rv.lerp_(u.square().mean(-1), 1 - beta2)
    rv_chunks = rv.chunk(len(params), dim=0) if is_3d else rv.unbind(0)
    for r, n in zip(row_v, rv_chunks):
        r.copy_(n)
    scaled = u * rv.clamp_min(1e-10).rsqrt().unsqueeze(-1)
    n = u.norm(dim=(-2, -1), keepdim=True)
    u = scaled * (n / scaled.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-10))
    if wd > 0:
        # Cautious weight decay: only decay weights when the orthogonalized update and weight align
        p_stack = (torch.cat(params, dim=0) if is_3d else torch.stack(params)).float()
        u = u + wd * torch.where(u * p_stack > 0, p_stack, 0.0)
    u_chunks = u.chunk(len(params), dim=0) if is_3d else u.unbind(0)
    torch._foreach_add_(params, [x.to(params[0].dtype) * -lr for x in u_chunks])


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5, normuon_beta2=0.95, weight_decay=0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, ns_steps=ns_steps, normuon_beta2=normuon_beta2, weight_decay=weight_decay),
        )
        self.fn = torch.compile(bucket, mode=os.environ.get("MUON_COMPILE_MODE", "max-autotune-no-cudagraphs"))
        self.lr_t = {}
        self.beta_t = {}

    @torch.no_grad()
    def step(self):
        for gi, g in enumerate(self.param_groups):
            dev = g["params"][0].device
            lr = self.lr_t.setdefault(gi, torch.zeros((), device=dev))
            lr.fill_(g["lr"])
            beta = self.beta_t.setdefault(gi, torch.zeros((), device=dev))
            beta.fill_(g["momentum"])
            buckets = {}
            for p in g["params"]:
                buckets.setdefault(tuple(p.shape), []).append(p)
            for ps in buckets.values():
                for p in ps:
                    s = self.state[p]
                    if not s:
                        s["momentum_buffer"] = torch.zeros_like(p)
                        s["row_v"] = torch.zeros(p.shape[:-1], device=p.device)
                self.fn(
                    ps,
                    [p.grad for p in ps],
                    [self.state[p]["momentum_buffer"] for p in ps],
                    [self.state[p]["row_v"] for p in ps],
                    lr,
                    beta,
                    g["normuon_beta2"],
                    g["ns_steps"],
                    g["weight_decay"],
                )
