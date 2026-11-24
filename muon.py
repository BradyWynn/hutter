import torch


def ns5(X, steps):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = X.bfloat16()
    t = X.size(-2) > X.size(-1)
    if t:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.mT if t else X


def bucket(params, grads, bufs, row_v, lr, beta, beta2, ns_steps, wd):
    torch._foreach_lerp_(bufs, grads, 1 - beta)
    upd = [g.lerp(m, beta) for g, m in zip(grads, bufs)]
    shape = params[0].shape
    u = ns5(torch.stack(upd), ns_steps).float() * max(1, shape[-2] / shape[-1]) ** 0.5
    rv = torch.stack(row_v)
    rv.lerp_(u.square().mean(-1), 1 - beta2)
    for r, n in zip(row_v, rv.unbind(0)):
        r.copy_(n)
    scaled = u * rv.clamp_min(1e-10).rsqrt().unsqueeze(-1)
    n = u.norm(dim=(-2, -1), keepdim=True)
    u = scaled * (n / scaled.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-10))
    if wd > 0:
        # Cautious weight decay: only decay weights when the orthogonalized update and weight align
        p_stack = torch.stack(params).float()
        u = u + wd * torch.where(u * p_stack > 0, p_stack, 0.0)
    torch._foreach_add_(params, [x.to(params[0].dtype) * -lr for x in u.unbind(0)])


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5, normuon_beta2=0.95, weight_decay=0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, ns_steps=ns_steps, normuon_beta2=normuon_beta2, weight_decay=weight_decay),
        )
        self.fn = torch.compile(bucket, mode="max-autotune-no-cudagraphs")
        self.lr_t = {}

    @torch.no_grad()
    def step(self):
        for gi, g in enumerate(self.param_groups):
            lr = self.lr_t.setdefault(gi, torch.zeros((), device=g["params"][0].device))
            lr.fill_(g["lr"])
            buckets = {}
            for p in g["params"]:
                buckets.setdefault(tuple(p.shape), []).append(p)
            for ps in buckets.values():
                for p in ps:
                    s = self.state[p]
                    if not s:
                        s["momentum_buffer"] = torch.zeros_like(p)
                        s["row_v"] = torch.zeros(p.shape[-2], device=p.device)
                self.fn(ps, [p.grad for p in ps], [self.state[p]["momentum_buffer"] for p in ps],
                        [self.state[p]["row_v"] for p in ps], lr, g["momentum"], g["normuon_beta2"], g["ns_steps"], g["weight_decay"])
