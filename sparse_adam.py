"""Adam for embeddings whose gradient is sparse, with per-row state advanced on demand.

torch.optim.SparseAdam keeps two dense moment tensors and applies no decay to rows a step did
not touch, so a row's second moment reflects however many steps ago it was last seen rather than
the current step. This keeps a per-row clock instead and advances the missed decay arithmetically
when a row is next reached, which is exact rather than lazy.

Momentum is dropped outright (beta1 = 0). Under lazy updates a row touched every k steps would
otherwise carry `beta1 * (gradient from k steps ago)` into its update, which for a table where k
runs into the thousands means the step is dominated by a stale gradient. Removing it also removes
the exp_avg tensor, cutting state from two dense [rows, dim] tensors to one.
"""
import torch


class SparseRowAdam(torch.optim.Optimizer):
    """Adam without momentum over rows carrying gradient.

    Memory per row is `dim * itemsize(state_dtype) + 4` bytes against `2 * dim * 4` for
    SparseAdam, so at dim 256 it is 1028 bytes in fp32 or 516 in bf16, versus 2048.
    """

    def __init__(self, params, lr=1e-3, beta2=0.999, eps=1e-8, state_dtype=torch.float32):
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must be in [0, 1)")
        super().__init__(params, dict(lr=lr, beta2=beta2, eps=eps, state_dtype=state_dtype))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, beta2, eps = group["lr"], group["beta2"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if not p.grad.is_sparse:
                    raise RuntimeError("SparseRowAdam requires a sparse gradient")
                grad = p.grad.coalesce()          # unique row indices, duplicates summed
                idx = grad.indices()[0]
                val = grad.values().float()
                state = self.state[p]
                if not state:
                    state["v"] = torch.zeros(p.shape, dtype=group["state_dtype"], device=p.device)
                    state["seen"] = torch.zeros(p.shape[0], dtype=torch.int32, device=p.device)
                    state["t"] = 0
                state["t"] += 1
                t = state["t"]

                # advance each touched row's decay over the steps it sat out, then take its step
                missed = (t - 1 - state["seen"][idx].long()).clamp_(min=0)
                v = state["v"][idx].float() * beta2 ** missed.to(torch.float32).unsqueeze(1)
                v = beta2 * v + (1 - beta2) * val.square()
                update = val / ((v / (1 - beta2 ** t)).sqrt_() + eps)

                # idx is unique after coalesce, so this scatter has no accumulation conflict and
                # is deterministic -- which the decoder depends on.
                # The cast matters when the table is stored below fp32: the update is computed
                # in fp32 and rounded once, on the way in. Safe here only because this table's
                # updates run ~18% of row magnitude (ngram_lr=1.0 with Adam normalising), so
                # almost nothing is lost to rounding -- 0.18% of coordinates in fp16. It would
                # NOT be safe for a normally-parameterised weight.
                p.index_add_(0, idx, update.to(p.dtype), alpha=-lr)
                state["v"][idx] = v.to(state["v"].dtype)
                state["seen"][idx] = t
