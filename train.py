import os
import copy
import json
import time
import math
import numpy as np
import torch
import wandb
import torch.nn.functional as F
from model import GPT, GPTConfig, FP8Linear, QATLinear, quantize_4bit_stats
from muon import Muon
import matchfeat

B, ACCUM, T, STEPS, LOG, CKPT = 16, 1, 2048, 100_000, 100, 1000
LR_MUON, LR_WTE, LR_HEAD, LR_CONV, LR_SCALAR, LR_BIAS = 0.015, 0.005, 0.001818, 0.003, 0.003, 0.0209
WD_MUON, WD_ADAM = 0.005, 0.005
WARMUP, COOLDOWN = 0, 0.30
# Optional checkpointing:
# - INIT_FROM: load model weights (and seed EMA from them) for a fresh schedule / cooldown / QAT branch
# - RESUME: resume full state (model, ema, optimizers, step, RNG) from a saved ckpt.pt
INIT_FROM = None
RESUME = None
# EMA of all weights, updated after every optimizer step (decay warms up as (1+t)/(10+t));
# None disables. Every CKPT steps both models are scored on the same EVAL_BATCHES fixed batches
EMA_DECAY, EVAL_BATCHES = 0.999, 32
# distillation: set TEACHER to a state_dict path (e.g. a model.pt from a past run)
# loss = (1 - KD_ALPHA) * CE + KD_ALPHA * KD_TEMP^2 * KL(teacher || student)
TEACHER = None
TEACHER_CFG = dict(n_layer=12, n_head=12, n_embd=768)
KD_ALPHA, KD_TEMP = 1.0, 1.0
# data: "enwik9" trains on raw bytes; a bpe_tok.py prefix such as "enwik9.split4096"
# trains on its token file (DATA + ".i16"), with the vocab from DATA + ".json"
DATA = "enwik9.split4096"
# match features (matchfeat.py): one column per context length, counted in bytes or tokens
# to match DATA; () disables. The token lengths cover about the same text at ~2.3 bytes/token
MATCH_K = (5, 12, 32, 64) if DATA == "enwik9" else (2, 3, 5, 8, 13, 27)

torch.set_float32_matmul_precision("high")
N_BYTES = os.path.getsize("enwik9")
if DATA == "enwik9":
    data = torch.frombuffer(
        bytearray(open("enwik9", "rb").read()), dtype=torch.uint8
    ).cuda()
    VOCAB = 256
else:
    data = torch.from_numpy(np.fromfile(DATA + ".i16", dtype=np.int16)).cuda()
    VOCAB = 256 + len(json.load(open(DATA + ".json"))["merges"])
N = len(data)  # training positions: bytes or tokens
print(
    f"data: {DATA} | {N:,} positions, vocab {VOCAB}, {N_BYTES / N:.3f} bytes/position"
)
match_tables = ()
if MATCH_K:
    # [N, len(MATCH_K)] each, as numpy memmaps on disk (~8 GB for enwik9, too much for VRAM);
    # each micro-step gathers only its B windows of T rows
    match_tables = matchfeat.cached(data, MATCH_K, DATA)
    torch.cuda.empty_cache()  # hand the build's scratch memory back before CUDA graph capture
offs = torch.arange(T + 1, device="cuda")


class Batches:
    """Random window starts, plus the match feature rows of each window, gathered on the CPU
    into a ring of pinned buffers and copied to the GPU asynchronously. A buffer is refilled
    only after the event of its previous copy has fired."""

    def __init__(self, tables, n_buf=8):
        self.gen = torch.Generator().manual_seed(1234)
        self.tables = tables
        self.ring = [
            [torch.empty(B, dtype=torch.long).pin_memory()]
            + [
                torch.empty(
                    B, T, a.shape[1], dtype=torch.from_numpy(np.empty(0, a.dtype)).dtype
                ).pin_memory()
                for a in tables
            ]
            for _ in range(n_buf)
        ]
        self.events = [None] * n_buf
        self.i = 0

    def next(self):
        i, self.i = self.i, (self.i + 1) % len(self.ring)
        if self.events[i] is not None:
            self.events[i].synchronize()
        bufs = self.ring[i]
        torch.randint(0, N - T, (B,), generator=self.gen, out=bufs[0])
        for a, buf in zip(self.tables, bufs[1:]):
            bn = buf.numpy()
            # match features describe the byte being predicted, i.e. the target positions s+1..s+T
            for r, s in enumerate(bufs[0].tolist()):
                bn[r] = a[s + 1 : s + 1 + T]
        out = [b.cuda(non_blocking=True) for b in bufs]
        self.events[i] = torch.cuda.Event()
        self.events[i].record()
        return out[0], out[1:]


batches = Batches(match_tables)

torch.manual_seed(0)
cfg = GPTConfig(
    context_length=T,
    vocab_size=VOCAB,
    n_match=len(MATCH_K),
    match_codes=matchfeat.N_CODE,
)
model = GPT(cfg).cuda()
if INIT_FROM and not RESUME:
    model.load_state_dict(torch.load(INIT_FROM, map_location="cuda", weights_only=True))
    print(f"initialized weights from {INIT_FROM}")
print(f"params: {sum(p.numel() for p in model.parameters())}")

teacher = None
if TEACHER:
    t_sd = torch.load(TEACHER, map_location="cuda")
    # a teacher trained with match features gets them too; its columns must be the first
    # t_match entries of MATCH_K, built with the same code layout
    t_match = t_sd["match_gate"].shape[0] if "match_gate" in t_sd else 0
    assert t_sd["wte.weight"].shape[0] == VOCAB, (
        "teacher was trained on a different vocab than DATA"
    )
    if t_match:
        assert t_match <= len(MATCH_K), (
            f"teacher has {t_match} match columns, MATCH_K has {len(MATCH_K)}"
        )
        assert t_sd["match_gate"].shape[1] == matchfeat.N_CODE, (
            "teacher uses an older match code layout"
        )
    teacher = GPT(
        GPTConfig(
            context_length=T,
            vocab_size=t_sd["wte.weight"].shape[0],
            n_match=t_match,
            match_codes=matchfeat.N_CODE,
            **TEACHER_CFG,
        )
    ).cuda()
    teacher.load_state_dict(t_sd)
    del t_sd
    teacher.eval().requires_grad_(False)
    print(f"teacher params: {sum(p.numel() for p in teacher.parameters())}")

lin = [m for m in model.h.modules() if isinstance(m, (FP8Linear, QATLinear))]
convs = [m for m in model.h.modules() if isinstance(m, torch.nn.Conv1d)]
attn_gates = [b.attn_gate for b in model.h if b.attn_gate is not None]
muon_ps = [m.weight for m in lin]
wte_ps = [model.wte.weight]
head_ps = [] if cfg.tie_weights else [model.lm_head.weight]
conv_ps = (
    [m.weight for m in convs]
    + [m.weight for m in attn_gates]
    + ([model.match_ctx.weight, model.match_hit_emb, model.match_miss_emb] if cfg.n_match else [])
)
scalar_ps = [
    p
    for p in (
        model.resid_lambdas,
        model.x0_lambdas,
        model.v_lambdas,
        model.skip_lambdas,
        model.pass_scale,
        model.pass_shift,
        model.loop_emb,
        model.logit_scale,
        *(
            (
                model.match_hit_scale,
                model.match_miss_scale,
                model.match_agree_scale,
                model.match_hit_bias,
                model.match_miss_bias,
            )
            if cfg.n_match
            else ()
        ),
    )
    if p is not None
]
assigned = {id(p) for p in muon_ps + wte_ps + head_ps + conv_ps + scalar_ps}
bias_ps = [p for p in model.parameters() if id(p) not in assigned]
assert len(assigned) + len(bias_ps) == len(list(model.parameters()))
adamw = lambda ps, lr, wd=0.0: torch.optim.AdamW(ps, lr=lr, weight_decay=wd, fused=True)
opts = [
    adamw(wte_ps, LR_WTE, WD_ADAM),
    *([adamw(head_ps, LR_HEAD, WD_ADAM)] if head_ps else []),
    *([adamw(conv_ps, LR_CONV, WD_ADAM)] if conv_ps else []),
    adamw(scalar_ps, LR_SCALAR, 0.0),
    adamw(bias_ps, LR_BIAS, 0.0),
    Muon(muon_ps, lr=LR_MUON, weight_decay=WD_MUON),
]
for o in opts:
    for g in o.param_groups:
        g["initial_lr"] = g["lr"]

ema = None
if EMA_DECAY:
    # fp32 like the master weights: a bf16 EMA would round away the (1 - decay) * delta steps
    ema = copy.deepcopy(model).requires_grad_(False)
    ema_p, model_p = list(ema.parameters()), list(model.parameters())
model.build_swa_mask()
if cfg.qat_group:
    qs = quantize_4bit_stats(model)
    print(f"4-bit QAT (g={cfg.qat_group}): raw={qs['raw_mb']:.2f} MB, entropy-est={qs['entropy_mb']:.2f} MB ({qs['bits_per_qweight']:.2f} bpw)")
eval_batches = [batches.next() for _ in range(EVAL_BATCHES)]
start_step = 0
if RESUME:
    ckpt = torch.load(RESUME, map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if ema is not None and ckpt.get("ema") is not None:
        ema.load_state_dict(ckpt["ema"])
    for o, osd in zip(opts, ckpt["opts"]):
        o.load_state_dict(osd)
    start_step = ckpt["step"]
    if "batch_gen_state" in ckpt:
        batches.gen.set_state(ckpt["batch_gen_state"].cpu())
    del ckpt
    print(f"resumed full checkpoint from {RESUME} at step {start_step}")
elif INIT_FROM:
    # Keep eval_batches identical (seed=1234), then advance RNG so cooldown sees fresh windows
    batches.gen.manual_seed(5678)


@torch.no_grad()
def eval_ce(m):
    """Mean CE (nats/position) of m on the fixed eval batches, eager and uncompiled."""
    tot = 0.0
    for starts, match in eval_batches:
        w = data[starts[:, None] + offs].long()
        mp, ml = match if MATCH_K else (None, None)
        with torch.autocast("cuda", torch.bfloat16):
            logits = m(w[:, :-1], mp, ml)
        tot += F.cross_entropy(logits.flatten(0, 1).float(), w[:, 1:].flatten()).item()
    return tot / len(eval_batches)


# the teacher runs as its own compiled function, outside the student's CUDA graph: capturing
# both in one graph crashed with "invalid stream capture status" inside an FP8 _scaled_mm
@torch.compile(mode="max-autotune-no-cudagraphs")
def teacher_fn(x, mp, ml):
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        return (
            teacher(x, mp[..., :t_match], ml[..., :t_match]) if t_match else teacher(x)
        )


@torch.compile(mode="max-autotune")
def loss_fn(x, y, mp, ml, t_logits):
    with torch.autocast("cuda", torch.bfloat16):
        logits = model(x, mp, ml)
    logits = logits.flatten(0, 1).float()
    ce = F.cross_entropy(logits, y.flatten())
    if teacher is None:
        return ce, ce, torch.zeros_like(ce)
    t_logp = F.log_softmax(t_logits.flatten(0, 1).float() / KD_TEMP, -1)
    s_logp = F.log_softmax(logits / KD_TEMP, -1)
    kl = F.kl_div(s_logp, t_logp, log_target=True, reduction="batchmean")
    return (1 - KD_ALPHA) * ce + KD_ALPHA * KD_TEMP**2 * kl, ce, kl


def lr_mult(step):
    if WARMUP > 0 and step < WARMUP:
        return (step + 1) / WARMUP
    if COOLDOWN >= 1.0:
        return max(0.0, 1.0 - (step - WARMUP) / max(STEPS - WARMUP, 1))
    return min(1.0, (1.0 - step / STEPS) / COOLDOWN)


wandb.init(
    project="hutter",
    config=dict(
        B=B,
        ACCUM=ACCUM,
        T=T,
        STEPS=STEPS,
        LR_MUON=LR_MUON,
        LR_WTE=LR_WTE,
        LR_HEAD=LR_HEAD,
        LR_CONV=LR_CONV,
        LR_SCALAR=LR_SCALAR,
        LR_BIAS=LR_BIAS,
        WD_MUON=WD_MUON,
        WD_ADAM=WD_ADAM,
        WARMUP=WARMUP,
        COOLDOWN=COOLDOWN,
        INIT_FROM=INIT_FROM,
        RESUME=RESUME,
        EMA_DECAY=EMA_DECAY,
        EVAL_BATCHES=EVAL_BATCHES,
        DATA=DATA,
        VOCAB=VOCAB,
        TEACHER=TEACHER,
        TEACHER_CFG=TEACHER_CFG if TEACHER else None,
        KD_ALPHA=KD_ALPHA if TEACHER else None,
        KD_TEMP=KD_TEMP if TEACHER else None,
        MATCH_K=MATCH_K,
        MATCH_TABLE_BITS=matchfeat.TABLE_BITS,
        MATCH_DIST_EDGES=matchfeat.DIST_EDGES,
        params=sum(p.numel() for p in model.parameters()),
        **vars(cfg),
    ),
)

# Warm up cuBLASLt FP8 workspace eagerly before CUDA graph capture (prevents
# intermittent "invalid stream capture status" inside _scaled_mm on cached restarts)
init_ce = eval_ce(model)
if INIT_FROM or RESUME:
    init_bpb = init_ce / math.log(2) * N / N_BYTES
    print(f"step {start_step} | init_eval_loss {init_ce:.4f} | init_eval_bpb {init_bpb:.4f}")

log = open("log.csv", "a" if RESUME else "w", buffering=1)
if not RESUME:
    log.write("step,loss,bpb,bytes,ms_per_step\n")
t0 = time.time()
for step in range(start_step, STEPS):
    torch.compiler.cudagraph_mark_step_begin()
    loss = kl = 0.0
    for _ in range(ACCUM):
        starts, match = batches.next()
        w = data[starts[:, None] + offs].long()
        mp, ml = match if MATCH_K else (None, None)
        x = w[:, :-1]
        t_logits = teacher_fn(x, mp, ml) if teacher is not None else None
        l, ce, k = loss_fn(x, w[:, 1:], mp, ml, t_logits)
        (l / ACCUM).backward()
        loss += ce.detach() / ACCUM
        kl += k.detach() / ACCUM
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    for o in opts:
        for g in o.param_groups:
            g["lr"] = g["initial_lr"] * lr_mult(step)
        o.step()
    model.zero_grad(set_to_none=True)
    if ema is not None:
        decay = EMA_DECAY if (INIT_FROM or RESUME) else min(EMA_DECAY, (1 + step) / (10 + step))
        torch._foreach_lerp_(ema_p, model_p, 1 - decay)

    if (step + 1) % LOG == 0 or step == STEPS - 1:
        loss, kl = loss.item(), kl.item()
        # loss is nats per position; total archive-equivalent bytes, then bits per raw byte
        total = loss / math.log(2) * N / 8
        bpb = total * 8 / N_BYTES
        dt = (time.time() - t0) / LOG
        t0 = time.time()
        print(
            f"step {step + 1} | loss {loss:.4f}{f' | kl {kl:.4f}' if teacher is not None else ''} | bpb {bpb:.4f} | bytes {total:,.0f} | {dt * 1000:.1f}ms/step"
        )
        log.write(f"{step + 1},{loss:.6f},{bpb:.6f},{total:.0f},{dt * 1000:.2f}\n")
        wandb.log(
            dict(
                loss=loss,
                bpb=bpb,
                **(dict(kl=kl) if teacher is not None else {}),
                bytes=total,
                ms_per_step=dt * 1000,
                grad_norm=gnorm.item(),
                lr_mult=lr_mult(step),
            ),
            step=step + 1,
        )

    if (step + 1) % CKPT == 0 or step == STEPS - 1:
        to_bpb = lambda ce: ce / math.log(2) * N / N_BYTES
        ev = dict(eval_loss=eval_ce(model))
        if ema is not None:
            ev["ema_eval_loss"] = eval_ce(ema)
        ev.update({k.replace("loss", "bpb"): to_bpb(v) for k, v in list(ev.items())})
        print(f"step {step + 1} | " + " | ".join(f"{k} {v:.4f}" for k, v in ev.items()))
        wandb.log(ev, step=step + 1)
        torch.save(model.state_dict(), "model.pt")
        if ema is not None:
            torch.save(ema.state_dict(), "model_ema.pt")
        torch.save(
            dict(
                step=step + 1,
                model=model.state_dict(),
                ema=ema.state_dict() if ema is not None else None,
                opts=[o.state_dict() for o in opts],
                batch_gen_state=batches.gen.get_state(),
            ),
            "ckpt.pt",
        )

wandb.finish()
