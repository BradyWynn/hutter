import os
import copy
import json
import time
import math
import numpy as np
import torch
import wandb
import torch.nn.functional as F
from model import GPT, GPTConfig, FP8Linear, QATLinear, MoEMLP, quantize_stats
from muon import Muon
from eval_hutter import vocab_info, extra_overhead_bytes, make_strided_starts, eval_hutter_suite
import matchfeat

B, ACCUM, T = 16, 1, 2048
STEPS = int(os.environ.get("STEPS", "100000"))
LOG = int(os.environ.get("LOG", "100"))
CKPT = int(os.environ.get("CKPT", "1000"))
RUN_NAME = os.environ.get("RUN_NAME", "") or None
OUT_DIR = os.environ.get("OUT_DIR", ".")
COMPILE_MODE = os.environ.get("COMPILE_MODE", "max-autotune")
os.makedirs(OUT_DIR, exist_ok=True)
LR_MUON, LR_WTE, LR_HEAD, LR_CONV, LR_SCALAR, LR_BIAS = 0.015, 0.005, 0.001818, 0.003, 0.003, 0.0209
WD_MUON, WD_ADAM = 0.005, 0.005
MUON_MOM_MIN, MUON_MOM_MAX, MUON_MOM_WARMUP = 0.85, 0.95, 300
WARMUP, COOLDOWN = 0, 0.50
# Late-cooldown mixed-precision QAT:
# Train in BF16 during main training, then turn on STE QAT only in the final (1 - QAT_START) of steps
# (Int6 for tied wte/lm_head, Int5/Int4 for dense blocks, Int4/Int3 for routed MoE experts).
_qs_env = os.environ.get("QAT_START", "0.85")
QAT_START = None if _qs_env.lower() == "none" else float(_qs_env)
QAT_GROUP = int(os.environ.get("QAT_GROUP", "64"))
QAT_BLOCK_BITS = int(os.environ.get("QAT_BLOCK_BITS", "4"))
QAT_EXPERT_BITS = int(os.environ.get("QAT_EXPERT_BITS", "4"))
QAT_EMB_BITS = int(os.environ.get("QAT_EMB_BITS", "6"))
N_LAYER = int(os.environ.get("N_LAYER", "7"))
N_LOOP = int(os.environ.get("N_LOOP", "2"))
N_EMBD = int(os.environ.get("N_EMBD", "256"))
N_HEAD = int(os.environ.get("N_HEAD", "4"))
N_KV_HEAD = int(os.environ.get("N_KV_HEAD", "2"))
N_EXPERTS = int(os.environ.get("N_EXPERTS", "0"))
TOP_K = int(os.environ.get("TOP_K", "2"))
SHARED_H = int(os.environ.get("SHARED_H", "256"))
EXPERT_H = int(os.environ.get("EXPERT_H", "256"))
ENGRAM_ROWS = int(os.environ.get("ENGRAM_ROWS", "0"))
ENGRAM_DIM = int(os.environ.get("ENGRAM_DIM", "32"))
if N_EXPERTS > 0 and "COMPILE_MODE" not in os.environ:
    COMPILE_MODE = "default"
# Optional checkpointing:
# - INIT_FROM: load model weights (and seed EMA from them) for a fresh schedule / cooldown / QAT branch
# - RESUME: resume full state (model, ema, optimizers, step, RNG) from a saved ckpt.pt
INIT_FROM = None
RESUME = None
# EMA of all weights, updated after every optimizer step (decay warms up as (1+t)/(10+t));
# None disables. Every CKPT steps we run the full strided Hutter-MB eval suite.
EMA_DECAY = 0.999
EVAL_WINDOWS = int(os.environ.get("EVAL_WINDOWS", "2048"))
EVAL_STRIDE = T // 2  # evenly spaced full-context windows across all 1 GB of enwik9
# distillation: set TEACHER to a state_dict path (e.g. a model.pt from a past run)
# loss = (1 - KD_ALPHA) * CE + KD_ALPHA * KD_TEMP^2 * KL(teacher || student)
TEACHER = None
TEACHER_CFG = dict(n_layer=12, n_head=12, n_embd=768)
KD_ALPHA, KD_TEMP = 1.0, 1.0
# data: "enwik9" trains on raw bytes; a bpe_tok.py prefix such as "enwik9.split4096"
# trains on its token file (DATA + ".i16"), with the vocab from DATA + ".json"
DATA = os.environ.get("DATA", "enwik9.split4096")
# match features (matchfeat.py): one column per context length, counted in bytes or tokens
# to match DATA; () disables. The token lengths cover about the same text at ~2.8 bytes/token
_mk_env = os.environ.get("MATCH_K", "")
if _mk_env:
    MATCH_K = () if _mk_env.lower() == "none" else tuple(int(x) for x in _mk_env.split(","))
else:
    MATCH_K = (5, 12, 32, 64) if DATA == "enwik9" else (2, 3, 5, 8, 13, 27)

torch.set_float32_matmul_precision("high")
N_BYTES = os.path.getsize("enwik9")
TOK_LENS, VOCAB_BYTES, VOCAB = vocab_info(DATA)
EXTRA_BYTES = extra_overhead_bytes(DATA)
if DATA == "enwik9":
    data = torch.frombuffer(
        bytearray(open("enwik9", "rb").read()), dtype=torch.uint8
    ).cuda()
else:
    data = torch.from_numpy(np.fromfile(DATA + ".i16", dtype=np.int16)).cuda()
N = len(data)  # training positions: bytes or tokens
print(
    f"data: {DATA} | {N:,} positions, vocab {VOCAB} (zlib {VOCAB_BYTES/1e3:.1f} KB, extra {EXTRA_BYTES/1e6:.3f} MB), {N_BYTES / N:.3f} bytes/position"
)
match_tables = ()
if MATCH_K:
    # [N, len(MATCH_K)] each, as numpy memmaps on disk (~6.4 GB for split4096, too much for VRAM);
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
eval_starts = make_strided_starts(N, T, EVAL_STRIDE, EVAL_WINDOWS)

torch.manual_seed(0)
cfg = GPTConfig(
    context_length=T,
    vocab_size=VOCAB,
    n_layer=N_LAYER,
    n_loop=N_LOOP,
    n_head=N_HEAD,
    n_kv_head=N_KV_HEAD,
    n_embd=N_EMBD,
    n_experts=N_EXPERTS,
    n_experts_active=TOP_K,
    shared_h=SHARED_H,
    expert_h=EXPERT_H,
    engram_rows=ENGRAM_ROWS,
    engram_dim=ENGRAM_DIM,
    n_match=len(MATCH_K),
    match_codes=matchfeat.N_CODE,
    qat_block_bits=QAT_BLOCK_BITS,
    qat_expert_bits=QAT_EXPERT_BITS,
    qat_emb_bits=QAT_EMB_BITS,
    fp8=bool(int(os.environ.get("FP8", "0"))),
)
model = GPT(cfg).cuda()
if INIT_FROM and not RESUME:
    model.load_state_dict(torch.load(INIT_FROM, map_location="cuda", weights_only=True))
    print(f"initialized weights from {INIT_FROM}")
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

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
    print(f"teacher params: {sum(p.numel() for p in teacher.parameters()):,}")

lin = [m for m in model.h.modules() if isinstance(m, (FP8Linear, QATLinear))]
moes = [m for m in model.h.modules() if isinstance(m, MoEMLP)]
convs = [m for m in model.h.modules() if isinstance(m, torch.nn.Conv1d)]
attn_gates = [b.attn_gate for b in model.h if b.attn_gate is not None]
muon_ps = [m.weight for m in lin] + [p for m in moes for p in (m.w_fc, m.w_out)]
wte_ps = [model.wte.weight] + ([model.engram.table] if model.engram is not None else [])
head_ps = [] if cfg.tie_weights else [model.lm_head.weight]
conv_ps = (
    [m.weight for m in convs]
    + [m.weight for m in attn_gates]
    + [m.router.weight for m in moes]
    + ([model.engram.key_proj.weight, model.engram.val_proj.weight] if model.engram is not None else [])
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
muon_opt = Muon(muon_ps, lr=LR_MUON, momentum=MUON_MOM_MIN, weight_decay=WD_MUON)
opts = [
    adamw(wte_ps, LR_WTE, WD_ADAM),
    *([adamw(head_ps, LR_HEAD, WD_ADAM)] if head_ps else []),
    *([adamw(conv_ps, LR_CONV, WD_ADAM)] if conv_ps else []),
    adamw(scalar_ps, LR_SCALAR, 0.0),
    adamw(bias_ps, LR_BIAS, 0.0),
    muon_opt,
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
    batches.gen.manual_seed(5678)

qat_step = int(STEPS * QAT_START) if (QAT_START is not None and QAT_GROUP) else None
qat_active = False


def enable_qat_now(step):
    global qat_active
    if qat_active:
        return
    model.set_qat(group_size=QAT_GROUP, block_bits=QAT_BLOCK_BITS, emb_bits=QAT_EMB_BITS, expert_bits=QAT_EXPERT_BITS)
    if ema is not None:
        ema.set_qat(group_size=QAT_GROUP, block_bits=QAT_BLOCK_BITS, emb_bits=QAT_EMB_BITS, expert_bits=QAT_EXPERT_BITS)
    qat_active = True
    qs = quantize_stats(model, block_bits=QAT_BLOCK_BITS, emb_bits=QAT_EMB_BITS, expert_bits=QAT_EXPERT_BITS, group_size=QAT_GROUP)
    exp_str = f", exp {qs['expert_bpw']:.2f} bpw ({qs['expert_entropy_mb']:.2f}MB)" if qs["expert_params"] else ""
    print(
        f"step {step} | enabled mixed-precision QAT (emb={QAT_EMB_BITS}b, blk={QAT_BLOCK_BITS}b, exp={QAT_EXPERT_BITS}b, g={QAT_GROUP}): "
        f"raw={qs['raw_mb']:.2f} MB, entropy-est={qs['entropy_mb']:.2f} MB "
        f"(emb {qs['emb_bpw']:.2f} bpw, blk {qs['block_bpw']:.2f} bpw{exp_str})"
    )


if cfg.qat_group or (qat_step is not None and start_step >= qat_step):
    enable_qat_now(start_step)


# the teacher runs as its own compiled function, outside the student's CUDA graph: capturing
# both in one graph crashed with "invalid stream capture status" inside an FP8 _scaled_mm
@torch.compile(mode="max-autotune-no-cudagraphs")
def teacher_fn(x, mp, ml):
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        return (
            teacher(x, mp[..., :t_match], ml[..., :t_match]) if t_match else teacher(x)
        )


def _raw_loss(x, y, mp, ml, t_logits):
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


# Separate compiled graphs for FP8 main phase and BF16 late-cooldown QAT phase
# so switching QAT on mid-run never triggers a full dynamo reset or CUDA graph clash.
@torch.compile(mode=COMPILE_MODE)
def loss_fn_fp8(x, y, mp, ml, t_logits):
    return _raw_loss(x, y, mp, ml, t_logits)


@torch.compile(mode=COMPILE_MODE)
def loss_fn_qat(x, y, mp, ml, t_logits):
    return _raw_loss(x, y, mp, ml, t_logits)


def lr_mult(step):
    if WARMUP > 0 and step < WARMUP:
        return (step + 1) / WARMUP
    if COOLDOWN >= 1.0:
        return max(0.0, 1.0 - (step - WARMUP) / max(STEPS - WARMUP, 1))
    return min(1.0, (1.0 - step / STEPS) / COOLDOWN)


def muon_momentum(step):
    """Warm up Muon momentum 0.85 -> 0.95 early, hold 0.95, then cool down 0.95 -> 0.85."""
    if MUON_MOM_WARMUP > 0 and step < MUON_MOM_WARMUP:
        frac = step / MUON_MOM_WARMUP
        return MUON_MOM_MIN + (MUON_MOM_MAX - MUON_MOM_MIN) * frac
    cd_start = STEPS * (1.0 - COOLDOWN) if COOLDOWN < 1.0 else WARMUP
    if step >= cd_start and STEPS > cd_start:
        frac = min(1.0, (step - cd_start) / (STEPS - cd_start))
        return MUON_MOM_MAX - (MUON_MOM_MAX - MUON_MOM_MIN) * frac
    return MUON_MOM_MAX


wandb.init(
    project="hutter",
    name=RUN_NAME,
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
        MUON_MOM_MIN=MUON_MOM_MIN,
        MUON_MOM_MAX=MUON_MOM_MAX,
        MUON_MOM_WARMUP=MUON_MOM_WARMUP,
        WARMUP=WARMUP,
        COOLDOWN=COOLDOWN,
        QAT_START=QAT_START,
        QAT_GROUP=QAT_GROUP,
        QAT_BLOCK_BITS=QAT_BLOCK_BITS,
        QAT_EXPERT_BITS=QAT_EXPERT_BITS,
        QAT_EMB_BITS=QAT_EMB_BITS,
        INIT_FROM=INIT_FROM,
        RESUME=RESUME,
        EMA_DECAY=EMA_DECAY,
        EVAL_WINDOWS=EVAL_WINDOWS,
        EVAL_STRIDE=EVAL_STRIDE,
        DATA=DATA,
        VOCAB=VOCAB,
        VOCAB_BYTES=VOCAB_BYTES,
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
with torch.no_grad():
    _s0 = torch.from_numpy(eval_starts[:1]).cuda()
    _w0 = data[_s0[:, None] + offs].long()
    _mp0 = torch.from_numpy(match_tables[0][int(eval_starts[0]) + 1 : int(eval_starts[0]) + 1 + T][None]).cuda() if MATCH_K else None
    _ml0 = torch.from_numpy(match_tables[1][int(eval_starts[0]) + 1 : int(eval_starts[0]) + 1 + T][None]).cuda() if MATCH_K else None
    with torch.autocast("cuda", torch.bfloat16):
        _ = model(_w0[:, :-1], _mp0, _ml0)
    del _s0, _w0, _mp0, _ml0

if INIT_FROM or RESUME:
    ev0 = eval_hutter_suite(
        model, ema, data, match_tables, eval_starts, T,
        stride=EVAL_STRIDE, batch_size=B, tok_lens=TOK_LENS,
        n_bytes_total=N_BYTES, vocab_bytes=VOCAB_BYTES, extra_bytes=EXTRA_BYTES,
        block_bits=QAT_BLOCK_BITS, emb_bits=QAT_EMB_BITS, expert_bits=QAT_EXPERT_BITS, qat_group=QAT_GROUP,
    )
    print(
        f"step {start_step} | init eval_bpb {ev0['eval_bpb']:.4f} | "
        f"strided {ev0.get('ema_strided_bpb', ev0['strided_bpb']):.4f} | "
        f"q_strided {ev0['q_strided_bpb']:.4f} | hutter {ev0['hutter_mb']:.2f} MB"
    )

log = open(os.path.join(OUT_DIR, "log.csv"), "a" if RESUME else "w", buffering=1)
if not RESUME:
    log.write("step,loss,bpb,bytes,ms_per_step\n")
t0 = time.time()
for step in range(start_step, STEPS):
    if qat_step is not None and not qat_active and step >= qat_step:
        enable_qat_now(step)
    active_loss_fn = loss_fn_qat if qat_active else loss_fn_fp8

    torch.compiler.cudagraph_mark_step_begin()
    loss = kl = 0.0
    for _ in range(ACCUM):
        starts, match = batches.next()
        w = data[starts[:, None] + offs].long()
        mp, ml = match if MATCH_K else (None, None)
        x = w[:, :-1]
        t_logits = teacher_fn(x, mp, ml) if teacher is not None else None
        l, ce, k = active_loss_fn(x, w[:, 1:], mp, ml, t_logits)
        (l / ACCUM).backward()
        loss += ce.detach() / ACCUM
        kl += k.detach() / ACCUM
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    lrm = lr_mult(step)
    mom = muon_momentum(step)
    for g in muon_opt.param_groups:
        g["momentum"] = mom
    for o in opts:
        for g in o.param_groups:
            g["lr"] = g["initial_lr"] * lrm
        o.step()
    model.zero_grad(set_to_none=True)
    moe_max_load = model.update_router_biases() if cfg.n_experts > 0 else 1.0
    if ema is not None:
        decay = EMA_DECAY if (INIT_FROM or RESUME) else min(EMA_DECAY, (1 + step) / (10 + step))
        torch._foreach_lerp_(ema_p, model_p, 1 - decay)
        if cfg.n_experts > 0:
            for eb, mb in zip(ema.h, model.h):
                eb.moe.router_bias.copy_(mb.moe.router_bias)

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
                lr_mult=lrm,
                muon_momentum=mom,
                qat_active=int(qat_active),
                **(dict(moe_max_load=moe_max_load) if cfg.n_experts > 0 else {}),
            ),
            step=step + 1,
        )

    if (step + 1) % CKPT == 0 or step == STEPS - 1:
        ev = eval_hutter_suite(
            model, ema, data, match_tables, eval_starts, T,
            stride=EVAL_STRIDE, batch_size=B, tok_lens=TOK_LENS,
            n_bytes_total=N_BYTES, vocab_bytes=VOCAB_BYTES, extra_bytes=EXTRA_BYTES,
            block_bits=QAT_BLOCK_BITS, emb_bits=QAT_EMB_BITS, expert_bits=QAT_EXPERT_BITS, qat_group=QAT_GROUP,
        )
        print(
            f"step {step + 1} | eval_bpb {ev['eval_bpb']:.4f}"
            + (f" | ema_eval_bpb {ev['ema_eval_bpb']:.4f} | ema_strided {ev['ema_strided_bpb']:.4f}" if ema is not None else f" | strided {ev['strided_bpb']:.4f}")
            + f" | q_strided {ev['q_strided_bpb']:.4f} ({ev['q_drop_bpb']:+.4f})"
            + f" | wt {ev['weight_mb']:.2f}MB | HUTTER {ev['hutter_mb']:.2f} MB"
        )
        wandb.log(ev, step=step + 1)
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "model.pt"))
        if ema is not None:
            torch.save(ema.state_dict(), os.path.join(OUT_DIR, "model_ema.pt"))
        torch.save(
            dict(
                step=step + 1,
                model=model.state_dict(),
                ema=ema.state_dict() if ema is not None else None,
                opts=[o.state_dict() for o in opts],
                batch_gen_state=batches.gen.get_state(),
                qat_active=qat_active,
            ),
            os.path.join(OUT_DIR, "ckpt.pt"),
        )

wandb.finish()
