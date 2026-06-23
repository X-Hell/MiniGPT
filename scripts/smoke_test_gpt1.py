#!/usr/bin/env python3
"""
Smoke test for the modern (RoPE + SwiGLU + RMSNorm) architecture at small scale.

Run with:
    MINIGPT_BACKEND=numpy python3 scripts/smoke_test_gpt1.py

Checks: tied embedding, param count, forward+backward (no NaN), weight-decay
grouping (only RMSNorm gammas excluded), LR schedule, and config defaults.
"""
import os
import sys
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
from minigpt.backend import xp
from minigpt.config import ModelConfig
from minigpt.model import MiniTransformer
from minigpt.optimizer import LRSchedule, build_param_groups

# ---- 1. Small-scale instantiation (d=384, L=6 — much faster than 30M) --------
cfg = ModelConfig(d_model=384, n_layers=6, n_heads=6,
                  vocab_size=4096, d_ff=1024, max_len=128, dropout=0.1)
model = MiniTransformer(cfg)

# ---- 2. Tied-embedding uniqueness -------------------------------------------
names = [n for n, _ in model.named_parameters()]
w_emb_count = sum(1 for n in names if n == "embeddings.W_emb")
assert w_emb_count == 1, f"W_emb should appear exactly once, got {w_emb_count}"
assert not any("W_out" in n or "lm_head" in n for n in names), \
    "Found a separate output head — tied embedding violated"
assert not any("W_pos" in n for n in names), "RoPE model must have no W_pos"
assert not any(".b_" in n or "beta" in n for n in names), \
    "Modern model is bias-free and uses RMSNorm (no beta)"
print("[OK] Tied embedding, no W_pos, no biases/beta.")

# ---- 3. Parameter count -------------------------------------------------------
total = sum(p.size for _, p in model.named_parameters())
print(f"[OK] Parameter count: {total:,}  (small-scale d=384/L=6/V=4096)")

# ---- 4. Forward + backward on dummy batch ------------------------------------
B, T = 2, 32
rng = np.random.RandomState(42)
x_np = rng.randint(0, cfg.vocab_size, size=(B, T)).astype(np.int64)
y_np = rng.randint(0, cfg.vocab_size, size=(B, T)).astype(np.int64)
x = xp.asarray(x_np)
y = xp.asarray(y_np)

model.train()
logits, _ = model.forward(x, training=True)
assert logits.shape == (B, T, cfg.vocab_size), \
    f"Unexpected logits shape: {logits.shape}"

# Cross-entropy loss + gradient
logits_flat = logits.reshape(-1, cfg.vocab_size)
y_flat = y.reshape(-1)
mx = xp.max(logits_flat, axis=1, keepdims=True)
ex = xp.exp(logits_flat - mx)
probs = ex / xp.sum(ex, axis=1, keepdims=True)
N = logits_flat.shape[0]
loss = float(-xp.mean(xp.log(probs[xp.arange(N), y_flat] + 1e-9)))
dlogits = probs.copy()
dlogits[xp.arange(N), y_flat] -= 1
dlogits = (dlogits / N).reshape(B, T, cfg.vocab_size)

grads = model.backward(dlogits)
dW_emb_out, layer_grads, dX_emb, d_gamma_final = grads

# Check no NaN / Inf
def has_nan(a):
    return bool(xp.any(xp.isnan(a)) | xp.any(xp.isinf(a)))

assert not has_nan(dW_emb_out),   "NaN/Inf in dW_emb_out"
assert not has_nan(d_gamma_final), "NaN/Inf in d_gamma_final"
for i, (swiglu_g, attn_g, rms1_dg, rms2_dg) in enumerate(layer_grads):
    for j, g in enumerate(tuple(swiglu_g) + tuple(attn_g)):
        assert not has_nan(g), f"NaN/Inf in layer {i} weight grad index {j}"
    for g in (rms1_dg, rms2_dg):
        assert not has_nan(g), f"NaN/Inf in layer {i} RMSNorm grad"

expected_loss = math.log(cfg.vocab_size)
print(f"[OK] Forward+backward. step-0 loss = {loss:.4f}  "
      f"(expect ~ln(V) = {expected_loss:.4f})")

# ---- 5. Param-group decay verification ----------------------------------------
pg = build_param_groups(model, weight_decay=0.1)
decay_names, no_decay_names = [], []
for name, p, group in model.named_parameters_with_groups():
    (decay_names if group == "decay" else no_decay_names).append(name)

print(f"[OK] Decay group ({len(decay_names)} params), first 3:")
for n in decay_names[:3]:
    print(f"      {n}")
print(f"[OK] No-decay group ({len(no_decay_names)} params):")
for n in no_decay_names:
    print(f"      {n}")

# Modern model: ONLY RMSNorm gammas are excluded from weight decay.
required_no_decay = {
    "final_norm.gamma",
    "layers.0.rms1.gamma", "layers.0.rms2.gamma",
}
missing = required_no_decay - set(no_decay_names)
assert not missing, f"These params must be no_decay but aren't: {missing}"
assert all(n.endswith(".gamma") for n in no_decay_names), \
    "Only RMSNorm gammas should be in the no-decay group"
print("[OK] Only RMSNorm gammas excluded from weight decay.")

# ---- 6. LR schedule sanity ---------------------------------------------------
sched = LRSchedule(peak_lr=3e-4, min_lr=3e-5,
                   warmup_steps=2000, max_steps=100_000)
assert sched(0) < 1e-6,                          f"Step-0 LR too high: {sched(0)}"
assert abs(sched(2000) - 3e-4) < 1e-8,           f"Warmup-end LR wrong: {sched(2000)}"
assert sched(100_000) == 3e-5,                    f"Final LR wrong: {sched(100_000)}"
assert sched(50_000) > sched(100_000),            "Mid-training LR not above floor"
print(f"[OK] LR schedule: step0={sched(0):.2e}  "
      f"warmup_end={sched(2000):.2e}  final={sched(100_000):.2e}")

# ---- 7. Config defaults -------------------------------------------------------
from minigpt.config import ModelConfig as MC, TrainConfig as TC
mc = MC()
tc = TC()
assert mc.d_model == 512,    f"d_model={mc.d_model}"
assert mc.n_layers == 8,     f"n_layers={mc.n_layers}"
assert mc.n_heads == 8,      f"n_heads={mc.n_heads}"
assert mc.d_ff == 1024,      f"d_ff={mc.d_ff}"
assert mc.vocab_size == 16384, f"vocab_size={mc.vocab_size}"
assert mc.rope_theta == 10000.0, f"rope_theta={mc.rope_theta}"
assert tc.beta2 == 0.95,     f"beta2={tc.beta2}  (modern: 0.95)"
assert tc.warmup_steps == 2000, f"warmup_steps={tc.warmup_steps}"
default_model = MiniTransformer(MC())
default_params = sum(p.size for _, p in default_model.named_parameters())
assert 28_000_000 < default_params < 31_000_000, \
    f"Default model should be ~30M, got {default_params:,}"
print(f"[OK] Config defaults: d={mc.d_model} L={mc.n_layers} H={mc.n_heads} "
      f"d_ff={mc.d_ff} V={mc.vocab_size} theta={mc.rope_theta:.0f} "
      f"-> {default_params:,} params")

# ---- 8. One optimizer step (no NaN) ------------------------------------------
from minigpt.optimizer import build_param_groups, Adam as _Adam
_opt = _Adam(lr=1e-4, weight_decay=0.1)
_pg = build_param_groups(model, weight_decay=0.1)
_gg = [{'params': [xp.zeros_like(p) for p in g['params']]} for g in _pg]
_opt.step_grouped(_pg, _gg, lr=1e-4)
print("[OK] Adam.step_grouped ran without error.")

print("\n========= ALL SMOKE TESTS PASSED =========")
