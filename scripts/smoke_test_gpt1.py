#!/usr/bin/env python3
"""
Smoke test for Milestones 1+2 at small scale before the 117M training run.

Run with:
    MINIGPT_BACKEND=numpy python3 scripts/smoke_test_gpt1.py

Expected output (last lines):
    [OK] Tied embedding: W_emb appears exactly once, no W_out.
    [OK] Parameter count: ~14,000,000  (at d=384/L=6/V=4096)
    [OK] Forward+backward. step-0 loss ≈ 8.3
    [OK] W_pos, LN gamma/beta, all biases correctly excluded from WD.
    [OK] LR schedule: step0=1.25e-07 warmup_end=2.50e-04 final=1.00e-05
    ========= ALL SMOKE TESTS PASSED =========
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

# ---- 1. Small-scale instantiation (d=384, L=6 — much faster than 117M) -------
cfg = ModelConfig(d_model=384, n_layers=6, n_heads=6,
                  vocab_size=4096, max_len=128, dropout=0.1)
model = MiniTransformer(cfg)

# ---- 2. Tied-embedding uniqueness -------------------------------------------
names = [n for n, _ in model.named_parameters()]
w_emb_count = sum(1 for n in names if n == "embeddings.W_emb")
assert w_emb_count == 1, f"W_emb should appear exactly once, got {w_emb_count}"
assert not any("W_out" in n or "lm_head" in n for n in names), \
    "Found a separate output head — tied embedding violated"
print("[OK] Tied embedding: W_emb appears exactly once, no W_out.")

# ---- 3. Parameter count -------------------------------------------------------
total = sum(p.size for _, p in model.named_parameters())
print(f"[OK] Parameter count: {total:,}  (expect ~14M at d=384/L=6/V=4096)")

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
dW_emb_out, dW_pos, layer_grads, dX_emb = grads

# Check no NaN / Inf
def has_nan(a):
    return bool(xp.any(xp.isnan(a)) | xp.any(xp.isinf(a)))

assert not has_nan(dW_emb_out), "NaN/Inf in dW_emb_out"
assert not has_nan(dW_pos),     "NaN/Inf in dW_pos"
for i, (ffn_g, attn_g, (ln1_dg, ln1_db), (ln2_dg, ln2_db)) in enumerate(layer_grads):
    for j, g in enumerate(ffn_g + attn_g):
        assert not has_nan(g), f"NaN/Inf in layer {i} grad tuple index {j}"
    for g in (ln1_dg, ln1_db, ln2_dg, ln2_db):
        assert not has_nan(g), f"NaN/Inf in layer {i} LN grad"

expected_loss = math.log(cfg.vocab_size)
print(f"[OK] Forward+backward. step-0 loss = {loss:.4f}  "
      f"(expect ~ln(V) = {expected_loss:.4f})")

# ---- 5. Param-group decay verification ----------------------------------------
pg = build_param_groups(model, weight_decay=0.01)
decay_names, no_decay_names = [], []
for name, p, group in model.named_parameters_with_groups():
    (decay_names if group == "decay" else no_decay_names).append(name)

print(f"[OK] Decay group ({len(decay_names)} params), first 3:")
for n in decay_names[:3]:
    print(f"      {n}")
print(f"[OK] No-decay group ({len(no_decay_names)} params):")
for n in no_decay_names:
    print(f"      {n}")

required_no_decay = {
    "embeddings.W_pos",
    "layers.0.ln1.gamma", "layers.0.ln1.beta",
    "layers.0.ln2.gamma", "layers.0.ln2.beta",
    "layers.0.attn.b_qkv", "layers.0.attn.b_o",
    "layers.0.ffn.b_fc",  "layers.0.ffn.b_proj",
}
missing = required_no_decay - set(no_decay_names)
assert not missing, f"These params must be no_decay but aren't: {missing}"
print("[OK] W_pos, LN gamma/beta, all biases correctly excluded from WD.")

# ---- 6. LR schedule sanity ---------------------------------------------------
sched = LRSchedule(peak_lr=2.5e-4, min_lr=1e-5,
                   warmup_steps=2000, max_steps=100_000)
assert sched(0) < 1e-6,                          f"Step-0 LR too high: {sched(0)}"
assert abs(sched(2000) - 2.5e-4) < 1e-8,         f"Warmup-end LR wrong: {sched(2000)}"
assert sched(100_000) == 1e-5,                    f"Final LR wrong: {sched(100_000)}"
assert sched(50_000) > sched(100_000),            "Mid-training LR not above floor"
print(f"[OK] LR schedule: step0={sched(0):.2e}  "
      f"warmup_end={sched(2000):.2e}  final={sched(100_000):.2e}")

# ---- 7. Config defaults -------------------------------------------------------
from minigpt.config import ModelConfig as MC, TrainConfig as TC
mc = MC()
tc = TC()
assert mc.d_model == 768,    f"d_model={mc.d_model}"
assert mc.n_layers == 12,    f"n_layers={mc.n_layers}"
assert mc.n_heads == 12,     f"n_heads={mc.n_heads}"
assert mc.d_ff == 3072,      f"d_ff={mc.d_ff}  (expect 4*768=3072)"
assert mc.vocab_size == 40000, f"vocab_size={mc.vocab_size}"
assert mc.dropout == 0.1,    f"dropout={mc.dropout}"
assert tc.learning_rate == 2.5e-4, f"lr={tc.learning_rate}"
assert tc.beta2 == 0.98,     f"beta2={tc.beta2}  (GPT-1: 0.98)"
assert tc.weight_decay == 0.01, f"weight_decay={tc.weight_decay}"
assert tc.warmup_steps == 2000, f"warmup_steps={tc.warmup_steps}"
print(f"[OK] Config defaults: d_model={mc.d_model} n_layers={mc.n_layers} "
      f"n_heads={mc.n_heads} d_ff={mc.d_ff} vocab={mc.vocab_size} dropout={mc.dropout}")

# ---- 8. One optimizer step (no NaN) ------------------------------------------
from minigpt.optimizer import build_param_groups, Adam as _Adam
_opt = _Adam(lr=1e-4, weight_decay=0.01)
_pg = build_param_groups(model, weight_decay=0.01)
# Build fake grad groups (all-zeros) to check the step runs without error
_gg = [{'params': [xp.zeros_like(p) for p in g['params']]} for g in _pg]
_opt.step_grouped(_pg, _gg, lr=1e-4)
print("[OK] Adam.step_grouped ran without error.")

print("\n========= ALL SMOKE TESTS PASSED =========")
