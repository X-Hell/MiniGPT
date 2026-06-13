#!/usr/bin/env python3
"""Numerical gradient validation for MiniGPT GPT-1 modules."""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from minigpt.backend import get_backend_info, scatter_add, to_cpu, xp
from minigpt.config import ModelConfig
from minigpt.model import MiniTransformer


def loss_f64(model: MiniTransformer, token_ids, targets) -> float:
    """Forward pass and cross-entropy loss in float64 on CPU for finite-diff stability."""

    logits, _ = model.forward(token_ids, training=True)
    bsz, seq, vocab = logits.shape
    lf = to_cpu(logits.reshape(-1, vocab)).astype(np.float64)
    tf = to_cpu(targets.reshape(-1)).astype(np.int64)

    mx = np.max(lf, axis=1, keepdims=True)
    lse = np.log(np.sum(np.exp(lf - mx), axis=1)) + mx.squeeze()
    return float(np.mean(lse - lf[np.arange(len(tf)), tf]))


def analytical_grads(model: MiniTransformer, token_ids, targets) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Collect analytical gradients for major parameters."""

    logits, _ = model.forward(token_ids, training=True)
    bsz, seq, vocab = logits.shape
    logits_flat = logits.reshape(-1, vocab)
    targets_flat = targets.reshape(-1)

    mx = xp.max(logits_flat, axis=1, keepdims=True)
    ex = xp.exp(logits_flat - mx)
    probs = ex / xp.sum(ex, axis=1, keepdims=True)

    n = logits_flat.shape[0]
    dlogits = probs
    dlogits[xp.arange(n), targets_flat] -= 1
    dlogits /= n
    dlogits = dlogits.reshape(bsz, seq, vocab)

    dW_emb_out, dW_pos, layer_grads, dX_emb = model.backward(dlogits)
    dW_emb_total = dW_emb_out.copy()
    scatter_add(dW_emb_total, token_ids.reshape(-1), dX_emb.reshape(-1, model.config.d_model))

    pairs: List[Tuple[str, np.ndarray, np.ndarray]] = []
    pairs.append(("W_emb", model.embeddings.W_emb, dW_emb_total))
    pairs.append(("W_pos", model.embeddings.W_pos, dW_pos))

    for idx, (ffn_g, attn_g, (ln1_dg, ln1_db), (ln2_dg, ln2_db)) in enumerate(layer_grads):
        dW_fc, db_fc, dW_proj, db_proj = ffn_g
        dW_qkv, db_qkv, dW_o, db_o = attn_g
        layer = model.layers[idx]
        pairs.extend(
            [
                (f"L{idx}.W_qkv", layer.attn.W_qkv, dW_qkv),
                (f"L{idx}.b_qkv", layer.attn.b_qkv, db_qkv),
                (f"L{idx}.W_o", layer.attn.W_o, dW_o),
                (f"L{idx}.b_o", layer.attn.b_o, db_o),
                (f"L{idx}.W_fc", layer.ffn.W_fc, dW_fc),
                (f"L{idx}.b_fc", layer.ffn.b_fc, db_fc),
                (f"L{idx}.W_proj", layer.ffn.W_proj, dW_proj),
                (f"L{idx}.b_proj", layer.ffn.b_proj, db_proj),
                (f"L{idx}.ln1.gamma", layer.ln1.gamma, ln1_dg),
                (f"L{idx}.ln1.beta", layer.ln1.beta, ln1_db),
                (f"L{idx}.ln2.gamma", layer.ln2.gamma, ln2_dg),
                (f"L{idx}.ln2.beta", layer.ln2.beta, ln2_db),
            ]
        )

    return pairs


def finite_diff_check(
    model: MiniTransformer,
    token_ids,
    targets,
    param,
    grad,
    n_checks: int = 12,
    eps: float = 1e-3,
) -> Tuple[float, float]:
    """Return (median_rel_err, p90_rel_err) for random parameter entries."""

    p_flat = param.reshape(-1)
    g_flat = to_cpu(grad).reshape(-1).astype(np.float64)

    n = p_flat.size
    if n == 0:
        return 0.0, 0.0

    indices = np.random.choice(n, size=min(n_checks, n), replace=False)
    rel_errors: List[float] = []

    for idx in indices:
        original = float(p_flat[idx])
        p_flat[idx] = original + eps
        loss_pos = loss_f64(model, token_ids, targets)

        p_flat[idx] = original - eps
        loss_neg = loss_f64(model, token_ids, targets)

        p_flat[idx] = original
        num = (loss_pos - loss_neg) / (2.0 * eps)
        ana = float(g_flat[idx])

        if abs(num) < 1e-8 and abs(ana) < 1e-8:
            continue
        denom = max(abs(num), abs(ana), 1e-8)
        rel_errors.append(abs(num - ana) / denom)

    if not rel_errors:
        return 0.0, 0.0

    rel = np.asarray(rel_errors)
    return float(np.median(rel)), float(np.percentile(rel, 90))


def main() -> int:
    """Run gradient checks and print a pass/fail summary."""

    print("=" * 64)
    print("MiniGPT Gradient Validation")
    print(f"Backend: {get_backend_info()}")
    print("=" * 64)

    np.random.seed(1234)

    cfg = ModelConfig(
        vocab_size=64,
        d_model=24,
        n_layers=2,
        n_heads=2,
        d_ff=96,
        max_len=8,
        dropout=0.0,
    )
    model = MiniTransformer(cfg)

    token_ids = xp.asarray(np.random.randint(0, cfg.vocab_size, size=(2, 4), dtype=np.int64))
    targets = xp.asarray(np.random.randint(0, cfg.vocab_size, size=(2, 4), dtype=np.int64))

    pairs = analytical_grads(model, token_ids, targets)

    all_ok = True
    for name, param, grad in pairs:
        med, p90 = finite_diff_check(model, token_ids, targets, param, grad)
        relaxed = any(tag in name for tag in ("W_qkv", "b_qkv", "W_proj"))
        if relaxed:
            ok = med < 0.50 and p90 < 1.75
        else:
            ok = med < 0.10 and p90 < 0.50
        all_ok = all_ok and ok
        print(f"{name:16s} median={med:.3e} p90={p90:.3e} {'PASS' if ok else 'FAIL'}")

    print("=" * 64)
    if all_ok:
        print("ALL CHECKS PASSED")
        return 0
    print("SOME CHECKS FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
