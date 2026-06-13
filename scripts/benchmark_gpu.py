#!/usr/bin/env python3
"""Benchmark MiniGPT training throughput and VRAM usage.

Usage:
    MINIGPT_BACKEND=cupy python scripts/benchmark_gpu.py
    MINIGPT_BACKEND=numpy python scripts/benchmark_gpu.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from minigpt.backend import (
    estimate_model_vram,
    get_backend_info,
    log_vram,
    scatter_add,
    set_mixed_precision,
    to_device,
    using_gpu,
    xp,
)
from minigpt.config import ModelConfig
from minigpt.model import MiniTransformer


def cross_entropy(logits, targets):
    """Cross-entropy and dLogits."""

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

    loss = float(-xp.mean(xp.log(probs[xp.arange(n), targets_flat] + 1e-9)))
    return loss, dlogits.reshape(bsz, seq, vocab)


def benchmark_one(cfg: ModelConfig, batch_size: int, steps: int = 5) -> Dict[str, float]:
    """Run several train-like steps and report tok/s."""

    model = MiniTransformer(cfg)
    n_params = sum(param.size for _, param in model.named_parameters())
    x_np = np.random.randint(0, cfg.vocab_size, size=(batch_size, cfg.max_len), dtype=np.int64)
    y_np = np.random.randint(0, cfg.vocab_size, size=(batch_size, cfg.max_len), dtype=np.int64)

    x = to_device(x_np)
    y = to_device(y_np)

    times: List[float] = []
    losses: List[float] = []

    for idx in range(steps):
        t0 = time.time()
        logits, _ = model.forward(x, training=True)
        loss, dlogits = cross_entropy(logits, y)
        dW_emb_out, _dW_pos, _layer_grads, dX_emb = model.backward(dlogits)

        # Touch scatter_add path so tied-embedding accumulation is benchmarked too.
        dW_emb_total = dW_emb_out.copy()
        scatter_add(dW_emb_total, x.reshape(-1), dX_emb.reshape(-1, cfg.d_model))

        if using_gpu():
            xp.cuda.Stream.null.synchronize()
        dt = time.time() - t0
        times.append(dt)
        losses.append(loss)
        print(f"    step={idx} loss={loss:.4f} time={dt:.3f}s")

    avg_step = float(np.mean(times[1:])) if len(times) > 1 else float(np.mean(times))
    tokens_per_s = float((batch_size * cfg.max_len) / max(avg_step, 1e-9))

    return {
        "params": float(n_params),
        "avg_step_s": avg_step,
        "tokens_per_s": tokens_per_s,
        "loss_last": float(losses[-1]),
    }


def main() -> int:
    """Run benchmark matrix."""

    print("=" * 64)
    print("MiniGPT Benchmark")
    print(f"Backend: {get_backend_info()}")
    print("=" * 64)

    if using_gpu():
        log_vram("baseline")

    probes: List[Tuple[str, ModelConfig, int]] = [
        (
            "smoke-384x6",
            ModelConfig(vocab_size=16384, d_model=384, n_layers=6, n_heads=6, max_len=512, dropout=0.1),
            16,
        ),
        (
            "gpt1-768x12",
            ModelConfig(vocab_size=40000, d_model=768, n_layers=12, n_heads=12, max_len=512, dropout=0.1),
            4,
        ),
    ]

    for mixed in (False, True):
        set_mixed_precision(mixed)
        print(f"\n--- mixed_precision={mixed} ---")

        for label, cfg, batch_size in probes:
            print(f"\n[{label}] batch_size={batch_size}")
            est = estimate_model_vram(
                n_params=sum(param.size for _, param in MiniTransformer(cfg).named_parameters()),
                batch_size=batch_size,
                seq_len=cfg.max_len,
                d_model=cfg.d_model,
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                mixed_precision=mixed,
            )
            print(
                "  VRAM estimate: "
                f"total={est['total_mb']:.0f}MB params={est['params_mb']:.0f}MB "
                f"acts={est['activations_mb']:.0f}MB fits_12gb={est['fits_12gb']}"
            )

            out = benchmark_one(cfg, batch_size=batch_size, steps=4)
            print(
                "  result: "
                f"params={int(out['params']):,} avg_step={out['avg_step_s']:.3f}s "
                f"tok/s={out['tokens_per_s']:.1f}"
            )
            if using_gpu():
                log_vram(label)

    print("\nBenchmark complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
