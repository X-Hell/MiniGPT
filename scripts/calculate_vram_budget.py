#!/usr/bin/env python3
"""Estimate VRAM budget for GPT-1 72h training configuration."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from minigpt.backend import estimate_model_vram
from minigpt.config import ModelConfig
from minigpt.model import MiniTransformer


def main() -> int:
    """Compute batch-size feasibility and recommended accumulation."""

    cfg = ModelConfig(
        d_model=768,
        n_layers=12,
        n_heads=12,
        max_len=512,
        vocab_size=40000,
        dropout=0.1,
    )

    model = MiniTransformer(cfg)
    n_params = sum(param.size for _, param in model.named_parameters())

    candidates = [8, 16, 24, 32, 48, 64, 96, 128]
    safety_limit_mb = 11_000.0  # 12GB card minus ~1GB safety margin

    print("n_params:", n_params)
    print("=" * 50)

    best_batch = None
    best_exact_batch = None
    for batch_size in candidates:
        est = estimate_model_vram(
            n_params=n_params,
            batch_size=batch_size,
            seq_len=cfg.max_len,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            mixed_precision=True,
        )
        total_mb = est["total_mb"]
        status = "✅ Fits in 12GB" if total_mb <= safety_limit_mb else "❌ Exceeds 12GB safety target"
        print(f"batch_size={batch_size:3d} -> VRAM: {total_mb / 1024:.2f} GB ({total_mb:.0f} MB)  {status}")
        if total_mb <= safety_limit_mb:
            best_batch = batch_size
            if 64 % batch_size == 0:
                best_exact_batch = batch_size

    print("\n" + "=" * 50)
    print("RECOMMENDED CONFIG FOR 72H TRAINING:")

    if best_batch is None:
        print("No candidate fits under the safety limit. Reduce model size or sequence length.")
        return 1

    recommended_batch = best_exact_batch if best_exact_batch is not None else best_batch
    accum_steps = math.ceil(64 / recommended_batch)
    if best_exact_batch is not None:
        print("Selection policy: largest safe batch_size that preserves effective_batch=64 exactly")
    else:
        print("Selection policy: largest safe batch_size with nearest >=64 effective batch")
    print(f"batch_size: {recommended_batch}")
    print(f"gradient_accumulation_steps: {accum_steps}")
    print(f"effective_batch_size: {recommended_batch * accum_steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
