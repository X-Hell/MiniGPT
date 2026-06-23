#!/usr/bin/env python3
"""Compute the VRAM-maxing batch/accum config for the modern ~30M model on a 3060.

Targets ~11.5GB peak on the RTX 3060 12GB (FP16 mixed) at T=512, reaching an
effective batch of at least 128 via gradient accumulation.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from minigpt.backend import estimate_model_vram
from minigpt.config import ModelConfig, solve_batch_plan
from minigpt.model import MiniTransformer


def main() -> int:
    cfg = ModelConfig()  # modern ~30M defaults: d=512, L=8, H=8, d_ff=1024, V=16384

    model = MiniTransformer(cfg)
    n_params = sum(param.size for _, param in model.named_parameters())

    vram_budget_mb = 11_500.0          # 12GB card, ~500MB headroom
    target_effective_batch = 128
    # The estimator now models the FP32 logits/dlogits and backward recompute
    # (I-7, calibrated to the measured 11,294 MB @ batch 32), so the old 9,000 MB
    # derate hack is no longer needed -- recommend against the real budget.
    # benchmark_gpu.py remains the authoritative check on the real card.
    recommend_budget_mb = vram_budget_mb

    print(f"Model: d={cfg.d_model} L={cfg.n_layers} H={cfg.n_heads} "
          f"d_ff={cfg.d_ff} V={cfg.vocab_size} T={cfg.max_len}")
    print(f"Params: {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"VRAM budget: {vram_budget_mb:.0f} MB | target effective batch: {target_effective_batch}")
    print("=" * 64)

    for b in [8, 16, 24, 32, 48, 64, 96, 128]:
        est = estimate_model_vram(
            n_params=n_params, batch_size=b, seq_len=cfg.max_len,
            d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
            d_ff=cfg.d_ff, vocab_size=cfg.vocab_size, mixed_precision=False,
        )
        total = est["total_mb"]
        status = "fits" if total <= vram_budget_mb else "OOM"
        print(f"  micro_batch={b:4d} -> {total/1024:5.2f} GB ({total:6.0f} MB)  "
              f"[params {est['params_mb']:.0f} | optim {est['optimizer_mb']:.0f} | "
              f"act {est['activations_mb']:.0f}]  {status}")

    micro, accum, diag = solve_batch_plan(
        n_params=n_params, d_model=cfg.d_model, n_layers=cfg.n_layers,
        n_heads=cfg.n_heads, d_ff=cfg.d_ff, seq_len=cfg.max_len,
        vocab_size=cfg.vocab_size,
        vram_budget_mb=recommend_budget_mb, target_effective_batch=target_effective_batch,
    )

    print("=" * 64)
    print(f"RECOMMENDED RTX 3060 CONFIG (derated budget {recommend_budget_mb:.0f} MB "
          f"for FP32 backward recompute; verify with benchmark_gpu.py):")
    print(f"  batch_size                  : {micro}")
    print(f"  gradient_accumulation_steps : {accum}")
    print(f"  effective_batch             : {int(diag['effective_batch'])}")
    print(f"  estimated peak VRAM         : {diag['estimated_peak_mb']/1024:.2f} GB "
          f"({diag['estimated_peak_mb']:.0f} MB, headroom {diag['headroom_mb']:.0f} MB)")
    print()
    print("  Launch:")
    print(f"    MINIGPT_BACKEND=cupy python scripts/train.py \\")
    print(f"        --batch_size {micro} --accum_steps {accum} \\")
    print(f"        --data_path data/train.bin --output_dir outputs/modern_30m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
