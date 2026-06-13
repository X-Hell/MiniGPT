#!/usr/bin/env python3
"""Compute Jetson Orin Nano batch/accum plan for MiniGPT."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from minigpt.config import calculate_jetson_batch_plan
from minigpt.model import MiniTransformer
from minigpt.config import ModelConfig


def main() -> int:
    cfg = ModelConfig(d_model=768, n_layers=12, n_heads=12, max_len=512, vocab_size=40000)
    model = MiniTransformer(cfg)
    n_params = int(sum(param.size for _, param in model.named_parameters()))

    max_batch, accum, diag = calculate_jetson_batch_plan(
        n_params=n_params,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        seq_len=cfg.max_len,
    )

    print("=== Jetson Orin Nano Memory Budget ===")
    print(f"params={n_params:,}")
    print(f"budget_mb={diag['budget_mb']:.1f}")
    print(f"estimated_peak_mb@batch{max_batch}={diag['estimated_peak_mb']:.1f}")
    print(f"headroom_mb={diag['headroom_mb']:.1f}")
    print(f"max_batch_size={max_batch}")
    print(f"gradient_accumulation_steps={accum}")
    print(f"effective_batch={int(diag['effective_batch'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
