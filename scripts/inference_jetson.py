#!/usr/bin/env python3
"""Headless inference validation for Jetson checkpoint.

Loads a retrained `.npz` checkpoint and generates 100 tokens using nucleus
sampling (top-p).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Use CuPy backend on Jetson by default.
os.environ.setdefault("MINIGPT_BACKEND", "cupy")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from minigpt.backend import get_backend_info, set_mixed_precision
from minigpt.config import ModelConfig, TokenizerConfig
from minigpt.inference import InferenceEngine, load_jetson_checkpoint_npz
from minigpt.model import MiniTransformer
from minigpt.tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="MiniGPT Jetson inference validation")
    parser.add_argument(
        "--checkpoint",
        default="outputs/jetson_1hr/jetson_retrained_checkpoint.npz",
        help="Path to Jetson retrained checkpoint (.npz)",
    )
    parser.add_argument(
        "--tokenizer",
        default="assets/tokenizer.model",
        help="Path to tokenizer model",
    )
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--max_tokens", type=int, default=100, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.90, help="Nucleus sampling top-p")
    parser.add_argument("--top_k", type=int, default=0, help="Optional top-k cutoff; 0 disables")
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=40000)
    return parser.parse_args()


def main() -> int:
    """Run one headless generation for validation."""
    args = parse_args()

    set_mixed_precision(True)

    model_cfg = ModelConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_len=args.max_len,
        dropout=0.0,
    )
    model = MiniTransformer(model_cfg).eval()

    meta = load_jetson_checkpoint_npz(model, args.checkpoint)

    # If metadata includes a model config, verify compatibility before inference.
    meta_cfg = meta.get("model_config") if isinstance(meta, dict) else None
    if isinstance(meta_cfg, dict):
        for key in ("d_model", "n_layers", "n_heads", "max_len", "vocab_size"):
            if key in meta_cfg and int(meta_cfg[key]) != int(getattr(model_cfg, key)):
                raise ValueError(
                    f"Model config mismatch for {key}: checkpoint={meta_cfg[key]} current={getattr(model_cfg, key)}"
                )

    tok = BPETokenizer(TokenizerConfig(vocab_size=model_cfg.vocab_size))
    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")
    tok.load(args.tokenizer)

    engine = InferenceEngine(model, tok)
    texts, stats_list = engine.generate(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=1.05,
        num_return_sequences=1,
    )

    output_text = texts[0]
    stats = stats_list[0]
    avg_conf = sum(s["confidence"] for s in stats) / max(1, len(stats))
    avg_entropy = sum(s["entropy"] for s in stats) / max(1, len(stats))

    print("=== Jetson Inference Validation ===")
    print(f"Backend: {get_backend_info()}")
    print(f"Checkpoint: {Path(args.checkpoint).resolve()}")
    print(f"Prompt: {args.prompt}")
    print("--- Generated (100-token target) ---")
    print(output_text)
    print("--- Metrics ---")
    print(f"generated_tokens={len(stats)} avg_confidence={avg_conf:.4f} avg_entropy={avg_entropy:.4f}")

    if meta:
        print("--- Checkpoint Meta ---")
        print(json.dumps(meta, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
