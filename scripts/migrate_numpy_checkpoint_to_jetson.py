#!/usr/bin/env python3
"""Secure NumPy -> Jetson CuPy checkpoint migration.

This script loads a MacBook-trained NumPy checkpoint (`.npz` or `.npy`) with
`allow_pickle=False` by default, validates shapes against MiniTransformer, casts
weights through CuPy FP16 tensors, and saves a Jetson-ready `.npz` checkpoint.

Output checkpoint format:
  - model::<param_name>  -> FP16 NumPy arrays (serialized from CuPy tensors)
  - meta_json            -> JSON metadata string
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Force CuPy backend before MiniGPT imports.
os.environ.setdefault("MINIGPT_BACKEND", "cupy")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from minigpt.backend import get_backend_info, to_cpu, using_gpu, xp
from minigpt.config import ModelConfig
from minigpt.model import MiniTransformer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for migration."""
    parser = argparse.ArgumentParser(description="Migrate NumPy checkpoint to Jetson CuPy FP16 format")
    parser.add_argument("--input", required=True, help="Path to source .npz or .npy checkpoint")
    parser.add_argument(
        "--output",
        default="checkpoints/jetson_init_fp16.npz",
        help="Path to output Jetson checkpoint (.npz)",
    )
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=40000)
    parser.add_argument(
        "--allow_pickle",
        action="store_true",
        help="Allow pickle loading for legacy object-dtype .npy checkpoints (less secure).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Require every model parameter to be present in the source checkpoint.",
    )
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Allow unresolved parameters (disables strict mode).",
    )
    return parser.parse_args()


def normalize_key(key: str) -> str:
    """Normalize source parameter key naming conventions."""
    clean = key.strip().replace("/", ".")
    for prefix in ("model::", "model.", "model_state.", "state_dict.", "params.", "module."):
        if clean.startswith(prefix):
            clean = clean[len(prefix) :]
    return clean


def load_source_arrays(path: Path, allow_pickle: bool) -> Dict[str, np.ndarray]:
    """Load source checkpoint arrays into a flat mapping."""
    if not path.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=allow_pickle) as payload:
            return {k: payload[k] for k in payload.files}

    if suffix == ".npy":
        arr = np.load(path, allow_pickle=allow_pickle)

        # Structured-array payload: fields become parameter names.
        if arr.dtype.fields:
            return {name: arr[name] for name in arr.dtype.fields.keys()}

        # Object scalar dict payload (legacy checkpoints).
        if arr.shape == () and arr.dtype == object:
            obj = arr.item()
            if isinstance(obj, dict):
                out: Dict[str, np.ndarray] = {}
                for key, value in obj.items():
                    out[str(key)] = np.asarray(value)
                return out

        raise ValueError(
            "Unsupported .npy checkpoint format. Use .npz with named tensors, "
            "or pass --allow_pickle for legacy object-dict .npy files."
        )

    raise ValueError(f"Unsupported checkpoint extension: {suffix}")


def resolve_mapping(
    source: Dict[str, np.ndarray],
    target_shapes: Dict[str, Tuple[int, ...]],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Map source keys to model parameter names.

    Returns:
      - mapping: target_name -> source_key
      - unresolved: target_name -> reason
    """
    normalized_source = {normalize_key(k): k for k in source.keys()}
    mapping: Dict[str, str] = {}
    unresolved: Dict[str, str] = {}

    for target_name, target_shape in target_shapes.items():
        if target_name in normalized_source:
            src_key = normalized_source[target_name]
            if tuple(source[src_key].shape) == tuple(target_shape):
                mapping[target_name] = src_key
                continue

        # Fallback: suffix match with exact shape.
        candidates = []
        for norm_key, raw_key in normalized_source.items():
            if norm_key.endswith(target_name) or target_name.endswith(norm_key):
                if tuple(source[raw_key].shape) == tuple(target_shape):
                    candidates.append(raw_key)

        if len(candidates) == 1:
            mapping[target_name] = candidates[0]
        elif len(candidates) > 1:
            unresolved[target_name] = f"ambiguous candidates={candidates}"
        else:
            unresolved[target_name] = "missing"

    return mapping, unresolved


def main() -> int:
    """Run checkpoint migration."""
    args = parse_args()
    if args.allow_missing:
        args.strict = False

    if not using_gpu():
        raise RuntimeError(
            "CuPy backend is not active. Install CuPy on Jetson and set MINIGPT_BACKEND=cupy."
        )

    model_cfg = ModelConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_len=args.max_len,
        dropout=0.1,
    )
    model = MiniTransformer(model_cfg)

    target_shapes = {name: tuple(param.shape) for name, param in model.named_parameters()}
    source = load_source_arrays(Path(args.input), allow_pickle=args.allow_pickle)

    mapping, unresolved = resolve_mapping(source, target_shapes)
    if args.strict and unresolved:
        missing = "\n".join(f"  - {k}: {v}" for k, v in unresolved.items())
        raise ValueError(f"Strict mode failed, unresolved parameters:\n{missing}")

    out: Dict[str, np.ndarray] = {}
    migrated = 0
    for name, param in model.named_parameters():
        if name not in mapping:
            continue
        src_key = mapping[name]
        src = np.asarray(source[src_key])

        if tuple(src.shape) != tuple(param.shape):
            raise ValueError(
                f"Shape mismatch for {name}: source {src.shape} != target {param.shape}"
            )

        # Cast through CuPy to guarantee Jetson-compatible device conversion path.
        gpu_fp16 = xp.asarray(src, dtype=xp.float16)
        out[f"model::{name}"] = to_cpu(gpu_fp16).astype(np.float16, copy=False)
        migrated += 1

    meta = {
        "source_checkpoint": str(Path(args.input).resolve()),
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
        "backend": get_backend_info(),
        "dtype": "float16",
        "migrated_params": migrated,
        "total_params": len(target_shapes),
        "strict": bool(args.strict),
        "unresolved": unresolved,
        "model_config": {
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "max_len": args.max_len,
            "vocab_size": args.vocab_size,
        },
    }
    out["meta_json"] = np.asarray(json.dumps(meta, separators=(",", ":")))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **out)

    print("=== Migration Complete ===")
    print(f"Backend: {get_backend_info()}")
    print(f"Input:   {Path(args.input).resolve()}")
    print(f"Output:  {output.resolve()}")
    print(f"Weights migrated: {migrated}/{len(target_shapes)}")
    if unresolved:
        print(f"Unresolved parameters: {len(unresolved)} (see meta_json in output checkpoint)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
