#!/usr/bin/env python3
"""Jetson Orin Nano 1-hour retraining script.

Workflow:
1) Load a migrated `.npz` checkpoint (NumPy->CuPy FP16 handoff output).
2) Upcast weights to FP32 model parameters (optimizer master weights).
3) Train with CuPy backend and mixed precision matmuls for exactly 3600 seconds.
4) Save `jetson_retrained_checkpoint.npz`.

The script reuses the core optimizer and gradient logic from `scripts/train.py`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Force CuPy backend before importing MiniGPT modules.
os.environ.setdefault("MINIGPT_BACKEND", "cupy")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, SRC)
sys.path.insert(0, SCRIPTS)

from minigpt.backend import get_backend_info, set_mixed_precision, to_cpu, to_device, using_gpu, xp
from minigpt.config import ModelConfig, calculate_jetson_batch_plan
from minigpt.model import MiniTransformer
from minigpt.optimizer import Adam, LRSchedule

# Reuse proven training utilities.
from train import (
    apply_grads,
    clip_grads,
    consolidate_tied_embedding_grad,
    cross_entropy_loss,
    discover_training_tokens,
    load_config_file,
    recursive_add,
    recursive_scale,
    sample_batch,
    split_train_val,
)


@dataclass
class JetsonRunConfig:
    """Jetson-focused run configuration for timed retraining."""

    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    max_len: int = 512
    vocab_size: int = 40000
    dropout: float = 0.1

    lr: float = 2.5e-4
    min_lr: float = 1e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Calculated with Jetson 8GB - 1.5GB OS - 0.5GB safety margin, T=512.
    batch_size: int = 14
    gradient_accumulation_steps: int = 5

    total_steps: int = 100000
    log_interval: int = 10
    checkpoint_interval: int = 200

    fp16: bool = True
    data_path: Optional[str] = "data/tokens_v256.npy"
    output_dir: str = "outputs/jetson_1hr"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for timed retraining."""
    parser = argparse.ArgumentParser(description="MiniGPT Jetson 1-hour retraining")
    parser.add_argument(
        "--config",
        default="configs/jetson_orin_nano_1hr.yaml",
        help="Path to YAML/JSON run config.",
    )
    parser.add_argument(
        "--init_checkpoint",
        required=True,
        help="Path to migrated Jetson init checkpoint (.npz).",
    )
    parser.add_argument(
        "--duration_seconds",
        type=int,
        default=3600,
        help="Exact wall-clock retraining duration in seconds.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--output_checkpoint",
        default="jetson_retrained_checkpoint.npz",
        help="Final checkpoint filename (inside output_dir unless absolute).",
    )
    return parser.parse_args()


def build_run_config(config_path: str) -> JetsonRunConfig:
    """Load YAML/JSON file and merge values into JetsonRunConfig defaults."""
    cfg = JetsonRunConfig()
    file_cfg = load_config_file(config_path)

    mapping = {
        "d_model": "d_model",
        "n_layers": "n_layers",
        "n_heads": "n_heads",
        "max_len": "max_len",
        "vocab_size": "vocab_size",
        "dropout": "dropout",
        "lr": "lr",
        "min_lr": "min_lr",
        "warmup_steps": "warmup_steps",
        "weight_decay": "weight_decay",
        "betas": "betas",
        "eps": "eps",
        "max_grad_norm": "max_grad_norm",
        "batch_size": "batch_size",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
        "accum_steps": "gradient_accumulation_steps",
        "total_steps": "total_steps",
        "log_interval": "log_interval",
        "checkpoint_interval": "checkpoint_interval",
        "fp16": "fp16",
        "data_path": "data_path",
        "output_dir": "output_dir",
    }
    for src_key, dst_key in mapping.items():
        if src_key in file_cfg:
            setattr(cfg, dst_key, file_cfg[src_key])

    if isinstance(cfg.betas, list):
        cfg.betas = tuple(cfg.betas)  # type: ignore[assignment]
    if not isinstance(cfg.betas, tuple) or len(cfg.betas) != 2:
        raise ValueError(f"Invalid betas value: {cfg.betas}")

    return cfg


def load_npz_checkpoint(model: MiniTransformer, optimizer: Adam, checkpoint_path: str) -> Dict[str, Any]:
    """Load model/optimizer state from Jetson `.npz` checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with np.load(checkpoint_path, allow_pickle=False) as ckpt:
        keys = set(ckpt.files)

        for name, param in model.named_parameters():
            candidates = (f"model::{name}", name)
            src_key = next((k for k in candidates if k in keys), None)
            if src_key is None:
                raise ValueError(f"Missing parameter in checkpoint: {name}")
            value = ckpt[src_key]
            if tuple(value.shape) != tuple(param.shape):
                raise ValueError(
                    f"Shape mismatch for {name}: checkpoint {value.shape} vs model {param.shape}"
                )
            # Keep trainable master params in FP32 for optimizer stability.
            param[...] = xp.asarray(value, dtype=xp.float32)

        optimizer.state = {}
        name_to_param = dict(model.named_parameters())
        for name, param in name_to_param.items():
            m_key = f"optim::{name}::m"
            v_key = f"optim::{name}::v"
            t_key = f"optim::{name}::t"
            if m_key in keys and v_key in keys and t_key in keys:
                optimizer.state[id(param)] = {
                    "m": xp.asarray(ckpt[m_key], dtype=xp.float32),
                    "v": xp.asarray(ckpt[v_key], dtype=xp.float32),
                    "t": int(ckpt[t_key]),
                }

        meta: Dict[str, Any] = {}
        if "meta_json" in keys:
            try:
                meta = json.loads(str(ckpt["meta_json"]))
            except Exception:
                meta = {}

    return meta


def save_npz_checkpoint(path: str, model: MiniTransformer, optimizer: Adam, meta: Dict[str, Any]) -> None:
    """Save a Jetson checkpoint as `.npz` with FP16 model weights."""
    out: Dict[str, np.ndarray] = {}

    for name, param in model.named_parameters():
        out[f"model::{name}"] = to_cpu(param).astype(np.float16, copy=False)

    id_to_name = {id(param): name for name, param in model.named_parameters()}
    for param_id, state in optimizer.state.items():
        pname = id_to_name.get(param_id)
        if pname is None:
            continue
        out[f"optim::{pname}::m"] = to_cpu(state["m"]).astype(np.float32, copy=False)
        out[f"optim::{pname}::v"] = to_cpu(state["v"]).astype(np.float32, copy=False)
        out[f"optim::{pname}::t"] = np.asarray(int(state["t"]), dtype=np.int64)

    out["meta_json"] = np.asarray(json.dumps(meta, separators=(",", ":")))

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)


def main() -> int:
    """Run exactly one hour of CuPy retraining on Jetson."""
    args = parse_args()
    run_cfg = build_run_config(args.config)

    if not using_gpu():
        raise RuntimeError(
            "CuPy backend is not active. Install CuPy on Jetson and set MINIGPT_BACKEND=cupy."
        )

    np.random.seed(args.seed)
    set_mixed_precision(bool(run_cfg.fp16))

    model_cfg = ModelConfig(
        vocab_size=run_cfg.vocab_size,
        d_model=run_cfg.d_model,
        n_layers=run_cfg.n_layers,
        n_heads=run_cfg.n_heads,
        max_len=run_cfg.max_len,
        dropout=run_cfg.dropout,
    )
    model = MiniTransformer(model_cfg)

    total_params = int(sum(param.size for _, param in model.named_parameters()))
    max_batch, recommended_accum, diag = calculate_jetson_batch_plan(
        n_params=total_params,
        d_model=run_cfg.d_model,
        n_layers=run_cfg.n_layers,
        n_heads=run_cfg.n_heads,
        seq_len=run_cfg.max_len,
    )

    if run_cfg.batch_size > max_batch:
        raise ValueError(
            f"Configured batch_size={run_cfg.batch_size} exceeds calculated Jetson max={max_batch} "
            f"for seq_len={run_cfg.max_len}."
        )

    optimizer = Adam(
        lr=run_cfg.lr,
        betas=(float(run_cfg.betas[0]), float(run_cfg.betas[1])),
        eps=run_cfg.eps,
        weight_decay=run_cfg.weight_decay,
    )

    ckpt_meta = load_npz_checkpoint(model, optimizer, args.init_checkpoint)

    schedule = LRSchedule(
        peak_lr=run_cfg.lr,
        min_lr=run_cfg.min_lr,
        warmup_steps=min(run_cfg.warmup_steps, max(1, run_cfg.total_steps - 1)),
        max_steps=run_cfg.total_steps,
    )

    tokens = discover_training_tokens(run_cfg.data_path, run_cfg.vocab_size, args.seed)
    train_tokens, _ = split_train_val(tokens, seq_len=run_cfg.max_len, val_fraction=0.01)

    output_dir = Path(run_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt_path = Path(args.output_checkpoint)
    if not final_ckpt_path.is_absolute():
        final_ckpt_path = output_dir / final_ckpt_path

    print("=== Jetson 1-Hour Retraining ===")
    print(f"Backend: {get_backend_info()}")
    print(f"Init checkpoint: {Path(args.init_checkpoint).resolve()}")
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Model params: {total_params:,}")
    print(
        f"Jetson budget: {diag['budget_mb']:.1f}MB | peak@max_batch({max_batch})={diag['estimated_peak_mb']:.1f}MB "
        f"| headroom={diag['headroom_mb']:.1f}MB"
    )
    print(
        f"Configured batch={run_cfg.batch_size}, accum={run_cfg.gradient_accumulation_steps} "
        f"(recommended accum for effective batch target: {recommended_accum})"
    )
    print(f"Duration target: {args.duration_seconds}s")

    model.train()
    wall_start = time.time()
    deadline = wall_start + float(args.duration_seconds)

    step = int(ckpt_meta.get("global_step", 0))
    updates_applied = 0
    tokens_seen = int(ckpt_meta.get("tokens_seen", 0))
    avg_micro_step_seconds: Optional[float] = None

    while True:
        now = time.time()
        if now >= deadline:
            break

        accum_grads: Optional[Tuple[Any, Any, Any]] = None
        accum_loss = 0.0
        micro_steps = 0

        while micro_steps < run_cfg.gradient_accumulation_steps:
            remaining = deadline - time.time()
            if remaining <= 0.0:
                break

            # Avoid starting a new micro-step when there is not enough time left.
            # This makes the run end at the exact wall-clock target after final sleep.
            if avg_micro_step_seconds is not None and remaining < (avg_micro_step_seconds * 1.05):
                break

            micro_start = time.time()
            x_np, y_np = sample_batch(train_tokens, run_cfg.batch_size, run_cfg.max_len)
            x = to_device(x_np)
            y = to_device(y_np)

            logits, _ = model.forward(x, training=True)
            loss, dlogits = cross_entropy_loss(logits, y)
            grads_raw = model.backward(dlogits)
            grads = consolidate_tied_embedding_grad(model, grads_raw, x)

            if accum_grads is None:
                accum_grads = grads
            else:
                accum_grads = recursive_add(accum_grads, grads)  # type: ignore[assignment]

            accum_loss += float(loss)
            micro_steps += 1
            tokens_seen += run_cfg.batch_size * run_cfg.max_len
            micro_elapsed = time.time() - micro_start
            if avg_micro_step_seconds is None:
                avg_micro_step_seconds = micro_elapsed
            else:
                avg_micro_step_seconds = 0.9 * avg_micro_step_seconds + 0.1 * micro_elapsed

        if accum_grads is None or micro_steps == 0:
            break

        accum_grads = recursive_scale(accum_grads, 1.0 / float(micro_steps))
        accum_loss /= float(micro_steps)

        step += 1
        updates_applied += 1

        lr_now = schedule(step)
        clipped_grads, grad_norm = clip_grads(accum_grads, run_cfg.max_grad_norm)
        apply_grads(model, optimizer, clipped_grads, run_cfg.weight_decay, lr_now)

        if updates_applied == 1 or updates_applied % run_cfg.log_interval == 0:
            elapsed = time.time() - wall_start
            remaining = max(0.0, deadline - time.time())
            tok_per_sec = tokens_seen / max(elapsed, 1e-9)
            print(
                f"update={updates_applied:6d} global_step={step:7d} loss={accum_loss:.4f} "
                f"grad_norm={grad_norm:.4f} lr={lr_now:.6e} tok/s={tok_per_sec:.1f} "
                f"elapsed={elapsed:.1f}s remaining={remaining:.1f}s"
            )

        if run_cfg.checkpoint_interval > 0 and updates_applied % run_cfg.checkpoint_interval == 0:
            inter_path = output_dir / f"jetson_intermediate_step_{step:07d}.npz"
            save_npz_checkpoint(
                str(inter_path),
                model,
                optimizer,
                {
                    "global_step": step,
                    "tokens_seen": tokens_seen,
                    "created_at": dt.datetime.utcnow().isoformat() + "Z",
                    "type": "intermediate",
                },
            )

    # Align to exact wall-clock boundary when update loop exits slightly early.
    remaining = deadline - time.time()
    if remaining > 0.0:
        time.sleep(remaining)

    elapsed = time.time() - wall_start
    final_meta = {
        "global_step": step,
        "updates_applied": updates_applied,
        "tokens_seen": tokens_seen,
        "duration_seconds": args.duration_seconds,
        "actual_elapsed_seconds": elapsed,
        "backend": get_backend_info(),
        "init_checkpoint": str(Path(args.init_checkpoint).resolve()),
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
        "model_config": {
            "d_model": run_cfg.d_model,
            "n_layers": run_cfg.n_layers,
            "n_heads": run_cfg.n_heads,
            "max_len": run_cfg.max_len,
            "vocab_size": run_cfg.vocab_size,
        },
        "train_config": {
            "batch_size": run_cfg.batch_size,
            "gradient_accumulation_steps": run_cfg.gradient_accumulation_steps,
            "lr": run_cfg.lr,
            "min_lr": run_cfg.min_lr,
            "warmup_steps": run_cfg.warmup_steps,
            "weight_decay": run_cfg.weight_decay,
            "max_grad_norm": run_cfg.max_grad_norm,
            "fp16": run_cfg.fp16,
        },
    }
    save_npz_checkpoint(str(final_ckpt_path), model, optimizer, final_meta)

    print("=== Retraining Complete ===")
    print(f"Elapsed: {elapsed:.3f}s (target={args.duration_seconds}s)")
    print(f"Updates applied: {updates_applied}")
    print(f"Final checkpoint: {final_ckpt_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
