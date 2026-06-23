#!/usr/bin/env python3
"""Production training entrypoint for MiniGPT GPT-1 runs.

This script supports:
- YAML/JSON-like config files (simple key-value format)
- CLI overrides for smoke tests and production launches
- Global-norm gradient clipping
- Adam (coupled L2) with per-parameter-group weight decay exclusions
- Linear warmup + cosine decay LR schedule
- CPU-safe, self-contained checkpoints (model, optimizer, config, tokenizer)
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import math
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from minigpt.backend import (
    get_backend_info,
    log_vram,
    scatter_add,
    set_mixed_precision,
    to_cpu,
    to_device,
    using_gpu,
    xp,
)
from minigpt.config import ModelConfig, TokenizerConfig, TrainConfig
from minigpt.model import MiniTransformer
from minigpt.optimizer import Adam, LRSchedule, build_param_groups
from minigpt.stability import apply_residual_scaling
from minigpt.stability import save_checkpoint as save_rolling_checkpoint
from minigpt.tokenizer import BPETokenizer

ArrayLike = Any
NestedGrad = Union[ArrayLike, Tuple["NestedGrad", ...], List["NestedGrad"]]


@dataclass
class RunConfig:
    """Flattened run configuration resolved from file + CLI overrides."""

    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    max_len: int = 512
    vocab_size: int = 16384
    dropout: float = 0.0

    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    max_grad_norm: float = 1.0

    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    total_steps: int = 36000
    log_interval: int = 10
    checkpoint_interval: int = 1000
    eval_interval: int = 500

    fp16: bool = False
    data_path: Optional[str] = None
    tokenizer_path: Optional[str] = "assets/tokenizer_modern_16k.json"
    output_dir: str = "outputs/default_run"


class Logger:
    """Simple stdout + file logger."""

    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._fh = open(log_path, "a", encoding="utf-8")

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _parse_scalar(raw: str) -> Any:
    """Parse a scalar string from simple YAML into bool/int/float/list/str."""

    value = raw.strip()
    if not value:
        return ""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None

    try:
        return ast.literal_eval(value)
    except Exception:
        pass

    try:
        if any(c in value for c in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")


def load_config_file(path: Optional[str]) -> Dict[str, Any]:
    """Load a config file as a flat dictionary.

    Supported formats:
    - `.json`
    - simple YAML `key: value` lines (comments allowed)
    """

    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    text = open(path, "r", encoding="utf-8").read()
    if path.endswith(".json"):
        data = json.loads(text)
    else:
        data: Dict[str, Any] = {}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = _parse_scalar(value)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must deserialize to a dictionary: {path}")

    return data


def build_run_config(file_cfg: Dict[str, Any], args: argparse.Namespace) -> RunConfig:
    """Merge config-file values with CLI overrides."""

    cfg = RunConfig()

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
        "steps": "total_steps",
        "log_interval": "log_interval",
        "checkpoint_interval": "checkpoint_interval",
        "eval_interval": "eval_interval",
        "fp16": "fp16",
        "data_path": "data_path",
        "tokenizer_path": "tokenizer_path",
        "output_dir": "output_dir",
    }

    for src_key, dst_key in mapping.items():
        if src_key in file_cfg:
            setattr(cfg, dst_key, file_cfg[src_key])

    if getattr(args, "steps", None) is not None:
        cfg.total_steps = args.steps
    if getattr(args, "batch_size", None) is not None:
        cfg.batch_size = args.batch_size
    if getattr(args, "accum_steps", None) is not None:
        cfg.gradient_accumulation_steps = args.accum_steps
    if getattr(args, "log_interval", None) is not None:
        cfg.log_interval = args.log_interval
    if getattr(args, "checkpoint_interval", None) is not None:
        cfg.checkpoint_interval = args.checkpoint_interval
    if getattr(args, "eval_interval", None) is not None:
        cfg.eval_interval = args.eval_interval
    if getattr(args, "output_dir", None):
        cfg.output_dir = args.output_dir
    if getattr(args, "data_path", None):
        cfg.data_path = args.data_path
    if getattr(args, "tokenizer_path", None):
        cfg.tokenizer_path = args.tokenizer_path

    if getattr(args, "no_mixed_precision", False):
        cfg.fp16 = False

    if isinstance(cfg.betas, list):
        cfg.betas = tuple(cfg.betas)  # type: ignore[assignment]
    if not isinstance(cfg.betas, tuple) or len(cfg.betas) != 2:
        raise ValueError(f"betas must be a 2-tuple/list, got: {cfg.betas}")

    return cfg


def discover_training_tokens(data_path: Optional[str], vocab_size: int, seed: int) -> np.ndarray:
    """Load token IDs from local files, or generate a synthetic fallback stream."""

    candidate_paths: List[str] = []
    if data_path:
        candidate_paths.append(data_path)
    candidate_paths.extend([
        "data/train.bin",
        "data/fineweb_train_00000.npy",
    ])

    for path in candidate_paths:
        if path and os.path.exists(path):
            # Memory-map the token stream so it stays file-backed (reclaimable)
            # instead of committed RAM. sample_batch() only slices small windows
            # and upcasts each batch to int64, so the full stream never needs to
            # be resident or int64 (which would cost ~4x the RAM). This is the
            # 16GB-RAM-safe path for the FineWeb sample-10BT corpus.
            if path.endswith(".bin"):
                # Flat uint16 token stream written by prepare_fineweb.py.
                return np.memmap(path, dtype=np.uint16, mode="r")
            if path.endswith(".npy"):
                arr = np.load(path, mmap_mode="r")
                if np.issubdtype(arr.dtype, np.integer):
                    return arr
                return np.asarray(arr, dtype=np.int64)

    # Fallback: deterministic synthetic stream
    rng = np.random.RandomState(seed)
    return rng.randint(0, vocab_size, size=2_000_000, dtype=np.int64)


def split_train_val(tokens: np.ndarray, seq_len: int, val_fraction: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """Split token stream into train/val slices."""

    if len(tokens) < (seq_len + 2) * 2:
        raise ValueError(
            f"Token stream too short: len={len(tokens)} for seq_len={seq_len}. Need at least {(seq_len + 2) * 2}."
        )
    val_size = max(seq_len + 2, int(len(tokens) * val_fraction))
    val_tokens = tokens[-val_size:]
    train_tokens = tokens[:-val_size]
    return train_tokens, val_tokens


def sample_batch(tokens: np.ndarray, batch_size: int, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a random contiguous-token batch from a 1D token stream."""

    starts = np.random.randint(0, len(tokens) - seq_len - 1, size=batch_size)
    x = np.stack([tokens[i : i + seq_len] for i in starts], axis=0).astype(np.int64)
    y = np.stack([tokens[i + 1 : i + seq_len + 1] for i in starts], axis=0).astype(np.int64)
    return x, y


def cross_entropy_loss(logits: ArrayLike, targets: ArrayLike) -> Tuple[float, ArrayLike]:
    """Compute cross-entropy loss and dLogits."""

    bsz, seq, vocab = logits.shape
    logits_flat = logits.reshape(-1, vocab)
    targets_flat = targets.reshape(-1)

    mx = xp.max(logits_flat, axis=1, keepdims=True)
    ex = xp.exp(logits_flat - mx)
    probs = ex / xp.sum(ex, axis=1, keepdims=True)

    n = logits_flat.shape[0]
    loss = float(-xp.mean(xp.log(probs[xp.arange(n), targets_flat] + 1e-9)))

    dlogits = probs
    dlogits[xp.arange(n), targets_flat] -= 1
    dlogits /= n

    return loss, dlogits.reshape(bsz, seq, vocab)


def consolidate_tied_embedding_grad(
    model: MiniTransformer,
    grads: Tuple[ArrayLike, List[Any], ArrayLike, ArrayLike],
    token_ids: ArrayLike,
) -> Tuple[ArrayLike, List[Any], ArrayLike]:
    """Merge output-projection and input-lookup embedding gradients.

    model.backward returns (dW_emb_out, layer_grads, dX_emb, d_gamma_final).
    Returns the consolidated (dW_emb_total, layer_grads, d_gamma_final).
    """

    dW_emb_out, layer_grads, dX_emb, d_gamma_final = grads
    dW_emb_total = dW_emb_out.copy()
    flat_ids = token_ids.reshape(-1)
    flat_dX = dX_emb.reshape(-1, model.config.d_model)
    scatter_add(dW_emb_total, flat_ids, flat_dX)
    return dW_emb_total, layer_grads, d_gamma_final


def recursive_scale(value: NestedGrad, scale: float) -> NestedGrad:
    """Scale every array leaf in a nested tuple/list structure."""

    if isinstance(value, tuple):
        return tuple(recursive_scale(v, scale) for v in value)
    if isinstance(value, list):
        return [recursive_scale(v, scale) for v in value]
    return value * scale


def recursive_add(a: NestedGrad, b: NestedGrad) -> NestedGrad:
    """Elementwise add for nested tuple/list gradient structures."""

    if isinstance(a, tuple) and isinstance(b, tuple):
        return tuple(recursive_add(x, y) for x, y in zip(a, b))
    if isinstance(a, list) and isinstance(b, list):
        return [recursive_add(x, y) for x, y in zip(a, b)]
    return a + b


def global_grad_norm(grads: Tuple[ArrayLike, List[Any], ArrayLike]) -> float:
    """Compute global L2 norm across all trainable parameter gradients."""

    dW_emb, layer_grads, d_gamma_final = grads
    sq = float(xp.sum(dW_emb ** 2) + xp.sum(d_gamma_final ** 2))

    for swiglu_g, attn_g, rms1_dg, rms2_dg in layer_grads:
        for g in swiglu_g:
            sq += float(xp.sum(g ** 2))
        for g in attn_g:
            sq += float(xp.sum(g ** 2))
        sq += float(xp.sum(rms1_dg ** 2) + xp.sum(rms2_dg ** 2))

    return math.sqrt(sq)


def clip_grads(grads: Tuple[ArrayLike, List[Any], ArrayLike], max_norm: float) -> Tuple[Tuple[ArrayLike, List[Any], ArrayLike], float]:
    """Clip gradients by global norm and return clipped grads + original norm."""

    norm = global_grad_norm(grads)
    if norm <= max_norm:
        return grads, norm
    scale = max_norm / (norm + 1e-6)
    clipped = recursive_scale(grads, scale)
    return clipped, norm


def apply_grads(
    model: MiniTransformer,
    optimizer: Adam,
    grads: Tuple[ArrayLike, List[Any], ArrayLike],
    weight_decay: float,
    lr: float,
) -> None:
    """Apply consolidated gradients using grouped Adam updates."""

    dW_emb, layer_grads, d_gamma_final = grads
    grad_map: Dict[int, ArrayLike] = {
        id(model.embeddings.W_emb): dW_emb,
        id(model.final_norm.gamma): d_gamma_final,
    }

    for layer, (swiglu_g, attn_g, rms1_dg, rms2_dg) in zip(model.layers, layer_grads):
        dW_gate, dW_up, dW_down = swiglu_g
        dW_qkv, dW_o = attn_g
        grad_map[id(layer.ffn.W_gate)] = dW_gate
        grad_map[id(layer.ffn.W_up)] = dW_up
        grad_map[id(layer.ffn.W_down)] = dW_down
        grad_map[id(layer.attn.W_qkv)] = dW_qkv
        grad_map[id(layer.attn.W_o)] = dW_o
        grad_map[id(layer.rms1.gamma)] = rms1_dg
        grad_map[id(layer.rms2.gamma)] = rms2_dg

    param_groups = build_param_groups(model, weight_decay=weight_decay)
    grad_groups = [{"params": [grad_map[id(p)] for p in group["params"]]} for group in param_groups]
    optimizer.step_grouped(param_groups, grad_groups, lr=lr)


def evaluate(model: MiniTransformer, tokens: np.ndarray, batch_size: int, seq_len: int, n_batches: int = 5) -> float:
    """Evaluate mean validation loss across random batches."""

    model.eval()
    losses: List[float] = []
    for _ in range(n_batches):
        x_np, y_np = sample_batch(tokens, batch_size, seq_len)
        x = to_device(x_np)
        y = to_device(y_np)
        logits, _ = model.forward(x, training=False)
        loss, _ = cross_entropy_loss(logits, y)
        losses.append(loss)
    model.train()
    return float(np.mean(losses))


def model_state_to_cpu(model: MiniTransformer) -> Dict[str, np.ndarray]:
    """Extract a CPU float32 parameter state dictionary from the model."""

    return {name: to_cpu(param).astype(np.float32, copy=True) for name, param in model.named_parameters()}


def load_model_state(model: MiniTransformer, state: Dict[str, np.ndarray]) -> None:
    """Load model parameters from a CPU state dict into active backend tensors."""

    name_to_param = dict(model.named_parameters())
    missing = [name for name in name_to_param if name not in state]
    if missing:
        raise ValueError(f"Checkpoint missing parameters: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    for name, param in name_to_param.items():
        value = state[name]
        if tuple(value.shape) != tuple(param.shape):
            raise ValueError(
                f"Shape mismatch for {name}: checkpoint {value.shape} != model {param.shape}"
            )
        param[...] = xp.asarray(value, dtype=xp.float32)


def optimizer_state_to_cpu(optimizer: Adam, model: MiniTransformer) -> Dict[str, Dict[str, Any]]:
    """Serialize optimizer moments keyed by parameter name."""

    id_to_name = {id(param): name for name, param in model.named_parameters()}
    out: Dict[str, Dict[str, Any]] = {}
    for param_id, state in optimizer.state.items():
        name = id_to_name.get(param_id)
        if not name:
            continue
        out[name] = {
            "m": to_cpu(state["m"]).astype(np.float32, copy=True),
            "v": to_cpu(state["v"]).astype(np.float32, copy=True),
            "t": int(state["t"]),
        }
    return out


def load_optimizer_state(optimizer: Adam, model: MiniTransformer, state: Dict[str, Dict[str, Any]]) -> None:
    """Restore optimizer moments keyed by parameter name."""

    name_to_param = dict(model.named_parameters())
    optimizer.state = {}
    for name, payload in state.items():
        if name not in name_to_param:
            continue
        param = name_to_param[name]
        optimizer.state[id(param)] = {
            "m": xp.asarray(payload["m"], dtype=xp.float32),
            "v": xp.asarray(payload["v"], dtype=xp.float32),
            "t": int(payload["t"]),
        }


def bundle_tokenizer(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Bundle tokenizer state into checkpoint payload when available."""

    if not path or not os.path.exists(path):
        return None

    if path.endswith(".model"):
        tok = BPETokenizer(TokenizerConfig())
        tok.load(path)
        return {
            "format": "bpe_pickle",
            "source_path": path,
            "vocab_size": int(tok.vocab_size),
            "eos_id": int(tok.eos_id),
            "merges": tok.merges,
            "vocab": tok.vocab,
            "config": {
                "vocab_size": tok.config.vocab_size,
                "min_frequency": tok.config.min_frequency,
                "pattern": tok.config.pattern,
            },
        }

    with open(path, "rb") as fh:
        payload = fh.read()
    return {
        "format": "raw_bytes",
        "source_path": path,
        "payload": payload,
    }


def save_checkpoint(
    path: str,
    model: MiniTransformer,
    optimizer: Adam,
    run_cfg: RunConfig,
    step: int,
    best_val_loss: float,
    tokens_seen: int,
) -> None:
    """Save a self-contained, CPU-safe checkpoint."""

    # On Jetson (unified memory), reclaim CuPy's cached-but-unused pool blocks
    # before copying params+optimizer to host. Without this, the ~1.2GB host
    # copy stacks on top of the activation reserve still held by the pool and
    # OOM-kills the process. No-op on the NumPy/CPU backend.
    try:
        import cupy as _cp
        _cp.get_default_memory_pool().free_all_blocks()
        _cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

    ckpt = {
        "model_state": model_state_to_cpu(model),
        "optimizer_state": optimizer_state_to_cpu(optimizer, model),
        "model_config": vars(model.config).copy(),
        "run_config": vars(run_cfg).copy(),
        "tokenizer": bundle_tokenizer(run_cfg.tokenizer_path),
        "step": int(step),
        "best_val_loss": float(best_val_loss),
        "tokens_seen": int(tokens_seen),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(ckpt, fh)

    # Release the ~1.2GB of host float32 copies promptly so training resumes
    # with full unified-memory headroom (critical on the 8GB Jetson).
    del ckpt
    gc.collect()


def load_checkpoint(path: str, model: MiniTransformer, optimizer: Adam) -> Dict[str, Any]:
    """Load checkpoint into model/optimizer and return training metadata."""

    with open(path, "rb") as fh:
        ckpt = pickle.load(fh)

    if "model_state" not in ckpt:
        raise ValueError(f"Unsupported checkpoint format at {path}")

    ckpt_cfg = ckpt.get("model_config", {})
    ckpt_vocab = int(ckpt_cfg.get("vocab_size", -1))
    model_vocab = int(model.config.vocab_size)
    if ckpt_vocab != model_vocab:
        raise ValueError(
            f"Checkpoint vocab_size={ckpt_vocab} != model vocab_size={model_vocab}. "
            "Provide a migration script or retrain."
        )

    load_model_state(model, ckpt["model_state"])
    load_optimizer_state(optimizer, model, ckpt.get("optimizer_state", {}))

    return {
        "step": int(ckpt.get("step", 0)),
        "best_val_loss": float(ckpt.get("best_val_loss", float("inf"))),
        "tokens_seen": int(ckpt.get("tokens_seen", 0)),
        "numpy_rng_state": ckpt.get("numpy_rng_state"),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="MiniGPT GPT-1 training")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML/JSON")
    parser.add_argument("--steps", type=int, default=None, help="Override total steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Override micro-batch size")
    parser.add_argument("--accum_steps", type=int, default=None, help="Override gradient accumulation steps")
    parser.add_argument("--log_interval", type=int, default=None, help="Override log interval")
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=None,
        help="Override checkpoint save interval",
    )
    parser.add_argument("--eval_interval", type=int, default=None, help="Override eval interval")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Run output directory (overrides config; falls back to config/default)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Checkpoint path to resume")
    parser.add_argument("--data_path", type=str, default=None, help="Optional token .npy path")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Tokenizer file to bundle")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--no_mixed_precision", action="store_true", help="Disable FP16 matmuls")
    return parser.parse_args()


def main() -> int:
    """Run training end-to-end."""

    args = parse_args()
    file_cfg = load_config_file(args.config)
    run_cfg = build_run_config(file_cfg, args)

    os.makedirs(run_cfg.output_dir, exist_ok=True)
    log_path = os.path.join(run_cfg.output_dir, "training.log")
    logger = Logger(log_path)

    random.seed(args.seed)
    np.random.seed(args.seed)

    set_mixed_precision(bool(run_cfg.fp16))

    logger.log("=== MiniGPT GPT-1 Training ===")
    logger.log(f"Backend: {get_backend_info()}")
    logger.log(f"Output directory: {os.path.abspath(run_cfg.output_dir)}")

    model_cfg = ModelConfig(
        vocab_size=run_cfg.vocab_size,
        d_model=run_cfg.d_model,
        n_layers=run_cfg.n_layers,
        n_heads=run_cfg.n_heads,
        max_len=run_cfg.max_len,
        dropout=run_cfg.dropout,
    )

    train_cfg = TrainConfig(
        learning_rate=run_cfg.lr,
        min_lr=run_cfg.min_lr,
        batch_size=run_cfg.batch_size,
        accum_steps=run_cfg.gradient_accumulation_steps,
        max_steps=run_cfg.total_steps,
        warmup_steps=run_cfg.warmup_steps,
        weight_decay=run_cfg.weight_decay,
        grad_clip=run_cfg.max_grad_norm,
        beta1=float(run_cfg.betas[0]),
        beta2=float(run_cfg.betas[1]),
        eps=run_cfg.eps,
        eval_interval=run_cfg.eval_interval,
        log_interval=run_cfg.log_interval,
        save_interval=run_cfg.checkpoint_interval,
        save_dir=run_cfg.output_dir,
        seq_len=run_cfg.max_len,
    )

    model = MiniTransformer(model_cfg)
    if not args.resume_from_checkpoint:
        # I-2: verify/enforce residual-stream init (idempotent; init-only).
        apply_residual_scaling(model, model_cfg.n_layers)
    _rep = model.init_report()
    logger.log(
        f"Init report: tok_emb_std={_rep['tok_emb_std']:.4f} "
        f"has_W_pos={_rep['has_W_pos']} "
        f"residual_target_std={_rep['residual_target_std']:.5f} "
        f"out_proj[0]_std=(W_o={_rep['out_proj'][0][1]:.5f}, W_down={_rep['out_proj'][0][2]:.5f})"
    )
    optimizer = Adam(
        lr=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
        eps=train_cfg.eps,
        weight_decay=train_cfg.weight_decay,
    )
    if train_cfg.max_steps <= 1:
        schedule = lambda _step: train_cfg.learning_rate
        effective_warmup = 0
    else:
        effective_warmup = min(train_cfg.warmup_steps, train_cfg.max_steps - 1)
        schedule = LRSchedule(
            peak_lr=train_cfg.learning_rate,
            min_lr=train_cfg.min_lr,
            warmup_steps=effective_warmup,
            max_steps=train_cfg.max_steps,
        )

    tokens = discover_training_tokens(run_cfg.data_path, run_cfg.vocab_size, args.seed)
    train_tokens, val_tokens = split_train_val(tokens, seq_len=run_cfg.max_len, val_fraction=0.01)
    logger.log(
        f"Token stream loaded: train={len(train_tokens):,} val={len(val_tokens):,} (source data_path={run_cfg.data_path})"
    )

    total_params = sum(param.size for _, param in model.named_parameters())
    logger.log(f"Model params: {total_params:,}")
    logger.log(
        "Run config: "
        f"d={run_cfg.d_model} L={run_cfg.n_layers} H={run_cfg.n_heads} "
        f"V={run_cfg.vocab_size} T={run_cfg.max_len} "
        f"batch={run_cfg.batch_size} accum={run_cfg.gradient_accumulation_steps} "
        f"steps={run_cfg.total_steps}"
    )
    if effective_warmup != train_cfg.warmup_steps:
        logger.log(
            f"Adjusted warmup_steps from {train_cfg.warmup_steps} to {effective_warmup} "
            f"for total_steps={train_cfg.max_steps}."
        )

    start_step = 0
    best_val_loss = float("inf")
    tokens_seen = 0

    if args.resume_from_checkpoint:
        meta = load_checkpoint(args.resume_from_checkpoint, model, optimizer)
        start_step = meta["step"]
        best_val_loss = meta["best_val_loss"]
        tokens_seen = meta["tokens_seen"]
        if meta.get("numpy_rng_state") is not None:
            # I-5: continue the data stream instead of replaying the same early
            # batches. Older checkpoints (no RNG state) fall through unchanged.
            np.random.set_state(meta["numpy_rng_state"])
            logger.log("Restored NumPy RNG state from checkpoint.")
        logger.log(
            f"Resumed from {args.resume_from_checkpoint} at step={start_step} "
            f"best_val_loss={best_val_loss:.4f} tokens_seen={tokens_seen:,}"
        )

    model.train()
    wall_start = time.time()

    # Step-0 sanity check
    x0, y0 = sample_batch(train_tokens, run_cfg.batch_size, run_cfg.max_len)
    logits0, _ = model.forward(to_device(x0), training=True)
    init_loss, _ = cross_entropy_loss(logits0, to_device(y0))
    logger.log(f"Initial loss (sanity): {init_loss:.4f} | ln(vocab)={math.log(run_cfg.vocab_size):.4f}")

    for step in range(start_step + 1, run_cfg.total_steps + 1):
        step_start = time.time()
        accum_grads: Optional[Tuple[ArrayLike, List[Any], ArrayLike]] = None
        accum_loss = 0.0

        for _ in range(run_cfg.gradient_accumulation_steps):
            x_np, y_np = sample_batch(train_tokens, run_cfg.batch_size, run_cfg.max_len)
            x = to_device(x_np)
            y = to_device(y_np)

            logits, _ = model.forward(x, training=True)
            loss, dlogits = cross_entropy_loss(logits, y)
            accum_loss += loss / run_cfg.gradient_accumulation_steps
            tokens_seen += run_cfg.batch_size * run_cfg.max_len

            grads_raw = model.backward(dlogits)
            grads = consolidate_tied_embedding_grad(model, grads_raw, x)
            grads = recursive_scale(grads, 1.0 / run_cfg.gradient_accumulation_steps)

            if accum_grads is None:
                accum_grads = grads
            else:
                accum_grads = recursive_add(accum_grads, grads)  # type: ignore[assignment]

        if accum_grads is None:
            raise RuntimeError("Gradient accumulation produced no gradients")

        model.latest_grads = accum_grads  # pre-clip grads for monitor_gradient_norms()
        clipped_grads, grad_norm = clip_grads(accum_grads, run_cfg.max_grad_norm)
        lr_now = schedule(step)
        apply_grads(model, optimizer, clipped_grads, run_cfg.weight_decay, lr_now)

        step_time = time.time() - step_start
        tokens_per_sec = (run_cfg.batch_size * run_cfg.gradient_accumulation_steps * run_cfg.max_len) / max(step_time, 1e-9)

        if step % run_cfg.log_interval == 0 or step == 1:
            logger.log(
                f"step={step:6d} loss={accum_loss:.4f} grad_norm={grad_norm:.4f} "
                f"lr={lr_now:.6e} tok/s={tokens_per_sec:.1f}"
            )

        if step % run_cfg.eval_interval == 0:
            # I-6: 20x16 = 320 sequences (vs Run-1's 5x8=40) to cut eval variance
            # below the EvaluationMonitor's 0.05-nat regression margin.
            val_loss = evaluate(model, val_tokens, batch_size=16, seq_len=run_cfg.max_len, n_batches=20)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            logger.log(f"eval step={step:6d} val_loss={val_loss:.4f} best_val_loss={best_val_loss:.4f}")
            model.train()

        if step % run_cfg.checkpoint_interval == 0:
            # I-5: rolling (keep_last=3), atomic, with optimizer + scheduler +
            # RNG state. Refreshes latest.pkl and prunes stale step_*.pkl itself.
            meta = {
                "best_val_loss": best_val_loss,
                "tokens_seen": tokens_seen,
                "numpy_rng_state": np.random.get_state(),
            }
            ckpt_path = save_rolling_checkpoint(
                model, optimizer, schedule, step, meta,
                output_dir=run_cfg.output_dir, run_cfg=run_cfg, keep_last=3,
            )
            logger.log(f"checkpoint saved: {ckpt_path}")

        if using_gpu() and step % 100 == 0:
            log_vram(f"step-{step}")

    final_ckpt = os.path.join(run_cfg.output_dir, "checkpoints", "final.pkl")
    save_checkpoint(final_ckpt, model, optimizer, run_cfg, run_cfg.total_steps, best_val_loss, tokens_seen)

    elapsed = time.time() - wall_start
    logger.log(
        f"training complete: steps={run_cfg.total_steps} tokens_seen={tokens_seen:,} "
        f"best_val_loss={best_val_loss:.4f} wall_time={elapsed:.1f}s"
    )
    logger.log(f"Final checkpoint: {final_ckpt}")
    logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
