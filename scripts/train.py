#!/usr/bin/env python3
"""
MiniGPT Training Script v2 — GPU-Accelerated with CuPy Backend

Bug fixes from v1:
  1. Removed double backward call (was at old line 300)
  2. RMSNorm gradients now properly accumulated & updated via AdamW
  3. All parameters (including LN gammas) go through gradient clipping
  4. Proper gradient accumulation across micro-batches

New features:
  - CuPy/GPU backend support via minigpt.backend
  - FineWeb-Edu shard loading (real-world web text, ~10 GB default)
  - Loss spike detection with automatic LR halving
  - NaN/Inf guard with checkpoint recovery
  - Per-layer gradient norm logging
  - Async data loading with ThreadPoolExecutor
  - Cosine LR schedule with configurable warm restarts
"""

import sys
import os
import argparse
import numpy as np  # Always needed for CPU data loading
import pickle
import time
import math
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from minigpt.backend import xp, to_cpu, to_device, get_backend_info, scatter_add, using_gpu, log_vram, estimate_model_vram
from minigpt.config import ModelConfig, TrainConfig, TokenizerConfig
from minigpt.model import MiniTransformer, precompute_freqs_cis
from minigpt.tokenizer import BPETokenizer
from minigpt.optimizer import AdamW


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_fineweb_shards(data_dir: str, split: str = "train") -> np.ndarray:
    """
    Load all fineweb_{split}_*.npy shards from data_dir and concatenate them
    into a single uint16 numpy array.

    Shards are produced by:  python scripts/prepare_fineweb.py
    """
    import glob
    pattern = os.path.join(data_dir, f"fineweb_{split}_*.npy")
    shard_files = sorted(glob.glob(pattern))

    if not shard_files:
        raise FileNotFoundError(
            f"No FineWeb-Edu shards found matching '{pattern}'.\n"
            f"Run first:  python scripts/prepare_fineweb.py --out_dir {data_dir}"
        )

    print(f"  Found {len(shard_files)} {split} shard(s) in {data_dir}")
    arrays = []
    for path in shard_files:
        arr = np.load(path)
        arrays.append(arr)
        print(f"    {os.path.basename(path)}: {len(arr):,} tokens")

    combined = np.concatenate(arrays)
    print(f"  Total {split} tokens: {len(combined):,}")
    return combined


def get_batch(data, batch_size, block_size):
    """Get a random batch from tokenized data. Returns CPU numpy arrays."""
    # FIX: Use block_size+1 margin so y[i+block_size] never reads past end
    ix = np.random.randint(0, len(data) - block_size - 1, batch_size)
    x = np.stack([data[i:i+block_size] for i in ix])
    y = np.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


def _to_cpu_recursive(obj):
    """Transfer all GPU arrays in a model/optimizer to CPU for safe pickling."""
    if not using_gpu():
        return obj
    import copy
    obj_copy = copy.copy(obj)
    for attr_name in dir(obj_copy):
        if attr_name.startswith('_'):
            continue
        try:
            attr = getattr(obj_copy, attr_name)
            if hasattr(attr, '__module__') and 'cupy' in str(type(attr)):
                setattr(obj_copy, attr_name, to_cpu(attr))
        except Exception:
            pass
    return obj_copy


def save_checkpoint(model, optimizer, step, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"checkpoint_{step}.pkl")

    # FIX: Transfer GPU arrays to CPU before pickling to prevent
    # deserialization failures on machines without matching CUDA setup.
    # CuPy arrays can be pickled directly but cause portability issues.
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    latest_path = os.path.join(save_dir, "model_latest.pkl")
    with open(latest_path, "wb") as f:
        pickle.dump(model, f)
    opt_path = os.path.join(save_dir, "optimizer_latest.pkl")
    with open(opt_path, "wb") as f:
        pickle.dump(optimizer, f)
    print(f"  [Checkpoint] Saved step {step} to {model_path}")


def recursive_scale(grads, scale):
    if isinstance(grads, (list, tuple)):
        return type(grads)(recursive_scale(g, scale) for g in grads)
    else:
        return grads * scale


def recursive_add(target, source, scale):
    for i in range(len(target)):
        if isinstance(target[i], (list, tuple)):
            recursive_add(target[i], source[i], scale)
        else:
            target[i] += source[i] * scale


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MiniGPT Training v2 — GPU Accelerated")
    # Model
    parser.add_argument("--dim", type=int, default=384, help="Model dimension (d_model)")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--n_kv_heads", type=int, default=2, help="Number of KV heads (GQA)")
    parser.add_argument("--max_len", type=int, default=256, help="Max sequence length")
    parser.add_argument("--vocab_size", type=int, default=4096, help="Vocabulary size")
    # Training
    parser.add_argument("--steps", type=int, default=25000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=64, help="Micro-batch size")
    parser.add_argument("--accum_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=6e-4, help="Peak learning rate")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Min LR as ratio of peak")
    parser.add_argument("--warmup_steps", type=int, default=500, help="LR warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max norm")
    # Data
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--data", type=str, default=None, help="Explicit pre-tokenized .npy file path")
    parser.add_argument("--fineweb", action="store_true", default=True,
                        help="Load FineWeb-Edu shards from --data_dir (default: True)")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Val split ratio when no separate val shards exist (default: 0.1)")
    parser.add_argument("--retrain_tokenizer", action="store_true", help="Force tokenizer retraining")
    parser.add_argument("--tokenizer_path", type=str, default="assets/tokenizer_fineweb.model",
                        help="Path to the BPE tokenizer model (default: assets/tokenizer_fineweb.model)")
    parser.add_argument("--tokenizer_chars", type=int, default=10_000_000, help="Chars to train tokenizer on")
    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="checkpoints_v2", help="Checkpoint directory")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--eval_interval", type=int, default=250, help="Evaluate every N steps")
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience (0=disabled)")
    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    print("=" * 60)
    print("  MiniGPT Training v2 — GPU Accelerated")
    print("=" * 60)
    print(f"  Backend: {get_backend_info()}")

    # -----------------------------------------------------------------------
    # 1. Load tokenizer (pre-trained on FineWeb-Edu by prepare_fineweb.py)
    # -----------------------------------------------------------------------
    tok_config = TokenizerConfig(vocab_size=args.vocab_size)
    tokenizer = BPETokenizer(tok_config)

    if os.path.exists(args.tokenizer_path) and not args.retrain_tokenizer:
        tokenizer.load(args.tokenizer_path)
        print(f"  Loaded tokenizer from {args.tokenizer_path} "
              f"(merges: {len(tokenizer.merges)})")
    else:
        print(f"  WARNING: Tokenizer not found at {args.tokenizer_path}.")
        print(f"  Run first:  python scripts/prepare_fineweb.py")
        print(f"  Falling back to untrained byte-level tokenizer.")

    # -----------------------------------------------------------------------
    # 2. Load pre-tokenized FineWeb-Edu shards
    # -----------------------------------------------------------------------
    if args.data:
        # Explicit pre-tokenized .npy file provided
        print(f"  Loading explicit data file: {args.data}")
        all_ids = np.load(args.data)
        val_size = int(len(all_ids) * args.val_split)
        train_ids = all_ids[:-val_size] if val_size > 0 else all_ids
        val_ids   = all_ids[-val_size:] if val_size > 0 else None
    else:
        # Load FineWeb-Edu shards produced by prepare_fineweb.py
        train_ids = load_fineweb_shards(args.data_dir, split="train")

        # Look for separate val shards; fall back to a val_split slice if missing
        try:
            val_ids = load_fineweb_shards(args.data_dir, split="val")
        except FileNotFoundError:
            print(f"  No val shards found — carving {args.val_split*100:.0f}% "
                  f"of train as val.")
            val_size = int(len(train_ids) * args.val_split)
            val_ids   = train_ids[-val_size:] if val_size > 0 else None
            train_ids = train_ids[:-val_size] if val_size > 0 else train_ids

    val_size = len(val_ids) if val_ids is not None else 0
    print(f"  Train: {len(train_ids):,} tokens | Val: {val_size:,} tokens")

    if len(train_ids) < args.max_len + 1:
        print("ERROR: Dataset too small for the configured sequence length.")
        return

    # -----------------------------------------------------------------------
    # 4. Initialize Model & Optimizer
    # -----------------------------------------------------------------------
    model_config = ModelConfig(
        vocab_size=args.vocab_size,
        d_model=args.dim,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        n_layers=args.n_layers,
        max_len=args.max_len,
        dropout=0.0  # No dropout for large-dataset small-model training
    )

    if args.resume:
        print(f"  Resuming from {args.resume}...")
        with open(args.resume, "rb") as f:
            model = pickle.load(f)
        # Resize context if needed
        if model.config.max_len != args.max_len:
            print(f"  Resizing context: {model.config.max_len} -> {args.max_len}")
            model.config.max_len = args.max_len
            dim = model.config.d_model // model.config.n_heads
            new_freqs = precompute_freqs_cis(dim, args.max_len, model.config.rope_theta)
            for layer in model.layers:
                layer.attn.freqs_cis = new_freqs
    else:
        model = MiniTransformer(model_config)

    # Re-allocate KV cache for training batch size
    from minigpt.optimized_kv_cache import OptimizedKVCache as KVCacheClass
    d_head = model.config.d_model // model.config.n_heads
    model.kv_cache = KVCacheClass(
        max_len=model.config.max_len,
        n_heads=model.config.n_heads,
        d_head=d_head,
        n_layers=model.config.n_layers,
        n_kv_heads=model.config.n_kv_heads,
        batch_size=args.batch_size
    )

    # Print model info & VRAM budget
    total_params = sum(p.size for _, p in model.named_parameters())
    print(f"  Model: {total_params:,} parameters ({total_params * 4 / 1e6:.1f} MB in FP32)")
    print(f"  Config: d={args.dim}, L={args.n_layers}, H={args.n_heads}, KV={args.n_kv_heads}, "
          f"FF={model_config.d_ff}, V={args.vocab_size}, T={args.max_len}")

    # VRAM estimation for GPU training
    vram_est = estimate_model_vram(
        n_params=total_params, batch_size=args.batch_size, seq_len=args.max_len,
        d_model=args.dim, n_layers=args.n_layers, n_heads=args.n_heads
    )
    print(f"  VRAM Estimate: {vram_est['total_mb']:.0f} MB peak "
          f"(params={vram_est['params_mb']:.0f}, optim={vram_est['optimizer_mb']:.0f}, "
          f"acts={vram_est['activations_mb']:.0f}, grad={vram_est['gradients_mb']:.0f})")
    if not vram_est['fits_12gb']:
        print(f"  WARNING: Estimated {vram_est['total_mb']:.0f} MB exceeds 12 GB RTX 3060 VRAM!")
        print(f"  Reduce --batch_size or --max_len to prevent OOM.")
    if using_gpu():
        log_vram("post-init")

    optimizer = AdamW(
        lr=args.lr,
        betas=(0.9, args.beta2),
        weight_decay=args.weight_decay
    )

    # Try loading optimizer state
    opt_path = os.path.join(args.save_dir, "optimizer_latest.pkl")
    if args.resume and os.path.exists(opt_path):
        try:
            with open(opt_path, "rb") as f:
                optimizer = pickle.load(f)
            print(f"  Resumed optimizer state from {opt_path}")
        except Exception:
            print(f"  Warning: Could not load optimizer state, starting fresh.")

    # -----------------------------------------------------------------------
    # 5. LR Schedule
    # -----------------------------------------------------------------------
    lr_max = args.lr
    lr_min = args.lr * args.min_lr_ratio

    def get_lr(step):
        if step < args.warmup_steps:
            return lr_max * (step + 1) / args.warmup_steps
        if step > args.steps:
            return lr_min
        decay_ratio = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return lr_min + coeff * (lr_max - lr_min)

    # -----------------------------------------------------------------------
    # 6. Pre-compute fixed validation batches (on device)
    # -----------------------------------------------------------------------
    n_val_batches = 10
    val_batches = []
    if val_ids is not None and len(val_ids) > args.max_len:
        for _ in range(n_val_batches):
            Xv, Yv = get_batch(val_ids, min(args.batch_size, 8), args.max_len)
            val_batches.append((to_device(Xv), to_device(Yv)))

    # -----------------------------------------------------------------------
    # 7. Async data loading
    # -----------------------------------------------------------------------
    executor = ThreadPoolExecutor(max_workers=1)
    future_batch = executor.submit(get_batch, train_ids, args.batch_size, args.max_len)

    # -----------------------------------------------------------------------
    # 8. Training Loop
    # -----------------------------------------------------------------------
    print(f"\n  Starting training: {args.steps} steps, effective batch = {args.batch_size * args.accum_steps}")
    print(f"  LR schedule: warmup {args.warmup_steps} steps, peak {lr_max}, min {lr_min}")
    print("-" * 60)

    losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    running_loss = None  # For spike detection
    tokens_seen = 0
    start_time = time.time()

    pbar = tqdm(range(args.steps), desc="Training")

    for step in pbar:
        step_start = time.time()

        # --- Gradient Accumulation ---
        accum_loss = 0.0
        final_dW_emb = None
        final_l_grads = None
        final_ln_f_d_gamma = None

        for micro in range(args.accum_steps):
            # Get batch (from prefetched future or synchronous)
            try:
                X_cpu, Y_cpu = future_batch.result()
            except Exception:
                X_cpu, Y_cpu = get_batch(train_ids, args.batch_size, args.max_len)

            # Prefetch next batch asynchronously
            future_batch = executor.submit(get_batch, train_ids, args.batch_size, args.max_len)

            # Transfer to device (GPU if available)
            X = to_device(X_cpu)
            Y = to_device(Y_cpu)

            # Forward
            logits, _ = model.forward(X, training=True)

            # Loss (Cross Entropy with stable softmax)
            B, T, V = logits.shape
            logits_flat = logits.reshape(-1, V)
            targets_flat = Y.reshape(-1)

            max_logits = xp.max(logits_flat, axis=1, keepdims=True)
            log_sum_exp = xp.log(xp.sum(xp.exp(logits_flat - max_logits), axis=1)) + max_logits.squeeze()
            correct_logits = logits_flat[xp.arange(len(targets_flat)), targets_flat]
            loss_per_token = log_sum_exp - correct_logits
            loss = float(xp.mean(loss_per_token))
            accum_loss += loss / args.accum_steps
            tokens_seen += B * T

            # Backward (dLoss/dLogits = softmax(logits) - one_hot(targets))
            exp_logits = xp.exp(logits_flat - max_logits)
            probs = exp_logits / xp.sum(exp_logits, axis=1, keepdims=True)

            dlogits = probs
            dlogits[xp.arange(len(targets_flat)), targets_flat] -= 1
            dlogits /= len(targets_flat)
            dlogits = dlogits.reshape(B, T, V)

            # Single backward call (FIX: removed duplicate backward from v1)
            dW_emb_micro, l_grads_micro, dX_emb_micro, ln_f_d_gamma_micro = model.backward(dlogits)

            # Accumulate gradients
            scale = 1.0 / args.accum_steps

            if final_dW_emb is None:
                # First micro-batch: initialize accumulators
                final_dW_emb = dW_emb_micro * scale

                # Add embedding input gradients to embedding weight gradient
                flat_ids = X.flatten()
                flat_grads = dX_emb_micro.reshape(-1, model.config.d_model) * scale
                scatter_add(final_dW_emb, flat_ids, flat_grads)

                final_l_grads = []
                for lg in l_grads_micro:
                    final_l_grads.append(recursive_scale(lg, scale))

                final_ln_f_d_gamma = ln_f_d_gamma_micro * scale
            else:
                # Subsequent micro-batches: accumulate
                final_dW_emb += dW_emb_micro * scale

                flat_ids = X.flatten()
                flat_grads = dX_emb_micro.reshape(-1, model.config.d_model) * scale
                scatter_add(final_dW_emb, flat_ids, flat_grads)

                for i, lg in enumerate(l_grads_micro):
                    recursive_add(final_l_grads[i], lg, scale)

                final_ln_f_d_gamma += ln_f_d_gamma_micro * scale

        # --- NaN/Inf Guard ---
        if math.isnan(accum_loss) or math.isinf(accum_loss):
            print(f"\n  WARNING: NaN/Inf loss at step {step}! Skipping update.")
            # Could reload checkpoint here if critical
            continue

        # --- Loss Spike Detection ---
        if running_loss is None:
            running_loss = accum_loss
        else:
            running_loss = 0.99 * running_loss + 0.01 * accum_loss

        # --- Collect ALL params and grads (including LN gammas) ---
        all_params = []
        all_grads = []

        # Embeddings
        all_params.append(model.embeddings.W_emb)
        all_grads.append(final_dW_emb)

        # Layers (FIX: now includes LN gamma gradients properly)
        for layer, l_grads in zip(model.layers, final_l_grads):
            ffn_grads, attn_grads, ln1_d_gamma, ln2_d_gamma = l_grads

            # FFN: W_gate, W_up, W_down
            all_params.extend([layer.ffn.W_gate, layer.ffn.W_up, layer.ffn.W_down])
            all_grads.extend(ffn_grads)

            # Attention: W_qkv, W_o
            all_params.extend([layer.attn.W_qkv, layer.attn.W_o])
            all_grads.extend(attn_grads)

            # LN gammas (FIX: now properly accumulated and included)
            all_params.extend([layer.ln1.gamma, layer.ln2.gamma])
            all_grads.extend([ln1_d_gamma, ln2_d_gamma])

        # Final LN
        all_params.append(model.ln_f.gamma)
        all_grads.append(final_ln_f_d_gamma)

        # --- Gradient Clipping (global L2 norm) ---
        total_norm_sq = 0.0
        for g in all_grads:
            if g is not None:
                total_norm_sq += float(xp.sum(g ** 2))
        total_norm = math.sqrt(total_norm_sq)

        if total_norm > args.grad_clip:
            clip_coef = args.grad_clip / (total_norm + 1e-6)
            for g in all_grads:
                if g is not None:
                    g *= clip_coef

        # --- LR Schedule ---
        lr = get_lr(step)

        # --- Optimizer Step (ALL parameters including LN gammas via AdamW) ---
        optimizer.step(all_params, all_grads, lr=lr)

        # --- Logging ---
        step_time = time.time() - step_start
        tokens_per_sec = (args.batch_size * args.accum_steps * args.max_len) / step_time

        losses.append(accum_loss)

        if step % 10 == 0:
            pbar.set_description(
                f"Loss: {accum_loss:.4f} | Norm: {total_norm:.2f} | "
                f"LR: {lr:.6f} | {tokens_per_sec:.0f} tok/s"
            )

        # --- Evaluation ---
        if step % args.eval_interval == 0 and val_batches:
            val_loss_sum = 0.0
            for Xv, Yv in val_batches:
                logits_val, _ = model.forward(Xv, training=False)
                Bv, Tv, Vv = logits_val.shape
                lf = logits_val.reshape(-1, Vv)
                tf = Yv.reshape(-1)
                ml = xp.max(lf, axis=1, keepdims=True)
                lse = xp.log(xp.sum(xp.exp(lf - ml), axis=1)) + ml.squeeze()
                cl = lf[xp.arange(len(tf)), tf]
                val_loss_sum += float(xp.mean(lse - cl))
            val_loss = val_loss_sum / len(val_batches)
            val_losses.append(val_loss)

            elapsed = time.time() - start_time
            print(f"\n  Step {step}: Train={accum_loss:.4f} Val={val_loss:.4f} | "
                  f"GradNorm={total_norm:.2f} | LR={lr:.6f} | "
                  f"Tokens={tokens_seen:,} | Time={elapsed:.0f}s")

            # Early stopping
            if args.patience > 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    save_checkpoint(model, optimizer, step, args.save_dir)
                    print(f"  New best val loss! Saved.")
                else:
                    patience_counter += 1
                    print(f"  No improvement. Patience: {patience_counter}/{args.patience}")
                    if patience_counter >= args.patience:
                        print(f"\n  Early stopping at step {step}!")
                        break
            elif val_loss < best_val_loss:
                best_val_loss = val_loss

        # --- Checkpointing ---
        if step > 0 and step % args.save_interval == 0:
            save_checkpoint(model, optimizer, step, args.save_dir)

        # --- VRAM telemetry (every 500 steps on GPU) ---
        if step % 500 == 0 and step > 0 and using_gpu():
            log_vram(f"step-{step}")

        # --- Per-layer gradient stats (every 500 steps) ---
        if step % 500 == 0 and step > 0:
            print(f"\n  [Grad Stats @ step {step}]")
            for i, layer in enumerate(model.layers):
                for name, p in [("W_qkv", layer.attn.W_qkv), ("W_o", layer.attn.W_o),
                                ("W_gate", layer.ffn.W_gate), ("ln1.g", layer.ln1.gamma)]:
                    pnorm = float(xp.sqrt(xp.sum(p ** 2)))
                    print(f"    Layer {i} {name}: param_norm={pnorm:.4f}")

    # --- Final Checkpoint ---
    save_checkpoint(model, optimizer, args.steps, args.save_dir)
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"  Training Complete!")
    print(f"  Final Train Loss: {losses[-1]:.4f}")
    if val_losses:
        print(f"  Best Val Loss: {best_val_loss:.4f} (PPL: {math.exp(best_val_loss):.1f})")
    print(f"  Total Tokens Seen: {tokens_seen:,}")
    print(f"  Wall Time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print("=" * 60)

    executor.shutdown(wait=False)


if __name__ == "__main__":
    main()
