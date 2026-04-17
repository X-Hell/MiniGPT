"""
Trainer for MiniGPT (GPT-1 faithful).

Gradient tuple layout expected from model.backward():
    (dW_emb_out, dW_pos, layer_grads, dX_emb)
    layer_grads[i] = (ffn_grads, attn_grads, (ln1_dg, ln1_db), (ln2_dg, ln2_db))
    ffn_grads  = (dW_fc, db_fc, dW_proj, db_proj)
    attn_grads = (dW_qkv, db_qkv, dW_o, db_o)

This trainer uses:
  - Adam.step_grouped  — per-group coupled L2 weight decay
  - LRSchedule         — linear warmup + cosine decay
  - build_param_groups — decay / no-decay split (excludes W_pos, biases, LN params)
"""

import os
import pickle
import math
import time
from typing import Optional
import numpy as np

from .backend import xp, scatter_add, to_cpu
from .model import MiniTransformer
from .config import TrainConfig
from .optimizer import Adam, LRSchedule, build_param_groups
from .tokenizer import BPETokenizer


class Trainer:
    """Trainer for MiniGPT (GPT-1 style)."""

    def __init__(self, model: MiniTransformer, config: TrainConfig,
                 tokenizer: Optional[BPETokenizer] = None):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer

        self.optimizer = Adam(
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=getattr(config, "eps", 1e-8),
            weight_decay=config.weight_decay,
        )
        self.schedule = LRSchedule(
            peak_lr=config.learning_rate,
            min_lr=config.min_lr,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
        )
        self.steps = 0

    # -------------------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------------------

    def cross_entropy_loss(self, logits, targets):
        """
        Standard cross-entropy, no label smoothing.

        logits:  (B, T, V) — may be on GPU (xp array)
        targets: (B, T)
        Returns: (scalar loss as Python float, dlogits (B, T, V))
        """
        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        targets_flat = targets.reshape(-1)

        mx = xp.max(logits_flat, axis=1, keepdims=True)
        ex = xp.exp(logits_flat - mx)
        probs = ex / xp.sum(ex, axis=1, keepdims=True)

        N = logits_flat.shape[0]
        loss = float(-xp.mean(xp.log(probs[xp.arange(N), targets_flat] + 1e-9)))

        dlogits = probs.copy()
        dlogits[xp.arange(N), targets_flat] -= 1
        dlogits /= N

        return loss, dlogits.reshape(B, T, V)

    # -------------------------------------------------------------------------
    # Gradient clipping
    # -------------------------------------------------------------------------

    def clip_grads(self, grads):
        """
        Compute global gradient norm over all parameter gradients (excluding
        dX_emb which is scatter-added later, not a standalone parameter grad).

        grads = (dW_emb_out, dW_pos, layer_grads, dX_emb)
        """
        dW_emb_out, dW_pos, layer_grads, _dX_emb = grads
        sq = float(xp.sum(dW_emb_out ** 2)) + float(xp.sum(dW_pos ** 2))

        for ffn_g, attn_g, (ln1_dg, ln1_db), (ln2_dg, ln2_db) in layer_grads:
            for g in ffn_g:    # dW_fc, db_fc, dW_proj, db_proj
                sq += float(xp.sum(g ** 2))
            for g in attn_g:   # dW_qkv, db_qkv, dW_o, db_o
                sq += float(xp.sum(g ** 2))
            sq += (float(xp.sum(ln1_dg ** 2)) + float(xp.sum(ln1_db ** 2)) +
                   float(xp.sum(ln2_dg ** 2)) + float(xp.sum(ln2_db ** 2)))

        grad_norm = math.sqrt(sq)
        scale = 1.0
        if grad_norm > self.config.grad_clip:
            scale = self.config.grad_clip / (grad_norm + 1e-6)

        if scale < 1.0:
            def s(g):
                if isinstance(g, tuple): return tuple(s(x) for x in g)
                if isinstance(g, list):  return [s(x) for x in g]
                if hasattr(g, "__mul__"): return g * scale
                return g
            grads = s(grads)

        return grads, grad_norm

    # -------------------------------------------------------------------------
    # Parameter update
    # -------------------------------------------------------------------------

    def apply_grads(self, grads, token_ids, lr: float):
        """
        Scatter-add the input-embedding gradient into the tied weight matrix,
        then call Adam.step_grouped with the decay / no-decay groups.
        """
        dW_emb_out, dW_pos, layer_grads, dX_emb = grads

        # Accumulate input-lookup contribution into tied embedding gradient
        dW_emb_total = dW_emb_out.copy()
        flat_ids = token_ids.flatten()
        flat_grads = dX_emb.reshape(-1, self.model.config.d_model)
        scatter_add(dW_emb_total, flat_ids, flat_grads)

        # Build param groups (same order every call — determined by model topology)
        pg = build_param_groups(self.model, weight_decay=self.config.weight_decay)

        # Build a gradient lookup by param identity
        grad_map = {}
        grad_map[id(self.model.embeddings.W_emb)] = dW_emb_total
        grad_map[id(self.model.embeddings.W_pos)] = dW_pos

        for layer, (ffn_g, attn_g, (ln1_dg, ln1_db), (ln2_dg, ln2_db)) in zip(
            self.model.layers, layer_grads
        ):
            dW_fc, db_fc, dW_proj, db_proj = ffn_g
            dW_qkv, db_qkv, dW_o, db_o    = attn_g
            grad_map[id(layer.ffn.W_fc)]   = dW_fc
            grad_map[id(layer.ffn.b_fc)]   = db_fc
            grad_map[id(layer.ffn.W_proj)] = dW_proj
            grad_map[id(layer.ffn.b_proj)] = db_proj
            grad_map[id(layer.attn.W_qkv)] = dW_qkv
            grad_map[id(layer.attn.b_qkv)] = db_qkv
            grad_map[id(layer.attn.W_o)]   = dW_o
            grad_map[id(layer.attn.b_o)]   = db_o
            grad_map[id(layer.ln1.gamma)]  = ln1_dg
            grad_map[id(layer.ln1.beta)]   = ln1_db
            grad_map[id(layer.ln2.gamma)]  = ln2_dg
            grad_map[id(layer.ln2.beta)]   = ln2_db

        grad_groups = [
            {'params': [grad_map.get(id(p)) for p in g['params']]}
            for g in pg
        ]
        self.optimizer.step_grouped(pg, grad_groups, lr=lr)

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------

    def train(self, train_data: np.ndarray):
        """
        Main training loop.

        train_data: 1-D int array of token ids (NumPy, CPU)
        """
        print(f"Starting training for {self.config.max_steps} steps...")

        tokens = train_data
        B = self.config.batch_size
        T = self.config.seq_len

        start_time = time.time()

        for step in range(self.config.max_steps):
            lr_now = self.schedule(step)
            self.model.train()

            # Sample batch
            ix = np.random.randint(0, len(tokens) - T - 1, size=(B,))
            x_np = np.stack([tokens[i:i + T] for i in ix]).astype(np.int64)
            y_np = np.stack([tokens[i + 1:i + T + 1] for i in ix]).astype(np.int64)

            x = xp.asarray(x_np)
            y = xp.asarray(y_np)

            # Forward
            logits, _ = self.model.forward(x, start_pos=0, training=True)
            loss, dlogits = self.cross_entropy_loss(logits, y)

            # Backward
            grads = self.model.backward(dlogits)

            # Clip
            grads, grad_norm = self.clip_grads(grads)

            # Update
            self.apply_grads(grads, x, lr=lr_now)

            self.steps += 1

            if step % self.config.log_interval == 0:
                dt = time.time() - start_time
                print(f"Step {step:5d} | Loss: {loss:.4f} | LR: {lr_now:.2e} | "
                      f"GradNorm: {grad_norm:.3f} | Elapsed: {dt:.1f}s")

            if step % self.config.eval_interval == 0 and step > 0:
                self.save_checkpoint(f"checkpoint_{step}.pkl")

    # -------------------------------------------------------------------------
    # Checkpoint
    # -------------------------------------------------------------------------

    def save_checkpoint(self, filename: str):
        path = os.path.join(self.config.save_dir, filename)
        os.makedirs(self.config.save_dir, exist_ok=True)
        state = {
            "model": self.model,
            "optimizer_state": self.optimizer.state,
            "steps": self.steps,
            "config": self.config,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"Saved checkpoint: {path}")
