"""Run-2 stability + observability utilities for MiniGPT.

This module is additive: it does NOT change the model architecture or the
hand-derived backward gradient tuple. It provides the four production
implementations the Run-2 plan requires, plus a gradient spike/clustering
detector used by both the trainer and the live monitor:

    apply_residual_scaling(model, n_layers)   -> idempotent init-time verifier (I-2)
    monitor_gradient_norms(model, clip_value) -> per-step grad diagnostics (I-1/I-10)
    GradientSpikeDetector                      -> isolated-vs-clustered spikes (I-1)
    EvaluationMonitor                          -> two-confirmation eval gating (I-6)
    save_checkpoint(...)                        -> atomic, rolling, restartable (I-5)

Backend imports (`xp`, `to_cpu`) are done lazily inside the functions that need
them so that the pure-Python classes (EvaluationMonitor, GradientSpikeDetector)
can be imported by the standalone monitor and the test-suite without pulling in
NumPy/CuPy.
"""

from __future__ import annotations

import math
import os
import pickle
import shutil
import tempfile
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# I-2 · Residual-stream init verification (idempotent, init-only)
# ---------------------------------------------------------------------------
def apply_residual_scaling(model, n_layers: int):
    """Enforce GPT-2/Llama residual-stream init on output projections (I-2).

    Sets std(W_o) == std(W_down) == 0.02 / sqrt(2 * n_layers) for every block by
    rescaling in place. IDEMPOTENT and INIT-ONLY: on the current codebase the
    init already applies this (model.py FeedForward/MultiHeadAttention), so this
    is effectively a no-op verifier that also catches regressions if the init is
    ever changed.

    MUST be called immediately after construction, before any optimizer step --
    rescaling a *trained* weight would destroy it.

    Returns the model (chainable). Side effects: stamps `model._residual_scaled`
    and `model._residual_target_std`.
    """
    from minigpt.backend import xp

    target = 0.02 / math.sqrt(2.0 * max(1, n_layers))
    eps = 1e-12
    for layer in model.layers:
        for w in (layer.attn.W_o, layer.ffn.W_down):
            cur = float(xp.std(w))
            if cur > eps:
                w *= (target / cur)  # in-place rescale to the residual target std
    model._residual_scaled = True
    model._residual_target_std = target
    return model


# ---------------------------------------------------------------------------
# I-1/I-10 · Per-step gradient diagnostics
# ---------------------------------------------------------------------------
def monitor_gradient_norms(model, clip_value: float = 1.0) -> Dict[str, Any]:
    """Per-step gradient diagnostics from `model.latest_grads`.

    `model.latest_grads` is the consolidated 3-tuple the trainer builds before
    clipping: (dW_emb, layer_grads, d_gamma_final), where each
    layer_grads[i] = (swiglu_grads, attn_grads, rms1_dgamma, rms2_dgamma).

    Returns global L2 norm, whether the global-norm clip will fire, the clip
    scale, the worst-contributing layer index/norm, and spike/critical flags.
    The traversal mirrors scripts/train.py::global_grad_norm exactly so the
    numbers match the training log.
    """
    from minigpt.backend import xp

    grads = getattr(model, "latest_grads", None)
    if grads is None:
        raise RuntimeError(
            "model.latest_grads is unset; assign it (pre-clip grads) before calling."
        )
    dW_emb, layer_grads, d_gamma_final = grads
    sq = float(xp.sum(dW_emb ** 2) + xp.sum(d_gamma_final ** 2))
    per_layer = []
    for swiglu_g, attn_g, rms1_dg, rms2_dg in layer_grads:
        ls = sum(float(xp.sum(g ** 2)) for g in swiglu_g)
        ls += sum(float(xp.sum(g ** 2)) for g in attn_g)
        ls += float(xp.sum(rms1_dg ** 2) + xp.sum(rms2_dg ** 2))
        per_layer.append(math.sqrt(ls))
        sq += ls
    gnorm = math.sqrt(sq)
    worst = int(max(range(len(per_layer)), key=lambda j: per_layer[j])) if per_layer else -1
    return {
        "global_norm": gnorm,
        "will_clip": gnorm > clip_value,
        "clip_scale": (clip_value / (gnorm + 1e-6)) if gnorm > clip_value else 1.0,
        "worst_layer": worst,
        "worst_layer_norm": per_layer[worst] if per_layer else 0.0,
        "spike": gnorm > 12.0,
        "critical": gnorm > 40.0,
    }


# ---------------------------------------------------------------------------
# I-1 · Gradient spike clustering (isolated vs. sustained)
# ---------------------------------------------------------------------------
class GradientSpikeDetector:
    """Distinguish isolated gradient spikes from sustained clusters (I-1).

    Run 1 had grad_norm > 5 on 43.8% of steps (noise) but the destabilization
    that mattered was *sustained* (post-warmup mean 10.81, max 55.28). This
    detector only fires on clusters:

        WARN     : >= `warn_consec` consecutive samples above `warn`
        CRITICAL : any sample above `crit`, OR >= `crit_consec` consecutive
                   samples above `warn`

    `update` returns "WARN", "CRITICAL", or None for the latest window.
    """

    def __init__(self, warn: float = 12.0, crit: float = 40.0, window: int = 5,
                 warn_consec: int = 3, crit_consec: int = 5):
        self.warn, self.crit = warn, crit
        self.window, self.warn_consec, self.crit_consec = window, warn_consec, crit_consec
        self.recent: list[float] = []
        self.status: Optional[str] = None

    def update(self, grad_norm: float) -> Optional[str]:
        self.recent.append(float(grad_norm))
        self.recent = self.recent[-self.window:]
        consec = 0
        for g in reversed(self.recent):
            if g > self.warn:
                consec += 1
            else:
                break
        if any(g > self.crit for g in self.recent) or consec >= self.crit_consec:
            self.status = "CRITICAL"
        elif consec >= self.warn_consec:
            self.status = "WARN"
        else:
            self.status = None
        return self.status


# ---------------------------------------------------------------------------
# I-6 · Two-confirmation evaluation gating
# ---------------------------------------------------------------------------
class EvaluationMonitor:
    """Two-confirmation eval with a pending-alarm state (I-6).

    States:  HEALTHY -> PENDING (one regression) -> CONFIRMED (two consecutive).
    A new best val OR a new best train loss CANCELS a pending alarm. This makes a
    single noisy eval unable to trigger a failure call. Replaying Run 1's eval
    sequence (7.7983 -> 7.9072 -> 7.8534 -> 7.7749 -> 7.9052) with the
    interleaved train lows never reaches CONFIRMED through step 2000 and only
    reaches PENDING at step 2500.

    A "regression" is `val_loss > best_val + margin` (margin defaults to 0.05
    nats, comfortably above the 320-sequence eval noise floor).
    """

    def __init__(self, margin: float = 0.05):
        self.margin = float(margin)
        self.best_val = float("inf")
        self.best_train = float("inf")
        self.status = "HEALTHY"
        self.detail = ""
        self.pending_step: Optional[int] = None

    def record_train(self, step: int, train_loss: float) -> None:
        if train_loss < self.best_train:
            self.best_train = train_loss
            if self.status == "PENDING":  # a new train low cancels the pending alarm
                self.status = "HEALTHY"
                self.detail = f"pending cancelled by new train low {train_loss:.4f} @step {step}"
                self.pending_step = None

    def record_eval(self, step: int, val_loss: float) -> str:
        if val_loss < self.best_val - 1e-9:
            self.best_val = val_loss
            self.status = "HEALTHY"
            self.detail = f"new best val {val_loss:.4f} @step {step}"
            self.pending_step = None
            return self.status

        regressed = val_loss > self.best_val + self.margin
        if not regressed:
            self.status = "HEALTHY"  # within the noise band -- no alarm
            return self.status

        if self.status in ("HEALTHY", "CONFIRMED"):
            self.status = "PENDING"
            self.pending_step = step
            self.detail = (f"val {val_loss:.4f} > best {self.best_val:.4f}"
                           f"+{self.margin} @step {step} (1/2)")
        elif self.status == "PENDING":
            self.status = "CONFIRMED"
            self.detail = (f"2 consecutive regressions, last val {val_loss:.4f} "
                           f"@step {step}")
        return self.status


# ---------------------------------------------------------------------------
# I-5 · Atomic, rolling, fully-restartable checkpoint
# ---------------------------------------------------------------------------
def save_checkpoint(model, optimizer, scheduler, step: int, metadata: dict,
                    *, output_dir: str, run_cfg=None, keep_last: int = 3) -> str:
    """Atomic, rolling, fully-restartable checkpoint (I-5).

    Saves model params, optimizer moments (m, v, t), an explicit scheduler block
    (LRSchedule is stateless -> store its config so a resume reconstructs the
    EXACT LR even if the config file later changes), the step, the NumPy RNG
    state (so the data stream continues rather than restarting), and caller
    metadata. Writes to a temp file + os.replace (atomic), refreshes latest.pkl,
    and prunes numbered checkpoints to `keep_last`.

    The payload is a superset of the schema scripts/train.py::load_checkpoint
    already reads, so existing load logic stays compatible.

    Returns the written checkpoint path.
    """
    from minigpt.backend import to_cpu

    # Reclaim CuPy pool blocks before the host copy (unified-memory safe; no-op on CPU).
    try:
        import cupy as _cp
        _cp.get_default_memory_pool().free_all_blocks()
        _cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

    sched = {
        "type": "LRSchedule",
        "peak_lr": getattr(scheduler, "peak", None),
        "min_lr": getattr(scheduler, "floor", None),
        "warmup_steps": getattr(scheduler, "warmup", None),
        "max_steps": getattr(scheduler, "total", None),
    }

    id_to_name = {id(p): n for n, p in model.named_parameters()}
    opt_state = {
        id_to_name[i]: {
            "m": to_cpu(s["m"]).astype(np.float32, copy=True),
            "v": to_cpu(s["v"]).astype(np.float32, copy=True),
            "t": int(s["t"]),
        }
        for i, s in optimizer.state.items() if i in id_to_name
    }

    ckpt = {
        "model_state": {n: to_cpu(p).astype(np.float32, copy=True)
                        for n, p in model.named_parameters()},
        "optimizer_state": opt_state,
        "scheduler": sched,
        "model_config": vars(model.config).copy(),
        "run_config": (vars(run_cfg).copy() if run_cfg is not None else {}),
        "step": int(step),
        "metadata": dict(metadata),
        "best_val_loss": float(metadata.get("best_val_loss", float("inf"))),
        "tokens_seen": int(metadata.get("tokens_seen", 0)),
        "numpy_rng_state": metadata.get("numpy_rng_state"),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    ck_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)
    path = os.path.join(ck_dir, f"step_{step:07d}.pkl")

    fd, tmp = tempfile.mkstemp(dir=ck_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            pickle.dump(ckpt, fh)
        os.replace(tmp, path)  # atomic on POSIX
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    shutil.copyfile(path, os.path.join(ck_dir, "latest.pkl"))

    numbered = sorted(f for f in os.listdir(ck_dir)
                      if f.startswith("step_") and f.endswith(".pkl"))
    for stale in numbered[:-keep_last] if keep_last > 0 else []:
        try:
            os.remove(os.path.join(ck_dir, stale))
        except OSError:
            pass

    del ckpt
    return path
