"""
Adam Optimizer with L2 Weight Decay (GPT-1 style, 2018).

This is the CLASSIC Adam formulation used in the original GPT-1 paper, NOT the
modern decoupled AdamW (Loshchilov & Hutter, 2019). The difference:

    AdamW (decoupled):   p <- p - lr * (m_hat / (sqrt(v_hat) + eps) + wd * p)
    Adam + L2 (coupled): g <- g + wd * p  (then standard Adam on modified g)

In the coupled formulation, weight decay flows through the adaptive moments
(m, v), so the effective decay is scaled by 1/sqrt(v_hat). This is the
historically accurate behavior for GPT-1.

Hyperparameters (GPT-1 defaults):
    lr        = 2.5e-4
    betas     = (0.9, 0.98)
    eps       = 1e-8
    weight_decay = 0.01

RMSNorm gammas (1D) are excluded from weight decay. This is standard practice --
decaying the learnable scale of normalization layers destabilizes training. The
modern model has no biases and no learned positional table, so the no-decay
group is exactly the set of RMSNorm gammas.
"""

import math
from minigpt.backend import xp
from typing import Any, Dict, List, Optional, Tuple


class Adam:
    """
    Adam optimizer with COUPLED L2 weight decay, as used in GPT-1 (2018).

    Weight decay is added to the gradient (L2 penalty) rather than applied
    directly to the parameter after the Adam update (which would be AdamW).
    """

    def __init__(self,
                 lr: float = 2.5e-4,
                 betas: Tuple[float, float] = (0.9, 0.98),
                 eps: float = 1e-8,
                 weight_decay: float = 0.01):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # State: dict mapping param_id -> {'m': m, 'v': v, 't': 0}
        self.state: Dict[int, Dict] = {}

    def step(self, params: List, grads: List, lr: Optional[float] = None) -> None:
        """
        Updates params in-place using Adam + L2 weight decay.

        Args:
            params: list of parameter arrays
            grads:  list of gradient arrays corresponding to params
            lr:     optional learning rate override (for scheduling)
        """
        current_lr = lr if lr is not None else self.lr
        b1, b2 = self.betas
        wd = self.weight_decay

        for p, g in zip(params, grads):
            if g is None:
                continue

            p_id = id(p)
            if p_id not in self.state:
                self.state[p_id] = {
                    'm': xp.zeros_like(p),
                    'v': xp.zeros_like(p),
                    't': 0,
                }

            s = self.state[p_id]
            s['t'] += 1
            t = s['t']

            # 1. COUPLED L2 weight decay: add wd*p to the gradient.
            #    Skip 1D params (biases, LN gammas, LN betas). This naturally
            #    excludes every bias term in the GPT-1 model (QKV bias, output
            #    projection bias, FFN biases) and both LayerNorm parameters.
            if p.ndim >= 2 and wd > 0.0:
                g = g + wd * p

            # 2. Update biased first and second moment estimates.
            s['m'] = b1 * s['m'] + (1 - b1) * g
            s['v'] = b2 * s['v'] + (1 - b2) * (g * g)

            # 3. Bias-correct moments.
            m_hat = s['m'] / (1 - b1 ** t)
            v_hat = s['v'] / (1 - b2 ** t)

            # 4. Parameter update.
            p -= current_lr * m_hat / (xp.sqrt(v_hat) + self.eps)

    def step_grouped(self, param_groups: List[Dict], grad_groups: List[Dict],
                     lr: Optional[float] = None) -> None:
        """
        Per-group Adam update. Each entry in param_groups is a dict with keys
        'params' and 'weight_decay'; grad_groups mirrors the shape.

        This is the primary update path used by trainer.py — it respects the
        per-group weight_decay override (0.0 for RMSNorm gammas).
        """
        current_lr = lr if lr is not None else self.lr
        b1, b2 = self.betas

        for pg, gg in zip(param_groups, grad_groups):
            wd = pg['weight_decay']
            for p, g in zip(pg['params'], gg['params']):
                if g is None:
                    continue

                p_id = id(p)
                if p_id not in self.state:
                    self.state[p_id] = {
                        'm': xp.zeros_like(p),
                        'v': xp.zeros_like(p),
                        't': 0,
                    }

                s = self.state[p_id]
                s['t'] += 1
                t = s['t']

                # Coupled L2 weight decay (only for the decay group)
                if wd > 0.0:
                    g = g + wd * p

                s['m'] = b1 * s['m'] + (1 - b1) * g
                s['v'] = b2 * s['v'] + (1 - b2) * (g * g)

                m_hat = s['m'] / (1 - b1 ** t)
                v_hat = s['v'] / (1 - b2 ** t)

                p -= current_lr * m_hat / (xp.sqrt(v_hat) + self.eps)


# Backwards-compatibility alias. Existing training scripts import `AdamW`;
# keep the name available so the rest of the codebase does not break, but the
# implementation is now the classic Adam + L2 optimizer described above.
AdamW = Adam


class LRSchedule:
    """
    Linear warmup 0 -> peak_lr over warmup_steps, then cosine decay to min_lr
    over (max_steps - warmup_steps).

    Usage:
        sched = LRSchedule(peak_lr=2.5e-4, min_lr=1e-5,
                           warmup_steps=2000, max_steps=100_000)
        for step in range(max_steps):
            lr_now = sched(step)
            optimizer.step(params, grads, lr=lr_now)
    """

    def __init__(self, peak_lr: float, min_lr: float,
                 warmup_steps: int, max_steps: int):
        assert 0 < warmup_steps < max_steps, \
            f"warmup_steps={warmup_steps} must be in (0, max_steps={max_steps})"
        self.peak = peak_lr
        self.floor = min_lr
        self.warmup = warmup_steps
        self.total = max_steps

    def __call__(self, step: int) -> float:
        if step <= 0:
            return 0.0
        if step < self.warmup:
            return self.peak * step / self.warmup
        if step == self.warmup:
            return self.peak
        if step >= self.total:
            return self.floor
        progress = (step - self.warmup) / (self.total - self.warmup)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.floor + coeff * (self.peak - self.floor)


def build_param_groups(model: Any, weight_decay: float = 0.01) -> List[Dict[str, Any]]:
    """
    Returns a list of two param-group dicts suitable for `Adam.step_grouped`:
        [{'params': [...], 'weight_decay': weight_decay},   # 2-D weight matrices
         {'params': [...], 'weight_decay': 0.0}]            # biases, LN, W_pos

    The model is expected to expose `named_parameters_with_groups()`, which puts
    RMSNorm gammas in the no-decay group and all weight matrices in decay.
    """
    decay_params, no_decay_params = [], []
    for _, p, group in model.named_parameters_with_groups():
        if group == "decay":
            decay_params.append(p)
        else:
            no_decay_params.append(p)
    return [
        {'params': decay_params,    'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
