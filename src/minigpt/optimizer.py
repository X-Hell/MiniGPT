from minigpt.backend import xp
from typing import List, Tuple, Dict, Optional

class AdamW:
    """AdamW Optimizer with decoupled weight decay and simplified interface."""
    def __init__(self, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.95),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # State: dict mapping param_id -> {'m': m, 'v': v, 't': 0}
        self.state: Dict[int, Dict] = {}

    def step(self, params: List, grads: List, lr: Optional[float] = None) -> None:
        """
        Updates params in-place.
        Args:
            params: list of parameter arrays
            grads: list of gradient arrays corresponding to params
            lr: optional learning rate override (for scheduling)
        """
        current_lr = lr if lr is not None else self.lr
        b1, b2 = self.betas

        for p, g in zip(params, grads):
            if g is None:
                continue

            p_id = id(p)
            if p_id not in self.state:
                self.state[p_id] = {
                    'm': xp.zeros_like(p),
                    'v': xp.zeros_like(p),
                    't': 0
                }

            s = self.state[p_id]
            s['t'] += 1
            t = s['t']

            # AdamW logic
            # 1. Weight Decay (applied to param directly, before momentum)
            # FIX: Skip weight decay for 1D params (LN gammas, biases).
            # Standard practice (GPT-2, LLaMA): weight decay on 2D+ weight
            # matrices only. Decaying LN gammas pushes them toward zero,
            # destabilizing the learnable scale in RMSNorm.
            if p.ndim >= 2:
                p -= current_lr * self.weight_decay * p

            # 2. Moments
            s['m'] = b1 * s['m'] + (1 - b1) * g
            s['v'] = b2 * s['v'] + (1 - b2) * (g ** 2)

            # 3. Bias Correction
            m_hat = s['m'] / (1 - b1 ** t)
            v_hat = s['v'] / (1 - b2 ** t)

            # 4. Update
            p -= current_lr * m_hat / (xp.sqrt(v_hat) + self.eps)
