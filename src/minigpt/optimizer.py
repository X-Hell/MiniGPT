import numpy as np
from typing import List, Tuple, Dict, Optional

class CosineSchedule:
    def __init__(self, warmup_steps: int, max_steps: int, lr_min: float, lr_max: float):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.lr_min = lr_min
        self.lr_max = lr_max

    def get_lr(self, step: int) -> float:
        # 1. Warmup
        if step < self.warmup_steps:
             return self.lr_max * (step + 1) / self.warmup_steps
        
        # 2. Min LR after max_steps
        if step > self.max_steps:
            return self.lr_min
            
        # 3. Cosine Decay
        decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
        return self.lr_min + coeff * (self.lr_max - self.lr_min)

class AdamW:
    """AdamW Optimizer with decoupled weight decay and simplified interface."""
    def __init__(self, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999), 
                 eps: float = 1e-8, weight_decay: float = 0.01):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # State: dict mapping param_id -> {'m': m, 'v': v, 't': 0}
        self.state: Dict[int, Dict[str, np.ndarray]] = {}
        
    @staticmethod
    def clip_grad_norm(grads: List[np.ndarray], max_norm: float) -> float:
        """
        Clips gradient norm of an iterable of parameters.
        Returns the total norm.
        """
        total_norm = 0.0
        for g in grads:
            if g is not None:
                param_norm = np.sum(g ** 2)
                total_norm += param_norm
        total_norm = np.sqrt(total_norm)
        
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grads:
                if g is not None:
                    g *= clip_coef
        return total_norm

    def step(self, params: List[np.ndarray], grads: List[np.ndarray], lr: Optional[float] = None) -> None:
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
                    'm': np.zeros_like(p),
                    'v': np.zeros_like(p),
                    't': 0
                }
                
            s = self.state[p_id]
            s['t'] += 1
            t = s['t']
            
            # AdamW logic
            # 1. Weight Decay (applied to param directly, before momentum)
            p -= current_lr * self.weight_decay * p
            
            # 2. Moments
            s['m'] = b1 * s['m'] + (1 - b1) * g
            s['v'] = b2 * s['v'] + (1 - b2) * (g ** 2)
            
            # 3. Bias Correction
            m_hat = s['m'] / (1 - b1 ** t)
            v_hat = s['v'] / (1 - b2 ** t)
            
            # 4. Update
            p -= current_lr * m_hat / (np.sqrt(v_hat) + self.eps)
