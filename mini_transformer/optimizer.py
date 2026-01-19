import numpy as np

class AdamW:
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # State: dict mapping param_id -> {'m': m, 'v': v, 't': 0}
        self.state = {}
        
    def step(self, params, grads):
        """
        params: list of np.ndarray
        grads: list of np.ndarray
        Updates params in-place.
        """
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
            # 1. Weight Decay (decoupled)
            p -= self.lr * self.weight_decay * p
            
            # 2. Moments
            s['m'] = b1 * s['m'] + (1 - b1) * g
            s['v'] = b2 * s['v'] + (1 - b2) * (g ** 2)
            
            # 3. Bias Correction
            m_hat = s['m'] / (1 - b1 ** t)
            v_hat = s['v'] / (1 - b2 ** t)
            
            # 4. Update
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
