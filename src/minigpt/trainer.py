import numpy as np
import os
import pickle
import math
import time
from typing import Optional, List, Any
from .model import MiniTransformer
from .config import TrainConfig
from .optimizer import AdamW
from .tokenizer import BPETokenizer

class Trainer:
    """
    Trainer class for MiniGPT.
    """
    def __init__(self, model: MiniTransformer, config: TrainConfig, 
                 tokenizer: Optional[BPETokenizer] = None):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        
        self.optimizer = AdamW(
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
        self.steps = 0
        
    def cross_entropy_loss(self, logits: np.ndarray, targets: np.ndarray, 
                           label_smoothing: float = 0.1) -> Any:
        # logits: (B, T, V)
        # targets: (B, T)
        
        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        targets_flat = targets.reshape(-1)
        
        # Softmax stable
        max_logits = np.max(logits_flat, axis=1, keepdims=True)
        exp_logits = np.exp(logits_flat - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        N = logits_flat.shape[0]
        
        if label_smoothing > 0:
            smooth_targets = np.full((N, V), label_smoothing / (V - 1), dtype=np.float32)
            smooth_targets[np.arange(N), targets_flat] = 1.0 - label_smoothing
            
            log_probs_all = np.log(probs + 1e-9)
            loss = -np.sum(smooth_targets * log_probs_all) / N
            dlogits = (probs - smooth_targets) / N
        else:
            relevant_probs = probs[np.arange(N), targets_flat]
            log_probs = -np.log(relevant_probs + 1e-9)
            loss = np.mean(log_probs)
            
            dlogits = probs.copy()
            dlogits[np.arange(N), targets_flat] -= 1
            dlogits /= N
            
        return loss, dlogits.reshape(B, T, V)

    def get_lr(self, step: int) -> float:
        # Cosine Schedule with Warmup
        max_lr = self.config.learning_rate
        min_lr = max_lr * 0.1
        warmup_iters = int(self.config.max_steps * 0.05)
        
        if step < warmup_iters:
            return max_lr * step / warmup_iters
        if step > self.config.max_steps:
            return min_lr
            
        decay_ratio = (step - warmup_iters) / (self.config.max_steps - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)
        
    def clip_grads(self, grads):
        dW_emb, layer_grads, _ = grads
        sq_sum = np.sum(dW_emb**2)
        
        for l_grads in layer_grads:
            ffn_grads, attn_grads = l_grads
            for g in ffn_grads:
                sq_sum += np.sum(g**2)
            dW_qkv, dW_o = attn_grads
            sq_sum += np.sum(dW_qkv**2) + np.sum(dW_o**2)
            
        grad_norm = np.sqrt(sq_sum)
        clip_scale = 1.0
        if grad_norm > self.config.grad_clip:
            clip_scale = self.config.grad_clip / grad_norm
            
        def scale_grads(g, scale):
            if isinstance(g, tuple):
                return tuple(scale_grads(x, scale) for x in g)
            elif isinstance(g, list):
                return [scale_grads(x, scale) for x in g]
            elif isinstance(g, np.ndarray):
                return g * scale
            return g
            
        if clip_scale < 1.0:
            grads = scale_grads(grads, clip_scale)
            
        return grads, grad_norm

    def train(self, train_data: np.ndarray):
        print(f"Starting training for {self.config.max_steps} steps...")
        self.model.train = True # Logical state if needed
        
        tokens = train_data
        B = self.config.batch_size
        T_schedule_start = 16
        T_schedule_end = self.config.seq_len
        
        start_time = time.time()
        
        for step in range(self.config.max_steps):
            self.optimizer.lr = self.get_lr(step)
            
            # Curriculum Learning for T
            train_frac = step / self.config.max_steps
            current_T = int(T_schedule_start + (T_schedule_end - T_schedule_start) * train_frac)
            current_T = min(current_T, T_schedule_end)
            
            # Sample Batch (Simple Random Sampling)
            ix = np.random.randint(0, len(tokens) - current_T - 1, size=(B,))
            x = np.stack([tokens[i:i+current_T] for i in ix])
            y = np.stack([tokens[i+1:i+current_T+1] for i in ix])
            
            # Forward
            logits, _ = self.model.forward(x, start_pos=0, training=True)
            loss, dlogits = self.cross_entropy_loss(logits, y)
            
            # Backward
            grads = self.model.backward(dlogits)
            
            # Clip
            grads, grad_norm = self.clip_grads(grads)
            
            # Update
            self.model.apply_grads(grads, x, lr=self.optimizer.lr, optimizer=self.optimizer)
            
            self.steps += 1
            
            if step % self.config.log_interval == 0:
                dt = time.time() - start_time
                print(f"Step {step:4d} | Loss: {loss:.4f} | LR: {self.optimizer.lr:.5f} | "
                      f"Norm: {grad_norm:.2f} | T: {current_T} | Time: {dt:.1f}s")
                
            if step % self.config.eval_interval == 0 and step > 0:
                self.save_checkpoint(f"checkpoint_{step}.pkl")

    def save_checkpoint(self, filename: str):
        path = os.path.join(self.config.save_dir, filename)
        os.makedirs(self.config.save_dir, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Saved checkpoint: {path}")
