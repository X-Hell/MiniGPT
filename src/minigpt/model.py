import numpy as np
from typing import Tuple, Optional, List, Union
from .config import ModelConfig
from .optimized_kv_cache import OptimizedKVCache as KVCache

def explicit_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Wrapper for np.matmul."""
    return np.matmul(A, B)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class RMSNorm:
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.gamma = np.ones(d_model, dtype=np.float32)
        self.eps = eps
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (..., D)
        ms = np.mean(x**2, axis=-1, keepdims=True)
        self.rms = np.sqrt(ms + self.eps)
        self.x_norm = x / self.rms
        return self.gamma * self.x_norm
        
    def backward(self, dout: np.ndarray) -> np.ndarray:
        # dout: (..., D)
        orig_shape = dout.shape
        D = orig_shape[-1]
        dout_flat = dout.reshape(-1, D)
        x_norm_flat = self.x_norm.reshape(-1, D)
        
        # dGamma
        self.d_gamma = np.sum(dout_flat * x_norm_flat, axis=0)
        
        # dX
        dx_norm = dout_flat * self.gamma
        mean_compound = np.mean(dx_norm * x_norm_flat, axis=-1, keepdims=True)
        dx = (dx_norm - x_norm_flat * mean_compound) / self.rms.reshape(-1, 1)
        
        return dx.reshape(orig_shape)

    def apply_grads(self, lr: float = 1e-3, optimizer=None):
        if optimizer:
            optimizer.step([self.gamma], [self.d_gamma])
        else:
            self.gamma -= lr * self.d_gamma

class FeedForward:
    """SwiGLU Feed-Forward Network."""
    def __init__(self, config: ModelConfig):
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        
        scale = 1.0 / np.sqrt(self.d_model)
        self.W_gate = np.random.normal(scale=scale, size=(self.d_model, self.d_ff)).astype(np.float32)
        self.W_up = np.random.normal(scale=scale, size=(self.d_model, self.d_ff)).astype(np.float32)
        
        scale_down = 1.0 / np.sqrt(self.d_ff)
        self.W_down = np.random.normal(scale=scale_down, size=(self.d_ff, self.d_model)).astype(np.float32)
        
    @staticmethod
    def silu(x: np.ndarray) -> np.ndarray:
        # Optimized: Avoid clip overhead if possible, use robust sigmoid approx or native exp with safeguards
        # Approx: x * sigmoid(x) where sigmoid(x) = 0.5 * (1 + tanh(0.5*x))? Tanh is also expensive.
        # Just use guarded exp.
        # x * (1 / (1 + exp(-x)))
        return x * (1.0 / (1.0 + np.exp(-x))) # Removed clip for speed, assuming reasonable range from LayerNorm
    
    @staticmethod
    def silu_backward(x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
        sig = 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))
        return grad_out * (sig + x * sig * (1 - sig))
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (..., D)
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        
        gate_pre = explicit_matmul(x_flat, self.W_gate)
        gate_out = self.silu(gate_pre)
        up_out = explicit_matmul(x_flat, self.W_up)
        
        hidden = gate_out * up_out
        out = explicit_matmul(hidden, self.W_down)
        return out.reshape(orig_shape)

    def backward(self, dout: np.ndarray, x_in: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        orig_shape = x_in.shape
        x_flat = x_in.reshape(-1, orig_shape[-1])
        dout_flat = dout.reshape(-1, orig_shape[-1])
        
        # Recompute Forward
        gate_pre = np.matmul(x_flat, self.W_gate)
        gate_out = self.silu(gate_pre)
        up_out = np.matmul(x_flat, self.W_up)
        hidden = gate_out * up_out
        
        # Backend Pass
        dW_down = np.matmul(hidden.T, dout_flat)
        d_hidden = np.matmul(dout_flat, self.W_down.T)
        
        d_gate_out = d_hidden * up_out
        d_up_out = d_hidden * gate_out
        
        d_gate_pre = self.silu_backward(gate_pre, d_gate_out)
        
        dW_gate = np.matmul(x_flat.T, d_gate_pre)
        dW_up = np.matmul(x_flat.T, d_up_out)
        
        dx_flat = np.matmul(d_gate_pre, self.W_gate.T) + np.matmul(d_up_out, self.W_up.T)
        
        return dx_flat.reshape(orig_shape), (dW_gate, dW_up, dW_down)

    def apply_grads(self, grads, lr=1e-3, optimizer=None):
        dW_gate, dW_up, dW_down = grads
        if optimizer:
            optimizer.step([self.W_gate, self.W_up, self.W_down], [dW_gate, dW_up, dW_down])
        else:
            self.W_gate -= lr * dW_gate
            self.W_up -= lr * dW_up
            self.W_down -= lr * dW_down

def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0) -> np.ndarray:
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(np.float32) / dim))
    t = np.arange(end, dtype=np.float32)
    freqs = np.outer(t, freqs)
    freqs_cis = np.exp(1j * freqs)
    return freqs_cis

def apply_rope(xq: np.ndarray, xk: np.ndarray, freqs_cis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xq = np.ascontiguousarray(xq).astype(np.float32)
    xk = np.ascontiguousarray(xk).astype(np.float32)
    
    xq_ = xq.view(np.complex64)
    xk_ = xk.view(np.complex64)
    
    freqs_cis = freqs_cis[None, None, :, :]
    
    xq_out = xq_ * freqs_cis
    xk_out = xk_ * freqs_cis
    
    return xq_out.view(np.float32).reshape(*xq.shape), xk_out.view(np.float32).reshape(*xk.shape)

def apply_rope_backward(dxq, dxk, freqs_cis):
     freqs_cis_conj = np.conj(freqs_cis)
     return apply_rope(dxq, dxk, freqs_cis_conj)

class MultiHeadAttention:
    def __init__(self, config: ModelConfig):
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        
        self.d_head = self.d_model // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        
        scale = 1.0 / np.sqrt(self.d_head)
        
        self.dim_q = self.n_heads * self.d_head
        self.dim_kv = self.n_kv_heads * self.d_head
        self.total_head_dim = self.dim_q + 2 * self.dim_kv
        
        self.W_qkv = np.random.normal(scale=scale, size=(self.d_model, self.total_head_dim)).astype(np.float32)
        self.W_o = np.random.normal(scale=scale, size=(self.n_heads * self.d_head, self.d_model)).astype(np.float32)
        
        self.freqs_cis = precompute_freqs_cis(self.d_head, config.max_len, config.rope_theta)
        
        # Head Scaling
        base_scales = [1.0, 0.9, 1.1, 1.2]
        scales = [base_scales[h % len(base_scales)] for h in range(self.n_heads)]
        self.head_scale = np.array(scales, dtype=np.float32).reshape(1, self.n_heads, 1, 1)

    def forward(self, x: np.ndarray, kv_cache: KVCache, start_pos: int, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)
        
        qkv = explicit_matmul(x_flat, self.W_qkv)
        
        q = qkv[:, :self.dim_q]
        k = qkv[:, self.dim_q : self.dim_q + self.dim_kv]
        v = qkv[:, self.dim_q + self.dim_kv :]
        
        q = q.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3) 
        k = k.reshape(B, T, self.n_kv_heads, self.d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_heads, self.d_head).transpose(0, 2, 1, 3)
        
        freqs_cis = self.freqs_cis[start_pos : start_pos + T]
        q, k = apply_rope(q, k, freqs_cis)
        
        q = q * self.head_scale
        
        k, v = kv_cache.update(k, v, start_pos, layer_idx)
        
        if self.n_rep > 1:
            k = np.repeat(k, self.n_rep, axis=1)
            v = np.repeat(v, self.n_rep, axis=1)
            
        # k is now (B, H, D, T) from cache directly (already transposed during update)
        # q is (B, H, 1, D) -> want (B, H, T_q, D)
        
        # Current logic:
        # q: (B, H, T, D)
        # k: (B, H, D, T) from cache
        
        # Matmul: (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
        scores = np.matmul(q, k) / np.sqrt(self.d_head)
        
        if T > 1:
            T_q = T
            T_k = scores.shape[-1]
            mask = np.full((T_q, T_k), -1e9, dtype=np.float32)
            idx_q = np.arange(T_q)[:, None]
            idx_k = np.arange(T_k)[None, :]
            mask[idx_k <= (idx_q + start_pos)] = 0.0
            scores = scores + mask
            
        attn_weights = softmax(scores, axis=-1)
        attn_output = np.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, T, self.dim_q)
        
        out_flat = attn_output.reshape(-1, self.dim_q)
        final_out = explicit_matmul(out_flat, self.W_o)
        
        return final_out.reshape(B, T, self.d_model), attn_weights

    def backward(self, dout, x_in, mask=None):
        # ... Recompute Forward logic (Simplified for readability, full version should be copied if training is needed) ...
        # Since this is a refactor and the user wants to keep logic, I MUST copy the backward pass fully.
        
        B, T, D = x_in.shape
        x_flat = x_in.reshape(-1, D)
        
        # --- Recompute Forward (Partial) ---
        # --- Recompute Forward (Partial) ---
        qkv = np.matmul(x_flat, self.W_qkv)
        
        q = qkv[:, :self.dim_q]
        k = qkv[:, self.dim_q : self.dim_q + self.dim_kv]
        v = qkv[:, self.dim_q + self.dim_kv :]
        
        q = q.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3) 
        k = k.reshape(B, T, self.n_kv_heads, self.d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_heads, self.d_head).transpose(0, 2, 1, 3)
        
        # q, k, v are (B, n_heads/kv, T, d_head)
        
        freqs_cis = self.freqs_cis[0:T] # Assuming start_pos=0 for backward training usually
        # Note: If start_pos was not 0, this recompute is wrong. Trainer uses start_pos=0.
        
        q_rot, k_rot = apply_rope(q, k, freqs_cis)
        
        q_scaled = q_rot * self.head_scale
        
        if self.n_rep > 1:
            k_rep = np.repeat(k_rot, self.n_rep, axis=1)
            v_rep = np.repeat(v, self.n_rep, axis=1)
        else:
            k_rep = k_rot
            v_rep = v
            
        # scores: (B, H, T, T)
        scores = np.matmul(q_scaled, k_rep.transpose(0, 1, 3, 2)) / np.sqrt(self.d_head)
        
        if mask is None:
             # Causal mask
             full_mask = np.triu(np.ones((T, T)) * -1e9, k=1)
             scores += full_mask
        
        attn_weights = softmax(scores) # (B, H, T, T)
        
        # attn_output: (B, H, T, d_head)
        attn_output = np.matmul(attn_weights, v_rep).transpose(0, 2, 1, 3).reshape(B, T, self.dim_q)
        
        # --- Backward ---
        dout_flat = dout.reshape(-1, D)
        attn_out_flat = attn_output.reshape(-1, self.dim_q)
        
        dW_o = np.matmul(attn_out_flat.T, dout_flat)
        # d_attn_out: (B*T, dim_q) -> (B, T, H, D) -> (B, H, T, D)
        d_attn_out = np.matmul(dout_flat, self.W_o.T).reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        
        # attn_weights: (B, H, T, T) -> Transpose to (B, H, T, T) (swap last two)
        attn_weights_T = attn_weights.transpose(0, 1, 3, 2)
        # dV_rep: (B, H, T, D) = (B, H, T, T) @ (B, H, T, D)
        dV_rep = np.matmul(attn_weights_T, d_attn_out)
        
        # d_weights: (B, H, T, T) = (B, H, T, D) @ (B, H, T, D)^T
        d_weights = np.matmul(d_attn_out, v_rep.transpose(0, 1, 3, 2))
        
        term1 = d_weights 
        term2 = np.sum(d_weights * attn_weights, axis=-1, keepdims=True)
        d_scores = attn_weights * (term1 - term2) / np.sqrt(self.d_head)
        
        # dQ_scaled: (B, H, T, D)
        dQ_scaled = np.matmul(d_scores, k_rep)
        # dK_rep: (B, H, T, D)
        dK_rep = np.matmul(d_scores.transpose(0, 1, 3, 2), q_scaled)
        
        if self.n_rep > 1:
            # Sum over repetition group
            # (B, H, T, D) -> (B, n_kv, n_rep, T, D) -> Sum dim 2 -> (B, n_kv, T, D)
            dK_rot = dK_rep.reshape(B, self.n_kv_heads, self.n_rep, T, self.d_head).sum(axis=2)
            dV = dV_rep.reshape(B, self.n_kv_heads, self.n_rep, T, self.d_head).sum(axis=2)
        else:
            dK_rot = dK_rep
            dV = dV_rep
            
        dQ_rot = dQ_scaled * self.head_scale
        
        # RoPE Backward
        # dQ_rot, dK_rot are (B, H, T, D). apply_rope handles broadcasting of freqs_cis (1, 1, T, D)
        dQ, dK = apply_rope_backward(dQ_rot, dK_rot, freqs_cis)
        
        # Transpose/Flatten for final projection
        # (B, H/KV, T, D) -> (B, T, H/KV, D) -> (B*T, dim)
        dQ_flat = dQ.transpose(0, 2, 1, 3).reshape(B*T, self.dim_q)
        dK_flat = dK.transpose(0, 2, 1, 3).reshape(B*T, self.dim_kv)
        dV_flat = dV.transpose(0, 2, 1, 3).reshape(B*T, self.dim_kv)
        
        dQKV_flat = np.concatenate([dQ_flat, dK_flat, dV_flat], axis=1)
        
        dW_qkv = np.matmul(x_flat.T, dQKV_flat)
        dx = np.matmul(dQKV_flat, self.W_qkv.T)
        
        # Reshape dx back to (B, T, D)
        return dx.reshape(B, T, D), (dW_qkv, dW_o)

    def apply_grads(self, grads, lr=1e-3, optimizer=None):
        dW_qkv, dW_o = grads
        if optimizer:
            optimizer.step([self.W_qkv, self.W_o], [dW_qkv, dW_o])
        else:
            self.W_qkv -= lr * dW_qkv
            self.W_o -= lr * dW_o


class TransformerBlock:
    def __init__(self, config: ModelConfig):
        self.ln1 = RMSNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = RMSNorm(config.d_model)
        self.ffn = FeedForward(config)
        self.p_dropout = config.dropout
        self.dropout_mask_attn = None
        self.dropout_mask_ffn = None
        
    def forward(self, x, kv_cache, start_pos, layer_idx, training=False):
        # 1. Attn Block
        resid1 = x
        x_norm_1 = self.ln1.forward(x)
        attn_out, attn_weights = self.attn.forward(x_norm_1, kv_cache, start_pos, layer_idx)
        
        if training and self.p_dropout > 0:
            mask = (np.random.rand(*attn_out.shape) > self.p_dropout).astype(np.float32) / (1.0 - self.p_dropout)
            self.dropout_mask_attn = mask
            attn_out = attn_out * mask
        else:
            self.dropout_mask_attn = None
            
        x2 = resid1 + attn_out
        
        # 2. FFN Block
        resid2 = x2
        x_norm_2 = self.ln2.forward(x2)
        ffn_out = self.ffn.forward(x_norm_2)
        
        if training and self.p_dropout > 0:
            mask = (np.random.rand(*ffn_out.shape) > self.p_dropout).astype(np.float32) / (1.0 - self.p_dropout)
            self.dropout_mask_ffn = mask
            ffn_out = ffn_out * mask
        else:
            self.dropout_mask_ffn = None
            
        x3 = resid2 + ffn_out
        return x3, attn_weights

    def backward(self, dout):
        # Implementation of backward call, delegating to sub-blocks
        # Assuming identical logic to original implementation
        dX3 = dout
        dX2_resid = dX3
        
        dFFN_out = dX3
        if self.dropout_mask_ffn is not None:
            dFFN_out = dFFN_out * self.dropout_mask_ffn
            
        dX_norm_2, ffn_grads = self.ffn.backward(dFFN_out, self.ln2.x_norm)
        dX2_branch = self.ln2.backward(dX_norm_2)
        
        dX2 = dX2_resid + dX2_branch
        
        dX_resid = dX2
        
        dAttn_out = dX2
        if self.dropout_mask_attn is not None:
             dAttn_out = dAttn_out * self.dropout_mask_attn
             
        dX_norm_1, attn_grads = self.attn.backward(dAttn_out, self.ln1.x_norm)
        dX_branch = self.ln1.backward(dX_norm_1)
        
        dX = dX_resid + dX_branch
        
        return dX, (ffn_grads, attn_grads)

    def apply_grads(self, grads, lr, optimizer=None):
        ffn_grads, attn_grads = grads
        self.ffn.apply_grads(ffn_grads, lr, optimizer)
        self.attn.apply_grads(attn_grads, lr, optimizer)
        self.ln1.apply_grads(lr, optimizer)
        self.ln2.apply_grads(lr, optimizer)

class EmbeddingLayer:
    def __init__(self, config: ModelConfig):
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.W_emb = np.random.normal(scale=0.02, size=(self.vocab_size, self.d_model)).astype(np.float32)
        
    def forward_seq(self, token_ids: np.ndarray) -> np.ndarray:
        return self.W_emb[token_ids]

class MiniTransformer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.embeddings = EmbeddingLayer(config)
        self.kv_cache = KVCache(config.max_len, config.n_heads, config.d_model // config.n_heads, config.n_layers, n_kv_heads=config.n_kv_heads)
        
        self.layers = []
        for _ in range(config.n_layers):
            self.layers.append(TransformerBlock(config))
            
        self.ln_f = RMSNorm(config.d_model)
        self.training = False

        

    def named_parameters(self):
        """Yield (name, param) pairs for all learnable parameters."""
        yield "embeddings.W_emb", self.embeddings.W_emb
        
        for i, layer in enumerate(self.layers):
            yield f"layers.{i}.attn.W_qkv", layer.attn.W_qkv
            yield f"layers.{i}.attn.W_o", layer.attn.W_o
            yield f"layers.{i}.ffn.W_gate", layer.ffn.W_gate
            yield f"layers.{i}.ffn.W_up", layer.ffn.W_up
            yield f"layers.{i}.ffn.W_down", layer.ffn.W_down
            yield f"layers.{i}.ln1.weight", layer.ln1.gamma
            yield f"layers.{i}.ln2.weight", layer.ln2.gamma
            
        yield "ln_f.weight", self.ln_f.gamma

    def forward(self, token_ids: np.ndarray, start_pos: int = 0, training: bool = False):
        if len(token_ids.shape) == 1:
            token_ids = token_ids[np.newaxis, :]
            
        B, T = token_ids.shape
        x = self.embeddings.forward_seq(token_ids)
        self.cache_x_emb = x.copy()
        
        all_attn_weights = []
        for i, layer in enumerate(self.layers):
            x, attn_w = layer.forward(x, self.kv_cache, start_pos, layer_idx=i, training=training)
            all_attn_weights.append(attn_w)
            
        x_final = self.ln_f.forward(x)
        self.cache_x_final = x_final.copy()
        
        x_flat = x_final.reshape(-1, self.config.d_model)
        logits = explicit_matmul(x_flat, self.embeddings.W_emb.T)
        logits = logits.reshape(B, T, self.config.vocab_size)

        
        return logits, all_attn_weights[-1]

    def train(self):
        """Set training mode to True."""
        self.training = True

    def eval(self):
        """Set training mode to False."""
        self.training = False


    def backward(self, dlogits):
        B, T, V = dlogits.shape
        dlogits_flat = dlogits.reshape(-1, V)
        x_final_flat = self.cache_x_final.reshape(-1, self.config.d_model)
        
        dX_final = np.matmul(dlogits_flat, self.embeddings.W_emb)
        dW_emb = np.matmul(dlogits_flat.T, x_final_flat)
        
        dX_final = dX_final.reshape(B, T, self.config.d_model)
        dX = self.ln_f.backward(dX_final)
        
        layer_grads = []
        for layer in reversed(self.layers):
            dX, l_grads = layer.backward(dX)
            layer_grads.append(l_grads)
        layer_grads.reverse()
        
        dX_emb = dX
        return dW_emb, layer_grads, dX_emb

    def apply_grads(self, grads, token_ids, lr=1e-3, optimizer=None):
        dW_emb_out, layer_grads, dX_emb = grads
        
        for layer, l_grads in zip(self.layers, layer_grads):
            layer.apply_grads(l_grads, lr, optimizer)
            
        self.ln_f.apply_grads(lr, optimizer)
        
        dW_total = dW_emb_out.copy()
        flat_ids = token_ids.flatten()
        flat_grads = dX_emb.reshape(-1, self.config.d_model)
        np.add.at(dW_total, flat_ids, flat_grads)
        
        if optimizer:
            optimizer.step([self.embeddings.W_emb], [dW_total])
        else:
            self.embeddings.W_emb -= lr * dW_total
