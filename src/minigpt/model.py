from minigpt.backend import xp, scatter_add, fuse, fp16_matmul
from typing import Tuple, Optional, List, Union
from .config import ModelConfig
from .optimized_kv_cache import OptimizedKVCache as KVCache


# ---------------------------------------------------------------------------
# Fused element-wise kernels (single CUDA kernel on GPU, plain Python on CPU)
# ---------------------------------------------------------------------------

@fuse(kernel_name='fused_exp')
def _fused_exp(x, x_max):
    return xp.exp(x - x_max)


@fuse(kernel_name='fused_silu')
def _fused_silu(x):
    return x * (1.0 / (1.0 + xp.exp(-x)))


@fuse(kernel_name='fused_silu_backward')
def _fused_silu_backward(x, grad_out):
    sig = 1.0 / (1.0 + xp.exp(-xp.clip(x, -88, 88)))
    return grad_out * (sig + x * sig * (1.0 - sig))


def softmax(x, axis: int = -1):
    """Numerically stable softmax with fused exp kernel."""
    x_max = xp.max(x, axis=axis, keepdims=True)
    exp_x = _fused_exp(x, x_max)
    return exp_x / xp.sum(exp_x, axis=axis, keepdims=True)

class RMSNorm:
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.gamma = xp.ones(d_model, dtype=xp.float32)
        self.eps = eps

    def forward(self, x):
        # x: (..., D)
        ms = xp.mean(x**2, axis=-1, keepdims=True)
        self.rms = xp.sqrt(ms + self.eps)
        self.x_norm = x / self.rms
        return self.gamma * self.x_norm

    def backward(self, dout):
        """
        Returns (dX, d_gamma) -- d_gamma is now explicitly returned
        instead of stored statelessly, enabling proper gradient accumulation.
        """
        orig_shape = dout.shape
        D = orig_shape[-1]
        dout_flat = dout.reshape(-1, D)
        x_norm_flat = self.x_norm.reshape(-1, D)

        # dGamma -- returned, not stored
        d_gamma = xp.sum(dout_flat * x_norm_flat, axis=0)

        # dX
        dx_norm = dout_flat * self.gamma
        mean_compound = xp.mean(dx_norm * x_norm_flat, axis=-1, keepdims=True)
        dx = (dx_norm - x_norm_flat * mean_compound) / self.rms.reshape(-1, 1)

        return dx.reshape(orig_shape), d_gamma

    def apply_grads(self, d_gamma, lr: float = 1e-3, optimizer=None):
        """Apply gradients with explicit d_gamma (no longer stateful)."""
        if optimizer:
            optimizer.step([self.gamma], [d_gamma], lr=lr)
        else:
            self.gamma -= lr * d_gamma

class FeedForward:
    """SwiGLU Feed-Forward Network."""
    def __init__(self, config: ModelConfig):
        self.d_model = config.d_model
        self.d_ff = config.d_ff

        scale = 1.0 / xp.sqrt(xp.float32(self.d_model))
        self.W_gate = xp.random.normal(scale=float(scale), size=(self.d_model, self.d_ff)).astype(xp.float32)
        self.W_up = xp.random.normal(scale=float(scale), size=(self.d_model, self.d_ff)).astype(xp.float32)

        scale_down = 1.0 / xp.sqrt(xp.float32(self.d_ff))
        self.W_down = xp.random.normal(scale=float(scale_down), size=(self.d_ff, self.d_model)).astype(xp.float32)

    @staticmethod
    def silu(x):
        return _fused_silu(x)

    @staticmethod
    def silu_backward(x, grad_out):
        return _fused_silu_backward(x, grad_out)

    def forward(self, x):
        # x: (..., D)
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])

        gate_pre = fp16_matmul(x_flat, self.W_gate)
        gate_out = self.silu(gate_pre)
        up_out = fp16_matmul(x_flat, self.W_up)

        hidden = gate_out * up_out
        out = fp16_matmul(hidden, self.W_down)
        return out.reshape(orig_shape)

    def backward(self, dout, x_in):
        orig_shape = x_in.shape
        x_flat = x_in.reshape(-1, orig_shape[-1])
        dout_flat = dout.reshape(-1, orig_shape[-1])

        # Recompute Forward
        gate_pre = xp.matmul(x_flat, self.W_gate)
        gate_out = self.silu(gate_pre)
        up_out = xp.matmul(x_flat, self.W_up)
        hidden = gate_out * up_out

        # Backward Pass
        dW_down = xp.matmul(hidden.T, dout_flat)
        d_hidden = xp.matmul(dout_flat, self.W_down.T)

        d_gate_out = d_hidden * up_out
        d_up_out = d_hidden * gate_out

        d_gate_pre = self.silu_backward(gate_pre, d_gate_out)

        dW_gate = xp.matmul(x_flat.T, d_gate_pre)
        dW_up = xp.matmul(x_flat.T, d_up_out)

        dx_flat = xp.matmul(d_gate_pre, self.W_gate.T) + xp.matmul(d_up_out, self.W_up.T)

        return dx_flat.reshape(orig_shape), (dW_gate, dW_up, dW_down)

    def apply_grads(self, grads, lr=1e-3, optimizer=None):
        dW_gate, dW_up, dW_down = grads
        if optimizer:
            optimizer.step([self.W_gate, self.W_up, self.W_down], [dW_gate, dW_up, dW_down], lr=lr)
        else:
            self.W_gate -= lr * dW_gate
            self.W_up -= lr * dW_up
            self.W_down -= lr * dW_down

def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0):
    freqs = 1.0 / (theta ** (xp.arange(0, dim, 2)[: (dim // 2)].astype(xp.float32) / dim))
    t = xp.arange(end, dtype=xp.float32)
    freqs = xp.outer(t, freqs)
    freqs_cis = xp.exp(1j * freqs)
    return freqs_cis

def apply_rope(xq, xk, freqs_cis):
    xq = xp.ascontiguousarray(xq).astype(xp.float32)
    xk = xp.ascontiguousarray(xk).astype(xp.float32)

    xq_ = xq.view(xp.complex64)
    xk_ = xk.view(xp.complex64)

    freqs_cis = freqs_cis[None, None, :, :]

    xq_out = xq_ * freqs_cis
    xk_out = xk_ * freqs_cis

    return xq_out.view(xp.float32).reshape(*xq.shape), xk_out.view(xp.float32).reshape(*xk.shape)

def apply_rope_backward(dxq, dxk, freqs_cis):
     freqs_cis_conj = xp.conj(freqs_cis)
     return apply_rope(dxq, dxk, freqs_cis_conj)

class MultiHeadAttention:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads

        self.d_head = self.d_model // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.dim_q = self.n_heads * self.d_head
        self.dim_kv = self.n_kv_heads * self.d_head
        self.total_head_dim = self.dim_q + 2 * self.dim_kv

        # FIX: Use 1/sqrt(d_model) for input projections instead of 1/sqrt(d_head)
        scale_in = 1.0 / float(xp.sqrt(xp.float32(self.d_model)))
        self.W_qkv = xp.random.normal(scale=scale_in, size=(self.d_model, self.total_head_dim)).astype(xp.float32)

        # FIX: Scale output projection by 1/sqrt(d_model * n_layers) to prevent residual explosion
        scale_out = 1.0 / float(xp.sqrt(xp.float32(self.d_model * config.n_layers)))
        self.W_o = xp.random.normal(scale=scale_out, size=(self.n_heads * self.d_head, self.d_model)).astype(xp.float32)

        self.freqs_cis = precompute_freqs_cis(self.d_head, config.max_len, config.rope_theta)

        # FIX: Removed non-standard head scaling [1.0, 0.9, 1.1, 1.2].
        # All heads now use uniform scaling (1.0).
        self.head_scale = 1.0

    def forward(self, x, kv_cache: KVCache, start_pos: int, layer_idx: int):
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)

        qkv = fp16_matmul(x_flat, self.W_qkv)

        q = qkv[:, :self.dim_q]
        k = qkv[:, self.dim_q : self.dim_q + self.dim_kv]
        v = qkv[:, self.dim_q + self.dim_kv :]

        q = q.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_kv_heads, self.d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_heads, self.d_head).transpose(0, 2, 1, 3)

        freqs_cis = self.freqs_cis[start_pos : start_pos + T]
        q, k = apply_rope(q, k, freqs_cis)

        q = q * self.head_scale

        if kv_cache is not None:
            # Inference mode: use KV cache for autoregressive generation
            k, v = kv_cache.update(k, v, start_pos, layer_idx)
        else:
            # Training mode: bypass KV cache entirely for ~10x speedup.
            # The per-token Python loop in KVCache.update() is the #1 bottleneck.
            # During training (start_pos=0, full sequence), cache is unnecessary.
            # Transpose K to match cache output layout: (B, H, T, D) -> (B, H, D, T)
            k = k.transpose(0, 1, 3, 2)

        if self.n_rep > 1:
            k = xp.repeat(k, self.n_rep, axis=1)
            v = xp.repeat(v, self.n_rep, axis=1)

        scores = fp16_matmul(q, k) / xp.sqrt(xp.float32(self.d_head))

        if T > 1:
            T_q = T
            T_k = scores.shape[-1]
            mask = xp.full((T_q, T_k), -1e9, dtype=xp.float32)
            idx_q = xp.arange(T_q)[:, None]
            idx_k = xp.arange(T_k)[None, :]
            mask = xp.where(idx_k <= (idx_q + start_pos), xp.float32(0.0), mask)
            scores = scores + mask

        attn_weights = softmax(scores, axis=-1)
        attn_output = fp16_matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, T, self.dim_q)

        out_flat = attn_output.reshape(-1, self.dim_q)
        final_out = fp16_matmul(out_flat, self.W_o)

        return final_out.reshape(B, T, self.d_model), attn_weights

    def backward(self, dout, x_in, mask=None):
        B, T, D = x_in.shape
        x_flat = x_in.reshape(-1, D)

        # --- Recompute Forward ---
        qkv = xp.matmul(x_flat, self.W_qkv)

        q = qkv[:, :self.dim_q]
        k = qkv[:, self.dim_q : self.dim_q + self.dim_kv]
        v = qkv[:, self.dim_q + self.dim_kv :]

        q = q.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_kv_heads, self.d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_heads, self.d_head).transpose(0, 2, 1, 3)

        freqs_cis = self.freqs_cis[0:T]

        q_rot, k_rot = apply_rope(q, k, freqs_cis)

        q_scaled = q_rot * self.head_scale

        if self.n_rep > 1:
            k_rep = xp.repeat(k_rot, self.n_rep, axis=1)
            v_rep = xp.repeat(v, self.n_rep, axis=1)
        else:
            k_rep = k_rot
            v_rep = v

        # scores: (B, H, T, T)
        scores = xp.matmul(q_scaled, k_rep.transpose(0, 1, 3, 2)) / xp.sqrt(xp.float32(self.d_head))

        if mask is None:
             # Causal mask
             full_mask = xp.triu(xp.ones((T, T), dtype=xp.float32) * -1e9, k=1)
             scores += full_mask

        attn_weights = softmax(scores) # (B, H, T, T)

        # attn_output: (B, H, T, d_head)
        attn_output = xp.matmul(attn_weights, v_rep).transpose(0, 2, 1, 3).reshape(B, T, self.dim_q)

        # --- Backward ---
        dout_flat = dout.reshape(-1, D)
        attn_out_flat = attn_output.reshape(-1, self.dim_q)

        dW_o = xp.matmul(attn_out_flat.T, dout_flat)
        d_attn_out = xp.matmul(dout_flat, self.W_o.T).reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        attn_weights_T = attn_weights.transpose(0, 1, 3, 2)
        dV_rep = xp.matmul(attn_weights_T, d_attn_out)

        d_weights = xp.matmul(d_attn_out, v_rep.transpose(0, 1, 3, 2))

        term1 = d_weights
        term2 = xp.sum(d_weights * attn_weights, axis=-1, keepdims=True)
        d_scores = attn_weights * (term1 - term2) / xp.sqrt(xp.float32(self.d_head))

        # dQ_scaled: (B, H, T, D)
        dQ_scaled = xp.matmul(d_scores, k_rep)
        # dK_rep: (B, H, T, D)
        dK_rep = xp.matmul(d_scores.transpose(0, 1, 3, 2), q_scaled)

        if self.n_rep > 1:
            dK_rot = dK_rep.reshape(B, self.n_kv_heads, self.n_rep, T, self.d_head).sum(axis=2)
            dV = dV_rep.reshape(B, self.n_kv_heads, self.n_rep, T, self.d_head).sum(axis=2)
        else:
            dK_rot = dK_rep
            dV = dV_rep

        dQ_rot = dQ_scaled * self.head_scale

        # RoPE Backward
        dQ, dK = apply_rope_backward(dQ_rot, dK_rot, freqs_cis)

        # Transpose/Flatten for final projection
        dQ_flat = dQ.transpose(0, 2, 1, 3).reshape(B*T, self.dim_q)
        dK_flat = dK.transpose(0, 2, 1, 3).reshape(B*T, self.dim_kv)
        dV_flat = dV.transpose(0, 2, 1, 3).reshape(B*T, self.dim_kv)

        dQKV_flat = xp.concatenate([dQ_flat, dK_flat, dV_flat], axis=1)

        dW_qkv = xp.matmul(x_flat.T, dQKV_flat)
        dx = xp.matmul(dQKV_flat, self.W_qkv.T)

        return dx.reshape(B, T, D), (dW_qkv, dW_o)

    def apply_grads(self, grads, lr=1e-3, optimizer=None):
        dW_qkv, dW_o = grads
        if optimizer:
            optimizer.step([self.W_qkv, self.W_o], [dW_qkv, dW_o], lr=lr)
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
            mask = (xp.random.rand(*attn_out.shape) > self.p_dropout).astype(xp.float32) / (1.0 - self.p_dropout)
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
            mask = (xp.random.rand(*ffn_out.shape) > self.p_dropout).astype(xp.float32) / (1.0 - self.p_dropout)
            self.dropout_mask_ffn = mask
            ffn_out = ffn_out * mask
        else:
            self.dropout_mask_ffn = None

        x3 = resid2 + ffn_out
        return x3, attn_weights

    def backward(self, dout):
        """
        Returns (dX, (ffn_grads, attn_grads, ln1_d_gamma, ln2_d_gamma))

        FIX: LN gradients (d_gamma) are now explicitly returned in the tuple
        instead of stored statelessly on the RMSNorm objects.
        """
        dX3 = dout
        dX2_resid = dX3

        dFFN_out = dX3
        if self.dropout_mask_ffn is not None:
            dFFN_out = dFFN_out * self.dropout_mask_ffn

        dX_norm_2, ffn_grads = self.ffn.backward(dFFN_out, self.ln2.gamma * self.ln2.x_norm)
        dX2_branch, ln2_d_gamma = self.ln2.backward(dX_norm_2)

        dX2 = dX2_resid + dX2_branch

        dX_resid = dX2

        dAttn_out = dX2
        if self.dropout_mask_attn is not None:
             dAttn_out = dAttn_out * self.dropout_mask_attn

        dX_norm_1, attn_grads = self.attn.backward(dAttn_out, self.ln1.gamma * self.ln1.x_norm)
        dX_branch, ln1_d_gamma = self.ln1.backward(dX_norm_1)

        dX = dX_resid + dX_branch

        return dX, (ffn_grads, attn_grads, ln1_d_gamma, ln2_d_gamma)

    def apply_grads(self, grads, lr, optimizer=None):
        ffn_grads, attn_grads, ln1_d_gamma, ln2_d_gamma = grads
        self.ffn.apply_grads(ffn_grads, lr, optimizer)
        self.attn.apply_grads(attn_grads, lr, optimizer)
        self.ln1.apply_grads(ln1_d_gamma, lr, optimizer)
        self.ln2.apply_grads(ln2_d_gamma, lr, optimizer)

class EmbeddingLayer:
    def __init__(self, config: ModelConfig):
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.W_emb = xp.random.normal(scale=0.02, size=(self.vocab_size, self.d_model)).astype(xp.float32)

    def forward_seq(self, token_ids):
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

    def named_parameters(self):
        """Yield (name, param) pairs for all learnable parameters."""
        yield "embeddings.W_emb", self.embeddings.W_emb

        for i, layer in enumerate(self.layers):
            yield f"layers.{i}.attn.W_qkv", layer.attn.W_qkv
            yield f"layers.{i}.attn.W_o", layer.attn.W_o
            yield f"layers.{i}.ffn.W_gate", layer.ffn.W_gate
            yield f"layers.{i}.ffn.W_up", layer.ffn.W_up
            yield f"layers.{i}.ffn.W_down", layer.ffn.W_down
            yield f"layers.{i}.ln1.gamma", layer.ln1.gamma
            yield f"layers.{i}.ln2.gamma", layer.ln2.gamma

        yield "ln_f.gamma", self.ln_f.gamma

    def forward(self, token_ids, start_pos: int = 0, training: bool = False):
        if len(token_ids.shape) == 1:
            token_ids = token_ids[xp.newaxis, :]

        B, T = token_ids.shape
        x = self.embeddings.forward_seq(token_ids)
        self.cache_x_emb = x.copy()

        # FIX: Bypass KV cache during training for massive speedup.
        # The KV cache's per-token Python loop is the single biggest bottleneck
        # during training (~10x slower than direct matmul). Cache is only needed
        # for autoregressive inference where we decode one token at a time.
        cache = None if training else self.kv_cache

        all_attn_weights = []
        for i, layer in enumerate(self.layers):
            x, attn_w = layer.forward(x, cache, start_pos, layer_idx=i, training=training)
            all_attn_weights.append(attn_w)

        x_final = self.ln_f.forward(x)
        self.cache_x_final = x_final.copy()

        x_flat = x_final.reshape(-1, self.config.d_model)
        logits = fp16_matmul(x_flat, self.embeddings.W_emb.T)
        logits = logits.reshape(B, T, self.config.vocab_size)

        return logits, all_attn_weights[-1]

    def backward(self, dlogits):
        """
        Returns (dW_emb, layer_grads, dX_emb) where layer_grads now includes
        LN gradients: each element is (ffn_grads, attn_grads, ln1_d_gamma, ln2_d_gamma).
        """
        B, T, V = dlogits.shape
        dlogits_flat = dlogits.reshape(-1, V)
        x_final_flat = self.cache_x_final.reshape(-1, self.config.d_model)

        dX_final = xp.matmul(dlogits_flat, self.embeddings.W_emb)
        dW_emb = xp.matmul(dlogits_flat.T, x_final_flat)

        dX_final = dX_final.reshape(B, T, self.config.d_model)
        dX, ln_f_d_gamma = self.ln_f.backward(dX_final)

        layer_grads = []
        for layer in reversed(self.layers):
            dX, l_grads = layer.backward(dX)
            layer_grads.append(l_grads)
        layer_grads.reverse()

        dX_emb = dX
        return dW_emb, layer_grads, dX_emb, ln_f_d_gamma

    def apply_grads(self, grads, token_ids, lr=1e-3, optimizer=None):
        dW_emb_out, layer_grads, dX_emb, ln_f_d_gamma = grads

        for layer, l_grads in zip(self.layers, layer_grads):
            layer.apply_grads(l_grads, lr, optimizer)

        self.ln_f.apply_grads(ln_f_d_gamma, lr, optimizer)

        dW_total = dW_emb_out.copy()
        flat_ids = token_ids.flatten()
        flat_grads = dX_emb.reshape(-1, self.config.d_model)
        scatter_add(dW_total, flat_ids, flat_grads)

        if optimizer:
            optimizer.step([self.embeddings.W_emb], [dW_total], lr=lr)
        else:
            self.embeddings.W_emb -= lr * dW_total
