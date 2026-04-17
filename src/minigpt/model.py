"""
MiniGPT — GPT-1 (2018) architectural replica.

Architecture:
  - Token + Learned Absolute Positional Embeddings (tied output projection)
  - 12x TransformerBlock with Post-Norm:
        x = LayerNorm(x + MultiHeadAttention(x))
        x = LayerNorm(x + FFN(x))
  - Full Multi-Head Attention (NO GQA, NO RoPE)
  - FFN: Linear -> GELU -> Dropout -> Linear
  - Standard LayerNorm (gamma + beta)
  - Residual dropout on attention and FFN branches
  - NO final LayerNorm (GPT-1 does not use one)

Notes on numerical layout vs. prior RMSNorm+RoPE+SwiGLU model:
  - LayerNorm carries BOTH gamma (scale) and beta (shift). Both are 1D.
  - Every Linear layer has a bias (b_*). All 1D tensors are excluded from
    weight decay by the optimizer.
  - Position information is handled additively at the input layer only
    (no rotary transform of Q/K inside attention).
  - Backward still assumes the custom NumPy/CuPy abstraction (xp).

Gradient tuple layout (per TransformerBlock, returned by backward):
    (ffn_grads, attn_grads, (ln1_dgamma, ln1_dbeta), (ln2_dgamma, ln2_dbeta))
    ffn_grads  = (dW_fc, db_fc, dW_proj, db_proj)
    attn_grads = (dW_qkv, db_qkv, dW_o, db_o)

MiniTransformer.backward() returns:
    (dW_emb_out, dW_pos, layer_grads, dX_emb)
  where dW_emb_out is the *output-projection* gradient for the tied token
  matrix; the input-lookup contribution must still be scatter-added by the
  trainer at apply-time.
"""

from minigpt.backend import xp, scatter_add, fuse, fp16_matmul
from typing import Tuple, Optional, List, Union
from .config import ModelConfig
from .optimized_kv_cache import OptimizedKVCache as KVCache


# ---------------------------------------------------------------------------
# Fused element-wise kernels (single CUDA kernel on GPU, plain on CPU)
# ---------------------------------------------------------------------------

# sqrt(2 / pi) for the GELU tanh approximation (GPT-1 / GPT-2 exact recipe)
_GELU_C = 0.7978845608028654


@fuse(kernel_name='fused_exp')
def _fused_exp(x, x_max):
    return xp.exp(x - x_max)


@fuse(kernel_name='fused_gelu')
def _fused_gelu(x):
    """GELU (tanh approximation). Matches GPT-1's Gaussian Error Linear Unit."""
    inner = _GELU_C * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + xp.tanh(inner))


@fuse(kernel_name='fused_gelu_backward')
def _fused_gelu_backward(x, grad_out):
    """d/dx of GELU(x) multiplied by grad_out. Single fused kernel on GPU."""
    inner = _GELU_C * (x + 0.044715 * x * x * x)
    t = xp.tanh(inner)
    # d/dx = 0.5*(1+t) + 0.5*x*(1 - t^2) * c * (1 + 3*0.044715*x^2)
    dgelu = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * _GELU_C * (1.0 + 0.134145 * x * x)
    return grad_out * dgelu


def softmax(x, axis: int = -1):
    """Numerically stable softmax with fused exp kernel."""
    x_max = xp.max(x, axis=axis, keepdims=True)
    exp_x = _fused_exp(x, x_max)
    return exp_x / xp.sum(exp_x, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# LayerNorm (with gamma AND beta — standard, not RMSNorm)
# ---------------------------------------------------------------------------

class LayerNorm:
    """Standard LayerNorm: y = gamma * (x - mu) / sqrt(var + eps) + beta."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        self.gamma = xp.ones(d_model, dtype=xp.float32)
        self.beta = xp.zeros(d_model, dtype=xp.float32)
        self.eps = eps

    def forward(self, x):
        # x: (..., D)
        mu = xp.mean(x, axis=-1, keepdims=True)
        var = xp.mean((x - mu) ** 2, axis=-1, keepdims=True)
        self.inv_std = 1.0 / xp.sqrt(var + self.eps)
        self.x_hat = (x - mu) * self.inv_std
        return self.gamma * self.x_hat + self.beta

    def backward(self, dout):
        """
        Returns (dX, d_gamma, d_beta).
        """
        orig_shape = dout.shape
        D = orig_shape[-1]
        dout_flat = dout.reshape(-1, D)
        x_hat_flat = self.x_hat.reshape(-1, D)
        inv_std_flat = self.inv_std.reshape(-1, 1)

        # Parameter grads
        d_gamma = xp.sum(dout_flat * x_hat_flat, axis=0)
        d_beta = xp.sum(dout_flat, axis=0)

        # Input grad (compact form of LayerNorm Jacobian-vector product)
        #   dx = inv_std * (dx_hat - mean(dx_hat) - x_hat * mean(dx_hat * x_hat))
        dx_hat = dout_flat * self.gamma
        mean_dx_hat = xp.mean(dx_hat, axis=-1, keepdims=True)
        mean_dx_hat_xhat = xp.mean(dx_hat * x_hat_flat, axis=-1, keepdims=True)
        dx = inv_std_flat * (dx_hat - mean_dx_hat - x_hat_flat * mean_dx_hat_xhat)

        return dx.reshape(orig_shape), d_gamma, d_beta

    def apply_grads(self, ln_grads, lr: float = 1e-3, optimizer=None):
        d_gamma, d_beta = ln_grads
        if optimizer:
            optimizer.step([self.gamma, self.beta], [d_gamma, d_beta], lr=lr)
        else:
            self.gamma -= lr * d_gamma
            self.beta -= lr * d_beta


# ---------------------------------------------------------------------------
# Feed-Forward Network (GPT-1 style): Linear -> GELU -> Dropout -> Linear
# ---------------------------------------------------------------------------

class FeedForward:
    def __init__(self, config: ModelConfig):
        self.d_model = config.d_model
        self.d_ff = config.d_ff          # 4 * d_model by default
        self.p_dropout = config.dropout

        # GPT-style init: N(0, 0.02)
        self.W_fc = xp.random.normal(scale=0.02, size=(self.d_model, self.d_ff)).astype(xp.float32)
        self.b_fc = xp.zeros(self.d_ff, dtype=xp.float32)

        # Output projection scaled by 1/sqrt(2 * n_layers) to dampen residual
        # stream growth (the GPT-2 residual-stream init trick — harmless & helpful here too).
        out_scale = 0.02 / float(xp.sqrt(xp.float32(2 * max(1, config.n_layers))))
        self.W_proj = xp.random.normal(scale=out_scale, size=(self.d_ff, self.d_model)).astype(xp.float32)
        self.b_proj = xp.zeros(self.d_model, dtype=xp.float32)

        self.dropout_mask_hidden = None

    def forward(self, x, training: bool = False):
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.d_model)

        # Linear 1 (FP16 matmul on GPU, FP32 on CPU)
        h_pre = fp16_matmul(x_flat, self.W_fc) + self.b_fc

        # GELU (fused)
        h = _fused_gelu(h_pre)

        # Dropout on MLP intermediate (GPT-1)
        if training and self.p_dropout > 0:
            mask = (xp.random.rand(*h.shape) > self.p_dropout).astype(xp.float32) / (1.0 - self.p_dropout)
            h = h * mask
            self.dropout_mask_hidden = mask
        else:
            self.dropout_mask_hidden = None

        # Linear 2
        out = fp16_matmul(h, self.W_proj) + self.b_proj
        return out.reshape(orig_shape)

    def backward(self, dout, x_in):
        """
        Returns (dX, (dW_fc, db_fc, dW_proj, db_proj)).
        x_in is the FFN *input* (the LN1 output in post-norm).
        """
        orig_shape = x_in.shape
        x_flat = x_in.reshape(-1, self.d_model)
        dout_flat = dout.reshape(-1, self.d_model)

        # Recompute forward (FP32 for stability in backward)
        h_pre = xp.matmul(x_flat, self.W_fc) + self.b_fc
        h = _fused_gelu(h_pre)
        if self.dropout_mask_hidden is not None:
            h_drop = h * self.dropout_mask_hidden
        else:
            h_drop = h

        # ---- Backward ----
        # Linear 2
        dW_proj = xp.matmul(h_drop.T, dout_flat)
        db_proj = xp.sum(dout_flat, axis=0)
        dh_drop = xp.matmul(dout_flat, self.W_proj.T)

        # Dropout backward
        if self.dropout_mask_hidden is not None:
            dh = dh_drop * self.dropout_mask_hidden
        else:
            dh = dh_drop

        # GELU backward (fused)
        dh_pre = _fused_gelu_backward(h_pre, dh)

        # Linear 1
        dW_fc = xp.matmul(x_flat.T, dh_pre)
        db_fc = xp.sum(dh_pre, axis=0)
        dx_flat = xp.matmul(dh_pre, self.W_fc.T)

        return dx_flat.reshape(orig_shape), (dW_fc, db_fc, dW_proj, db_proj)

    def apply_grads(self, grads, lr=1e-3, optimizer=None):
        dW_fc, db_fc, dW_proj, db_proj = grads
        params = [self.W_fc, self.b_fc, self.W_proj, self.b_proj]
        grads_l = [dW_fc, db_fc, dW_proj, db_proj]
        if optimizer:
            optimizer.step(params, grads_l, lr=lr)
        else:
            for p, g in zip(params, grads_l):
                p -= lr * g


# ---------------------------------------------------------------------------
# Multi-Head Attention (full, no GQA, no RoPE)
# ---------------------------------------------------------------------------

class MultiHeadAttention:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = self.d_model // self.n_heads
        self.p_dropout = config.dropout
        self.dropout_mask_softmax = None

        # GPT-style init
        self.W_qkv = xp.random.normal(scale=0.02, size=(self.d_model, 3 * self.d_model)).astype(xp.float32)
        self.b_qkv = xp.zeros(3 * self.d_model, dtype=xp.float32)

        # Output projection with residual-stream scaling
        out_scale = 0.02 / float(xp.sqrt(xp.float32(2 * max(1, config.n_layers))))
        self.W_o = xp.random.normal(scale=out_scale, size=(self.d_model, self.d_model)).astype(xp.float32)
        self.b_o = xp.zeros(self.d_model, dtype=xp.float32)

    def forward(self, x, kv_cache: Optional[KVCache], start_pos: int, layer_idx: int,
                training: bool = False):
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)

        # QKV projection (FP16 matmul on GPU)
        qkv = fp16_matmul(x_flat, self.W_qkv) + self.b_qkv

        q = qkv[:, : self.d_model]
        k = qkv[:, self.d_model : 2 * self.d_model]
        v = qkv[:, 2 * self.d_model :]

        q = q.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        if kv_cache is not None:
            # Inference (autoregressive decode): use KV cache
            k, v = kv_cache.update(k, v, start_pos, layer_idx)
        else:
            # Training: bypass cache, transpose K so xp.matmul(q, k) works directly.
            # (B, H, T, D_h) -> (B, H, D_h, T)
            k = k.transpose(0, 1, 3, 2)

        # Scaled dot-product
        scores = fp16_matmul(q, k) / float(xp.sqrt(xp.float32(self.d_head)))

        # Causal mask
        if T > 1:
            T_k = scores.shape[-1]
            idx_q = xp.arange(T)[:, None]
            idx_k = xp.arange(T_k)[None, :]
            mask = xp.where(
                idx_k <= (idx_q + start_pos),
                xp.float32(0.0),
                xp.float32(-1e9),
            )
            scores = scores + mask

        attn_weights = softmax(scores, axis=-1)

        # Dropout on attention weights (GPT-1 site 1)
        if training and self.p_dropout > 0:
            mask = (xp.random.rand(*attn_weights.shape) > self.p_dropout).astype(xp.float32) \
                   / (1.0 - self.p_dropout)
            attn_weights = attn_weights * mask
            self.dropout_mask_softmax = mask
        else:
            self.dropout_mask_softmax = None

        attn_output = fp16_matmul(attn_weights, v)

        # (B, H, T, D_h) -> (B, T, D)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, T, D)
        out_flat = attn_output.reshape(-1, D)
        final_out = fp16_matmul(out_flat, self.W_o) + self.b_o

        return final_out.reshape(B, T, D), attn_weights

    def backward(self, dout, x_in):
        """
        Returns (dX, (dW_qkv, db_qkv, dW_o, db_o)).
        x_in is the attention *input* (the raw x — NOT normalized, post-norm).
        """
        B, T, D = x_in.shape
        x_flat = x_in.reshape(-1, D)

        # ---- Recompute forward (FP32 for gradient stability) ----
        qkv = xp.matmul(x_flat, self.W_qkv) + self.b_qkv

        q = qkv[:, : self.d_model]
        k = qkv[:, self.d_model : 2 * self.d_model]
        v = qkv[:, 2 * self.d_model :]

        q = q.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        # scores: (B, H, T, T)
        scores = xp.matmul(q, k.transpose(0, 1, 3, 2)) / float(xp.sqrt(xp.float32(self.d_head)))

        # Causal mask (training always passes full T>1 sequence)
        full_mask = xp.triu(xp.ones((T, T), dtype=xp.float32) * -1e9, k=1)
        scores = scores + full_mask

        attn_weights = softmax(scores)
        attn_output = xp.matmul(attn_weights, v).transpose(0, 2, 1, 3).reshape(B, T, D)

        # ---- Backward ----
        dout_flat = dout.reshape(-1, D)
        attn_out_flat = attn_output.reshape(-1, D)

        # Output projection
        dW_o = xp.matmul(attn_out_flat.T, dout_flat)
        db_o = xp.sum(dout_flat, axis=0)
        d_attn_out = xp.matmul(dout_flat, self.W_o.T).reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        # Through attn = softmax(scores) @ V
        attn_weights_T = attn_weights.transpose(0, 1, 3, 2)
        dV = xp.matmul(attn_weights_T, d_attn_out)

        d_weights = xp.matmul(d_attn_out, v.transpose(0, 1, 3, 2))

        # Softmax dropout backward (re-apply the stored mask)
        if self.dropout_mask_softmax is not None:
            d_weights = d_weights * self.dropout_mask_softmax

        # Softmax Jacobian-vector product
        term1 = d_weights
        term2 = xp.sum(d_weights * attn_weights, axis=-1, keepdims=True)
        d_scores = attn_weights * (term1 - term2) / float(xp.sqrt(xp.float32(self.d_head)))

        dQ = xp.matmul(d_scores, k)
        dK = xp.matmul(d_scores.transpose(0, 1, 3, 2), q)

        # (B, H, T, D_h) -> (B*T, D)
        dQ_flat = dQ.transpose(0, 2, 1, 3).reshape(B * T, D)
        dK_flat = dK.transpose(0, 2, 1, 3).reshape(B * T, D)
        dV_flat = dV.transpose(0, 2, 1, 3).reshape(B * T, D)

        dQKV = xp.concatenate([dQ_flat, dK_flat, dV_flat], axis=1)

        dW_qkv = xp.matmul(x_flat.T, dQKV)
        db_qkv = xp.sum(dQKV, axis=0)
        dx = xp.matmul(dQKV, self.W_qkv.T)

        return dx.reshape(B, T, D), (dW_qkv, db_qkv, dW_o, db_o)

    def apply_grads(self, grads, lr=1e-3, optimizer=None):
        dW_qkv, db_qkv, dW_o, db_o = grads
        params = [self.W_qkv, self.b_qkv, self.W_o, self.b_o]
        grads_l = [dW_qkv, db_qkv, dW_o, db_o]
        if optimizer:
            optimizer.step(params, grads_l, lr=lr)
        else:
            for p, g in zip(params, grads_l):
                p -= lr * g


# ---------------------------------------------------------------------------
# Transformer block — Post-Norm (GPT-1)
#    x = LN1(x + Attention(x))
#    x = LN2(x + FFN(x))
# ---------------------------------------------------------------------------

class TransformerBlock:
    def __init__(self, config: ModelConfig):
        self.attn = MultiHeadAttention(config)
        self.ln1 = LayerNorm(config.d_model)
        self.ffn = FeedForward(config)
        self.ln2 = LayerNorm(config.d_model)
        self.p_dropout = config.dropout
        self.dropout_mask_attn = None
        # Caches for backward
        self.cache_x = None
        self.cache_b = None

    def forward(self, x, kv_cache, start_pos, layer_idx, training=False):
        # Cache raw x (attention's input in post-norm)
        self.cache_x = x

        # -- Sub-block 1: Attention + residual + LayerNorm --
        # Softmax dropout is handled inside MultiHeadAttention.forward (site 1).
        attn_out, attn_weights = self.attn.forward(x, kv_cache, start_pos, layer_idx,
                                                    training=training)

        # Dropout on attention output branch (site 2)
        if training and self.p_dropout > 0:
            mask = (xp.random.rand(*attn_out.shape) > self.p_dropout).astype(xp.float32) \
                   / (1.0 - self.p_dropout)
            attn_out = attn_out * mask
            self.dropout_mask_attn = mask
        else:
            self.dropout_mask_attn = None

        a = x + attn_out
        b = self.ln1.forward(a)
        self.cache_b = b  # FFN's input in post-norm

        # -- Sub-block 2: FFN + residual + LayerNorm --
        # FFN dropout (site 3, after GELU) is handled inside FeedForward.forward.
        ffn_out = self.ffn.forward(b, training=training)

        c = b + ffn_out
        y = self.ln2.forward(c)

        return y, attn_weights

    def backward(self, dy):
        """
        Returns (dX, (ffn_grads, attn_grads, (ln1_dgamma, ln1_dbeta), (ln2_dgamma, ln2_dbeta))).
        """
        # ---- LN2 backward ----
        dc, ln2_dgamma, ln2_dbeta = self.ln2.backward(dy)
        db_res = dc

        # ---- FFN backward (input was self.cache_b = LN1(a)) ----
        # FFN dropout (post-GELU) is handled inside FeedForward.backward.
        db_ffn, ffn_grads = self.ffn.backward(dc, self.cache_b)
        db = dc + db_ffn

        # ---- LN1 backward ----
        da, ln1_dgamma, ln1_dbeta = self.ln1.backward(db)
        dx_res = da
        dattn_out = da
        if self.dropout_mask_attn is not None:
            dattn_out = dattn_out * self.dropout_mask_attn

        # ---- Attention backward (input was self.cache_x = raw x) ----
        dx_attn, attn_grads = self.attn.backward(dattn_out, self.cache_x)
        dx = dx_res + dx_attn

        return dx, (ffn_grads, attn_grads, (ln1_dgamma, ln1_dbeta), (ln2_dgamma, ln2_dbeta))

    def apply_grads(self, grads, lr, optimizer=None):
        ffn_grads, attn_grads, ln1_grads, ln2_grads = grads
        self.ffn.apply_grads(ffn_grads, lr, optimizer)
        self.attn.apply_grads(attn_grads, lr, optimizer)
        self.ln1.apply_grads(ln1_grads, lr, optimizer)
        self.ln2.apply_grads(ln2_grads, lr, optimizer)


# ---------------------------------------------------------------------------
# Token + Learned Positional Embeddings (GPT-1)
# ---------------------------------------------------------------------------

class EmbeddingLayer:
    """
    Combined token embedding + learned absolute positional embedding.
    The token matrix (W_emb) is TIED to the output softmax projection
    at the MiniTransformer level.
    """

    def __init__(self, config: ModelConfig):
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.max_len = config.max_len
        self.W_emb = xp.random.normal(scale=0.02, size=(self.vocab_size, self.d_model)).astype(xp.float32)
        # Smaller init for positional embeddings (GPT-2 recipe)
        self.W_pos = xp.random.normal(scale=0.01, size=(self.max_len, self.d_model)).astype(xp.float32)

    def forward_seq(self, token_ids, start_pos: int = 0):
        B, T = token_ids.shape
        tok_emb = self.W_emb[token_ids]                         # (B, T, D)
        pos_emb = self.W_pos[start_pos : start_pos + T]         # (T, D)
        return tok_emb + pos_emb[None, :, :]                    # broadcast


# ---------------------------------------------------------------------------
# Top-level GPT-1 Transformer
# ---------------------------------------------------------------------------

class MiniTransformer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.embeddings = EmbeddingLayer(config)

        # KV cache (inference-only); GPT-1 has no GQA so n_kv_heads = n_heads.
        self.kv_cache = KVCache(
            config.max_len,
            config.n_heads,
            config.d_model // config.n_heads,
            config.n_layers,
            n_kv_heads=config.n_heads,
        )

        self.layers = [TransformerBlock(config) for _ in range(config.n_layers)]

        # NOTE: GPT-1 has NO final LayerNorm. The last block's output feeds
        # directly into the tied-embedding output projection.

        self.p_dropout = config.dropout
        self.embed_dropout_mask = None
        self._training = False

        # Cached for backward
        self.cache_x_final = None
        self.cache_token_ids = None
        self.cache_start_pos = 0

    # ---- mode toggle -------------------------------------------------------
    def train(self, mode: bool = True):
        """Enable dropout globally. Returns self for chaining."""
        self._training = mode
        return self

    def eval(self):
        """Disable dropout globally. Returns self for chaining."""
        self._training = False
        return self

    # ---- parameter iteration -----------------------------------------------
    def named_parameters(self):
        """Yield (name, param) pairs for all learnable parameters."""
        yield "embeddings.W_emb", self.embeddings.W_emb
        yield "embeddings.W_pos", self.embeddings.W_pos

        for i, layer in enumerate(self.layers):
            yield f"layers.{i}.attn.W_qkv", layer.attn.W_qkv
            yield f"layers.{i}.attn.b_qkv", layer.attn.b_qkv
            yield f"layers.{i}.attn.W_o", layer.attn.W_o
            yield f"layers.{i}.attn.b_o", layer.attn.b_o
            yield f"layers.{i}.ffn.W_fc", layer.ffn.W_fc
            yield f"layers.{i}.ffn.b_fc", layer.ffn.b_fc
            yield f"layers.{i}.ffn.W_proj", layer.ffn.W_proj
            yield f"layers.{i}.ffn.b_proj", layer.ffn.b_proj
            yield f"layers.{i}.ln1.gamma", layer.ln1.gamma
            yield f"layers.{i}.ln1.beta", layer.ln1.beta
            yield f"layers.{i}.ln2.gamma", layer.ln2.gamma
            yield f"layers.{i}.ln2.beta", layer.ln2.beta

    def named_parameters_with_groups(self):
        """
        Yield (name, param, group) tuples where group is 'decay' or 'no_decay'.

        No-decay covers: LayerNorm gamma/beta, all biases, and W_pos
        (GPT-1 convention — positional embeddings are not regularised).
        """
        no_decay_suffixes = (
            ".gamma", ".beta",
            ".b_qkv", ".b_o", ".b_fc", ".b_proj",
            "embeddings.W_pos",
        )
        for name, p in self.named_parameters():
            is_no_decay = any(name.endswith(s) or s in name for s in no_decay_suffixes)
            yield name, p, ("no_decay" if is_no_decay else "decay")

    # ---- forward ----------------------------------------------------------
    def forward(self, token_ids, start_pos: int = 0, training: Optional[bool] = None):
        if training is None:
            training = self._training

        if len(token_ids.shape) == 1:
            token_ids = token_ids[xp.newaxis, :]

        B, T = token_ids.shape
        x = self.embeddings.forward_seq(token_ids, start_pos=start_pos)

        # Embedding dropout (GPT-1 style)
        if training and self.p_dropout > 0:
            mask = (xp.random.rand(*x.shape) > self.p_dropout).astype(xp.float32) / (1.0 - self.p_dropout)
            x = x * mask
            self.embed_dropout_mask = mask
        else:
            self.embed_dropout_mask = None

        # Cache for backward
        self.cache_token_ids = token_ids
        self.cache_start_pos = start_pos

        # KV cache is only used for inference (autoregressive decode)
        cache = None if training else self.kv_cache

        all_attn_weights = []
        for i, layer in enumerate(self.layers):
            x, attn_w = layer.forward(x, cache, start_pos, layer_idx=i, training=training)
            all_attn_weights.append(attn_w)

        # Final output — tied embedding projection (NO final LayerNorm in GPT-1)
        self.cache_x_final = x
        x_flat = x.reshape(-1, self.config.d_model)
        logits = fp16_matmul(x_flat, self.embeddings.W_emb.T)
        logits = logits.reshape(B, T, self.config.vocab_size)

        return logits, all_attn_weights[-1]

    # ---- backward ---------------------------------------------------------
    def backward(self, dlogits):
        """
        Returns (dW_emb_out, dW_pos, layer_grads, dX_emb).

        - dW_emb_out: output-projection contribution to the tied embedding.
                      The trainer must scatter_add(dX_emb) into this afterward
                      to capture the input-lookup contribution.
        - dW_pos:     gradient for the learned positional embeddings.
        - layer_grads: list of per-block grads (see TransformerBlock.backward).
        - dX_emb:     gradient at the embedding layer output (used by the
                      trainer to scatter-add into dW_emb for tied weights).
        """
        B, T, V = dlogits.shape
        D = self.config.d_model
        dlogits_flat = dlogits.reshape(-1, V)
        x_final_flat = self.cache_x_final.reshape(-1, D)

        # Output-projection gradient for tied embedding:
        #   logits = x_final @ W_emb.T
        #   dW_emb_out = dlogits.T @ x_final       shape (V, D)
        #   dX_final   = dlogits @ W_emb           shape (N, D)
        dX_final = xp.matmul(dlogits_flat, self.embeddings.W_emb)
        dW_emb_out = xp.matmul(dlogits_flat.T, x_final_flat)

        dX = dX_final.reshape(B, T, D)

        # Backprop through transformer blocks (reverse order)
        layer_grads = []
        for layer in reversed(self.layers):
            dX, l_grads = layer.backward(dX)
            layer_grads.append(l_grads)
        layer_grads.reverse()

        # Embedding dropout backward
        if self.embed_dropout_mask is not None:
            dX = dX * self.embed_dropout_mask

        # Positional embedding gradient: reduce over batch, place at [start_pos:start_pos+T]
        dW_pos = xp.zeros_like(self.embeddings.W_pos)
        dX_reduce = xp.sum(dX, axis=0)  # (T, D)
        dW_pos[self.cache_start_pos : self.cache_start_pos + T] = dX_reduce

        # dX_emb is the gradient at the (tok + pos) sum — the trainer
        # scatter_add's it into dW_emb using the input token ids.
        dX_emb = dX

        return dW_emb_out, dW_pos, layer_grads, dX_emb

    # ---- apply_grads (used by the simple non-accumulated trainer) --------
    def apply_grads(self, grads, token_ids, lr=1e-3, optimizer=None):
        dW_emb_out, dW_pos, layer_grads, dX_emb = grads

        # Per-block grads
        for layer, l_grads in zip(self.layers, layer_grads):
            layer.apply_grads(l_grads, lr, optimizer)

        # Tied embedding: combine output-projection + input-lookup gradients
        dW_emb_total = dW_emb_out.copy()
        flat_ids = token_ids.flatten()
        flat_grads = dX_emb.reshape(-1, self.config.d_model)
        scatter_add(dW_emb_total, flat_ids, flat_grads)

        if optimizer:
            optimizer.step(
                [self.embeddings.W_emb, self.embeddings.W_pos],
                [dW_emb_total, dW_pos],
                lr=lr,
            )
        else:
            self.embeddings.W_emb -= lr * dW_emb_total
            self.embeddings.W_pos -= lr * dW_pos
