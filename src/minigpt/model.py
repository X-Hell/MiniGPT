"""
MiniGPT — modern Llama-style decoder-only Transformer (~30M).

Architecture:
  - Token embedding (tied to the output projection); positions via RoPE.
  - Pre-norm TransformerBlocks:
        h = x + Attention(RMSNorm1(x))
        y = h + SwiGLU(RMSNorm2(h))
  - Full Multi-Head Attention with Rotary Positional Embeddings on Q/K.
  - SwiGLU feed-forward:  down( silu(gate(x)) * up(x) )   (3 matrices, no bias).
  - RMSNorm everywhere (scale gamma only, no beta), plus a final RMSNorm
    before the tied output projection.
  - No biases anywhere (Llama convention).

Notes vs. the prior GPT-1 (LayerNorm + GELU + learned-pos) model:
  - RMSNorm carries gamma only (1D). All 1D tensors are excluded from weight
    decay by the optimizer.
  - Position is injected by rotating Q/K inside attention (RoPE), not by an
    additive learned positional embedding. There is no W_pos.
  - Backward is hand-written against the NumPy/CuPy `xp` abstraction.

Gradient tuple layout (per TransformerBlock, returned by backward):
    (swiglu_grads, attn_grads, rms1_dgamma, rms2_dgamma)
    swiglu_grads = (dW_gate, dW_up, dW_down)
    attn_grads   = (dW_qkv, dW_o)

MiniTransformer.backward() returns:
    (dW_emb_out, layer_grads, dX_emb, d_gamma_final)
  where dW_emb_out is the *output-projection* gradient for the tied token
  matrix; the input-lookup contribution must still be scatter-added by the
  trainer at apply-time. d_gamma_final is the gradient of the final RMSNorm.
"""

from minigpt.backend import xp, scatter_add, fuse, fp16_matmul
from typing import Any, Iterator, List, Optional, Tuple
from .config import ModelConfig
from .optimized_kv_cache import OptimizedKVCache as KVCache

Array = Any


# ---------------------------------------------------------------------------
# Fused element-wise kernels (single CUDA kernel on GPU, plain on CPU)
# ---------------------------------------------------------------------------

@fuse(kernel_name='fused_exp')
def _fused_exp(x, x_max):
    return xp.exp(x - x_max)


@fuse(kernel_name='fused_silu')
def _fused_silu(x):
    """SiLU / swish activation: x * sigmoid(x). Single fused kernel on GPU."""
    return x / (1.0 + xp.exp(-x))


@fuse(kernel_name='fused_silu_backward')
def _fused_silu_backward(x, grad_out):
    """d/dx of SiLU(x) times grad_out.  silu'(x) = sig(x)*(1 + x*(1 - sig(x)))."""
    s = 1.0 / (1.0 + xp.exp(-x))
    return grad_out * (s * (1.0 + x * (1.0 - s)))


@fuse(kernel_name='fused_swiglu')
def _fused_swiglu(gate, up):
    """SwiGLU gate: silu(gate) * up, fused into one kernel."""
    return (gate / (1.0 + xp.exp(-gate))) * up


@fuse(kernel_name='fused_rmsnorm_apply')
def _fused_rmsnorm_apply(x, inv_rms, gamma):
    """RMSNorm element-wise apply: gamma * x * inv_rms (reduction done outside)."""
    return gamma * (x * inv_rms)


@fuse(kernel_name='fused_rope_combine')
def _fused_rope_combine(x, rot, cos, sin):
    """RoPE element-wise combine: x*cos + rot*sin  (rotate_half done outside)."""
    return x * cos + rot * sin


def softmax(x: Array, axis: int = -1) -> Array:
    """Numerically stable softmax with fused exp kernel."""
    x_max = xp.max(x, axis=axis, keepdims=True)
    exp_x = _fused_exp(x, x_max)
    return exp_x / xp.sum(exp_x, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# RoPE helpers (rotary positional embedding)
# ---------------------------------------------------------------------------

def build_rope_tables(max_len: int, d_head: int, theta: float) -> Tuple[Array, Array]:
    """Precompute (cos, sin) tables of shape (max_len, d_head) for RoPE.

    Uses the Llama 'rotate_half' convention: the head dim is split into two
    halves and each frequency angle is duplicated across the two halves.
    """
    half = d_head // 2
    inv_freq = 1.0 / (theta ** (xp.arange(0, half, dtype=xp.float32) * 2.0 / d_head))
    pos = xp.arange(max_len, dtype=xp.float32)
    angles = pos[:, None] * inv_freq[None, :]          # (max_len, half)
    angles = xp.concatenate([angles, angles], axis=-1)  # (max_len, d_head)
    return xp.cos(angles).astype(xp.float32), xp.sin(angles).astype(xp.float32)


def _rotate_half(x: Array) -> Array:
    """[x1, x2] -> [-x2, x1] along the last (head) dimension."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return xp.concatenate([-x2, x1], axis=-1)


def apply_rope(x: Array, cos: Array, sin: Array) -> Array:
    """Rotate x (B, H, T, d_head) by the (T, d_head) cos/sin tables."""
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    return _fused_rope_combine(x, _rotate_half(x), c, s)


def apply_rope_backward(dy: Array, cos: Array, sin: Array) -> Array:
    """Adjoint of apply_rope (rotation by -theta): dy*cos - rotate_half(dy)*sin."""
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    return dy * c - _rotate_half(dy) * s


# ---------------------------------------------------------------------------
# RMSNorm (scale gamma only — no mean-subtraction, no beta)
# ---------------------------------------------------------------------------

class RMSNorm:
    """RMSNorm: y = gamma * x / sqrt(mean(x^2) + eps)."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        self.gamma = xp.ones(d_model, dtype=xp.float32)
        self.eps = eps
        self.inv_rms = None
        self.x_hat = None

    def forward(self, x: Array) -> Array:
        """Apply RMS normalization on the last dimension."""
        ms = xp.mean(x * x, axis=-1, keepdims=True)
        self.inv_rms = 1.0 / xp.sqrt(ms + self.eps)
        self.x_hat = x * self.inv_rms
        return _fused_rmsnorm_apply(x, self.inv_rms, self.gamma)

    def backward(self, dout: Array) -> Tuple[Array, Array]:
        """Returns (dX, d_gamma)."""
        orig_shape = dout.shape
        D = orig_shape[-1]
        dout_flat = dout.reshape(-1, D)
        x_hat_flat = self.x_hat.reshape(-1, D)
        inv_rms_flat = self.inv_rms.reshape(-1, 1)

        # Parameter gradient
        d_gamma = xp.sum(dout_flat * x_hat_flat, axis=0)

        # Input gradient (RMSNorm Jacobian-vector product):
        #   dx = inv_rms * (dx_hat - x_hat * mean(dx_hat * x_hat))
        dx_hat = dout_flat * self.gamma
        mean_term = xp.mean(dx_hat * x_hat_flat, axis=-1, keepdims=True)
        dx = inv_rms_flat * (dx_hat - x_hat_flat * mean_term)

        return dx.reshape(orig_shape), d_gamma

    def apply_grads(self, d_gamma: Array, lr: float = 1e-3,
                    optimizer: Optional[Any] = None) -> None:
        if optimizer:
            optimizer.step([self.gamma], [d_gamma], lr=lr)
        else:
            self.gamma -= lr * d_gamma


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward:  down( silu(gate(x)) * up(x) )   — no biases
# ---------------------------------------------------------------------------

class FeedForward:
    def __init__(self, config: ModelConfig):
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.p_dropout = config.dropout

        self.W_gate = xp.random.normal(scale=0.02, size=(self.d_model, self.d_ff)).astype(xp.float32)
        self.W_up = xp.random.normal(scale=0.02, size=(self.d_model, self.d_ff)).astype(xp.float32)
        # Output projection scaled by 1/sqrt(2*n_layers) (residual-stream init).
        out_scale = 0.02 / float(xp.sqrt(xp.float32(2 * max(1, config.n_layers))))
        self.W_down = xp.random.normal(scale=out_scale, size=(self.d_ff, self.d_model)).astype(xp.float32)

        self.dropout_mask_hidden = None

    def forward(self, x: Array, training: bool = False) -> Array:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.d_model)

        g = fp16_matmul(x_flat, self.W_gate)        # (N, d_ff)
        u = fp16_matmul(x_flat, self.W_up)          # (N, d_ff)
        h = _fused_swiglu(g, u)                      # silu(g) * u

        if training and self.p_dropout > 0:
            mask = (xp.random.rand(*h.shape) > self.p_dropout).astype(xp.float32) / (1.0 - self.p_dropout)
            h = h * mask
            self.dropout_mask_hidden = mask
        else:
            self.dropout_mask_hidden = None

        out = fp16_matmul(h, self.W_down)
        return out.reshape(orig_shape)

    def backward(self, dout: Array, x_in: Array) -> Tuple[Array, Tuple[Array, Array, Array]]:
        """Returns (dX, (dW_gate, dW_up, dW_down))."""
        orig_shape = x_in.shape
        x_flat = x_in.reshape(-1, self.d_model)
        dout_flat = dout.reshape(-1, self.d_model)

        # Recompute forward (FP32 for stability in backward)
        g = xp.matmul(x_flat, self.W_gate)
        u = xp.matmul(x_flat, self.W_up)
        silu_g = _fused_silu(g)
        h = silu_g * u
        if self.dropout_mask_hidden is not None:
            h_drop = h * self.dropout_mask_hidden
        else:
            h_drop = h

        # ---- Backward ----
        dW_down = xp.matmul(h_drop.T, dout_flat)
        dh = xp.matmul(dout_flat, self.W_down.T)

        if self.dropout_mask_hidden is not None:
            dh = dh * self.dropout_mask_hidden

        # h = silu(g) * u
        du = dh * silu_g
        d_silu = dh * u
        dg = _fused_silu_backward(g, d_silu)

        dW_gate = xp.matmul(x_flat.T, dg)
        dW_up = xp.matmul(x_flat.T, du)
        dx_flat = xp.matmul(dg, self.W_gate.T) + xp.matmul(du, self.W_up.T)

        return dx_flat.reshape(orig_shape), (dW_gate, dW_up, dW_down)

    def apply_grads(self, grads: Tuple[Array, Array, Array], lr: float = 1e-3,
                    optimizer: Optional[Any] = None) -> None:
        dW_gate, dW_up, dW_down = grads
        params = [self.W_gate, self.W_up, self.W_down]
        grads_l = [dW_gate, dW_up, dW_down]
        if optimizer:
            optimizer.step(params, grads_l, lr=lr)
        else:
            for p, g in zip(params, grads_l):
                p -= lr * g


# ---------------------------------------------------------------------------
# Multi-Head Attention with RoPE (full MHA, no biases)
# ---------------------------------------------------------------------------

class MultiHeadAttention:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = self.d_model // self.n_heads
        self.p_dropout = config.dropout
        self.dropout_mask_softmax = None

        self.W_qkv = xp.random.normal(scale=0.02, size=(self.d_model, 3 * self.d_model)).astype(xp.float32)
        out_scale = 0.02 / float(xp.sqrt(xp.float32(2 * max(1, config.n_layers))))
        self.W_o = xp.random.normal(scale=out_scale, size=(self.d_model, self.d_model)).astype(xp.float32)

        # RoPE tables injected by MiniTransformer (shared across layers).
        self.rope_cos = None
        self.rope_sin = None
        # Slices used in the most recent forward (reused in backward).
        self._cos_slice = None
        self._sin_slice = None

    def forward(self, x: Array, kv_cache: Optional[KVCache], start_pos: int,
                layer_idx: int, training: bool = False) -> Tuple[Array, Array]:
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)

        qkv = fp16_matmul(x_flat, self.W_qkv)
        q = qkv[:, : self.d_model]
        k = qkv[:, self.d_model : 2 * self.d_model]
        v = qkv[:, 2 * self.d_model :]

        q = q.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        # RoPE on Q and K for the current positions.
        cos = self.rope_cos[start_pos : start_pos + T]
        sin = self.rope_sin[start_pos : start_pos + T]
        self._cos_slice, self._sin_slice = cos, sin
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            # Inference: append roped K and V to the cache. update() returns K
            # already in (B, H, D, T) layout, ready for the scores matmul.
            k_t, v = kv_cache.update(k, v, start_pos, layer_idx)
        else:
            # Training: transpose K so xp.matmul(q, k_t) works directly.
            k_t = k.transpose(0, 1, 3, 2)

        scores = fp16_matmul(q, k_t) / float(xp.sqrt(xp.float32(self.d_head)))

        if T > 1:
            T_k = scores.shape[-1]
            idx_q = xp.arange(T)[:, None]
            idx_k = xp.arange(T_k)[None, :]
            mask = xp.where(idx_k <= (idx_q + start_pos), xp.float32(0.0), xp.float32(-1e9))
            scores = scores + mask

        attn_weights = softmax(scores, axis=-1)

        if training and self.p_dropout > 0:
            mask = (xp.random.rand(*attn_weights.shape) > self.p_dropout).astype(xp.float32) \
                   / (1.0 - self.p_dropout)
            attn_weights = attn_weights * mask
            self.dropout_mask_softmax = mask
        else:
            self.dropout_mask_softmax = None

        attn_output = fp16_matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, T, D)
        out_flat = attn_output.reshape(-1, D)
        final_out = fp16_matmul(out_flat, self.W_o)

        return final_out.reshape(B, T, D), attn_weights

    def backward(self, dout: Array, x_in: Array) -> Tuple[Array, Tuple[Array, Array]]:
        """Returns (dX, (dW_qkv, dW_o)). Training path only (no KV cache)."""
        B, T, D = x_in.shape
        x_flat = x_in.reshape(-1, D)

        # ---- Recompute forward (FP32) ----
        qkv = xp.matmul(x_flat, self.W_qkv)
        q = qkv[:, : self.d_model]
        k = qkv[:, self.d_model : 2 * self.d_model]
        v = qkv[:, 2 * self.d_model :]

        q = q.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        cos, sin = self._cos_slice, self._sin_slice
        q_r = apply_rope(q, cos, sin)
        k_r = apply_rope(k, cos, sin)

        scores = xp.matmul(q_r, k_r.transpose(0, 1, 3, 2)) / float(xp.sqrt(xp.float32(self.d_head)))
        full_mask = xp.triu(xp.ones((T, T), dtype=xp.float32) * -1e9, k=1)
        scores = scores + full_mask

        attn_weights = softmax(scores)
        attn_output = xp.matmul(attn_weights, v).transpose(0, 2, 1, 3).reshape(B, T, D)

        # ---- Backward ----
        dout_flat = dout.reshape(-1, D)
        attn_out_flat = attn_output.reshape(-1, D)

        dW_o = xp.matmul(attn_out_flat.T, dout_flat)
        d_attn_out = xp.matmul(dout_flat, self.W_o.T).reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        attn_weights_T = attn_weights.transpose(0, 1, 3, 2)
        dV = xp.matmul(attn_weights_T, d_attn_out)
        d_weights = xp.matmul(d_attn_out, v.transpose(0, 1, 3, 2))

        if self.dropout_mask_softmax is not None:
            d_weights = d_weights * self.dropout_mask_softmax

        term2 = xp.sum(d_weights * attn_weights, axis=-1, keepdims=True)
        d_scores = attn_weights * (d_weights - term2) / float(xp.sqrt(xp.float32(self.d_head)))

        dQ_r = xp.matmul(d_scores, k_r)
        dK_r = xp.matmul(d_scores.transpose(0, 1, 3, 2), q_r)

        # Backprop through RoPE (rotation adjoint).
        dQ = apply_rope_backward(dQ_r, cos, sin)
        dK = apply_rope_backward(dK_r, cos, sin)

        dQ_flat = dQ.transpose(0, 2, 1, 3).reshape(B * T, D)
        dK_flat = dK.transpose(0, 2, 1, 3).reshape(B * T, D)
        dV_flat = dV.transpose(0, 2, 1, 3).reshape(B * T, D)

        dQKV = xp.concatenate([dQ_flat, dK_flat, dV_flat], axis=1)
        dW_qkv = xp.matmul(x_flat.T, dQKV)
        dx = xp.matmul(dQKV, self.W_qkv.T)

        return dx.reshape(B, T, D), (dW_qkv, dW_o)

    def apply_grads(self, grads: Tuple[Array, Array], lr: float = 1e-3,
                    optimizer: Optional[Any] = None) -> None:
        dW_qkv, dW_o = grads
        params = [self.W_qkv, self.W_o]
        grads_l = [dW_qkv, dW_o]
        if optimizer:
            optimizer.step(params, grads_l, lr=lr)
        else:
            for p, g in zip(params, grads_l):
                p -= lr * g


# ---------------------------------------------------------------------------
# Transformer block — Pre-Norm (Llama-style)
#    h = x + Attention(RMSNorm1(x))
#    y = h + SwiGLU(RMSNorm2(h))
# ---------------------------------------------------------------------------

class TransformerBlock:
    def __init__(self, config: ModelConfig):
        self.rms1 = RMSNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.rms2 = RMSNorm(config.d_model)
        self.ffn = FeedForward(config)
        self.p_dropout = config.dropout
        self.dropout_mask_attn = None
        self.dropout_mask_ffn = None
        # Caches for backward (normalized inputs feeding the sublayers)
        self.cache_n1 = None
        self.cache_n2 = None

    def forward(self, x: Array, kv_cache: Optional[KVCache], start_pos: int,
                layer_idx: int, training: bool = False) -> Tuple[Array, Array]:
        # -- Sub-block 1: Attention (pre-norm) --
        n1 = self.rms1.forward(x)
        self.cache_n1 = n1
        attn_out, attn_weights = self.attn.forward(n1, kv_cache, start_pos, layer_idx, training=training)

        if training and self.p_dropout > 0:
            mask = (xp.random.rand(*attn_out.shape) > self.p_dropout).astype(xp.float32) / (1.0 - self.p_dropout)
            attn_out = attn_out * mask
            self.dropout_mask_attn = mask
        else:
            self.dropout_mask_attn = None

        h = x + attn_out

        # -- Sub-block 2: SwiGLU FFN (pre-norm) --
        n2 = self.rms2.forward(h)
        self.cache_n2 = n2
        ffn_out = self.ffn.forward(n2, training=training)

        if training and self.p_dropout > 0:
            mask = (xp.random.rand(*ffn_out.shape) > self.p_dropout).astype(xp.float32) / (1.0 - self.p_dropout)
            ffn_out = ffn_out * mask
            self.dropout_mask_ffn = mask
        else:
            self.dropout_mask_ffn = None

        y = h + ffn_out
        return y, attn_weights

    def backward(self, dy: Array) -> Tuple[Array, Tuple[Any, Any, Array, Array]]:
        """Returns (dX, (swiglu_grads, attn_grads, rms1_dgamma, rms2_dgamma))."""
        # y = h + ffn_out
        dh = dy
        dffn_out = dy
        if self.dropout_mask_ffn is not None:
            dffn_out = dffn_out * self.dropout_mask_ffn

        # ffn_out = SwiGLU(n2);  n2 = RMSNorm2(h)
        dn2, swiglu_grads = self.ffn.backward(dffn_out, self.cache_n2)
        dh_from_norm, rms2_dgamma = self.rms2.backward(dn2)
        dh_total = dh + dh_from_norm

        # h = x + attn_out
        dx_res = dh_total
        dattn_out = dh_total
        if self.dropout_mask_attn is not None:
            dattn_out = dattn_out * self.dropout_mask_attn

        # attn_out = Attention(n1);  n1 = RMSNorm1(x)
        dn1, attn_grads = self.attn.backward(dattn_out, self.cache_n1)
        dx_from_norm, rms1_dgamma = self.rms1.backward(dn1)
        dx = dx_res + dx_from_norm

        return dx, (swiglu_grads, attn_grads, rms1_dgamma, rms2_dgamma)

    def apply_grads(self, grads: Tuple[Any, Any, Array, Array], lr: float,
                    optimizer: Optional[Any] = None) -> None:
        swiglu_grads, attn_grads, rms1_dgamma, rms2_dgamma = grads
        self.ffn.apply_grads(swiglu_grads, lr, optimizer)
        self.attn.apply_grads(attn_grads, lr, optimizer)
        self.rms1.apply_grads(rms1_dgamma, lr, optimizer)
        self.rms2.apply_grads(rms2_dgamma, lr, optimizer)


# ---------------------------------------------------------------------------
# Token Embedding (positions handled by RoPE — no learned positional table)
# ---------------------------------------------------------------------------

class EmbeddingLayer:
    """Token embedding. The matrix (W_emb) is TIED to the output projection."""

    def __init__(self, config: ModelConfig):
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.W_emb = xp.random.normal(scale=0.02, size=(self.vocab_size, self.d_model)).astype(xp.float32)

    def forward_seq(self, token_ids: Array, start_pos: int = 0) -> Array:
        """Lookup token embeddings for a full sequence (positions via RoPE)."""
        return self.W_emb[token_ids]                            # (B, T, D)


# ---------------------------------------------------------------------------
# Top-level Transformer
# ---------------------------------------------------------------------------

class MiniTransformer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.embeddings = EmbeddingLayer(config)

        d_head = config.d_model // config.n_heads
        self.kv_cache = KVCache(
            config.max_len, config.n_heads, d_head, config.n_layers,
            n_kv_heads=config.n_heads,
        )

        self.layers = [TransformerBlock(config) for _ in range(config.n_layers)]
        self.final_norm = RMSNorm(config.d_model)

        # Shared RoPE tables, injected into each attention sublayer.
        cos, sin = build_rope_tables(config.max_len, d_head, config.rope_theta)
        for layer in self.layers:
            layer.attn.rope_cos = cos
            layer.attn.rope_sin = sin
        self.rope_cos = cos
        self.rope_sin = sin

        self.p_dropout = config.dropout
        self.embed_dropout_mask = None
        self._training = False

        self.cache_x_final = None      # final-norm OUTPUT (feeds the tied head)
        self.cache_token_ids = None
        self.cache_start_pos = 0

    # ---- mode toggle -------------------------------------------------------
    def train(self, mode: bool = True) -> "MiniTransformer":
        self._training = mode
        return self

    def eval(self) -> "MiniTransformer":
        self._training = False
        return self

    # ---- init-time sanity report (no state change; used by pre-flight) ------
    def init_report(self) -> dict:
        """Return init-time stats for the pre-flight check (I-2).

        Reports the token-embedding std, absence of a learned positional table
        (RoPE handles position), the residual-stream target std, and the
        measured std of every output projection (attn.W_o, ffn.W_down). Called
        once at startup; the device sync from `xp.std` is negligible there.
        """
        target = 0.02 / (2.0 * max(1, self.config.n_layers)) ** 0.5
        rep = {
            "tok_emb_std": float(xp.std(self.embeddings.W_emb)),
            "has_W_pos": any("W_pos" in n for n, _ in self.named_parameters()),
            "residual_target_std": target,
            "out_proj": [],
        }
        for i, layer in enumerate(self.layers):
            rep["out_proj"].append(
                (i, float(xp.std(layer.attn.W_o)), float(xp.std(layer.ffn.W_down)))
            )
        return rep

    # ---- parameter iteration -----------------------------------------------
    def named_parameters(self) -> Iterator[Tuple[str, Array]]:
        yield "embeddings.W_emb", self.embeddings.W_emb
        for i, layer in enumerate(self.layers):
            yield f"layers.{i}.attn.W_qkv", layer.attn.W_qkv
            yield f"layers.{i}.attn.W_o", layer.attn.W_o
            yield f"layers.{i}.ffn.W_gate", layer.ffn.W_gate
            yield f"layers.{i}.ffn.W_up", layer.ffn.W_up
            yield f"layers.{i}.ffn.W_down", layer.ffn.W_down
            yield f"layers.{i}.rms1.gamma", layer.rms1.gamma
            yield f"layers.{i}.rms2.gamma", layer.rms2.gamma
        yield "final_norm.gamma", self.final_norm.gamma

    def named_parameters_with_groups(self) -> Iterator[Tuple[str, Array, str]]:
        """Yield (name, param, group). No-decay = RMSNorm gammas (all 1D)."""
        for name, p in self.named_parameters():
            is_no_decay = name.endswith(".gamma")
            yield name, p, ("no_decay" if is_no_decay else "decay")

    # ---- forward ----------------------------------------------------------
    def forward(self, token_ids: Array, start_pos: int = 0,
                training: Optional[bool] = None) -> Tuple[Array, Array]:
        if training is None:
            training = self._training

        if len(token_ids.shape) == 1:
            token_ids = token_ids[xp.newaxis, :]

        B, T = token_ids.shape
        if start_pos + T > self.config.max_len:
            raise ValueError(
                f"Sequence length overflow: start_pos({start_pos}) + T({T}) > max_len({self.config.max_len})"
            )
        x = self.embeddings.forward_seq(token_ids, start_pos=start_pos)

        if training and self.p_dropout > 0:
            mask = (xp.random.rand(*x.shape) > self.p_dropout).astype(xp.float32) / (1.0 - self.p_dropout)
            x = x * mask
            self.embed_dropout_mask = mask
        else:
            self.embed_dropout_mask = None

        self.cache_token_ids = token_ids
        self.cache_start_pos = start_pos

        cache = None if training else self.kv_cache

        all_attn_weights = []
        for i, layer in enumerate(self.layers):
            x, attn_w = layer.forward(x, cache, start_pos, layer_idx=i, training=training)
            all_attn_weights.append(attn_w)

        # Final RMSNorm, then tied-embedding output projection.
        x = self.final_norm.forward(x)
        self.cache_x_final = x
        x_flat = x.reshape(-1, self.config.d_model)
        logits = fp16_matmul(x_flat, self.embeddings.W_emb.T)
        logits = logits.reshape(B, T, self.config.vocab_size)

        return logits, all_attn_weights[-1]

    # ---- backward ---------------------------------------------------------
    def backward(self, dlogits: Array) -> Tuple[Array, List[Any], Array, Array]:
        """Returns (dW_emb_out, layer_grads, dX_emb, d_gamma_final)."""
        B, T, V = dlogits.shape
        D = self.config.d_model
        dlogits_flat = dlogits.reshape(-1, V)
        x_final_flat = self.cache_x_final.reshape(-1, D)

        # logits = x_final @ W_emb.T
        dX_final = xp.matmul(dlogits_flat, self.embeddings.W_emb)
        dW_emb_out = xp.matmul(dlogits_flat.T, x_final_flat)
        dX = dX_final.reshape(B, T, D)

        # Final RMSNorm backward
        dX, d_gamma_final = self.final_norm.backward(dX)

        # Backprop through transformer blocks (reverse order)
        layer_grads = []
        for layer in reversed(self.layers):
            dX, l_grads = layer.backward(dX)
            layer_grads.append(l_grads)
        layer_grads.reverse()

        # Embedding dropout backward
        if self.embed_dropout_mask is not None:
            dX = dX * self.embed_dropout_mask

        # dX_emb is scatter_add'd into W_emb by the trainer (tied weights).
        dX_emb = dX
        return dW_emb_out, layer_grads, dX_emb, d_gamma_final

    # ---- apply_grads (used by the simple non-accumulated trainer) ---------
    def apply_grads(self, grads: Tuple[Array, List[Any], Array, Array],
                    token_ids: Array, lr: float = 1e-3,
                    optimizer: Optional[Any] = None) -> None:
        dW_emb_out, layer_grads, dX_emb, d_gamma_final = grads

        for layer, l_grads in zip(self.layers, layer_grads):
            layer.apply_grads(l_grads, lr, optimizer)

        self.final_norm.apply_grads(d_gamma_final, lr, optimizer)

        dW_emb_total = dW_emb_out.copy()
        flat_ids = token_ids.flatten()
        flat_grads = dX_emb.reshape(-1, self.config.d_model)
        scatter_add(dW_emb_total, flat_ids, flat_grads)

        if optimizer:
            optimizer.step([self.embeddings.W_emb], [dW_emb_total], lr=lr)
        else:
            self.embeddings.W_emb -= lr * dW_emb_total
