import numpy as np
from .matmul import explicit_matmul

def softmax(x, axis=-1):
    # Numerically stable softmax
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class MultiHeadAttention:
    def __init__(self, d_model, n_heads, n_kv_heads=None, max_len=2048):
        self.d_model = d_model
        self.n_heads = n_heads
        
        # GQA: KV heads can be fewer than Q heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        assert n_heads % self.n_kv_heads == 0, f"n_heads ({n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        
        self.d_head = d_model // n_heads
        self.n_rep = n_heads // self.n_kv_heads # Repetition factor for GQA
        
        scale = 1.0 / np.sqrt(self.d_head)
        
        # Fusion QKV
        # Standard MHA: 3 * D_model
        # GQA: H*Dh + 2*Hkv*Dh
        self.dim_q = n_heads * self.d_head
        self.dim_kv = self.n_kv_heads * self.d_head
        self.total_head_dim = self.dim_q + 2 * self.dim_kv
        
        self.W_qkv = np.random.normal(scale=scale, size=(d_model, self.total_head_dim)).astype(np.float32)
        
        self.W_o = np.random.normal(scale=scale, size=(n_heads * self.d_head, d_model)).astype(np.float32)
        
        # RoPE
        from .rope import precompute_freqs_cis
        self.freqs_cis = precompute_freqs_cis(self.d_head, max_len)
        
        # Head Scaling
        base_scales = [1.0, 0.9, 1.1, 1.2]
        scales = []
        for h in range(n_heads):
            scales.append(base_scales[h % len(base_scales)])
        self.head_scale = np.array(scales, dtype=np.float32).reshape(1, n_heads, 1, 1)
        
        total_mem = (self.W_qkv.nbytes + self.W_o.nbytes + self.freqs_cis.nbytes)
        print(f"[Attention] Weights Mem (Fused GQA - {n_heads}Q/{self.n_kv_heads}KV): {total_mem/1024:.2f} KB")
        
        self.quantized = False

    def quantize(self):
        """Quantizes weights to INT8."""
        if hasattr(self, 'quantized') and self.quantized:
            return
            
        print(f"[Attention] Quantizing to INT8 (Per-Channel)...")
        from .quant_utils import quantize_matrix
        
        self.W_qkv_int8, self.w_qkv_scale = quantize_matrix(self.W_qkv)
        self.W_o_int8, self.w_o_scale = quantize_matrix(self.W_o)
        
        del self.W_qkv, self.W_o
        self.quantized = True
        
        mem = (self.W_qkv_int8.nbytes + self.w_qkv_scale.nbytes +
               self.W_o_int8.nbytes + self.w_o_scale.nbytes)
        print(f"[Attention] Quantized Mem: {mem/1024:.2f} KB")

    def forward(self, x, kv_cache, start_pos, layer_idx):
        # x: (B, T, D)
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)
        
        # 1. Fused QKV Projection
        # [Q | K | V]
        if hasattr(self, 'quantized') and self.quantized:
            W_qkv = self.W_qkv_int8.astype(np.float32) * self.w_qkv_scale
            qkv = explicit_matmul(x_flat, W_qkv, "Attn_QKV_Fused (INT8)")
        else:
            qkv = explicit_matmul(x_flat, self.W_qkv, "Attn_QKV_Fused")
            
        # Split
        # qkv: (d_q + d_k + d_v)
        q = qkv[:, :self.dim_q]
        k = qkv[:, self.dim_q : self.dim_q + self.dim_kv]
        v = qkv[:, self.dim_q + self.dim_kv :]
        
        # Reshape
        # q: (B*T, H*Dh) -> (B, T, H, Dh)
        q = q.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3) 
        k = k.reshape(B, T, self.n_kv_heads, self.d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_heads, self.d_head).transpose(0, 2, 1, 3)
        
        # 3. RoPE
        from .rope import apply_rope
        freqs_cis = self.freqs_cis[start_pos : start_pos + T]
        q, k = apply_rope(q, k, freqs_cis)
        
        # Head Scaling
        q = q * self.head_scale
        
        # 4. KV Cache
        k, v = kv_cache.update(k, v, start_pos, layer_idx)
        
        # 5. GQA - Repeat KV
        if self.n_rep > 1:
            k = np.repeat(k, self.n_rep, axis=1)
            v = np.repeat(v, self.n_rep, axis=1)
            
        # 6. Attention mechanism
        k_t = k.transpose(0, 1, 3, 2)
        scores = np.matmul(q, k_t) / np.sqrt(self.d_head)
        
        if T > 1:
            T_q = T
            T_k = scores.shape[-1]
            mask = np.full((T_q, T_k), -1e9, dtype=np.float32)
            idx_q = np.arange(T_q)[:, None]
            idx_k = np.arange(T_k)[None, :]
            legal_mask = idx_k <= (idx_q + start_pos)
            mask[legal_mask] = 0.0
            scores = scores + mask
            
        attn_weights = softmax(scores, axis=-1)
        
        # 7. Output
        attn_output = np.matmul(attn_weights, v) # (B, H, T, Dh)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, T, self.dim_q)
        
        out_flat = attn_output.reshape(-1, self.dim_q)
        
        if hasattr(self, 'quantized') and self.quantized:
             W_o = self.W_o_int8.astype(np.float32) * self.w_o_scale
             final_out = explicit_matmul(out_flat, W_o, "Attn_Out_Proj (INT8)")
        else:
             final_out = explicit_matmul(out_flat, self.W_o, "Attn_Out_Proj")
        
        return final_out.reshape(B, T, self.d_model), attn_weights

    def backward(self, dout, x_in, mask=None):
        B, T, D = x_in.shape
        x_flat = x_in.reshape(-1, D)
        
        # --- Recompute Forward (Partial) ---
        qkv = np.matmul(x_flat, self.W_qkv)
        q = qkv[:, :self.dim_q].reshape(T, self.n_heads, self.d_head).transpose(1, 0, 2)
        k = qkv[:, self.dim_q : self.dim_q + self.dim_kv].reshape(T, self.n_kv_heads, self.d_head).transpose(1, 0, 2)
        v = qkv[:, self.dim_q + self.dim_kv :].reshape(T, self.n_kv_heads, self.d_head).transpose(1, 0, 2)
        
        # Add batch dim for RoPE helper compatibility
        q_b, k_b = q[None, ...], k[None, ...]
        
        # RoPE Recompute
        from .rope import apply_rope, apply_rope_backward
        freqs_cis = self.freqs_cis[0:T]
        q_rot, k_rot = apply_rope(q_b, k_b, freqs_cis)
        q_rot, k_rot = q_rot[0], k_rot[0]
        
        # Head Scaling
        q_scaled = q_rot * self.head_scale.reshape(self.n_heads, 1, 1)
        
        # GQA Repeat
        if self.n_rep > 1:
            k_rep = np.repeat(k_rot, self.n_rep, axis=0)
            v_rep = np.repeat(v.transpose(1, 0, 2), self.n_rep, axis=1).transpose(1, 0, 2) 
            v_rep = np.repeat(v, self.n_rep, axis=0)
        else:
            k_rep = k_rot
            v_rep = v
            
        # Attention Scores
        scores = np.matmul(q_scaled, k_rep.transpose(0, 2, 1)) / np.sqrt(self.d_head)
        
        if mask is None:
             full_mask = np.triu(np.ones((T, T)) * -1e9, k=1)
             scores += full_mask
        
        attn_weights = softmax(scores)
        
        # Output recompute
        attn_output = np.matmul(attn_weights, v_rep).transpose(1, 0, 2).reshape(1, T, D)
        
        # --- Backward ---
        dout_flat = dout.reshape(-1, D)
        attn_out_flat = attn_output[0]
        
        # dW_o
        dW_o = np.matmul(attn_out_flat.T, dout_flat)
        d_attn_out = np.matmul(dout_flat, self.W_o.T).reshape(T, self.n_heads, self.d_head).transpose(1, 0, 2)
        
        # dV_rep
        attn_weights_T = attn_weights.transpose(0, 2, 1)
        dV_rep = np.matmul(attn_weights_T, d_attn_out)
        
        # dScores
        d_weights = np.matmul(d_attn_out, v_rep.transpose(0, 2, 1))
        
        # Softmax backward
        term1 = d_weights 
        term2 = np.sum(d_weights * attn_weights, axis=-1, keepdims=True)
        d_scores = attn_weights * (term1 - term2) / np.sqrt(self.d_head)
        
        # dQ_scaled
        dQ_scaled = np.matmul(d_scores, k_rep)
        
        # dK_rep
        dK_rep = np.matmul(d_scores.transpose(0, 2, 1), q_scaled)
        
        # Reverse GQA Repeat
        if self.n_rep > 1:
            dK_rot = dK_rep.reshape(self.n_kv_heads, self.n_rep, T, self.d_head).sum(axis=1)
            dV = dV_rep.reshape(self.n_kv_heads, self.n_rep, T, self.d_head).sum(axis=1)
        else:
            dK_rot = dK_rep
            dV = dV_rep
            
        # Reverse Head Scaling
        dQ_rot = dQ_scaled * self.head_scale.reshape(self.n_heads, 1, 1)
        
        # Reverse RoPE
        dQ_rot_b, dK_rot_b = dQ_rot[None, ...], dK_rot[None, ...]
        dQ, dK = apply_rope_backward(dQ_rot_b, dK_rot_b, freqs_cis)
        dQ, dK = dQ[0], dK[0]
        
        # Gradients for Q, K, V
        dQ_flat = dQ.transpose(1, 0, 2).reshape(T, self.dim_q)
        dK_flat = dK.transpose(1, 0, 2).reshape(T, self.dim_kv)
        dV_flat = dV.transpose(1, 0, 2).reshape(T, self.dim_kv)
        
        # Concat gradients: dQKV = [dQ | dK | dV]
        dQKV_flat = np.concatenate([dQ_flat, dK_flat, dV_flat], axis=1)
        
        # dW_qkv
        dW_qkv = np.matmul(x_flat.T, dQKV_flat)
        
        # dx
        dx = np.matmul(dQKV_flat, self.W_qkv.T)
        
        return dx[np.newaxis, ...], (dW_qkv, dW_o)

    def apply_grads(self, grads, lr=1e-3, optimizer=None):
        dW_qkv, dW_o = grads
        if optimizer:
            optimizer.step([self.W_qkv, self.W_o], [dW_qkv, dW_o])
        else:
            self.W_qkv -= lr * dW_qkv
            self.W_o -= lr * dW_o
