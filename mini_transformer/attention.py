import numpy as np
from .matmul import explicit_matmul

def softmax(x, axis=-1):
    # Numerically stable softmax
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class MultiHeadAttention:
    def __init__(self, d_model, n_heads, max_len=2048):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # W_qkv: (d_model, 3 * d_model)
        # W_o: (d_model, d_model)
        scale = 1.0 / np.sqrt(d_model)
        self.W_qkv = np.random.normal(scale=scale, size=(d_model, 3 * d_model)).astype(np.float32)
        self.W_o = np.random.normal(scale=scale, size=(d_model, d_model)).astype(np.float32)
        
        # RoPE
        from .rope import precompute_freqs_cis
        self.freqs_cis = precompute_freqs_cis(self.d_head, max_len)
        
        # Head Scaling (Fixed Diversity)
        # Pattern: [1.0, 0.9, 1.1, 1.2, 0.8, 1.0, ...]
        base_scales = [1.0, 0.9, 1.1, 1.2]
        scales = []
        for h in range(n_heads):
            scales.append(base_scales[h % len(base_scales)])
        self.head_scale = np.array(scales, dtype=np.float32).reshape(1, n_heads, 1, 1)
        
        total_mem = (self.W_qkv.nbytes + self.W_o.nbytes + self.freqs_cis.nbytes)
        print(f"[Attention] Weights Mem (Fused+RoPE): {total_mem/1024:.2f} KB")
        
        self.quantized = False

    def quantize(self):
        """Quantizes weights to INT8."""
        if hasattr(self, 'quantized') and self.quantized:
            return
            
        print(f"[Attention] Quantizing to INT8 (Per-Channel)...")
        from .quant_utils import quantize_matrix
        
        self.W_qkv_int8, self.w_qkv_scale = quantize_matrix(self.W_qkv)
        self.W_o_int8, self.w_o_scale = quantize_matrix(self.W_o)
        
        # Release float weights
        del self.W_qkv, self.W_o
        self.quantized = True
        
        mem = (self.W_qkv_int8.nbytes + self.w_qkv_scale.nbytes + 
               self.W_o_int8.nbytes + self.w_o_scale.nbytes)
        print(f"[Attention] Quantized Mem: {mem/1024:.2f} KB")

    def forward(self, x, kv_cache, start_pos, layer_idx):
        # x: (B, T, D)
        B, T, D = x.shape
        x_flat = x[0] # Assume B=1
        
        # 1. QKV Projection
        if hasattr(self, 'quantized') and self.quantized:
            W_qkv = self.W_qkv_int8.astype(np.float32) * self.w_qkv_scale
            qkv = explicit_matmul(x_flat, W_qkv, "Attn_QKV_Proj (INT8)")
        else:
            qkv = explicit_matmul(x_flat, self.W_qkv, "Attn_QKV_Proj")
            
        # Split Q, K, V
        qkv = qkv.reshape(T, 3, self.n_heads, self.d_head)
        qkv = qkv.transpose(1, 2, 0, 3)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        Q = Q[np.newaxis, ...] 
        K = K[np.newaxis, ...]
        V = V[np.newaxis, ...]
        
        # 2. RoPE
        from .rope import apply_rope
        freqs_cis = self.freqs_cis[start_pos : start_pos + T]
        Q, K = apply_rope(Q, K, freqs_cis)
        
        # 2b. Head Scaling
        Q = Q * self.head_scale
        
        # 3. KV Cache
        K, V = kv_cache.update(K, V, start_pos, layer_idx)
        
        # 4. Attention mechanism
        K_T = K.transpose(0, 1, 3, 2)
        scores = np.matmul(Q, K_T) / np.sqrt(self.d_head)
        
        if T > 1:
            # Masking
            B, n_heads, T_q, T_k = scores.shape
            full_mask = np.triu(np.ones((T_q, T_k)) * -1e9, k=1 + (T_k - T_q))
            scores = scores + full_mask
            
        attn_weights = softmax(scores, axis=-1)
        
        # 5. Output
        attn_output = np.matmul(attn_weights, V)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
        
        out_flat = attn_output[0]
        if hasattr(self, 'quantized') and self.quantized:
             W_o = self.W_o_int8.astype(np.float32) * self.w_o_scale
             final_out = explicit_matmul(out_flat, W_o, "Attn_Out_Proj (INT8)")
        else:
             final_out = explicit_matmul(out_flat, self.W_o, "Attn_Out_Proj")
        
        return final_out[np.newaxis, ...], attn_weights
        
    def backward(self, dout, x_in, mask=None):
        B, T, D = x_in.shape
        x_flat = x_in[0]
        
        # --- Recompute Forward (Partial) ---
        qkv = np.matmul(x_flat, self.W_qkv).reshape(T, 3, self.n_heads, self.d_head).transpose(1, 2, 0, 3)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        Q, K, V = Q[np.newaxis, ...], K[np.newaxis, ...], V[np.newaxis, ...]
        
        # RoPE Recompute
        from .rope import apply_rope, apply_rope_backward
        freqs_cis = self.freqs_cis[0:T] # Assume start_pos=0 for training
        Q, K = apply_rope(Q, K, freqs_cis)
        Q = Q * self.head_scale
        
        # Attention Scores
        K_T = K.transpose(0, 1, 3, 2)
        scores = np.matmul(Q, K_T) / np.sqrt(self.d_head)
        
        if mask is None:
             full_mask = np.triu(np.ones((T, T)) * -1e9, k=1)
             scores += full_mask
        
        attn_weights = softmax(scores)
        
        attn_output = np.matmul(attn_weights, V).transpose(0, 2, 1, 3).reshape(1, T, D)
        
        # --- Backward ---
        dout_flat = dout[0]
        attn_out_flat = attn_output[0]
        
        # dW_o
        dW_o = np.matmul(attn_out_flat.T, dout_flat)
        d_attn_out = np.matmul(dout_flat, self.W_o.T).reshape(1, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        
        # dV
        attn_weights_T = attn_weights.transpose(0, 1, 3, 2)
        dV = np.matmul(attn_weights_T, d_attn_out)
        
        d_weights = np.matmul(d_attn_out, V.transpose(0, 1, 3, 2))
        
        term1 = d_weights 
        term2 = np.sum(d_weights * attn_weights, axis=-1, keepdims=True)
        d_scores = attn_weights * (term1 - term2) / np.sqrt(self.d_head)
        
        dQ = np.matmul(d_scores, K)
        dK = np.matmul(d_scores.transpose(0, 1, 3, 2), Q)
        
        # Backprop through Head Scaling
        dQ = dQ * self.head_scale
        
        # Backprop through RoPE
        dQ, dK = apply_rope_backward(dQ, dK, freqs_cis)
        
        # Gradients for Q, K, V flattened
        dQ_flat = dQ.transpose(0, 2, 1, 3).reshape(T, D)
        dK_flat = dK.transpose(0, 2, 1, 3).reshape(T, D)
        dV_flat = dV.transpose(0, 2, 1, 3).reshape(T, D)
        
        # dW_qkv
        dQ = dQ.transpose(0, 2, 1, 3).reshape(T, self.n_heads, self.d_head) 
        dK = dK.transpose(0, 2, 1, 3).reshape(T, self.n_heads, self.d_head)
        dV = dV.transpose(0, 2, 1, 3).reshape(T, self.n_heads, self.d_head)
        
        dQKV = np.stack([dQ, dK, dV], axis=1) 
        dQKV_flat = dQKV.reshape(T, 3 * self.d_model)
        
        dW_qkv = np.matmul(x_flat.T, dQKV_flat)
        
        # dx = dQKV @ W_qkv.T
        dx = np.matmul(dQKV_flat, self.W_qkv.T)
        
        return dx[np.newaxis, ...], (dW_qkv, dW_o)

    def apply_grads(self, grads, lr=1e-3, optimizer=None):
        dW_qkv, dW_o = grads
        if optimizer:
            optimizer.step([self.W_qkv, self.W_o], [dW_qkv, dW_o])
        else:
            self.W_qkv -= lr * dW_qkv
            self.W_o -= lr * dW_o
