import numpy as np
from .matmul import explicit_matmul

def softmax(x, axis=-1):
    # Numerically stable softmax
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # W_q, W_k, W_v, W_o
        # Shapes: (d_model, d_model)
        # Random init
        scale = 1.0 / np.sqrt(d_model)
        self.W_q = np.random.normal(scale=scale, size=(d_model, d_model)).astype(np.float32)
        self.W_k = np.random.normal(scale=scale, size=(d_model, d_model)).astype(np.float32)
        self.W_v = np.random.normal(scale=scale, size=(d_model, d_model)).astype(np.float32)
        self.W_o = np.random.normal(scale=scale, size=(d_model, d_model)).astype(np.float32)
        
        total_mem = (self.W_q.nbytes + self.W_k.nbytes + self.W_v.nbytes + self.W_o.nbytes)
        print(f"[Attention] Weights Mem: {total_mem/1024:.2f} KB")

    def forward(self, x, kv_cache=None, start_pos=0, layer_idx=None, mask=None):
        """
        x: (1, seq_len, d_model) - Input embeddings
        kv_cache: KVCache object (optional)
        start_pos: current position index
        mask: attention mask (optional)
        """
        B, T, D = x.shape
        # assert B=1 for this simple engine
        
        # 1. Projections (Batched MatMul handled as generic 2D if we flatten B*T, but here B=1)
        # x is (T, D) effectively
        x_flat = x[0] # (T, D)
        
        Q = explicit_matmul(x_flat, self.W_q, "Attn_Q_Proj") # (T, D)
        K = explicit_matmul(x_flat, self.W_k, "Attn_K_Proj") # (T, D)
        V = explicit_matmul(x_flat, self.W_v, "Attn_V_Proj") # (T, D)
        
        # 2. Split Heads -> (B, n_heads, T, d_head)
        # Reshape: (T, n_heads, d_head) -> transpose -> (n_heads, T, d_head)
        # We manually reshape
        Q = Q.reshape(T, self.n_heads, self.d_head).transpose(1, 0, 2) # (H, T, Dh)
        K = K.reshape(T, self.n_heads, self.d_head).transpose(1, 0, 2)
        V = V.reshape(T, self.n_heads, self.d_head).transpose(1, 0, 2)
        
        # Add 'B' dim back for consistency with cache: (1, H, T, Dh)
        Q = Q[np.newaxis, ...]
        K = K[np.newaxis, ...]
        V = V[np.newaxis, ...]
        
        # 3. KV Cache Interaction
        if kv_cache:
            if layer_idx is None:
                raise ValueError("layer_idx must be provided when using KV Cache")
            K, V = kv_cache.update(K, V, start_pos, layer_idx)
            # K, V are now full history: (1, H, Total_Len, Dh)
            
        # 4. Scaled Dot Product Attention
        # Q: (1, H, T_q, Dh)
        # K: (1, H, T_k, Dh) -> Transpose to (1, H, Dh, T_k)
        # Attn = Q @ K.T / sqrt(d_head)
        
        # We process head by head or batched. Numpy supports batched matmul.
        # But to be "explicit" and visualize, let's do batched but log it.
        
        # K_T shape: (1, H, Dh, T_k)
        K_T = K.transpose(0, 1, 3, 2)
        
        scores = np.matmul(Q, K_T) # (1, H, T_q, T_k)
        scores = scores / np.sqrt(self.d_head)
        
        # Masking (Causal)
        # If we are generating token by token (T_q=1), we typically attend to all past (T_k). No masking needed if T_k is valid history.
        # If we are processing prefix (T_q > 1), we need causal mask.
        T_q = Q.shape[2]
        T_k = K.shape[2]
        
        if mask is not None:
            # mask assumed to be broadcastable
            scores = scores + mask
        elif T_q > 1:
            # Auto causal mask
            # We need to be careful with start_pos offset
            # Create a mask of shape (T_q, T_k)
            # The query at index i (relative to start_pos) can attend to keys up to start_pos + i
            full_mask = np.triu(np.ones((T_q, T_k)) * -1e9, k=1 + (T_k - T_q)) # Simplified causal
            scores = scores + full_mask
            
        attn_weights = softmax(scores, axis=-1)
        
        # 5. Aggregate V
        # Weights: (1, H, T_q, T_k)
        # V: (1, H, T_k, Dh)
        # Out: (1, H, T_q, Dh)
        attn_output = np.matmul(attn_weights, V)
        
        # 6. Merge Heads
        # (1, H, T_q, Dh) -> (1, T_q, H, Dh) -> (1, T_q, D)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, T_q, self.d_model)
        
        # 7. Output Projection
        # Flatten for matmul
        out_flat = attn_output[0]
        final_out = explicit_matmul(out_flat, self.W_o, "Attn_Out_Proj") # (T_q, D)
        
        return final_out[np.newaxis, ...], attn_weights

    def backward(self, dout, x_in, mask=None):
        """
        dout: Gradient of Loss w.r.t output (1, T, D)
        x_in: Input to forward (1, T, D)
        Recomputes necessary forward values for backward pass (checkpointing style)
        or uses cached values if we stored them (we didn't store all, so let's recompute cheaply or assume x_in valid).
        
        Ideally, we should have cached Q, K, V, scores, etc during forward.
        For simplicity of this exercise, we'll re-run partial forward or rely on what we can.
        Better: Use a 'cache' arg in forward to store internals if training.
        
        Let's implement a clean backward assuming we can recompute Q,K,V from x_in.
        """
        B, T, D = x_in.shape
        x_flat = x_in[0]
        
        # --- Recompute Forward (Partial) ---
        Q_flat = np.matmul(x_flat, self.W_q)
        K_flat = np.matmul(x_flat, self.W_k)
        V_flat = np.matmul(x_flat, self.W_v)
        
        Q = Q_flat.reshape(T, self.n_heads, self.d_head).transpose(1, 0, 2)[np.newaxis, ...]
        K = K_flat.reshape(T, self.n_heads, self.d_head).transpose(1, 0, 2)[np.newaxis, ...]
        V = V_flat.reshape(T, self.n_heads, self.d_head).transpose(1, 0, 2)[np.newaxis, ...]
        
        # Attention Scores
        K_T = K.transpose(0, 1, 3, 2)
        scores = np.matmul(Q, K_T) / np.sqrt(self.d_head)
        
        # Causal Mask
        if mask is None: # Assume causal training
             full_mask = np.triu(np.ones((T, T)) * -1e9, k=1)
             scores += full_mask
        
        attn_weights = softmax(scores) # (1, H, T, T)
        
        # Output before projection
        attn_output = np.matmul(attn_weights, V) # (1, H, T, Dh)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(1, T, D)
        
        # --- Backward ---
        # 1. Gradient of Output Projection
        # Out = Attn_Out @ W_o
        # dAttn_Out = dout @ W_o.T
        # dW_o = Attn_Out.T @ dout
        
        dout_flat = dout[0] # (T, D)
        attn_out_flat = attn_output[0] # (T, D)
        
        dW_o = np.matmul(attn_out_flat.T, dout_flat)
        d_attn_out = np.matmul(dout_flat, self.W_o.T) # (T, D)
        
        # Reshape d_attn_out to Heads
        d_attn_out = d_attn_out.reshape(1, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3) # (1, H, T, Dh)
        
        # 2. Gradient w.r.t V
        # Out = Weights @ V
        # dV = Weights.T @ dOut
        # dWeights = dOut @ V.T
        
        # dV: (1, H, T, Dh)
        attn_weights_T = attn_weights.transpose(0, 1, 3, 2) # (1, H, T, T)
        dV = np.matmul(attn_weights_T, d_attn_out)
        
        d_weights = np.matmul(d_attn_out, V.transpose(0, 1, 3, 2)) # (1, H, T, T)
        
        # 3. Gradient w.r.t Scores (Softmax)
        # dScores = dWeights * (Weights * (1 - Weights) if i==j else -WiWj)
        # Fast formula for Softmax-CrossEntropy is (P-Y), but here it's inside the net.
        # General Softmax backward: dS = W * (d_weights - sum(d_weights * W, axis=-1, keepdims=True))
        # d_weights is dL/dY. Y = softmax(S).
        
        term1 = d_weights 
        term2 = np.sum(d_weights * attn_weights, axis=-1, keepdims=True)
        d_scores = attn_weights * (term1 - term2)
        d_scores /= np.sqrt(self.d_head) # Scale
        
        # 4. Gradients w.r.t Q, K
        # S = Q @ K.T
        # dQ = dS @ K
        # dK = dS.T @ Q
        
        dQ = np.matmul(d_scores, K) # (1, H, T, Dh)
        dK_T = np.matmul(Q.transpose(0, 1, 3, 2), d_scores) # (1, H, Dh, T) -- wait, dS is (T, T)
        # dS (1, H, T_q, T_k). Q (1, H, T_q, Dh). K (1, H, T_k, Dh).
        # dQ = dS @ K -> (T_q, T_k) @ (T_k, Dh) -> (T_q, Dh). Correct.
        # dK = dS.T @ Q -> (T_k, T_q) @ (T_q, Dh) -> (T_k, Dh). Correct.
        dK = np.matmul(d_scores.transpose(0, 1, 3, 2), Q)
        
        # 5. Back to Projections
        # dQ, dK, dV are (1, H, T, Dh). Flatten to (T, D)
        dQ_flat = dQ.transpose(0, 2, 1, 3).reshape(T, D)
        dK_flat = dK.transpose(0, 2, 1, 3).reshape(T, D)
        dV_flat = dV.transpose(0, 2, 1, 3).reshape(T, D)
        
        # Gradients for W_q, W_k, W_v
        # Q = x @ W_q -> dW_q = x.T @ dQ
        dW_q = np.matmul(x_flat.T, dQ_flat)
        dW_k = np.matmul(x_flat.T, dK_flat)
        dW_v = np.matmul(x_flat.T, dV_flat)
        
        # Gradient w.r.t input x
        # dx_q = dQ @ W_q.T
        dx_q = np.matmul(dQ_flat, self.W_q.T)
        dx_k = np.matmul(dK_flat, self.W_k.T)
        dx_v = np.matmul(dV_flat, self.W_v.T)
        
        dx = dx_q + dx_k + dx_v
        
        return dx[np.newaxis, ...], (dW_q, dW_k, dW_v, dW_o)

    def apply_grads(self, grads, lr=1e-3):
        dW_q, dW_k, dW_v, dW_o = grads
        self.W_q -= lr * dW_q
        self.W_k -= lr * dW_k
        self.W_v -= lr * dW_v
        self.W_o -= lr * dW_o
