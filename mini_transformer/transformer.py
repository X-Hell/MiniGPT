import numpy as np
from .matmul import explicit_matmul
from .attention import MultiHeadAttention

class RMSNorm:
    def __init__(self, d_model, eps=1e-5):
        self.gamma = np.ones(d_model, dtype=np.float32)
        self.eps = eps
        
    def forward(self, x):
        # x: (N, D)
        self.ms = np.mean(x**2, axis=-1, keepdims=True)
        self.rms = np.sqrt(self.ms + self.eps)
        self.x_norm = x / self.rms
        return self.gamma * self.x_norm
        
    def backward(self, dout):
        # dout: (..., D)
        # Handle flattening
        orig_shape = dout.shape
        D = orig_shape[-1]
        dout_flat = dout.reshape(-1, D)
        x_norm_flat = self.x_norm.reshape(-1, D)
        x_flat = (self.x_norm * self.rms).reshape(-1, D) # Reconstruct x
        
        # dGamma
        self.d_gamma = np.sum(dout_flat * x_norm_flat, axis=0)
        
        # dX
        # dl/dx_norm = dout * gamma
        dx_norm = dout_flat * self.gamma
        
        # RMSNorm Backward
        # dx = 1/rms * (dx_norm - x_norm * mean(dx_norm * x_norm))
        # Wait, derivation check:
        # y = x / rms
        # dy = dx/rms - x * drms / rms^2
        # drms = 0.5/rms * dms
        # dms = 2x/D * dx
        # ...
        # Standard streamlined formula:
        # dx = (1/rms) * (dx_norm - x_norm * mean(dx_norm * x_norm))
        # Note: mean is over D dimension.
        
        mean_compound = np.mean(dx_norm * x_norm_flat, axis=-1, keepdims=True)
        dx = (dx_norm - x_norm_flat * mean_compound) / self.rms.reshape(-1, 1) # rms was (N, 1)
        
        return dx.reshape(orig_shape)

    def apply_grads(self, lr=1e-3, optimizer=None):
        if optimizer:
            optimizer.step([self.gamma], [self.d_gamma])
        else:
            self.gamma -= lr * self.d_gamma

class FeedForward:
    """
    SwiGLU Feed-Forward Network (Llama/Mistral standard)
    
    FFN(x) = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down
    
    SiLU(x) = x * sigmoid(x)
    """
    def __init__(self, d_model, d_ff, rank=None):
        # Ignore rank argument to maintain signature compatibility
        self.d_model = d_model
        # Reduce hidden dim to keep params ~constant with 3 matrices
        # Original: 2 * d_model * d_ff = 2 * 240 * 600 = 288K
        # SwiGLU: 3 * d_model * d_ff_new, solve for d_ff_new ≈ 512
        self.d_ff = int(d_ff * 2 / 3)  # 600 * 2/3 = 400, but we use 512 for power-of-2
        self.d_ff = max(256, min(512, self.d_ff))  # Clamp to [256, 512]
        
        # SwiGLU: three matrices
        scale = 1.0 / np.sqrt(d_model)
        self.W_gate = np.random.normal(scale=scale, size=(d_model, self.d_ff)).astype(np.float32)
        self.W_up = np.random.normal(scale=scale, size=(d_model, self.d_ff)).astype(np.float32)
        
        scale_down = 1.0 / np.sqrt(self.d_ff)
        self.W_down = np.random.normal(scale=scale_down, size=(self.d_ff, d_model)).astype(np.float32)
        
        self.quantized = False
        mem = (self.W_gate.nbytes + self.W_up.nbytes + self.W_down.nbytes)
        print(f"[FFN] SwiGLU ({d_model}->gate/up:{self.d_ff}->down:{d_model}) Weights Mem: {mem/1024:.2f} KB")
    
    @staticmethod
    def silu(x):
        """SiLU/Swish activation: x * sigmoid(x)"""
        return x * (1.0 / (1.0 + np.exp(-np.clip(x, -88, 88))))
    
    @staticmethod
    def silu_backward(x, grad_out):
        """
        d/dx[SiLU(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                      = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        """
        sig = 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))
        return grad_out * (sig + x * sig * (1 - sig))
    
    def quantize(self):
        """Quantizes weights to INT8."""
        if self.quantized:
            return
            
        print(f"[FFN] Quantizing SwiGLU matrices to INT8...")
        
        from .quant_utils import quantize_matrix
        
        self.W_gate_int8, self.gate_scale = quantize_matrix(self.W_gate)
        self.W_up_int8, self.up_scale = quantize_matrix(self.W_up)
        self.W_down_int8, self.down_scale = quantize_matrix(self.W_down)
        
        # Release float weights
        del self.W_gate, self.W_up, self.W_down
        self.quantized = True
        
        mem_new = (self.W_gate_int8.nbytes + self.W_up_int8.nbytes + self.W_down_int8.nbytes +
                   self.gate_scale.nbytes + self.up_scale.nbytes + self.down_scale.nbytes)
                   
        print(f"[FFN] Quantized Mem: {mem_new/1024:.2f} KB")

    def forward(self, x):
        # x: (..., D)
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        
        if self.quantized:
            W_gate = self.W_gate_int8.astype(np.float32) * self.gate_scale
            W_up = self.W_up_int8.astype(np.float32) * self.up_scale
            W_down = self.W_down_int8.astype(np.float32) * self.down_scale
        else:
            W_gate, W_up, W_down = self.W_gate, self.W_up, self.W_down
        
        # SwiGLU: (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down
        gate_pre = explicit_matmul(x_flat, W_gate, "FFN_Gate")
        gate_out = self.silu(gate_pre)
        
        up_out = explicit_matmul(x_flat, W_up, "FFN_Up")
        
        # Element-wise gating
        hidden = gate_out * up_out
        
        out = explicit_matmul(hidden, W_down, "FFN_Down")
            
        return out.reshape(orig_shape)

    def backward(self, dout, x_in):
        if self.quantized:
            raise NotImplementedError("Gradient calc on quantized model not supported")
            
        orig_shape = x_in.shape
        x_flat = x_in.reshape(-1, orig_shape[-1])
        dout_flat = dout.reshape(-1, orig_shape[-1])
        
        # Recompute Forward for gradient
        gate_pre = np.matmul(x_flat, self.W_gate)
        gate_out = self.silu(gate_pre)
        up_out = np.matmul(x_flat, self.W_up)
        hidden = gate_out * up_out
        
        # Backward Pass
        
        # d_W_down: hidden^T @ dout
        dW_down = np.matmul(hidden.T, dout_flat)
        
        # d_hidden: dout @ W_down^T
        d_hidden = np.matmul(dout_flat, self.W_down.T)
        
        # d_gate_out = d_hidden ⊙ up_out
        d_gate_out = d_hidden * up_out
        
        # d_up_out = d_hidden ⊙ gate_out
        d_up_out = d_hidden * gate_out
        
        # d_gate_pre = SiLU backward
        d_gate_pre = self.silu_backward(gate_pre, d_gate_out)
        
        # dW_gate, dW_up
        dW_gate = np.matmul(x_flat.T, d_gate_pre)
        dW_up = np.matmul(x_flat.T, d_up_out)
        
        # dx
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

class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff, n_kv_heads=None, ffn_rank=32):
        from .attention import MultiHeadAttention
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, n_kv_heads=n_kv_heads)
        self.ln2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, rank=ffn_rank)
        
    def forward(self, x, kv_cache, start_pos, layer_idx):
        # x: (B, T, D)
        
        # 1. Attn Block
        resid1 = x
        x_norm_1 = self.ln1.forward(x)
        attn_out, attn_weights = self.attn.forward(x_norm_1, kv_cache, start_pos, layer_idx)
        x2 = resid1 + attn_out
        
        # 2. FFN Block
        resid2 = x2
        x_norm_2 = self.ln2.forward(x2)
        ffn_out = self.ffn.forward(x_norm_2)
        x3 = resid2 + ffn_out
        
        # Cache for backward
        self.cache_x = x
        self.cache_x2 = x2
        
        return x3, attn_weights

    def backward(self, dout):
        """
        dout: Gradient from next layer (dL/dX3)
        """
        # FFN Block Backprop
        # X3 = X2 + FFN(LN2(X2))
        dX3 = dout
        dX2_resid = dX3
        
        dFFN_out = dX3
        dX_norm_2, ffn_grads = self.ffn.backward(dFFN_out, self.ln2.x_norm)
        dX2_branch = self.ln2.backward(dX_norm_2)
        
        dX2 = dX2_resid + dX2_branch
        
        # Attn Block Backprop
        # X2 = X + Attn(LN1(X))
        dX_resid = dX2
        
        dAttn_out = dX2
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
        
    def quantize(self):
        self.ffn.quantize()
        self.attn.quantize()

class MiniTransformer:
    def __init__(self, vocab_size, d_model=240, n_heads=4, n_kv_heads=2, max_len=256, n_layers=2):
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads 
        self.n_kv_heads = n_kv_heads
        
        from .embeddings import EmbeddingLayer
        from .kv_cache import KVCache
        
        self.embeddings = EmbeddingLayer(vocab_size, d_model, max_len)
        self.kv_cache = KVCache(max_len, n_kv_heads, d_model // n_heads, n_layers) 
        # Note: KVCache needs to support n_layers now!
        
        self.layers = []
        for _ in range(n_layers):
            # FFN: 2.5x expansion instead of 4x (better for small data, prevents memorization)
            self.layers.append(TransformerBlock(d_model, n_heads, int(d_model * 2.5), n_kv_heads=n_kv_heads, ffn_rank=32))
            
        self.ln_f = RMSNorm(d_model)
        
        print(f"[Transformer] Initialized {n_layers} layers (Weight Tying Enabled).")

    def forward(self, token_ids, start_pos=0, targets=None, return_intermediates=False):
        if len(token_ids.shape) == 1:
            token_ids = token_ids[np.newaxis, :]
            
        B, T = token_ids.shape
        
        # Embeddings
        x_emb = []
        for i in range(B):
            x_emb.append(self.embeddings.forward_seq(token_ids[i], start_pos))
        x = np.array(x_emb) # (B, T, D)
        
        self.cache_x_emb = x.copy()
        
        # Layers
        all_attn_weights = []
        intermediate_states = [x.copy()] if return_intermediates else None
        
        for i, layer in enumerate(self.layers):
            x, attn_w = layer.forward(x, self.kv_cache, start_pos, layer_idx=i)
            all_attn_weights.append(attn_w)
            if return_intermediates:
                intermediate_states.append(x.copy())
            
        # Final Norm
        x_final = self.ln_f.forward(x)
        self.cache_x_final = x_final.copy()
        
        # Output Projection (Tied)
        x_flat = x_final.reshape(-1, self.d_model)
        logits = explicit_matmul(x_flat, self.embeddings.W_emb.T, "Logits (Tied)")
        logits = logits.reshape(B, T, self.embeddings.vocab_size)
        
        # Return last layer attn for viz or all? 
        # Run inference expects (B, H, T, T). We can return list or just last.
        # For compatibility with visualize.py, let's return the last one for now or stack.
        # visualize.py plots one set.
        if return_intermediates:
            return logits, all_attn_weights[-1], intermediate_states
        return logits, all_attn_weights[-1] # Return last layer attention
    
    def logit_lens(self, hidden_states, position=-1):
        """
        Apply output projection to each layer's hidden state.
        
        This is a powerful interpretability tool that shows what the model
        would predict at each layer, revealing how the prediction evolves.
        
        Args:
            hidden_states: List of hidden states from forward_with_intermediates
            position: Token position to analyze (-1 for last token)
            
        Returns:
            List of (layer_idx, logits) tuples
        """
        results = []
        for layer_idx, h in enumerate(hidden_states):
            # Apply final layer norm
            h_norm = self.ln_f.forward(h)
            
            # Get hidden state at specified position
            h_pos = h_norm[0, position, :]  # (D,)
            
            # Project to vocabulary (tied weights)
            logits = np.dot(h_pos, self.embeddings.W_emb.T)  # (V,)
            
            results.append((layer_idx, logits))
        
        return results

    def backward(self, dlogits):
        B, T, V = dlogits.shape
        dlogits_flat = dlogits.reshape(-1, V)
        x_final_flat = self.cache_x_final.reshape(-1, self.d_model)
        
        # 1. Output Head (Embedding Tying)
        dX_final = np.matmul(dlogits_flat, self.embeddings.W_emb)
        dW_emb = np.matmul(dlogits_flat.T, x_final_flat)
        
        dX_final = dX_final.reshape(B, T, self.d_model)
        
        # 2. Final Norm
        dX = self.ln_f.backward(dX_final)
        
        # 3. Layers (Reverse)
        layer_grads = []
        for layer in reversed(self.layers):
            dX, l_grads = layer.backward(dX)
            layer_grads.append(l_grads)
        layer_grads.reverse()
        
        # 4. Embeddings (Leaf)
        dX_emb = dX
        # dX_emb needed for embedding update?
        
        return dW_emb, layer_grads, dX_emb

    def apply_grads(self, grads, token_ids, lr=1e-3, optimizer=None):
        dW_emb_out, layer_grads, dX_emb = grads
        
        # Layers
        for layer, l_grads in zip(self.layers, layer_grads):
            layer.apply_grads(l_grads, lr, optimizer)
            
        self.ln_f.apply_grads(lr, optimizer)
        
        # Embeddings
        # Note: dW_emb_out is gradient from output projection
        # dX_emb is gradient from input lookup
        # We need to combine them or update separately?
        # AdamW needs one update per param.
        # So we should accumulate gradients first.
        
        # Accumulate input gradients into a full dW_emb matrix
        # dX_emb: (B, T, D). token_ids: (B, T)
        # We need to scatter add.
        
        dW_total = dW_emb_out.copy() # Already full shape (V, D)
        
        # Scatter add dX_emb
        B, T = token_ids.shape
        flat_ids = token_ids.flatten()
        flat_grads = dX_emb.reshape(-1, self.d_model)
        
        # Use simple loop or ufunc (numpy doesn't have partial scatter add easily for Adam state, 
        # but for gradient accumulation we can use at)
        # Wait, if we use Adam, we pass (W, dW).
        # We must construct the full dW.
        
        np.add.at(dW_total, flat_ids, flat_grads)
        
        if optimizer:
            optimizer.step([self.embeddings.W_emb], [dW_total])
        else:
            self.embeddings.W_emb -= lr * dW_total
