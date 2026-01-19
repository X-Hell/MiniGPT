import numpy as np
from .matmul import explicit_matmul
from .attention import MultiHeadAttention

class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.gamma = np.ones(d_model, dtype=np.float32)
        self.beta = np.zeros(d_model, dtype=np.float32)
        self.eps = eps
        
    def forward(self, x):
        # x: (N, D)
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)
        self.x_norm = (x - self.mean) / self.std
        return self.gamma * self.x_norm + self.beta
        
    def backward(self, dout):
        # dout: (N, D)
        # dBeta = sum(dout, axis=0)
        # dGamma = sum(dout * x_norm, axis=0)
        
        D = dout.shape[-1]
        
        if len(dout.shape) == 3:
             # handle (B, T, D) -> flatten to (B*T, D)
             dout_flat = dout.reshape(-1, D)
             x_norm_flat = self.x_norm.reshape(-1, D)
             N = dout_flat.shape[0]
        else:
             dout_flat = dout
             x_norm_flat = self.x_norm
             
        # Debug
        # print(f"LN Backward: dout_flat {dout_flat.shape}, gamma {self.gamma.shape}")
        
        self.d_beta = np.sum(dout_flat, axis=0)
        self.d_gamma = np.sum(dout_flat * x_norm_flat, axis=0)
        
        # dX
        # dl/dx_norm = dout * gamma
        dx_norm = dout_flat * self.gamma
        
        # dl/dvar = sum(dl/dx_norm * (x-mean) * -0.5 * std^-3)
        # dl/dmean = sum(dl/dx_norm * -1/std) + dl/dvar * sum(-2*(x-mean))/M
        
        # Standard LN gradient formula (for simplicity)
        # dx = 1/std * (dx_norm - mean(dx_norm) - x_norm * mean(dx_norm * x_norm))
        # Note: mean() here is over dimension D
        
        mean_dx_norm = np.mean(dx_norm, axis=-1, keepdims=True)
        mean_dx_norm_x_norm = np.mean(dx_norm * x_norm_flat, axis=-1, keepdims=True)
        
        dx = (dx_norm - mean_dx_norm - x_norm_flat * mean_dx_norm_x_norm) / self.std # Note: self.std broadcast might need reshape if 3D
        
        if len(dout.shape) == 3:
            dx = dx.reshape(dout.shape)
            
        return dx

    def apply_grads(self, lr=1e-3):
        self.gamma -= lr * self.d_gamma
        self.beta -= lr * self.d_beta

class FeedForward:
    def __init__(self, d_model, d_ff):
        # W1: d_model -> d_ff
        # W2: d_ff -> d_model
        scale = 1.0 / np.sqrt(d_model)
        self.W1 = np.random.normal(scale=scale, size=(d_model, d_ff)).astype(np.float32)
        
        scale_2 = 1.0 / np.sqrt(d_ff)
        self.W2 = np.random.normal(scale=scale_2, size=(d_ff, d_model)).astype(np.float32)
        
        self.quantized = False
        print(f"[FFN] Weights Mem: {(self.W1.nbytes + self.W2.nbytes)/1024:.2f} KB")
    
    def quantize(self):
        """Quantizes weights to INT8."""
        if self.quantized:
            return
            
        print("[FFN] Quantizing to INT8 (Per-Channel)...")
        
        # W1: (D, d_ff) -> Scale per column (d_ff)
        # Using axis=0: max over input dim, keeping output dim scales
        max_val_1 = np.max(np.abs(self.W1), axis=0) # Shape (d_ff,)
        self.w1_scale = max_val_1 / 127.0
        self.w1_scale[self.w1_scale == 0] = 1.0 # Avoid div by zero
        
        # Quantize: W / scale. Scale broadcast (1, d_ff)
        self.W1_int8 = np.round(self.W1 / self.w1_scale).astype(np.int8)
        
        # W2: (d_ff, D) -> Scale per column (D)
        max_val_2 = np.max(np.abs(self.W2), axis=0) # Shape (D,)
        self.w2_scale = max_val_2 / 127.0
        self.w2_scale[self.w2_scale == 0] = 1.0
        
        self.W2_int8 = np.round(self.W2 / self.w2_scale).astype(np.int8)
        
        # Release float weights
        del self.W1
        del self.W2
        self.quantized = True
        
        mem_new = self.W1_int8.nbytes + self.W2_int8.nbytes + self.w1_scale.nbytes + self.w2_scale.nbytes
        print(f"[FFN] Quantized Mem: {mem_new/1024:.2f} KB")

    def forward(self, x):
        # x: (..., D)
        # Handle quantization dequant on the fly
        
        # Flatten for matmul
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        
        if self.quantized:
            # Dequantize W1 for matmul
            # This is "fake" INT8 inference in that we dequantize to float for the mul,
            # but we save storage memory.
            W1_deq = self.W1_int8.astype(np.float32) * self.w1_scale
            h = explicit_matmul(x_flat, W1_deq, "FFN_1 (INT8)")
        else:
            h = explicit_matmul(x_flat, self.W1, "FFN_1")
            
        # ReLU (In-place)
        np.maximum(h, 0, out=h)
        
        if self.quantized:
            W2_deq = self.W2_int8.astype(np.float32) * self.w2_scale
            out = explicit_matmul(h, W2_deq, "FFN_2 (INT8)")
        else:
            out = explicit_matmul(h, self.W2, "FFN_2")
            
        return out.reshape(orig_shape)

    def backward(self, dout, x_in):
        """
        dout: (..., D)
        """
        if self.quantized:
            raise NotImplementedError("Training quantized FFN not supported")
            
        orig_shape = x_in.shape
        x_flat = x_in.reshape(-1, orig_shape[-1])
        dout_flat = dout.reshape(-1, orig_shape[-1])
        
        # Recompute forward
        h = np.matmul(x_flat, self.W1)
        h_relu = np.maximum(h, 0)
        
        # dW2
        # Out = H_relu @ W2
        dW2 = np.matmul(h_relu.T, dout_flat)
        dh_relu = np.matmul(dout_flat, self.W2.T)
        
        # dReLU
        dh = dh_relu * (h > 0)
        
        # dW1
        dW1 = np.matmul(x_flat.T, dh)
        dx_flat = np.matmul(dh, self.W1.T)
        
        return dx_flat.reshape(orig_shape), (dW1, dW2)

    def apply_grads(self, grads, lr=1e-3):
        dW1, dW2 = grads
        self.W1 -= lr * dW1
        self.W2 -= lr * dW2

class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff):
        from .attention import MultiHeadAttention
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        
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

    def apply_grads(self, grads, lr):
        ffn_grads, attn_grads = grads
        self.ffn.apply_grads(ffn_grads, lr)
        self.attn.apply_grads(attn_grads, lr)
        self.ln1.apply_grads(lr)
        self.ln2.apply_grads(lr)
        
    def quantize(self):
        self.ffn.quantize()

class MiniTransformer:
    def __init__(self, vocab_size, d_model=240, n_heads=4, max_len=128, n_layers=2):
        self.d_model = d_model
        self.n_layers = n_layers
        
        from .embeddings import EmbeddingLayer
        from .kv_cache import KVCache
        
        self.embeddings = EmbeddingLayer(vocab_size, d_model, max_len)
        self.kv_cache = KVCache(max_len, n_heads, d_model // n_heads, n_layers) 
        # Note: KVCache needs to support n_layers now!
        
        self.layers = []
        for _ in range(n_layers):
            self.layers.append(TransformerBlock(d_model, n_heads, d_model * 4))
            
        self.ln_f = LayerNorm(d_model)
        
        print(f"[Transformer] Initialized {n_layers} layers (Weight Tying Enabled).")

    def forward(self, token_ids, start_pos=0, targets=None):
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
        for i, layer in enumerate(self.layers):
            x, attn_w = layer.forward(x, self.kv_cache, start_pos, layer_idx=i)
            all_attn_weights.append(attn_w)
            
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
        return logits, all_attn_weights[-1] # Return last layer attention

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

    def apply_grads(self, grads, token_ids, lr=1e-3):
        dW_emb_out, layer_grads, dX_emb = grads
        
        # Layers
        for layer, l_grads in zip(self.layers, layer_grads):
            layer.apply_grads(l_grads, lr)
            
        self.ln_f.apply_grads(lr)
        
        # Embeddings
        self.embeddings.W_emb -= lr * dW_emb_out
        
        # Input gradients
        B, T = token_ids.shape
        flat_ids = token_ids.flatten()
        flat_grads = dX_emb.reshape(-1, self.d_model)
        np.add.at(self.embeddings.W_emb, flat_ids, -lr * flat_grads)
