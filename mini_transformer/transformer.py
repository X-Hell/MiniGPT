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
    def __init__(self, d_model, d_ff, rank=None):
        # Ignore rank argument to maintain signature compatibility but do standard FFN
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Standard FFN: Expansion -> Activation -> Projection
        # W1: (D, D_ff)
        # W2: (D_ff, D)
        
        scale_1 = 1.0 / np.sqrt(d_model)
        self.W1 = np.random.normal(scale=scale_1, size=(d_model, d_ff)).astype(np.float32)
        
        scale_2 = 1.0 / np.sqrt(d_ff)
        self.W2 = np.random.normal(scale=scale_2, size=(d_ff, d_model)).astype(np.float32)
        
        self.quantized = False
        mem = (self.W1.nbytes + self.W2.nbytes)
        print(f"[FFN] Standard ({d_model}->{d_ff}->{d_model}) Weights Mem: {mem/1024:.2f} KB")
    
    def quantize(self):
        """Quantizes weights to INT8."""
        if self.quantized:
            return
            
        print(f"[FFN] Quantizing Standard matrices to INT8...")
        
        from .quant_utils import quantize_matrix
        
        self.W1_int8, self.w1_scale = quantize_matrix(self.W1)
        self.W2_int8, self.w2_scale = quantize_matrix(self.W2)
        
        # Release float weights
        del self.W1, self.W2
        self.quantized = True
        
        mem_new = (self.W1_int8.nbytes + self.W2_int8.nbytes +
                   self.w1_scale.nbytes + self.w2_scale.nbytes)
                   
        print(f"[FFN] Quantized Mem: {mem_new/1024:.2f} KB")

    def forward(self, x):
        # x: (..., D)
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        
        if self.quantized:
            # Dequantize on fly
            W1 = self.W1_int8.astype(np.float32) * self.w1_scale
            h = explicit_matmul(x_flat, W1, "FFN_1 (INT8)")
        else:
            h = explicit_matmul(x_flat, self.W1, "FFN_1")
            
        # ReLU (In-place)
        np.maximum(h, 0, out=h)
        
        if self.quantized:
            W2 = self.W2_int8.astype(np.float32) * self.w2_scale
            out = explicit_matmul(h, W2, "FFN_2 (INT8)")
        else:
            out = explicit_matmul(h, self.W2, "FFN_2")
            
        return out.reshape(orig_shape)

    def backward(self, dout, x_in):
        if self.quantized:
            raise NotImplementedError("Gradient calc on quantized model not supported")
            
        orig_shape = x_in.shape
        x_flat = x_in.reshape(-1, orig_shape[-1])
        dout_flat = dout.reshape(-1, orig_shape[-1])
        
        # Recompute Forward
        # 1. W1
        h_pre = np.matmul(x_flat, self.W1)
        h_relu = np.maximum(h_pre, 0)
        
        # Backward Pass
        
        # dW2
        dW2 = np.matmul(h_relu.T, dout_flat)
        dh_relu = np.matmul(dout_flat, self.W2.T)
        
        # dReLU
        dh_pre = dh_relu * (h_pre > 0)
        
        # dW1
        dW1 = np.matmul(x_flat.T, dh_pre)
        dx_flat = np.matmul(dh_pre, self.W1.T)
        
        return dx_flat.reshape(orig_shape), (dW1, dW2)

    def apply_grads(self, grads, lr=1e-3, optimizer=None):
        dW1, dW2 = grads
        if optimizer:
            optimizer.step([self.W1, self.W2], [dW1, dW2])
        else:
            self.W1 -= lr * dW1
            self.W2 -= lr * dW2

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
    def __init__(self, vocab_size, d_model=240, n_heads=4, n_kv_heads=2, max_len=128, n_layers=2):
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
