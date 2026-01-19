import numpy as np
import sys
import os
import pickle
import time

# Ensure we can import modules
sys.path.append(os.getcwd())

from mini_transformer.tokenizer import MiniTokenizer, TokenizerConfig
from mini_transformer.transformer import MiniTransformer
from mini_transformer.matmul import explicit_matmul, _LOGGER
from mini_transformer.optimizer import AdamW

def cross_entropy_loss(logits, targets):
    """
    logits: (B, T, Vocab)
    targets: (B, T) - ints
    Returns: scalar loss, dLogits
    """
    B, T, V = logits.shape
    
    # Flatten
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)
    
    # Softmax stable
    max_logits = np.max(logits_flat, axis=1, keepdims=True)
    exp_logits = np.exp(logits_flat - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Loss
    N = logits_flat.shape[0]
    relevant_probs = probs[np.arange(N), targets_flat]
    log_probs = -np.log(relevant_probs + 1e-9)
    loss = np.mean(log_probs)
    
    # Gradient: P - 1 (at target)
    dlogits = probs.copy()
    dlogits[np.arange(N), targets_flat] -= 1
    dlogits /= N # Mean reduction
    
    return loss, dlogits.reshape(B, T, V)

def train_loop(text_path="mini_transformer/train_data.txt", steps=2000, lr=5e-4):
    print("=== Mini Transformer Training ===")
    
    # Config
    B = 1
    # T schedule implemented below
    
    # 1. Dataset
    # Check if we have data, logic inside main block ensures creation or we have downloaded it.
    if not os.path.exists(text_path):
         print("Generating default data...")
         print(f"Creating dummy train data at {text_path}")
         with open(text_path, "w") as f:
             f.write("Hello world! This is a mini transformer training test. " * 20)
            
    with open(text_path, "r") as f:
        text = f.read()
        
    config = TokenizerConfig(vocab_size=256)
    tokenizer = MiniTokenizer(config)
    tokens = tokenizer.encode(text)
    tokens = np.array(tokens)
    print(f"Dataset: {len(tokens)} tokens")
    
    # Init Model
    model = MiniTransformer(
        vocab_size=256,
        d_model=240,
        n_heads=4,
        max_len=128, # Must covers T
        n_layers=2
    )
    
    # Optimizer
    optimizer = AdamW(lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    
    # 3. Loop
    losses = []
    
    # Curriculum Schedule: T starts at 16, ends at 128
    # We linearly interpolate T
    
    for step in range(steps):
        # LR Schedule: Warmup + Cosine
        lr_now = lr 
        if step < 200: # Warmup
             lr_now = lr * (step / 200)
        else:
             progress = (step - 200) / (steps - 200)
             lr_now = 0.5 * lr * (1 + np.cos(np.pi * progress))
             
        # Update optimizer LR
        optimizer.lr = lr_now
             
        # Curriculum
        # Fraction of training done
        train_frac = step / steps
        # Scale T from 16 to 128 (max_len is 128)
        # We clamp it
        current_T = int(16 + (128 - 16) * train_frac)
        current_T = min(current_T, 128)
    
        # Sample batch
        ix = np.random.randint(0, len(tokens) - current_T - 1, size=(B,))
        x = np.stack([tokens[i:i+current_T] for i in ix])
        y = np.stack([tokens[i+1:i+current_T+1] for i in ix])
        
        # Forward
        logits, _ = model.forward(x)
        
        # Loss
        loss, dlogits = cross_entropy_loss(logits, y)
        
        # Backward
        grads = model.backward(dlogits)
        
        # Gradient Clipping (Global Norm)
        # Flatten main weights for norm calc
        dW_emb, layer_grads, _ = grads
        sq_sum = np.sum(dW_emb**2)
        
        for l_grads in layer_grads:
            ffn_grads, attn_grads = l_grads
            dW1a, dW1b, dW2a, dW2b = ffn_grads
            # attn_grads is now (dW_qkv, dW_o) due to fusion
            dW_qkv, dW_o = attn_grads
            
            sq_sum += np.sum(dW1a**2) + np.sum(dW1b**2) + np.sum(dW2a**2) + np.sum(dW2b**2)
            sq_sum += np.sum(dW_qkv**2) + np.sum(dW_o**2)
            
        grad_norm = np.sqrt(sq_sum)
        max_norm = 1.0
        
        clip_scale = 1.0
        if grad_norm > max_norm:
            clip_scale = max_norm / grad_norm
            
        # Helper to scale structured grads
        def scale_grads(g, scale):
            if isinstance(g, tuple):
                return tuple(scale_grads(x, scale) for x in g)
            elif isinstance(g, list):
                return [scale_grads(x, scale) for x in g]
            elif isinstance(g, np.ndarray):
                return g * scale
            return g
            
        if clip_scale < 1.0:
            grads = scale_grads(grads, clip_scale)
        
        # Update
        # Note: apply_grads signature update in transformer.py uses optimizer if passed
        model.apply_grads(grads, x, lr=lr_now, optimizer=optimizer)
        
        if step % 50 == 0:
            print(f"Step {step:4d} | Loss: {loss:.4f} | PPL: {np.exp(loss):.2f} | LR: {lr_now:.5f} | Norm: {grad_norm:.2f}")
            losses.append(loss)

    print(f"Final Loss: {losses[-1]:.4f}")
    
    # Save Weights
    print("Saving weights to model.npz...")
    # Helper to gather state (simplified)
    # We can use pickle for the whole object for simplicity in this prototype
    with open("mini_transformer_model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    # Check Quantization Memory Saving
    print("\n--- Quantization Check ---")
    for i, layer in enumerate(model.layers):
        # We only check/print for the first one to avoid spam
        if i == 0:
             print(f"[Layer {i} FFN] Quantizing to INT8 (Per-Channel)...")
        layer.quantize()
    
    print("\nTraining Complete.")
    
if __name__ == "__main__":
    train_loop()
