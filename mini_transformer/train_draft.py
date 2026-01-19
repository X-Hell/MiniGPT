import numpy as np
import sys
import os
import pickle

# Ensure we can import modules
sys.path.append(os.getcwd())

from mini_transformer.tokenizer import MiniTokenizer, TokenizerConfig
from mini_transformer.transformer import MiniTransformer
from mini_transformer.matmul import explicit_matmul

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
    
    # Softmax
    # Numerical stability
    max_logits = np.max(logits_flat, axis=1, keepdims=True)
    exp_logits = np.exp(logits_flat - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Select target probs
    # We want -log(probs[range(N), targets])
    N = logits_flat.shape[0]
    relevant_probs = probs[np.arange(N), targets_flat]
    log_probs = -np.log(relevant_probs + 1e-9)
    loss = np.mean(log_probs)
    
    # Gradient: P - 1 (at target)
    dlogits = probs.copy()
    dlogits[np.arange(N), targets_flat] -= 1
    dlogits /= N # Scale by batch size for mean
    
    return loss, dlogits.reshape(B, T, V)

def train_step(model, token_ids, targets, lr=1e-3):
    """
    Manual forward + backward + update
    """
    # 1. Forward
    # Note: We need to capture INTERMEDIATE ACTIVATIONS.
    # Our simple forward implementation doesn't return them currently.
    # We must hack or rely on `model.ln1.x_norm` etc being set during forward for the *last* batch.
    # Our modules currently store `self.x_norm` etc in forward, so this works for BATCH=1 or sequential simple.
    # BUT: `transformer.py` logic:
    # x_norm_1 = self.ln1.forward(x)
    # The `ln1` object updates its internal `self.x_norm`.
    # So if we run forward, the state is primed for backward.
    # THIS RELIES ON: No intervening calls. Functional safety is low, but acceptable for this "mini" constraint.
    
    # Wait, `MiniTransformer.forward` does:
    # x_emb = ...
    # x = resid + attn_out
    # We *don't* store 'resid' or 'x_emb' in the class.
    # We need those for the backward path (e.g. gradient passed to residual branch).
    # Since we can't easily change the signature too much without breaking everything,
    # let's modify `MiniTransformer` to store `self.cache` during training forward.
    # Or, we just re-run parts or pass them.
    #
    # Actually, let's just add state to MiniTransformer for the simple training loop.
    # It's stateful anyway with KV cache (though that's for inference).
    
    # Let's do a Forward that returns cache, or stores it.
    # Re-running forward is standard for checkpointing but maybe too complex here.
    # Let's TRUST the modules state for LN and FFN.
    # For residuals, we need the input to the block.
    # We can reconstruct? No.
    #
    # Let's augment MiniTransformer.forward to save 'x_in_block1', 'x_in_block2' if training.
    # But for now, we will add a `train_forward` method to `train.py` wrapper? No, needs access to weights.
    #
    # Simpler: Just add `self.last_x_1`, `self.last_x_2` etc to transformer forward.
    
    logits, _ = model.forward(token_ids)
    
    # 2. Loss
    loss, dlogits = cross_entropy_loss(logits, targets)
    
    # 3. Backward
    # Reverse of Forward
    
    # dLogits -> dX_final
    # Logits = X_final @ W_emb.T
    # dX_final = dLogits @ W_emb
    # dW_emb = dLogits.T @ X_final
    
    # Re-get intermediates. We need them.
    # We'll just assume we modified transformer.py? 
    # Let's modify transformer.py to store 'self.x_final_norm_in' etc?
    # Actually, let's just make the training loop simpler: Overfit a tiny string so we can verify components.
    #
    # Critical Missing Link: The residuals `x = resid + attn_out`. Gradient branches.
    # dResidual = dX_next
    # dAttnOrFFN = dX_next
    #
    # We need the input to the layer to compute its gradient.
    # Let's Update transformer.py to cache these inputs. 
    #
    # Wait, I cannot edit transformer.py in this step easily alongside train.py without tool switching.
    # I will assume I can update transformer.py next.
    pass

def train():
    # ...
    pass
