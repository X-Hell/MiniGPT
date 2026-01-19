import numpy as np
import sys
import os

# Ensure we can import modules
sys.path.append(os.getcwd())

from mini_transformer.tokenizer import BPETokenizer, TokenizerConfig
from mini_transformer.transformer import MiniTransformer
from mini_transformer.matmul import get_stats
from mini_transformer.visualize import plot_attention, plot_memory_log, plot_entropy, plot_head_similarity, plot_entropy_heatmap

def sample_next_token(logits, temperature=1.0, top_k=40, top_p=0.9, min_p=0.1, repetition_penalty=1.0, context_tokens=None):
    """
    Robust sampling: Temp -> RepPenalty -> TopK -> TopP -> MinP -> Softmax -> Sample
    
    Min-P: Discard tokens with probability < min_p * max_probability
    This is more adaptive than Top-P as it adjusts to confidence level.
    
    logits: (Vocab_Size,)
    """
    logits = logits.copy()  # Don't modify original
    
    # 1. Temperature
    logits = logits / (temperature + 1e-9)

    # 2. Repetition Penalty
    if repetition_penalty != 1.0 and context_tokens is not None:
        # Use a set for O(1)
        # Note with BPE: We might penalize bytes or merged tokens. 
        # The IDs are what matters.
        for token in set(context_tokens):
            if token < len(logits): # Bound check
                if logits[token] < 0:
                    logits[token] *= repetition_penalty
                else:
                    logits[token] /= repetition_penalty

    # 3. Top-K
    if top_k is not None and top_k > 0:
        kdict = min(top_k, len(logits))
        kth_val = np.partition(logits, -kdict)[-kdict]
        indices_to_remove = logits < kth_val
        logits[indices_to_remove] = -float('Inf')

    # 4. Top-P (Nucleus)
    if top_p is not None and top_p < 1.0:
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        
        # Softmax on sorted logits
        max_proto = np.max(sorted_logits)
        if max_proto == -float('Inf'):
            probs = np.ones_like(logits) / len(logits)
        else:
            exp_proto = np.exp(sorted_logits - max_proto)
            sorted_probs = exp_proto / np.sum(exp_proto)
            
        cumulative_probs = np.cumsum(sorted_probs)
        
        # Remove tokens with cumulative probability above the threshold
        # Shift rights to keep first
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1]
        sorted_indices_to_remove[0] = False
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float('Inf')

    # 5. Min-P (newer, better alternative to Top-P for low-confidence filtering)
    if min_p is not None and min_p > 0:
        # Compute probabilities
        max_logit = np.max(logits)
        if max_logit != -float('Inf'):
            exp_logits = np.exp(logits - max_logit)
            probs = exp_logits / np.sum(exp_logits)
            
            # Min-P threshold: discard tokens with prob < min_p * max_prob
            p_max = np.max(probs)
            threshold = min_p * p_max
            logits[probs < threshold] = -float('Inf')

    # 6. Final Softmax & Sample
    max_val = np.max(logits)
    if max_val == -float('Inf'):
         probs = np.ones_like(logits) / len(logits)
    else:
         exp_vals = np.exp(logits - max_val)
         probs = exp_vals / np.sum(exp_vals)
         
    return np.random.choice(len(probs), p=probs), probs

def main():
    print("=== Mini Transformer Inference Engine (BPE + GQA) ===")
    
    # Configuration
    # We load BPE model
    if not os.path.exists("models/tokenizer.model"):
        print("[Error] BPE model 'models/tokenizer.model' not found. Please run train.py first.")
        return

    config = TokenizerConfig(vocab_size=2048)  # Match train.py
    tokenizer = BPETokenizer(config)
    tokenizer.load("models/tokenizer.model")
    
    # Try to load trained model
    # Prefer quantized
    if os.path.exists("models/mini_transformer_model_quantized.pkl"):
        model_path = "models/mini_transformer_model_quantized.pkl"
    else:
        model_path = "models/mini_transformer_model.pkl"
        
    if os.path.exists(model_path):
        print(f"[Init] Loading trained model from {model_path}...")
        import pickle
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        # Reset KV Cache state from training
        from mini_transformer.kv_cache import KVCache
        # model.n_layers check
        if hasattr(model, 'layers'):
             n_heads = model.layers[0].attn.n_heads
             d_head = model.layers[0].attn.d_head
             n_kv_heads = getattr(model.layers[0].attn, 'n_kv_heads', n_heads)
             n_layers = getattr(model, 'n_layers', 2)
        else:
             n_heads = 4
             d_head = 60
             n_kv_heads = n_heads
             n_layers = 2
             
        model.kv_cache = KVCache(model.kv_cache.max_len, n_kv_heads, d_head, n_layers=n_layers)
    else:
        print("[Init] No trained model found. Initializing random weights.")
        model = MiniTransformer(
            vocab_size=vocab_size, # 2048 from prior edit (will be verified)
            d_model=384,
            n_heads=6,
            n_kv_heads=3,
            max_len=256,
            n_layers=6
        )
    
    # Input
    text = "Once upon a time,"
    print(f"\n[Input] '{text}'")
    
    # Encoding
    tokens = tokenizer.encode(text)
    print(f"[Tokens] {tokens}")
    
    # Inference "prefill" (processing the prompt)
    print("\n--- Phase 1: Prefill (Prompt Processing) ---")
    logits, attn_weights, hidden_states = model.forward(np.array(tokens), start_pos=0, return_intermediates=True)
    
    # Logit Lens: See what each layer predicts
    print("\n📊 Logit Lens (Per-Layer Predictions):")
    print("-" * 45)
    lens_results = model.logit_lens(hidden_states)
    for layer_idx, layer_logits in lens_results:
        top_k_indices = np.argsort(layer_logits)[-3:][::-1]
        top_tokens = [tokenizer.decode([idx]) for idx in top_k_indices]
        top_probs = np.exp(layer_logits[top_k_indices] - np.max(layer_logits))
        top_probs = top_probs / np.sum(np.exp(layer_logits - np.max(layer_logits)))
        
        layer_name = "Embed" if layer_idx == 0 else f"Layer {layer_idx}"
        tokens_str = ", ".join([f"'{t}' ({p:.1%})" for t, p in zip(top_tokens, top_probs)])
        print(f"  {layer_name:8} → {tokens_str}")
    print("-" * 45)
    
    # Decode last token prediction
    next_token_id, _ = sample_next_token(logits[0, -1], temperature=0.9, top_k=40, min_p=0.1)
    
    print(f"[Prediction] Next token ID: {next_token_id}")
    print(f"[Prediction] Next char: '{tokenizer.decode([next_token_id])}'")
    
    # Visualization items
    token_strs = [tokenizer.decode([t]) for t in tokens]
    from mini_transformer.visualize import (plot_attention, plot_memory_log, plot_entropy, 
                                              plot_head_similarity, plot_entropy_heatmap, 
                                              plot_interactive_attention, plot_logprobs, 
                                              plot_head_contribution)
    plot_attention(attn_weights, token_strs, save_path="attn_prefill.png")
    plot_interactive_attention(attn_weights, token_strs, save_path="attn_interactive.html")
    plot_entropy(attn_weights, save_path="entropy.png")
    plot_head_similarity(attn_weights, save_path="head_similarity.png")
    plot_entropy_heatmap(attn_weights, save_path="entropy_heatmap.png")
    plot_head_contribution(attn_weights, save_path="head_contribution.png")
    
    # Generation Step (Autoregressive)
    print("\n--- Phase 2: Generation (Step-by-Step) ---")
    
    generated_ids = list(tokens)
    current_ids = [next_token_id]
    logprobs_collected = []  # Track log probs for each generated token
    generated_tokens_decoded = []  # Track decoded tokens for labels
    
    # Early-exit tracking
    low_conf_streak = 0
    LOW_CONF_THRESHOLD = -4.5
    MAX_LOW_CONF_STREAK = 3
    
    # Let's generate 50 tokens (longer)
    max_tokens = 50
    for i in range(max_tokens):
        pos = len(generated_ids)
        
        # Forward pass for ONE token
        logits, attn_weights = model.forward(np.array(current_ids), start_pos=pos)
        
        # Compute attention entropy for adaptive decoding
        p_attn = attn_weights[0]  # (H, 1, T_k)
        log_p_attn = np.log(p_attn + 1e-9)
        entropy = -np.sum(p_attn * log_p_attn, axis=-1).mean()  # Average entropy across heads
        
        # Entropy-aware temperature/top_k adjustment (Simplified to Temp Decay for Phase 19)
        # Decay: Start creative (0.8), focus down (0.6)
        # temp = max(0.6, 0.8 * (0.99 ** generated_count))
        
        # Calculate generated count
        gen_count = len(generated_ids) - len(tokens)
        temp_decay = max(0.6, 0.8 * (0.99 ** gen_count))
        
        # Use simple logic for now, overriding entropy if valid
        temp = temp_decay
        top_k = 40
        
        # Keep entropy monitoring but use decay for control
        if entropy > 2.5:
             # If EXTREMELY uncertain, maybe boost temp slightly back up?
             pass 
        elif entropy < 1.0:
             # Logic aligns with decay mostly
             pass
        
        # Sample
        next_id, probs = sample_next_token(logits[0, -1], 
                                           temperature=temp, 
                                           top_k=top_k,
                                           top_p=0.9, 
                                           repetition_penalty=1.15, 
                                           context_tokens=generated_ids[-20:])  # Only last 20 for penalty
        
        # Track log prob of chosen token
        chosen_logprob = np.log(probs[next_id] + 1e-9)
        logprobs_collected.append(chosen_logprob)
        
        # Early-exit check
        if chosen_logprob < LOW_CONF_THRESHOLD:
            low_conf_streak += 1
            if low_conf_streak >= MAX_LOW_CONF_STREAK:
                print(f"\n[Early-Exit] Aborting after {MAX_LOW_CONF_STREAK} consecutive low-confidence tokens.")
                break
        else:
            low_conf_streak = 0
        
        char = tokenizer.decode([next_id])
        generated_tokens_decoded.append(char)
        sys.stdout.write(char)
        sys.stdout.flush()
        
        generated_ids.append(current_ids[0])
        current_ids = [next_id]

    print(f"\n\n[Final Output] '{tokenizer.decode(generated_ids)}'")
    
    # Stability Checks
    print("\n=== Stability Checks ===")
    avg_logprob = np.mean(logprobs_collected)
    min_logprob = np.min(logprobs_collected)
    low_conf_count = sum(1 for lp in logprobs_collected if lp < -4.0)
    print(f"Avg Log Prob: {avg_logprob:.3f}")
    print(f"Min Log Prob: {min_logprob:.3f}")
    print(f"Low Confidence Tokens (<-4.0): {low_conf_count}/{len(logprobs_collected)}")
    
    # Token repetition check
    from collections import Counter
    token_counts = Counter(generated_ids)
    max_repeat = max(token_counts.values())
    total_tokens = len(generated_ids)
    repetition_rate = max_repeat / total_tokens * 100
    print(f"Max Token Repetition: {max_repeat}/{total_tokens} ({repetition_rate:.1f}%)")
    if repetition_rate > 15:
        print("[WARNING] High token repetition detected!")
    
    # Plot logprobs
    plot_logprobs(logprobs_collected, tokens=generated_tokens_decoded, save_path="logprobs.png")
    
    # Stats
    stats = get_stats()
    print("\n=== Performance Metrics ===")
    print(f"Total MatMul Ops Logged: {len(stats)}")
    
    # Plot memory usage
    plot_memory_log(stats, save_path="memory_log.png")
    
    # Timeline
    from mini_transformer.visualize import plot_inference_timeline
    plot_inference_timeline(stats, save_path="inference_timeline.png")

if __name__ == "__main__":
    main()
