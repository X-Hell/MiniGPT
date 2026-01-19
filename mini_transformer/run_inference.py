import numpy as np
import sys
import os

# Ensure we can import modules
sys.path.append(os.getcwd())

from mini_transformer.tokenizer import BPETokenizer, TokenizerConfig
from mini_transformer.transformer import MiniTransformer
from mini_transformer.matmul import get_stats
from mini_transformer.visualize import plot_attention, plot_memory_log, plot_entropy, plot_head_similarity, plot_entropy_heatmap

def sample_next_token(logits, temperature=1.0, top_k=40, top_p=0.9, repetition_penalty=1.0, context_tokens=None):
    """
    Robust sampling: Temp -> RepPenalty -> TopK -> TopP -> Softmax -> Sample
    logits: (Vocab_Size,)
    """
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

    # 5. Final Softmax & Sample
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
    if not os.path.exists("tokenizer.model"):
        print("[Error] BPE model 'tokenizer.model' not found. Please run train.py first.")
        return

    config = TokenizerConfig(vocab_size=300) # Match train.py
    tokenizer = BPETokenizer(config)
    tokenizer.load("tokenizer.model")
    
    # Try to load trained model
    # Prefer quantized
    if os.path.exists("mini_transformer_model_quantized.pkl"):
        model_path = "mini_transformer_model_quantized.pkl"
    else:
        model_path = "mini_transformer_model.pkl"
        
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
            vocab_size=300,
            d_model=240,
            n_heads=4,
            n_kv_heads=2,
            max_len=128
        )
    
    # Input
    text = "Hello AI"
    print(f"\n[Input] '{text}'")
    
    # Encoding
    tokens = tokenizer.encode(text)
    print(f"[Tokens] {tokens}")
    
    # Inference "prefill" (processing the prompt)
    print("\n--- Phase 1: Prefill (Prompt Processing) ---")
    logits, attn_weights = model.forward(np.array(tokens), start_pos=0)
    
    # Decode last token prediction
    next_token_id, _ = sample_next_token(logits[0, -1], temperature=0.9, top_k=40)
    
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
        
        # Entropy-aware temperature/top_k adjustment
        # High entropy (>2.5) = model uncertain → increase temperature, increase top_k
        # Low entropy (<1.0) = model confident → lower temperature
        if entropy > 2.5:
            temp = 1.0
            top_k = 50
        elif entropy < 1.0:
            temp = 0.7
            top_k = 20
        else:
            temp = 0.9
            top_k = 40
        
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
