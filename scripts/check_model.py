
import sys
import os
import pickle
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def check_model():
    print("=== Model Health Check ===")
    ckpt_path = "checkpoints/model_latest.pkl"
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return

    try:
        with open(ckpt_path, "rb") as f:
            model = pickle.load(f)
        
        print(f"Model Class: {type(model).__name__}")
        print(f"Config: {model.config.__dict__}")
        
        # Check Weights
        print("\n[Weight Statistics]")
        has_issue = False
        
        # Helper to check a tensor
        def scan(name, tensor):
            nonlocal has_issue
            if tensor is None:
                return
            
            t_min, t_max = np.min(tensor), np.max(tensor)
            t_mean, t_std = np.mean(tensor), np.std(tensor)
            n_nans = np.isnan(tensor).sum()
            
            status = "OK"
            if n_nans > 0:
                status = "FAIL (NaNs)"
                has_issue = True
            elif np.abs(t_mean) > 10.0 or t_std > 10.0:
                 status = "WARN (Exploded?)"
            elif t_std < 1e-6:
                 status = "WARN (Collapsed?)"
                 
            print(f"{name:<25} | Shape: {str(tensor.shape):<15} | Mean: {t_mean:+.4f} | Std: {t_std:.4f} | Range: [{t_min:+.2f}, {t_max:+.2f}] | {status}")

        scan("Embeddings", model.embeddings.W_emb)
        for i, layer in enumerate(model.layers):
            scan(f"Layer {i} Attn W_qkv", layer.attn.W_qkv)
            scan(f"Layer {i} Attn W_o", layer.attn.W_o)
            scan(f"Layer {i} FFN W_gate", layer.ffn.W_gate)
            scan(f"Layer {i} FFN W_down", layer.ffn.W_down)
            
        scan("Final Norm", model.ln_f.gamma)
        
        if has_issue:
            print("\n[CONCLUSION] Model is BROKEN. Please run emergency_patch.py.")
        else:
            print("\n[CONCLUSION] Weights look statistically nominal (but could be untrained).")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    check_model()
