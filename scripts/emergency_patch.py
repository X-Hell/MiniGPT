
import sys
import os
import pickle
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from minigpt.config import ModelConfig
from minigpt.model import MiniTransformer

def emergency_patch():
    print("=== Emergency Model Patch ===")
    
    # 1. Create Fresh Configuration (Nano)
    print("Initializing fresh Nano model with Xavier Init...")
    # Minimal config for stability
    config = ModelConfig(
        vocab_size=4096,
        d_model=64,      # Tiny for testing
        n_layers=1,      # Single layer
        n_heads=2,       # Minimal heads
        n_kv_heads=1,
        max_len=256,
        dropout=0.0
    )
    
    # 2. Initialize Model
    model = MiniTransformer(config)
    print(f"Model initialized. Params: {sum(p.size for p in model.parameters()) if hasattr(model, 'parameters') else 'N/A'}")
    
    # CRITICAL: Xavier/Glorot initialization with proper scaling
    print("Applying Xavier Initialization...")
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.shape) >= 2:
                # Xavier uniform
                gain = 1.0
                if 'attention' in name or 'ffn' in name:
                    gain = 0.67
                std = gain * np.sqrt(2.0 / (param.shape[0] + param.shape[1]))
                param.data = np.random.uniform(-std, std, param.shape).astype(np.float32)
    
    # 3. Save as patched model
    save_path = "checkpoints/final_model_patched.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
        
    print(f"Saved fresh model to: {save_path}")
    print("\n[ACTION REQUIRED]")
    print(f"Please update scripts/rag_chat.py to load '{save_path}' instead of the old checkpoint.")

if __name__ == "__main__":
    emergency_patch()
