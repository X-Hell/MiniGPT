
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from minigpt.inference import InferenceEngine

def test_entropy():
    print("=== Entropy Calculation Unit Test ===")
    
    # helper
    def check(name, logits):
        logits = np.array(logits)
        entropy, probs = InferenceEngine.calculate_entropy(logits)
        print(f"[{name}]")
        print(f"  Logits: {logits}")
        print(f"  Probs sum: {np.sum(probs):.20f}")
        print(f"  Entropy: {entropy:.20f}")
        
        if entropy < 0:
            print("  FAIL: Negative Entropy!")
        else:
            print("  PASS")
        print("-" * 20)

    # 1. Uniform Distribution (High Entropy)
    check("Uniform", [1.0, 1.0, 1.0, 1.0])
    
    # 2. Peaked Distribution (Low Entropy)
    check("Peaked", [10.0, 0.0, 0.0, 0.0])
    
    # 3. Extreme Peak (Near Zero Entropy)
    check("Extreme", [100.0, 0.0, 0.0, 0.0])
    
    # 4. Negative Logits
    check("Negative", [-10.0, -10.0, -10.0])
    
    # 5. Mixed Large Range
    check("Mixed", [50.0, -50.0, 0.0])

if __name__ == "__main__":
    test_entropy()
