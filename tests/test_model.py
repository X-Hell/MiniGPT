import unittest
import numpy as np
from minigpt.model import MiniTransformer
from minigpt.config import ModelConfig

class TestMiniGPT(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(
            vocab_size=100,
            d_model=64,
            n_layers=2,
            n_heads=2,
            n_kv_heads=2,
            max_len=32
        )
        self.model = MiniTransformer(self.config)
        
    def test_forward_shape(self):
        B, T = 2, 10
        x = np.random.randint(0, 100, (B, T))
        logits, attn = self.model.forward(x)
        
        self.assertEqual(logits.shape, (B, T, 100))
        # Attn weights: (B, H, T, T) (with caching logic it might differ if seq passed entirely)
        # The current implementation returns last layer attn
        # Shape: (B, H, T, T)
        self.assertEqual(attn.shape, (B, 2, T, T))
        
    def test_kv_cache(self):
        # Infer one token at a time
        x = np.array([[1]])
        logits1, _ = self.model.forward(x, start_pos=0)
        
        self.assertEqual(self.model.kv_cache.current_len, 1)
        
        logits2, _ = self.model.forward(x, start_pos=1)
        self.assertEqual(self.model.kv_cache.current_len, 2)
        
if __name__ == "__main__":
    unittest.main()
