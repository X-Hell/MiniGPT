
import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from minigpt.config import ModelConfig, TokenizerConfig
from minigpt.model import MiniTransformer
from minigpt.tokenizer import BPETokenizer
from minigpt.inference import InferenceEngine
import pickle

def demo():
    print("=== Generation Demo ===")
    
    # Load Model
    ckpt = "checkpoints/model_latest.pkl"
    if not os.path.exists(ckpt):
        print("Checkpoint not found!")
        return

    print(f"Loading {ckpt}...")
    with open(ckpt, "rb") as f:
        model = pickle.load(f)

    # Load Tokenizer
    tok_path = "assets/tokenizer.model"
    tok_config = TokenizerConfig(vocab_size=model.config.vocab_size)
    tokenizer = BPETokenizer(tok_config)
    tokenizer.load(tok_path)

    engine = InferenceEngine(model, tokenizer)

    # Setup
    prompt_input = "User: What is the capital of France?\nAssistant:"
    temperature = 0.7
    
    print("\n" + "="*30)
    print(f"INPUT PROMPT (passed to model):")
    print(f"'{prompt_input}'")
    print(f"PARAMETER: Temperature = {temperature}")
    print("="*30 + "\n")

    print("Generating...")
    texts, stats = engine.generate(
        prompt_input, 
        max_tokens=64, 
        temperature=temperature, 
        top_k=40,
        num_return_sequences=1
    )
    
    print("\n" + "="*30)
    print(f"OUTPUT:")
    print(texts[0])
    print("="*30)
    
    print(f"\nStats (First Token): {stats[0][0] if stats[0] else 'N/A'}")

if __name__ == "__main__":
    demo()
