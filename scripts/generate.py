import sys
import os
import argparse
import pickle
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from minigpt.tokenizer import BPETokenizer, TokenizerConfig
from minigpt.inference import InferenceEngine
from minigpt.model import MiniTransformer

def main():
    parser = argparse.ArgumentParser(description="MiniGPT Text Generation")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text (if not set, enters interactive mode)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_latest.pkl", help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="assets/tokenizer.model", help="Path to tokenizer model")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0=deterministic)")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling (1=greedy)")
    args = parser.parse_args()
    
    # Load Model
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        with open(args.checkpoint, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully!")
    else:
        print(f"Warning: No checkpoint found at {args.checkpoint}")
        print("Initializing model with random weights (expect gibberish).")
        from minigpt.config import ModelConfig
        config = ModelConfig()
        model = MiniTransformer(config)
        
    print(f"Loading tokenizer: {args.tokenizer}")
    tok_config = TokenizerConfig(vocab_size=model.config.vocab_size)
    tokenizer = BPETokenizer(tok_config)
    if os.path.exists(args.tokenizer):
        tokenizer.load(args.tokenizer)
    else:
        print("Warning: Tokenizer not found. Output will be garbage.")
    
    engine = InferenceEngine(model, tokenizer)
    
    def stream_callback(token, *args, **kwargs):
        sys.stdout.write(token)
        sys.stdout.flush()
    
    if args.prompt:
        # Single prompt mode
        print(f"\nPrompt: {args.prompt}\n" + "="*50)
        texts, stats_list = engine.generate(
            args.prompt, 
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            stream=True, 
            callback=stream_callback,
            num_return_sequences=1
        )
        output = texts[0]
        stats = stats_list[0]
        print("\n" + "="*50 + "\nDone.")
    else:
        # Interactive mode
        print("\n=== MiniGPT Interactive Mode ===")
        print(f"Settings: temp={args.temperature}, top_k={args.top_k}, max_tokens={args.max_tokens}")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                prompt = input("> ")
            except (EOFError, KeyboardInterrupt):
                break
                
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            if not prompt.strip():
                continue
                
            texts, stats_list = engine.generate(
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                stream=True,
                callback=stream_callback,
                num_return_sequences=1
            )
            output = texts[0]
            stats = stats_list[0]
            print("\n")
        
        print("Goodbye!")

if __name__ == "__main__":
    main()
