import os
import sys
import pickle
import numpy as np
import argparse

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from minigpt.config import ModelConfig, TokenizerConfig
from minigpt.model import MiniTransformer, precompute_freqs_cis
from minigpt.tokenizer import BPETokenizer
from minigpt.inference import InferenceEngine
from minigpt.rag import VectorStore, Retriever

def main():
    parser = argparse.ArgumentParser(description="RAG Chat with Confidence Awareness")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final_model_patched.pkl", help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="assets/tokenizer.model", help="Path to tokenizer")
    parser.add_argument("--knowledge", type=str, default="README.md", help="Text file to index")
    parser.add_argument("--chunk_size", type=int, default=128, help="Chunk size for indexing")
    # Model Config Args (Match Nano)
    parser.add_argument("--dim", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--max_len", type=int, default=2048, help="Inference max length (RoPE Extrapolation)")
    args = parser.parse_args()

    print("=== MiniGPT RAG Chat ===")
    
    # 1. Load Model & Tokenizer
    print("Loading model...")
    if args.checkpoint and os.path.exists(args.checkpoint):
        with open(args.checkpoint, "rb") as f:
            model = pickle.load(f)
            
        # PATCH: Extend Context Window for RAG
        # The trained model might have max_len=128, but RAG prompts are long.
        # We recompute RoPE frequencies for 2048.
        if model.config.max_len < args.max_len:
            print(f"Patching model context window: {model.config.max_len} -> {args.max_len} (with PI)")
            # Linear Position Interpolation (PI)
            # We scale the positions effectively by compressing the input indices
            # scale = new_len / trained_len
            scale = args.max_len / model.config.max_len
            model.config.max_len = args.max_len
            
            # Recompute freqs with 'theta' scaled? 
            # Standard PI: freqs = 1 / (theta ** (2i/d)) * (pos / scale)
            # So freqs_cis(pos) -> freqs_cis(pos / scale)
            # In precompute_freqs_cis, t = np.arange(end).
            # We just hack t to be t / scale.
            
            # Using precompute_freqs_cis directly won't work unless we modify it to accept a scale factor
            # OR we pass a modified "end" and then interpolate?
            # Easiest way: Write a small inline helper or re-implement manually here.
            
            dim = model.config.d_model // model.config.n_heads
            end = args.max_len
            theta = model.config.rope_theta
            
            freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(np.float32) / dim))
            t = np.arange(end, dtype=np.float32) / scale # <--- PI Scale
            freqs = np.outer(t, freqs)
            new_freqs = np.exp(1j * freqs)

            for layer in model.layers:
                layer.attn.freqs_cis = new_freqs
                
            # CRITICAL: Re-initialize KV Cache with new size
            print(f"Re-allocating KV Cache for max_len={args.max_len}...")
            # We import OptimizedKVCache from model module (it is aliased as KVCache)
            # Or simplified: use the class attached to the object if possible?
            # Safer to verify type or import.
            # model.kv_cache is an instance. type(model.kv_cache) gives the class.
            KVCacheClass = type(model.kv_cache) 
            model.kv_cache = KVCacheClass(
                max_len=args.max_len,
                n_heads=model.config.n_heads,
                d_head=dim,  # 'dim' here is already d_model // n_heads (see line 58)
                n_layers=model.config.n_layers,
                n_kv_heads=model.config.n_kv_heads
            )
    else:
        print("No checkpoint found. Using random weights (Expect gibberish, but logic works).")
        config = ModelConfig(
            vocab_size=4096,
            d_model=args.dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_kv_heads=max(1, args.n_heads // 2),
            max_len=args.max_len
        )
        model = MiniTransformer(config)
        
    print(f"Loading tokenizer from {args.tokenizer}...")
    tok_config = TokenizerConfig(vocab_size=model.config.vocab_size)
    tokenizer = BPETokenizer(tok_config)
    if os.path.exists(args.tokenizer):
        tokenizer.load(args.tokenizer)
    else:
        print("Warning: Tokenizer not found/trained.")

    # 2. Initialize RAG
    vector_store = VectorStore()
    retriever = Retriever(model, tokenizer, vector_store)
    
    # 3. Index Knowledge Base
    if os.path.exists(args.knowledge):
        print(f"Indexing {args.knowledge}...")
        with open(args.knowledge, "r") as f:
            text = f.read()
            
        # Simple chunking
        words = text.split()
        chunks = []
        for i in range(0, len(words), args.chunk_size):
            chunk = " ".join(words[i:i+args.chunk_size])
            chunks.append(chunk)
            
        retriever.index_documents(chunks)
    else:
        print(f"Knowledge file {args.knowledge} not found. RAG will be empty.")
        
    engine = InferenceEngine(model, tokenizer)
    
    print("\nSystem Ready. Type 'quit' to exit.")
    print("Confidence Thresholds: High (>0.8), Med (>0.5), Low (<0.5)")
    
    import time
    
    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ["quit", "exit"]:
                break
                
            start_req = time.time()
            
            # Using the High-Level Pipeline
            print("    [Gen] ", end="", flush=True)
            
            # Note: respond() calls generate() internally. 
            # To get streaming during respond(), we would need to pass the callback to respond() -> generate()
            # But the detailed requirements didn't enforce streaming visualization for the pipeline test, 
            # just "Visualise these after each session". 
            # However, to be nice, let's keep streaming if possible. 
            # Implemented: Current `respond` does NOT take a callback. 
            # Decision: We will rely on printing the final response, but we can update InferenceEngine later to stream.
            # For now, we wait for full generation (or safety stop).
            
            result = engine.respond(user_input, retriever)
            
            req_time = time.time() - start_req
            
            response_text = result["response"]
            telemetry = result["telemetry"]
            
            # Print response
            print(f"{response_text}")
            
            # Telemetry Display
            print(f"\n    {'='*15} Advanced RAG Telemetry {'='*15}")
            print(f"    Query Refinement: {'TRIGGERED' if telemetry['refinement_triggered'] else 'No'}")
            print(f"    Retrieval Score : {telemetry['retrieval_score']:.4f}")
            print(f"    Query Class     : {telemetry.get('query_class', 'N/A')}")
            print(f"    Grounded Prompt : {'YES' if telemetry['grounded'] else 'No'}")
            
            if telemetry['fallback_triggered']:
                 print(f"    Fallback Mode   : {telemetry.get('fallback_mode', 'N/A')}")
                 if telemetry.get('generation_aborted'):
                     print(f"    Gen Aborted     : YES (Unsafe)")
            
            print(f"    Avg Log-Prob    : {telemetry.get('avg_logprob', 0.0):.4f}")
            print(f"    Avg Entropy     : {telemetry.get('avg_entropy', 0.0):.4f} nats")
            print(f"    Total Latency   : {req_time*1000:.2f} ms")
            print(f"    {'='*56}\n")

                
        except KeyboardInterrupt:
            break
            
if __name__ == "__main__":
    main()
