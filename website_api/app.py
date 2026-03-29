import sys
import os
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Optional

# Add the MiniGPT src path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from minigpt.tokenizer import BPETokenizer, TokenizerConfig
from minigpt.inference import InferenceEngine
from minigpt.model import MiniTransformer
import numpy as np

app = Flask(__name__)
# Enable CORS for all domains so React can talk to it
CORS(app)

def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj

engine: Optional[InferenceEngine] = None
tokenizer: Optional[BPETokenizer] = None

def init_model():
    global engine, tokenizer
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Canonical checkpoint location
    checkpoint_paths = [
        "checkpoints/model_latest.pkl",
    ]
    
    checkpoint_file = None
    for path in checkpoint_paths:
        full_path = os.path.join(base_dir, path)
        if os.path.exists(full_path):
            checkpoint_file = full_path
            break
            
    tokenizer_file = os.path.join(base_dir, "assets/tokenizer.model")
    
    # Load Model
    if checkpoint_file:
        print(f"Loading checkpoint: {checkpoint_file}")
        with open(checkpoint_file, "rb") as f:
            model = pickle.load(f)
        # Reset KV cache for inference (checkpoint may have training batch_size)
        model.kv_cache.reset()
        print("Model loaded.")
    else:
        print("No checkpoint found. Initializing random model.")
        from minigpt.config import ModelConfig
        config = ModelConfig()
        model = MiniTransformer(config)

    # Load Tokenizer
    print(f"Loading tokenizer: {tokenizer_file}")
    tok_config = TokenizerConfig(vocab_size=model.config.vocab_size)
    tokenizer = BPETokenizer(tok_config)
    if os.path.exists(tokenizer_file):
        tokenizer.load(tokenizer_file)
    else:
        print("Warning: Tokenizer not found.")
        
    engine = InferenceEngine(model, tokenizer)
    print("Inference Engine ready.")

# Initialize the model on startup
init_model()

@app.route("/chat", methods=["POST"])
def chat():
    global engine, tokenizer
    if not engine:
        return jsonify({"error": "Engine not loaded"}), 500
        
    data = request.json
    user_message = data.get("message", "").strip()
    max_tokens = data.get("max_tokens", 100)
    temperature = data.get("temperature", 0.01) # Force greedy for overfitted test
    top_k = data.get("top_k", 1)               # Force greedy for overfitted test
    
    # Format prompt to match training data format (train_clean.txt uses User/Assistant pairs)
    formatted_prompt = f"User: {user_message}\nAssistant:"
    
    # Suppress EOS token (id=256) to prevent immediate termination
    eos_id = getattr(tokenizer, 'eos_id', 256)
    logit_bias = {eos_id: -100.0}
    
    # Use only newline-prefixed stop sequences matching the dataset format
    stop_sequences = ["\nUser:", "\n\n"]
        
    try:
        # Cap max_tokens to prevent KV cache overflow
        prompt_tokens = tokenizer.encode(formatted_prompt)
        available = engine.model.config.max_len - len(prompt_tokens) - 2  # safety margin
        max_tokens = min(max_tokens, max(available, 10))
        
        texts, stats_list = engine.generate(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            logit_bias=logit_bias,
            stop_sequences=stop_sequences,
            repetition_penalty=1.0,
            num_return_sequences=1
        )
        reply = texts[0].strip()
        if not reply:
            reply = "I'm still learning! Try asking me about neural networks, AI, or how language models work."
        safe_stats = convert_to_serializable(stats_list[0])
        return jsonify({"reply": reply, "stats": safe_stats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(port=8000, debug=False)
