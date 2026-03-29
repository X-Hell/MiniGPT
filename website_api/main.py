import sys
import os
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict

# Add the MiniGPT src path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from minigpt.tokenizer import BPETokenizer, TokenizerConfig
from minigpt.inference import InferenceEngine
from minigpt.model import MiniTransformer

app = FastAPI(title="MiniGPT API")

# Add CORS so React frontend can call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: Optional[InferenceEngine] = None
tokenizer: Optional[BPETokenizer] = None

@app.on_event("startup")
async def startup_event():
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

class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 150
    temperature: float = 0.7
    top_k: int = 40

class ChatResponse(BaseModel):
    reply: str
    stats: Dict

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global engine
    if not engine:
        raise HTTPException(status_code=500, detail="Engine not loaded")
        
    try:
        texts, stats_list = engine.generate(
            request.message,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            num_return_sequences=1
        )
        return ChatResponse(reply=texts[0], stats=stats_list[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
