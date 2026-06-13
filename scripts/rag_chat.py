#!/usr/bin/env python3
"""Interactive RAG chat wrapper for MiniGPT."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from minigpt.config import ModelConfig, TokenizerConfig
from minigpt.inference import InferenceEngine
from minigpt.model import MiniTransformer
from minigpt.rag import Retriever, VectorStore
from minigpt.tokenizer import BPETokenizer


def load_model(checkpoint_path: str, dim: int, n_layers: int, n_heads: int, max_len: int, vocab_size: int) -> MiniTransformer:
    """Load a pickled model checkpoint or initialize a fresh model."""

    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as fh:
            payload = pickle.load(fh)

        # Handle both direct pickled model and dict checkpoint payloads.
        if isinstance(payload, MiniTransformer):
            return payload
        if isinstance(payload, dict) and "model_state" in payload:
            cfg = ModelConfig(
                vocab_size=vocab_size,
                d_model=dim,
                n_layers=n_layers,
                n_heads=n_heads,
                max_len=max_len,
            )
            model = MiniTransformer(cfg)
            state = payload["model_state"]
            for name, param in model.named_parameters():
                if name in state:
                    param[...] = state[name]
            return model

    cfg = ModelConfig(
        vocab_size=vocab_size,
        d_model=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_len=max_len,
    )
    return MiniTransformer(cfg)


def load_tokenizer(path: str, vocab_size: int) -> BPETokenizer:
    """Load BPETokenizer from file when present."""

    tok = BPETokenizer(TokenizerConfig(vocab_size=vocab_size))
    if os.path.exists(path):
        tok.load(path)
    else:
        print(f"[warn] tokenizer not found at {path}; using untrained tokenizer")
    return tok


def chunk_text(text: str, chunk_size_words: int) -> List[str]:
    """Split text into fixed-size word chunks."""

    words = text.split()
    return [" ".join(words[idx : idx + chunk_size_words]) for idx in range(0, len(words), chunk_size_words)]


def main() -> int:
    """Run interactive RAG chat."""

    parser = argparse.ArgumentParser(description="MiniGPT RAG chat")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_latest.pkl")
    parser.add_argument("--tokenizer", type=str, default="assets/tokenizer.model")
    parser.add_argument("--knowledge", type=str, default="README.md")
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=40000)
    args = parser.parse_args()

    print("=== MiniGPT RAG Chat ===")
    model = load_model(
        checkpoint_path=args.checkpoint,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_len=args.max_len,
        vocab_size=args.vocab_size,
    )
    tokenizer = load_tokenizer(args.tokenizer, vocab_size=model.config.vocab_size)

    store = VectorStore()
    retriever = Retriever(model, tokenizer, store)

    if os.path.exists(args.knowledge):
        with open(args.knowledge, "r", encoding="utf-8") as fh:
            content = fh.read()
        chunks = chunk_text(content, args.chunk_size)
        retriever.index_documents(chunks)
        print(f"Indexed {len(chunks)} chunks from {args.knowledge}")
    else:
        print(f"[warn] knowledge file missing: {args.knowledge}")

    engine = InferenceEngine(model, tokenizer)
    print("Type 'quit' to exit.")

    while True:
        try:
            user_query = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_query:
            continue
        if user_query.lower() in {"quit", "exit", "q"}:
            break

        t0 = time.time()
        result = engine.respond(user_query, retriever)
        dt = (time.time() - t0) * 1000.0

        print(result["response"])
        telemetry = result.get("telemetry", {})
        print(
            f"\n[telemetry] score={telemetry.get('retrieval_score', 0.0):.3f} "
            f"mode={telemetry.get('fallback_mode', 'GROUNDED')} "
            f"latency_ms={dt:.1f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
