#!/usr/bin/env python3
"""Simple test to isolate the issue"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/src')

# Test 1: Basic tokenizer
from minigpt.tokenizer import BPETokenizer
from minigpt.config import TokenizerConfig

tokenizer = BPETokenizer(TokenizerConfig())
tokenizer.load("assets/tokenizer.model")

test_texts = [
    "Hello world",
    "What is attention?",
    "The quick brown fox",
    "Artificial intelligence"
]

print("=== Tokenizer Test ===")
for text in test_texts:
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    print(f"'{text}' -> {len(tokens)} tokens -> '{decoded}'")
    if text != decoded:
        print(f"  WARNING: Mismatch! Original != Decoded")

# Test 2: Check if tokenizer is properly trained
print("\n=== Tokenizer Stats ===")
print(f"Vocab size: {len(tokenizer.vocab)}")
print(f"Merges: {len(tokenizer.merges)}")

# Test 3: Sample tokens
sample_tokens = list(range(260, 280))  # Check some early merge tokens
print("\n=== Sample Decoding ===")
for token_id in sample_tokens:
    if token_id in tokenizer.vocab:
        token_bytes = tokenizer.vocab[token_id]
        try:
            token_str = token_bytes.decode('utf-8', errors='replace')
            print(f"Token {token_id}: '{token_str}'")
        except:
            print(f"Token {token_id}: <binary: {token_bytes}>")
