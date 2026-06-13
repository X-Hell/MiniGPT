#!/usr/bin/env python3
"""Train a BPE tokenizer on the LOCAL TinyStories corpus and tokenize it to a
1-D .npy token stream that scripts/train.py consumes. CPU-only, no internet.

Produces:
  assets/tokenizer_tinystories_16k.json   (HF tokenizers JSON, vocab 16384)
  data/tinystories_train_16k.npy          (uint16 token stream, <eos> at doc bounds)
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from minigpt.config import TokenizerConfig
from minigpt.tokenizer import HFBPETokenizer

SRC = "data/TinyStoriesV2-GPT4-train.txt"
SEP = "<|endoftext|>"


def doc_stream(path, sep, max_chars, chunk=8_000_000):
    """Yield one document per <|endoftext|> boundary, streaming the file in
    bounded-memory chunks, stopping once max_chars have been read."""
    seen = 0
    buf = ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while max_chars is None or seen < max_chars:
            data = f.read(chunk)
            if not data:
                break
            seen += len(data)
            buf += data
            parts = buf.split(sep)
            buf = parts.pop()          # last (possibly partial) doc carries over
            for d in parts:
                d = d.strip()
                if d:
                    yield d
    if buf.strip():
        yield buf.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, default=16384)
    ap.add_argument("--train_chars", type=int, default=500_000_000)      # 500MB to learn merges
    ap.add_argument("--tokenize_chars", type=int, default=1_200_000_000) # 1.2GB -> ~280M tokens
    ap.add_argument("--tok_out", default="assets/tokenizer_tinystories_16k.json")
    ap.add_argument("--data_out", default="data/tinystories_train_16k.npy")
    ap.add_argument("--batch", type=int, default=8192)
    args = ap.parse_args()

    t0 = time.time()

    # ---- 1. Train tokenizer on a subset (BPE merges converge fast) ----------
    print(f"[1/2] Training BPE vocab={args.vocab_size} on up to {args.train_chars/1e6:.0f}MB ...", flush=True)
    cfg = TokenizerConfig(vocab_size=args.vocab_size, min_frequency=2,
                          special_tokens=("<pad>", "<eos>", "<unk>"))
    tok = HFBPETokenizer(cfg)
    tok.train_from_iterator(doc_stream(SRC, SEP, args.train_chars), min_frequency=2)
    os.makedirs(os.path.dirname(args.tok_out), exist_ok=True)
    tok.save(args.tok_out)
    assert tok.eos_id is not None, "eos token missing"
    print(f"    saved {args.tok_out}  pad={tok.pad_id} eos={tok.eos_id} unk={tok.unk_id}"
          f"  ({time.time()-t0:.0f}s)", flush=True)

    # ---- 2. Tokenize corpus with batched Rust encoder (fast) ----------------
    print(f"[2/2] Tokenizing up to {args.tokenize_chars/1e6:.0f}MB ...", flush=True)
    eos = tok.eos_id
    raw = tok._tk
    encode_batch = getattr(raw, "encode_batch_fast", None) or raw.encode_batch
    chunks = []          # one uint16 array per batch (keeps object count low)
    ntok = [0]
    ndoc = 0

    def flush(batch):
        encs = encode_batch(batch)
        acc = []
        for e in encs:
            acc.extend(e.ids)
            acc.append(eos)
        a = np.asarray(acc, dtype=np.uint16)
        chunks.append(a)
        ntok[0] += a.size

    batch = []
    for d in doc_stream(SRC, SEP, args.tokenize_chars):
        batch.append(d)
        ndoc += 1
        if len(batch) >= args.batch:
            flush(batch)
            batch = []
            if ndoc % (args.batch * 20) == 0:
                print(f"    {ndoc:,} docs  {ntok[0]:,} tokens  {time.time()-t0:.0f}s", flush=True)
    if batch:
        flush(batch)

    arr = np.concatenate(chunks)
    mx = int(arr.max())
    assert mx < args.vocab_size, f"max token id {mx} >= vocab {args.vocab_size}"
    os.makedirs(os.path.dirname(args.data_out), exist_ok=True)
    np.save(args.data_out, arr)
    print(f"DONE  {args.data_out}  tokens={arr.size:,}  dtype={arr.dtype}  max_id={mx}  "
          f"docs={ndoc:,}  total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
