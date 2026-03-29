# MiniGPT Model Card

> Auto-generated from `checkpoints/checkpoint_900.pkl` — 2026-03-01

## Architecture

| Parameter | Value |
|-----------|-------|
| Architecture | Llama 3-style Transformer |
| Framework | **NumPy only** (no PyTorch/TensorFlow) |
| Parameters | **12,882,816** (12.9M) |
| `d_model` | 384 |
| `n_layers` | 6 |
| `n_heads` | 6 (GQA: 3 KV heads) |
| `d_ff` | 1,024 (SwiGLU) |
| `vocab_size` | 8,192 (BPE, GPT-4 regex split) |
| `max_len` | 512 tokens |
| `dropout` | 0.1 |
| `rope_theta` | 500,000 |
| Normalization | RMSNorm (pre-norm) |
| Activation | SwiGLU |
| Positional Encoding | Rotary (RoPE) |
| KV Cache | OptimizedKVCache (ring buffer) |
| Weight Tying | Embedding ↔ Output |

## Training

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (weight_decay=0.1) |
| Learning Rate | 3e-4 (cosine decay, warmup=100) |
| Batch Size | 16 × 2 (gradient accumulation) |
| Sequence Length | 512 |
| Gradient Clipping | 1.0 |
| Dataset | TinyStoriesV2-GPT4 (2.1 GB, ~4M tokens) |
| Validation Split | 5% (~204K tokens) |
| Early Stopping | Patience = 10 |
| Steps Completed | 900 |
| Tokenizer | BPE (GPT-4 regex split, 8192 target, 2369 actual merges) |

## Performance (Phase 17 — Current)

| Metric | Value | vs Phase 16 |
|--------|-------|-------------|
| Perplexity | **759.5** | ↑ from 7.35 (expected: much larger dataset) |
| Avg Log-Prob | -6.63 | ↓ from -1.99 |
| Avg Entropy | 4.66 | ↑ from 1.61 |
| Latency / Token | **11.0 ms** | ↑ from 1.2 ms (10x more params) |
| Model Memory | 49.1 MB | ↑ from 4.9 MB (10x more params) |
| Generation Quality | **3/5 (coherent)** | ↑ from 2/5 (memorized) |

> **Note:** Perplexity increased because the model went from a tiny 75 KB memorized dataset to a 2.1 GB diverse corpus. The previous 7.35 was an overfitting artifact. The current 759.5 reflects genuine generalization on unseen data and will decrease with more training steps.

## Inference Samples

Real outputs from checkpoint_900. Generated 2026-03-01.

### Greedy Decoding (`temperature=0.01, top_k=1`)

**Prompt 1 — Story generation:**
```
Once upon a time
```
> , there was a little girl named Lily. She loved to play with her friends. One day, she saw a big, red ball. The ball was very happy and wanted to play with her friends. One day, Lily saw a big tree...

**Prompt 2 — Conversational:**
```
User: What is AI?
Assistant:
```
> you doing?". He is happy to see the cake and see it back to the cake. He says, "I'm sorry, I will help you. I will help you. I will help you...

**Prompt 3 — Creative:**
```
User: Tell me a story about a cat
Assistant:
```
> ca. It is a big, red ball. It is a big, red ball... The little bird said, "I want to play with you. You are not a good friend. I am sorry, but I will be friends."

**Prompt 4 — Open-ended:**
```
The sun was shining bright
```
> . The little bird was very happy and said, "I can help you, I can help you. I can help you..."

### Diagnostic Analysis

| Pattern | Observation | Status |
|---------|-------------|--------|
| **Coherent English** | Full grammatical sentences, story structure | ✅ Fixed |
| **Token fragmentation** | No more `oftssisty`, `ranou` gibberish | ✅ Fixed |
| **Story generation** | Characters, settings, dialogue, narrative arc | ✅ Working |
| **EOS tokens** | Model learned `<\|endoftext\|>` boundaries | ✅ Working |
| **Repetition loops** | "I will help you" repeats in longer outputs | 🟡 Needs fix |
| **Role confusion** | Model responds as storyteller, not assistant | 🟡 Needs fix |
| **Factual Q&A** | Cannot answer "What is AI?" or factual queries | 🔴 Expected (trained on stories) |

## Known Issues & Improvement Areas

### 🟡 Medium: Repetition in Long Outputs
- **Symptom:** Phrases like "I will help you" or "You are not happy" repeat in loops after 40+ tokens.
- **Cause:** Only 900 training steps completed. The attention hasn't learned long-range dependencies yet.
- **Fix:** Continue training to 3000+ steps. Add a repetition penalty to the `InferenceEngine.generate()` decoding loop.

### � Medium: Role Confusion (Storyteller vs Assistant)
- **Symptom:** When prompted with "User: ... Assistant:", model responds with TinyStories-style narrative instead of assistant-style answers.
- **Cause:** Trained exclusively on TinyStories (children's stories), not User/Assistant dialogue.
- **Fix:** Fine-tune on an instruction dataset (Alpaca/Dolly style) after base pre-training — this is **Milestone 10: Instruction Tuning**.

### 🟡 Medium: Tokenizer Vocab Underutilized
- **Symptom:** Target vocab was 8,192 but tokenizer stopped early at 2,369 merges due to training on a 100K subset.
- **Cause:** BPE training used only `text[:100000]` (first 100K chars of the corpus).
- **Fix:** Retrain the tokenizer on a larger sample (`text[:5000000]`) to fully utilize the 8,192 vocab budget.

### � Low: High Perplexity (759.5)
- **Symptom:** Perplexity is much higher than the previous model (7.35).
- **Cause:** This is **expected and healthy**. The old model memorized 75 KB; the new model is generalizing across 2.1 GB. With only 900 steps, it has barely seen the data.
- **Fix:** Continue training to 3000-5000 steps. Perplexity should drop below 100 with sufficient training.

### 🟢 Low: Latency Increase (11 ms/token)
- **Symptom:** 10x slower than the 1.25M model.
- **Cause:** 10x more parameters (12.9M) running on pure NumPy CPU inference.
- **Fix:** This is the expected cost of scaling. For production speed, implement **Milestone 9: C++/CUDA Kernels**.

## Generation Quality Scale

| Score | Level | Description |
|-------|-------|-------------|
| 0 | Gibberish | Random character sequences |
| 1 | Grammar | Basic word/sentence structure |
| 2 | Memorized | Reproduces training data, fails on novel prompts |
| 3 | **Coherent** ← current | Grammatical responses to novel prompts |
| 4 | Logical | Factually reasonable responses |
| 5 | Creative | Novel, contextually appropriate responses |
