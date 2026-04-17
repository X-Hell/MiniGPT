# MiniGPT Project Context

**Last updated:** 2026-04-17
**Current milestone:** M1 — Architecture Retrofit (🔄 in progress)
**Overall goal:** Replicate GPT-1 (117M params) faithfully in pure NumPy/CuPy on RTX 3060 12 GB.

---

## Project Snapshot

MiniGPT is a hand-written decoder-only Transformer implemented with a thin `xp` abstraction over NumPy (CPU) and CuPy (GPU) — no PyTorch, no TensorFlow, no autograd. The backward pass is written out explicitly. As of the last committed state (commit `a4ef455`), it trained a 12.9M-parameter Llama-3-style model (RMSNorm + SwiGLU + RoPE + GQA) on FineWeb-Edu with FP16 Tensor-Core matmuls and fused CuPy kernels, reaching roughly PPL 15 after 50K steps on an RTX 3060 12 GB. The pivot, now in progress in the working tree, is to replace every modern-LLM component with its historically-accurate GPT-1 (2018) equivalent and scale up to 117M parameters. `src/minigpt/` holds the library (model, optimizer, tokenizer, backend, inference, KV cache); `scripts/` holds the training loop, gradient validator, GPU benchmark, and FineWeb streaming pipeline. Training data is FineWeb-Edu shards (`.npy` uint16/uint32), tokenized by an in-repo BPE.

---

## Architecture: Current vs Target

| Component | Current State (working tree, 2026-04-17) | GPT-1 Target |
|---|---|---|
| Parameters | ~14.2M *(computed from `config.py` defaults d=384, L=6, H=6, V=4096, d_ff=1024, max_len=512, with new LN betas and biases)* | 117M |
| Layers (`n_layers`) | 6 *(config.py default)* | 12 |
| `d_model` | 384 *(config.py default)* | 768 |
| `n_heads` | 6 *(config.py default)* | 12 |
| `d_ff` | 1024 *(auto: 256-aligned `2*4*d/3` from the SwiGLU era — now ignored in model.py, which uses `config.d_ff` as-is for the GELU FFN; config should be updated to `4*d_model`)* | 3072 |
| Normalization | **LayerNorm (gamma + beta), post-norm, eps=1e-5** — GPT-1 faithful (uncommitted refactor in `model.py`) | LayerNorm, post-norm |
| Activation | **GELU (tanh approximation), fused forward + backward** — GPT-1 faithful (uncommitted refactor) | GELU (tanh approx) |
| Positional Encoding | **Learned absolute `W_pos` of shape `(max_len, d_model)`, init N(0, 0.01)** — GPT-1 faithful (uncommitted refactor, no RoPE in `model.py`) | Learned absolute |
| Block order | **Post-norm: `LN(x + Attn(x))`, then `LN(x + FFN(x))`** — GPT-1 faithful | Post-norm |
| Attention | **Full MHA, no GQA, no RoPE, with QKV+output biases** — GPT-1 faithful | Full MHA |
| Output projection | **Tied to `W_emb.T`, no final LayerNorm** — GPT-1 faithful | Tied, no final LN |
| Optimizer | **Adam + coupled L2 weight decay, `lr=2.5e-4`, `betas=(0.9, 0.98)`, `eps=1e-8`, `wd=0.01`** — GPT-1 faithful (uncommitted refactor in `optimizer.py`) | Adam + L2 |
| WD exclusion | LN gamma/beta ✅, all biases ✅ (via `p.ndim >= 2` gate) — `W_pos` is 2D and **currently receives decay** ⚠️ | LN gamma/beta, biases, `W_pos` all excluded |
| Dropout | p=0.1 wired at 4 locations in `model.py` (embed, MLP intermediate, attn-branch, ffn-branch), but **`ModelConfig.dropout` defaults to 0.0** | p=0.1 enabled |
| Vocab Size | 4096 BPE *(config.py default; FineWeb runs have used 8192)* | 40,000 BPE |
| Special tokens | `<EOS>` only (id 256) — **no `<pad>`, no `<unk>`** | `<pad>`, `<eos>`, `<unk>` |
| Context Length (`max_len`) | 512 *(config.py default)* | 512 |
| Precision | FP16 forward matmuls (Tensor Cores) + FP32 backward/optimizer — GPT-1 faithful for training stability | Same |

---

## File Map

| File | Contents | Owning Agent |
|---|---|---|
| `src/minigpt/backend.py` | `xp` CuPy/NumPy abstraction; `fuse` decorator (cupy.fuse or no-op); `fp16_matmul`; `scatter_add`; `to_cpu`/`to_device`; VRAM utilities (`vram_stats`, `log_vram`, `estimate_model_vram`). | BackendAgent |
| `src/minigpt/config.py` | Dataclasses: `TokenizerConfig`, `ModelConfig`, `TrainConfig`. **Still contains `rope_theta` and `n_kv_heads` legacy fields and 384/6/6/4096 defaults — pending update for 117M.** | ArchitectAgent (ModelConfig), OptimizerAgent (TrainConfig) |
| `src/minigpt/model.py` | GPT-1 Transformer: `LayerNorm` (gamma+beta), `FeedForward` (Linear→GELU→Dropout→Linear), `MultiHeadAttention` (full MHA, no GQA/RoPE), `TransformerBlock` (post-norm with cached inputs for backward), `EmbeddingLayer` (token + learned pos), `MiniTransformer` (tied output, no final LN). Fused `_fused_gelu`, `_fused_gelu_backward`, `_fused_exp`. | ArchitectAgent |
| `src/minigpt/optimizer.py` | `Adam` (coupled L2 weight decay) with `p.ndim >= 2` decay gate, `lr=2.5e-4`, `betas=(0.9, 0.98)`, `eps=1e-8`. `AdamW = Adam` alias kept for backward compat. | OptimizerAgent |
| `src/minigpt/tokenizer.py` | `BPETokenizer` with GPT-4 regex split, byte-level base vocab, `<EOS>` at id 256, merges from id 257, `save`/`load` via pickle. No `<pad>`/`<unk>` yet. | TokenizerAgent |
| `src/minigpt/inference.py` | `InferenceEngine`: generation with temperature / top-k / repetition penalty / logit bias / stopping criteria, batched `num_return_sequences`, entropy/confidence telemetry, RAG-aware `respond()` pipeline with query classification (`UNSAFE`/`OPINION`/`GENERAL`/`GROUNDED`). | *(no dedicated agent — revisit after M2)* |
| `src/minigpt/optimized_kv_cache.py` | Ring-buffer KV cache used during autoregressive inference (bypassed during training). | BackendAgent (allocation) + ArchitectAgent (shape semantics) |
| `src/minigpt/trainer.py` | Legacy training-loop utility. **Stale — still uses the v1 backward 3-tuple API per ROADMAP_72H.md §10. Does not know about the new `(dW_emb_out, dW_pos, layer_grads, dX_emb)` shape, new LN betas, new biases, or dropout masks. Needs a rewrite before M3.** | ArchitectAgent + OptimizerAgent |
| `src/minigpt/rag.py` | Retrieval-Augmented Generation module used by `inference.py`. Not in the GPT-1 refactor scope. | *(out of scope for M1–M4)* |
| `src/minigpt/__init__.py` | Empty init file. | — |

---

## Milestone Log

### ✅ Milestone 0: Production Foundation (Completed 2026-03-29, commit `a4ef455`)

**What was achieved:**
- FP16 mixed precision via `fp16_matmul` — activates Tensor Cores on RTX 3060 for all 9 forward matmuls, backward stays FP32 for gradient stability.
- Fused element-wise CUDA kernels via `cupy.fuse` — `_fused_silu`, `_fused_silu_backward`, `_fused_exp` cut ~36 kernel launches/step at 6 layers.
- FineWeb-Edu streaming pipeline (`scripts/prepare_fineweb.py`) — stream → tokenize → shard end-to-end tested.
- Numerical gradient validator (`scripts/validate_gradients.py`) — central finite differences, float64 loss, median error <10%, p90 <50%.
- KV-cache bypass during training (~10× forward pass speedup), VRAM monitoring, CPU-safe checkpoint portability, weight-decay exclusion for 1D params, LN gammas in grad clip, single (not double) backward call.

**Key metrics:** ~10× training speedup vs. CPU baseline; 2–4× GPU throughput from FP16; ~6.8 GB VRAM used at `d=384, L=6, T=512, B=128` on RTX 3060 12 GB; gradient check passes at float32 forward / float64 loss.

**Frozen — do not modify this section.**

---

### 🔄 Milestone 1: Architecture Retrofit (In Progress)

**Goal:** Replace all non-GPT-1 architectural components with GPT-1-faithful equivalents.

**Checklist:**
- [x] RMSNorm removed, standard `LayerNorm` (gamma + beta, eps=1e-5) implemented with compact Jacobian-vector-product backward
- [x] Transformer block reordered to post-norm (`LN(x + Attn(x))`, `LN(x + FFN(x))`)
- [x] SwiGLU replaced with GELU tanh-approximation, with fused forward and backward kernels
- [x] RoPE removed, learned absolute positional embeddings `W_pos` added (init N(0, 0.01))
- [x] Full Multi-Head Attention restored (no GQA, no head-scale vector)
- [x] Input/output embedding tying implemented: `logits = x_final @ W_emb.T`; `MiniTransformer.backward` returns `dW_emb_out` and `dX_emb` for the trainer to `scatter_add`
- [x] Final LayerNorm removed (GPT-1 canonical)
- [x] Dropout wired at 4 locations (embedding output, MLP intermediate, attention branch, FFN branch) with cached masks for backward
- [x] Biases added to all linears (QKV, output projection, FFN fc / proj)
- [ ] `scripts/train.py` updated to unpack the new 4-tuple `(dW_emb_out, dW_pos, layer_grads, dX_emb)` and new per-block grad shapes (ffn/attn both 4-tuples, LN a 2-tuple with gamma+beta)
- [ ] `src/minigpt/trainer.py` updated (still on v1 3-tuple API per ROADMAP §10)
- [ ] Gradient numerical check re-baselined on new modules (ArchitectAgent validation protocol step 3)
- [ ] Commit `model.py` refactor (currently uncommitted in working tree)

**Files being modified:** `src/minigpt/model.py`, `scripts/train.py`, `src/minigpt/trainer.py`
**Agent responsible:** ArchitectAgent
**Blocked by:** Nothing — this is first
**Unblocks:** Milestone 2 (OptimizerAgent needs the final parameter list, including LN betas and biases, to verify weight-decay exclusion)

---

### 🔄 Milestone 2: Optimizer & Regularization (In Progress, partial)

**Goal:** Replace AdamW with GPT-1-faithful Adam + L2, add dropout, implement LR schedule.

**Checklist:**
- [x] `AdamW` (decoupled) replaced with `Adam` (coupled L2: `g += wd * p` folded into gradient before moment updates)
- [x] Defaults set to GPT-1: `lr=2.5e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01`
- [x] LN gamma / LN beta / biases excluded from weight decay via `p.ndim >= 2` gate
- [x] Dropout plumbing present in `model.py` at 4 locations (owned by ArchitectAgent, validated here)
- [ ] `ModelConfig.dropout` default changed from 0.0 → 0.1 (currently still 0.0 in `config.py`)
- [ ] `W_pos` excluded from weight decay (it's 2D, so the `ndim >= 2` gate does not exclude it — needs an explicit name-based skip)
- [ ] Linear warmup (0 → 2.5e-4 over 2000 steps) + cosine decay to 1e-5 implemented and verified in `scripts/train.py`
- [ ] Gradient clipping confirmed at global norm 1.0 across **all** parameters including new LN betas and all biases (old clip set per ROADMAP §6 had only gammas; needs update)
- [ ] Commit `optimizer.py` refactor

**Files being modified:** `src/minigpt/optimizer.py`, `src/minigpt/config.py` (dropout default, TrainConfig betas/lr), `scripts/train.py` (LR schedule + expanded clip set)
**Agent responsible:** OptimizerAgent (with ArchitectAgent for dropout plumbing already done)
**Blocked by:** Milestone 1 final param list (so the clip set and no-decay list are complete)

---

### ⏳ Milestone 3: Tokenizer & Embedding (Pending)

**Goal:** Upgrade to 40K BPE tokenizer with full special-token set, bundle tokenizer into checkpoints.

**Checklist:**
- [ ] BPE tokenizer trained on FineWeb-Edu sample with `vocab_size=40000` (current defaults: 4096 in config, 8192 in recent runs)
- [ ] Special tokens added: `<pad>`, `<eos>`, `<unk>` (currently only `<EOS>` at id 256)
- [ ] Regex / pre-tokenizer updated so special-token literals encode atomically (not byte-by-byte)
- [ ] Data pipeline (`scripts/prepare_fineweb.py`) updated to emit `<eos>` at every document boundary
- [ ] `ModelConfig.vocab_size` default updated to 40000; embedding matrix and tied output projection verified to share the same matrix exactly (no duplication in `named_parameters()`)
- [ ] Checkpoint format updated to bundle tokenizer vocab + merges inline (see CheckpointAgent invariants)
- [ ] Config-mismatch guard: loading a 4K/8K checkpoint into a 40K model raises a clear `ValueError` rather than broadcasting
- [ ] FineWeb shards re-encoded with the new tokenizer, saved alongside a versioned `tokenizer_v3_40k.model`

**Files being modified:** `src/minigpt/tokenizer.py`, `src/minigpt/model.py` (tying is already there — this is a verification-only touch), `scripts/train.py` (checkpoint format), `scripts/prepare_fineweb.py` (doc-boundary `<eos>` and vocab size)
**Agent responsible:** TokenizerAgent + ArchitectAgent (embedding tying) + CheckpointAgent (bundling)
**Blocked by:** Milestone 1 (tied embedding path must be stable before a 40K matrix is trained)

---

### ⏳ Milestone 4: Scale to GPT-1 Dimensions (Pending)

**Goal:** Scale to `d_model=768, n_layers=12, n_heads=12, d_ff=3072, max_len=512` (~117M parameters) within 12 GB VRAM.

**Checklist:**
- [ ] `ModelConfig` defaults updated (and `n_kv_heads`, `rope_theta` legacy fields removed or defaulted to match `n_heads` / ignored)
- [ ] `ModelConfig.d_ff` default set explicitly to `4 * d_model = 3072` (auto-formula in `__post_init__` is currently the SwiGLU `2*4*d/3` ratio — replace)
- [ ] VRAM accounting rebuilt: `estimate_model_vram()` `d_ff` formula updated to `4*d_model`, biases and LN betas added to param count, FP16 activation paths re-accounted. Target: measured peak ≤ 11.0 GB at the chosen `batch_size × accum_steps`.
- [ ] `batch_size` and `accum_steps` chosen so effective batch = 64 at T=512 (candidate: `batch_size=8, accum_steps=8`; verify empirically)
- [ ] Smoke test: 10 training steps at full 117M scale, no OOM, loss decreases, grad norm in `[0.1, 5.0]` after warmup
- [ ] Step-0 loss ≈ `ln(40000) ≈ 10.60` within 0.1 nats (sanity check on random init)
- [ ] Activation checkpointing prototyped as fallback in case 12 GB is too tight (ArchitectAgent + BackendAgent co-lead)
- [ ] Projected total training wall-clock documented in the 72-hour plan update

**Files being modified:** `src/minigpt/config.py`, `src/minigpt/backend.py` (VRAM estimator), `scripts/train.py` (batch/accum defaults), possibly `scripts/benchmark_gpu.py` (new probe configs)
**Agent responsible:** BackendAgent + EvalAgent (VRAM verification) + ArchitectAgent (config defaults) + CheckpointAgent (117M ckpts are ~500 MB — confirm disk budget)
**Blocked by:** Milestone 3 (needs 40K vocab before scaling, otherwise `W_emb` dominates VRAM accounting incorrectly)

---

## Active Decisions & Tradeoffs

| Decision | Rationale | Revisit if... |
|---|---|---|
| FP16 forward / FP32 backward (not full mixed precision with loss scaling) | Tensor Core activation on RTX 3060 Ampere without the underflow problems that force loss scaling. Manual backward needs FP32 precision. | Moving to A100/H100 with BF16 support — BF16 has FP32 dynamic range and eliminates the split. |
| `cupy.fuse` instead of raw `RawKernel` CUDA | Decorator-based, no kernel code to maintain, automatic CPU fallback (no-op). Correct scope: element-wise ops only, not matmul (cuBLAS already handles matmul). | We need operation patterns `fuse` can't express (e.g., warp-level reductions, Flash Attention). |
| FineWeb-Edu over BooksCorpus | Larger (10+ GB), higher quality filter, streamable from HuggingFace without a one-time download. | We want strict GPT-1 historical replication — BooksCorpus is the original corpus. |
| Pure NumPy/CuPy (no PyTorch) | From-scratch learning objective — everything explicit, every gradient hand-derived. | Hitting performance ceiling that only a compiled-graph autograd framework can break (Flash Attention, fused multi-head kernels). |
| 40K vocab target | Matches original GPT-1. Also reduces subword fragmentation vs. 4K/8K runs, which helps at 117M scale. | VRAM too tight at 117M — `W_emb` alone is `40000 × 768 × 4 bytes = 117 MB` FP32, doubled by tied-output accumulation. |
| Coupled Adam + L2 (not AdamW) | GPT-1 (2018) used this formulation — historical fidelity. Effective decay scaled by `1/sqrt(v_hat)` via adaptive moments. | Training instability surfaces that decoupled AdamW is known to fix (rare at 117M). |
| Tied input/output embeddings | GPT-1 canonical, saves `vocab_size × d_model` parameters (30.7 M at 40K×768 — ~26% of total). Required: input-lookup `scatter_add` and output-projection `matmul` gradients both land on the same matrix. | We decide untied output helps perplexity enough to justify 30 M extra params. |
| KV cache bypass during training | Per-token Python loop was the #1 bottleneck. Training computes full sequence at `start_pos=0` so the cache is unnecessary. Backward recomputes — correctness preserved. | Ever training in a streaming / fine-tuning mode where `start_pos > 0`. |
| Weight decay exclusion via `ndim >= 2` gate | Simple and naturally excludes all GPT-1 biases, LN gammas, LN betas (all 1D). Matches GPT-2 / LLaMA practice. | Positional embedding `W_pos` is 2D but GPT-1 excludes it from decay — need an explicit name-based skip on top of the gate. |

---

## How to Update This File

At the end of each milestone:

1. Mark the completed milestone's checkbox items as done (`- [x]`).
2. Change its status emoji from 🔄 to ✅.
3. Add a "Key metrics" line with actual measured values (perplexity, VRAM in MB, tokens/sec, step-0 loss, grad-norm range).
4. Append `**Frozen — do not modify this section.**` to the completed milestone.
5. Change the next milestone's status from ⏳ to 🔄.
6. Update the "Last updated" date and "Current milestone" at the top of this file.
7. Update the **Architecture: Current vs Target** table's "Current State" column to reflect the new committed state.
8. Update the **File Map** if files were added, removed, or had their ownership change.

Do not rewrite history — completed milestone sections are append-only after freezing. If a later milestone invalidates a frozen milestone's claim (e.g., Milestone 3 changes the vocab size and the Milestone 0 perplexity number becomes meaningless at the new scale), add a single dated note under the frozen section ("2026-MM-DD: superseded by M3 — perplexity re-baselined to ..."), don't edit the original numbers.
