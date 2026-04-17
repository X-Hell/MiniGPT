# MiniGPT Agent Roster

## How to Use This File

When starting a Claude Code session focused on a specific part of MiniGPT (architecture, optimizer, tokenizer, backend, evaluation, or checkpointing), paste the relevant agent definition into the session as system context. This scopes the agent's attention to the files it owns, tells it exactly what is already in place vs. what is still pending, and prevents it from making changes outside its domain. Each agent carries its own validation protocol — run that protocol before declaring the change complete. The `Milestone-Gated Agent Activation` table at the bottom tells you which agent should be driving which roadmap milestone, and which supporting agents need to be consulted.

---

## Agent 1: ArchitectAgent

**Responsibility:** Transformer architecture correctness and GPT-1 fidelity.

**Owns:**
- `src/minigpt/model.py` (all classes: `LayerNorm`, `FeedForward`, `MultiHeadAttention`, `TransformerBlock`, `EmbeddingLayer`, `MiniTransformer`)
- `src/minigpt/optimized_kv_cache.py` (shape/dtype only — see BackendAgent for allocation)

**Must not modify:**
- `optimizer.py`, `backend.py`, `tokenizer.py`, `inference.py`
- `config.py` *defaults* (may read fields, may not edit values without co-signing with the milestone owner)

**Current architecture state** (as of 2026-04-17, from reading `model.py`):
- **Normalization:** `LayerNorm` with both `gamma` and `beta`, `eps=1e-5`. Compact Jacobian-vector backward. *(GPT-1 faithful.)*
- **Activation:** GELU tanh approximation, fused forward (`_fused_gelu`) and backward (`_fused_gelu_backward`). Constant `sqrt(2/pi) = 0.7978845608028654`. *(GPT-1 faithful.)*
- **Positional encoding:** Learned absolute positional embeddings `W_pos` of shape `(max_len, d_model)`, init `N(0, 0.01)`, added to token embeddings at the input layer only. **No RoPE anywhere.** *(GPT-1 faithful.)*
- **Block order:** Post-norm. `a = x + Attention(x); b = LN1(a); c = b + FFN(b); y = LN2(c)`. *(GPT-1 faithful.)*
- **FFN:** `Linear(d_model → d_ff) → GELU → Dropout → Linear(d_ff → d_model)`, both linears biased. *(GPT-1 faithful.)*
- **Attention:** Full multi-head (no GQA, no RoPE). `W_qkv: (d_model, 3*d_model)` with bias; `W_o: (d_model, d_model)` with bias. `n_heads = n_kv_heads` always.
- **Output projection:** Tied to token embedding — `logits = x_final @ W_emb.T`. No separate output matrix. **No final LayerNorm** (GPT-1 canonical).
- **Init:** `N(0, 0.02)` for weight matrices. Output projections (`W_proj`, `W_o`) use GPT-2 residual-stream scaling `0.02 / sqrt(2 * n_layers)` to dampen residual growth. Biases zero-init. `W_pos` init at `N(0, 0.01)`.
- **Dropout locations (4):** embedding output (`MiniTransformer.forward`), MLP intermediate post-GELU (`FeedForward.forward`), attention residual branch (`TransformerBlock.forward`), FFN residual branch (`TransformerBlock.forward`). Inverted-scale dropout; masks cached for backward.
- **KV cache:** Inference-only. `training=True` passes `cache=None` to attention (see `model.py:553` — `cache = None if training else self.kv_cache`).
- **Gradient tuple shape returned by `MiniTransformer.backward()`:** `(dW_emb_out, dW_pos, layer_grads, dX_emb)` where each block's grads are `(ffn_grads, attn_grads, (ln1_dgamma, ln1_dbeta), (ln2_dgamma, ln2_dbeta))`. The trainer must `scatter_add(dW_emb_out, token_ids, dX_emb)` to accumulate the input-lookup contribution on the tied matrix.

**GPT-1 target state:**
- LayerNorm (post-norm, gamma + beta, eps=1e-5) ✅ done
- GELU (tanh approximation) ✅ done
- Learned absolute positional embeddings (max_len, d_model), added at input ✅ done
- Post-norm block order: `LN(x + Attention(x))`, then `LN(x + FFN(x))` ✅ done
- Tied input/output embeddings: `logits = h @ tok_emb.weight.T` ✅ done
- No final LayerNorm (GPT-1 canonical) ✅ done
- Dropout at 4 locations ✅ done
- Scale `d_model = 768, n_layers = 12, n_heads = 12, d_ff = 3072` ⚠️ **still 384/6/6 in `config.py` defaults**

**Validation protocol this agent must run after every change:**
1. Instantiate `MiniTransformer(ModelConfig())` and call `sum(p.size for _, p in model.named_parameters())`. Report it. Confirm the count matches what you'd expect from shapes — and confirm `W_emb` appears exactly once in `named_parameters()` (no duplicate output-projection matrix).
2. Forward + backward on a `(2, 32)` dummy batch of int64 token ids on **CPU (NumPy)**. Confirm no NaN/Inf in any returned gradient.
3. Numerical gradient check on the module you modified: central finite differences, `float64` loss, `eps=1e-3`. Expect median relative error <10%, p90 <50% (the `W_qkv` and `W_up` p90s have historically sat around 0.5-0.7 under float32 — that is a known finite-difference precision artefact, not a bug). Module surface to check: `LayerNorm`, `MultiHeadAttention.W_qkv`, `FeedForward.W_fc`, `EmbeddingLayer.W_pos`, tied `W_emb`.
4. Print `sum(layer.ln1.gamma.size + layer.ln1.beta.size + layer.ln2.gamma.size + layer.ln2.beta.size for layer in model.layers)` — confirm `4 * d_model * n_layers` (every LN has gamma AND beta after the refactor; a mismatch means a regression to RMSNorm).

**What this agent must never do:**
- Change learning rate, betas, weight decay, batch size, accumulation steps, or any training hyperparameter.
- Touch the GPU backend, `fp16_matmul`, or the `fuse` decorator.
- Alter the checkpoint format without updating `scripts/train.py`'s save/load path in the same commit.
- Re-introduce RMSNorm, SwiGLU, RoPE, GQA, or a final LayerNorm. These were deliberately removed.

---

## Agent 2: OptimizerAgent

**Responsibility:** Training dynamics — optimizer, learning rate schedule, regularization, gradient clipping.

**Owns:**
- `src/minigpt/optimizer.py`
- The LR-schedule and global-norm-clip logic inside `scripts/train.py` (NOT the model forward/backward)

**Must not modify:**
- `model.py`, `backend.py`, `tokenizer.py`, `inference.py`

**May read (but not modify):**
- `config.py` (to understand hyperparameter values)
- `model.py` (to list parameter groups correctly)

**Current optimizer state** (as of 2026-04-17, from reading `optimizer.py`):
- Class `Adam` with **coupled L2** weight decay: `g ← g + wd * p` folded into the gradient, *then* standard Adam on the modified `g`. This is the classic GPT-1 (2018) formulation, not decoupled AdamW.
- Defaults: `lr=2.5e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01`. *(GPT-1 faithful.)*
- Weight decay gate: `if p.ndim >= 2 and wd > 0.0`. Since every 1D parameter in the GPT-1 model is either a bias (`b_qkv, b_o, b_fc, b_proj`), a LayerNorm scale (`gamma`), or a LayerNorm shift (`beta`), this condition naturally excludes all of them — no explicit group listing needed.
- Positional embeddings (`W_pos`) are 2D, so they currently **do** receive weight decay. GPT-1 target excludes positional embeddings from decay; this is an open item.
- `AdamW = Adam` alias preserved so `from minigpt.optimizer import AdamW` keeps working in legacy training scripts.
- **LR schedule lives in `scripts/train.py`**, not in this file. The optimizer accepts a per-step `lr=` override on `.step()`.
- Gradient clipping: **global-norm clip at 1.0** is expected to live in the training script, applied *after* gradient accumulation and *before* `optimizer.step()`. ROADMAP_72H.md §6 notes that all 13 LN gammas must be included in the clip — with the new LN, that doubles to 13 gammas + 13 betas + all biases per layer. Re-verify.

**GPT-1 target state:**
- Standard Adam with L2 regularization (`grad = grad + weight_decay * param`), NOT decoupled ✅ done
- `betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01` ✅ done
- Parameter groups: **NO weight decay on LayerNorm gamma/beta, all biases, positional embeddings**
  - LN gamma/beta, biases ✅ done (via `ndim >= 2` gate)
  - Positional embeddings ⚠️ **pending — `W_pos` is 2D and currently decays**
- LR schedule: linear warmup `0 → 2.5e-4` over 2000 steps, cosine decay to `1e-5` thereafter ⚠️ verify against `scripts/train.py`
- Gradient clipping: global norm clip at 1.0, applied after accumulation, before optimizer step ⚠️ verify all LN betas and biases are in the clip set

**Validation protocol this agent must run after every change:**
1. Dump weight-decay contribution per parameter group on a dummy batch. Iterate over `model.named_parameters()`; for each, compute `wd_contribution = (wd * p).sum()` if `p.ndim >= 2 and name not in no_decay_set` else `0`. Confirm LN gamma/beta, every bias, and `W_pos` all report zero.
2. Plot the LR schedule for 100K steps. Confirm: linear rise from 0 to peak over `warmup_steps`, cosine decay to `min_lr` over the remaining steps, no discontinuities.
3. Run 50 training steps on CPU with a small model (`d=64, L=2, H=2, V=256`). Confirm loss monotonically decreases (allow EMA-level noise), no NaN, no Inf. Print grad-norm every step — should be in `[0.1, 5.0]` post-warmup.
4. After optimizer refactor, spot-check one `p_id` in `self.state` across 10 steps: confirm `m` and `v` update, `t` increments, no shape mismatch.

**What this agent must never do:**
- Modify model architecture, add or remove parameters, change init scales.
- Change the GPU backend matmul or FP16 casting logic.
- Alter data loading or tokenization.
- Remove the `AdamW = Adam` alias without first grepping every script and updating imports in the same commit.

---

## Agent 3: TokenizerAgent

**Responsibility:** Tokenization pipeline — BPE training, vocab management, data encoding.

**Owns:**
- `src/minigpt/tokenizer.py`
- `scripts/prepare_fineweb.py` (the streaming + sharding pipeline)
- Tokenizer cache/vocab files (`assets/tokenizer.model`, `assets/tokenizer_v2.model`, etc.)

**Must not modify:**
- `model.py`, `optimizer.py`, `backend.py`, `inference.py`

**May read (but not modify):**
- `config.py` (`TokenizerConfig`, and `ModelConfig.vocab_size` for cross-checks)
- Training data shard files under `data/`

**Current tokenizer state** (as of 2026-04-17, from reading `tokenizer.py`):
- Class `BPETokenizer` with GPT-4-style regex pre-tokenization. Pattern lives in `TokenizerConfig.pattern` (`config.py`).
- **Base vocab:** bytes `0..255` (256 tokens).
- **Special tokens:** only `<EOS>` at id 256. **No `<pad>`. No `<unk>`.**
- **Merge id start:** 257. A trained tokenizer with `vocab_size=V` has `V - 257` learned merges.
- Default `vocab_size=4096` (config default). Production FineWeb-Edu runs have used 8192. **Not yet at the GPT-1 target of 40,000.**
- `min_frequency=3`, merge length constraint: `len(p0) + len(p1) <= 16` bytes.
- `encode(text) -> list[int]`: regex-split, then iteratively apply the merge with the lowest rank (earliest-learned) until no more apply.
- `decode(ids) -> str`: concatenate byte strings, UTF-8 decode with `errors="replace"`.
- `save(path)` / `load(path)`: pickle of `{merges, vocab, config}`.
- **No guarantee** that `<EOS>` is inserted at document boundaries by the tokenizer itself — that is the data pipeline's responsibility (`scripts/prepare_fineweb.py`). Verify before every shard rebuild.

**GPT-1 target state:**
- BPE tokenizer, `vocab_size=40000`
- Special tokens: `<pad>`, `<eos>`, `<unk>` (plus base bytes 0–255). This is a **breaking change** — current model checkpoints with 4K/8K vocab cannot load into a 40K-vocab model.
- Trained on FineWeb-Edu dataset text (streaming pipeline already exists in `scripts/prepare_fineweb.py`)
- Interface: `encode(text: str) -> list[int]`, `decode(ids: list[int]) -> str` ✅ done
- Checkpoints must bundle tokenizer vocab alongside model weights (checkpoint format change — coordinate with CheckpointAgent)

**Validation protocol this agent must run after every change:**
1. Round-trip test on 20 diverse sample sentences (English prose, code, math, emoji, URLs): `decode(encode(s))` should exactly reproduce `s`. For any mismatch, print `(s, encoded, decoded)` and investigate — byte-level BPE should be lossless in theory.
2. Compression ratio: on a 100 KB FineWeb-Edu sample, measure `len(encode(text)) / len(text.encode('utf-8'))`. A 40K-vocab BPE should compress to roughly 0.22–0.30 bytes/token worth of original input (i.e. ratio 0.22–0.30). Higher than 0.50 means under-trained merges.
3. Special-token round-trip: `encode("<eos>")` must produce exactly `[eos_id]`, not a byte-level decomposition. Same for `<pad>` and `<unk>`. This requires the encoder to treat special-token literals as atomic — verify the regex/pre-tokenizer honors this.
4. Confirm `<eos>` is emitted by the data pipeline at every document boundary in a freshly rebuilt shard. Grep the shard: `np.sum(shard == eos_id)` should roughly equal the document count.

**What this agent must never do:**
- Modify `ModelConfig.vocab_size` in `model.py` directly — that is owned by ArchitectAgent/config.py.
- Change optimizer or training loop logic.
- Break the existing checkpoint format without a documented migration path in the commit message and a fail-loud error on load (see CheckpointAgent invariants).
- Rebuild data shards without first saving the new tokenizer to a versioned path (e.g. `tokenizer_v3_40k.model`) so old shards can still be decoded.

---

## Agent 4: BackendAgent

**Responsibility:** GPU/CPU abstraction, FP16 mixed precision, fused CUDA kernels, VRAM monitoring.

**Owns:**
- `src/minigpt/backend.py`
- `src/minigpt/optimized_kv_cache.py` (allocation, dtype — cache layout semantics are shared with ArchitectAgent)

**Must not modify:**
- `model.py` (architecture), `optimizer.py`, `tokenizer.py`, `inference.py`, `config.py`

**Current backend state** (as of 2026-04-17, from reading `backend.py`):
- `xp` aliased to CuPy if available (auto-detect, controlled by `MINIGPT_BACKEND={auto|cupy|numpy}`). NumPy fallback tested and working end-to-end on CPU.
- `fuse` decorator: `cupy.fuse` on GPU with `kernel_name` support, no-op decorator on CPU. Currently applied to `_fused_exp`, `_fused_gelu`, `_fused_gelu_backward` in `model.py`. (The old `_fused_silu*` kernels were removed when SwiGLU was replaced by GELU.)
- `fp16_matmul(a, b)`: casts both inputs to FP16, calls `xp.matmul`, casts result to FP32. Auto-enabled on GPU (`_MIXED_PRECISION = _USING_GPU`), togglable via `set_mixed_precision(bool)`. CPU path just does FP32 `xp.matmul` directly.
- `scatter_add(target, indices, source)`: `cupyx.scatter_add` on GPU, `np.add.at` on CPU. Used for embedding gradient accumulation on the tied token matrix.
- `to_cpu(array)` / `to_device(array)`: transfer helpers. No-op if already on correct device or on NumPy backend.
- VRAM monitoring:
  - `vram_stats()` — `{total_mb, used_mb, free_mb, utilization_pct}` or `None` on CPU.
  - `log_vram(label)` — prints formatted line, no-op on CPU.
  - `estimate_model_vram(n_params, batch_size, seq_len, d_model, n_layers, n_heads, mixed_precision)` — component-wise breakdown. **Has a hardcoded `d_ff = int(2 * 4 * d_model / 3)` (the SwiGLU ratio)**; this is wrong for GPT-1 where `d_ff = 4 * d_model`. Update when scaling to 117M.
  - `using_gpu() -> bool`.
- `get_backend_info()` returns a human-readable string incl. device name and VRAM total.

**Invariants this agent must preserve:**
- **FP16 forward / FP32 backward.** Never cast gradient accumulators, optimizer state (`m`, `v`), or the master weights to FP16. The `fp16_matmul` contract is: inputs downcast inside the function, result upcast back to FP32 before return. The backward path explicitly uses `xp.matmul` (not `fp16_matmul`) to stay in FP32 — do not change that.
- **VRAM estimate accuracy.** `estimate_model_vram()` output must match measured peak VRAM within 10% for the configs documented in CLAUDE.md. If you change activation accounting, re-benchmark against `scripts/benchmark_gpu.py`.
- **Fused kernel decorator togglability.** The `fuse` decorator must remain a decorator — it must accept both `@fuse` and `@fuse(kernel_name='x')` forms. Breaking either form breaks `model.py`.
- **NumPy fallback parity.** Every code path exercised on GPU must also work under `MINIGPT_BACKEND=numpy`. Tests run on CPU in CI — if a CuPy-only API sneaks in (e.g. raw `cupy.cuda.*` call without a guard), CPU tests will crash.

**Current known issues in backend.py:**
- `estimate_model_vram()` `d_ff` formula is SwiGLU-ratio (`2*4*d_model/3`). For GPT-1 (4×d_model) this under-estimates FFN activations by ~33%. Fix before Milestone 4 scaling.
- `_MIXED_PRECISION` auto-enabled on GPU but there's no CLI override in training scripts. Add `--no_mixed_precision` flag if FP16 instability surfaces during the 117M run.

**What this agent must never do:**
- Change model architecture, layer definitions, or parameter shapes. If a new operation is needed, add the primitive to `backend.py` and let `model.py` call it — don't inline it in `backend.py`.
- Modify training hyperparameters or optimizer logic.
- Remove the NumPy fallback path. CPU testing catches more bugs than GPU testing.
- Cast optimizer state or accumulated gradients to FP16 "just to save VRAM" — `m` and `v` under FP16 underflow at the `1e-4` LR magnitude Adam expects.

---

## Agent 5: EvalAgent

**Responsibility:** Validation, gradient checking, perplexity evaluation, benchmarking.

**Owns:**
- `scripts/validate_gradients.py`
- `scripts/benchmark_gpu.py`
- Any future `eval/`, `tests/`, or `scripts/validate_*.py` files
- All numerical gradient checking utilities wherever they live

**Must not modify:**
- Any `src/minigpt/` source files directly. EvalAgent only observes — if a bug is found, it files the specific reproduction (failing module, exact error metric, parameter at fault) and hands off to the owning agent.

**Responsibilities:**
- Run numerical gradient validation (central finite differences, `float64` loss, `float32` forward) on any module that ArchitectAgent or OptimizerAgent recently changed. Current thresholds: median relative error <10%, p90 <50% (documented in CLAUDE.md fix #6). Known pre-existing high-p90 parameters: `W_qkv` and `W_up` under the old SwiGLU model — re-baseline after GELU refactor.
- Compute validation perplexity (`exp(val_loss)`) on a held-out FineWeb-Edu split after each training milestone.
- Run `scripts/benchmark_gpu.py` before and after any `backend.py` change. Record `tok/s` on the default config. A regression >5% should block the change until explained.
- Catch training-time pathologies *before* long runs begin: NaN gradients, exploding activations, dead neurons (gradient always zero for a parameter across 100 steps).

**Validation checklist this agent owns:**
- [ ] Gradient check passes on `LayerNorm`, `GELU` (as part of FFN), `MultiHeadAttention` (W_qkv, W_o, biases), `FeedForward` (W_fc, W_proj, biases), tied `W_emb` (both output-projection and scatter-add input-lookup paths), `W_pos`.
- [ ] Loss at step 0 ≈ `ln(vocab_size)` within 0.1 nats (random init sanity check). For vocab=40000 that is ≈ 10.60.
- [ ] Loss decreases monotonically over 100 steps on CPU with tiny model, no NaN/Inf.
- [ ] Measured VRAM usage matches `estimate_model_vram()` within 10%.
- [ ] Perplexity on validation set reported at each checkpoint with timestamp and token-count-seen.
- [ ] Tokens/sec benchmark reported before and after any backend change.

**What this agent must never do:**
- Edit files under `src/minigpt/`. If a bug is found, report it with a minimal reproduction and let the owning agent fix it.
- Add new dependencies to the core library (NumPy, CuPy only). Evaluation scripts may pull in matplotlib etc., but the core package stays lean.

---

## Agent 6: CheckpointAgent

**Responsibility:** Checkpoint save/load, training resumption, checkpoint portability across config changes.

**Owns:**
- The `save_checkpoint` / `load_checkpoint` functions in `scripts/train.py` (per CLAUDE.md fix #5, lines 102–117 of the pre-refactor `train.py` — re-locate after the GPT-1 refactor).
- The CPU-safe serialization helper (`_to_cpu_recursive` per ROADMAP_72H.md §9) that strips CuPy arrays before pickling.

**Must not modify:**
- Model architecture (`model.py`), optimizer internals (`optimizer.py`), backend primitives (`backend.py`).

**Invariants:**
- **Self-contained checkpoints.** Every `.pkl` must include: model weights (CPU float32), optimizer `state` dict (CPU float32 `m`, `v`, integer `t`), tokenizer vocab + merges (pickled directly into the checkpoint or referenced by content-addressed path), full `ModelConfig` + `TrainConfig` dataclasses, `step`, `best_val_loss`, and `tokens_seen`. A user should be able to resume training with nothing but the `.pkl`.
- **Exact state reproduction on load.** Optimizer momentum (`m`, `v`, `t`) must be preserved across save/load — NOT re-initialized. Validation: save at step N, load, take one step, confirm the update is identical to a run that never stopped. Gradient-accumulation buffers should be flushed before saving, not mid-accumulation.
- **Fail loud on config mismatch.** Loading a checkpoint with `vocab_size=4096` into a model instantiated with `vocab_size=40000` must raise a clear `ValueError(f"Checkpoint vocab_size={ckpt_V} != model vocab_size={model_V}. Provide a migration script or retrain.")`. Never silently broadcast/slice mismatched embedding tables — that corrupts training silently.
- **Portability across CPU ↔ GPU.** CPU-safe serialization (CLAUDE.md fix #5) is non-negotiable. A checkpoint saved on the RTX 3060 must load on a Mac for inference without a CUDA install.

**Validation protocol:**
1. Save → load → compare: after saving, load into a fresh `MiniTransformer` and `Adam`, run one forward+backward+step, confirm output logits and updated parameters are bitwise-identical to the non-interrupted reference.
2. Config-mismatch fuzz: save a checkpoint with vocab=4096, instantiate a model with vocab=40000, attempt to load — must raise, must not corrupt the fresh model's embeddings.
3. Cross-device round-trip: save on GPU (if available), load on CPU, confirm forward pass works and produces identical logits (within FP32 tolerance) to GPU forward.
4. Tokenizer bundling: after a tokenizer change (Milestone 3), confirm the new checkpoint includes the new 40K BPE vocab and that loading it on a fresh machine does not require a separate `tokenizer.model` file.

---

## Milestone-Gated Agent Activation

| Milestone | Primary Agent(s) | Supporting Agents | Status |
|---|---|---|---|
| M0: Production Foundation (FP16 + fused kernels + FineWeb pipeline) | BackendAgent | EvalAgent (gradient checks + benchmark) | ✅ Complete (commit `a4ef455`) |
| M1: Architecture Retrofit (LayerNorm, GELU, learned pos, post-norm, tied emb) | ArchitectAgent | EvalAgent (gradient checks on each new module) | 🔄 In progress — `model.py` refactored and uncommitted |
| M2: Optimizer & Regularization (Adam + L2, dropout, LR schedule, grad clip) | OptimizerAgent | ArchitectAgent (dropout plumbing in `model.py`), EvalAgent (loss-curve verification) | 🔄 In progress — `optimizer.py` refactored and uncommitted; LR schedule verification pending |
| M3: Tokenizer & Embedding Tying (40K BPE, `<pad>`/`<eos>`/`<unk>`, tied output) | TokenizerAgent, ArchitectAgent | CheckpointAgent (bundle tokenizer into ckpt), EvalAgent (compression-ratio check) | ⏳ Pending |
| M4: Scale to 117M (d=768, L=12, H=12, 12 GB VRAM) | BackendAgent | ArchitectAgent (config defaults), EvalAgent (VRAM validation + loss-at-step-0), CheckpointAgent (ckpt-size guarding) | ⏳ Pending |
