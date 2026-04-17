# GPT-1 Implementation Specification

**Target:** Apply Milestones 1 (Architecture Retrofit) and 2 (Optimizer & Regularization) to bring the MiniGPT codebase into GPT-1 (2018) fidelity. Milestones 3 (40K tokenizer, special tokens) and 4 (117M scale-up) are scoped here at the config/tokenizer level; the actual 117M training run is out of scope.

**Author's note:** This document is a spec — not code. Apply each file section in order and run the smoke test at the end. A large portion of Milestone 1/2 is already in the working tree (uncommitted); every section below begins with **Current State** so you can see what's already there.

---

## 0. Preamble: State of the working tree on 2026-04-17

| File | Current state | Needs work? |
|---|---|---|
| `src/minigpt/model.py` | GPT-1 refactor already in tree (uncommitted): `LayerNorm` with gamma+beta, post-norm block order, GELU with fused fwd/bwd, learned absolute positional embeddings `W_pos`, tied output via `W_emb.T`, no final LN, full MHA with biases, dropout at 4 locations (embed, MLP intermediate, attn-branch, ffn-branch). | Minor: add `model.train()`/`model.eval()` toggle; align dropout locations to user spec (softmax dropout in place of ffn-branch dropout); formalize `named_parameters_with_groups()` helper. |
| `src/minigpt/optimizer.py` | GPT-1 refactor already in tree (uncommitted): `Adam` class with coupled L2 weight decay; `lr=2.5e-4, betas=(0.9, 0.98), eps=1e-8, wd=0.01`; `p.ndim >= 2` WD gate; `AdamW = Adam` alias for back-compat. | Minor: add explicit positional-embedding WD exclusion; add `LRSchedule` helper class (linear warmup + cosine to min_lr); add `build_param_groups(model)` helper. |
| `src/minigpt/config.py` | Legacy RMSNorm/SwiGLU/RoPE/GQA defaults: `d_model=384, n_layers=6, n_heads=6, n_kv_heads=2, vocab_size=4096, dropout=0.0, rope_theta=500000.0`; `d_ff` auto-formula is SwiGLU ratio `2*4*d/3` rounded to 256. `TrainConfig.learning_rate=6e-4, beta2=0.95, warmup_steps=500, weight_decay=0.1`. | Major: update every default to GPT-1 target. |
| `src/minigpt/tokenizer.py` | Pure-Python `BPETokenizer` with byte-level base, single `<EOS>` at id 256, merges from 257, pickle-based save/load. No `<pad>`/`<unk>`. | Add a new `HFBPETokenizer` class using `huggingface/tokenizers` lib with 40K vocab and the three special tokens; keep legacy class for back-compat. |
| `src/minigpt/trainer.py` | **Broken** against the new `model.backward()`: unpacks `attn_grads` as 2-tuple (`dW_qkv, dW_o`) — the new model returns 4-tuples (with biases); unpacks `ln*_d_gamma` as scalar — the new LN returns `(dgamma, dbeta)` pairs; expects top-level `(dW_emb, layer_grads, _, ln_f_d_gamma)` — the new top-level is `(dW_emb_out, dW_pos, layer_grads, dX_emb)` and there is no final LN. | Major: rewrite `clip_grads`, `apply_grads` flow, and `train` loop to match the new gradient tuple shapes. |
| `src/minigpt/backend.py` | Preserved — do not touch per user directive. | None (separate task to update `estimate_model_vram` for 117M — M4 scope). |
| `src/minigpt/inference.py` | Preserved — do not touch per user directive. | None. |
| `src/minigpt/optimized_kv_cache.py` | OK. `n_kv_heads = n_heads` in the new model so GQA is a no-op. | None. |
| `scripts/train.py` | Not read in this spec; likely needs updating to match new `model.backward()` tuple and to import `LRSchedule`. | Verify after `trainer.py` is rewritten — copy the clip/apply pattern over. |

---

## 1. `src/minigpt/config.py` — rewrite defaults

### Current
```python
@dataclass
class TokenizerConfig:
    vocab_size: int = 4096
    min_frequency: int = 3
    pattern: str = r"""..."""

@dataclass
class ModelConfig:
    vocab_size: int = 4096
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 2                   # legacy (GQA)
    d_ff: int = 0                         # auto = 2*4*d/3 (SwiGLU)
    max_len: int = 512
    dropout: float = 0.0
    rope_theta: float = 500000.0          # legacy (RoPE)

    def __post_init__(self):
        if self.d_ff == 0:
            hidden = int(2 * 4 * self.d_model / 3)
            self.d_ff = 256 * ((hidden + 255) // 256)

@dataclass
class TrainConfig:
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    batch_size: int = 64
    accum_steps: int = 2
    max_steps: int = 50000
    warmup_steps: int = 500
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    ...
```

### Target (apply directly)
```python
from dataclasses import dataclass, field
from typing import Optional, Tuple

@dataclass
class TokenizerConfig:
    """GPT-1 BPE tokenizer, 40K vocab + 3 special tokens."""
    vocab_size: int = 40000
    min_frequency: int = 2
    pattern: str = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    special_tokens: Tuple[str, ...] = ("<pad>", "<eos>", "<unk>")

@dataclass
class ModelConfig:
    """GPT-1 target: 117M parameters (d=768, L=12, H=12)."""
    vocab_size: int = 40000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 0                         # auto = 4 * d_model
    max_len: int = 512
    dropout: float = 0.1

    # --- Legacy fields kept at no-op defaults for back-compat with old
    #     checkpoints and scripts/train.py CLI flags. Do NOT use them in new
    #     code; model.py ignores rope_theta entirely and forces n_kv_heads =
    #     n_heads.
    n_kv_heads: int = 12                  # always == n_heads for GPT-1
    rope_theta: float = 0.0               # unused (no RoPE in GPT-1)

    def __post_init__(self):
        if self.d_ff == 0:
            self.d_ff = 4 * self.d_model          # GPT-1 ratio
        if self.n_kv_heads != self.n_heads:
            # GPT-1 has full MHA; silently normalize
            self.n_kv_heads = self.n_heads

@dataclass
class TrainConfig:
    """GPT-1 training: Adam + L2, linear warmup + cosine decay."""
    learning_rate: float = 2.5e-4
    min_lr: float = 1e-5
    batch_size: int = 8                   # micro-batch for 117M at T=512 FP16
    accum_steps: int = 8                  # effective batch = 64
    max_steps: int = 100000
    warmup_steps: int = 2000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98                   # GPT-1 paper value
    eps: float = 1e-8

    # Checkpointing & Logging
    eval_interval: int = 500
    log_interval: int = 10
    save_interval: int = 1000
    save_dir: str = "checkpoints_gpt1"

    # Dataloader
    seq_len: int = 512
```

### Smoke
```bash
python3 -c "from minigpt.config import ModelConfig, TrainConfig; \
    m = ModelConfig(); t = TrainConfig(); \
    print(m.d_model, m.n_layers, m.n_heads, m.d_ff, m.vocab_size, m.dropout); \
    print(t.learning_rate, t.betas := (t.beta1, t.beta2), t.weight_decay, t.warmup_steps)"
# Expect: 768 12 12 3072 40000 0.1
#         0.00025 (0.9, 0.98) 0.01 2000
```

---

## 2. `src/minigpt/model.py` — minor additions on top of existing GPT-1 refactor

### 2.1 Add `model.train()` / `model.eval()` toggle

**Rationale:** The current `forward(training=True/False)` arg is per-call. The user-facing contract asks for a mode that sticks across calls. Easiest: store `self._training` on `MiniTransformer` and have `.forward()` default to it.

**Apply inside `class MiniTransformer`, after `__init__`:**

```python
    # ---- mode toggle -------------------------------------------------------
    def train(self, mode: bool = True):
        """Enable dropout globally. Returns self for chaining."""
        self._training = mode
        return self

    def eval(self):
        """Disable dropout globally. Returns self for chaining."""
        self._training = False
        return self
```

Then in `__init__` add: `self._training = False`.

Modify `forward` signature to accept `training: Optional[bool] = None` and resolve:
```python
def forward(self, token_ids, start_pos: int = 0, training: Optional[bool] = None):
    if training is None:
        training = self._training
    ...
```

Propagate `training` to every layer as already done.

### 2.2 Re-align dropout locations to user spec

User asked for exactly these four dropout sites:
1. **After attention softmax** (on `attn_weights`, pre-`@ V`) — **currently missing.**
2. **After attention output projection** (residual branch) — ✅ currently in `TransformerBlock` (the "attn mask").
3. **After GELU in FFN** (MLP intermediate) — ✅ currently in `FeedForward`.
4. **After input embedding sum** — ✅ currently in `MiniTransformer.forward`.

The current tree also has a 5th dropout (on the FFN residual branch in `TransformerBlock`). Remove it to match the user's 4-site spec. Add softmax dropout in `MultiHeadAttention`.

**Changes to `MultiHeadAttention.forward`:**

```python
# After: attn_weights = softmax(scores, axis=-1)
# Add:
if training and self.p_dropout > 0:
    mask = (xp.random.rand(*attn_weights.shape) > self.p_dropout).astype(xp.float32) \
           / (1.0 - self.p_dropout)
    attn_weights = attn_weights * mask
    self.dropout_mask_softmax = mask
else:
    self.dropout_mask_softmax = None

attn_output = fp16_matmul(attn_weights, v)
```

Add `self.p_dropout = config.dropout` and `self.dropout_mask_softmax = None` to `MultiHeadAttention.__init__`.

**Changes to `MultiHeadAttention.backward`:** after computing `d_weights` but before the softmax Jacobian-vector product, re-apply the stored mask (if any):

```python
if self.dropout_mask_softmax is not None:
    d_weights = d_weights * self.dropout_mask_softmax
# ... then the existing softmax-jacobian code
```

**Changes to `TransformerBlock.forward` / `backward`:** remove the FFN-branch dropout block (the one that creates `self.dropout_mask_ffn`). Keep the attn-branch dropout (`self.dropout_mask_attn`). The corresponding backward code that multiplies `dffn_out` by `self.dropout_mask_ffn` also goes away.

### 2.3 Add `named_parameters_with_groups()`

Used by the optimizer to build WD-include / WD-exclude groups.

```python
def named_parameters_with_groups(self):
    """
    Yield (name, param, group) tuples where group is either 'decay' or 'no_decay'.

    No-decay group covers: LayerNorm gamma/beta, all biases, and positional
    embeddings W_pos (GPT-1 convention).
    """
    no_decay_names = (
        ".gamma", ".beta",
        ".b_qkv", ".b_o", ".b_fc", ".b_proj",
        "embeddings.W_pos",
    )
    for name, p in self.named_parameters():
        group = "no_decay" if any(name.endswith(s) or s in name for s in no_decay_names) else "decay"
        yield name, p, group
```

---

## 3. `src/minigpt/optimizer.py` — add `LRSchedule` + `build_param_groups`

### 3.1 Add at bottom of the file, after the `AdamW = Adam` alias:

```python
import math


class LRSchedule:
    """
    Linear warmup 0 -> lr over warmup_steps, then cosine decay to min_lr
    over (max_steps - warmup_steps).

    Usage:
        sched = LRSchedule(peak_lr=2.5e-4, min_lr=1e-5,
                           warmup_steps=2000, max_steps=100_000)
        for step in range(max_steps):
            lr_now = sched(step)
            optimizer.step(params, grads, lr=lr_now)
    """
    def __init__(self, peak_lr: float, min_lr: float,
                 warmup_steps: int, max_steps: int):
        assert 0 < warmup_steps < max_steps
        self.peak = peak_lr
        self.floor = min_lr
        self.warmup = warmup_steps
        self.total = max_steps

    def __call__(self, step: int) -> float:
        if step < self.warmup:
            return self.peak * (step + 1) / self.warmup
        if step >= self.total:
            return self.floor
        progress = (step - self.warmup) / (self.total - self.warmup)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.floor + coeff * (self.peak - self.floor)


def build_param_groups(model, weight_decay: float = 0.01):
    """
    Returns a list of dicts:
        [{'params': [...], 'weight_decay': weight_decay},   # 2-D weights
         {'params': [...], 'weight_decay': 0.0}]            # biases, LN, W_pos

    The model is expected to expose `named_parameters_with_groups()`
    (added in model.py section 2.3).
    """
    decay_params, no_decay_params = [], []
    for _, p, group in model.named_parameters_with_groups():
        (decay_params if group == "decay" else no_decay_params).append(p)
    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
```

### 3.2 Modify `Adam.step` to accept per-group `weight_decay`

**Current signature keeps working** (single `self.weight_decay`). **Add a new method** that takes explicit groups:

```python
def step_grouped(self, param_groups, grad_groups, lr: Optional[float] = None) -> None:
    """
    Per-group Adam update. Each entry in param_groups is a dict with keys
    'params' and 'weight_decay'; grad_groups mirrors the shape.
    """
    current_lr = lr if lr is not None else self.lr
    b1, b2 = self.betas
    for pg, gg in zip(param_groups, grad_groups):
        wd = pg['weight_decay']
        for p, g in zip(pg['params'], gg['params']):
            if g is None:
                continue
            p_id = id(p)
            if p_id not in self.state:
                self.state[p_id] = {'m': xp.zeros_like(p), 'v': xp.zeros_like(p), 't': 0}
            s = self.state[p_id]
            s['t'] += 1
            t = s['t']
            if wd > 0.0:
                g = g + wd * p
            s['m'] = b1 * s['m'] + (1 - b1) * g
            s['v'] = b2 * s['v'] + (1 - b2) * (g * g)
            m_hat = s['m'] / (1 - b1 ** t)
            v_hat = s['v'] / (1 - b2 ** t)
            p -= current_lr * m_hat / (xp.sqrt(v_hat) + self.eps)
```

The existing `step(params, grads, lr)` stays untouched for back-compat; `trainer.py` (section 5) will switch to `step_grouped`.

---

## 4. `src/minigpt/tokenizer.py` — add `HFBPETokenizer`

Add this class at the **bottom** of the file (keep the existing pure-Python `BPETokenizer` for any scripts still using it):

```python
# =========================================================================
#   HuggingFace-tokenizers-backed BPE (GPT-1 target: 40K + 3 specials)
# =========================================================================

class HFBPETokenizer:
    """
    BPE tokenizer backed by the `tokenizers` library (Rust impl).
    Matches the public API of BPETokenizer: train / encode / decode / save / load.

    Requires: `pip install tokenizers`
    """
    def __init__(self, config: Optional[TokenizerConfig] = None):
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder

        self.config = config or TokenizerConfig()
        self.vocab_size = self.config.vocab_size
        self.special_tokens = list(getattr(
            self.config, "special_tokens", ("<pad>", "<eos>", "<unk>")
        ))

        self._tk = Tokenizer(BPE(unk_token="<unk>"))
        self._tk.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self._tk.decoder = ByteLevelDecoder()

        # Cached after train/load
        self.pad_id = None
        self.eos_id = None
        self.unk_id = None

    # ---- train / save / load ------------------------------------------------
    def train_from_iterator(self, text_iter, min_frequency: int = 2):
        """Train on an iterator of strings (one document per item)."""
        from tokenizers.trainers import BpeTrainer
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=min_frequency,
            special_tokens=self.special_tokens,
            show_progress=True,
        )
        self._tk.train_from_iterator(text_iter, trainer=trainer)
        self._cache_special_ids()

    def train(self, text: str):
        """Train on a single string (splits on newlines to feed the trainer)."""
        self.train_from_iterator(iter(text.split("\n")),
                                 min_frequency=self.config.min_frequency)

    def save(self, path: str = "tokenizer_hf.json"):
        self._tk.save(path)

    def load(self, path: str):
        from tokenizers import Tokenizer
        self._tk = Tokenizer.from_file(path)
        self._cache_special_ids()

    def _cache_special_ids(self):
        self.pad_id = self._tk.token_to_id("<pad>")
        self.eos_id = self._tk.token_to_id("<eos>")
        self.unk_id = self._tk.token_to_id("<unk>")

    # ---- encode / decode (matches BPETokenizer interface) -------------------
    def encode(self, text: str) -> List[int]:
        return self._tk.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self._tk.decode(ids, skip_special_tokens=False)
```

### 4.1 FineWeb-Edu training recipe (add as a `@classmethod`)

```python
    @classmethod
    def train_on_fineweb(cls, n_docs: int = 100_000,
                         vocab_size: int = 40_000,
                         save_path: str = "assets/tokenizer_gpt1_40k.json"):
        """
        Stream FineWeb-Edu, train, save. Requires `datasets` installed.
        """
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb-edu",
                          name="sample-10BT", split="train", streaming=True)
        def text_iter():
            for i, ex in enumerate(ds):
                if i >= n_docs:
                    break
                yield ex["text"]
        cfg = TokenizerConfig(vocab_size=vocab_size,
                              special_tokens=("<pad>", "<eos>", "<unk>"))
        tok = cls(cfg)
        tok.train_from_iterator(text_iter())
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tok.save(save_path)
        print(f"[HFBPETokenizer] Saved to {save_path} "
              f"(pad={tok.pad_id}, eos={tok.eos_id}, unk={tok.unk_id})")
        return tok
```

---

## 5. `src/minigpt/trainer.py` — rewrite for new backward API

This file is currently **broken** against the new `model.backward()`. Rewrite the relevant methods:

### 5.1 New `clip_grads`

```python
def clip_grads(self, grads):
    """
    grads = (dW_emb_out, dW_pos, layer_grads, dX_emb)
    where each layer's grads is:
        (ffn_grads, attn_grads, (ln1_dg, ln1_db), (ln2_dg, ln2_db))
        ffn_grads  = (dW_fc, db_fc, dW_proj, db_proj)
        attn_grads = (dW_qkv, db_qkv, dW_o, db_o)
    """
    dW_emb_out, dW_pos, layer_grads, _dX_emb = grads
    sq = float(xp.sum(dW_emb_out ** 2)) + float(xp.sum(dW_pos ** 2))

    for ffn_g, attn_g, (ln1_dg, ln1_db), (ln2_dg, ln2_db) in layer_grads:
        for g in ffn_g:
            sq += float(xp.sum(g ** 2))
        for g in attn_g:
            sq += float(xp.sum(g ** 2))
        sq += float(xp.sum(ln1_dg ** 2)) + float(xp.sum(ln1_db ** 2))
        sq += float(xp.sum(ln2_dg ** 2)) + float(xp.sum(ln2_db ** 2))

    # Note: dX_emb is not clipped — it is scatter-added into dW_emb_out
    # after clipping by the apply step.

    grad_norm = math.sqrt(sq)
    scale = 1.0
    if grad_norm > self.config.grad_clip:
        scale = self.config.grad_clip / (grad_norm + 1e-6)

    if scale < 1.0:
        def s(g):
            if isinstance(g, tuple): return tuple(s(x) for x in g)
            if isinstance(g, list):  return [s(x) for x in g]
            if hasattr(g, "__mul__"): return g * scale
            return g
        grads = s(grads)

    return grads, grad_norm
```

### 5.2 New `apply_grads` — uses `step_grouped`

```python
def apply_grads(self, grads, token_ids, lr: float):
    dW_emb_out, dW_pos, layer_grads, dX_emb = grads

    # Accumulate input-lookup contribution into tied embedding gradient
    dW_emb_total = dW_emb_out.copy()
    flat_ids = token_ids.flatten()
    flat_grads = dX_emb.reshape(-1, self.model.config.d_model)
    scatter_add(dW_emb_total, flat_ids, flat_grads)

    # Build flat param/grad lists matching param groups
    from .optimizer import build_param_groups
    pg = build_param_groups(self.model, weight_decay=self.config.weight_decay)
    # Construct grad groups in the same order as pg
    grad_map = {id(p): None for g in pg for p in g['params']}

    # Map gradients to params
    grad_map[id(self.model.embeddings.W_emb)] = dW_emb_total
    grad_map[id(self.model.embeddings.W_pos)] = dW_pos
    for layer, (ffn_g, attn_g, (ln1_dg, ln1_db), (ln2_dg, ln2_db)) in zip(
        self.model.layers, layer_grads
    ):
        dW_fc, db_fc, dW_proj, db_proj = ffn_g
        dW_qkv, db_qkv, dW_o, db_o = attn_g
        grad_map[id(layer.ffn.W_fc)]   = dW_fc
        grad_map[id(layer.ffn.b_fc)]   = db_fc
        grad_map[id(layer.ffn.W_proj)] = dW_proj
        grad_map[id(layer.ffn.b_proj)] = db_proj
        grad_map[id(layer.attn.W_qkv)] = dW_qkv
        grad_map[id(layer.attn.b_qkv)] = db_qkv
        grad_map[id(layer.attn.W_o)]   = dW_o
        grad_map[id(layer.attn.b_o)]   = db_o
        grad_map[id(layer.ln1.gamma)]  = ln1_dg
        grad_map[id(layer.ln1.beta)]   = ln1_db
        grad_map[id(layer.ln2.gamma)]  = ln2_dg
        grad_map[id(layer.ln2.beta)]   = ln2_db

    grad_groups = [
        {'params': [grad_map[id(p)] for p in g['params']]}
        for g in pg
    ]
    self.optimizer.step_grouped(pg, grad_groups, lr=lr)
```

### 5.3 New `__init__` wiring

```python
def __init__(self, model, config, tokenizer=None):
    self.model = model
    self.config = config
    self.tokenizer = tokenizer

    from .optimizer import Adam, LRSchedule
    self.optimizer = Adam(
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=getattr(config, "eps", 1e-8),
        weight_decay=config.weight_decay,
    )
    self.schedule = LRSchedule(
        peak_lr=config.learning_rate,
        min_lr=config.min_lr,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
    )
    self.steps = 0
```

### 5.4 In `train` loop: replace `self.optimizer.lr = self.get_lr(step)` with:

```python
lr_now = self.schedule(step)
self.model.train()    # set dropout mode
...
self.apply_grads(grads, x, lr=lr_now)
```

Delete the old `get_lr` method entirely.

---

## 6. Smoke test

Save as `scripts/smoke_test_gpt1.py`:

```python
#!/usr/bin/env python3
"""Verify Milestones 1+2 apply cleanly at small scale before 117M scale-up."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
from minigpt.backend import xp
from minigpt.config import ModelConfig
from minigpt.model import MiniTransformer
from minigpt.optimizer import Adam, LRSchedule, build_param_groups

# ----- 1. Small-scale instantiation (d=384, L=6) -------------------------
cfg = ModelConfig(d_model=384, n_layers=6, n_heads=6,
                  vocab_size=4096, max_len=128, dropout=0.1)
model = MiniTransformer(cfg)

# ----- 2. Tied-embedding uniqueness --------------------------------------
names = [n for n, _ in model.named_parameters()]
w_emb_count = sum(1 for n in names if n == "embeddings.W_emb")
assert w_emb_count == 1, f"W_emb should appear exactly once, got {w_emb_count}"
# Also: output projection must NOT be a separate matrix
assert not any("W_out" in n or "lm_head" in n for n in names), \
    "Found a separate output head — tied embedding violated"
print(f"[OK] Tied embedding: W_emb appears exactly once, no W_out.")

# ----- 3. Parameter count ------------------------------------------------
total = sum(p.size for _, p in model.named_parameters())
print(f"[OK] Parameter count: {total:,}  (expect ~14M at d=384/L=6/V=4096)")

# ----- 4. Forward + backward on dummy batch ------------------------------
B, T = 2, 32
x = xp.asarray(np.random.randint(0, cfg.vocab_size, size=(B, T)).astype(np.int64))
y = xp.asarray(np.random.randint(0, cfg.vocab_size, size=(B, T)).astype(np.int64))

model.train()
logits, _ = model.forward(x, training=True)
assert logits.shape == (B, T, cfg.vocab_size)

# cross-entropy loss + gradient
B_, T_, V = logits.shape
logits_flat = logits.reshape(-1, V)
y_flat = y.reshape(-1)
mx = xp.max(logits_flat, axis=1, keepdims=True)
ex = xp.exp(logits_flat - mx)
probs = ex / xp.sum(ex, axis=1, keepdims=True)
N = logits_flat.shape[0]
loss = float(-xp.mean(xp.log(probs[xp.arange(N), y_flat] + 1e-9)))
dlogits = probs.copy()
dlogits[xp.arange(N), y_flat] -= 1
dlogits = (dlogits / N).reshape(B, T, V)

grads = model.backward(dlogits)
dW_emb_out, dW_pos, layer_grads, dX_emb = grads

# Check no NaN
import math
def has_nan(a): return bool(xp.any(xp.isnan(a)) | xp.any(xp.isinf(a)))
assert not has_nan(dW_emb_out), "NaN in dW_emb_out"
assert not has_nan(dW_pos),      "NaN in dW_pos"
print(f"[OK] Forward+backward. step-0 loss = {loss:.4f}  "
      f"(expect ~ln(V)={math.log(cfg.vocab_size):.4f})")

# ----- 5. Param-group decay verification ---------------------------------
pg = build_param_groups(model, weight_decay=0.01)
decay_names, no_decay_names = [], []
for name, p, group in model.named_parameters_with_groups():
    (decay_names if group == "decay" else no_decay_names).append(name)
print(f"[OK] Decay group ({len(decay_names)} params):")
for n in decay_names[:3]:  print(f"      {n}")
print(f"[OK] No-decay group ({len(no_decay_names)} params):")
for n in no_decay_names:   print(f"      {n}")

# Must-have no-decay entries
required_no_decay = {
    "embeddings.W_pos",
    "layers.0.ln1.gamma", "layers.0.ln1.beta",
    "layers.0.ln2.gamma", "layers.0.ln2.beta",
    "layers.0.attn.b_qkv", "layers.0.attn.b_o",
    "layers.0.ffn.b_fc",  "layers.0.ffn.b_proj",
}
missing = required_no_decay - set(no_decay_names)
assert not missing, f"These params must be no_decay but aren't: {missing}"
print(f"[OK] W_pos, LN gamma/beta, all biases correctly excluded from WD.")

# ----- 6. LR schedule sanity --------------------------------------------
sched = LRSchedule(peak_lr=2.5e-4, min_lr=1e-5,
                   warmup_steps=2000, max_steps=100_000)
assert sched(0)     < 1e-6          # ~0 at step 0
assert abs(sched(2000) - 2.5e-4) < 1e-8  # peak at end of warmup
assert sched(100_000) == 1e-5       # floor at end
print(f"[OK] LR schedule: step0={sched(0):.2e} "
      f"warmup_end={sched(2000):.2e} final={sched(100_000):.2e}")

print("\n========= ALL SMOKE TESTS PASSED =========")
```

Run with:
```bash
MINIGPT_BACKEND=numpy python3 scripts/smoke_test_gpt1.py
```

Expected output (last lines):
```
[OK] Tied embedding: W_emb appears exactly once, no W_out.
[OK] Parameter count: ~14,000,000
[OK] Forward+backward. step-0 loss ≈ 8.3 (expect ~ln(V)=8.3178)
[OK] Decay group (~38 params): ...
[OK] No-decay group (~50 params): embeddings.W_pos, layers.0.ln1.gamma, ...
[OK] W_pos, LN gamma/beta, all biases correctly excluded from WD.
[OK] LR schedule: step0=1.25e-07 warmup_end=2.50e-04 final=1.00e-05

========= ALL SMOKE TESTS PASSED =========
```

---

## 7. Verification checklist

After applying sections 1–5, confirm each of these before committing:

- [ ] `python -c "from minigpt.config import ModelConfig; print(ModelConfig())"` shows `d_model=768, n_layers=12, n_heads=12, vocab_size=40000, d_ff=3072, dropout=0.1`
- [ ] `grep -n "RMSNorm\|SwiGLU\|RoPE\|rope_theta\|precompute_freqs" src/minigpt/model.py` returns nothing (the model.py refactor should have no legacy references).
- [ ] `scripts/smoke_test_gpt1.py` passes on CPU (`MINIGPT_BACKEND=numpy`).
- [ ] `scripts/validate_gradients.py` re-runs with median error <10%, p90 <50% on the new modules.
- [ ] `scripts/benchmark_gpu.py` runs without error on CPU (just for shape checks); the 117M config is not benchmarked here (M4 scope).
- [ ] `trainer.py` train loop runs for 5 steps on a tiny dataset without NaN.
- [ ] `HFBPETokenizer(cfg).encode("<eos>")` returns `[eos_id]` (atomic special token).
- [ ] Round-trip: `tok.decode(tok.encode("The quick brown fox jumps over the lazy dog."))` reproduces the input.

---

## 8. What's NOT in this spec

The following are explicitly **preserved as-is** per the user's directive:
- `src/minigpt/backend.py` — FP16 `fp16_matmul`, `fuse` decorator, `scatter_add`, VRAM utilities
- `src/minigpt/inference.py` — generation, sampling, RAG pipeline
- `src/minigpt/optimized_kv_cache.py` — ring-buffer KV cache
- Checkpoint save/load logic in `scripts/train.py`
- `scripts/validate_gradients.py`, `scripts/benchmark_gpu.py`
- FineWeb-Edu streaming pipeline in `scripts/prepare_fineweb.py`

The 117M scale-up (actual training on RTX 3060 12 GB) is Milestone 4 and is out of scope for this spec. The config changes in section 1 are sufficient to *enable* it, but the VRAM estimator in `backend.py` still uses the SwiGLU `d_ff` ratio and will under-estimate by ~33%. Update before the full-scale run.
