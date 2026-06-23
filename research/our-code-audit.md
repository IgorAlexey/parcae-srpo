# Code Audit: parcae-srpo Training Pipeline

**Date**: 2026-06-22  
**Scope**: `scripts/train_srpo.py`, `src/parcae/injection.py`, `src/parcae/model.py`  
**Model**: Gemma 4 E2B (4.6B params, 35 layers → 12+11+12, dim=1536)  
**Trainable**: 56,321 parameters (injection: 6,145 + DepthLoRA: 50,176)

---

## 1. `scripts/train_srpo.py` — Training Loop & SRPO Algorithm

### 1.1 `group_size=2` — CRITICAL for GRPO viability

- **File:Line**: `scripts/train_srpo.py:61`
- **Severity**: **BLOCKER**

**Finding**: GRPO computes a group-relative advantage normalized by the group's per-prompt mean and standard deviation. With `group_size=2`, the advantage space collapses:

| Rewards | Mean μ | Std σ (after 1e-8) | Adv A | Signal |
|---------|--------|---------------------|-------|--------|
| [0, 0]  | 0      | 1e-8               | [0, 0] | None — loss=0 |
| [1, 1]  | 1      | 1e-8               | [0, 0] | None — loss=0 |
| [0, 1]  | 0.5    | 0.707              | [-0.707, 0.707] | Binary only, no gradation |

With only 2 completions per prompt, advantage collapses to exactly three states: all-zero, all-positive, or binary signed. GRPO is designed to give _graded_ advantage signals across a distribution of completions — DeepSeekMath (2402.03300) uses G=64, DeepSeek-R1 used G=16.

**Evidence**: The README already acknowledges: _"The GRPO branch remains inactive until at least two completions in a group pass verification, which has not occurred consistently with the current builtin 20-problem dataset."_ The low G compounds with the small dataset.

**Recommendation**: Minimum G=4, target G=8–16 for meaningful GRPO. Even G=4 would allow advantage values in {-1.5, -0.5, 0.5, 1.5} (2 correct + 2 failed case) instead of the current binary split.

---

### 1.2 `gen_temperature=0.8` — too conservative for RL exploration

- **File:Line**: `scripts/train_srpo.py:63`
- **Severity**: **WARNING**

**Finding**: Temperature 0.8 is close to standard generation (1.0) and doesn't encourage exploration diversity. Combined with `group_size=2`, this means the two completions per prompt are likely very similar, further reducing the chance of getting a spread of rewards.

For RL exploration, temperatures in [1.0, 1.4] are typical (e.g., GRPO implementations in TRL use 1.0–1.2). The SDPO branch uses 0.6 for the teacher — that's appropriate since the teacher should be conservative. But for the student's exploration, 0.8 is low.

**Evidence**: `scripts/train_srpo.py:63` — `gen_temperature: float = 0.8`

**Recommendation**: Increase to 1.0–1.2. The SDPO teacher temperature of 0.6 (line ~898) is fine as-is.

---

### 1.3 `learning_rate=1e-4` — on the low side for 56K LoRA params

- **File:Line**: `scripts/train_srpo.py:70`
- **Severity**: **INFO**

**Finding**: With only 56,321 trainable parameters, 1e-4 is a conservative learning rate. Standard LoRA fine-tuning uses 1e-4 to 5e-4. For such a small parameter count relative to the backbone (56K / 4.6B = 0.0012%), a higher learning rate is typically safe.

**Evidence**: 
- `scripts/train_srpo.py:70`: `learning_rate: float = 1e-4`
- `scripts/train_srpo.py:72`: `weight_decay: float = 0.01` (standard)
- Trainable breakdown: 6,145 injection + 50,176 DepthLoRA = 56,321

**Recommendation**: Consider 2e-4 to 5e-4, especially since the injection's `B_raw` starts at magnitude 0.01 and needs to grow. Monitor `ρ(A)` to ensure it stays < 1.

---

### 1.4 GRPO NaN safety — handled correctly but with edge-case risks

- **File:Line**: `scripts/train_srpo.py:371-375` (sigma epsilon), `scripts/train_srpo.py:380` (B<2 guard)
- **Severity**: **INFO**

**Finding**: The GRPO implementation has two safeguards:
1. `sigma = rewards.std() + 1e-8` — prevents division by zero when all rewards are equal. When all rewards are 0 or all 1, advantage = 0, loss = 0. **No NaN**.
2. `if B < 2: return torch.tensor(0.0, ...)` — defensive guard. In practice, the caller (`train_step`) already checks `len(correct) >= 2` before calling `grpo_loss`, so this branch is never reached in normal operation. If reached, it returns 0.0 with no gradient.

**Edge case**: If `correct` contains 2+ samples but all rewards are identical (e.g., all 0, which is the common case), the loss is 0.0 but `.backward()` is still called. This produces zero gradients — correct behavior but wasteful compute (the old-policy forward for correct samples runs even though GRPO produces no signal).

**Evidence**: 
- `scripts/train_srpo.py:373`: `sigma = rewards.std() + 1e-8`
- `scripts/train_srpo.py:379-380`: `if B < 2: return torch.tensor(0.0, ...)`
- `scripts/train_srpo.py:937`: `if len(correct) >= 2:` (caller guard)

**Recommendation**: Add an early return when `rewards.std() < 1e-7` to skip the old-policy forward entirely. Currently the old-policy forward (line ~949) runs regardless.

---

### 1.5 `max_response_tokens=64` — adequate but limiting for complex code

- **File:Line**: `scripts/train_srpo.py:62`
- **Severity**: **WARNING**

**Finding**: 64 output tokens is already increased from the 16 mentioned in the README. For simple functions (e.g., `def factorial(n): return 1 if n==0 else n*factorial(n-1)` which is ~25 tokens), it's fine. For assembly generation, 64 tokens may constrain longer routines with multiple instructions.

For comparison: DeepSeek-R1 uses 4K–32K response tokens for math reasoning; SWE-bench generation targets 256–1024 tokens. The constraint here is memory — each additional response token compounds with the recurrent depth (T loops × N layers).

**Evidence**: `scripts/train_srpo.py:62`: `max_response_tokens: int = 64`

**Recommendation**: Consider 128 for code tasks, 256 for assembly. If memory is the bottleneck, reduce `poisson_mean` or `micro_batch_size` to compensate.

---

### 1.6 `loop_embedding_dim` mismatch between TrainConfig and RecurrentDepthConfig

- **File:Line**: `scripts/train_srpo.py:54` vs `src/parcae/model.py:101`
- **Severity**: **INFO**

**Finding**: `TrainConfig.loop_embedding_dim = 128` but `RecurrentDepthConfig.loop_embedding_dim = 256`. The training script overrides the model default at construction (`scripts/train_srpo.py:607`: `loop_embedding_dim=self.cfg.loop_embedding_dim`). Not a bug, but the inconsistency could cause confusion when using the model standalone vs. training.

**Recommendation**: Align defaults or document the difference.

---

### 1.7 README/code discrepancy: `find_unused_parameters`

- **File:Line**: `scripts/train_srpo.py:651` vs `README.md:49`
- **Severity**: **INFO**

**Finding**: README claims `find_unused_parameters=False` for DDP efficiency, but the code uses `find_unused_parameters=True`. Given the conditional branches (GRPO vs SDPO), `True` is safer — it prevents DDP crashes when a branch doesn't consume all trainable parameters. The performance tax is negligible for 56K params.

**Recommendation**: Update README to say `True`.

---

## 2. `src/parcae/injection.py` — LTI Injection

### 2.1 `B_raw` initialization at 0.01 — very weak initial injection

- **File:Line**: `src/parcae/injection.py:57`
- **Severity**: **INFO**

**Finding**: The effective injection magnitude at initialization is:

```
dt = exp(log_dt) = exp(-3.0) ≈ 0.05
B_discrete = dt × B_raw ≈ 0.05 × (0.01 × randn) ≈ 5e-4 × randn
```

So the injection term `B · e_norm` is ~5×10⁻⁴ of the normalized prelude output. This is essentially zero. Combined with `learning_rate=1e-4`, it may take many hundreds of steps before `B_raw` grows to a meaningful magnitude.

This is intentional (the model starts near the pretrained baseline and slowly learns injection), but it means early training provides no injection signal at all.

**Evidence**: 
- `src/parcae/injection.py:55`: `self.log_dt = nn.Parameter(torch.tensor(-3.0))`
- `src/parcae/injection.py:57`: `self.B_raw = nn.Parameter(torch.randn(dim) * 0.01)`
- `src/parcae/injection.py:84`: `B_discrete = dt * B_raw = exp(log_dt) * B_raw`

**Recommendation**: Consider 0.1 for faster injection learning, or increase the learning rate specifically for injection parameters. The Parcae paper uses small B initialization as well, so this is design-appropriate for stability.

---

### 2.2 `(A - I)·h` residual cancellation — theoretically correct ✓

- **File:Line**: `src/parcae/injection.py:99-101`
- **Severity**: **INFO** (no bug)

**Finding**: The derivation is correct:

```
transformer_out = h_t + Σ_deltas     (residual baked into HF decoder layers)
injection: (A - 1)·h_t + B·e_norm + transformer_out
         = A·h_t - h_t + B·e_norm + h_t + Σ_deltas
         = A·h_t + B·e_norm + Σ_deltas  ← pure Parcae update
```

The cancellation works because:
- Every HuggingFace decoder layer wraps its attention+FFN in a residual block (pre-norm or post-norm depending on architecture) that adds the input back. This is guaranteed by the `Gemma4UnifiedTextDecoderLayer` implementation.
- The `_run_block` function passes `h` through each layer sequentially, accumulating deltas. The output is `h + Σδ_i` where δ_i are the layer deltas (including their internal residuals).

**Evidence**: 
- `src/parcae/injection.py:94-101`: `return (A - 1.0) * h + B * e_norm + transformer_out`
- `src/parcae/model.py:300-325`: `_recurrent_loop` calls `self.injection(h, e, trans_out)` where `h` is pre-block and `trans_out` is post-block
- `src/parcae/model.py:385`: `_run_block` returns post-block hidden states

**Recommendation**: No change needed. The formulation is sound.

---

### 2.3 `ρ(A) < 1` guarantee — verified by construction

- **File:Line**: `src/parcae/injection.py:70-78`
- **Severity**: **INFO** (no bug)

**Finding**: The spectral radius is guaranteed < 1:
```
A_discrete = exp(-exp(log_dt + log_A))
```
Since `exp(x) > 0` for all real x, and `-exp(anything) < 0`, each diagonal entry is in (0, 1). The max absolute value is therefore always < 1.

The clamp on `log_dt + log_A` to [-20, 20] prevents overflow (when > 88, exp would overflow). At the clamped value of 20: `A_discrete ≈ exp(-485,000,000) ≈ 0`, meaning that channel decays to zero instantly — safe.

**Recommendation**: No change. The guarantee is robust.

---

## 3. `src/parcae/model.py` — Model Architecture

### 3.1 Recurrent loop is a Python-native for-loop — low overhead but no torch.compile

- **File:Line**: `src/parcae/model.py:300-336` (`_recurrent_loop`)
- **Severity**: **WARNING**

**Finding**: The `_recurrent_loop` method uses a Python `for t in range(n_loops):` loop with conditional branches (`if t == 0`, `if self.loop_embed is not None`, etc.). This cannot be `torch.compile`'d because:

1. `n_loops` varies per forward pass (Poisson sampling), breaking static graph tracing
2. Conditional `t == 0` changes the computation graph per iteration
3. Module attribute access (`self.depth_lora`, `self.injection`) introduces Python-object-dependent control flow

**However**: The heavy computation (`_run_block` → ~11 layers of Gemma 4 decoder) dominates runtime. The Python loop overhead is negligible compared to 11×T transformer layer evaluations. With T ≤ 8, this is O(1) Python overhead vs. O(T) compute.

**Evidence**: 
- `src/parcae/model.py:310`: `for t in range(n_loops):`
- `src/parcae/model.py:318`: `if self.loop_embed is not None and t > 0:`
- `src/parcae/model.py:329`: `if t == 0: ... else: self.injection(h, e, trans_out)`

**Recommendation**: Not a priority. If T grows to 16+, consider restructuring to `torch.fx` or explicit unrolling. Current T ∈ [1, 8] makes this negligible.

---

### 3.2 Forward pass count per `train_step` — very high

- **File:Line**: `scripts/train_srpo.py:840-940` (entire `train_step`)
- **Severity**: **WARNING**

**Finding**: Each `train_step` issues many full pipeline forwards. Counting for `G=2`, worst case (both GRPO and SDPO active):

| Phase | Forward passes | What runs |
|-------|---------------|-----------|
| Generate (×G) | G = 2 | `forward()` for prompt + incremental decode |
| Old-policy for correct | 1 | `forward()` on padded correct batch |
| Teacher generate (SDPO) | ~1 generate | `generate()` for feedback-conditioned completions |
| Student forward (SDPO) | 1 | `forward()` on teacher outputs |
| Teacher forward (SDPO) | 1 | `forward()` on teacher outputs (old policy) |

That's **6 pipeline forwards** per `train_step`, each running 12 + T×11 + 12 = ~46 layers (T~2). With `gradient_accumulation_steps=4`, that's **24 pipeline forwards** per optimizer step.

Additionally, `generate()` calls `lm()` (the full 35-layer HF model) incrementally for each token. With `max_response_tokens=64`: ~G × 64 × 35 = **4,480 layer evaluations** for generation alone.

**Evidence**: The full orchestration spanning `scripts/train_srpo.py:842-940`.

**Recommendation**: This is inherent to recurrent depth + GRPO training. Potential optimizations:
- Skip old-policy forward when all correct rewards are identical (no GRPO signal)
- Merge teacher forward + student forward into a single pass when possible
- Reduce `max_response_tokens` further if code problems are simple enough

---

### 3.3 Memory footprint for `group_size=2` with 4.6B model

- **File:Line**: `src/parcae/model.py` architecture, `scripts/train_srpo.py:495-530` (_build_model)
- **Severity**: **INFO**

**Rough estimate for single GPU (RTX 5090, 32GB)**:

| Component | Size |
|-----------|------|
| Model weights (bf16) | ~9.2 GB |
| Optimizer states (AdamW for 56K) | ~0.4 MB (negligible) |
| KV cache during generation | ~2-4 GB (depending on context length) |
| Training activations (B=2, T~2, L~576) | ~2-4 GB |
| **Total estimate** | **~14-18 GB** |

This fits comfortably on 32GB with headroom for DDP overhead. The README's mention of 2× RTX 5090 provides safety margin but a single GPU should work for current settings.

**Evidence**: 
- 4.6B params × 2 bytes (bf16) = 9.2 GB
- 56,321 trainable params × (2 + 4 + 4) bytes ≈ 0.4 MB (fp32 param + momentum + variance)
- Activation memory: ~O(B × T × L × dim × n_layers × 2 bytes)

**Recommendation**: Monitor actual memory with `torch.cuda.memory_summary()`. If hitting OOM, reduce `micro_batch_size` to 1 or `max_response_tokens` to 32.

---

### 3.4 `generate()` uses per-token autoregression via HF's `_language_model` — not recurrent

- **File:Line**: `src/parcae/model.py:528-583`
- **Severity**: **WARNING**

**Finding**: During generation, after the prompt forward (which runs the full recurrent pipeline), each incremental token is produced by calling `self._language_model(input_ids=next_tok, ...)` — the **original 35-layer HF model**, NOT the recurrent-depth pipeline. Injection is applied manually after each step:

```python
h = inj(h, e, h)  # injection applied to hidden states
```

This means generation uses 35-layer forward passes per token (instead of potentially using the recurrent block). The injection is applied as a post-hoc correction to the HF model's output.

**Implications**:
- Token generation cost is ~35 layers/token, not 12 + T×11 + 12 = ~46 layers
- But the HF model forward uses KV caching, which the recurrent pipeline doesn't support natively
- Mixing HF's KV cache with our injection is a pragmatic engineering choice, not architecturally pure

**Recommendation**: This is a reasonable trade-off for now. If recurrent-depth KV caching becomes important (for longer generations), implement KV cache support directly in `_run_block`.

---

### 3.5 `_bptt_depth` detach mechanism — correct but has zero effect at T ≤ bptt_depth

- **File:Line**: `src/parcae/model.py:308-314`
- **Severity**: **INFO**

**Finding**: With `poisson_mean=2` and `bptt_ratio=0.5`:
```
T ~ Poisson(2) → typically 0–4 (clamped to [1, 8])
T_bwd = ceil(T * 0.5) → typically 1–2
detach_before = T - T_bwd → 0–2 iterations forward-only
```

At T=1: `detach_before = 1 - 1 = 0` → no detaching, full gradient flow
At T=2: `detach_before = 2 - 1 = 1` → first iteration detached
At T=3: `detach_before = 3 - 2 = 1` → first iteration detached
At T=4: `detach_before = 4 - 2 = 2` → first 2 iterations detached

The Truncated BPTT is working correctly but for low T values, most of the computation receives gradient. This is fine — Truncated BPTT is a memory optimization, and at low T there's no memory pressure.

**Recommendation**: No change needed. TBPBT becomes more important if `poisson_mean` is increased.

---

## 4. Summary

### Blockers

| # | Issue | Location |
|---|-------|----------|
| 1 | `group_size=2` makes GRPO advantage effectively binary (3 states). GRPO needs G≥4, ideally 8+. Readme already acknowledges GRPO rarely activates. | `train_srpo.py:61` |

### Warnings

| # | Issue | Location |
|---|-------|----------|
| 2 | `gen_temperature=0.8` is too low for RL exploration. Combined with G=2, this nearly guarantees identical completions. | `train_srpo.py:63` |
| 3 | `max_response_tokens=64` limits complex code/assembly generation. | `train_srpo.py:62` |
| 4 | Python for-loop in `_recurrent_loop` prevents `torch.compile` but overhead is negligible at T ≤ 8. | `model.py:310` |
| 5 | Very high forward-pass count per train_step (6+ pipeline forwards × 4 GA steps). | `train_srpo.py:840-940` |
| 6 | `generate()` uses HF model directly (not recurrent pipeline) for incremental decoding. KV cache not recurrent-depth-aware. | `model.py:528-583` |

### Info

| # | Issue | Location |
|---|-------|----------|
| 7 | `lr=1e-4` is conservative for 56K params; 2-5e-4 may converge faster. | `train_srpo.py:70` |
| 8 | GRPO NaN handling is correct (1e-8 epsilon). B<2 guard is defensive and never reached. Old-policy forward wasted when all rewards equal. | `train_srpo.py:371-380` |
| 9 | `B_raw` initialized at 0.01 → effective B_discrete ~5×10⁻⁴. Injection effectively zero at start. | `injection.py:57,84` |
| 10 | `(A-I)·h` residual cancellation is mathematically sound. | `injection.py:99-101` |
| 11 | `ρ(A) < 1` guaranteed by construction via `exp(-exp(...))`. | `injection.py:70-78` |
| 12 | Memory fits comfortably on 32GB (~14-18 GB estimated). | `model.py` |
| 13 | `loop_embedding_dim`: TrainConfig=128, RecurrentDepthConfig=256. Training overrides. Not a bug. | `train_srpo.py:54`, `model.py:101` |
| 14 | README says `find_unused_parameters=False`, code uses `True`. Code is safer. | `train_srpo.py:651`, `README.md:49` |
| 15 | `bptt_ratio=0.5` with low `poisson_mean=2` means most iterations get full gradient. Fine for current T range. | `model.py:308-314` |

---

## 5. Recommended Priority Fixes

1. **Increase `group_size` to at least 4** (`train_srpo.py:61`) — this is the single highest-impact change. Without it, GRPO is effectively binary and the model may never produce >1 correct completion per prompt.

2. **Increase `gen_temperature` to 1.0–1.2** (`train_srpo.py:63`) — for better exploration diversity during generation.

3. **Add early return when `rewards.std() < 1e-7`** in `train_step` — avoids wasted old-policy forward when all rewards are identical (the common case right now).
