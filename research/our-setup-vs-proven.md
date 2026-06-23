# GRPO Training: Our Setup vs. Proven Implementations

## 1. TRL's GRPOTrainer (huggingface/trl, main branch)

### 1.1 Advantage Normalization (PER-PROMPT vs GLOBAL)

**TRL** (`trl/trainer/grpo_trainer.py:2273-2275`):
```python
# sum_then_normalize (default)
mean_grouped_rewards = torch.nanmean(rewards.view(-1, num_generations), dim=1)  # per-prompt mean
mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
# Then: advantages = rewards - mean_grouped_rewards
# Optional: advantages = advantages / (std_rewards + 1e-4)  where std computed PER GROUP
```

→ **Per-prompt group-relative**: Each prompt's G completions are normalized AGAINST EACH OTHER independently.
Prompts don't compete across the batch. This is what the DeepSeekMath paper specifies.

**Our code** (`scripts/train_srpo.py:380-383`):
```python
# grpo_loss() — GLOBAL normalization
mu = rewards.mean()       # across ALL completions from ALL prompts
sigma = rewards.std() + 1e-8
A = (rewards - mu) / sigma
```

→ **BLOCKER**: We normalize rewards **globally across all prompts in the batch**, not per-prompt.
This means a trivially-easy prompt (all correct) dilutes the advantage of a hard prompt
(mostly wrong with one correct). TRL computes mean/std per prompt group; we don't.

**Consequence**: With global normalization, a correct answer on an easy prompt gets a LOWER advantage
than the same correct answer would get under per-prompt normalization (because the easy-prompt
mean is higher). Conversely, a correct answer on a hard prompt doesn't get properly rewarded.
This fundamentally changes the RL signal.

**Fix needed**: Group completions by prompt index, normalize within each group separately,
then concatenate.

### 1.2 KL Penalty

**TRL** (`grpo_trainer.py:2663-2667`):
```python
per_token_kl = (
    torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
)
# Then: per_token_loss = per_token_loss + self.beta * per_token_kl
```

This is the **unbiased reverse KL estimator**:
`KL[π_θ || π_ref] = exp(log π_ref - log π_θ) - (log π_ref - log π_θ) - 1`

It's an *estimator* of D_KL(π_θ || π_ref), not the true KL. It is exact in expectation
(`E_x~π_θ[...]`) but per-sample biased. Unsloth's blog confirms: "The reference GRPO
implementation uses the reverse KL divergence."

**Our code** (`scripts/train_srpo.py:27`):
```python
kl_beta: float = 0.0   # KL DISABLED
```

→ **OK (intentional)**: We don't use a KL penalty at all. DeepSeek-R1-Zero also used β=0
for pure-RL reasoning training. However, TRL's default is β=0.0 too in recent versions,
so this is not a divergence. We use β=0 because we train a tiny injection+Lora adapter
on a frozen backbone, so catastrophic forgetting is minimal.

**DAPO (TRL's default loss_type)** also uses β=0. So KL=0 is standard for many
setups. Not a blocker.

**WARNING for future**: If you ever unfreeze backbone layers or increase trainable params,
add KL regularization against a reference model. TRL's `beta=0.04` was the DeepSeekMath
default for full-model training.

### 1.3 Batching Completions (Importance Sampling Level)

**TRL** (`grpo_config.py:219`):
```python
importance_sampling_level: str = "token"  # default: per-token ratios
```
And `loss_type` defaults to `"dapo"` (not `"grpo"`).

With `importance_sampling_level="token"`:
- Log-ratio `coef_1 = exp(log π_θ - log π_old)` has shape `(B, T)` — one ratio per token
- Each token gets its own importance weight
- Loss is averaged over tokens, not sequences

**Our code** (`scripts/train_srpo.py:380-383`):
```python
# Sequence-level importance ratio (length-normalized)
seq_lp = (log_probs * response_mask).sum(dim=-1) / n_tokens
seq_lp_old = (log_probs_old * response_mask).sum(dim=-1) / n_tokens
rho = torch.exp(seq_lp - seq_lp_old)  # ONE scalar per sequence
```

→ **WARNING**: We use GSPO-style sequence-level importance ratios. This is **intentional**
(we label our loss as GSPO) and the GSPO paper shows this can be more stable for
sequence-level rewards. However, TRL's default is token-level ratios (`"grpo"` loss type)
or `"dapo"` which also token-level but with different loss aggregation.

**Key difference**: In TRL's `"grpo"` loss, the importance ratio is per-token:
```
per_token_loss1 = coef_1 * advantages    # coef_1 is (B, T)
per_token_loss2 = coef_2 * advantages    # coef_2 is (B, T) after clipping
per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
```
Then averaged per-sequence: `loss = (per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)`

We compute: one scalar ratio per sequence, then expand to token level.

**Risk**: Sequence-level ratios with length normalization interact differently with
the gradient. Tokens that change little (ratio≈1) but length changes (ratio≠1) get
the same weight as tokens where the distribution actually shifted. This can introduce
noise. The GSPO paper validates this works, but TRL's token-level is more
fine-grained and the standard default.

### 1.4 Old-Policy Log-Prob Computation

**TRL** (`grpo_trainer.py:2628-2631`):
```python
old_per_token_logps = inputs.get("old_per_token_logps")
old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
```

→ When `num_iterations == 1` (single update per generation batch), TRL uses
`per_token_logps.detach()` as the old-policy log-probs. This means: **the model
that generated the completion IS the old policy**, and no separate forward pass
is needed. Only when `num_iterations > 1` does TRL compute separate old-policy
log-probs from the pre-update model.

**Our code** (`scripts/train_srpo.py:460-481`):
```python
# We ALWAYS do a separate old-policy forward with module swaps
with self._old_policy_ctx():
    with torch.no_grad():
        logits_old = self._model_unwrapped.forward(
            input_ids=batch_ids, n_loops=T, return_logits=True)
log_probs_old = F.log_softmax(logits_old.float(), dim=-1).to(logits_old.dtype)
```

→ **WARNING**: We do a separate forward pass with swapped old-policy modules.
This is correct for the `num_iterations=1` case because we DO update between
generation and training (gradient accumulation → optimizer step → then generate
again in the next step). So our old-policy snapshot is genuinely older. TRL's
optimization for `old_per_token_logps = per_token_logps.detach()` only works
because they batch generation and training within the same micro-step.

However, our approach has a **correctness concern**: The log-probs cached from
`_generate()` use the CURRENT policy (post-last-update), but we then swap to
old-policy modules for the separate forward. So we have:
- Current-policy log-probs: from `_generate()` (correct)
- Old-policy log-probs: from separate `forward()` with swapped modules (correct)

But the `coef_1 = exp(current_lp - old_lp)` is correct. **OK for our single-iteration
setup.**

**Efficiency**: TRL's approach is much more efficient when `num_iterations=1` because
it skips the separate old-policy forward. We always pay that cost. Since we only train
a small adapter, this cost is modest. **OK.**

### 1.5 Reference Model Handling

**TRL**: Maintains a separate reference model copy. When `beta != 0`, computes
`ref_per_token_logps` by forwarding the reference model on all completions.
The reference model is loaded as a separate `PreTrainedModel` instance.

**Our code**: We don't have a reference model at all (β=0). The old-policy modules
serve a different purpose (importance sampling), not KL regularization.

→ **OK**: β=0 is a valid choice, used by DeepSeek-R1-Zero and DAPO.

### 1.6 Epsilon / Clipping

**TRL defaults** (`grpo_config.py`):
```python
epsilon: float = 0.2          # lower bound
epsilon_high: Optional[float] = None  # defaults to epsilon if not set
loss_type: str = "dapo"       # different normalization than standard GRPO
```

**Our code** (`scripts/train_srpo.py:50-52`):
```python
clip_epsilon: float = 0.2
clip_epsilon_high: float = 0.28  # GSPO Clip-Higher (asymmetric)
```

→ **OK (intentional)**: We use asymmetric clipping (GSPO), which is a valid variant.
TRL supports this via `epsilon_high` parameter. Our 0.2/0.28 values match DAPO/GSPO
recommendations.

### 1.7 Loss Aggregation

**TRL** (`grpo_trainer.py:2715-2717, for loss_type="grpo"`):
```python
loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
```

→ Per-sequence normalized, then averaged across all sequences.

**Our code** (`scripts/train_srpo.py:392-393`):
```python
per_token = surr.unsqueeze(-1) * response_mask  # (B, L)
loss = -per_token.sum() / n_tokens.sum().clamp(min=1)
```

→ **WARNING**: We sum ALL tokens across ALL sequences and divide by total token count.
This is more like TRL's `"bnpo"` loss type (local batch normalization) and can
introduce length bias — longer sequences get more weight in the loss.

TRL's `"dapo"` (default) normalizes by *global accumulated batch token count* to
eliminate length bias. Our aggregation over-weights longer sequences within the
correct-samples group.

**Consequence**: A correct 200-token completion contributes 2× the loss of a
correct 100-token completion. This can bias the model toward verbose answers.

### 1.8 DIVERGENCE SUMMARY: TRL

| Aspect | TRL | Our Code | Severity |
|--------|-----|----------|----------|
| **Advantage normalization** | Per-prompt group (`rewards.view(-1, G)`) | **Global (all prompts)** | **BLOCKER** |
| KL penalty | β=0.0 default (same) | β=0.0 | OK |
| Importance ratio level | Token-level (default) | Sequence-level (GSPO) | WARNING |
| Old-policy log-probs | `per_token_logps.detach()` optim | Separate forward with module swap | OK |
| Reference model | Separate model (when β≠0) | None (β=0) | OK |
| Epsilon | 0.2 symmetric (default) | 0.2/0.28 asymmetric | OK |
| Loss aggregation | Per-sequence average | Token-sum/total-tokens | WARNING |

---

## 2. Unsloth's GRPO Notebook Training Code

Unsloth effectively wraps TRL's GRPOTrainer with memory optimizations. They don't
change the algorithm — they change the **implementation efficiency**.

### 2.1 Reverse KL Formula

**Unsloth**: Uses the same unbiased reverse KL estimator as TRL:
```python
per_token_kl = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1
```

**Our code**: β=0, so no KL at all.

→ **OK**: We don't need KL with frozen backbone + tiny adapter.

### 2.2 Linear Cross Entropy / Chunked Loss (Memory Efficiency)

**Unsloth blog** (unsloth.ai/blog/grpo):
> "We got inspired from Horace He's linear cross entropy implementation, and managed
> to make it work for GRPO! We actually found a few surprising points: The reference
> GRPO implementation uses the reverse KL divergence."

The key Unsloth efficiency techniques:

1. **Chunked cross-entropy**: Instead of computing `lm_head(x) → logits (B, T, V)` then
   cross-entropy, they fuse the linear layer + log-softmax into a chunked kernel. This
   avoids materializing the full `(B, T, V)` logits tensor, which for a 256K vocab ×
   batch × seqlen is enormous. Memory drops from O(BTV) to O(BT).

2. **Chunked GRPO loss**: Same idea — compute loss in chunks rather than materializing
   all logits at once.

3. **torch.compile**: The chunked kernels are compiled for speed.

**Our code** (`model.py:341-342`):
```python
h = self.norm(h)
logits = self.lm_head(h)  # Full (B, T, V) tensor!
```

→ **WARNING**: We materialize the full `(B, T, 256000)` logits tensor in `forward()`.
For `B=12` (6 prompts × 2 correct samples at G=6), `T=256` (prompt+response),
this is `12 × 256 × 256000 × 2 bytes (bf16) ≈ 1.5 GB` just for logits.
This doubles with teacher logits in SDPO. On a 32GB RTX 5090 this is tight but
manageable at our scale. However:

- In SDPO, we have BOTH `stu_logits` AND `tea_logits` — two full (B, T, V) tensors.
- With G=6, max_response_tokens=128, and B=2 prompts, the worst-case is ~12 completions
  of ~200 tokens each = 2400 tokens × 256K vocab ≈ 1.2 GB per logits tensor.

**What Unsloth does**: Never materializes full logits. Uses fused `linear_cross_entropy()`
which computes `log_softmax(Wx)` without storing the intermediate `Wx`.

**Our mitigation**: We limit `max_response_tokens=128` and use bf16 autocast.
This is OK for our scale. But for longer sequences or larger batches,
we'd want chunked cross-entropy.

### 2.3 vLLM Integration

**Unsloth/TRL**: Can use vLLM for generation, separate from training. This decouples
generation memory from training memory and allows continuous batching.

**Our code**: Generation runs in-process via `model.generate()`, sharing GPU memory
with training. With 32GB GPUs and a ~7.4GB model (E2B in bf16), we have room.

→ **OK**: At our scale, in-process generation works. vLLM would help for larger models
or longer sequences.

### 2.4 Liger Kernel

**Unsloth**: Also supports Liger's fused GRPO loss kernel for further speedup.

**Our code**: No fused kernels.

→ **OK**: Not needed at our scale.

### 2.5 DIVERGENCE SUMMARY: Unsloth

| Aspect | Unsloth | Our Code | Severity |
|--------|---------|----------|----------|
| **Chunked logits** | Fused linear+CE (no full logits) | Full (B,T,V) logits materialized | WARNING |
| vLLM generation | Supported (decoupled) | In-process generation | OK |
| Fused GRPO loss | Liger kernel | Manual PyTorch | OK |
| Reverse KL | Unbiased estimator | Not needed (β=0) | OK |

The main thing to adopt from Unsloth for memory efficiency: **use
`F.linear_cross_entropy()` (PyTorch 2.6+) or write a chunked log_prob
computation for the old-policy forward**. This would avoid materializing
the full logits tensor in the old-policy forward pass.

In our `_old_policy_ctx()` forward:
```python
logits_old = self._model_unwrapped.forward(input_ids=batch_ids, n_loops=T, return_logits=True)
```
We could instead compute log-probs via chunked forward or direct
`F.linear_cross_entropy` if we refactor to use it. However, PyTorch's
`linear_cross_entropy` only works for the standard cross-entropy
(single label per position), not for our general log-prob extraction
(we need log_probs at ALL positions, not just CE loss). So this would
require custom chunking.

---

## 3. DeepSeekMath Paper (Section 2.3 — GRPO Algorithm)

### 3.1 The Advantage Formula

**DeepSeekMath paper (arxiv 2402.03300, Eq. 2)**:
```
Â_i = (r_i - mean(r_group)) / std(r_group)
```
where `r_group = {r_1, r_2, ..., r_G}` for a single question q.

→ **Per-prompt group normalization**: ONLY completions for the SAME prompt are
compared. This is the defining feature of GRPO.

**TRL's implementation**: Matches exactly: `rewards.view(-1, num_generations)` → normalize along dim=1.

**Our code**: Normalizes across ALL prompts. **Does NOT match the paper.**

This is the same issue as Section 1.1 above — **BLOCKER**.

### 3.2 The GRPO Objective

**DeepSeekMath paper (Eq. 3)**:
```
J_GRPO(θ) = E_{q~P(Q), {o_i}~π_old}[
    1/G Σ_{i=1}^G min(
        π_θ(o_i|q) / π_old(o_i|q) · Â_i,
        clip(π_θ(o_i|q) / π_old(o_i|q), 1-ε, 1+ε) · Â_i
    ) - β·D_KL(π_θ || π_ref)
]
```

Key aspects:
1. **The probability ratio `π_θ/π_old` is PER-SEQUENCE**, not per-token. The DeepSeekMath
   paper uses sequence-level probabilities: `π_θ(o_i|q) = Π_t π_θ(o_{i,t} | q, o_{i,<t})`.
   In log-space: `log π_θ(o_i|q) = Σ_t log π_θ(o_{i,t} | q, o_{i,<t})`.

2. **The advantage `Â_i` is a **scalar** per sequence**, computed from per-prompt
   group normalization.

3. **KL penalty is subtracted**, not added to the advantage.

**Our code**: We use sequence-level log-prob ratios (matching the paper) but with
length normalization:
```python
seq_lp = (log_probs * response_mask).sum(dim=-1) / n_tokens    # mean over tokens
```
The paper uses **sum** over tokens (product of probabilities), not mean.

**WARNING**: Length-normalized log-probs change the importance ratio scale:
- Paper: `ratio = exp(Σ_t lp_θ(t) - Σ_t lp_old(t))`
- Ours: `ratio = exp(mean(lp_θ) - mean(lp_old)) = exp((Σ lp_θ - Σ lp_old) / L)`

For a 100-token sequence with log-ratio 0.01 per token:
- Paper ratio: `exp(100 × 0.01) = exp(1.0) ≈ 2.72`
- Our ratio: `exp(0.01) ≈ 1.01`

This **dramatically understates** the true importance ratio for long sequences,
making the clip nearly useless (ratio almost never exceeds 1±ε). For a 128-token
sequence, our ratio needs `|seq_lp - seq_lp_old| > 0.0016` to exceed the clip
(at ε=0.2), while the paper's formula would exceed it at `|lp_diff| > 0.0018`
per token cumulatively.

**This is the GSPO formulation** — GSPO (arxiv 2507.18071) explicitly uses
sequence-level log-prob means. So this is **OK (intentional GSPO)** but diverges
from the DeepSeekMath paper's "product of token probabilities" approach.

### 3.3 ε Clipping

**DeepSeekMath paper**: Uses symmetric clipping with ε=0.2. No asymmetric Clip-Higher.

**Our code**: Asymmetric: ε_low=0.2, ε_high=0.28 (GSPO Clip-Higher).

→ **OK (intentional GSPO variant)**. The GSPO paper shows asymmetric clipping
can improve training for sequence-level rewards.

### 3.4 KL Penalty in the Paper

**DeepSeekMath paper**: Uses β > 0 (value not specified in section 2.3 but the
algorithm includes D_KL). Later DeepSeek-R1-Zero used β=0 for pure RL.

**Our code**: β=0.

→ **OK**: Pure RL without KL penalty is valid for verifiable reward domains
when training small adapters on frozen backbones.

### 3.5 DIVERGENCE SUMMARY: DeepSeekMath

| Aspect | DeepSeekMath Paper | Our Code | Severity |
|--------|-------------------|----------|----------|
| **Advantage normalization** | Per-prompt group: `(r_i - mean)/std` | **Global batch: `(r - mean)/std`** | **BLOCKER** |
| Probability ratio | Sum of token log-probs | Mean of token log-probs (GSPO) | WARNING |
| Epsilon clipping | Symmetric ε=0.2 | Asymmetric 0.2/0.28 | OK |
| KL penalty | β > 0 | β = 0 | OK |

---

## 4. Additional Divergences Found

### 4.1 No Per-Prompt Grouping in Data Flow

**Issue**: Our `train_step` receives a batch of 2 prompts. For each prompt, we generate
G=6 completions → 12 total completions. But then all 12 completions are pooled into
a single flat list. The `grpo_loss()` function receives ALL correct completions
(across potentially different problems) as a single flat batch and computes a global
mean/std.

**TRL**: Keeps track of `num_generations` per prompt, reshapes into
`(num_prompts, num_generations)`, normalizes along dim=1 (per-prompt).

**Fix**: In `grpo_loss()`, reshape rewards into `(B_per_prompt, G_per_prompt)` and
normalize per row.

### 4.2 SDPO Branch Uses Old Policy, Not Reference Model

**Issue**: Our SDPO branch uses old-policy modules for both:
1. Teacher generation (correct — we want the pre-update model to generate corrections)
2. Teacher log-probs in KL computation

This is subtle: In the DeepSeekMath paper, the KL is against a **reference model**
(frozen snapshot of the initial model). We don't have a reference model — we use the
**old policy** (pre-update snapshot) as the teacher. This means the teacher keeps
moving (it's updated every step via `_snapshot_old_policy()`), which means the SDPO
distillation target drifts.

**OK for our use case**: SDPO is self-distillation from a feedback-conditioned model,
not KL-penalty against a frozen reference. The teacher being the old policy is by design.

### 4.3 BPTT / Truncated Backprop

**Issue**: We set `_bptt_depth = T_bwd = ceil(T * 0.5)`, meaning gradients only flow
through the last half of recurrent loops. This is a Parcae-specific optimization
that has no GRPO equivalent.

→ **OK (Parcae design)**. Not a GRPO divergence — it's our recurrent architecture's
memory-saving technique.

### 4.4 No Entropy Masking

**TRL**: Supports `top_entropy_quantile < 1.0` to mask out low-entropy tokens
(reduce gradient noise from confident-but-wrong tokens).

**Our code**: No entropy masking.

→ **OK**: Optional feature that helps stability with small group sizes.

---

## 5. Prioritized Action Items

### BLOCKER: Fix Advantage Normalization (Section 1.1, 3.1)

**What**: Normalize advantages per-prompt, not globally.

**Where**: `scripts/train_srpo.py`, `grpo_loss()` function, line ~380.

**Current**:
```python
mu = rewards.mean()        # global
sigma = rewards.std() + 1e-8
A = (rewards - mu) / sigma
```

**Should be**:
```python
# Reshape into (n_prompts, G) — need to pass prompt grouping info
rewards_grouped = rewards.view(-1, G_per_prompt)  # (n_prompts, G)
mu = rewards_grouped.mean(dim=1, keepdim=True)    # per-prompt mean
sigma = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
A_grouped = (rewards_grouped - mu) / sigma
A = A_grouped.view(-1)  # flatten back
```

This requires passing the per-prompt group size into `grpo_loss()` (or deriving it
from rewards shape + metadata). The function signature needs to know how many
prompts are in the batch.

**Same fix needed for log_probs grouping** — the sequence-level log-probs and
log_probs_old must also be grouped per-prompt so the importance ratio is
computed per-prompt.

### WARNING: Consider Per-Prompt Importance Ratios

Currently the importance ratio `rho = exp(seq_lp - seq_lp_old)` is computed
globally. If we fix advantage normalization to per-prompt, the importance
ratios should also be computed per-prompt to stay consistent. The DeepSeekMath
formula computes `π_θ(o_i|q) / π_old(o_i|q)` within each prompt group.

### WARNING: Check Response Token Count Handling

Our length normalization (`/n_tokens`) for the log-prob mean is by design (GSPO).
But make sure that:
1. `n_tokens` counts only response tokens (not prompt tokens) — this appears correct.
2. The `response_mask` is correctly set to 1 for response tokens and 0 for prompt tokens.
3. When computing old-policy log-probs, we only index into the generated positions.

### Memory: Consider Chunked Log-Prob Extraction

For the old-policy forward, instead of:
```python
logits_old = self._model_unwrapped.forward(... , return_logits=True)
log_probs_old = F.log_softmax(logits_old.float(), dim=-1)
lp_old[j, PL:L] = log_probs_old[j, gen_pos - 1, batch_ids[j, gen_pos]]
```

We could compute log-probs directly without materializing full logits:
```python
# For each position, compute lm_head forward on the hidden state at that position
# and immediately do log_softmax + gather. This is O(B * T * V) still, but peak
# memory is O(B * V) instead of O(B * T * V).
```

However, this is a micro-optimization for our scale. Not urgent.

---

## 6. Summary Table

| # | Divergence | Source File:Line | Severity | Fix |
|---|-----------|-----------------|----------|-----|
| 1 | Advantage normalization is GLOBAL, not per-prompt group | `train_srpo.py:380-383` | **BLOCKER** | Group rewards by prompt, normalize per-group |
| 2 | Log-prob ratio uses token-mean (GSPO), paper uses token-sum | `train_srpo.py:377-379` | WARNING | OK if intentional GSPO; document clearly |
| 3 | Loss aggregation: token-sum/total-tokens (over-weights long seqs) | `train_srpo.py:392-393` | WARNING | Consider per-sequence average like TRL |
| 4 | Full logits materialization in old-policy forward | `train_srpo.py:478-479` | WARNING | Consider chunked log-prob extraction |
| 5 | β=0 (no KL penalty) | `train_srpo.py:52` | OK | Valid for frozen-backbone RL |
| 6 | Asymmetric epsilon clipping (GSPO) | `train_srpo.py:51-52` | OK | Valid GSPO variant |
| 7 | Sequence-level importance (GSPO, not token-level) | `train_srpo.py:381` | OK | Valid GSPO formulation |
| 8 | BPTT truncation (Parcae-specific) | `train_srpo.py:412` | OK | Architecture-specific optimization |
| 9 | No vLLM / fused kernels | N/A | OK | Not needed at our scale |
| 10 | Teacher uses old-policy, not reference model | `train_srpo.py:540-544` | OK | SDPO design choice |
