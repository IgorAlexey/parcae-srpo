# Training Code Correctness Audit

**Date:** 2026-06-23
**Files audited:** `scripts/train_srpo.py`, `src/parcae/model.py`, `src/parcae/injection.py`

---

## 1. SDPO Teacher Generation (feedback-conditioned prompt & old-policy usage)

### Finding: CORRECT — teacher uses old-policy + feedback-conditioned prompt

**Evidence:**

- `build_feedback()` at `train_srpo.py:452-473` constructs the feedback string from failed code + error + optional correct demo.
- `train_srpo.py:837-839` builds the teacher prompt by concatenating feedback with the original problem:
  ```python
  teacher_prompts = [
      f + "\n\nNow write the corrected code for:\n" + prompts[c["batch_idx"]]
      for c, f in [(c, c["sdpo_feedback"]) for c in failed]
  ]
  ```
- `train_srpo.py:853-860` wraps teacher generation in `_old_policy_ctx()`:
  ```python
  with self._old_policy_ctx():
      with torch.no_grad():
          teacher_gen = self._model_unwrapped.generate(...)
  ```
- `_old_policy_ctx()` at `train_srpo.py:908-925` swaps `m.injection` → `self.injection_old` and `m.depth_lora` → `self.depth_lora_old` on the unwrapped model. Guarantees restoration via `try/finally`.
- Teacher logits for KL computation also use `_old_policy_ctx()` at `train_srpo.py:869-872`.
- Student logits at `train_srpo.py:866-868` use `self.model.forward(...)` (DDP wrapper, grad enabled, current policy).

**Verdict:** ✅ Correct. The teacher generation and teacher logit computation both use old-policy weights. The student forward uses current-policy weights with grad. The feedback string is correctly incorporated into the prompt.

---

## 2. GRPO Loss Gradient Flow & Old-Policy Snapshot

### Finding: BUG — GRPO branch contributes zero gradient

**Evidence:**

**Bug A: Current log-probs are detached (`@torch.no_grad()`).**

- `_generate()` is decorated with `@torch.no_grad()` at `train_srpo.py:685`.
- The cached `token_lp` values in each completion dict come from this no-grad context.
- `train_srpo.py:812-816` populates the `lp` tensor from cached values:
  ```python
  lp = torch.zeros(len(correct), L_max, device=self.device)
  for j, c in enumerate(correct):
      LP[j, PL:L] = c["token_lp"][:L - PL]
  ```
- `train_srpo.py:827` passes `lp` (detached) as the first argument to `grpo_loss()`.
- The `grpo_loss()` at `train_srpo.py:326-358` computes `seq_lp` from `lp` and `rho = exp(seq_lp - seq_lp_old)`. Since `lp` is detached, `rho` has no gradient connection to model parameters.
- The old-policy log-probs (`lp_old`) are also computed under `torch.no_grad()` at `train_srpo.py:823-826`.

**Result:** `grpo_loss` output has `requires_grad=False` (it's a detached scalar). The GRPO branch gradient is zero.

**Bug B: Old-policy == current-policy (importance ratio always 1.0).**

- `_snapshot_old_policy()` at `train_srpo.py:901-905` copies current weights immediately before `_generate()`.
- Both `_generate()` (current policy) and the old-policy forward use the SAME weights (no `optimizer.step()` has occurred between snapshot and generation).
- Therefore `seq_lp == seq_lp_old`, so `rho = exp(0) = 1.0` always.
- The clipping in `grpo_loss()` is dead code: `rho_clip = clamp(1.0, 1-ε, 1+ε_high) = 1.0`, so `surr = min(1.0*A, 1.0*A) = A`.
- Even if gradient DID flow, the policy gradient would be `-A` with no clipping effect.

**Verdict:** ❌ **Bug.** The GRPO branch is dead weight. Only the SDPO branch actually trains the model. All policy improvement comes from self-distillation, not from the GRPO policy gradient.

**Fix required:** Either (a) remove `@torch.no_grad()` from `_generate()` and recompute log-probs under the GRPO branch with grad enabled, or (b) run a second forward pass on correct completions with grad for GRPO. The old-policy snapshot must capture the policy BEFORE the current training step (i.e., the policy that generated the data), not the current policy. With single-epoch GRPO, the ratio will still be 1.0 on the first update; multi-epoch inner loop or a separate rollout phase is needed.

---

### Finding: CORRECT — old-policy snapshot mechanism (`swap_module_params`)

**Evidence:**

- `_snapshot_old_policy()` at `train_srpo.py:901-905` uses `load_state_dict()` to copy parameters:
  ```python
  self.injection_old.load_state_dict(m.injection.state_dict())
  if self.depth_lora_old is not None and m.depth_lora is not None:
      self.depth_lora_old.load_state_dict(m.depth_lora.state_dict())
  ```
- `_old_policy_ctx()` at `train_srpo.py:908-925` swaps Python object references (not tensors), which avoids DDP inplace-op tracking.
- The DDP wrapper only wraps `self.model`; operations on `self._model_unwrapped` are invisible to DDP's reducer hooks.
- Old-policy modules are deep-copied at init (`train_srpo.py:632-641`), with `requires_grad=False`.
- Restoration is guaranteed by `try/finally`.

**Verdict:** ✅ Correct. The module-reference swap pattern is DDP-safe and correctly copies all trainable parameters. The old-policy modules hold the exact weights at snapshot time.

---

## 3. DDP Setup: Gradient Sync & DistributedSampler

### Finding: CORRECT — DDP setup is properly configured

**Evidence:**

- Process group init at `train_srpo.py:489`: `dist.init_process_group(backend="nccl")` with `torch.cuda.set_device(local_rank)`.
- DDP wrapper at `train_srpo.py:647-651`:
  ```python
  self.model = DDP(self.model, device_ids=[self.local_rank],
                    find_unused_parameters=True)
  ```
- `find_unused_parameters=True` is correct because the frozen backbone parameters receive no gradients and would otherwise trigger DDP errors.
- `DistributedSampler` at `train_srpo.py:498-504` with `shuffle=True` and `seed=cfg.seed` ensures deterministic shuffling across ranks.
- `set_epoch(epoch)` at `train_srpo.py:958` is called on `StopIteration` when the dataloader exhausts:
  ```python
  except StopIteration:
      epoch += 1
      if self._sampler is not None:
          self._sampler.set_epoch(epoch)
      data_iter = iter(self.dataloader)
  ```
- Gradient accumulation with `no_sync()` at `train_srpo.py:952`:
  ```python
  sync_ctx = self.model.no_sync() if self.world_size > 1 and not is_last \
             else contextlib.nullcontext()
  ```
  All micro-batches except the last skip gradient all-reduce, then the last micro-batch syncs. This is the standard pattern.

**Verdict:** ✅ Correct. Gradients synchronize properly across GPUs. Sampler shuffling works correctly across epochs.

---

## 4. Optimizer Step: Gradient Clipping & LR Scheduler

### Finding: CORRECT — gradient clipping is properly wired

**Evidence:**

- `train_srpo.py:966`: `self.scaler.unscale_(self.optimizer)` — unscales gradients before clipping (required for `GradScaler`).
- `train_srpo.py:967`: `torch.nn.utils.clip_grad_norm_(self.trainable_params(), cfg.max_grad_norm)` — clips gradients of trainable params with `max_grad_norm=1.0`.
- `train_srpo.py:968-969`: `self.scaler.step(self.optimizer)` + `self.scaler.update()` — standard AMP pattern.
- `train_srpo.py:970`: `self.optimizer.zero_grad()` — zeros gradients after step.

**Verdict:** ✅ Correct. The unscale→clip→step→update→zero_grad sequence is correct for AMP training.

### Finding: BUG — No LR scheduler is wired

**Evidence:**

- `_build_optimizer()` at `train_srpo.py:676-680` creates `AdamW` with constant `lr=cfg.learning_rate`.
- Grep for `scheduler`, `lr_scheduler`, `Cosine`, `warmup` in `train_srpo.py` returns **zero matches**.
- The `_save()` method at `train_srpo.py:1018-1026` saves only `optimizer.state_dict()` — no scheduler state dict.
- The `resume()` method at `train_srpo.py:1028-1052` never loads a scheduler.

**Verdict:** ❌ **Bug.** The learning rate is constant throughout training. For a 1000-step training run with AdamW at 1e-4, this may work acceptably, but a cosine decay or linear warmup+decay schedule is standard practice for LLM fine-tuning. The `TrainConfig` has no scheduler-related fields.

**Note:** This may be intentional for a research prototype, but it means the model never reduces its learning rate, which can cause training instability in later steps.

---

## 5. Checkpoint Save/Load

### Finding: PARTIAL BUG — sampler epoch not preserved

**Evidence:**

- `_save()` at `train_srpo.py:1018-1026` saves:
  ```python
  torch.save({
      "step": step,
      "trainable": {n: p.data.clone() for n, p in ...},
      "optimizer": self.optimizer.state_dict(),
      "config": self.cfg,
  }, f"checkpoints/step_{step}.pt")
  ```
- `resume()` at `train_srpo.py:1028-1052` restores:
  - `trainer.step = ckpt["step"]`
  - Trainable params via `param.data.copy_()` (preserves parameter identity for optimizer state mapping)
  - `trainer.optimizer.load_state_dict(ckpt["optimizer"])`

**What works:**
- Model weights (trainable injection + depth_lora) are correctly saved and restored. ✅
- Optimizer state (Adam momentum/variance buffers) is correctly saved and restored. ✅
- Training step is correctly restored. ✅

**What's missing:**
- **Sampler epoch is NOT saved.** The `_sampler` is recreated in `__init__` with the initial seed. On resume, the sampler starts at epoch 0, causing the first "epoch" of resumed data to repeat the shuffle pattern of the original epoch 0. Since the dataset shuffles with a fixed seed (deterministic), this means resumed training sees previously-seen data order. ❌
- **Sampler state (internal index) is NOT saved.** Even within an epoch, the sampler's internal position counter is lost. On resume, the sampler starts from the beginning of its shuffle order. ❌
- **`GradScaler` state is NOT saved.** The AMP scaler's internal scale factor is lost on resume; it restarts at the default scale. This is minor but can cause unnecessary gradient-skip steps. ⚠️
- Old-policy modules are NOT saved (acceptable — they're re-snapshotted at the start of each step). ✅

**Verdict:** ❌ **Partial bug.** Optimizer state and trainable weights are correctly preserved, which is the critical path. But sampler epoch and position are lost on resume, causing data-order regression. The `GradScaler` state is also lost (minor).

---

## 6. `_forward_pipeline`: Prelude → Recurrent → Coda Order & `intermediate_norm`

### Finding: CORRECT — pipeline order is correct; intermediate_norm is only used for thought extraction

**Evidence for pipeline order:**

- `_forward_pipeline()` at `model.py:510-569` executes in this order:
  1. `model.py:541-547`: **Prelude** — `self._run_block(h, self.prelude_indices, ...)`
  2. `model.py:548-549`: `e = h.clone().detach()` — cache prelude output for injection
  3. `model.py:558-561`: **Recurrent** — `self._recurrent_loop(h, e, n_loops, _run_rec, ...)`
  4. `model.py:563-568`: **Coda** — `self._run_block(h, self.coda_indices, ...)`
- The prelude runs once, then the recurrent block runs `n_loops` times with injection, then the coda runs once final. ✅

**Evidence for intermediate_norm isolation:**

- `intermediate_norm` is defined at `model.py:221` as `nn.LayerNorm(...)`.
- It is ONLY accessed in `_recurrent_loop()` at `model.py:496-504`:
  ```python
  if show_work:
      h_norm = self.intermediate_norm(h)
      inter_logits = self.lm_head(h_norm)
      ...
  ```
- `forward()` at `model.py:608` passes `show_work=False` to `_forward_pipeline()`. ✅
- `_forward_pipeline()` passes `show_work` through to `_recurrent_loop()`. ✅
- `forward_with_thoughts()` at `model.py:657` passes `show_work=True`. This is the ONLY path that uses `intermediate_norm`.
- The actual final logit projection in `forward()` uses `self.norm` (the pretrained Gemma final LayerNorm) at `model.py:622`:
  ```python
  h = self.norm(h)
  logits = self.lm_head(h)
  ```

**Verdict:** ✅ Correct. The pipeline runs in the correct order. `intermediate_norm` is strictly gated behind `show_work=True` and never used in training forward passes or in `generate()`.

---

## 7. `trainable_parameters()`: Injection + DepthLoRA Only

### Finding: CORRECT — returns only injection + depth_lora (not full model)

**Evidence:**

- `model.py:747-753`:
  ```python
  def trainable_parameters(self):
      params = []
      params.extend(self.injection.parameters())
      if self.depth_lora is not None:
          params.extend(self.depth_lora.parameters())
      return params
  ```
- `loop_embed` uses `register_buffer` at `model.py:122` (no learnable parameters), so excluding it is correct.
- `intermediate_norm` has learnable parameters (`weight`, `bias`), but is intentionally excluded because it's only used for thought extraction (inference-only diagnostic).
- The trainer's `_build_model()` at `train_srpo.py:618-626` sets `requires_grad=True` only for `injection` and `depth_lora`.
- The trainer's `trainable_params()` at `train_srpo.py:665-671` mirrors this:
  ```python
  def trainable_params(self):
      m = self._model_unwrapped
      for p in m.injection.parameters():
          yield p
      if m.depth_lora:
          for p in m.depth_lora.parameters():
              yield p
  ```

**Verdict:** ✅ Correct. Only the new (non-pretrained) modules are trained. The frozen backbone (pretrained Gemma weights) is never included.

**Minor documentation issue:** The comment at `train_srpo.py:618` says `# freeze backbone, train injection + lora + loop emb` but `loop_emb` is NOT trained (it uses fixed sinusoidal buffers, so this is correct behavior — the comment is just misleading).

---

## 8. Injection `clip` Values: `log_dt + log_A` Clamped to `[-20, 20]`

### Finding: CORRECT — the clamp range is not too restrictive

**Evidence:**

- `injection.py:89-93`:
  ```python
  def get_A(self) -> torch.Tensor:
      log_product = (self.log_dt + self.log_A).clamp(-20, 20)
      return torch.exp(-torch.exp(log_product))
  ```

**Numerical analysis of the clamp range:**

| `log_product` | Inner `exp(log_product)` | `A_discrete = exp(-inner)` | Interpretation |
|---|---|---|---|
| -20 | ~2.06e-9 | ~1.0 (0.999999998) | Near-identity; channel decays negligibly |
| -10 | ~4.54e-5 | ~0.99995 | Very slow decay |
| -5 | ~6.74e-3 | ~0.9933 | Slow decay |
| -3 | ~0.050 | ~0.951 | Initial value (log_dt=-3, log_A=0) |
| 0 | 1.0 | ~0.368 | Moderate decay |
| 3 | ~20.1 | ~1.9e-9 ≈ 0 | Near-instant reset |
| 20 | ~4.85e8 | ~0 | Instant reset |

- **Lower clamp (-20):** Allows `A_i ≈ 0.999999998`, which is effectively an identity channel. The theoretical maximum is 1.0 (but must be strictly < 1 for stability). -20 is very close to this limit. ✅
- **Upper clamp (20):** Prevents overflow of `exp(log_product)` while still allowing `A_i ≈ 0` (instant decay). ✅
- **Initialization:** `log_dt = -3.0`, `log_A = 0` → `log_product = -3.0` → `A ≈ 0.95`. This is a reasonable starting point where each channel retains ~95% of its signal per iteration. ✅
- **The clamp ensures `A` always stays in (0, 1), guaranteeing ρ(A) < 1 as claimed.** ✅

**Is -20 restrictive?** To achieve `A = 0.9999` (0.01% decay), you'd need `log_product ≈ -9.2`, which is well within the clamp. To achieve `A = 0.999999` (0.0001% decay), you'd need `log_product ≈ -13.8`, also within bounds. The -20 clamp allows values as high as ~`1 - 2e-9`, which is beyond any practical precision difference from 1.0 in bf16.

**Verdict:** ✅ Correct. The [-20, 20] clamp is not too restrictive. It allows the full practical range of decay rates from near-identity to instant reset, while preventing numerical overflow.

---

## Additional Finding: `sdpo_loss` KL Direction Mismatch

### Finding: POTENTIAL BUG — documentation says "reverse KL" but implementation computes forward KL

**Evidence:**

- `train_srpo.py:366-396`:
  ```python
  student_lp = F.log_softmax(student_logits.float(), dim=-1)  # log π_s
  teacher_p  = F.softmax(teacher_logits.float(), dim=-1)      # π_t

  kl = F.kl_div(
      student_lp.float(), teacher_p.float(),
      reduction="none", log_target=False,
  ).sum(dim=-1)
  ```
- `F.kl_div(input=student_lp, target=teacher_p, log_target=False)` computes:
  ```
  Σ teacher_p * (log(teacher_p) - student_lp)
  = Σ π_t * log(π_t / π_s)
  = KL(teacher || student)   ← FORWARD KL
  ```
- The docstring at `train_srpo.py:366` says "reverse KL: KL(p_student || p_teacher)".
- Forward KL (teacher→student) is mass-covering: the student is penalized wherever the teacher assigns probability, even if the student doesn't. This is the standard distillation KL direction.
- Reverse KL (student→teacher, i.e., `Σ π_s * log(π_s / π_t)`) is mode-seeking: the student only covers the teacher's modes.

**Impact:** The implementation computes forward KL, not reverse KL. This may be intentional (forward KL is standard for distillation), but the docstring is wrong. If the SRPO paper specifically calls for reverse KL (mode-seeking), then the KL direction is backwards.

**Verdict:** ⚠️ **Documentation bug** (or implementation bug, depending on SRPO paper specification). The docstring says "reverse KL" but the code computes forward KL.

---

## Summary

| # | Item | Verdict | Severity |
|---|---|---|---|
| 1 | SDPO teacher uses feedback-conditioned prompt + old-policy | ✅ Correct | — |
| 2a | GRPO log-probs detached (no grad flow) | ❌ Bug | **High** |
| 2b | Old-policy == current-policy (ratio always 1.0) | ❌ Bug | **High** |
| 2c | Old-policy snapshot mechanism | ✅ Correct | — |
| 3 | DDP setup: gradient sync, DistributedSampler | ✅ Correct | — |
| 4a | Gradient clipping | ✅ Correct | — |
| 4b | LR scheduler | ❌ Missing | Medium |
| 5a | Checkpoint: model weights + optimizer state | ✅ Correct | — |
| 5b | Checkpoint: sampler epoch not preserved | ❌ Bug | Low |
| 6 | Pipeline order + intermediate_norm isolation | ✅ Correct | — |
| 7 | trainable_parameters() returns only injection + depth_lora | ✅ Correct | — |
| 8 | Injection clamp [-20, 20] | ✅ Correct | — |
| A | sdpo_loss KL direction docstring | ⚠️ Doc bug | Low |

**Critical finding:** The GRPO branch (`grpo_loss`) contributes **zero gradient** to the model. All training signal comes from the SDPO self-distillation branch. The model is effectively learning only through the teacher-student distillation loss, not through the policy gradient. The `@torch.no_grad()` decorator on `_generate()` causes cached log-probs to be detached, and the old-policy snapshot captures the same weights used for generation, making the importance ratio always 1.0 even if gradients did flow.
