# Research: Unsloth GRPO Training Practices

## Summary
Unsloth wraps HuggingFace TRL's `GRPOTrainer`/`GRPOConfig` and does not alter the core hyperparameter defaults. Unsloth's own notebooks use **`num_generations=6–8` (G=6–8)**, **`temperature=0.9`** (TRL default), **`learning_rate=5e-6`**, **`beta=0.04`**, and **`warmup_ratio=0.1`**. Unsloth recommends starting from an **instruct-tuned model** (not a base model) for GRPO, and for base models they recommend an explicit SFT warmup/priming stage before GRPO. The well-known "advantage collapse" problem (all group rewards identical → zero gradients) is a fundamental GRPO limitation that Unsloth's advanced notebooks attempt to mitigate through better reward functions.

---

## Findings

### 1. group_size / num_generations (G) — Is 2 too low?

**Unsloth's notebooks use `num_generations = 6` or `8` as the standard default.** The Llama3.1 (8B)-GRPO notebook uses `num_generations = 6` [Source](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb). The earlier Unsloth notebooks and community guides use `num_generations = 8` [Source](https://github.com/unslothai/unsloth/issues/1836). The TRL `GRPOConfig` default is 8 [Source](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py).

**G=2 is generally considered too low by the community.** The GRPO algorithm computes advantages as `(reward - group_mean) / group_std`. With only 2 samples, the advantage collapses to ±constant (one positive, one negative), providing a binary gradient signal with zero variance information. A larger G (4–16) gives better statistics for advantage estimation. One community guide states: "Start with 8, increase to 16 if GPU allows" [Source](https://lobehub.com/skills/frank-luongt-faos-skills-marketplace-grpo-rl-training). The HuggingFace blog on GRPO with Unsloth notes: "A larger group gives better statistics" [Source](https://huggingface.co/blog/shivance/post-training-llm-for-reasoning-with-grpo). Minimum practical value is generally **G=4**, with G=6–8 being the sweet spot for VRAM-constrained setups. Unsloth users have reported trying G=2, 3, 4 and hitting OOM for other reasons but not reporting success with G=2 [Source](https://github.com/unslothai/unsloth/issues/3864).

**Recommendation from sources:** G ≥ 4, ideally G = 6–8 for GRPO to have meaningful advantage variance. G=2 is too low.

### 2. Temperature for GRPO sampling — Is 1.2 typical?

**Unsloth follows TRL's default: `temperature = 0.9`.** This is the TRL `GRPOConfig` default [Source](https://www.stephendiehl.com/posts/grpotrainer/). Unsloth's own documentation for VLM RL and GSPO notebooks also shows `temperature = 0.9` [Source](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/vision-reinforcement-learning-vlm-rl).

However, community practice varies:
- **`temperature = 0.8`** with `top_p = 0.95` is used in the HuggingFace course exercise for GRPO with Unsloth for *inference* (post-training sampling) [Source](https://huggingface.co/learn/llm-course/en/chapter12/6)
- **`temperature = 1.0`** is recommended by the HuggingFace GRPO blog post: "A higher temperature (like 1.0) encourages the model to generate more diverse and creative responses for the group" [Source](https://huggingface.co/blog/shivance/post-training-llm-for-reasoning-with-grpo)
- **`temperature = 1.2`** is NOT a standard Unsloth recommendation. It may be used in some community experiments but is higher than Unsloth's defaults.

**Recommendation:** 0.9 (Unsloth/TRL default) or 1.0 (for more diversity). 1.2 is on the high side and risks too much randomness, degrading response quality.

### 3. Learning rate for LoRA/QLoRA + GRPO

**Unsloth's standard: `learning_rate = 5e-6`.** This appears consistently across ALL Unsloth GRPO notebooks:
- Llama3.1 (8B)-GRPO: `learning_rate = 5e-6` [Source](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
- Qwen2.5 (3B)-GRPO: `learning_rate = 5e-6` [Source](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(3B)-GRPO.ipynb)
- Advanced Llama3.2 (3B)-GRPO: `learning_rate = 5e-6`
- Phi-4 (14B)-GRPO: `learning_rate = 5e-6`
- Community guides consistently use `5e-6` [Source](https://softmaxdata.com/blog/how-to-tune-your-own-llm-with-grpo-common-crawl-and-unsloth/)

This is ~10× smaller than typical SFT LoRA learning rates (which are ~2e-4 to 5e-5). The rationale: RL reward signals are sparser and noisier than SFT labels, so a very conservative LR prevents destabilizing the pre-trained policy [Source](https://medium.com/@Thumar_Rushik/step-by-step-breakdown-fine-tuning-gemma-3-1b-with-grpo-in-unsloth-bc3bce6b7a52). Community skill guides state: "5e-6 (safe), 1e-5 (faster, riskier)" [Source](https://lobehub.com/skills/frank-luongt-faos-skills-marketplace-grpo-rl-training).

**Additional optimizer settings** from Unsloth notebooks:
- `adam_beta1 = 0.9`, `adam_beta2 = 0.99`
- `weight_decay = 0.1`
- `optim = "paged_adamw_8bit"`
- `lr_scheduler_type = "cosine"`
- `warmup_ratio = 0.1` (10% of steps)
- `max_grad_norm = 0.1`

### 4. SFT warmup stage before GRPO

**Unsloth's position: yes, for base models; not needed for instruct models.**

Unsloth's standard GRPO notebooks start from **instruct-tuned models** (e.g., `meta-llama/Llama-3.1-8B-Instruct`, `google/gemma-3-1b-it`), which have already undergone SFT. No additional SFT warmup is applied before GRPO in these notebooks.

For training from a **base model** (no instruction tuning), Unsloth explicitly recommends an SFT warmup/priming stage. The Reddit AMA with the Unsloth team confirms: "to do SFT warmup or priming, which involves a small fast finetuning run to convert a base model into a instruct model for RL" [Source](https://www.reddit.com/r/LocalLLaMA/comments/1ndjxdt/ama_with_the_unsloth_team/). Unsloth provides a dedicated notebook: **"Qwen 3 4B Base GRPO"** which includes an SFT priming stage before GRPO training [Source](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Advanced_Llama3_2_(3B)_GRPO_LoRA.ipynb) (referenced in the Advanced notebook description).

Additional evidence from Unsloth's own RL Guide: "If you're not getting any reasoning, make sure you have enough training steps and ensure your reward function/verifier is working" — implying that GRPO on an already-capable instruct model should work without extra warmup [Source](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide).

The broader research community also favors SFT-then-RL: "SFT-then-RL Outperforms Mixed-Policy Methods" [Source](https://arxiv.org/html/2604.23747v1). A typical three-stage pipeline is SFT → Preference Optimization → RL (GRPO) [Source](https://zylos.ai/research/2026-04-10-rl-posttraining-tool-using-agents-grpo-async-rl/).

**Recommendation:** If starting from an instruct model → GRPO directly. If starting from a base model → do a brief SFT warmup first to teach format-following and basic task structure.

### 5. Handling mixed results (some correct, some wrong) in groups

This is GRPO's **normal operating mode** — the algorithm is designed for it. The advantage formula is: `A_i = (r_i - mean(r_group)) / std(r_group)`. When rewards are mixed, advantages are properly scaled: correct responses get positive advantage, incorrect get negative advantage.

Unsloth does not add special handling for mixed-result groups beyond what TRL's GRPO does. The key mechanisms:
- **KL penalty (`beta=0.04`):** Prevents the policy from diverging too far from the reference model, which is important when advantages are noisy [Source](https://www.stephendiehl.com/posts/grpotrainer/)
- **Gradient clipping (`max_grad_norm=0.1`):** Unsloth uses aggressive gradient clipping to prevent large updates from noisy rewards
- **Reward function design:** Unsloth's advanced notebooks use multi-component reward functions (format + correctness) to provide smoother reward signals [Source](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo)

The real edge case is when all rewards are identical (not mixed), which leads to the stagnation problem discussed in Finding 6.

### 6. Known issues with reward=0 stagnation in GRPO

**This is a well-documented fundamental GRPO problem called "advantage collapse."**

When all completions in a group receive identical rewards (all correct → all reward=1, or all wrong → all reward=0), the group standard deviation becomes 0, making all advantages = 0. The gradient signal vanishes, and the model stops learning for that batch. This is formally studied in the ICML 2026 paper "Advantage Collapse in Group Relative Policy Optimization: Diagnosis and Mitigation" [Source](https://arxiv.org/html/2605.21125).

**Unsloth-specific manifestations:**
- GitHub Issue #3260: Users report "constant loss of zero throughout most of the training" when fine-tuning with GRPO [Source](https://github.com/unslothai/unsloth/issues/3260)
- GitHub Issue #2291: "Loss Always Zero While Training GRPO Model" — user trying to overfit on a single sample [Source](https://github.com/unslothai/unsloth/issues/2291)
- GitHub Issue #2614: "When using GRPO for training, the loss is 0" [Source](https://github.com/unslothai/unsloth/issues/2614)
- HuggingFace Forum: "GRPO loss is always zero" — despite non-zero rewards and grad_norm [Source](https://discuss.huggingface.co/t/huggingface-trl-grpo-loss-is-always-zero/155597)

**Unsloth's mitigations:**
- **Better reward functions:** Unsloth's Advanced GRPO notebooks use multi-component reward functions (e.g., format reward + correctness reward) that produce more granular scores (not just 0/1 binary), reducing the chance of all-identical rewards [Source](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo)
- **Larger group sizes (G=6–8):** More samples per group reduce the probability of all rewards being identical
- **DAPO/GSPO support:** Unsloth now supports GSPO (Group Sequence Policy Optimization) and DAPO-style techniques that include mechanisms like `epsilon_high` (0.28 recommended) to handle low-variance reward scenarios [Source](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation)

The broader research community's mitigations (not Unsloth-specific) include: AVSPO (injects virtual samples when collapse detected), DrGRPO (length-normalized token-level losses), and importance sampling [Source](https://arxiv.org/html/2605.21125).

---

## Sources

### Kept
- **TRL GRPOConfig source** (https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py) — Definitive source for TRL defaults (temperature=0.9, num_generations=8, beta=0.04)
- **Stephen Diehl's GRPOTrainer guide** (https://www.stephendiehl.com/posts/grpotrainer/) — Clear documentation of all GRPO defaults with explanations
- **Unsloth Llama3.1 (8B)-GRPO notebook** (https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb) — Primary source: exact Unsloth config (num_generations=6, lr=5e-6, warmup_ratio=0.1)
- **Unsloth RL Guide** (https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide) — Official Unsloth documentation on GRPO
- **Unsloth Advanced RL Documentation** (https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation) — GSPO/DAPO settings, epsilon_high, importance sampling
- **Unsloth Tutorial: Train Reasoning Model with GRPO** (https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo) — Troubleshooting advice for GRPO not learning
- **HuggingFace Blog: Post-training LLM for reasoning with GRPO using Unsloth** (https://huggingface.co/blog/shivance/post-training-llm-for-reasoning-with-grpo) — temperature=1.0 recommendation, group size discussion
- **Reddit AMA with Unsloth team** (https://www.reddit.com/r/LocalLLaMA/comments/1ndjxdt/ama_with_the_unsloth_team/) — Confirms SFT warmup/priming for base models
- **Advantage Collapse paper (ICML 2026)** (https://arxiv.org/html/2605.21125) — Formal analysis of reward=0 stagnation problem
- **GitHub Issue #3260 (unsloth)** (https://github.com/unslothai/unsloth/issues/3260) — Zero loss / stagnation bug report
- **GitHub Issue #2291 (unsloth)** (https://github.com/unslothai/unsloth/issues/2291) — Loss always zero bug report
- **GitHub Issue #1836 (unsloth)** (https://github.com/unslothai/unsloth/issues/1836) — Early GRPO issue showing num_generations=8, lr=5e-6 config
- **Unsloth Long-context GRPO blog** (https://unsloth.ai/blog/grpo) — Memory analysis showing num_generations=8 in GRPO loss computation
- **Unsloth VLM RL docs** (https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/vision-reinforcement-learning-vlm-rl) — temperature=0.9, max_grad_norm=0.1 defaults
- **Cloudera GRPO guide** (https://community.cloudera.com/t5/Community-Articles/A-Practical-Guide-to-Fine-Tuning-Language-Models-with-GRPO/ta-p/411583) — LR=5e-6, warmup_ratio=0.1, cosine schedule
- **SFT-then-RL paper** (https://arxiv.org/html/2604.23747v1) — Evidence for SFT warmup before RL
- **Medium: Gemma 3 GRPO breakdown** (https://medium.com/@Thumar_Rushik/step-by-step-breakdown-fine-tuning-gemma-3-1b-with-grpo-in-unsloth-bc3bce6b7a52) — LR rationale for GRPO

### Dropped
- **Modelscope/ms-swift docs** — Different framework, not Unsloth-specific
- **Analytics Vidhya GRPO article** — High-level, no concrete Unsloth config numbers
- **Various Japanese/Chinese tutorial pages** — Redundant with English sources above
- **LinkedIn posts** — Low information density, summaries of other sources
- **arXiv papers not directly about Unsloth** — General GRPO theory, not Unsloth practice
- **Predibase GRPOConfig docs** — Different platform, different defaults (num_generations=16)

---

## Gaps

1. **Exact Unsloth `UnslothGRPOConfig` vs. TRL `GRPOConfig` differences:** Unsloth may override certain defaults internally, but the exact overrides are in the compiled Python source (`unsloth/models/llama.py`, etc.), which was not directly inspected. The notebooks use standard TRL `GRPOConfig` with explicit parameter overrides, suggesting Unsloth doesn't silently change defaults.

2. **Temperature=1.2 origin:** Could not trace where 1.2 comes from. It is not in Unsloth docs, TRL defaults, or community guides. May originate from a specific blog post or paper not covered in this search.

3. **G=2 experimental evidence:** No published ablation studies comparing G=2 vs G=4/6/8 specifically for Unsloth were found. The recommendation against G=2 is based on algorithmic reasoning and community consensus, not direct benchmarks.

4. **Unsloth's internal handling of zero-advantage batches:** Whether Unsloth has custom logic to skip or reweight batches with zero advantage (beyond what TRL provides) is not documented publicly. The source code in `unsloth/models/` would need direct inspection.

---

## Quick Reference: Unsloth GRPO Default Config

| Parameter | Unsloth Default | TRL Default | Notes |
|---|---|---|---|
| `num_generations` (G) | **6** (notebooks) | 8 | Reduce if OOM; G≥4 recommended |
| `temperature` | **0.9** | 0.9 | 1.0 for more diversity |
| `learning_rate` | **5e-6** | 1e-5 | 10× smaller than SFT LR |
| `beta` (KL penalty) | **0.04** | 0.04 | Range: 0.01–0.1 |
| `warmup_ratio` | **0.1** | 0.0 | 10% of total steps |
| `lr_scheduler_type` | **cosine** | linear | |
| `max_grad_norm` | **0.1** | 1.0 | Aggressive clipping |
| `optim` | **paged_adamw_8bit** | adamw_torch | |
| `weight_decay` | **0.1** | 0.0 | |
| `lora_rank` | **32** (basic) / **64** (advanced) | N/A | Larger = smarter |
| `max_prompt_length` | **256–768** | 512 | Task-dependent |
| `per_device_train_batch_size` | **1** | 1 | Must divide evenly by num_generations |
