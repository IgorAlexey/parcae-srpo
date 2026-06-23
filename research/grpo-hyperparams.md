# Research: GRPO Hyperparameter Best Practices Across Implementations

## Summary

GRPO (Group Relative Policy Optimization) uses **group_size (G)=8–64**, **temperature=0.9–1.0**, **learning rate=1×10⁻⁶ to 5×10⁻⁶** (10–100× lower than SFT), and **KL penalty β=0.001–0.04** in practice. The canonical DeepSeekMath paper uses G=64, LR=1e-6, β=0.04, temp=1.0. Production recipes converge on G=8 (GSPO/Unsloth) to G=16 (DAPO), with β often tuned down to 0.001–0.01 for stability. A critical operational insight: when all group rewards are equal (zero advantage), GRPO produces no gradient—frameworks handle this with batch-skipping (Axolotl), dynamic resampling (DAPO), or per-reward decoupled normalization (GDPO).

## Findings

### 1. Group Size (G) — Production Values by Framework

| Framework / Paper | Default G | Notes |
|---|---|---|
| **DeepSeekMath (canonical)** | 64 | Batch of 1024 prompts, G=64 completions per prompt [Source](https://ar5iv.labs.arxiv.org/html/2402.03300) |
| **DAPO (ByteDance)** | 16 | Train Qwen2.5-32B; with dynamic sampling filter [Source](https://blog.diffio.ai/grpo-dapo/) |
| **TRL (HuggingFace)** | 8 | `num_generations=8` default in `GRPOConfig` [Source](https://huggingface.co/docs/trl/grpo_trainer) |
| **Unsloth (production)** | 4–8 | Notebooks use `num_generations=4` to `8`; GSPO recipe uses `group_size=8` [Source](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb) |
| **GSPO (Unsloth advanced)** | 8 | Production recipe: group_size=8, kl_coeff=0.1, clip_eps=0.2, temperature=1.0 [Source](https://megacpp.com/blog/distillation-bestofn-and-rl/) |
| **veRL** | Configurable >1 | `data.train_batch_size` controls global batch; `rollout.n` for group sampling [Source](https://verl.readthedocs.io/en/latest/algo/grpo.html) |
| **Axolotl** | Configurable | Supports streaming partial batch + zero-advantage skipping [Source](https://docs.axolotl.ai/docs/grpo.html) |
| **OpenRLHF** | Configurable | Uses `--advantage_estimator group_norm` flag; group size per-prompt sampling [Source](https://claudeskill.wiki/ar/skills/ai-research/post-training-openrlhf/) |
| **GRPO-LEAD (2024)** | 8 | Batch size 32, group size 8; KL penalty removed entirely [Source](https://arxiv.org/html/2504.09696v1) |
| **Enhanced LLM Reasoning (2026)** | 4 | Small-scale GRPO with G=4, batch=4, LR=5e-6 [Source](https://arxiv.org/html/2605.02073) |

**Key insight from Lian et al. (2025) PPO/GRPO/DAPO comparison:** Increasing group size leads to more stable training dynamics and higher accuracy. Larger G reduces advantage estimate variance at the cost of proportionally more inference compute. [Source](https://arxiv.org/abs/2512.07611)

### 2. Temperature — Exploration vs. Exploitation

- **TRL default: 0.9** — This is the `GRPOConfig.temperature` default. [Source](https://www.stephendiehl.com/posts/grpotrainer/)
- **DeepSeekMath: 1.0** — The canonical paper uses temperature=1.0 for generation during GRPO training. [Source](https://ar5iv.labs.arxiv.org/html/2402.03300)
- **DAPO: 1.0** — Same as DeepSeekMath. The asymmetric clipping (Clip-Higher) provides exploration, reducing reliance on temperature tuning. [Source](https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/DAPO.html)
- **GSPO production: 1.0** — Matches DeepSeekMath. [Source](https://megacpp.com/blog/distillation-bestofn-and-rl/)
- **veRL default: 1.0** — Rollout config YAML defaults `temperature: 1.0`. [Source](https://github.com/verl-project/verl/blob/main/verl/trainer/config/rollout/rollout.yaml)
- **JAX/Tunix implementations: 1.0** — "High for diverse responses." [Source](https://medium.com/@ktiyab_42514/tune-gemma-3-1b-in-jax-with-grpo-for-reasoning-part-2-grpo-in-tunix-4cb58a5d2402)
- **Critical perspective paper (R1-Zero-Like Training):** Explored temperatures 0.6–1.0. Shows that base model pass@8 accuracy at different temperatures is indicative of exploration ability. Suggests 0.8 as a sweet spot for some base models. [Source](https://arxiv.org/abs/2503.20783)
- **Consensus:** Temperature 1.0 is the safe default. GRPO's group-based advantage normalization reduces the need for temperature annealing. Lower temperatures (0.7–0.8) may help in later training stages to reduce noise when the policy has converged, but 1.0 is standard for the exploration phase.

### 3. Learning Rates — LoRA + GRPO vs SFT

**The 10–100× Rule:** GRPO RL learning rates are consistently 10–100× lower than SFT learning rates for the same model.

| Training Type | Typical LR (LoRA) | Typical LR (Full) | Source |
|---|---|---|---|
| **SFT (LoRA)** | 1×10⁻⁴ to 5×10⁻⁵ | 5×10⁻⁶ to 5×10⁻⁵ | [Source](https://futureagi.com/blog/llm-fine-tuning-techniques-i-ii/) |
| **SFT (full)** | — | 1×10⁻⁵ to 5×10⁻⁵ | General practice |
| **GRPO RL (LoRA)** | 5×10⁻⁷ to 5×10⁻⁶ | 1×10⁻⁶ | Multiple sources below |
| **GRPO RL (full)** | — | 1×10⁻⁶ | DeepSeekMath canonical |

**Specific values from practice:**

- **DeepSeekMath (full fine-tune):** LR = 1×10⁻⁶, constant schedule. [Source](https://ar5iv.labs.arxiv.org/html/2402.03300)
- **Unsloth Llama 3.1 8B GRPO (LoRA):** `learning_rate=5e-6`, adam_beta1=0.9, weight_decay=0.1. [Source](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
- **Phil Schmid Mini-R1 (LoRA):** Started at 1e-6 (unstable), reduced to **5×10⁻⁷** for stable training. [Source](https://www.philschmid.de/mini-deepseek-r1)
- **AlignRL notebook (LoRA):** `learning_rate=5e-6` — explicitly notes "Lower than SFT since RL training is more sensitive." [Source](https://colab.research.google.com/github/sacredvoid/alignrl/blob/main/notebooks/02_grpo_math_reasoning.ipynb)
- **Agent RFT guide:** `learning_rate=1.0e-5` for LoRA, warns "10–100× lower than SFT." [Source](https://tensorops.ai/blog/practical-guide-to-agent-reinforcement-fine-tuning)
- **Lian et al. ablation:** Learning rates tested at {1e-4, 5e-5, 1e-5, 5e-6}. "Larger learning rates (e.g., 1e-4) lead to unstable training behavior." Best results at 1×10⁻⁵ to 5×10⁻⁶ for LoRA. [Source](https://arxiv.org/pdf/2606.10931)
- **Reasoning-SQL (full fine-tune):** LR = 1×10⁻⁶, constant schedule with 0.1% warmup. [Source](https://arxiv.org/html/2503.23157v1)
- **DAPO (Qwen2.5-32B, full):** LR = 1×10⁻⁵ for 3 epochs. [Source](https://aman.ai/primers/ai/deepseek-R1/)
- **ATLAS production config:** `learning_rate: 1e-6` for RL. [Source](https://docs.arc.computer/training/configuration)

**Practical guidance:** Start GRPO LoRA at 5×10⁻⁶. If loss diverges or outputs become degenerate, drop to 1×10⁻⁶ or 5×10⁻⁷. Aayush Garg's ablation blog states: "The learning rate is the most critical hyperparameter. A high lr in GRPO doesn't just cause loss divergence—it can collapse the policy onto degenerate outputs before learning anything useful." [Source](https://huggingface.co/blog/garg-aayush/grpo-from-scratch)

### 4. Zero-Advantage (All Rewards Equal) Edge Case

When all G completions for a prompt receive identical rewards, the group standard deviation σ=0, and normalized advantages become 0/0 or all-zero. This means **no gradient signal** for that prompt.

**How frameworks handle it:**

- **Standard GRPO (DeepSeekMath):** Uses `σ + 1e-8` epsilon in denominator to prevent division by zero, but the advantage is still ≡0 for all completions. The batch is effectively wasted compute. [Source](https://enricopiovano.com/blog/grpo-group-relative-policy-optimization/)
- **TRL (HuggingFace):** Logs `frac_reward_zero_std` metric — the fraction of samples with reward std of zero. Does NOT skip these batches by default; gradients simply cancel out. [Source](https://huggingface.co/docs/trl/grpo_trainer)
- **Axolotl:** Implements **"zero-advantage batch skipping"** — detects zero-variance groups and skips the gradient step entirely. Also supports **"streaming partial batch"** mode that scores one prompt group at a time, enabling finer-grained skipping. [Source](https://docs.axolotl.ai/docs/grpo.html)
- **DAPO:** **Dynamic Sampling** — actively oversamples and filters to ensure every group has non-zero reward variance before computing the loss. "Keep sampling new prompts and responses until every batch has non-zero reward variance." [Source](https://pub.towardsai.net/the-death-of-rlhf-a-practitioners-guide-to-the-new-post-training-stack-84b2ff6d4e74)
- **GDPO (NVIDIA):** **Per-reward decoupled normalization** — normalizes each reward component (e.g., correctness, format) independently before summing. This prevents one high-variance reward from drowning out another, and avoids the scenario where all summed rewards are equal. Does NOT fully solve the all-equal case but reduces its frequency. [Source](https://arxiv.org/pdf/2601.05242)
- **Real-world impact:** One HuggingFace model (thawait/qwen2.5-7b-math-reasoning-grpo) reported `frac_reward_zero_std = 0.63` — meaning **63% of batches produced near-zero gradient signal** throughout training. [Source](https://huggingface.co/thawait/qwen2.5-7b-math-reasoning-grpo)

**Practical recommendations:**
1. Monitor `frac_reward_zero_std` — if consistently >0.5, your reward function is too coarse
2. Use multi-component reward functions (correctness + format + style) to increase variance
3. Consider Axolotl's zero-advantage skipping or implement DAPO-style dynamic sampling
4. If using TRL, implement a custom callback to skip updates when `reward_std == 0`

### 5. KL Penalty (β) Values in Practice

| Source | β Value | Context |
|---|---|---|
| **DeepSeekMath (canonical)** | 0.04 | Full fine-tune, rule-based reward [Source](https://ar5iv.labs.arxiv.org/html/2402.03300) |
| **TRL default** | 0.04 | `GRPOConfig.beta` default [Source](https://www.stephendiehl.com/posts/grpotrainer/) |
| **ATLAS production** | 0.04 | Matches DeepSeekMath [Source](https://docs.arc.computer/training/configuration) |
| **Phil Schmid Mini-R1** | 0.001 | Reduced from 0.04 for stability on LoRA [Source](https://www.philschmid.de/mini-deepseek-r1) |
| **Unsloth notebooks** | 0.001 | Common in LoRA GRPO recipes [Source](https://forums.fivetechsupport.com/viewtopic.php?t=45371) |
| **GSPO production** | 0.1 | Higher for strong KL anchoring [Source](https://megacpp.com/blog/distillation-bestofn-and-rl/) |
| **Agent RFT guide** | 0.01 | "Tune carefully" [Source](https://tensorops.ai/blog/practical-guide-to-agent-reinforcement-fine-tuning) |
| **GRPO-LEAD** | 0.0 | KL penalty removed entirely "as it was found to suppress exploration" [Source](https://arxiv.org/html/2504.09696v1) |
| **Some LoRA recipes** | 0.002 | Very low to allow more policy movement [Source](https://blog.ando.ai/posts/ai-grpo/) |

**Key insights:**
- β=0.04 is the canonical starting point from DeepSeekMath
- **LoRA typically needs lower β (0.001–0.01)** because the low-rank constraint already limits policy divergence
- KL divergence is only logged when `beta > 0`. Running β=0 means "flying without this instrument — risky for long training runs." [Source](https://chrisvoncsefalvay.com/posts/post-training-instrument-cluster-grpo/)
- Lian et al. found the impact of KL coefficient is **non-monotonic** — too low causes reward hacking, too high stifles learning [Source](https://arxiv.org/abs/2512.07611)
- With rule-based verifiable rewards (RLVR), β can be set lower or zero since there's no learned reward model to over-optimize against

### 6. Ablation Studies on GRPO Hyperparameters

**Aayush Garg — "GRPO: Building Intuition Through Ablation Studies" (2026):**
- 20+ experiments on Qwen2.5-Math-1.5B
- **Learning rate sweep:** Most critical hyperparameter. 5e-7 was stable; 1e-6 showed better convergence; 5e-6 caused collapse on some seeds
- **Group size sweep:** G=4 vs G=8 vs G=16. Larger G improved stability but increased compute linearly
- **Normalization types:** Batch norm vs group norm vs no norm. Group normalization essential for GRPO
- **On-policy vs off-policy:** On-policy (regenerating completions each step) significantly outperformed reusing stale completions
- Key conclusion: "Run a lot of ablation studies to understand and build intuition on what matters in GRPO training." [Source](https://huggingface.co/blog/garg-aayush/grpo-from-scratch)

**Lian et al. — "Comparative Analysis and Parametric Tuning of PPO, GRPO, and DAPO" (2025):**
- Systematic comparison on Countdown Game → transfer to general reasoning benchmarks
- **Group size effect:** Larger G → more stable training + higher accuracy in both GRPO and DAPO
- **KL penalty effect:** Non-monotonic impact; moderate β values best
- **Dynamic Sampling (DAPO):** Did NOT improve performance; best results achieved without it
- **Token-level vs sample-level:** Token-level loss (DAPO) showed marginal improvements
- **Entropy bonus:** PPO benefited; GRPO/DAPO didn't need it
- **Overall:** DAPO with Clip-Higher + token-level loss achieved best results, but gains over well-tuned GRPO were modest [Source](https://arxiv.org/abs/2512.07611)

**DAPO paper — Component ablation (ByteDance, 2025):**
- Ablated 4 key components on Qwen2.5-32B
- Clip-Higher: Biggest individual gain (~3 pts on AIME)
- Dynamic Sampling: Mixed results; helps on hard prompts, hurts throughput
- Token-level loss: Moderate gain (~1.5 pts)
- Overlong filtering: Necessary for training stability on long CoT [Source](https://blog.diffio.ai/grpo-dapo/)

**"Understanding R1-Zero-Like Training: A Critical Perspective" (2025):**
- Showed GRPO's std-based normalization introduces **question-level difficulty bias**
- Proposed Dr.GRPO: removes length normalization and std normalization from GRPO's loss
- Found that Dr.GRPO improves token efficiency while maintaining reasoning performance [Source](https://arxiv.org/abs/2503.20783)

### 7. DeepSeekMath Paper (2402.03300) — Exact Training Configuration

The DeepSeekMath paper is the canonical introduction of GRPO. Here is the RL training configuration extracted from the paper (Section 4.2):

| Parameter | Value |
|---|---|
| **Algorithm** | GRPO (Group Relative Policy Optimization) |
| **Base model** | DeepSeekMath-Base 7B (initialized from DeepSeek-Coder-Base-v1.5 7B) |
| **RL training data** | 144K Chain-of-Thought prompts from SFT dataset (math problems) |
| **Group size (G)** | 64 completions per prompt |
| **Batch size** | 1,024 prompts per training step |
| **Learning rate (policy)** | 1×10⁻⁶ (constant) |
| **KL penalty coefficient (β)** | 0.04 |
| **Optimizer** | AdamW |
| **Reward model** | Rule-based verifier (not a learned neural model) |
| **Temperature** | 1.0 (implied — standard for RL phase) |
| **Epochs** | 1 epoch (iterative RL) |
| **Advantage normalization** | Group-level: (r - mean(r_group)) / std(r_group) |
| **Clipping (ε)** | 0.2 (symmetric PPO-style clipping) |
| **Loss type** | Clipped surrogate objective with KL penalty |
| **KL estimator** | k3 estimator (unbiased estimate of KL divergence) |

**Key architectural decisions:**
- **No critic/value model** — advantages computed purely from group statistics
- **Rule-based reward** — no learned reward model, eliminating reward hacking surface
- **Single epoch** — avoids overfitting to the reward signal
- **Iterative RL** — the paper trains, then evaluates, then continues training (Figure 6 shows iterative improvement) [Source](https://ar5iv.labs.arxiv.org/html/2402.03300)

## Sources

### Kept
- **DeepSeekMath ar5iv (2402.03300)** — Canonical GRPO paper with exact training configuration in Section 4.2
- **TRL GRPOConfig source (GitHub)** — Default hyperparameter values for HuggingFace's implementation
- **TRL GRPO Trainer docs (HuggingFace)** — Official documentation with parameter descriptions and defaults
- **diffio.ai GRPO/DAPO comparison** — Detailed side-by-side of GRPO vs DAPO with exact hyperparameters from both papers
- **Phil Schmid Mini-R1 tutorial** — Real-world debugging of DeepSeekMath hyperparameters with LoRA
- **Axolotl GRPO docs** — Zero-advantage batch skipping, streaming partial batch, production monitoring
- **veRL GRPO docs + rollout config** — Production RL framework configuration reference
- **GSPO production recipe (MegaCpp)** — Real production hyperparameter values
- **Aayush Garg GRPO ablation blog (HuggingFace)** — 20+ controlled experiments on hyperparameter sensitivity
- **Lian et al. PPO/GRPO/DAPO comparison (arXiv 2512.07611)** — Systematic parametric tuning study
- **GDPO paper (arXiv 2601.05242)** — Multi-reward normalization fix for GRPO edge cases
- **TensorOps Agent RFT guide** — Practical practitioner guidance with specific LR, KL, group_size values
- **Unsloth Llama 3.1 8B GRPO notebook** — Production LoRA+GRPO hyperparameters
- **Stephendiehl GRPOTrainer guide** — Clear documentation of all TRL defaults
- **HuggingFace model card (thawait/qwen2.5-7b-math-reasoning-grpo)** — Real-world zero-std metric (63%)
- **Understanding R1-Zero-Like Training (arXiv 2503.20783)** — Critical analysis of GRPO biases including temperature effects

### Dropped
- **w3tutorials.net / generic ML hyperparameter pages** — Not GRPO-specific
- **Symbl.ai LLM hyperparameters guide** — Generic LLM, not GRPO
- **Wikipedia DeepSeek** — Too high-level, no hyperparameter detail
- **Various Medium/LinkedIn summaries** — Derivative of primary sources already captured
- **RecBole / XGBoost / CatBoost pages** — Irrelevant to GRPO
- **GDPO search results about standard deviation** — Wikipedia math, not relevant

## Gaps

1. **OpenRLHF exact GRPO hyperparameter schema:** While I found that OpenRLHF supports GRPO via `--advantage_estimator group_norm`, the exact argument names and defaults were not fully resolved. The framework uses token-in-token-out architecture and separates actor/reward/reference models. Suggested: inspect OpenRLHF's GRPO example scripts directly in the repo.

2. **veRL GRPO group_size exact default:** The veRL docs reference `rollout.n` and `data.train_batch_size` for group sampling, but the exact per-prompt group_size default was ambiguous. Suggested: review veRL's example GRPO training YAML configs.

3. **Temperature ablation studies:** No publication specifically ablates GRPO temperature in isolation. The critical perspective paper touches on it, but a dedicated study comparing temperatures 0.6/0.8/1.0/1.2 for GRPO would be valuable.

4. **LoRA rank interaction with GRPO:** No systematic study found on how LoRA rank (r) interacts with GRPO learning rate, KL penalty, or group size. This is a significant gap given how commonly LoRA+GRPO is used.

5. **FP8/quantization interaction with GRPO stability:** veRL's rollout config supports FP8. No published guidance on how quantization affects GRPO hyperparameter sensitivity.
