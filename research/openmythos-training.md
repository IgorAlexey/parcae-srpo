# Research: OpenMythos Training Approach (Kye Gomez / Agora)

## Summary

OpenMythos is Kye Gomez's open-source, first-principles theoretical reconstruction of Anthropic's Claude Mythos, implemented as a **Recurrent-Depth Transformer (RDT)** with Mixture-of-Experts. The planned training pipeline is **pretraining on FineWeb-Edu → GRPO + high-quality RL fine-tuning**, with no explicit SFT warmup stage between pretraining and GRPO. The architecture achieves parameter efficiency through weight-shared recurrent loops: a 770M RDT matches a 1.3B standard transformer on identical data, and community runs show 2.67× faster validation convergence vs. nanoGPT. Training stability is guaranteed by design via **LTI-stable injection (spectral radius < 1)**, making the model robust to high learning rates.

## Findings

### 1. Training Pipeline: Pretraining → GRPO/RL (No SFT Warmup)

OpenMythos follows a two-phase plan. **Phase 1** is standard autoregressive pretraining on FineWeb-Edu using the script `training/3b_fine_web_edu.py`. **Phase 2** is GRPO-based RL fine-tuning. According to Kye Gomez on X/Twitter (April 21, 2026): *"After the base 3B model is pretrained, we will further fine-tune it with GRPO and a series of high-quality RL datasets."* [Source](https://x.com/KyeGomezB/status/2046387358040301800)

There is **no SFT stage** mentioned between pretraining and GRPO — the base pretrained model goes directly into RL, similar to the DeepSeek-R1 "no cold-start SFT" approach. [Source (blockchain.news)](https://blockchain.news/ainews/openmythos-breakthrough-looped-transformer-moe-rebuild-of-claude-mythos-shows-2-67x-faster-validation-steps)

**Important distinction**: There is a separate project also called "OpenMythos" by KingNish (a cybersecurity LLM trained from scratch) that uses SFT → RLVR. That is a different project from Kye Gomez's architecture work. [Source](https://huggingface.co/blog/KingNish/openmythos)

### 2. Pretraining Hyperparameters

The `3b_fine_web_edu.py` training script uses the following configuration:

- **Optimizer**: AdamW
- **Learning rate schedule**: Linear warmup of 2000 steps → cosine decay
- **Precision**: bfloat16 on H100/A100; float16 + GradScaler on older GPUs
- **Sequence length**: 2048 tokens
- **Distributed training**: PyTorch FSDP (Fully Sharded Data Parallel) via torchrun
- **Dataset**: HuggingFaceFW/fineweb-edu, default `sample-10BT` (30B tokens) for validation; `sample-100BT` for full runs
- **Tokenizer**: openai/gpt-oss-20b via MythosTokenizer
- **Global batch size**: configurable; exact default not publicly specified but dev.to teardown references a 3B throughput benchmark at batch=32

[Source (GitHub README)](https://github.com/kyegomez/OpenMythos) | [Source (Medium/eng.fadishaar)](https://medium.com/@eng.fadishaar/openmythos-the-open-source-reconstruction-of-claude-mythos-that-reframes-what-ai-scaling-actually-32297d4be231) | [Source (dev.to teardown)](https://dev.to/clintjosy/openmythos-teardown-dissecting-the-open-source-reconstruction-of-claude-mythos-9e5)

**Note**: Specific GRPO hyperparameters (group size G, temperature, KL penalty β, learning rate for RL phase) are **not yet specified** in the repository or public communications. The GRPO phase is described as aspirational ("will further fine-tune").

### 3. Model Architecture and Recurrent Depth

OpenMythos implements a three-stage Recurrent-Depth Transformer:

1. **Prelude**: Standard transformer blocks that run once to encode the input
2. **Recurrent Block**: A looped block that iterates up to `max_loop_iters` times (default 16 during training), with weight sharing across iterations
3. **Coda**: Final transformer blocks producing logits

Key architectural features:
- **LTI-stable injection**: Injection matrix with spectral radius ρ(A) < 1 guaranteed by construction. This prevents hidden states from diverging exponentially across loops, making training stable even at high learning rates. From the Parcae architecture (Prairie et al., 2026). [Source](https://github.com/kyegomez/OpenMythos)
- **Sparse MoE**: Fine-grained expert segmentation (DeepSeekMoE-style) with shared and routed experts. Router bias is a buffer (not a gradient parameter) for load balancing.
- **Switchable attention**: MLA (Multi-Latent Attention) or GQA (Grouped Query Attention)
- **ACT (Adaptive Computation Time)**: Dynamic halting that learns when to stop looping per token
- **Model variants**: Pre-configured at 1B, 3B, 10B, 50B, 100B, 500B, 1T parameter scales

[Source](https://github.com/kyegomez/OpenMythos/blob/main/open_mythos/main.py) | [Source (MarkTechPost)](https://www.marktechpost.com/2026/04/19/meet-openmythos-an-open-source-pytorch-reconstruction-of-claude-mythos-where-770m-parameters-match-a-1-3b-transformer/)

### 4. Training Stability via Architecture, Not Hyperparameter Tuning

The key insight for recurrent-depth training is that **the LTI constraint (spectral radius < 1) guarantees ρ(A) < 1 regardless of learning rate or batch noise**. The README states: *"The result: the looped model becomes significantly more robust to hyperparameter selection and trains cleanly even at high learning rates. This is the Parcae architecture."* [Source](https://github.com/kyegomez/OpenMythos)

Practical guidance for training looped models:
- Start with smaller `n_loops` (e.g., 4) for initial validation, then increase gradually
- Memory pressure scales with loop count: 16 loops × seq_len 4096 × batch 8 can be several times that of a normal model
- Train with N loops, evaluate with N+M to exploit depth extrapolation at inference

[Source (openclawapi getting-started)](https://openclawapi.org/en/blog/2026-04-26-openmythos-getting-started)

### 5. Small-Model Efficiency (770M–3B range)

**Parameter efficiency**: A 770M RDT matches a 1.3B standard transformer on identical training data — roughly half the parameters for equivalent downstream quality. [Source](https://www.marktechpost.com/2026/04/19/meet-openmythos-an-open-source-pytorch-reconstruction-of-claude-mythos-where-770m-parameters-match-a-1-3b-transformer/)

**Sample efficiency (convergence speed)**: A community training run on Tiny Shakespeare showed OpenMythos reaching its best validation in **2.67× fewer steps** than nanoGPT. On the docs/datasets.md: *"The looped architecture is more sample-efficient than a standard transformer — same validation loss is reachable with fewer tokens due to faster convergence."* The recommended token budgets for looped models are lower than for standard transformers. [Source](https://github.com/kyegomez/OpenMythos/blob/main/docs/datasets.md) | [Source (blockchain.news)](https://blockchain.news/ainews/openmythos-breakthrough-looped-transformer-moe-rebuild-of-claude-mythos-shows-2-67x-faster-validation-steps)

**Inference throughput**: OpenMythos 3B (MoE) achieves 2,510 tokens/sec vs. 940 tokens/sec for a dense 3B baseline on an A100 at batch=32 — **2.67× faster**. [Source (dev.to)](https://dev.to/clintjosy/openmythos-teardown-dissecting-the-open-source-reconstruction-of-claude-mythos-9e5)

### 6. Scaling Laws for Looped Models

The README claims that both optimal recurrence and optimal token count follow power laws with consistent exponents across scales:
- More test-time loops improve quality following a predictable, saturating exponential decay
- Training dynamics are predictable in a way that looped models were previously thought not to be
- Optimal recurrence and optimal token count follow power laws with consistent exponents across scales

[Source](https://github.com/kyegomez/OpenMythos) | [Source (Medium/eng.fadishaar)](https://medium.com/@eng.fadishaar/openmythos-the-open-source-reconstruction-of-claude-mythos-that-reframes-what-ai-scaling-actually-32297d4be231)

### 7. Related Work: CART (Context-Anchored Recurrent Transformer)

The CART paper (arXiv:2606.01495, May 2026) provides a closely related architecture with published training details:
- **Training setup**: 3,000 steps (~49M tokens), seq_len=512, batch=4, grad_accum=4 (16,384 tokens/step)
- **Key difference**: CART computes K and V once from a multi-layer prelude and reuses them throughout the recurrent core (unlike OpenMythos which recomputes at each loop)
- **Finding**: Prelude depth dominates loop count for performance — P=6 prelude layers is the hyperparameter ordering sweet spot
- The paper explicitly references OpenMythos as prior work

[Source](https://arxiv.org/abs/2606.01495)

## Sources

### Kept
- **GitHub: kyegomez/OpenMythos** (https://github.com/kyegomez/OpenMythos) — Primary source: README, training scripts, architecture docs
- **Blockchain.news article** (https://blockchain.news/ainews/openmythos-breakthrough-looped-transformer-moe-rebuild-of-claude-mythos-shows-2-67x-faster-validation-steps) — Aggregates Kye Gomez's X/Twitter statements about GRPO plans, 2.67× convergence claim
- **Kye Gomez X/Twitter** (https://x.com/KyeGomezB/status/2046387358040301800) — Direct statement about GRPO + RL plans after pretraining
- **DEV Community Teardown** (https://dev.to/clintjosy/openmythos-teardown-dissecting-the-open-source-reconstruction-of-claude-mythos-9e5) — Sequence length (2048), throughput benchmarks, training config summary
- **Medium/eng.fadishaar** (https://medium.com/@eng.fadishaar/openmythos-the-open-source-reconstruction-of-claude-mythos-that-reframes-what-ai-scaling-actually-32297d4be231) — Training config details (AdamW, 2000-step warmup, cosine decay, model variants)
- **OpenClawAPI Getting Started** (https://openclawapi.org/en/blog/2026-04-26-openmythos-getting-started) — Practical training guidance, memory pressure scaling with loops, max_loop_iters=16
- **MarkTechPost** (https://www.marktechpost.com/2026/04/19/meet-openmythos-an-open-source-pytorch-reconstruction-of-claude-mythos-where-770m-parameters-match-a-1-3b-transformer/) — 770M = 1.3B claim, architecture overview
- **DeepWiki: OpenMythos Training** (https://deepwiki.com/jnsereko/OpenMythos/5-training) — FSDP pipeline, streaming dataloader, distributed coordination
- **CART paper (arXiv:2606.01495)** (https://arxiv.org/abs/2606.01495) — Related recurrent-depth architecture with published training hyperparameters
- **HuggingFace: KingNish OpenMythos** (https://huggingface.co/blog/KingNish/openmythos) — Separate cybersecurity project using SFT→RLVR (not Kye Gomez's work, but informative contrast)

### Dropped
- General GRPO explainers (TowardsDataScience, TRL docs, GRPO++ blog) — not OpenMythos-specific
- LinkedIn reposts — derivative content, no unique information
- Swarms/Agora multi-agent papers — unrelated to model training per se
- nanochat-openmythos fork — derivative project, no novel training insights
- Behavioral distillation issue (#20) — interesting but about distillation, not core training approach

## Gaps

1. **GRPO hyperparameters are unspecified**. Kye Gomez has stated intent to use GRPO but has not published the specific configuration (group size G, temperature for rollouts, KL penalty β, learning rate, number of RL steps/epochs). The training repository currently only contains the pretraining script.

2. **No trained weights or checkpoint released**. As of the research date, OpenMythos is an architecture blueprint — no pretrained weights from Kye Gomez exist on HuggingFace. The community fork `fartinalbania/OpenMythos-1.5b-transplant-shell-preview` exists but uses transplanted Qwen weights.

3. **Actual learning rate values** for pretraining are not publicly documented in the README or Medium posts — only the schedule type (warmup + cosine) is specified, not the peak LR or min LR values. These would need to be extracted directly from the `3b_fine_web_edu.py` source.

4. **The 770M = 1.3B equivalence claim** has not been independently reproduced. The 2.67× faster validation claim comes from a single community training run on Tiny Shakespeare. Larger-scale verification on standard benchmarks is missing.

5. **GRPO group size / temperature for small models** is not addressed in any OpenMythos-specific source. Best-practice guidance for GRPO on 4-7B models would need to be sourced from general literature (e.g., GRPO++ blog suggests G=4-16, temperature=1.0 for small models).

### Suggested next steps
- Clone the repository and extract exact hyperparameters from `training/3b_fine_web_edu.py` (peak LR, weight decay, max steps, gradient clipping, batch size per GPU)
- Monitor Kye Gomez's X/Twitter (@KyeGomezB) and the GitHub repo for the GRPO training script when it ships
- Check the OpenMythos Discord for community training run logs with actual hyperparameters
- Compare against the CART paper's published training recipe as a baseline for recurrent-depth transformer training
