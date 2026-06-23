# Research: DeepSeek's Architecture, Training Pipeline, and Recurrent/Looped Work

## Summary
DeepSeek uses standard (non-recurrent) Transformer architectures with their proprietary MLA attention and DeepSeekMoE routing. Their looped/recurrent contributions are limited to attention and MoE components used *by others* in looped architectures (e.g., Ouroboros). R1 training is a 4-stage pipeline: Cold-Start SFT → Reasoning RL (GRPO) → Rejection Sampling SFT → Full RL. DeepSeekMath's GRPO uses G=64, LR=1e-6, KL coefficient=0.04. V3 training uses 16-way PP, 64-way EP, ZeRO-1 DP on 2048 H800 GPUs with their custom DualPipe algorithm.

## Findings

### 1. DeepSeek-R1 Training Pipeline — 4-Stage Recipe (No SFT Warmup Before GRPO)

Contrary to the "SFT warmup before GRPO" framing, the R1 pipeline is more nuanced:

**DeepSeek-R1-Zero** (exploratory variant): DeepSeek-V3-Base → **pure GRPO RL** with zero SFT. Emerged with powerful reasoning but had readability/language-mixing issues. [Source](https://arxiv.org/abs/2501.12948)

**DeepSeek-R1** (production) — 4 stages:

- **Stage 1 — Cold Start SFT**: "Thousands" of long Chain-of-Thought (CoT) examples (exact count not disclosed — described only as "thousands") collected and used to fine-tune DeepSeek-V3-Base. These CoT examples use a structured format (e.g., `<reasoning>` + `<answer>` tags) to establish readable, human-friendly output patterns before any RL. This is a small, quality-focused dataset designed to bootstrap readability without compromising reasoning potential. [Source](https://arxiv.org/html/2501.12948v1)

- **Stage 2 — Reasoning-Oriented RL**: GRPO applied on top of the Stage-1 SFT model with **rule-based rewards** (exact match for math, test case execution for code, format adherence, language consistency penalties). No reward model — pure verifiable rewards. This is where reasoning capabilities emerge.

- **Stage 3 — Rejection Sampling + SFT**: From the converged Stage-2 RL checkpoint, generate multiple completions per prompt via rejection sampling. Collect **~600k high-quality reasoning samples** (correct solutions only), combine with **~200k non-reasoning data** (writing, Q&A, translation, etc.), and SFT **DeepSeek-V3-Base** (fresh base model, not the RL checkpoint). Total ≈ 800k SFT samples.

- **Stage 4 — Full RL**: A second RL pass on the Stage-3 SFT model, combining **rule-based reasoning rewards** + a **helpful/safety reward model** for alignment. This produces the final DeepSeek-R1.

**Key insight**: SFT and RL are interleaved, not just "warmup then RL." The Cold Start SFT (Stage 1) is minimal — just enough to make outputs readable before RL takes over. [Source](https://aman.ai/primers/ai/deepseek-R1/)

### 2. DeepSeek-V3 Architecture — No Recurrent Depth, Standard Transformer

DeepSeek-V3 is a **standard feed-forward Transformer** with **no recurrent depth or loop blocks**:

- **61 layers**, hidden dimension **7168**
- Uses **Multi-head Latent Attention (MLA)** (inherited from DeepSeek-V2): low-rank KV joint compression to reduce KV cache memory. Keys and values are compressed into a latent space, then up-projected per attention head. RoPE applied via decoupled mechanism. [Source](https://arxiv.org/html/2412.19437v1)
- **DeepSeekMoE architecture**: 
  - **1 shared expert** + **256 routed experts** per MoE layer
  - Each expert has intermediate hidden dimension **2048**
  - **Top-8 routing** (K_r=8): 8 routed experts activated per token, plus the always-on shared expert → total 9 experts active
  - **671B total parameters, 37B activated per token**
- **Auxiliary-loss-free load balancing**: Instead of a traditional auxiliary loss that can interfere with model quality, DeepSeek-V3 uses a **bias update strategy**: a per-expert bias term is dynamically adjusted after each training step. If an expert is overloaded, its bias is reduced; if underloaded, increased. Bias update speed γ=0.001 for first 14.3T tokens, γ=0 for final 500B tokens. A tiny sequence-level auxiliary loss (α=0.0001) is kept only to prevent extreme intra-sequence imbalance. [Source](https://arxiv.org/pdf/2412.19437)
- **Multi-Token Prediction (MTP)**: auxiliary training objective predicting future tokens at multiple depths — improves training signal density. [Source](https://arxiv.org/html/2412.19437v1)
- Uses SwiGLU activation, RoPE, RMSNorm — standard Transformer components.
- The architecture is described as "still within the Transformer (Vaswani et al., 2017) framework." [Source](https://verbraucherschutzforum.berlin/wp-content/uploads/2025/01/DeepSeek_V3.pdf)

### 3. DeepSeekMath GRPO Hyperparameters (arXiv 2402.03300)

From Section 4 / Appendix of the DeepSeekMath paper:

| Hyperparameter | Value |
|---|---|
| **Group size (G)** | **64** outputs sampled per problem |
| **Policy model LR** | **1e-6** |
| **KL coefficient (β)** | **0.04** |
| **Batch size** | **1024** |
| **Max generation length** | **1024** tokens |
| **Reward model LR** | 2e-5 (trained on DeepSeekMath-Base 7B) |
| **Temperature** | **Not explicitly stated** in accessible materials (likely 1.0, standard GRPO practice) |

**Pre-training hyperparameters** (for the base DeepSeekMath 7B model, before GRPO):
- AdamW optimizer: β₁=0.9, β₂=0.95, weight_decay=0.1
- Max LR: **5.3e-4**, multi-step schedule: peak after 2000 warmup steps, drops to 31.6% after 80% of training, 10% after 90%
- Batch size: 4M tokens, 4K context length
- Continued pre-training from DeepSeek-Coder-Base-v1.5 7B with 120B math tokens

[Source 1](https://arxiv.org/pdf/2402.03300v2) | [Source 2](https://deepseek-r1.com/the-secret-behind-deepseek-1-deepseekmath-and-grpo-details/) | [Source 3](https://deepseekai.guide/models/deepseek-math/)

### 4. DeepSeek's Multi-GPU Training Strategy

**DeepSeek-V3 training infrastructure** (14.8T tokens, 2.788M H800 GPU hours):

| Parallelism Type | Configuration |
|---|---|
| **Pipeline Parallelism (PP)** | 16-way, using proprietary **DualPipe** algorithm |
| **Expert Parallelism (EP)** | 64-way, spanning 8 nodes (8 GPUs per node) |
| **Data Parallelism (DP)** | ZeRO-1 (optimizer state sharding only) |

Total: 16 PP × 64 EP × 2 DP = **2048 NVIDIA H800 GPUs** across 256 nodes (8 GPUs/node). Inter-node: InfiniBand. Intra-node: NVLink + NVSwitch.

[Source](https://arxiv.org/pdf/2412.19437) | [Source](https://aman.ai/primers/ai/deepseekV3/)

**DualPipe** — key innovation:
- Bidirectional pipeline parallelism that overlaps forward and backward **computation with communication** in both directions simultaneously
- Eliminates pipeline bubbles more effectively than standard 1F1B schedules
- Critical for making cross-node expert routing (all-to-all communication) economically feasible
- Open-source: [github.com/deepseek-ai/DualPipe](https://github.com/deepseek-ai/DualPipe)

**Training framework**: Custom **HAI-LLM** framework (built from scratch by DeepSeek engineers), **not** Megatron, DeepSpeed, or FSDP. The framework handles the hybrid parallelism, FP8 mixed precision, and custom communication kernels.

**FP8 training**: First large-scale validation of FP8 mixed-precision training on a 671B model. Used block-wise quantization with per-tile scaling for both GEMM operations in forward and backward passes.

**For smaller models (DeepSeek LLM 7B/67B)**: Standard data parallelism on a smaller cluster. 7B: batch size 2304, LR 4.2e-4. 67B: batch size 4608, LR 3.2e-4. Both trained on 2T tokens. [Source](https://github.com/deepseek-ai/DeepSeek-LLM)

### 5. DeepSeek Papers on Recurrent Depth / Looped Transformers

**DeepSeek has NO papers on recurrent depth or looped transformer architectures.**

The looped/recurrent transformer work comes from other research groups:
- **Parcae** (arXiv 2604.12946, Apr 2026) — "Scaling Laws For Stable Looped Language Models" — authors: Hayden Prairie, Zachary Novack, Taylor Berg-Kirkpatrick, Daniel Y. Fu (UCSD/Berkeley, **not DeepSeek**). Introduces LTI stability constraints for looped training and first predictable scaling laws for looped models. [Source](https://arxiv.org/pdf/2604.12946) | [GitHub](https://github.com/sandyresearch/parcae)
- **Ouroboros** (GitHub: asherk7/Ouroboros) — an open-source recurrent-depth/P/R/C looped transformer that *uses* DeepSeek components (MLA, DeepSeekMoE) but is not a DeepSeek project. [Source](https://github.com/asherk7/Ouroboros)
- **"Two-Scale Latent Dynamics for Recurrent-Depth Transformers"** (arXiv 2509.23314, Nov 2025) — studies geometry of looped block iterates. Not from DeepSeek.
- **"Loop, Think, & Generalize"** (arXiv 2604.07822, Apr 2026) — implicit reasoning in recurrent-depth transformers. Not from DeepSeek.

**DeepSeek's indirect contributions** to looped architectures:
- **MLA** (Multi-head Latent Attention from DeepSeek-V2): adopted in Ouroboros and other looped projects for KV cache efficiency during repeated iterations
- **DeepSeekMoE**: fine-grained expert segmentation + shared expert isolation, adopted for MoE in looped transformer projects

DeepSeek's V3.2 introduced **DeepSeek Sparse Attention (DSA)** for long-context efficiency, but this is about token-level sparsity, not depth-wise recurrence.

### 6. DeepSeek's Small-Model Training Efficiency (4-7B Range)

**DeepSeek LLM 7B** (arXiv 2401.02954, Jan 2024):
- 30 layers, trained on 2T tokens, batch size 2304, LR 4.2e-4
- Standard dense Transformer, 4K context
- Scaling laws paper: established compute-optimal token/parameter ratios for 7B and 67B scales
- Architecture adjusted specifically for "pipeline partitioning to optimize training and inference" (30 layers chosen for PP-friendly partitioning)

**DeepSeek-Coder** (arXiv 2401.14196, Jan 2024):
- Sizes: **1.3B, 5.7B, 6.7B, 33B**
- All trained from scratch on 2T tokens (87% code, 13% natural language, 80+ languages)
- 16K context window
- Architecture: standard dense Transformer (no MoE at these scales)

**DeepSeekMath 7B** (arXiv 2402.03300, Feb 2024):
- Continued from DeepSeek-Coder-Base-v1.5 7B + 120B math tokens
- **GRPO fine-tuning** (see Finding 3 for hyperparameters)
- 4K context during training

**DeepSeek-R1 Distilled Models** (Jan 2025):
- **SFT-only distillation** (no RL on distilled models)
- 1.5B, 7B, 8B (based on Qwen2.5/Llama-3 series), plus 14B, 32B, 70B
- Trained on **800k samples** (600k reasoning chains + 200k non-reasoning) from DeepSeek-R1's intermediate checkpoint
- This is purely supervised — no GRPO or RL at distill scale

**DeepSeek's overall small-model philosophy**: Dense architectures up to ~7B, standard training recipes with well-tuned LR/batch size. MoE only at 671B (V3) and larger (V4). No public work on looped/recurrent architectures at any scale.

## Sources

### Kept (primary sources)
- **DeepSeek-R1 paper** (arXiv 2501.12948) — definitive source for R1 training pipeline. [Link](https://arxiv.org/abs/2501.12948)
- **DeepSeek-V3 Technical Report** (arXiv 2412.19437) — architecture, MoE routing, DualPipe, parallelism strategy. [Link](https://arxiv.org/pdf/2412.19437)
- **DeepSeekMath paper** (arXiv 2402.03300) — GRPO hyperparameters, math pre-training recipe. [Link](https://arxiv.org/pdf/2402.03300v2)
- **DeepSeek LLM paper** (arXiv 2401.02954) — 7B/67B scaling laws, small-model training recipes. [Link](https://arxiv.org/abs/2401.02954)
- **DeepSeek-Coder paper** (arXiv 2401.14196) — small code model training. [Link](https://arxiv.org/abs/2401.14196)
- **DeepSeekMoE paper** (arXiv 2401.06066) — MoE architecture details. [Link](https://arxiv.org/pdf/2401.06066)
- **Aman's AI Journal — DeepSeek V3 primer** — well-structured summary of V3 architecture specifics. [Link](https://aman.ai/primers/ai/deepseekV3/)
- **Aman's AI Journal — DeepSeek R1 primer** — accurate pipeline breakdown. [Link](https://aman.ai/primers/ai/deepseek-R1/)
- **DeepSeek DualPipe GitHub** — open-source bidirectional pipeline parallelism. [Link](https://github.com/deepseek-ai/DualPipe)
- **Parcae paper** (arXiv 2604.12946) — confirmed NOT a DeepSeek paper (UCSD/Berkeley). [Link](https://arxiv.org/pdf/2604.12946)
- **DeepSeek-R1.com GRPO details** — summary of DeepSeekMath hyperparameters. [Link](https://deepseek-r1.com/the-secret-behind-deepseek-1-deepseekmath-and-grpo-details/)
- **Creative Strategies — Dispelling DeepSeek Myths** — clean breakdown of V3 parallelism numbers. [Link](https://creativestrategies.com/dispelling-deepseek-myths-studying-v3/)

### Dropped
- Multiple Medium blog posts summarizing the same papers — redundant with primary sources.
- Ouroboros GitHub — informative for context (uses DeepSeekMoE/MLA in looped architecture) but not a DeepSeek project.
- OpenMythos / various looped transformer papers — not relevant to DeepSeek specifically.
- "Two-Scale Latent Dynamics" paper — not a DeepSeek paper, but cited for looped transformer dynamics understanding.

## Gaps

1. **DeepSeekMath GRPO temperature**: The paper does not explicitly state the sampling temperature used for generating G=64 samples. Common practice (and TRL defaults) suggest temperature=1.0, but this is unconfirmed. Worth checking the paper's appendix section directly.

2. **R1 Cold Start SFT exact count**: Described only as "thousands" — no exact number disclosed in the paper. Could be anywhere from 2,000 to 20,000.

3. **HAI-LLM framework details**: DeepSeek's custom training framework is mentioned but not publicly documented beyond the V3 paper's brief description. No open-source release.

4. **GRPO hyperparameters for R1 Stage 2**: The R1 paper references GRPO but does not re-specify hyperparameters — likely reused DeepSeekMath settings (G=64, LR=1e-6, KL=0.04) but not confirmed in the paper.

5. **DeepSeek V3 MoE layer placement**: The V3 paper states 61 layers but doesn't specify which layers are MoE vs. dense. Likely MoE in all layers except the first few (common practice), but exact placement not verified.

## Supervisor Coordination

No coordination needed — research completed successfully across all six angles.
