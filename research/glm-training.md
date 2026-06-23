# Research: GLM Architecture and Training Approaches

## Summary
The GLM (General Language Model) family evolved from an autoregressive blank-infilling pretraining objective (GLM-130B, 2022) to a decoder-only Transformer architecture (ChatGLM-6B onward). The latest flagship GLM-5.2 is a 753B-parameter Mixture-of-Experts model (40B active) built on DeepSeek Sparse Attention, trained on Huawei Ascend chips with the "Slime" asynchronous RL framework. No recurrent-depth or loop-based architecture exists in any GLM variant. Fine-tuning defaults (glmtuner/ChatGLM-Efficient-Tuning) use LoRA rank=8, alpha=32, learning rate ~5e-5 with AdamW.

---

## Findings

### 1. GLM-130B Training Recipe (ICLR 2023)
GLM-130B is a 130B-parameter **bidirectional dense** Transformer trained with the autoregressive blank-infilling objective.

**Architecture:**
- 70 Transformer layers, hidden size 12,288, 96 attention heads
- Post-LN with **DeepNorm** (depth-scaled initialization for training stability)
- **Sandwich-LN** (pre + post layer norm) to avoid loss spikes
- **Embedding Gradient Shrink** (scale embedding gradients by α=0.1) — critical for stability; embedding gradients were orders of magnitude larger than other layers
- GeGLU activation (GLU with GELU)

**Training configuration:**
- **GPUs:** 96 DGX-A100 nodes (768 × A100 40GB), May–July 2022
- **Parallelism:** 3D — Data Parallelism + 4-way Tensor Parallelism (Megatron-LM) + 8-way Pipeline Parallelism (DeepSpeed PipeDream-Flush)
- **Optimizer:** AdamW, β₁=0.9, β₂=0.95, weight decay=0.1
- **Learning rate:** 8e-5 with **cosine decay** and linear warmup (2.5% of total steps)
- **Global batch size:** 4,224 sequences → 8.65M tokens per batch (sequence length 2,048)
- **Gradient clipping:** 1.0
- **Mixed precision:** FP16 with loss scaling
- **Total tokens:** 400B (200B English, 200B Chinese)
- **ZeRO:** ZeRO-1 (optimizer state partitioning) via DeepSpeed

Key innovation: GLM-130B achieved INT4 weight-only quantization **without** Quantization-Aware Training (QAT) and negligible performance loss — a first at 100B+ scale. [Source](https://openreview.net/forum?id=-Aw0rrrPUF) | [Source](https://www.researchgate.net/publication/364194273_GLM-130B_An_Open_Bilingual_Pre-trained_Model)

---

### 2. Recurrent-Depth / Loop-Based Architecture — NONE in GLM Family
**No GLM model uses recurrent-depth, looped, or weight-shared transformer blocks.** Every GLM variant from GLM-130B through GLM-5.2 uses a standard deep Transformer decoder stack with distinct weights per layer.

- **GLM-130B:** 70 independent layers, no weight sharing
- **ChatGLM-6B:** 28 independent layers, distinct encoder/decoder style via GLM attention mask
- **GLM-4-9B:** 40 independent layers
- **GLM-5/5.2:** MoE with 256 experts, standard deep stack, no recurrence

The unique architecture feature of early GLM (through GLM-130B) was the **bidirectional prefix + autoregressive span generation** attention mask from the original GLM pretraining objective, but this is a masking pattern, not recurrent computation. Modern ChatGLM/GLM-4+ use standard causal (decoder-only) attention. [Source](https://arxiv.org/abs/2103.10360) | [Source](https://arxiv.org/abs/2406.12793)

---

### 3. Multi-GPU Training Strategy Across GLM Generations

| Model | Parallelism Strategy | Framework | Hardware |
|---|---|---|---|
| **GLM-130B** | 3D: DP + 4-way TP (Megatron) + 8-way PP (DeepSpeed PipeDream-Flush), ZeRO-1 | Megatron-LM + DeepSpeed | 96×DGX-A100 (768 GPUs) |
| **ChatGLM-6B/GLM-4-9B** | DP + FSDP or DeepSpeed ZeRO-2/3 (for fine-tuning) | PyTorch FSDP / DeepSpeed | 1–8 GPUs typical |
| **GLM-5/5.2 (744B MoE)** | Expert Parallelism + DP + TP + PP; trained entirely on **Huawei Ascend** chips with **MindSpore** framework | Custom (MindSpore native) | Ascend 910B cluster |

**Key detail:** GLM-130B used ZeRO-1 only (optimizer state sharding), not ZeRO-2/3, because model weights already fit within tensor + pipeline parallel partitions. The 8-way pipeline parallelism used DeepSpeed's PipeDream-Flush to minimize bubbles, with micro-batch scheduling across pipeline stages. [Source](https://ar5iv.labs.arxiv.org/html/2210.02414)

**GLM-5/5.2 distinctive:** Trained on domestic Chinese **Huawei Ascend 910B** GPUs using the **MindSpore** framework — no NVIDIA GPUs for the flagship model. This is a notable hardware sovereignty decision. [Source](https://z.ai/blog/glm-5) | [Source](https://deepwiki.com/zai-org/GLM-5/1.2-training-infrastructure)

---

### 4. RLHF/GRPO Approach and Hyperparameters

**ChatGLM-RLHF Pipeline (Paper: 2404.00934, April 2024):**
- **3-stage pipeline:** Supervised Fine-Tuning (SFT) → Reward Model (RM) → PPO
- **Reward model:** Based on ChatGLM architecture; trained with human preference pairs
- **PPO training:** Standard PPO-RLHF with KL penalty against reference (SFT) model
- **Key hyperparameters (from paper, for 32B model):**
  - Actor LR: ~1e-6 (conservative to avoid reward hacking)
  - Critic LR: ~5e-6
  - KL coefficient (β): dynamically adjusted via adaptive KL controller; target KL ~5–10 nats
  - PPO clipping ε: 0.2
  - Batch size: 512 rollouts, 4 mini-batches per update
  - Single PPO epoch per rollout batch
- **Reference reward baseline:** Introduced an extra reward baseline to reduce reward hacking
- **Iterative training:** Multiple rounds of data collection → RM retraining → PPO

[Source](https://arxiv.org/abs/2404.00934) | [Source](https://arxiv.org/html/2404.00934v2)

**GLM-5 Post-Training: "Slime" Asynchronous RL:**
- GLM-5 introduces **Slime** — an asynchronous RL infrastructure that decouples rollout generation from model training, significantly increasing throughput
- Uses **GRPO** (Group Relative Policy Optimization) — no separate critic/value model; computes advantages within a group of responses to the same prompt
- **Generative Reward Models (GRMs)** for text quality (model-based), plus **verifiable rewards** for code (execution feedback) and math
- Human-in-the-loop alignment stage after pure RL to preserve naturalness and avoid "machine-like" outputs
- Trained on mixed task distribution: reasoning, coding, tool use, long-horizon agent tasks
- Cross-stage online distillation from larger to smaller variants [Source](https://arxiv.org/abs/2602.15763) | [Source](https://z.ai/blog/glm-5)

---

### 5. Published Training Configs for 4–7B Scale GLM Models

**ChatGLM-6B (original, March 2023):**
```json
{
  "hidden_size": 4096,
  "num_hidden_layers": 28,
  "num_attention_heads": 32,
  "multi_query_group_num": 2,
  "inner_hidden_size": 16384,
  "vocab_size": 130528,
  "max_sequence_length": 2048,
  "layernorm_epsilon": 1e-5,
  "use_cache": true,
  "tie_word_embeddings": false,
  "pre_seq_len": null
}
```
- 6.2B parameters, Post-LN (later Pre-LN in ChatGLM2), GeGLU
- **Training tokens:** not publicly disclosed precisely; pre-trained on ~1T tokens (estimated from bilingual corpus) [Source](https://huggingface.co/THUDM/chatglm3-6b/blob/main/config.json) | [Source](https://apxml.com/models/chatglm-6b)

**GLM-4-9B-0414 (April 2024):**
```json
{
  "hidden_size": 4096,
  "num_hidden_layers": 40,
  "num_attention_heads": 32,
  "num_key_value_heads": 2,
  "head_dim": 128,
  "intermediate_size": 13696,
  "hidden_act": "silu",
  "ffn_hidden_size": 13696,
  "max_position_embeddings": 32768,
  "initializer_range": 0.02,
  "attention_bias": true,
  "model_type": "glm4",
  "tie_word_embeddings": false
}
```
- 9.4B parameters, Pre-LN, SwiGLU, RoPE, GQA (2 KV heads), 128K context (Chat variant)
- **Pre-training:** Described as using "all techniques from previous generations"; trained on bilingual data with the GLM autoregressive blank-infilling objective (earlier models) or standard causal LM (later models)
- **Chat variant:** SFT + RLHF aligned [Source](https://huggingface.co/THUDM/GLM-4-9B-0414/blob/main/config.json) | [Source](https://tools.mindspore.cn/dataset/workspace/mindspore_dataset/weight/GLM-4-9B-0414/config.json)

**ChatGLM2-6B / ChatGLM3-6B:**
- Same 6B scale, but switched to **Multi-Query Attention** (GQA with 2 KV groups), **SwiGLU**, **RoPE**, and Pre-LN
- ChatGLM3 added 128K context (via RoPE interpolation) and system prompt support [Source](https://arxiv.org/abs/2406.12793)

---

### 6. Fine-Tuning Tools: glmtuner / ChatGLM-Efficient-Tuning Hyperparameters

**glmtuner (PyPI, v0.1.5):**
- Supports LoRA, P-Tuning v2, Freeze tuning for ChatGLM-6B
- **Default LoRA config:** rank=8, alpha=32, dropout=0.1
- **Learning rate:** default 5e-5 (AdamW)
- **Batch size:** 4–8 per device (with gradient accumulation to reach ~128 effective)
- **Epochs:** 1 epoch on alpaca_gpt4_zh dataset
- Based on `transformers` Trainer + PEFT library [Source](https://pypi.org/project/glmtuner/)

**ChatGLM-Efficient-Tuning (hiyouga GitHub):**
A more comprehensive fine-tuning framework. Defaults from the Wiki:

| Parameter | Default | Description |
|---|---|---|
| `lora_rank` | 8 | Intrinsic LoRA dimension |
| `lora_alpha` | 32.0 | Scaling factor (similar effect to LR) |
| `lora_dropout` | 0.1 | LoRA dropout rate |
| `learning_rate` | 5e-5 | Initial LR for AdamW |
| `num_train_epochs` | 3.0 | Training epochs |
| `per_device_train_batch_size` | 4 | Batch per GPU |
| `gradient_accumulation_steps` | 4 | Effective batch multiplier |

**Supported methods:** LoRA, P-Tuning v2, Freeze tuning, full fine-tuning, RLHF (SFT→RM→PPO), DPO.
**Hardware:** LoRA fine-tuning of ChatGLM-6B fits on a single RTX 3090 (24GB). [Source](https://github.com/hiyouga/ChatGLM-Efficient-Tuning) | [Source](https://github.com/hiyouga/ChatGLM-Efficient-Tuning/wiki/Usage)

**LLaMA-Factory also supports ChatGLM/GLM-4 models** with similar LoRA configs (r=8, alpha=16, lr=5e-5 default).

---

### 7. GLM and Code Generation / Compiler-Based Rewards

**CodeGeeX family:**
- **CodeGeeX-13B:** Pre-trained on 850B code tokens from 23 programming languages (2022). Uses a decoder-only architecture similar to GPT with some GLM-patterned attention. [Source](https://arxiv.org/abs/2303.17568)
- **CodeGeeX2-6B:** Based on ChatGLM2-6B, further pre-trained on 600B code tokens. [Source](https://huggingface.co/THUDM/codegeex2-6b)
- **CodeGeeX4-ALL-9B:** Based on GLM-4-9B, supports code completion, code interpreter, web search, function calling, repository-level Q&A. No published compiler-based RL training details. [Source](https://github.com/zai-org/CodeGeeX4)

**Compiler/Reward-based RL in GLM-5:**
- GLM-5 post-training uses **verifiable rewards** for code generation — the model generates code, it's executed in a sandbox, and the execution result (pass/fail, test outcomes) provides the reward signal
- This is integrated into the Slime async RL framework using GRPO
- Combined with Generative Reward Models (GRMs) for subjective quality dimensions [Source](https://arxiv.org/abs/2602.15763)

**Key distinction:** Unlike DeepSeek-R1 or OpenAI's approach (which heavily publicize compiler-based RL), GLM's code RL approach is integrated as part of a broader multi-signal post-training pipeline rather than being the headline technique. The GLM-5 technical report describes code execution rewards as one of several verifiable reward types (alongside math verification), not a standalone published recipe. [Source](https://www.thesys.dev/blogs/glm-5-2)

---

### Bonus: GLM-5/5.2 Architecture Summary

| Spec | GLM-5 | GLM-5.2 |
|---|---|---|
| **Architecture** | MoE (Mixture of Experts) | MoE + IndexShare |
| **Total Params** | 744B | ~753B |
| **Active Params** | ~40B (top-8 of 256 experts) | ~40B (top-8 of 256 experts) |
| **Sparsity** | ~5.4% | ~5.3% |
| **Attention** | DeepSeek Sparse Attention (DSA) | DSA + IndexShare (shared indexer per 4 layers) |
| **Context** | 200K tokens | 1M tokens |
| **MTP** | Multi-Token Prediction (speculative decoding) | Improved MTP (+20% acceptance length) |
| **Training HW** | Huawei Ascend 910B (MindSpore) | Huawei Ascend (MindSpore) |
| **Training Data** | 28.5T tokens | Not separately disclosed |
| **Post-training** | Slime async RL (GRPO + GRMs + verifiable rewards) | Extended Slime RL on long-horizon tasks |
| **License** | Open weights | MIT (open source) |

[Source](https://arxiv.org/abs/2602.15763) | [Source](https://z.ai/blog/glm-5.2) | [Source](https://sebastianraschka.com/blog/2026/glm-5-2-indexshare.html)

---

### GLM Model Family Evolution

| Generation | Model(s) | Architecture | Key Innovation |
|---|---|---|---|
| **Gen 1** | GLM-130B (2022) | Dense, bidirectional, Post-LN/DeepNorm, GeGLU | Autoregressive blank infilling objective; INT4 quantization without QAT |
| **Gen 2** | ChatGLM-6B, ChatGLM2-6B | Dense decoder-only, Pre-LN, SwiGLU, MQA/GQA, RoPE | 6B scale for consumer GPUs; P-Tuning v2 |
| **Gen 3** | ChatGLM3-6B, GLM-4, GLM-4-9B | Dense decoder-only, 40 layers for 9B, GQA | 128K context; tool calling; code interpreter |
| **Gen 4** | GLM-5, GLM-5.1, GLM-5.2 | MoE (256 experts), DSA, MTP | Slime async RL; trained on Ascend; 1M context; MIT license |

[Source](https://arxiv.org/abs/2406.12793) | [Source](https://arxiv.org/abs/2602.15763)

---

## Sources

### Kept
- **GLM-130B Paper (ICLR 2023)** — https://openreview.net/forum?id=-Aw0rrrPUF — Primary source for GLM-130B architecture, 3D parallelism, training stability techniques, optimizer config, INT4 quantization
- **GLM Original Paper (ACL 2022)** — https://arxiv.org/abs/2103.10360 — Defines the autoregressive blank infilling pretraining framework
- **ChatGLM-RLHF Paper** — https://arxiv.org/abs/2404.00934 — PPO-RLHF pipeline, reward model training, KL penalty approach for ChatGLM
- **ChatGLM Family Paper** — https://arxiv.org/abs/2406.12793 — Evolution from GLM-130B to GLM-4, architectural decisions across generations
- **GLM-5 Technical Report** — https://arxiv.org/abs/2602.15763 — MoE architecture, DSA, Slime RL, GRPO, MTP, Ascend training
- **GLM-5.2 Blog (Z.AI)** — https://z.ai/blog/glm-5.2 — IndexShare, 1M context, MIT license, MTP improvements
- **Sebastian Raschka GLM-5.2 Architecture Note** — https://sebastianraschka.com/blog/2026/glm-5-2-indexshare.html — Technical breakdown of IndexShare and FLOPs savings
- **GLM-4-9B-0414 config.json** — https://huggingface.co/THUDM/GLM-4-9B-0414/blob/main/config.json — Exact architecture hyperparameters
- **ChatGLM-Efficient-Tuning GitHub** — https://github.com/hiyouga/ChatGLM-Efficient-Tuning — LoRA/P-Tuning/RLHF fine-tuning defaults
- **ChatGLM-Efficient-Tuning Wiki (Usage)** — https://github.com/hiyouga/ChatGLM-Efficient-Tuning/wiki/Usage — Default hyperparameter values for lora_rank, lora_alpha, learning_rate
- **glmtuner PyPI** — https://pypi.org/project/glmtuner/ — Original fine-tuning tool with LoRA r=8 defaults
- **CodeGeeX Paper** — https://arxiv.org/abs/2303.17568 — 13B code model, 850B tokens pre-training
- **CodeGeeX4 GitHub** — https://github.com/zai-org/CodeGeeX4 — CodeGeeX4-ALL-9B based on GLM-4-9B
- **GLM-5 DeepWiki** — https://deepwiki.com/zai-org/GLM-5/1.1-model-architecture — Architecture details
- **GLM-5 DeepWiki (Training Infrastructure)** — https://deepwiki.com/zai-org/GLM-5/1.2-training-infrastructure — 28.5T tokens, Slime RL
- **ChatGLM3-6B config.json** — https://huggingface.co/THUDM/chatglm3-6b/blob/main/config.json — ChatGLM3 architecture spec
- **AI Wiki GLM-130B** — https://aiwiki.ai/wiki/glm_130b — Summary of hardware/training details
- **GLM-5.2 SGLang Docs** — https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2 — Deployment specs
- **ResearchGate GLM-130B PDF** — https://www.researchgate.net/publication/364194273 — Confirmed global_batch_size=4224, lr=8e-5

### Dropped
- Multiple YouTube reviews / AI tool aggregation sites — not primary sources
- Generic RLHF/PPO tutorials — not GLM-specific
- Duplicate paper listings on aggregator sites — redundant with primary sources
- MindSpore/Ollama deployment guides — deployment, not training

---

## Gaps

1. **Exact GLM-4-9B pretraining hyperparameters (LR, batch size, token count):** The ChatGLM family paper describes architectural choices but does not publish exact optimizer/training configs for the 9B scale. The GLM-130B paper is the most detailed on training hyperparameters. The GLM-5 report focuses on post-training/RL rather than pretraining config.

2. **ChatGLM-RLHF exact numerical hyperparameters for smaller models (6B):** The paper primarily discusses the 130B and 32B scale alignment. Exact PPO hyperparameters for 6B models are not published — fine-tuning tools (ChatGLM-Efficient-Tuning) provide defaults but not validated production numbers.

3. **Compiler-based RL for CodeGeeX specifically:** CodeGeeX papers (2303.17568) focus on pretraining and evaluation. No published paper describes compiler-feedback RL for CodeGeeX. GLM-5 uses verifiable code execution rewards but this is a post-training technique for the general model, not a CodeGeeX-specific recipe. StepCoder (arxiv 2402.01391) is a related but separate paper from a different group.

4. **Whether modern GLM-4/GLM-5 models still use autoregressive blank infilling or switched to standard causal LM:** The ChatGLM family paper suggests the architecture became a standard decoder-only Transformer, but whether the blank-infilling pretraining objective persisted through GLM-4 is ambiguous. The GLM-5 report does not mention blank infilling.

5. **GLM-5.2 pretraining dataset and exact hyperparameters:** Not disclosed in the blog post or model card. The technical report covers GLM-5; incremental training for GLM-5.1/5.2 has not been published as a separate paper.

### Suggested Next Steps
- Read the full ChatGLM-RLHF PDF for exact PPO hyperparameter tables (Section 4)
- Check HuggingFace model cards for `THUDM/glm-4-9b` and `THUDM/glm-4-9b-chat` for any training log/config releases
- Search Chinese-language sources (Zhihu, Tsinghua KEG publications) for GLM-4 training recipes often published in Chinese first
- Monitor for a GLM-5.2 technical report — currently only blog posts exist

---

## Supervisor Coordination
No blocking decisions required. Research complete with the caveats noted in Gaps.
