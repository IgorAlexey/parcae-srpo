# Research: RWKV7/RWKV8 Training Configurations — Multi-GPU, RLHF/GRPO, and Model Parallelism

## Summary
RWKV uses PyTorch Lightning + DeepSpeed ZeRO (Stage 1–3) for multi-GPU training with pure data parallelism — no tensor or pipeline parallelism needed at moderate scales. The offical pretraining recipe uses bf16 precision, AdamW with **beta2=0.99** (not 0.999), a "delayed" cosine LR schedule from 4e-4→1e-5, weight decay 0.1, and increasing batch sizes mid-training. For GRPO/RLHF, the ecosystem is nascent: `rwkvtune` provides a TRL-style GRPOTrainer (v0.1.0), and OpenMOSE/RWKV-LM-RLHF offers DPO/ORPO tooling for RWKV v6/v7. The key RWKV architectural advantage for parallel training is that its WKV operation is fully parallelizable at training time (like a Transformer) while running recurrently at inference (O(1) memory per token, no KV cache).

---

## Findings

### 1. Multi-GPU Training: DeepSpeed ZeRO + PyTorch Lightning
- **Official pipeline**: PyTorch Lightning `Trainer` with `strategy="deepspeed_stage_2"` (or stage 1 for fine-tuning). [Source](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/train_temp/demo-training-prepare.sh)
- **Large-scale verified config**: 7.2B RWKV7 trained on **4 nodes × 8 H100s (32 GPUs)** at ctx8192 with DeepSpeed ZeRO Stage 2 + gradient checkpointing, achieving **263k tokens/s** (~36% MFU). [Source](https://github.com/BlinkDL/RWKV-LM)
- **Fine-tuning preference**: `deepspeed_stage_1` recommended; fall back to stage 2 if VRAM-limited. [Source](https://wiki.rwkv.com/RWKV-Fine-Tuning/LoRA-Fine-Tuning.html)
- **ZeRO Stage 3**: Supported but JIT compilation must be disabled to avoid compatibility issues. [Source](https://deepwiki.com/BlinkDL/RWKV-LM/4-training-pipeline)
- **Precision**: bf16 universally preferred; tf32 used for smaller-scale experimentation.
- **Older confusion**: Some older references claimed RWKV "does not support Data Parallelism" — this was a documentation error. RWKV v5+ uses standard DeepSpeed DP+ZeRO. [Source](https://githubissues.com/microsoft/unilm/1243)
- **GPT-NeoX integration**: EleutherAI's gpt-neox (Megatron-based) supports RWKV with pipeline parallelism, though this is a separate codebase. [Source](https://github.com/EleutherAI/gpt-neox)

### 2. Optimizer, Learning Rates, Batch Sizes (Official Pretraining)
- **Optimizer**: AdamW with **beta1=0.9, beta2=0.99** (notably lower than the standard 0.999), eps=1e-8. The lower beta2 is a deliberate choice for RWKV training stability. [Source](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/train_temp/demo-training-prepare.sh)
- **LR schedule**: "Delayed" cosine decay — LR held constant at 3e-4–4e-4 for the first ~15B tokens, then cosine-decayed to 1e-5. [Source](https://github.com/BlinkDL/RWKV-LM) — README RWKV-loss graph
- **RWKV7 World3 1.5B/2.9B config**: `bf16, lr 4e-4 to 1e-5 "delayed" cosine decay, wd 0.1` with increasing batch sizes during middle of training. [Source](https://huggingface.co/RWKV/RWKV7-Goose-World3-1.5B-HF)
- **Batch size schedule**: Starts small, increases in the middle (around 15G tokens), then may decrease in late stages. LR and bsz changes are coordinated for smooth transitions ("smooth training — no loss spikes"). [Source](https://github.com/BlinkDL/RWKV-LM)
- **Context length**: 4096 for pretraining, but models trained at ctx4k extrapolate to ctx32k+ automatically. [Source](https://huggingface.co/BlinkDL)
- **Per-parameter configuration**: RWKV7 uses carefully tuned per-tensor initialization, learning rates, and weight decay — this is a key design choice for stability and scalability. Each weight tensor has its own hyperparameters defined at the model level. [Source](https://deepwiki.com/BlinkDL/RWKV-LM/4.1-rwkv-v7-training)

### 3. Fine-Tuning Configs (LoRA, State-Tuning, SFT)
| Method | LR Init | LR Final | Optimizer | DeepSpeed | LoRA r/alpha | Notes |
|--------|---------|----------|-----------|-----------|--------------|-------|
| **LoRA** | 5e-5 (max 1e-4) | same | AdamW (β=0.9, 0.999) | Stage 1 | r=32, α=64 | Recommended starting config |
| **State-Tuning** | **1.0** | 0.01 | Adam (LR=0.001) | Stage 1 | N/A | Extremely high LR, 10 warmup steps |
| **SFT (full)** | 1e-5 → 1e-6 | cosine | AdamW | Stage 2 | N/A | From CommerAI RWKV-7-Goose-Arith |

- **LoRA defaults**: `--lr_init 5e-5`, `--lora_r 32`, `--lora_alpha 64`, `--lora_dropout 0.01`, `--strategy deepspeed_stage_1`. [Source](https://wiki.rwkv.com/RWKV-Fine-Tuning/LoRA-Fine-Tuning.html)
- **State-Tuning** (RWKV-unique): Tunes the initial hidden state. Zero inference overhead. Uses `--train_type "states" --lr_init 1 --lr_final 0.01 --warmup_steps 10`. Adam optimizer at lr=0.001 for 5 epochs. [Source](https://arxiv.org/abs/2504.05097) and [GitHub](https://github.com/BlinkDL/RWKV-LM)
- **Memory benchmarks (RTX 4090, 24GB)**: 0.4B model LoRA fine-tuning at ctx1024, micro_bsz=1, deepspeed_stage_1 uses ~8–12GB VRAM. 1.5B model needs stage 2 or lower ctx. [Source](https://github.com/Joluck/RWKV-PEFT)
- **Dropout note**: "RWKV-LM dropout is very effective — use 1/4 of your usual value." [Source](https://github.com/BlinkDL/RWKV-LM)

### 4. GRPO / RLHF Training Recipes (Community)
- **rwkvtune** (PyPI, v0.1.0): Most complete GRPO tool for RWKV. Provides `GRPOTrainer` class inspired by HuggingFace TRL. Supports multi-GPU via DeepSpeed ZeRO, gradient checkpointing. Still very early (Feb 2026 release, ~11 downloads/month). [Source](https://pypi.org/project/rwkvtune/)
- **OpenMOSE/RWKV-LM-RLHF**: Reinforcement learning toolkit supporting DPO, ORPO, SFT, and distillation for RWKV v6, v7, and ARWKV. Focused on preference optimization (DPO/ORPO), not GRPO. Has working examples for distillation and SFT. [Source](https://github.com/OpenMOSE/RWKV-LM-RLHF)
- **State-Tuning as RLHF alternative**: Because RWKV is a pure RNN, fine-tuning the initial state acts as a form of alignment — the tuned state transfers across tasks. This is a uniquely RWKV approach to preference alignment without RL. [Source](https://arxiv.org/abs/2504.05097)
- **No production GRPO recipe exists**: The RWKV community has not yet published a battle-tested GRPO training recipe. The rwkvtune package is the closest but is pre-production.
- **DPO pipeline (OpenMOSE)**: A more mature path — SFT → DPO using the OpenMOSE toolkit, which is documented and has example configs for 2.9B models. [Source](https://huggingface.co/OpenMOSE/RWKV-x070-2B9-CJE-Instruct)

### 5. Model Parallelism Lessons for Recurrent-Depth Architectures
- **RWKV trains like a Transformer**: The WKV operation is formulated as linear attention, making it fully parallelizable across sequence positions during training. This means standard Transformer parallelism strategies apply. [Source](https://arxiv.org/abs/2305.13048)
- **No tensor/pipeline parallelism needed at moderate scale**: 7.2B model trained with only ZeRO Stage 2 across 32 H100s. The architecture's efficiency means less memory pressure than equivalently-sized Transformers.
- **Recurrent state is constant-size**: Unlike Transformers' growing KV cache, RWKV maintains O(1) state memory regardless of context length. This eliminates a major scaling bottleneck for both training and inference. [Source](https://arxiv.org/abs/2503.14456)
- **Per-parameter config is critical**: RWKV7's stability comes from per-tensor initialization and hyperparameters — a lesson for any recurrent architecture: careful per-component tuning matters more than in Transformers. [Source](https://deepwiki.com/BlinkDL/RWKV-LM/4.1-rwkv-v7-training)
- **Beta2=0.99 for stability**: RWKV uses lower Adam beta2 (0.99 vs standard 0.999) — this is likely important for recurrent architectures where gradient statistics differ from Transformers. The training README shows this consistently across configurations. [Source](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/train_temp/demo-training-prepare.sh)
- **Batch ramp scheduling**: Increasing batch size mid-training (around 15G tokens) while holding LR constant, then decaying both — this "delayed cosine" pattern is specifically tuned for RWKV loss curves. [Source](https://github.com/BlinkDL/RWKV-LM)
- **RWKV7 kernels scale with Bsz×HeadCount**: Throughput improves with larger models and batch sizes due to CUDA kernel design. [Source](https://github.com/BlinkDL/RWKV-LM)

### 6. RWKV8 Status
- As of June 2026, RWKV8 has not been publicly released. The RWKV-LM repository is at v7 "Goose" (x070). The "v8" designation appears in some community discussions but no official training configs, papers, or code exist yet. The architecture survey paper from Jan 2025 covers through RWKV-6. [Source](https://arxiv.org/html/2412.14847v2)

---

## Sources

### Kept
- **BlinkDL/RWKV-LM GitHub README** (https://github.com/BlinkDL/RWKV-LM) — Primary source for training commands, multi-GPU configs, RWKV7 7.2B perf numbers, stability notes
- **RWKV7 Goose Paper (arXiv 2503.14456)** (https://arxiv.org/abs/2503.14456) — Architecture, training schedule, batch size table (Appendix Table 13)
- **RWKV7 World3 Model Cards (HuggingFace)** (https://huggingface.co/RWKV/RWKV7-Goose-World3-1.5B-HF) — Hyperparameters: bf16, lr 4e-4→1e-5 delayed cosine, wd 0.1, increasing batch sizes
- **RWKV wiki LoRA Fine-Tuning Tutorial** (https://wiki.rwkv.com/RWKV-Fine-Tuning/LoRA-Fine-Tuning.html) — Specific LR, LoRA config, DeepSpeed strategy recommendations
- **RWKV wiki Pretrain Tutorial** (https://wiki.rwkv.com/advance/pretrain.html) — Full pretraining pipeline, script parameters
- **RWKV-PEFT GitHub (JL-er/RWKV-PEFT)** (https://github.com/JL-er/RWKV-PEFT) — Memory benchmarks for fine-tuning, state-tuning demo scripts
- **State Tuning Paper (arXiv 2504.05097)** (https://arxiv.org/abs/2504.05097) — State-tuning methodology, Adam LR=0.001, 5 epochs
- **OpenMOSE/RWKV-LM-RLHF** (https://github.com/OpenMOSE/RWKV-LM-RLHF) — DPO/ORPO/SFT recipes for RWKV
- **rwkvtune PyPI** (https://pypi.org/project/rwkvtune/) — GRPOTrainer for RWKV
- **DeepWiki: RWKV-LM Training Pipeline** (https://deepwiki.com/BlinkDL/RWKV-LM/4-training-pipeline) — PyTorch Lightning + DeepSpeed integration details
- **DeepWiki: RWKV-v7 Training** (https://deepwiki.com/BlinkDL/RWKV-LM/4.1-rwkv-v7-training) — Per-parameter configuration system
- **DeepWiki: Training RWKV-v7** (https://deepwiki.com/BlinkDL/RWKV-LM/7.2-training-rwkv-v7) — Quickstart guide, GPU requirements
- **RWKV-infctx-trainer** (https://github.com/RWKV/RWKV-infctx-trainer) — Arbitrary context length training with Lightning CLI + YAML configs
- **GWKV: Reinventing RNNs (arXiv 2305.13048)** (https://arxiv.org/abs/2305.13048) — Original RWKV paper establishing parallel training + recurrent inference
- **gpt-neox RWKV support** (https://github.com/EleutherAI/gpt-neox) — Pipeline parallelism for RWKV in Megatron/DeepSpeed framework

### Dropped
- General GRPO/PPO/RLHF tutorial pages (Unsloth, Analytics Vidhya, Medium) — Not RWKV-specific
- Spheron blog posts on xLSTM/RWKV deployment — Marketing content, no training config details
- RWKV Runner/colab notebooks — Inference-focused, not training
- Multiple duplicate HuggingFace model card mirrors — Same info as primary model card

---

## Gaps

1. **No production GRPO recipe for RWKV**: rwkvtune is pre-release, and no published results show RWKV+GRPO convergence characteristics, reward hacking behavior, or optimal KL penalty coefficients for recurrent models.

2. **RWKV8 training configs**: No public information available. If RWKV8 follows the RWKV pattern, configs will likely appear first in the RWKV-LM GitHub under `RWKV-v8/train_temp/`.

3. **Batch size specifics for 7.2B+ models**: The exact batch size ramp schedule (specific token counts and batch sizes at each stage) is only available in the Goose paper's Appendix Table 13, which was not fully retrieved. Worth extracting directly from the PDF.

4. **Convergence comparison: GRPO on recurrent vs Transformer**: No published study comparing RL training dynamics between RWKV's constant-state architecture and Transformer's attention-based architecture. The state-tuning paper suggests recurrent-specific optimization may behave differently.

5. **Multi-node scaling beyond 32 GPUs**: No published data on RWKV training at 64+ GPU scale. The architecture's linear complexity suggests it should scale well with pure data parallelism, but this is unconfirmed.

6. **RWKV dropout specifics**: The README mentions "use 1/4 of your usual value" for dropout but doesn't specify what "usual" refers to. Transformer-standard dropout (0.1) would imply ~0.025 for RWKV, but this needs verification.

---

## Recommendations for Parcae

1. **Adopt AdamW β2=0.99**: RWKV's lower momentum term may benefit recurrent-depth architectures generally.
2. **Use "delayed cosine" LR schedule**: Hold LR constant for initial tokens, then decay — proven effective for recurrent models.
3. **Start with ZeRO Stage 2 + gradient checkpointing**: Proven config for models up to 7.2B on 32 GPUs.
4. **Consider state-based alignment**: RWKV's state-tuning approach (tune initial hidden state with high LR) is a recurrent-specific technique that could transfer to our architecture.
5. **Test rwkvtune for GRPO prototyping**: Even though pre-production, the API matches HuggingFace TRL which our team is familiar with.
