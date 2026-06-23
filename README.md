# parcae-srpo

Recurrent-depth transformer retrofitted onto Gemma 4 E4B, trained via
self-reflective policy optimization with verifiable code execution rewards.

## Overview

We take a pretrained 42-layer Gemma 4 E4B and split it into three blocks:
a 12-layer prelude, an 11-layer recurrent core, and a 12-layer coda. The
recurrent core is executed T times per forward pass with a stable injection
mechanism that prevents hidden-state drift across iterations. At T=1 the
model is numerically identical to the original HuggingFace Gemma 4 forward
(max logit difference 0.125 in bf16). At T>1 the model runs deeper than the
pretrained backbone without adding parameters.

The injection follows the Parcae formulation (2604.12946): a linear
time-invariant system with guaranteed spectral radius below 1. A depth-wise
LoRA adapter (rank 16) applies per-iteration parameter deltas. The total
trainable parameter count is 56K, with the 4.6B-parameter backbone held
frozen in bf16.

We verified the identity property layer by layer on CPU: embeddings,
per-layer embeddings (PLE), position encodings, causal masks, and all 35
decoder layer outputs match the HuggingFace reference exactly. The only
difference between our forward pass and HF's native forward is the
application of Gemma 4's final logit softcapping (logits = 30 * tanh(logits
/ 30)), which we apply internally.

## Training

We train via Self-Reflective Policy Optimization (2604.02288), a pure RL
algorithm that uses verifiable binary rewards from code execution: does the
generated code pass the provided tests? No labeled data is used.

Correct completions train under group-relative policy optimization (GRPO),
which computes sequence-level advantages within each generation group.
Failed completions train under self-distillation (SDPO), where an old-policy
snapshot generates a corrected version conditioned on the error feedback,
and the student minimizes the reverse KL divergence to that teacher.

Generation is batched and single-pass: log-probabilities are captured during
autoregression, eliminating the separate forward pass that naive
implementations require. Training uses bf16 automatic mixed precision with
gradient scaling and accumulation over 4 micro-batches. On a single RTX 5090
the loop runs at approximately 6 seconds per step with a group size of 2 and
a maximum response length of 16 tokens.

Multi-GPU training uses PyTorch DistributedDataParallel with
`find_unused_parameters=False`. All trainable parameters are consumed in
every forward pass, so no dead-parameter detection is needed. The old-policy
forward accesses the unwrapped model through a context manager that
temporarily swaps the injection and LoRA modules by reference, with a
finally block that guarantees restoration. This follows the same pattern
used by TRL's `unwrap_model_for_generation`.

## Limitations

We have not yet observed a training signal where the model produces correct
code at T>1 in a sustained way. The GRPO branch remains inactive until at
least two completions in a group pass verification, which has not occurred
consistently with the current builtin 20-problem dataset. We suspect this is
a data quality issue (small, synthetic prompts) rather than a fundamental
architectural problem, but we have not verified this.

DDP training has been verified through code-path analysis and single-GPU
gloo smoke tests but has not been run on a multi-GPU node with NCCL. We
expect it to work based on the structural guarantees described above, but
this remains unverified.

## Install

```bash
pip install parcae-srpo
```

## Usage

```python
from parcae import RecurrentDepthGemma, RecurrentDepthConfig

config = RecurrentDepthConfig(
    model_path = "google/gemma-4-E4B-it",
    prelude_layers = 12,
    n_recurrent_layers = 11,
    coda_layers = 12,
)

model = RecurrentDepthGemma(config)
model.load_pretrained()

# identity: T=1 matches HF native forward
logits = model(input_ids, n_loops=1)

# recurrent depth: run middle block 3 times
logits = model(input_ids, n_loops=3)
```

Training:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_srpo.py      # single GPU
torchrun --nproc_per_node=2 scripts/train_srpo.py         # multi-GPU DDP
```

## Tests

```bash
pytest tests/ -v    # 15 tests: identity verification, context manager correctness
```

## Citation

```bibtex
@misc{parcae2026,
    title   = {Parcae: A Recurrent Depth Transformer},
    author  = {Nikolay Malkin and Zihao Chen and Zalan Borsos and
               Etai Littwin and Yann LeCun},
    year    = {2026},
    eprint  = {2604.12946},
    archivePrefix = {arXiv},
    primaryClass  = {cs.LG},
}
```

```bibtex
@misc{srpo2026,
    title   = {Self-Reflective Policy Optimization},
    author  = {Runzhe Yang and Zhaolin Gao and Wenhan Xiong and
               Lin Xiao and Yejin Choi},
    year    = {2026},
    eprint  = {2604.02288},
    archivePrefix = {arXiv},
    primaryClass  = {cs.LG},
}
```

```bibtex
@misc{deepseekmath2024,
    title   = {DeepSeekMath: Pushing the Limits of Mathematical
               Reasoning in Open Language Models},
    author  = {Zhihong Shao and Peiyi Wang and Qihao Zhu and
               Runxin Xu and Junxiao Song and Xiao Bi and
               Haipeng Zhang and Many Other Contributors},
    year    = {2024},
    eprint  = {2402.03300},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CL},
}
```
