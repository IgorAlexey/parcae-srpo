Research complete. The full brief is at `research/vram-reality.md`. Here's the bottom line:

---

## G=6 × 128 tokens GRPO on Gemma 4 E2B (~4.6B) — **YES, fits in 32 GB**

**Estimated peak VRAM: ~15 GB** with Unsloth + LoRA (bf16 base weights). Even with 1024-token prompts, stays under 20 GB.

### The breakdown that matters:

| What | Why it's fine |
|------|---------------|
| **9.2 GB** base weights (bf16) | Frozen with LoRA, only adapters trained |
| **~0.2 GB** LoRA + optimizer | Rank 64 adapters are tiny |
| **~0.8 GB** KV cache (prompt+G×completions) | 128-token completions are cheap; Unsloth shares prefix KV |
| **~0.1 GB** logits tensor | Unsloth's chunked kernel makes this negligible at 128 tokens |
| **~2.5 GB** activations (gradient checkpointed) | Unsloth's smart recompute patterns |
| **~2 GB** framework overhead | vLLM + CUDA context |

### What would break it:
- **Full fine-tuning** — weights + gradients + Adam = 55 GB → instant OOM
- **Vanilla TRL without Unsloth** — still fits (~18 GB) but with less headroom
- **Long completions** (e.g., 2048 tokens) — KV cache would grow to ~5 GB

### The one caveat:
Official sources say E2B is **2.1B effective parameters**, not 4.6B. If that's the true weight count, VRAM drops to ~10-12 GB and it's very comfortable. Verify with `model.num_parameters()` before committing to a hardware budget.