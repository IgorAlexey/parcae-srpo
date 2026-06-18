"""
Script to load Gemma 4 12B and convert to recurrent-depth architecture.
Run with: python -m parcae.convert

This does NOT train; it just validates the architecture loads,
dimensions match, and a forward pass completes.
"""

import sys
import torch

sys.path.insert(0, ".")

from parcae import RecurrentDepthGemma, RecurrentDepthConfig, LTIInjection


def main():
    print("=== Gemma 4 12B → Recurrent-Depth Conversion ===\n")

    # Config: 16 prelude + 16 recurrent + 16 coda = 48 layers
    cfg = RecurrentDepthConfig(
        model_path="models/gemma-4-12B-it",
        prelude_layers=16,
        n_recurrent_layers=16,
        coda_layers=16,
        default_loops=3,
        use_depth_lora=True,
        lora_rank=16,
        use_loop_embedding=True,
        loop_embedding_dim=256,
    )

    print("Step 1: Load pretrained weights...")
    model = RecurrentDepthGemma(cfg)
    model.load_pretrained()

    # Parameter counts
    counts = model.count_parameters()
    print(f"\nParameter breakdown:")
    print(f"  Total:      {counts['total']/1e9:.2f}B")
    print(f"  Pretrained: {counts['pretrained']/1e9:.2f}B")
    print(f"  Trainable:  {counts['trainable']/1e6:.2f}M")
    print(f"  Trainable:  {counts['trainable']:,} params")

    # Verify injection stability
    rho = model.injection.compute_spectral_radius()
    print(f"\n  Injection ρ(A) = {rho:.6f} (must be < 1)")
    assert rho < 1.0, f"Spectral radius {rho} >= 1; unstable!"

    # Test forward pass (CPU, small input)
    print("\nStep 2: Test forward pass (CPU, n_loops=1)...")
    model.eval()
    dummy_ids = torch.randint(0, 1000, (1, 8))  # batch=1, seq_len=8

    with torch.no_grad():
        logits = model.forward(dummy_ids, n_loops=1, return_logits=True)

    print(f"  Input shape:    {dummy_ids.shape}")
    print(f"  Output shape:   {logits.shape}")
    print(f"  Output dtype:   {logits.dtype}")
    print(f"  Logit range:    [{logits.min().item():.2f}, {logits.max().item():.2f}]")
    print(f"  No NaN:         {not torch.isnan(logits).any().item()}")
    V = model.lm_head.out_features
    assert logits.shape == (1, 8, V), f"Bad shape: {logits.shape}, expected (1, 8, {V})"

    # Test with n_loops=3
    print("\nStep 3: Test forward pass (CPU, n_loops=3)...")
    with torch.no_grad():
        logits3 = model.forward(dummy_ids, n_loops=3, return_logits=True)

    print(f"  Output shape:   {logits3.shape}")
    print(f"  No NaN:         {not torch.isnan(logits3).any().item()}")
    print(f"  Diff from T=1:  {(logits3 - logits).abs().mean().item():.6f}")

    # Verify T=1 gives different output than T=3 (the injection should change things)
    diff = (logits3 - logits).abs().mean().item()
    if diff < 1e-6:
        print("  WARNING: n_loops has negligible effect; injection may be too small")
    else:
        print("  OK: injection parameters change the output")

    print("\n=== Architecture validated ===")
    print(f"Prelude layers:  {model.prelude_indices}")
    print(f"Recurrent layers: {model.recurrent_indices}")
    print(f"Coda layers:     {model.coda_indices}")
    print(f"\nAt n_loops=3, effective depth: {16 + 16*3 + 16} layers")


if __name__ == "__main__":
    main()
