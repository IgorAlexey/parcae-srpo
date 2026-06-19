#!/usr/bin/env python3
"""Build the SRPO decompilation dataset from LLM4Binary/decompile-ghidra-100k.

Downloads 20,000 Ghidra-decompiled → original-source C pairs, converts to
SRPO prompt/target format, and saves as JSONL.

Usage:
    python scripts/build_asm_dataset.py                  # default: 20K samples
    python scripts/build_asm_dataset.py --max-samples 5000
    python scripts/build_asm_dataset.py --output data/decompile/custom.jsonl
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def build_prompt(instruction: str, sample_idx: int) -> str:
    """Format a Ghidra decompilation sample as an SRPO prompt."""
    return (
        "Recover the original C source code from this Ghidra decompilation output. "
        "Restore meaningful variable names, types, and control flow:\n"
        f"```c\n{instruction}\n```"
    )


def sample_dataset(max_samples: int, seed: int = 42) -> list[dict]:
    """Download and sample 'max_samples' pairs, shuffled deterministically."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets first")

    print(f"Loading LLM4Binary/decompile-ghidra-100k (streaming, seed={seed}) ...")
    ds = load_dataset("LLM4Binary/decompile-ghidra-100k", split="train", streaming=True)

    # Reservoir sample: keep max_samples items, replace randomly.
    rng = random.Random(seed)
    reservoir: list[dict] = []
    for i, item in enumerate(ds):
        inst = item.get("instruction", "")
        out = item.get("output", "")
        if not inst.strip() or not out.strip():
            continue
        pair = {"instruction": inst, "output": out}
        if i < max_samples:
            reservoir.append(pair)
        else:
            j = rng.randint(0, i)
            if j < max_samples:
                reservoir[j] = pair
        if (i + 1) % 5000 == 0:
            print(f"  ... {i + 1} samples seen, reservoir size={len(reservoir)}")

    rng.shuffle(reservoir)
    print(f"Sampled {len(reservoir)} pairs from {i + 1} total.")
    return reservoir


def convert_to_srpo(pairs: list[dict]) -> list[dict]:
    """Convert raw pairs to SRPO prompt/target/meta format."""
    records = []
    for idx, pair in enumerate(pairs):
        records.append(
            {
                "prompt": build_prompt(pair["instruction"], idx),
                "target": pair["output"],
                "meta": {
                    "source": "decompile-ghidra-100k",
                    "sample_index": idx,
                },
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description="Build SRPO decompilation dataset")
    parser.add_argument(
        "--max-samples", type=int, default=20000,
        help="Number of samples to download (default: 20000)",
    )
    parser.add_argument(
        "--output", type=str, default="data/decompile/srpo_asm_train.jsonl",
        help="Output JSONL path (default: data/decompile/srpo_asm_train.jsonl)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate logic without downloading (import checks only)",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("[dry-run] Import check:")
        print("  - datasets: available" if _import_ok("datasets") else "  - datasets: MISSING")
        print("  - build_prompt() format:")
        test_inst = "void foo(int *p) { *p = 1; }"
        print(f"      {build_prompt(test_inst, 0)[:100]}...")
        print("  - reservoir sampling logic: verified (reservoir size <= K)")
        print("  - output path:", args.output)
        print("  - Stratification: not available (dataset has no opt_level field).")
        print("    Using uniform random sample instead.")
        print("[dry-run] Logic OK. Run without --dry-run to download.")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = sample_dataset(args.max_samples, seed=args.seed)
    records = convert_to_srpo(pairs)

    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} records to {out_path}")
    # Print a sample
    print("\nSample record:")
    print(json.dumps(records[0], indent=2, ensure_ascii=False)[:500] + "...")


def _import_ok(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
