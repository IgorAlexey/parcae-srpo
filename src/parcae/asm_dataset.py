"""Assembly-code datasets for SRPO training.

Two sources:
  1. decompile-ghidra-100k  — 100K asm→C pairs (MIT, via HuggingFace)
  2. nl-spec-triples        — 209 NL-spec → C → asm triples (built locally)

Format: uniform {"prompt": str, "target": str} JSONL.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


class AssemblyDataset(Dataset):
    """Loads and wraps JSONL files of {"prompt": str, "target": str}."""

    def __init__(self, paths: list[str], seed: int = 42, max_samples: int = 0):
        self.items: list[dict] = []
        for p in paths:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.items.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        rng = random.Random(seed)
        rng.shuffle(self.items)
        if max_samples > 0 and len(self.items) > max_samples:
            self.items = self.items[:max_samples]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]

    def shuffle(self, seed: Optional[int] = None):
        rng = random.Random(seed or 42)
        rng.shuffle(self.items)


def build_assembly_paths(data_dir: str = "data") -> list[str]:
    """Return all available JSONL data file paths under data_dir."""
    paths = []
    for pattern in ["decompile/srpo_asm_train.jsonl", "nl_asm/srpo_nl_train.jsonl"]:
        p = Path(data_dir) / pattern
        if p.exists():
            paths.append(str(p))
    return paths
