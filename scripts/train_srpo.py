"""
SRPO training for recurrent-depth Gemma E2B.

Self-Reflective Policy Optimization (ICML 2026, arxiv 2604.02288):
  Correct samples → GRPO branch (sequence-level group-relative advantage)
  Failed samples  → SDPO branch (self-distillation from feedback-conditioned teacher)
  Entropy-aware dynamic weighting suppresses unreliable teacher predictions.

Parcae training recipe (arxiv 2604.12946):
  Variable depth sampling T ~ Poisson(μ_rec)
  Truncated BPTT through μ_bwd = ceil(T * bptt_ratio)
  LTI-stable injection ρ(A) < 1 guaranteed by construction.

Target hardware: 2× RTX 5090 (32GB each), CUDA 13.0.

Run: torchrun --nproc_per_node=2 train_srpo.py
"""

import os
import sys
import math
import time
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Generator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from parcae import RecurrentDepthGemma, RecurrentDepthConfig

# ── Configuration ──────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # ── Model ──
    model_name: str = "google/gemma-4-E2B-it"
    model_path: Optional[str] = None       # set to local cache path to skip download
    prelude_layers: int = 12               # E2B: 35 layers → 12 + 11 + 12 split
    n_recurrent_layers: int = 11
    coda_layers: int = 12
    lora_rank: int = 16
    loop_embedding_dim: int = 128

    # ── Recurrent depth (Parcae) ──
    poisson_mean: int = 2                  # μ_rec (lower for memory)
    min_loops: int = 1
    max_loops: int = 8
    bptt_ratio: float = 0.5                # μ_bwd = ceil(T * bptt_ratio)

    # ── SRPO algorithm ──
    group_size: int = 6                    # G completions per prompt (≥4 for meaningful GRPO advantage; Unsloth/TRL use 6-8)
    max_response_tokens: int = 128          # enough for code/assembly (DeepSeekMath: 4K; we trade length for memory)
    gen_temperature: float = 1.0            # 1.0 standard for RL exploration (TRL/DeepSeekMath/veRL all default 1.0)
    clip_epsilon: float = 0.2
    clip_epsilon_high: float = 0.28        # GSPO Clip-Higher
    kl_beta: float = 0.0
    entropy_weight: float = 0.01           # SRPO entropy-aware weighting

    # ── Optimization ──
    micro_batch_size: int = 2              # prompts per micro-batch (per GPU)
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # ── Training schedule ──
    total_steps: int = 1000
    save_every: int = 200
    eval_every: int = 50
    log_every: int = 10
    seed: int = 42

    # ── Dataset ──
    dataset: str = "asm"                  # asm | mbpp | both | humaneval | bigcodebench | builtin
    asm_data_dir: str = "data"             # assembly JSONL directory
    max_prompts: int = 500

    # ── Distributed ──
    world_size: int = 2                    # set by torchrun


# ── Dataset ────────────────────────────────────────────────────────────

class CodeProblemDataset:
    """Streaming dataset of coding problems with verifiable unit tests."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self._items = self._load()

    def _load(self) -> list[dict]:
        name = self.cfg.dataset

        # asm handled separately — must not fall through to try/except
        if name in ("asm", "both"):
            from parcae.asm_dataset import AssemblyDataset, build_assembly_paths
            paths = build_assembly_paths(self.cfg.asm_data_dir)
            if not paths:
                raise FileNotFoundError(
                    f"No JSONL under {self.cfg.asm_data_dir}. "
                    "Run scripts/build_asm_dataset.py first.")
            if name == "asm":
                ds = AssemblyDataset(paths, seed=self.cfg.seed, max_samples=self.cfg.max_prompts)
                return [{"prompt": i["prompt"], "target": i["target"], "kind": "asm"} for i in ds]
            # both: load code problems + assembly, round-robin interleave
            code_items = self._load_code_problems()
            asm_ds = AssemblyDataset(paths, seed=self.cfg.seed, max_samples=self.cfg.max_prompts)
            asm_items = [{"prompt": i["prompt"], "target": i["target"], "kind": "asm"} for i in asm_ds]
            half = self.cfg.max_prompts // 2
            code_items = code_items[:half]
            asm_items = asm_items[:half]
            rng = random.Random(self.cfg.seed)
            rng.shuffle(code_items)
            rng.shuffle(asm_items)
            items = []
            for k in range(max(len(code_items), len(asm_items))):
                if k < len(code_items):
                    items.append(code_items[k])
                if k < len(asm_items):
                    items.append(asm_items[k])
            return items[: self.cfg.max_prompts]

        try:
            items = self._load_code_problems(name)
            rng = random.Random(self.cfg.seed)
            rng.shuffle(items)
            return items[: self.cfg.max_prompts]
        except Exception:
            items = _builtin_problems(self.cfg.max_prompts)
            for i in items:
                i["kind"] = "code"
            return items

    def _load_code_problems(self, name: str = "mbpp") -> list[dict]:
        from datasets import load_dataset

        if name == "mbpp":
            ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
            items = []
            for item in ds:
                tests = item["test_list"]
                test_str = "\n".join(tests)
                entry = tests[0].split("assert ")[1].split("(")[0] if tests else "solution"
                func_sig = item["code"].split("\n")[0]
                full_prompt = f"{item['prompt']}\n{func_sig}"
                items.append({
                    "prompt": full_prompt,
                    "test": test_str,
                    "entry": entry,
                    "kind": "code",
                })
        elif name == "humaneval":
            ds = load_dataset("openai/openai_humaneval", split="test")
            items = []
            for item in ds:
                items.append({
                    "prompt": item["prompt"],
                    "test": "\n".join([f"assert {t}" for t in item.get("test", "").split("\n") if t.strip()]) if item.get("test") else "",
                    "entry": item.get("entry_point", "solution"),
                    "kind": "code",
                })
        elif name == "bigcodebench":
            ds = load_dataset("bigcode/bigcodebench", split="v0.1")
            items = [{"prompt": i["prompt"], "test": i.get("test",""), "entry": i.get("entry_point","solution"), "kind": "code"} for i in ds]
        else:
            raise ValueError(f"Unknown dataset: {name}")
        return items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int) -> dict:
        return self._items[i]

    def shuffle(self):
        rng = random.Random(self.cfg.seed + int(time.time() * 1e6) % 10000)
        rng.shuffle(self._items)


def _builtin_problems(n: int) -> list[dict]:
    """Fallback: 50 diverse coding problems with multi-assert tests."""
    problems = [
        # ── Math / numbers ──
        {"prompt": "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n", "test": "assert add(2,3)==5; assert add(-1,1)==0; assert add(0,0)==0\n", "entry": "add", "kind": "code"},
        {"prompt": "def factorial(n):\n    \"\"\"Return n! recursively.\"\"\"\n", "test": "assert factorial(0)==1; assert factorial(1)==1; assert factorial(5)==120; assert factorial(7)==5040\n", "entry": "factorial"},
        {"prompt": "def is_prime(n):\n    \"\"\"Return True if n is prime.\"\"\"\n", "test": "assert is_prime(2); assert is_prime(17); assert is_prime(97); assert not is_prime(1); assert not is_prime(4); assert not is_prime(100)\n", "entry": "is_prime"},
        {"prompt": "def gcd(a, b):\n    \"\"\"Return greatest common divisor.\"\"\"\n", "test": "assert gcd(48,18)==6; assert gcd(7,13)==1; assert gcd(100,10)==10; assert gcd(0,5)==5\n", "entry": "gcd"},
        {"prompt": "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number (0-indexed).\"\"\"\n", "test": "assert fibonacci(0)==0; assert fibonacci(1)==1; assert fibonacci(2)==1; assert fibonacci(10)==55; assert fibonacci(20)==6765\n", "entry": "fibonacci"},
        {"prompt": "def power(x, n):\n    \"\"\"Return x raised to power n (integer).\"\"\"\n", "test": "assert power(2,10)==1024; assert power(3,0)==1; assert power(5,3)==125; assert power(2,-1)==0.5\n", "entry": "power"},
        {"prompt": "def is_even(n):\n    \"\"\"Return True if n is even.\"\"\"\n", "test": "assert is_even(2); assert is_even(0); assert is_even(-4); assert not is_even(1); assert not is_even(99)\n", "entry": "is_even"},
        {"prompt": "def sum_of_digits(n):\n    \"\"\"Return sum of decimal digits of positive int n.\"\"\"\n", "test": "assert sum_of_digits(123)==6; assert sum_of_digits(0)==0; assert sum_of_digits(9999)==36\n", "entry": "sum_of_digits"},
        {"prompt": "def is_perfect_square(n):\n    \"\"\"Return True if n is a perfect square.\"\"\"\n", "test": "assert is_perfect_square(0); assert is_perfect_square(1); assert is_perfect_square(16); assert is_perfect_square(100); assert not is_perfect_square(2); assert not is_perfect_square(99)\n", "entry": "is_perfect_square"},
        {"prompt": "def lcm(a, b):\n    \"\"\"Return least common multiple of a and b.\"\"\"\n", "test": "assert lcm(4,6)==12; assert lcm(7,13)==91; assert lcm(1,99)==99; assert lcm(10,10)==10\n", "entry": "lcm"},

        # ── Strings ──
        {"prompt": "def is_palindrome(s):\n    \"\"\"Return True if s reads the same backward.\"\"\"\n", "test": "assert is_palindrome('racecar'); assert is_palindrome(''); assert is_palindrome('a'); assert not is_palindrome('hello'); assert not is_palindrome('ab')\n", "entry": "is_palindrome"},
        {"prompt": "def count_vowels(s):\n    \"\"\"Return number of vowels in s (case-insensitive).\"\"\"\n", "test": "assert count_vowels('hello')==2; assert count_vowels('HELLO')==2; assert count_vowels('xyz')==0; assert count_vowels('aeiou')==5\n", "entry": "count_vowels"},
        {"prompt": "def char_frequency(s):\n    \"\"\"Return dict of character frequencies (case-sensitive).\"\"\"\n", "test": "assert char_frequency('aba')=={'a':2,'b':1}; assert char_frequency('')=={}; assert char_frequency('zzz')=={'z':3}\n", "entry": "char_frequency"},
        {"prompt": "def anagram(s1, s2):\n    \"\"\"Return True if s1 and s2 are anagrams.\"\"\"\n", "test": "assert anagram('listen','silent'); assert anagram('',''); assert not anagram('hello','world'); assert not anagram('a','ab')\n", "entry": "anagram"},
        {"prompt": "def longest_common_prefix(strs):\n    \"\"\"Return longest common prefix of list of strings.\"\"\"\n", "test": "assert longest_common_prefix(['flower','flow','flight'])=='fl'; assert longest_common_prefix(['dog','racecar','car'])==''; assert longest_common_prefix(['a'])=='a'\n", "entry": "longest_common_prefix"},
        {"prompt": "def reverse_string(s):\n    \"\"\"Return reversed copy of string.\"\"\"\n", "test": "assert reverse_string('hello')=='olleh'; assert reverse_string('')==''; assert reverse_string('a')=='a'\n", "entry": "reverse_string"},
        {"prompt": "def capitalize_words(s):\n    \"\"\"Return string with first letter of each word capitalized.\"\"\"\n", "test": "assert capitalize_words('hello world')=='Hello World'; assert capitalize_words('a b c')=='A B C'; assert capitalize_words('')==''\n", "entry": "capitalize_words"},
        {"prompt": "def remove_vowels(s):\n    \"\"\"Return s with all vowels removed (case-insensitive).\"\"\"\n", "test": "assert remove_vowels('hello')=='hll'; assert remove_vowels('AEIOU')==''; assert remove_vowels('xyz')=='xyz'\n", "entry": "remove_vowels"},
        {"prompt": "def word_count(s):\n    \"\"\"Return number of words in s (split by whitespace).\"\"\"\n", "test": "assert word_count('hello world')==2; assert word_count('  a  b  ')==2; assert word_count('')==0\n", "entry": "word_count"},

        # ── Lists / arrays ──
        {"prompt": "def reverse_list(lst):\n    \"\"\"Return reversed copy of list.\"\"\"\n", "test": "assert reverse_list([1,2,3])==[3,2,1]; assert reverse_list([])==[]; assert reverse_list([1])==[1]; assert reverse_list(['a','b'])==['b','a']\n", "entry": "reverse_list"},
        {"prompt": "def binary_search(arr, x):\n    \"\"\"Return index of x in sorted arr, or -1.\"\"\"\n", "test": "assert binary_search([1,3,5,7],5)==2; assert binary_search([1,3,5,7],4)==-1; assert binary_search([],1)==-1; assert binary_search([1],1)==0\n", "entry": "binary_search"},
        {"prompt": "def merge_sorted(a, b):\n    \"\"\"Merge two sorted lists into one sorted list.\"\"\"\n", "test": "assert merge_sorted([1,3],[2,4])==[1,2,3,4]; assert merge_sorted([],[1])==[1]; assert merge_sorted([5,6],[1,2])==[1,2,5,6]\n", "entry": "merge_sorted"},
        {"prompt": "def max_subarray(nums):\n    \"\"\"Kadane's algorithm: max subarray sum.\"\"\"\n", "test": "assert max_subarray([-2,1,-3,4,-1,2,1,-5,4])==6; assert max_subarray([1])==1; assert max_subarray([-1,-2,-3])==-1; assert max_subarray([5,4,-1,7,8])==23\n", "entry": "max_subarray"},
        {"prompt": "def two_sum(nums, target):\n    \"\"\"Return indices of two numbers summing to target.\"\"\"\n", "test": "assert set(two_sum([2,7,11,15],9))=={0,1}; assert set(two_sum([3,2,4],6))=={1,2}; assert set(two_sum([3,3],6))=={0,1}\n", "entry": "two_sum"},
        {"prompt": "def remove_duplicates(lst):\n    \"\"\"Return list with duplicates removed, preserving order.\"\"\"\n", "test": "assert remove_duplicates([1,2,2,3,1])==[1,2,3]; assert remove_duplicates([])==[]; assert remove_duplicates([1,1,1])==[1]\n", "entry": "remove_duplicates"},
        {"prompt": "def rotate_list(lst, k):\n    \"\"\"Rotate list right by k positions.\"\"\"\n", "test": "assert rotate_list([1,2,3,4,5],2)==[4,5,1,2,3]; assert rotate_list([1,2,3],0)==[1,2,3]; assert rotate_list([1,2,3],3)==[1,2,3]; assert rotate_list([1,2,3],1)==[3,1,2]\n", "entry": "rotate_list"},
        {"prompt": "def flatten(lst):\n    \"\"\"Flatten a nested list one level deep.\"\"\"\n", "test": "assert flatten([[1,2],[3,4]])==[1,2,3,4]; assert flatten([])==[]; assert flatten([[1],[],[2,3]])==[1,2,3]\n", "entry": "flatten"},
        {"prompt": "def sort_by_length(words):\n    \"\"\"Sort list of words by length ascending.\"\"\"\n", "test": "assert sort_by_length(['a','abc','ab'])==['a','ab','abc']; assert sort_by_length([])==[]; assert sort_by_length(['xyz','a'])==['a','xyz']\n", "entry": "sort_by_length"},
        {"prompt": "def list_product(nums):\n    \"\"\"Return product of all numbers in list.\"\"\"\n", "test": "assert list_product([1,2,3,4])==24; assert list_product([5])==5; assert list_product([])==1; assert list_product([0,1,2])==0\n", "entry": "list_product"},
        {"prompt": "def running_sum(nums):\n    \"\"\"Return list of running totals (prefix sums).\"\"\"\n", "test": "assert running_sum([1,2,3,4])==[1,3,6,10]; assert running_sum([1])==[1]; assert running_sum([])==[]\n", "entry": "running_sum"},
        {"prompt": "def find_min_max(nums):\n    \"\"\"Return (min, max) tuple from list. Assume non-empty.\"\"\"\n", "test": "assert find_min_max([3,1,4,1,5])==(1,5); assert find_min_max([7])==(7,7); assert find_min_max([-5,0,5])==(-5,5)\n", "entry": "find_min_max"},
        {"prompt": "def count_occurrences(lst, x):\n    \"\"\"Return number of times x appears in lst.\"\"\"\n", "test": "assert count_occurrences([1,2,2,3,2],2)==3; assert count_occurrences([],1)==0; assert count_occurrences([1,2,3],4)==0\n", "entry": "count_occurrences"},
        {"prompt": "def interleave(a, b):\n    \"\"\"Interleave two lists: [a0,b0,a1,b1,...]. Assume equal length.\"\"\"\n", "test": "assert interleave([1,2,3],[4,5,6])==[1,4,2,5,3,6]; assert interleave([],[])==[]; assert interleave(['a'],['b'])==['a','b']\n", "entry": "interleave"},
        {"prompt": "def chunk_list(lst, size):\n    \"\"\"Split lst into chunks of given size.\"\"\"\n", "test": "assert chunk_list([1,2,3,4,5],2)==[[1,2],[3,4],[5]]; assert chunk_list([1],3)==[[1]]; assert chunk_list([],1)==[]\n", "entry": "chunk_list"},

        # ── Dictionaries / sets ──
        {"prompt": "def merge_dicts(a, b):\n    \"\"\"Merge two dicts; b overwrites a on conflict.\"\"\"\n", "test": "assert merge_dicts({'a':1},{'b':2})=={'a':1,'b':2}; assert merge_dicts({'a':1},{'a':2})=={'a':2}; assert merge_dicts({},{})=={}\n", "entry": "merge_dicts"},
        {"prompt": "def invert_dict(d):\n    \"\"\"Invert dict: keys become values and vice versa. Assume unique values.\"\"\"\n", "test": "assert invert_dict({'a':1,'b':2})=={1:'a',2:'b'}; assert invert_dict({})=={}; assert invert_dict({'x':10})=={10:'x'}\n", "entry": "invert_dict"},
        {"prompt": "def set_union(a, b):\n    \"\"\"Return sorted list of elements in either set a or b.\"\"\"\n", "test": "assert set_union({1,2},{2,3})==[1,2,3]; assert set_union(set(),{1})==[1]; assert set_union({},set())==[]\n", "entry": "set_union"},
        {"prompt": "def set_intersection(a, b):\n    \"\"\"Return sorted list of elements in both sets.\"\"\"\n", "test": "assert set_intersection({1,2,3},{2,3,4})==[2,3]; assert set_intersection({1},{2})==[]; assert set_intersection(set(),{1})==[]\n", "entry": "set_intersection"},
        {"prompt": "def set_difference(a, b):\n    \"\"\"Return sorted list of elements in a but not in b.\"\"\"\n", "test": "assert set_difference({1,2,3},{2})==[1,3]; assert set_difference({1},{1})==[]; assert set_difference(set(),{1})==[]\n", "entry": "set_difference"},

        # ── Classes / OOP ──
        {"prompt": "class Counter:\n    \"\"\"Count from 0, incrementing by 1 each call.\"\"\"\n    def __init__(self):\n        self.n = 0\n    def next(self):\n", "test": "c=Counter(); assert c.next()==0; assert c.next()==1; assert c.next()==2\n", "entry": "Counter"},
        {"prompt": "class Stack:\n    \"\"\"Simple stack with push, pop, peek, is_empty.\"\"\"\n    def __init__(self):\n        self._items = []\n    def push(self, x):\n", "test": "s=Stack(); s.push(1); s.push(2); assert s.peek()==2; assert s.pop()==2; assert s.pop()==1; assert s.is_empty()\n", "entry": "Stack"},
        {"prompt": "class Queue:\n    \"\"\"Simple queue with enqueue, dequeue, peek, is_empty.\"\"\"\n    def __init__(self):\n        self._items = []\n    def enqueue(self, x):\n", "test": "q=Queue(); q.enqueue(1); q.enqueue(2); assert q.peek()==1; assert q.dequeue()==1; assert q.dequeue()==2; assert q.is_empty()\n", "entry": "Queue"},

        # ── Recursion ──
        {"prompt": "def sum_nested(lst):\n    \"\"\"Return sum of all integers in a nested list (any depth).\"\"\"\n", "test": "assert sum_nested([1,[2,[3,4]],5])==15; assert sum_nested([])==0; assert sum_nested([1,2,3])==6\n", "entry": "sum_nested"},
        {"prompt": "def tree_depth(obj):\n    \"\"\"Return max nesting depth of a list. A plain value has depth 0, [] has depth 1.\"\"\"\n", "test": "assert tree_depth([[]])==2; assert tree_depth([1,[2,[3]]])==3; assert tree_depth(5)==0; assert tree_depth([])==1\n", "entry": "tree_depth"},

        # ── Search / sort ──
        {"prompt": "def linear_search(arr, x):\n    \"\"\"Return first index of x in arr, or -1.\"\"\"\n", "test": "assert linear_search([5,3,1,4],3)==1; assert linear_search([5,3,1,4],6)==-1; assert linear_search([],1)==-1\n", "entry": "linear_search"},
        {"prompt": "def bubble_sort(arr):\n    \"\"\"Sort list in place using bubble sort. Return the sorted list.\"\"\"\n", "test": "assert bubble_sort([3,1,2])==[1,2,3]; assert bubble_sort([])==[]; assert bubble_sort([5,5,1])==[1,5,5]\n", "entry": "bubble_sort"},

        # ── String manipulation (harder) ──
        {"prompt": "def compress_string(s):\n    \"\"\"Basic run-length encoding: 'aaabb' -> 'a3b2'.\"\"\"\n", "test": "assert compress_string('aaabb')=='a3b2'; assert compress_string('')==''; assert compress_string('abc')=='a1b1c1'\n", "entry": "compress_string"},
        {"prompt": "def longest_word(sentence):\n    \"\"\"Return the longest word in a sentence (by length). On tie, return first.\"\"\"\n", "test": "assert longest_word('the quick brown fox')=='quick'; assert longest_word('a')=='a'; assert longest_word('')==''\n", "entry": "longest_word"},
        {"prompt": "def is_substring(s, sub):\n    \"\"\"Return True if sub appears in s (case-sensitive). Do NOT use 'in'.\"\"\"\n", "test": "assert is_substring('hello','ell'); assert is_substring('abc','abc'); assert not is_substring('abc','abcd'); assert not is_substring('','a')\n", "entry": "is_substring"},
        {"prompt": "def title_case(s):\n    \"\"\"Convert to title case: first letter upper, rest lower, per word.\"\"\"\n", "test": "assert title_case('hello world')=='Hello World'; assert title_case('HELLO')=='Hello'; assert title_case('a b c')=='A B C'\n", "entry": "title_case"},
    ]
    # Ensure every item has kind="code"
    for p in problems:
        p.setdefault("kind", "code")
    rng = random.Random(42)
    items = []
    while len(items) < n:
        items.append(rng.choice(problems))
    return items


# ── Reward ─────────────────────────────────────────────────────────────

def extract_code(text: str) -> str:
    """Extract Python function from model completion. Robust to missing fences."""
    # Try markdown fences first
    if "```python" in text:
        parts = text.split("```python", 1)
        if len(parts) > 1:
            code = parts[1].split("```", 1)[0]
            if code.strip():
                return code.strip()
    if "```" in text:
        parts = text.split("```", 1)
        if len(parts) > 1:
            code = parts[1].split("```", 1)[0]
            if code.strip():
                return code.strip()
    # Fallback: return raw text (model may output code directly)
    return text.strip()


def extract_code_c(text: str) -> str:
    """Extract C code from model completion. Handles ```c, ```C, ```cpp."""
    for fence in ("```cpp", "```c", "```C"):
        if fence in text:
            parts = text.split(fence, 1)
            if len(parts) > 1:
                code = parts[1].split("```", 1)[0]
                if code.strip():
                    return code.strip()
    # Try any code fence
    if "```" in text:
        parts = text.split("```", 1)
        if len(parts) > 1:
            code = parts[1].split("```", 1)[0]
            if code.strip():
                return code.strip()
    return text.strip()


def verify_compile(code: str, timeout: float = 15.0) -> tuple[float, str]:
    """Compile C code with GCC. Returns (1.0, feedback) on success.

    Runs on rented GPU instances — no host compromise risk.
    """
    import tempfile
    import shutil
    if not shutil.which("gcc"):
        return 0.0, "GCC not found."
    if not code.strip():
        return 0.0, "Empty code."
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".c", delete=False,
    ) as f:
        f.write(code)
        src_path = f.name
    bin_path = src_path.replace(".c", "")
    try:
        r = subprocess.run(
            ["gcc", "-O2", "-Wall", "-o", bin_path, src_path],
            capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode == 0:
            return 1.0, "Compiled successfully."
        return 0.0, r.stderr.strip()[:600]
    except subprocess.TimeoutExpired:
        return 0.0, "Compilation timed out."
    except Exception as e:
        return 0.0, str(e)[:600]
    finally:
        Path(src_path).unlink(missing_ok=True)
        Path(bin_path).unlink(missing_ok=True)


def verify(code: str, test: str, timeout: float = 10.0) -> tuple[float, str]:
    """Run code + tests in subprocess. Returns (reward 0/1, feedback)."""
    script_lines = [code, ""]
    script_lines.append("try:")
    for t in test.strip().split("\n"):
        if t.strip():
            script_lines.append(f"    {t}")
    script_lines.append("    print('__ALL_PASSED__')")
    script_lines.append("except Exception as exc:")
    script_lines.append("    import traceback")
    script_lines.append("    print('__FAILED__')")
    script_lines.append("    traceback.print_exc()")
    full_script = "\n".join(script_lines)
    try:
        r = subprocess.run(["python3", "-c", full_script], capture_output=True, text=True, timeout=timeout)
        out = r.stdout + r.stderr
        if "__ALL_PASSED__" in out:
            return 1.0, "All tests passed."
        return 0.0, out.strip()[:600]
    except subprocess.TimeoutExpired:
        return 0.0, "Timed out."
    except Exception as e:
        return 0.0, str(e)[:600]


# ── SRPO losses ────────────────────────────────────────────────────────

def grpo_loss(
    log_probs: torch.Tensor,         # (B, L)
    log_probs_old: torch.Tensor,     # (B, L)
    rewards: torch.Tensor,           # (B,)
    response_mask: torch.Tensor,     # (B, L)
    epsilon: float,
    epsilon_high: float,
) -> torch.Tensor:
    """
    GRPO branch: sequence-level group-relative advantage.  One scalar
    importance ratio per sequence (GSPO sequence-level formulation),
    clipped PPO-style with asymmetric Clip-Higher.

    B is the group size (all completions for one or more prompts).
    """
    B = rewards.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=rewards.device)

    # sequence-level log‑prob (mean over response tokens)
    n_tokens = response_mask.sum(dim=-1).clamp(min=1)       # (B,)
    seq_lp = (log_probs * response_mask).sum(dim=-1) / n_tokens
    seq_lp_old = (log_probs_old * response_mask).sum(dim=-1) / n_tokens

    # group-relative advantage
    mu = rewards.mean()
    sigma = rewards.std() + 1e-8
    A = (rewards - mu) / sigma                                    # (B,)

    # sequence-level importance ratio (length-normalized)
    rho = torch.exp(seq_lp - seq_lp_old)                          # (B,)

    # clipped surrogate (sequence level)
    rho_clip = torch.clamp(rho, 1.0 - epsilon, 1.0 + epsilon_high)
    surr = torch.min(rho * A, rho_clip * A)                       # (B,)

    # expand to token level: every token in sequence i gets the same
    # scalar surrogate weight
    per_token = surr.unsqueeze(-1) * response_mask                # (B, L)
    loss = -per_token.sum() / n_tokens.sum().clamp(min=1)
    return loss


def sdpo_loss(
    student_logits: torch.Tensor,    # (L, V) or (B, L, V)
    teacher_logits: torch.Tensor,    # (L, V) or (B, L, V)
    response_mask: torch.Tensor,     # (L,) or (B, L)
    entropy_weight: float,
) -> torch.Tensor:
    """
    SDPO branch: reverse KL with entropy-aware weighting.
    Supports batched (B, L, V) or single (L, V) inputs.
    Weight in (0, 1] per token; uncertain teacher predictions
    suppressed; confident ones emphasized.
    """
    if student_logits.dim() == 2:  # single sequence: (L, V)
        student_logits = student_logits.unsqueeze(0)   # (1, L, V)
        teacher_logits = teacher_logits.unsqueeze(0)
        response_mask = response_mask.unsqueeze(0)      # (1, L)
    B, L, V = student_logits.shape

    student_lp = F.log_softmax(student_logits.float(), dim=-1).to(student_logits.dtype)
    teacher_p  = F.softmax(teacher_logits.float(), dim=-1).to(student_logits.dtype)

    # token-level reverse KL: KL(p_student || p_teacher)
    kl = F.kl_div(
        student_lp.float(), teacher_p.float(),
        reduction="none", log_target=False,
    ).sum(dim=-1).to(student_logits.dtype)  # (B, L)

    # entropy-aware weight
    teacher_log_p = F.log_softmax(teacher_logits.float(), dim=-1).to(teacher_logits.dtype)
    H = -(teacher_p.float() * teacher_log_p.float()).sum(dim=-1).to(teacher_logits.dtype)
    H_max = math.log(V)
    w = 1.0 - entropy_weight * (H / H_max)  # (B, L) in (0, 1]

    weighted = kl * w * response_mask  # (B, L)
    return (weighted.float().sum() / response_mask.sum().clamp(min=1).float()).to(weighted.dtype)


# ── Feedback ───────────────────────────────────────────────────────────

def build_feedback(
    failed_code: str,
    error: str,
    problem: str,
    correct_demos: list[str],
    kind: str = "code",
) -> str:
    """Build self-distillation feedback for a failed completion."""
    lang = "c" if kind == "asm" else "python"
    if correct_demos:
        demo = correct_demos[0]
        return (
            f"Your code:\n```{lang}\n{failed_code}\n```\n\n"
            f"Error: {error}\n\n"
            f"Here is a working solution:\n```{lang}\n{demo}\n```\n\n"
            f"Write a corrected version."
        )
    return (
        f"Your code:\n```{lang}\n{failed_code}\n```\n\n"
        f"Error: {error}\n\nIdentify and fix the mistake."
    )


# ── Trainer ────────────────────────────────────────────────────────────

class SRPOTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.device = torch.device(f"cuda:{self.local_rank}")

        if self.world_size > 1:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)

        self._seed()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path or cfg.model_name)
        self._load_chat_template(cfg)
        self._build_model()
        self._build_optimizer()
        self.dataset = CodeProblemDataset(cfg)

        if self.world_size > 1:
            self._sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=True,
                seed=cfg.seed,
            )
            self.dataloader = DataLoader(
                self.dataset, batch_size=cfg.micro_batch_size,
                sampler=self._sampler,
                collate_fn=lambda x: list(x))
        else:
            self._sampler = None
            self.dataloader = DataLoader(
                self.dataset, batch_size=cfg.micro_batch_size,
                shuffle=True, collate_fn=lambda x: list(x))

        self.scaler = torch.amp.GradScaler('cuda')
        self.step = 0

    def _seed(self):
        torch.manual_seed(self.cfg.seed + self.local_rank)
        np.random.seed(self.cfg.seed + self.local_rank)
        random.seed(self.cfg.seed + self.local_rank)

    def _load_chat_template(self, cfg):
        """Ensure the tokenizer has a chat template.

        Gemma 4 E2B ships chat_template.jinja separately (HF issue 45205).
        Other model families embed it in tokenizer_config.json; skip if
        already present.

        Template is small (~17 KB); every DDP rank reads it independently
        at startup.  No broadcast needed.
        """
        if self.tokenizer.chat_template:
            return  # already loaded from tokenizer_config.json

        template = None

        if cfg.model_path:
            local_dir = Path(cfg.model_path)
            if not local_dir.is_dir():
                raise NotADirectoryError(
                    f"model_path={cfg.model_path} is not a directory")
            local_tmpl = local_dir / "chat_template.jinja"
            if local_tmpl.exists():
                template = local_tmpl.read_text()
                if self.local_rank == 0:
                    print(f"Loaded chat template from {local_tmpl}")

        # Try HF hub if no local template found.
        if template is None and cfg.model_name:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import (
                EntryNotFoundError, RepositoryNotFoundError)
            try:
                cache = hf_hub_download(
                    repo_id=cfg.model_name, filename="chat_template.jinja")
                template = Path(cache).read_text()
                if self.local_rank == 0:
                    print(f"Loaded chat template from {cache}")
            except (EntryNotFoundError, RepositoryNotFoundError, OSError):
                pass

        if template is None:
            return  # no separate template file; raw text is fine

        self.tokenizer.chat_template = template

    def _apply_chat_template(self, content: str) -> str:
        """Wrap content in chat template if available, otherwise return raw."""
        if self.tokenizer.chat_template is None:
            return content
        messages = [{"role": "user", "content": content}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    def _tokenize_safe(self, texts: list[str], max_length: int = 512):
        """Tokenize with truncation and one-shot overflow warning."""
        # Chat template already renders all special tokens.
        add_special = self.tokenizer.chat_template is None
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
            add_special_tokens=add_special,
        )
        # Detect truncation: a sample reached max_length and has no trailing pad.
        at_limit = (enc["input_ids"].shape[1] >= max_length)
        full_mask = enc["attention_mask"].sum(dim=1) >= max_length
        truncated = (at_limit and full_mask.any())
        if truncated and not hasattr(self, '_trunc_warned'):
            self._trunc_warned = True
            if self.local_rank == 0:
                n = full_mask.sum().item()
                print(f"[WARN] {n} prompts at {max_length}-token limit "
                      "(may be truncated)")
        return enc.to(self.device)

    def _build_model(self):
        rd = RecurrentDepthConfig(
            model_path=self.cfg.model_path or self.cfg.model_name,
            prelude_layers=self.cfg.prelude_layers,
            n_recurrent_layers=self.cfg.n_recurrent_layers,
            coda_layers=self.cfg.coda_layers,
            default_loops=self.cfg.poisson_mean,
            use_depth_lora=True,
            lora_rank=self.cfg.lora_rank,
            use_loop_embedding=True,
            loop_embedding_dim=self.cfg.loop_embedding_dim,
        )
        if self.cfg.model_path is None:
            from huggingface_hub import snapshot_download
            rd.model_path = snapshot_download(
                self.cfg.model_name, cache_dir="/tmp/hf-cache",
                ignore_patterns=["*.md", ".gitattributes"])
        self.model = RecurrentDepthGemma(rd)
        self.model.load_pretrained()
        self.model.to(self.device)

        # freeze backbone, train injection + lora + loop emb
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.injection.train()
        for p in self.model.injection.parameters():
            p.requires_grad = True
        if self.model.depth_lora:
            self.model.depth_lora.train()
            for p in self.model.depth_lora.parameters():
                p.requires_grad = True

        # Parallel old-policy modules for GRPO importance sampling.
        # We swap MODULE REFERENCES (not tensor data) to avoid DDP inplace-op tracking.
        import copy
        self.injection_old = copy.deepcopy(self.model.injection)
        self.injection_old.to(self.device)
        for p in self.injection_old.parameters():
            p.requires_grad = False
        if self.model.depth_lora:
            self.depth_lora_old = copy.deepcopy(self.model.depth_lora)
            self.depth_lora_old.to(self.device)
            for p in self.depth_lora_old.parameters():
                p.requires_grad = False
        else:
            self.depth_lora_old = None

        # Wrap in DDP for multi-GPU.
        # Module-reference swaps (for old-policy log-probs) don't trigger DDP versioning
        # because they're Python object assignments, not tensor inplace ops.
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=True
            )
            self._model_unwrapped = self.model.module
        else:
            self._model_unwrapped = self.model

        n_train = sum(p.numel() for p in self.trainable_params())
        n_total = sum(p.numel() for p in self._model_unwrapped.parameters())
        if self.local_rank == 0:
            print(f"Model loaded. {n_total/1e9:.2f}B total, {n_train:,} trainable")

    def trainable_params(self):
        """Generator over trainable parameters (works through DDP wrapper)."""
        m = self._model_unwrapped
        for p in m.injection.parameters():
            yield p
        if m.depth_lora:
            for p in m.depth_lora.parameters():
                yield p

    def _build_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.trainable_params(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

    # ── generation with log‑prob caching ───────────────────────────

    @torch.no_grad()
    def _generate(self, prompts: list[str], T: int) -> list[dict]:
        """Generate G completions per prompt, batched across prompts.

        Calls generate() with return_logprobs=True. Single forward pass
        per batch, no separate log-prob extraction pass.
        """
        self._model_unwrapped._bptt_depth = None
        results = []
        G = self.cfg.group_size

        # Tokenize all prompts together (apply chat template first)
        formatted = [self._apply_chat_template(p) for p in prompts]
        enc = self._tokenize_safe(formatted)
        prompt_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]
        B = len(prompts)

        # Generate G completions per prompt
        for g in range(G):
            gen_out = self._model_unwrapped.generate(
                input_ids=prompt_ids,
                max_new_tokens=self.cfg.max_response_tokens,
                n_loops=T,
                temperature=self.cfg.gen_temperature,
                top_k=50,
                return_logprobs=True,
            )
            if isinstance(gen_out, tuple):
                full_ids, token_lps_batch = gen_out
            else:
                full_ids = gen_out
                token_lps_batch = None

            for b in range(B):
                pl = attn_mask[b].sum().item()
                ids = full_ids[b]
                L = ids.shape[0]
                eos_mask = (ids[pl:] == self.tokenizer.eos_token_id)
                if eos_mask.any():
                    first_eos = eos_mask.nonzero(as_tuple=True)[0][0].item() + pl
                    ids = ids[:first_eos + 1]
                    L = ids.shape[0]

                text = self.tokenizer.decode(ids[pl:], skip_special_tokens=True)
                resp_mask = torch.zeros(L, device=self.device)
                resp_mask[pl:] = 1

                if token_lps_batch is not None:
                    gen_len = min(L - pl, token_lps_batch.shape[1])
                    token_lp = token_lps_batch[b, :gen_len]
                else:
                    token_lp = torch.zeros(0, device=self.device)

                results.append({
                    "text": text,
                    "full_ids": ids,
                    "token_lp": token_lp,
                    "resp_mask": resp_mask,
                    "prompt_len": pl,
                })
        return results

    def train_step(self, batch: list[dict]) -> dict:
        cfg = self.cfg
        G = cfg.group_size
        prompts = [b["prompt"] for b in batch]
        B = len(prompts)

        # sample depth
        T = max(cfg.min_loops, min(cfg.max_loops, np.random.poisson(cfg.poisson_mean)))
        T_bwd = max(1, math.ceil(T * cfg.bptt_ratio))

        # Set BPTT depth on model; gradients only flow through last T_bwd iterations
        self._model_unwrapped._bptt_depth = T_bwd

        # snapshot old policy: save current trainable params into old-policy modules
        self._snapshot_old_policy()

        # generate + cache log‑probs
        if self.local_rank == 0:
            print(f"  [generate] T={T}, prompts={B}, G={G}...", flush=True)
        comps = self._generate(prompts, T)   # list of B*G dicts
        # Free GPU memory: move full_ids to CPU, drop token_lp (used only in GRPO branch)
        for c in comps:
            c["full_ids"] = c["full_ids"].cpu()

        # verify
        for i, c in enumerate(comps):
            b = i // G
            prob = batch[b]
            if prob.get("kind") == "asm":
                code = extract_code_c(c["text"])
                reward, fb = verify_compile(code)
            else:
                code = extract_code(c["text"])
                reward, fb = verify(code, prob["test"])
            c["reward"] = reward
            c["feedback"] = fb
            c["batch_idx"] = b

        # ── build per‑prompt correct demonstrations ──
        correct_by_prompt = {b: [] for b in range(B)}
        for c in comps:
            if c["reward"] > 0:
                correct_by_prompt[c["batch_idx"]].append(c["text"])

        for c in comps:
            if c["reward"] <= 0:
                demos = correct_by_prompt[c["batch_idx"]]
                k = batch[c["batch_idx"]].get("kind", "code")
                c["sdpo_feedback"] = build_feedback(c["text"], c["feedback"], prompts[c["batch_idx"]], demos, kind=k)

        # --- GRPO branch: correct samples (batched) ---
        correct = [c for c in comps if c["reward"] > 0]
        grpo_l = torch.tensor(0.0, device=self.device)
        if len(correct) >= 2:
            # Pad all correct completions to uniform length
            L_max = max(c["full_ids"].shape[0] for c in correct)
            batch_ids = torch.zeros(len(correct), L_max, dtype=torch.long, device=self.device)
            batch_mask = torch.zeros(len(correct), L_max, device=self.device)
            for j, c in enumerate(correct):
                L = c["full_ids"].shape[0]
                PL = c["prompt_len"]
                batch_ids[j, :L] = c["full_ids"].to(self.device)
                batch_mask[j, PL:] = 1
            rwd = torch.tensor([c["reward"] for c in correct], device=self.device)

            # Skip old-policy forward when all rewards identical (no GRPO signal).
            # Common at start: all correct → all reward=1, advantage=0.
            if rwd.std() < 1e-7:
                grpo_l = torch.tensor(0.0, device=self.device)
            else:
                # Current log-probs: already cached from _generate (no extra forward)
                lp = torch.zeros(len(correct), L_max, device=self.device)
                for j, c in enumerate(correct):
                    L = c["full_ids"].shape[0]
                    PL = c["prompt_len"]
                    lp[j, PL:L] = c["token_lp"][:L - PL]

                # Old-policy log-probs: forward with old-policy modules swapped in.
                # Uses context manager to guarantee restoration even on exception.
                with self._old_policy_ctx():
                    with torch.no_grad():
                        logits_old = self._model_unwrapped.forward(
                            input_ids=batch_ids, n_loops=T, return_logits=True)
                log_probs_old = F.log_softmax(logits_old.float(), dim=-1).to(logits_old.dtype)
                lp_old = torch.zeros(len(correct), L_max, device=self.device)
                for j, c in enumerate(correct):
                    L = c["full_ids"].shape[0]
                    PL = c["prompt_len"]
                    gen_pos = torch.arange(PL, L, device=self.device)
                    lp_old[j, PL:L] = log_probs_old[j, gen_pos - 1, batch_ids[j, gen_pos]]

                grpo_l = grpo_loss(lp, lp_old, rwd, batch_mask, cfg.clip_epsilon, cfg.clip_epsilon_high)

        # --- SDPO branch: failed samples (batched) ---
        failed = [c for c in comps if c["reward"] <= 0 and c.get("sdpo_feedback")]
        sdpo_l = torch.tensor(0.0, device=self.device)
        if failed:
            teacher_prompts = [
                f + "\n\nNow write the corrected code for:\n" + prompts[c["batch_idx"]]
                for c, f in [(c, c["sdpo_feedback"]) for c in failed]
            ]
            teacher_formatted = [self._apply_chat_template(p) for p in teacher_prompts]
            tp_enc = self._tokenize_safe(teacher_formatted)
            tp_lens = tp_enc["attention_mask"].sum(dim=-1).long()

            # Teacher generates batched (old policy, no grad).
            # Tokenization above is on CPU path: feedback text varies per
            # step. Acceptable for research (20 prompts, G=2). Production
            # would pre-tokenize or use async pipeline.
            # _old_policy_ctx is a zero-copy reference swap on the unwrapped
            # model (see definition at _old_policy_ctx). No state_dict copy,
            # no CUDA sync. Old-policy weights stored in same dtype (bf16)
            # as active model via load_state_dict in _snapshot_old_policy.
            with self._old_policy_ctx():
                with torch.no_grad():
                    teacher_gen = self._model_unwrapped.generate(
                        input_ids=tp_enc["input_ids"],
                        max_new_tokens=self.cfg.max_response_tokens,
                        n_loops=T,
                        temperature=0.6,
                        top_k=50,
                    )
            teacher_full_ids = teacher_gen
            TL = teacher_full_ids.shape[1]

            # Student: current policy (DDP, grad enabled)
            stu_logits = self.model.forward(
                input_ids=teacher_full_ids, n_loops=T, return_logits=True)
            # Teacher: old policy forward (no grad)
            with self._old_policy_ctx():
                with torch.no_grad():
                    tea_logits = self._model_unwrapped.forward(
                        input_ids=teacher_full_ids, n_loops=T, return_logits=True)

            # Batched response mask: (B, TL) where B = len(failed)
            resp_mask = torch.zeros(len(failed), TL, device=self.device)
            for j, pl in enumerate(tp_lens):
                resp_mask[j, pl.item():] = 1

            # Compute SDPO loss over entire batch (vectorized)
            sdpo_l = sdpo_loss(
                stu_logits, tea_logits, resp_mask, cfg.entropy_weight)

        total_loss = grpo_l + sdpo_l
        # Loss returned to caller for gradient accumulation.
        # Backward is called in train() after accumulating over micro-batches.

        metrics = {
            "total_loss": total_loss,
            "loss": total_loss.item(),
            "grpo_loss": grpo_l.item(),
            "sdpo_loss": sdpo_l.item(),
            "reward_mean": sum(c["reward"] for c in comps) / len(comps) if comps else 0,
            "T": T,
            "T_bwd": T_bwd,
            "rho": self._model_unwrapped.injection.compute_spectral_radius(),
            "n_correct": len(correct),
            "n_failed": len(failed),
        }
        return metrics

    def _snapshot_old_policy(self):
        """Copy current trainable params into old-policy modules."""
        m = self._model_unwrapped
        self.injection_old.load_state_dict(m.injection.state_dict())
        if self.depth_lora_old is not None and m.depth_lora is not None:
            self.depth_lora_old.load_state_dict(m.depth_lora.state_dict())

    @contextlib.contextmanager
    def _old_policy_ctx(self):
        """Context manager: temporarily swap model to old-policy modules.

        Swaps injection and depth_lora on the unwrapped model for
        old-policy forward() calls. Guarantees restoration on exit,
        so DDP gradient sync always sees the current (trainable) modules.

        DDP-safe: only touches _model_unwrapped, not the DDP wrapper.
        Pattern derived from TRL's reference model context managers
        (unwrap_model_for_generation in trl/models/utils.py).
        """
        m = self._model_unwrapped
        saved_inj = m.injection
        saved_lora = m.depth_lora
        m.injection = self.injection_old
        if self.depth_lora_old is not None and saved_lora is not None:
            m.depth_lora = self.depth_lora_old
        try:
            yield
        finally:
            m.injection = saved_inj
            m.depth_lora = saved_lora

    # ── training loop ──────────────────────────────────────────────

    def train(self):
        cfg = self.cfg
        if self.local_rank == 0:
            print(f"{'='*60}")
            print(f"SRPO · Recurrent-Depth Gemma E2B · {cfg.total_steps} steps")
            print(f"G={cfg.group_size} · T~Poisson({cfg.poisson_mean}) · μ_bwd≈{cfg.bptt_ratio}T")
            print(f"Device: {self.device} · World: {self.world_size}")
            print(f"{'='*60}")

        self.optimizer.zero_grad()
        data_iter = iter(self.dataloader)
        epoch = 0
        time_hist = []

        for step_idx in range(cfg.total_steps):
            t0 = time.time()

            # accumulate gradients over micro-batches (no_sync until last)
            for acc in range(cfg.gradient_accumulation_steps):
                is_last = (acc == cfg.gradient_accumulation_steps - 1)
                sync_ctx = self.model.no_sync() if self.world_size > 1 and not is_last else contextlib.nullcontext()
                try:
                    batch = next(data_iter)
                except StopIteration:
                    epoch += 1
                    if self._sampler is not None:
                        self._sampler.set_epoch(epoch)
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)

                # forward (autocast bf16), then backward scaled by GA steps
                with sync_ctx:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        metrics = self.train_step(list(batch))
                self.scaler.scale(metrics["total_loss"] / cfg.gradient_accumulation_steps).backward()

            # optimizer step
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.trainable_params(), cfg.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.step += 1

            dt = time.time() - t0
            time_hist.append(dt)

            # logging (rank 0 only)
            if self.local_rank == 0 and (step_idx % cfg.log_every == 0 or step_idx < 5):
                avg_t = sum(time_hist[-10:]) / len(time_hist[-10:])
                print(
                    f"step {step_idx:4d}/{cfg.total_steps} | "
                    f"loss={metrics.get('loss',0):.3f} | "
                    f"grpo={metrics.get('grpo_loss',0):.3f} | "
                    f"sdpo={metrics.get('sdpo_loss',0):.3f} | "
                    f"R̄={metrics.get('reward_mean',0):.3f} | "
                    f"T={metrics['T']} | "
                    f"ρ(A)={metrics['rho']:.4f} | "
                    f"{dt:.1f}s"
                )

            # eval
            if step_idx % cfg.eval_every == 0 and step_idx > 0:
                self._eval()

            # checkpoint
            if step_idx % cfg.save_every == 0 and step_idx > 0:
                self._save(step_idx)

        if self.local_rank == 0:
            print("Training complete.")

    def _eval(self):
        rho = self._model_unwrapped.injection.compute_spectral_radius()
        ok = rho < 1.0
        if self.local_rank == 0:
            print(f"  [eval] ρ(A)={rho:.6f} {'✓' if ok else '✗ UNSTABLE'}")

    def _save(self, step: int):
        if self.local_rank != 0:
            return
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            "step": step,
            "trainable": {n: p.data.clone() for n, p in self._model_unwrapped.named_parameters() if p.requires_grad},
            "optimizer": self.optimizer.state_dict(),
            "config": self.cfg,
        }, f"checkpoints/step_{step}.pt")
        print(f"  [save] checkpoints/step_{step}.pt")

    @classmethod
    def resume(cls, checkpoint_path: str, cfg_override: Optional[TrainConfig] = None):
        """Resume training from a checkpoint saved by _save().

        Restores model weights, optimizer state (including momentum buffers),
        and training step.  The pretrained backbone is reloaded via
        load_pretrained(); only trainable injection/LoRA weights and
        optimizer state come from the checkpoint.

        Args:
            checkpoint_path: Path to a step_N.pt file.
            cfg_override:    Optional TrainConfig override (e.g., to change
                              total_steps for a longer run).
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_cfg = ckpt["config"]
        cfg = cfg_override if cfg_override is not None else saved_cfg

        trainer = cls(cfg)
        trainer.step = ckpt["step"]

        # Restore trainable parameters into the freshly-loaded model
        m = trainer._model_unwrapped
        for name, param in m.named_parameters():
            if name in ckpt["trainable"]:
                param.data.copy_(ckpt["trainable"][name].to(param.device))

        # Restore optimizer state (momentum / variance buffers)
        # Must happen AFTER parameter data is restored so IDs match.
        trainer.optimizer.load_state_dict(ckpt["optimizer"])

        if trainer.local_rank == 0:
            print(f"Resumed from {checkpoint_path} at step {trainer.step}")
        return trainer


# ── entry ──────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from")
    ap.add_argument("--steps", type=int, default=None,
                    help="Override total_steps")
    args = ap.parse_args()

    if args.resume:
        cfg = TrainConfig()
        if args.steps is not None:
            cfg.total_steps = args.steps
        trainer = SRPOTrainer.resume(args.resume, cfg_override=cfg)
    else:
        cfg = TrainConfig()
        trainer = SRPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
