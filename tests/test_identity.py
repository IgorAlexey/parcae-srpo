#!/usr/bin/env python3
"""Identity test: at n_loops=1, our model must match HuggingFace Gemma 4."""

import sys, os
import torch
from parcae import RecurrentDepthGemma, RecurrentDepthConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "google/gemma-4-E2B-it",
)

PROMPTS = [
    "def add(a, b): return a + b",
    "def factorial(n):",
    "import torch\n",
    "class MyModel(nn.Module):",
    "# This is a comment",
    "x = [1, 2, 3]",
    "for i in range(10):",
]

TOL = 0.2  # bf16 tolerance

import pytest


class TestIdentity:
    @pytest.fixture(scope="class")
    @classmethod
    def hf_model(cls):
        hf = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cpu",
            low_cpu_mem_usage=True)
        hf.eval()
        return hf

    @pytest.fixture(scope="class")
    @classmethod
    def our_model(cls):
        cfg = RecurrentDepthConfig(
            model_path=MODEL_PATH, prelude_layers=12,
            n_recurrent_layers=11, coda_layers=12)
        model = RecurrentDepthGemma(cfg)
        model.load_pretrained()
        model.eval()
        return model

    @pytest.fixture(scope="class")
    @classmethod
    def tokenizer(cls):
        return AutoTokenizer.from_pretrained(MODEL_PATH)

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_identity_prompt(self, hf_model, our_model, tokenizer, prompt):
        ids = tokenizer(prompt, return_tensors='pt')
        hf_logits = hf_model(ids.input_ids).logits
        our_logits = our_model(ids.input_ids, n_loops=1)

        diff = (hf_logits.float() - our_logits.float()).abs().max().item()
        top_hf = hf_logits[0, -1].argmax().item()
        top_our = our_logits[0, -1].argmax().item()

        assert diff < TOL, f"max diff {diff:.6f} exceeds tolerance {TOL}"
        assert top_hf == top_our, f"top token mismatch: {top_hf} vs {top_our}"
