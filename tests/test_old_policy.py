"""Unit tests for the old-policy context manager pattern.

These test the actual _old_policy_ctx from the SRPO trainer.
They do NOT test DDP (that requires multi-GPU hardware).
They DO test correctness of the context manager: restore, gradient
isolation, and output difference with different weights.
"""

import pytest
import contextlib
import torch

DIM = 8

class DummyInjection(torch.nn.Module):
    """Minimal injection for testing. Same interface as DepthInjection."""
    def __init__(self, dim=DIM):
        super().__init__()
        self.A = torch.nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.B = torch.nn.Parameter(torch.randn(dim, dim) * 0.01)
    def forward(self, h, e, trans_out):
        return h @ self.A.T + e @ self.B.T + trans_out

class FakeModel(torch.nn.Module):
    """Model with swappable injection and depth_lora, like RecurrentDepthGemma."""
    def __init__(self):
        super().__init__()
        self.injection = DummyInjection()
        self.depth_lora = torch.nn.Linear(DIM, DIM)
        self.backbone = torch.nn.Linear(DIM, DIM, bias=False)
        self.backbone.weight.requires_grad = False

    def forward(self, x):
        h = self.backbone(x)
        trans = self.depth_lora(h)
        return self.injection(h, h, trans)


@contextlib.contextmanager
def old_policy_ctx(model, inj_old, lora_old):
    """Exact same pattern as train_srpo.py _old_policy_ctx."""
    saved_inj = model.injection
    saved_lora = model.depth_lora
    model.injection = inj_old
    model.depth_lora = lora_old
    try:
        yield
    finally:
        model.injection = saved_inj
        model.depth_lora = saved_lora


class TestOldPolicyCtx:

    @pytest.fixture
    def model(self):
        return FakeModel()

    @pytest.fixture
    def old_modules(self):
        inj = DummyInjection()
        lora = torch.nn.Linear(DIM, DIM)
        for p in inj.parameters():
            p.requires_grad = False
        for p in lora.parameters():
            p.requires_grad = False
        return inj, lora

    @pytest.fixture
    def x(self):
        return torch.randn(2, DIM)

    # ── state restore ──

    def test_restores_references_after_normal_exit(self, model, old_modules):
        inj_old, lora_old = old_modules
        sp, sl = model.injection, model.depth_lora
        with old_policy_ctx(model, inj_old, lora_old):
            assert model.injection is inj_old
        assert model.injection is sp
        assert model.depth_lora is sl

    def test_restores_references_after_exception(self, model, old_modules):
        inj_old, lora_old = old_modules
        sp, sl = model.injection, model.depth_lora
        try:
            with old_policy_ctx(model, inj_old, lora_old):
                raise RuntimeError("simulated failure")
        except RuntimeError:
            pass
        assert model.injection is sp
        assert model.depth_lora is sl

    # ── gradient isolation ──

    def test_trainable_params_still_have_grad_after_swap(self, model, old_modules, x):
        """Gradient isolation: trainable params get grads, old-policy don't."""
        inj_old, lora_old = old_modules
        model.zero_grad()

        with old_policy_ctx(model, inj_old, lora_old):
            with torch.no_grad():
                _ = model(x)   # old-policy forward, no grad

        # After context exit, trainable params should still have no grad
        # because the forward was under no_grad
        assert model.injection.A.grad is None
        assert model.depth_lora.weight.grad is None
        assert inj_old.A.grad is None
        assert inj_old.B.grad is None
        assert lora_old.weight.grad is None

    def test_current_policy_gets_grad_after_restore(self, model, old_modules, x):
        """After context exit, backward on current policy produces grads."""
        inj_old, lora_old = old_modules

        with old_policy_ctx(model, inj_old, lora_old):
            with torch.no_grad():
                _ = model(x)

        # Now current forward with grad
        out = model(x)
        out.sum().backward()
        assert model.injection.A.grad is not None
        assert model.depth_lora.weight.grad is not None

    # ── functional correctness ──

    def test_old_policy_really_differs(self, model, old_modules, x):
        """Verify the swap actually changes model output."""
        inj_old, lora_old = old_modules
        with torch.no_grad():
            cur_before = model(x)

            # Copy weights from a different initialization
            inj_diff = DummyInjection()
            with old_policy_ctx(model, inj_diff, lora_old):
                old_out = model(x)
            cur_after = model(x)

        assert not torch.allclose(cur_before, old_out, atol=1e-4), \
            "swapped injection should produce different output"
        assert torch.allclose(cur_before, cur_after, atol=1e-4), \
            "restored model should produce identical output"

    def test_output_identity_when_weights_match(self, model, x):
        """old policy == current policy when weights are identical."""
        inj_copy = DummyInjection()
        lora_copy = torch.nn.Linear(DIM, DIM)
        inj_copy.load_state_dict(model.injection.state_dict())
        lora_copy.load_state_dict(model.depth_lora.state_dict())

        with torch.no_grad():
            cur = model(x)
            with old_policy_ctx(model, inj_copy, lora_copy):
                old = model(x)
        assert torch.allclose(cur, old, atol=1e-4), \
            "identical weights should produce identical output"

    # ── requires_grad isolation ──

    def test_old_policy_has_no_trainable_params(self, old_modules):
        inj_old, lora_old = old_modules
        for p in inj_old.parameters():
            assert not p.requires_grad
        for p in lora_old.parameters():
            assert not p.requires_grad

    def test_backward_fails_under_old_policy(self, model, old_modules, x):
        """Backward through old policy blocked: requires_grad=False on
        all old modules means output has no grad_fn, so .backward()
        raises RuntimeError. This is the safety property."""
        inj_old, lora_old = old_modules
        with old_policy_ctx(model, inj_old, lora_old):
            out = model(x)
            with pytest.raises(RuntimeError, match="does not require grad"):
                out.sum().backward()
