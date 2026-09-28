# Correctness of the FlashAttention-2 forward and backward against PyTorch's SDPA.
# CPU runs the PyTorch version; the Triton tests need a CUDA GPU and are skipped otherwise.
import pytest
import torch
import torch.nn.functional as F

from flash.flash_attention import FlashForward

cuda = torch.cuda.is_available()


def inputs(device, T=128, d=64):
    torch.manual_seed(0)
    q, k, v, do = (torch.randn(4, T, d, device=device) for _ in range(4))
    return [t.requires_grad_() for t in (q, k, v)], do


def check(fn, device, causal):
    (q, k, v), do = inputs(device)
    o = fn(q, k, v, causal)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    torch.testing.assert_close(o, ref, rtol=1e-2, atol=1e-2)
    g = torch.autograd.grad(o, (q, k, v), do)
    g_ref = torch.autograd.grad(ref, (q, k, v), do)
    for a, b in zip(g, g_ref):
        torch.testing.assert_close(a, b, rtol=1e-2, atol=1e-2)


def test_pytorch():
    check(FlashForward.apply, "cpu", False)


@pytest.mark.skipif(not cuda, reason="Triton needs a CUDA GPU")
@pytest.mark.parametrize("causal", [False, True])
def test_triton(causal):
    from flash.flash_attention_triton import FlashForwardTriton
    check(FlashForwardTriton.apply, "cuda", causal)
