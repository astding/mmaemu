"""torch.compile and CPU flush-to-zero: the public entry points are wrapped in torch.compiler.disable, so
a compiled caller gets the same bits as eager (Inductor/Triton <= 3.8 fp8 conversion double-rounds, as the tcemu
audit found); torch.set_flush_denormal(True) makes the CPU path raise MmaemuError instead of returning flushed
results."""
from __future__ import annotations

import pytest
import torch

from mmaemu.api import accumulate, scaled_mm
from mmaemu.errors import MmaemuError
from mmaemu.testing import assert_bitwise_equal

E4, E5 = torch.float8_e4m3fn, torch.float8_e5m2


def _caller(a, b, sa, sb):
    y = scaled_mm(a, b.t(), sa, sb, out_dtype=torch.bfloat16, use_fast_accum=True, splitk=1)
    y8 = scaled_mm(a, b.t(), sa, sb, out_dtype=E4, use_fast_accum=False, splitk=1)
    acc = accumulate(a, b, use_fast_accum=True, splitk=1)
    return y, y8, acc, (acc * 2.0).float()


def _inputs(device, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = (torch.randn(48, 256, generator=g) * 40).clamp(-448, 448).to(E4)
    b = (torch.randn(64, 256, generator=g) * 40).clamp(-448, 448).to(E4)
    return a.to(device), b.to(device), torch.tensor(0.0123, device=device), torch.tensor(3.7, device=device)


def _check(device):
    args = _inputs(device)
    eager = _caller(*args)
    torch._dynamo.reset()
    compiled = torch.compile(_caller)
    try:
        got = compiled(*args)
    except Exception as e:  # noqa: BLE001
        if "mmaemu" in repr(e) or isinstance(e, MmaemuError):
            raise
        pytest.skip(f"torch.compile unavailable here: {type(e).__name__}: {str(e)[:120]}")
    for x, y in zip(got, eager):
        assert_bitwise_equal(x, y)


def test_compiled_caller_identical_bits_cpu():
    _check("cpu")


@pytest.mark.gpu
def test_compiled_caller_identical_bits_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    _check("cuda")


def test_flush_denormal_raises():
    a, b, sa, sb = _inputs("cpu")
    if not torch.set_flush_denormal(True):
        pytest.skip("this CPU cannot flush denormals")
    try:
        with pytest.raises(MmaemuError, match="flush"):
            scaled_mm(a, b.t(), sa, sb, out_dtype=torch.float32, splitk=1)
        with pytest.raises(MmaemuError, match="flush"):
            accumulate(a, b, use_fast_accum=True, splitk=1)
    finally:
        torch.set_flush_denormal(False)
    assert scaled_mm(a, b.t(), sa, sb, out_dtype=torch.float32, splitk=1).shape == (48, 64)
