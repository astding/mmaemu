"""T4: test mutated NVRTC kernels directly, before smoke gates can disable them."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

import mmaemu
from mmaemu.backends import ChainRequest
from mmaemu.backends.cuda import kernels, modules, nvrtc
from mmaemu.testing import assert_bitwise_equal

pytestmark = pytest.mark.gpu
ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("_mutation_vectors", ROOT / "tests/test_kernels_ieee.py")
vectors = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vectors)


@pytest.fixture(scope="module")
def direct_v1():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    dev = torch.device("cuda", 0)
    return kernels.chain_fn_for(kernels.V1, ChainRequest(dev, "e4m3", "e4m3", "ada"))


def test_mutated_nvrtc_conformance(direct_v1):
    for case in vectors.CASES:
        a, b = vectors.make(*case)
        got = direct_v1(a.cuda(), b.cuda())
        assert_bitwise_equal(got, vectors.cached_reference(case), msg=str(case))
    assert modules.module_info("v1", torch.device("cuda", 0)).source == "nvrtc"


def test_mutated_nan_propagation(direct_v1):
    a = torch.ones(16, 32).to(torch.float8_e4m3fn).cuda()
    b = a.clone()
    a.view(torch.uint8)[0, 0] = 0x7F
    want = torch.full((16, 16), 32.0)
    want.view(torch.int32)[0] = 0x7FFFFFFF
    got = mmaemu.accumulate(a, b, use_fast_accum=True, splitk=1, backend="cuda:v1")
    assert_bitwise_equal(got, want)


def test_pinned_arithmetic_options(direct_v1):
    # This is a required compiler contract, even when a particular source
    # happens to generate the same arithmetic under a more permissive flag.
    opts = nvrtc.build_options(120)
    assert "--fmad=false" in opts
    assert "-default-device" in opts
    assert "--std=c++17" in opts
