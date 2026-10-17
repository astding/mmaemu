"""Reference backend: protocol, equality with the model, the temporary-memory bound, and
(-m gpu) CPU == CUDA bitwise on every toolbox set plus peak CUDA memory of chain_gemm at 2048x5760x1920."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from mmaemu.backends import ChainRequest
from mmaemu.backends.reference import REFERENCE, ReferenceBackend
from mmaemu.errors import ExperimentalPresetWarning
from mmaemu.model import chain as C
from mmaemu.model.chain import chain_gemm, dot
from mmaemu.testing import assert_bitwise_equal

E4, E5 = torch.float8_e4m3fn, torch.float8_e5m2
TB = Path(__file__).parent / "data" / "toolbox"


def _req(dev="cpu", fa="e4m3", fb="e4m3", preset="ada"):
    return ChainRequest(device=torch.device(dev), fmt_a=fa, fmt_b=fb, preset=preset)


def test_protocol():
    assert isinstance(REFERENCE, ReferenceBackend) and REFERENCE.name == "reference"
    for fa in ("e4m3", "e5m2"):
        for fb in ("e4m3", "e5m2"):
            for preset in ("ada", "hopper", "b200_mma_sync"):
                assert REFERENCE.supports(_req(fa=fa, fb=fb, preset=preset)) is None
    g = torch.Generator().manual_seed(2)
    a = (torch.randn(5, 64, generator=g) * 40).clamp(-448, 448).to(E4).view(torch.uint8)
    b = (torch.randn(3, 64, generator=g) * 40).clamp(-448, 448).to(E4).view(torch.uint8)
    fn = REFERENCE.chain_fn(_req())
    assert_bitwise_equal(fn(a, b), chain_gemm(a, b, fmt_a="e4m3", fmt_b="e4m3"))


def test_select_returns_reference_on_cpu():
    from mmaemu.backends import select
    be = select(_req(), "auto")
    assert be.name == "reference"


@pytest.mark.parametrize("M,N,L", [(2048, 5760, 1920), (16384, 24576, 8192), (1, 1 << 20, 64), (1 << 20, 1, 64),
                                   (4, 768, 1920)])
@pytest.mark.parametrize("dev", ["cpu", "cuda"])
def test_tile_plan_respects_budget(M, N, L, dev):
    for nfma in (16, 32):
        tm, tn, kc = C._tile_plan(M, N, L, nfma, torch.device(dev))
        nbytes = tm * tn * (6 * kc + 2 * (kc // nfma) + 4 * nfma + 64) + 14 * (tm + tn) * kc
        assert nbytes <= C.TEMP_BYTES <= 256 << 20 and kc % nfma == 0


def _toolbox_sets():
    meta = json.loads((TB / "SOURCES.json").read_text())
    z = np.load(TB / "vectors.npz", allow_pickle=False)

    def f32(k):
        return torch.from_numpy(z[k].view(np.int32).copy()).view(torch.float32)
    preset = {"Ada": "ada", "L40S": "ada", "H100": "hopper", "H200": "hopper", "B200": "b200_mma_sync"}
    for s in meta["sets"]:
        c = f32(s["c"]) if s["gpu"] not in ("H100", "H200") else torch.zeros(5000)
        yield s, preset[s["gpu"]], f32(s["a"]).double(), f32(s["b"]).double(), c, f32(s["d"])


@pytest.mark.gpu
def test_cpu_equals_cuda_on_toolbox_sets():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalPresetWarning)
        for s, preset, a, b, c, d in _toolbox_sets():
            cpu = dot(a, b, c, preset=preset, fmt=s["fmt"])
            gpu = dot(a.cuda(), b.cuda(), c.cuda(), preset=preset, fmt=s["fmt"])
            assert_bitwise_equal(gpu, cpu, msg=f"dot {s['gpu']} {s['fmt']}")
            assert_bitwise_equal(cpu, d, msg="vs toolbox")
            # GEMM-shaped: all 1024 x 1024 cross products of the first 1024 a / b vectors
            ac = a[:1024].float().to(E4 if s["fmt"] == "e4m3" else E5).view(torch.uint8)
            bc = b[:1024].float().to(E4 if s["fmt"] == "e4m3" else E5).view(torch.uint8)
            g_cpu = chain_gemm(ac, bc, fmt_a=s["fmt"], fmt_b=s["fmt"], preset=preset)
            g_gpu = chain_gemm(ac.cuda(), bc.cuda(), fmt_a=s["fmt"], fmt_b=s["fmt"], preset=preset)
            assert_bitwise_equal(g_gpu, g_cpu, msg=f"chain_gemm {s['gpu']} {s['fmt']}")


@pytest.mark.gpu
def test_scaled_mm_reference_cpu_equals_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from mmaemu.api import scaled_mm
    g = torch.Generator().manual_seed(8)
    a = (torch.randn(96, 1024, generator=g) * 40).clamp(-448, 448).to(E4)
    b = (torch.randn(160, 1024, generator=g) * 40).clamp(-448, 448).to(E4)
    a.view(torch.uint8)[3, 7] = 0x7F                                 # a NaN row
    sa, sb = torch.tensor(0.0123), torch.tensor(-0.000731)
    for od in (torch.float32, torch.bfloat16, torch.float16, E4):
        for fast in (True, False):
            for sk in (1, (3, "pairwise"), (8, "serial")):
                kw = dict(out_dtype=od, use_fast_accum=fast, splitk=sk, backend="reference",
                          profile="ada")
                cpu = scaled_mm(a, b.t(), sa, sb, **kw)
                gpu = scaled_mm(a.cuda(), b.cuda().t(), sa.cuda(), sb.cuda(), **kw)
                assert gpu.device.type == "cuda"
                assert_bitwise_equal(gpu, cpu, msg=f"{od} fast={fast} splitk={sk}")


@pytest.mark.gpu
def test_chain_gemm_cuda_memory_bound():
    """Peak extra CUDA memory of chain_gemm at 2048 x 5760 x 1920 (Evo 2 1B projection, M = 2048) <= 256 MB above
    the inputs (the fp32 output itself, 47 MB, included)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    g = torch.Generator(device="cuda").manual_seed(0)
    a = (torch.randn(2048, 1920, device="cuda", generator=g) * 40).clamp(-448, 448).to(E4).view(torch.uint8)
    b = (torch.randn(5760, 1920, device="cuda", generator=g) * 40).clamp(-448, 448).to(E4).view(torch.uint8)
    chain_gemm(a[:16], b[:16], fmt_a="e4m3", fmt_b="e4m3")          # LUTs and allocator warm-up
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    out = chain_gemm(a, b, fmt_a="e4m3", fmt_b="e4m3")
    t1.record()
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - base
    secs = t0.elapsed_time(t1) / 1000
    print(f"chain_gemm 2048x5760x1920 on {torch.cuda.get_device_name()}: peak extra {extra / 2**20:.1f} MiB, "
          f"{secs:.2f} s (CUDA events), {2048 * 5760 * 1920 / secs / 1e9:.1f} GMAC/s")
    assert extra <= 256 << 20
    # spot-check a block of the result against the CPU reference
    sub = chain_gemm(a[:64].cpu(), b[:96].cpu(), fmt_a="e4m3", fmt_b="e4m3")
    assert_bitwise_equal(out[:64, :96].cpu(), sub)
