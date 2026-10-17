"""Device independence (T3; runbook gate G3). Markers: gpu + truth.

Every CUDA backend that mmaemu reports available on this device must give the same bits as the hardware-defined
reference, on fixed inputs recorded on the RTX 4060 (read from $MMAEMU_TRUTH_DIR, never written):

1. Evo 2 1B (truth evo2.evo2_1b_w2k, the 21 FP8 projections of one 2 kb window, M=2048, N=5760, K=1920; unit-scale
   fp32 accumulator, fast accumulation, S=1):
   - test_evo2_acc_rows: rows 0..R-1 of all 21 accumulators equal the stored hardware rows 0..255, for EVERY
     available backend (R = 256 for kernels; 4 for "reference" on the GPU, whose float64 emulation is slow);
   - test_evo2_acc_sha256: the sha256 of the FULL [2048, 5760] accumulator equals the recorded hardware sha256, for
     the backend that backend="auto" selects, and for every backend when MMAEMU_DI_FULL=all (or a comma list of
     backend names). Items are per (backend, call) so tools/run_gpu_tests.sh can split slow backends into several
     locked runs of < 10 min.
2. Emulator group (truth emulator.tcemu_kernels_fixed_inputs: the fixed inputs on which the audited prototype's CUDA
   kernels were recorded): every backend equals the recorded kernel output except at the outputs whose row of A or
   column of B holds an E4M3 NaN code (the prototype kernels decoded NaN as finite), where the expected
   value is the canonical NaN 0x7FFFFFFF; the CPU reference gives the same bits (checked on the small cases in full
   and on rows 0..3 of the large ones).
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import mmaemu
from mmaemu import errors as E
from mmaemu.testing import assert_bitwise_equal

pytestmark = [pytest.mark.gpu, pytest.mark.truth]
BACKENDS = ("reference", "cuda", "cuda:v1", "cuda:v4", "cuda:v9", "cuda:tc3")
CALLS = [f"call{i:02d}" for i in range(21)]
_TRUTH = {}


def truth_loader():
    d = os.environ.get("MMAEMU_TRUTH_DIR")
    if not d:
        pytest.skip("MMAEMU_TRUTH_DIR not set")
    if "loader" not in _TRUTH:
        sys.dont_write_bytecode = True
        root = str(Path(d).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        from tcemu_ada_truth import loader  # noqa: WPS433
        _TRUTH["loader"] = loader
    return _TRUTH["loader"]


def rec(rid):
    key = ("rec", rid)
    if key not in _TRUTH:
        _TRUTH[key] = truth_loader().load(rid)
    return _TRUTH[key]


def device():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return torch.device("cuda", 0)


def available(backend, dev):
    st = mmaemu.backends_status(dev)
    if backend not in st:
        pytest.skip(f"{backend!r} not reported by mmaemu.backends_status()")
    if not st[backend].available:
        pytest.skip(f"{backend!r} unavailable on {torch.cuda.get_device_name(dev)}: {st[backend].reason}")


def auto_backend(dev):
    req = mmaemu.backends.ChainRequest(device=dev, fmt_a="e4m3", fmt_b="e4m3", preset="ada")
    return mmaemu.backends.select(req, "auto").name


def u8(a, dev):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.uint8).copy()).view(torch.float8_e4m3fn).to(dev)


def u32_to_f32(bits):
    return torch.from_numpy(np.ascontiguousarray(bits, dtype=np.uint32).view(np.int32).copy()).view(torch.float32)


def bits32(t):
    return t.detach().cpu().contiguous().view(torch.int32).numpy().view(np.uint32)


def evo2_case(call):
    r = rec("evo2.evo2_1b_w2k")
    case = next(c for c in r.cases if c.id == call)
    outs = {o.name: o for o in case.outputs}
    return case, outs


def accumulate(a, b, backend):
    return mmaemu.accumulate(a, b, use_fast_accum=True, splitk=1, backend=backend, profile="ada")


# ---------------------------------------------------------------------------------------------------------------
IDS = [b.replace(":", "_") for b in BACKENDS]


@pytest.mark.parametrize("backend", BACKENDS, ids=IDS)
def test_evo2_acc_rows(backend):
    dev = device()
    available(backend, dev)
    R = 4 if backend == "reference" else 256
    for call in CALLS:
        case, outs = evo2_case(call)
        ins = case.inputs()
        acc_rows = outs["acc"]                       # stored hardware rows 0..255 of the unit-scale accumulator
        assert acc_rows.rows == (0, 256)
        try:
            got = accumulate(u8(ins["a"][:R], dev), u8(ins["b"], dev), backend).cpu()
        except E.BackendUnavailableError as e:
            pytest.fail(f"{backend} reported available but raised BackendUnavailableError: {e}")
        want = u32_to_f32(acc_rows.bits()[:R])
        assert_bitwise_equal(got, want, msg=f"{backend} evo2 {call} rows 0..{R - 1}")
        case.drop_inputs()


@pytest.mark.parametrize("call", CALLS)
@pytest.mark.parametrize("backend", BACKENDS, ids=IDS)
def test_evo2_acc_sha256(backend, call):
    dev = device()
    full = os.environ.get("MMAEMU_DI_FULL", "")
    wanted = full == "all" or backend in full.split(",")
    if not wanted:
        available(backend, dev)
        if backend != auto_backend(dev):
            pytest.skip(f"full accumulator for {backend} only with MMAEMU_DI_FULL=all or ={backend}")
    if backend == "reference" and not wanted:
        pytest.skip("reference on the GPU is too slow for full Evo 2 accumulators (use MMAEMU_DI_FULL=reference)")
    available(backend, dev)
    case, outs = evo2_case(call)
    ins = case.inputs()
    acc_full = outs["acc_full"]                      # hash-only: sha256 of the full hardware accumulator
    got = accumulate(u8(ins["a"], dev), u8(ins["b"], dev), backend)
    b = bits32(got)
    assert list(b.shape) == list(acc_full.shape)
    sha = hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest()
    case.drop_inputs()
    assert sha == acc_full.sha256, f"{backend} evo2 {call}: accumulator sha256 {sha} != hardware {acc_full.sha256}"


# ---------------------------------------------------------------------------------------------------------------
def emulator_expected(case, out):
    """Recorded old-kernel bits, with canonical NaN where a row of A / column of B holds an E4M3 NaN code."""
    ins = case.inputs()
    hw = out.bits().copy()
    nan_a = ((ins["a"].astype(np.int16) & 0x7F) == 0x7F).any(1)
    nan_b = ((ins["b"].astype(np.int16) & 0x7F) == 0x7F).any(1)
    mask = nan_a[:, None] | nan_b[None, :]
    hw[mask] = 0x7FFFFFFF
    return hw, int(mask.sum())


@pytest.mark.parametrize("backend", BACKENDS, ids=IDS)
def test_emulator_group(backend):
    dev = device()
    available(backend, dev)
    r = rec("emulator.tcemu_kernels_fixed_inputs")
    n_nan = 0
    for case in r.cases:
        out = next(o for o in case.outputs if o.name == "v1_cfg0")
        exp, nm = emulator_expected(case, out)
        n_nan += nm
        ins = case.inputs()
        a, b = ins["a"], ins["b"]
        rows = a.shape[0] if backend != "reference" else min(a.shape[0], 4)
        got = accumulate(u8(a[:rows], dev), u8(b, dev), backend).cpu()
        assert_bitwise_equal(got, u32_to_f32(exp[:rows]), msg=f"{backend} emulator {case.id}")
        # the CPU reference agrees (all of the small cases, rows 0..3 of the large ones)
        small = a.shape[0] * b.shape[0] * a.shape[1] <= 3e7
        rr = a.shape[0] if small else 4
        ref = accumulate(u8(a[:rr], "cpu"), u8(b, "cpu"), "reference")
        assert_bitwise_equal(ref, u32_to_f32(exp[:rr]), msg=f"CPU reference emulator {case.id}")
        case.drop_inputs()
    assert n_nan > 0, "the emulator group has no NaN-decode outputs (wrong recording?)"
