"""Instruction-level chain model: M1, canonical NaN, the B13 input rules, chain_gemm == dot on
adversarial data, and (-m "truth and slow") every FP8 mma.sync recording of the RTX 4060 truth dataset."""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pytest
import torch

from mmaemu.errors import ExperimentalPresetWarning
from mmaemu.formats import decode, nonfinite_mask
from mmaemu.model import chain as C
from mmaemu.model.chain import chain_gemm, dot
from mmaemu.model.presets import get_preset
from mmaemu.testing import assert_bitwise_equal

E4, E5 = torch.float8_e4m3fn, torch.float8_e5m2
F8 = {"e4m3": E4, "e5m2": E5}


def f32_bits(t: torch.Tensor) -> list:
    return [int(v) & 0xFFFFFFFF for v in t.contiguous().view(torch.int32).flatten().tolist()]


# --------------------------------------------------------------------------------------------------------------------
# M1 (c2-M1): an exactly-zero or truncated-to-zero block result is +0
@pytest.mark.parametrize("c", [-(2.0 ** -139 - 2.0 ** -149), -2.0 ** -140, -2.0 ** -145, -0.0])
def test_m1_zero_block_result_is_plus_zero(c):
    a = torch.zeros(1, 16, dtype=torch.float64)
    cc = torch.tensor([c], dtype=torch.float32)
    d = dot(a, a, cc, fmt="e4m3")
    assert f32_bits(d) == [0], c
    # without the M1 rule the toolbox formula gives -0 for these c (what the hardware never returns)
    if c != -0.0:
        E = -132.0                                            # no products: E = max(exp(c), min_exp)
        dd = torch.trunc(torch.tensor([c], dtype=torch.float64) * 2.0 ** (13 - E)) * 2.0 ** (E - 13)
        if dd.item() != 0:
            ed = max(float(torch.frexp(dd)[1].item() - 1), -126.0)
            dd = torch.trunc(dd * 2.0 ** (13 - ed)) * 2.0 ** (ed - 13)
        assert f32_bits(dd.float()) == [0x80000000]


def test_m1_cancellation_gives_plus_zero():
    a = torch.tensor([[1.0, -1.0] + [0.0] * 14], dtype=torch.float64)
    b = torch.ones(1, 16, dtype=torch.float64)
    assert f32_bits(dot(a, b, fmt="e4m3")) == [0]
    assert f32_bits(dot(-a, b, torch.tensor([-0.0]), fmt="e4m3")) == [0]


def test_dot_returns_canonical_nan():
    a = torch.zeros(3, 32, dtype=torch.uint8)
    a[0, 3] = 0xFF                       # negative E4M3 NaN
    a[1, 0] = 0x7F
    b = torch.full((3, 32), 0x38, dtype=torch.uint8)    # 1.0
    d = dot(a.view(E4), b.view(E4))
    assert f32_bits(d)[:2] == [0x7FFFFFFF, 0x7FFFFFFF]
    # E5M2 Inf * 0 -> NaN; c = -NaN -> canonical NaN
    a5 = torch.zeros(2, 16, dtype=torch.uint8)
    a5[0, 0] = 0xFC                      # -Inf
    d = dot(a5.view(E5), torch.zeros(2, 16, dtype=torch.uint8).view(E4),
            torch.tensor([0.0, 0.0]))
    assert f32_bits(d) == [0x7FFFFFFF, 0]
    neg_nan = torch.tensor([-1], dtype=torch.int32).view(torch.float32)
    d = dot(torch.ones(1, 16).to(E4), torch.ones(1, 16).to(E4), neg_nan)
    assert f32_bits(d) == [0x7FFFFFFF]


# --------------------------------------------------------------------------------------------------------------------
# B13: format inference, explicit fmt for float64, no broadcasting
def test_b13_fmt_inferred_from_dtypes_v05_minimal_m1():
    """verify-a1-reference-review-4/scripts/v05_minimal.py M1: a = [2^-9, 2^-12] (E5M2), b = [1, 2^-9] (E4M3).
    The RTX 4060 (torch._scaled_mm, 16x32x16) returned 0x1.001p-9; the old default fmt='e4m3' gave 0x1p-9."""
    a = torch.zeros(1, 16, dtype=torch.float64)
    a[0, 0], a[0, 1] = 2.0 ** -9, 2.0 ** -12
    b = torch.zeros(1, 16, dtype=torch.float64)
    b[0, 0], b[0, 1] = 1.0, 2.0 ** -9
    hw = 2.0 ** -9 + 2.0 ** -21
    assert dot(a.to(E5), b.to(E4)).item() == hw
    assert dot(a, b, fmt=("e5m2", "e4m3")).item() == hw
    with pytest.raises(ValueError, match="not e4m3 values"):
        dot(a, b, fmt="e4m3")            # 2^-12 is not an E4M3 value: the old silent hazard now raises


def test_b13_fmt_required_for_float64():
    x = torch.ones(2, 16, dtype=torch.float64)
    with pytest.raises(TypeError, match="fmt"):
        dot(x, x)
    with pytest.raises(TypeError):
        dot(x.float(), x.float(), fmt="e4m3")      # float32 is neither FP8 nor float64
    with pytest.raises(ValueError, match="dtype"):
        dot(x.to(E4), x.to(E4), fmt="e5m2")         # fmt contradicts the dtype


@pytest.mark.parametrize("sa,sb", [((1, 16), (1, 32)), ((1, 32), (1, 16)), ((1, 16), (1, 1)), ((2, 16), (3, 16)),
                                   ((2, 1, 16), (1, 3, 16))])
def test_b13_no_broadcasting(sa, sb):
    with pytest.raises(ValueError, match="same shape"):
        dot(torch.ones(sa).to(E4), torch.ones(sb).to(E4))


def test_b13_values_must_be_fp8():
    x = torch.full((1, 16), 0.1, dtype=torch.float64)      # v05_minimal M3: 0.1 was silently accepted
    with pytest.raises(ValueError, match="not e4m3 values"):
        dot(x, x, fmt="e4m3")


def test_c_checks():
    x = torch.ones(2, 16).to(E4)
    with pytest.raises(ValueError, match="c must have shape"):
        dot(x, x, torch.zeros(3))
    with pytest.raises(ValueError, match="not fp32 values"):
        dot(x, x, torch.tensor([0.1, 0.0], dtype=torch.float64))
    with pytest.raises(TypeError):
        dot(x, x, torch.zeros(2, dtype=torch.bfloat16))


def test_ragged_k_is_zero_padded():
    g = torch.Generator().manual_seed(3)
    a = (torch.randn(8, 37, generator=g) * 40).clamp(-448, 448).to(E4)
    b = (torch.randn(8, 37, generator=g) * 40).clamp(-448, 448).to(E4)
    pad = torch.zeros(8, 11).to(E4)
    assert_bitwise_equal(dot(a, b), dot(torch.cat([a, pad], 1), torch.cat([b, pad], 1)))


# --------------------------------------------------------------------------------------------------------------------
# chain_gemm == dot (the restructured fast path against the readable model)
def _codes(shape, fmt, kind, g):
    n = int(np.prod(shape))
    if kind == "allcodes":
        c = torch.randint(0, 256, (n,), generator=g, dtype=torch.int32).to(torch.uint8)
    elif kind == "gauss40":
        c = (torch.randn(n, generator=g) * 40).clamp(-448, 448).to(F8[fmt]).view(torch.uint8)
    elif kind == "tiny":                 # subnormals and the smallest normals: the finest alignment grids
        c = torch.randint(0, 12, (n,), generator=g, dtype=torch.int32).to(torch.uint8)
        c |= (torch.randint(0, 2, (n,), generator=g, dtype=torch.int32).to(torch.uint8) << 7)
    elif kind == "maxmag":
        top = 0x7E if fmt == "e4m3" else 0x7B
        c = torch.randint(top - 8, top + 1, (n,), generator=g, dtype=torch.int32).to(torch.uint8)
        c |= (torch.randint(0, 2, (n,), generator=g, dtype=torch.int32).to(torch.uint8) << 7)
    elif kind == "sparse":
        c = torch.randint(0, 256, (n,), generator=g, dtype=torch.int32).to(torch.uint8)
        c[torch.rand(n, generator=g) < 0.85] = 0
    elif kind == "spread":               # one huge term per block plus tiny ones: deep truncation
        c = torch.randint(0, 24, (n,), generator=g, dtype=torch.int32).to(torch.uint8)
        c[::16] = 0x70 if fmt == "e4m3" else 0x6C
    elif kind == "cancel":               # +x, -x pairs: exact cancellation, zero results, sign of zero
        half = torch.randint(0, 0x7E, (n // 2,), generator=g, dtype=torch.int32).to(torch.uint8)
        c = torch.stack([half, half | 0x80], 1).flatten()
    else:
        raise ValueError(kind)
    c = c.reshape(shape)
    c[nonfinite_mask(c, fmt)] = 0x01
    return c


KINDS = ["allcodes", "gauss40", "tiny", "maxmag", "sparse", "spread", "cancel"]


@pytest.mark.parametrize("preset", ["ada", "hopper", "b200_mma_sync"])
@pytest.mark.parametrize("fa,fb", [("e4m3", "e4m3"), ("e4m3", "e5m2"), ("e5m2", "e4m3"), ("e5m2", "e5m2")])
def test_chain_gemm_equals_dot(preset, fa, fb):
    g = torch.Generator().manual_seed(hash((preset, fa, fb)) & 0xFFFF)
    p = get_preset(preset, warn=False)
    for kind in KINDS:
        M, N, L = 19, 23, 16 * 13
        a, b = _codes((M, L), fa, kind, g), _codes((N, L), fb, kind, g)
        if kind == "cancel":
            b = torch.full((N, L), 0x38 if fb == "e4m3" else 0x3C, dtype=torch.uint8)   # all ones
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalPresetWarning)
            got = chain_gemm(a, b, fmt_a=fa, fmt_b=fb, preset=preset)
        A = decode(a, fa)[:, None, :].expand(M, N, L)
        B = decode(b, fb)[None, :, :].expand(M, N, L)
        ref = C._canon_nan32(C._dot_values(A, B, None, p, (fa, fb)).float())
        assert_bitwise_equal(got, ref, msg=f"{preset} {fa}x{fb} {kind}")


def test_chain_gemm_tiling_does_not_change_bits(monkeypatch):
    g = torch.Generator().manual_seed(11)
    a, b = _codes((70, 512), "e4m3", "gauss40", g), _codes((90, 512), "e4m3", "gauss40", g)
    ref = chain_gemm(a, b, fmt_a="e4m3", fmt_b="e4m3")
    monkeypatch.setattr(C, "TEMP_BYTES", 1 << 18)            # forces tiny tiles and K chunks
    assert_bitwise_equal(chain_gemm(a, b, fmt_a="e4m3", fmt_b="e4m3"), ref)


def test_chain_gemm_contract():
    a = torch.zeros(4, 32, dtype=torch.uint8)
    with pytest.raises(ValueError, match="multiple of 16"):
        chain_gemm(a[:, :20], a[:, :20], fmt_a="e4m3", fmt_b="e4m3")
    bad = a.clone()
    bad[0, 0] = 0x7F
    with pytest.raises(ValueError, match="finite"):
        chain_gemm(bad, a, fmt_a="e4m3", fmt_b="e4m3")
    with pytest.raises(TypeError):
        chain_gemm(a.view(E4), a, fmt_a="e4m3", fmt_b="e4m3")
    with pytest.raises(ValueError):
        chain_gemm(a, a[:, :16], fmt_a="e4m3", fmt_b="e4m3")
    assert chain_gemm(a[:0], a, fmt_a="e4m3", fmt_b="e4m3").shape == (0, 4)
    z = chain_gemm(a[:, :0], a[:, :0], fmt_a="e4m3", fmt_b="e4m3")
    assert f32_bits(z) == [0] * 16                            # K = 0: +0


# --------------------------------------------------------------------------------------------------------------------
# truth: every FP8 mma.sync recording (c2.*, c6.*, *_mma*, c3.instr_specials; HMMA and fp16-acc excluded)
MMA_CHUNK = 1 << 16       # outputs per dot() call
MMA_ELEMS = 1 << 22       # products per dot() call (bounds the float64 operands to ~32 MB each)


def _mma_outputs(loader, root):
    out = []
    for e in loader.list_recordings(root):
        if e["kind"] != "mma":
            continue
        rec = loader.load(e["id"], root=root)
        fp8 = True
        for case in rec.cases:
            for o in case.outputs:
                p = o.params
                if p.get("fmt_a", "e4m3") not in F8 or p.get("fmt_b", "e4m3") not in F8 or p.get("acc", "f32") != "f32":
                    fp8 = False
        if fp8:
            out.append(e["id"])
    return out


def _replay_mma(a, b, c, p):
    fa, fb = p.get("fmt_a", "e4m3"), p.get("fmt_b", "e4m3")
    cf = torch.from_numpy(np.ascontiguousarray(c).view(np.int32).copy()).view(torch.float32)
    if p.get("layout", "vector") == "tile":
        T, mi, K = a.shape
        nj = b.shape[1]
        out = np.empty((T, mi, nj), np.uint32)
        step = max(1, min(MMA_CHUNK // (mi * nj), MMA_ELEMS // (mi * nj * K)))
        for t0 in range(0, T, step):
            A = torch.from_numpy(a[t0:t0 + step]).view(F8[fa])[:, :, None, :].expand(-1, mi, nj, K)
            B = torch.from_numpy(b[t0:t0 + step]).view(F8[fb])[:, None, :, :].expand(-1, mi, nj, K)
            d = dot(A, B, cf[t0:t0 + step])
            out[t0:t0 + step] = d.view(torch.int32).numpy().view(np.uint32)
        return out
    n = a.shape[0]
    out = np.empty((n,), np.uint32)
    step = max(1, min(MMA_CHUNK, MMA_ELEMS // a.shape[1]))
    for i0 in range(0, n, step):
        A = torch.from_numpy(a[i0:i0 + step]).view(F8[fa])
        B = torch.from_numpy(b[i0:i0 + step]).view(F8[fb])
        out[i0:i0 + step] = dot(A, B, cf[i0:i0 + step]).view(torch.int32).numpy().view(np.uint32)
    return out


@pytest.mark.truth
@pytest.mark.slow
def test_truth_mma_recordings():
    from test_epilogue import truth_dir, truth_loader
    loader = truth_loader()
    root = truth_dir()
    fixed = {r["id"]: r for r in json.loads((root / "replay" / "replay_fixed.json").read_text())["recordings"]}
    ids = _mma_outputs(loader, root)
    assert "c2.S16_zeros" in ids and "c3.instr_specials" in ids
    assert not any(i.startswith("hmma.") for i in ids) and "c3.instr_f16acc" not in ids
    t0 = time.monotonic()
    report, bad = {}, []
    for rid in ids:
        t = time.monotonic()
        rec = loader.load(rid, root=root)
        n = match = 0
        for case in rec.cases:
            ins = case.inputs()
            for o in case.outputs:
                hw = o.bits()
                got = _replay_mma(ins["a"], ins["b"], ins["c"], o.params).reshape(hw.shape)
                n += hw.size
                match += int((got == hw).sum())
            case.drop_inputs()
        report[rid] = dict(n=n, match=match, seconds=round(time.monotonic() - t, 1))
        if match != n or n != fixed[rid]["n"]:
            bad.append((rid, n, match, fixed[rid]["n"]))
    total = sum(r["n"] for r in report.values())
    print(json.dumps(dict(recordings=len(ids), outputs=total, seconds=round(time.monotonic() - t0, 1),
                          per_recording=report)))
    assert not bad, bad
