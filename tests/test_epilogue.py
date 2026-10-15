"""Epilogue of torch._scaled_mm: double rounding, zero sign, canonical NaN, fp8 saturation and
scale_result -- against RTX 4060 hardware outputs.

CPU data (no truth dir needed):
  tests/data/epilogue/epilogue_ties_rtx4060.*  constructed tie cases (verify-c1-claims-5); each is discriminating:
                                               single rounding or (acc*sa)*sb disagrees with the hardware somewhere
  tests/data/epilogue/epilogue_hw_rtx4060.*    zero-sign / underflow / zero-scale / NaN-Inf recordings extracted from
                                               the truth dataset (epilogue.c1_4_hw_underflow, epilogue.zero_nan_v2_a1_5,
                                               epilogue.c1_4_hw_zero_scale); they kill the fp32 "acc*alpha + 0" form
-m truth: every epilogue.* GEMM recording and misc.api_defaults_va1_3 through scaled_mm(backend="reference").
-m gpu:   CPU == CUDA epilogue on 1e6 (acc, alpha) pairs incl. subnormals, +-0, +-Inf and NaN, for every out dtype.

The helpers truth_loader / replay_gemm_output are shared with tests/test_plan.py and tests/test_api.py.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mmaemu.api import scaled_mm
from mmaemu.errors import ReferenceRejectsError, UnsupportedFeatureError
from mmaemu.model import epilogue
from mmaemu.testing import assert_bitwise_equal

DATA = Path(__file__).parent / "data" / "epilogue"
E4, E5 = torch.float8_e4m3fn, torch.float8_e5m2
TD = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16, "float8_e4m3fn": E4,
      "float8_e5m2": E5}
F8 = {"e4m3": E4, "e5m2": E5}
_IV = {4: torch.int32, 2: torch.int16, 1: torch.uint8}
_NV = {4: np.uint32, 2: np.uint16, 1: np.uint8}


def f32(bits) -> torch.Tensor:
    """fp32 scalar tensor from an int bit pattern."""
    return torch.from_numpy(np.array([int(bits)], dtype=np.uint32).view(np.int32)).view(torch.float32).reshape(())


def from_bits(arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    es = torch.empty((), dtype=dtype).element_size()
    a = np.ascontiguousarray(arr).view(_NV[es]).view({4: np.int32, 2: np.int16, 1: np.uint8}[es])
    return torch.from_numpy(a.copy()).view(dtype)


def bits_np(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu().contiguous()
    return t.view(_IV[t.element_size()]).numpy().view(_NV[t.element_size()])


# --------------------------------------------------------------------------------------------------------------------
# truth helpers (shared)
def truth_dir() -> Path:
    d = os.environ.get("MMAEMU_TRUTH_DIR")
    if not d:
        pytest.skip("MMAEMU_TRUTH_DIR not set (hardware ground-truth dataset)")
    return Path(d)


def truth_loader():
    d = truth_dir()
    sys.dont_write_bytecode = True              # never write into the truth dir
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    from tcemu_ada_truth import loader
    return loader


def replay_gemm_output(inputs: dict, out, *, backend: str = "reference", device: str = "cpu",
                       honour_call_variant: bool = True):
    """Replay one truth GEMM output through mmaemu.scaled_mm.

    Returns ("bits", ndarray) or ("error", class name) or ("na", reason). Stored operands that are a slice of the
    hardware call (params sub / hw_shape) are zero padded along N to a multiple of 16 and the result is sliced back:
    every output (m, n) depends only on row m of A and column n of B, under every plan."""
    p = out.params
    fa, fb = p.get("fmt_a", "e4m3"), p.get("fmt_b", "e4m3")
    od = TD[p["out_dtype"]]
    if p.get("scaling", "tensorwise") != "tensorwise":
        return ("na", "row-wise")
    if p.get("out_subset") or p.get("out_select"):
        return ("na", "output subset")
    a, b = inputs["a"], inputs["b"]
    if out.rows:
        a = a[out.rows[0]:out.rows[1]]
    M, K = a.shape
    N = b.shape[0]
    Np = -(-N // 16) * 16
    A = torch.from_numpy(np.ascontiguousarray(a)).view(F8[fa]).to(device)
    Bc = np.zeros((Np, K), np.uint8)
    Bc[:N] = b
    B = torch.from_numpy(Bc).view(F8[fb]).to(device)
    sa = f32(p["scale_a"]) if p.get("scale_a") is not None else torch.from_numpy(
        inputs["scale_a"].view(np.int32).copy()).view(torch.float32).reshape(())
    sb = f32(p["scale_b"]) if p.get("scale_b") is not None else torch.from_numpy(
        inputs["scale_b"].view(np.int32).copy()).view(torch.float32).reshape(())
    sa, sb = sa.to(device), sb.to(device)
    kw = {}
    variant = p.get("call_variant") if honour_call_variant else None
    if variant == "default_all":
        pass                                   # torch defaults: out_dtype = input dtype, use_fast_accum = False
    elif variant in ("default_fa", "default_fa_bf16"):
        kw["out_dtype"] = od                   # use_fast_accum omitted
    else:
        kw["out_dtype"] = od
        kw["use_fast_accum"] = bool(p.get("fast_accum", True))
    if p.get("scale_result") is not None:
        kw["scale_result"] = f32(p["scale_result"]).to(device)
    bname = p.get("bias_input") or "bias"
    if bname in inputs:
        bd = TD[p.get("bias_dtype", "bfloat16")]
        braw = from_bits(inputs[bname], bd)
        bias = torch.zeros(Np if braw.numel() == N else braw.numel(), dtype=bd)
        bias[:braw.numel()] = braw
        kw["bias"] = bias.to(device)
    sk = p.get("splitk")
    if isinstance(sk, dict) and int(sk.get("S", 1)) > 1:
        splitk = (int(sk["S"]), sk["scheme"])
    elif sk == "unknown":
        if not out.hw_error:
            return ("na", "split not recorded")
        splitk = 1                             # rejected calls never reach the split decision
    else:
        splitk = 1
    prefill = str(p.get("out_buffer_prefill", "none"))
    if prefill not in ("none", "fresh"):                 # out= buffer prefilled with this value (c1_4 / v01b)
        o = torch.full((M, Np), float(prefill), dtype=torch.float32)
        kw["out"] = o.to(od).to(device)
    try:
        y = scaled_mm(A, B.t(), sa, sb, splitk=splitk, backend=backend, **kw)
    except ReferenceRejectsError:
        return ("error", "ReferenceRejectsError")
    except UnsupportedFeatureError as e:
        return ("na", f"UnsupportedFeatureError: {str(e)[:60]}")
    return ("bits", bits_np(y[:, :N]))


# --------------------------------------------------------------------------------------------------------------------
# constructed tie cases (no truth needed)
TIES = json.loads((DATA / "epilogue_ties_rtx4060.json").read_text())
TZ = np.load(DATA / "epilogue_ties_rtx4060.npz", allow_pickle=False)
B_TIES = torch.from_numpy(TZ["b"].copy()).view(E4)
PE = {torch.bfloat16: (8, -126), torch.float16: (11, -14)}


def _round_sig(x: torch.Tensor, p: int, emin: int) -> torch.Tensor:
    """Single RNE rounding of float64 x to p significant bits (exponent floor emin), computed exactly."""
    e = (torch.frexp(x)[1] - 1).double()
    q = torch.exp2(torch.clamp(e - (p - 1), min=emin - (p - 1)))
    return torch.where(x == 0, x, torch.round(x / q) * q)


def _alt(kind: str, acc: torch.Tensor, sa, sb, od) -> torch.Tensor:
    if kind == "single":                                    # one rounding from the exact product
        return _round_sig(acc.double() * (sa * sb).double(), *PE[od]).float().to(od)
    if kind == "order":                                     # (acc * sa) * sb
        return ((acc * sa) * sb).to(od)
    if kind == "fp32":                                      # fp32 acc * alpha + 0 (wrong under underflow)
        return (acc * (sa * sb) + 0.0).to(od)
    raise ValueError(kind)


def _tie_inputs(c):
    a = torch.from_numpy(TZ[c["a"]].copy()).view(E4)
    hw = from_bits(TZ[c["out"]], TD[c["out_dtype"]])
    return a, hw, f32(c["scale_a_bits"]), f32(c["scale_b_bits"]), TD[c["out_dtype"]]


def test_ties_data_integrity():
    import hashlib
    assert TIES["format"] == "mmaemu-epilogue-ties/1" and len(TIES["cases"]) == 64
    assert hashlib.sha256((DATA / "epilogue_ties_rtx4060.npz").read_bytes()).hexdigest() == TIES["npz_sha256"]


@pytest.mark.parametrize("i", range(len(TIES["cases"])))
def test_epilogue_tie(i):
    c = TIES["cases"][i]
    a, hw, sa, sb, od = _tie_inputs(c)
    y = scaled_mm(a, B_TIES.t(), sa, sb, out_dtype=od, use_fast_accum=c["fast"], splitk=1, backend="reference")
    assert_bitwise_equal(y, hw, msg=f"tie case {i} ({c['kind']}, {c['out_dtype']}, fast={c['fast']})")


@pytest.mark.parametrize("kind", ["single", "order"])
def test_tie_alternatives_fail(kind):
    """Each alternative epilogue disagrees with the hardware on >= 1 recorded output (the data discriminates)."""
    from mmaemu.api import accumulate
    failing = 0
    for c in TIES["cases"]:
        if c["kind"] != kind:
            continue
        a, hw, sa, sb, od = _tie_inputs(c)
        acc = accumulate(a, B_TIES, use_fast_accum=c["fast"], splitk=1, backend="reference")
        alt = _alt(kind, acc, sa, sb, od)
        failing += int(not np.array_equal(bits_np(alt), bits_np(hw)))
    assert failing >= 1


# --------------------------------------------------------------------------------------------------------------------
# hardware zero-sign / underflow / NaN extract (no truth needed)
HWJ = json.loads((DATA / "epilogue_hw_rtx4060.json").read_text())
HWZ = np.load(DATA / "epilogue_hw_rtx4060.npz", allow_pickle=False)
HW_OUTS = [(ci, oi) for ci, c in enumerate(HWJ["cases"]) for oi in range(len(c["outputs"]))]


def _hw_case(ci, oi):
    c = HWJ["cases"][ci]
    o = c["outputs"][oi]
    a = torch.from_numpy(HWZ[c["a"]].copy()).view(F8[c["fmt_a"]])
    b = torch.from_numpy(HWZ[c["b"]].copy()).view(F8[c["fmt_b"]])
    return c, o, a, b


@pytest.mark.parametrize("ci,oi", HW_OUTS)
def test_epilogue_hw_extract(ci, oi):
    c, o, a, b = _hw_case(ci, oi)
    od = TD[o["out_dtype"]]
    sk = o["splitk"]
    y = scaled_mm(a, b.t(), f32(o["scale_a_bits"]), f32(o["scale_b_bits"]), out_dtype=od,
                  use_fast_accum=o["fast_accum"], splitk=(sk["S"], sk["scheme"]) if sk["S"] > 1 else 1,
                  backend="reference")
    assert_bitwise_equal(y, from_bits(HWZ[o["bits"]], od), msg=f"{c['truth_id']}/{c['case']}/{o['name']}")


def test_fp32_epilogue_alternative_fails_on_hardware_data():
    """The fp32 form acc*alpha + 0 (instead of the exact float64 product) disagrees with the hardware on >= 1 output
    (zero sign of underflowing products, a tcemu audit finding)."""
    from mmaemu.api import accumulate
    bad = 0
    for ci, oi in HW_OUTS:
        c, o, a, b = _hw_case(ci, oi)
        od = TD[o["out_dtype"]]
        sk = o["splitk"]
        acc = accumulate(a, b, use_fast_accum=o["fast_accum"], splitk=(sk["S"], sk["scheme"]) if sk["S"] > 1 else 1,
                         backend="reference")
        alt = epilogue.to_out_dtype(acc * (f32(o["scale_a_bits"]) * f32(o["scale_b_bits"])) + 0.0, od)
        bad += int(not np.array_equal(bits_np(alt), HWZ[o["bits"]]))
    assert bad >= 1


# --------------------------------------------------------------------------------------------------------------------
# unit rules
def _acc(vals):
    return torch.tensor(vals, dtype=torch.float32)


@pytest.mark.parametrize("od", [torch.float32, torch.bfloat16, torch.float16, E4, E5])
def test_exact_zero_product_is_plus_zero(od):
    acc = _acc([0.0, -0.0, 0.0, 1.0, -1.0])
    for sa in (-1.0, 0.0, -0.0):
        y = epilogue.apply(acc, torch.tensor(sa), torch.tensor(1.0), out_dtype=od)
        b = bits_np(y)
        zero_pos = [0, 1, 2] + ([3, 4] if sa in (0.0, -0.0) else [])
        assert all(int(b[i]) == 0 for i in zero_pos), (sa, od, b)


def test_underflow_keeps_sign():
    acc = _acc([-1.0, 1.0])
    y = epilogue.apply(acc, torch.tensor(2.0 ** -100), torch.tensor(2.0 ** -100), out_dtype=torch.float32)
    # alpha = 2^-200 underflows to +0 in fp32 -> exact zero product -> +0 for both signs
    assert bits_np(y).tolist() == [0, 0]
    y = epilogue.apply(acc, torch.tensor(2.0 ** -75), torch.tensor(2.0 ** -75), out_dtype=torch.float32)
    # alpha = 2^-150 rounds (tie to even) to 0 -> +0 as well; 2^-149 is the smallest subnormal
    assert bits_np(y).tolist() == [0, 0]
    y = epilogue.apply(_acc([-2.0 ** -30, 2.0 ** -30]), torch.tensor(2.0 ** -60), torch.tensor(2.0 ** -61),
                       out_dtype=torch.float32)
    # exact product -2^-151 / +2^-151 rounds to -0 / +0: the sign of an inexact zero follows the product
    assert bits_np(y).tolist() == [0x80000000, 0]


def test_inf_times_zero_is_canonical_nan():
    y = epilogue.apply(_acc([float("inf"), -float("inf"), 0.0]), torch.tensor(0.0), torch.tensor(1.0),
                       out_dtype=torch.float32)
    assert bits_np(y)[:2].tolist() == [0x7FFFFFFF, 0x7FFFFFFF]
    y = epilogue.apply(_acc([0.0]), torch.tensor(float("inf")), torch.tensor(1.0), out_dtype=torch.bfloat16)
    assert bits_np(y).tolist() == [0x7FFF]


@pytest.mark.parametrize("od,canon", [(torch.float32, 0x7FFFFFFF), (torch.bfloat16, 0x7FFF), (torch.float16, 0x7FFF),
                                      (E4, 0x7F), (E5, 0x7F)])
def test_canonical_nan(od, canon):
    nan_neg = torch.tensor([-1], dtype=torch.int32).view(torch.float32)      # 0xFFFFFFFF, a negative NaN
    acc = torch.cat([nan_neg, _acc([float("nan")])])
    y = epilogue.apply(acc, torch.tensor(-1.0), torch.tensor(1.0), out_dtype=od)
    assert [int(v) for v in bits_np(y)] == [canon, canon]


def test_fp8_saturation():
    acc = _acc([8192.0, -8192.0, 240.0, 480.0, 448.0, 464.0, float("inf"), -float("inf"), 1e30])
    y = epilogue.apply(acc, torch.tensor(1.0), torch.tensor(1.0), out_dtype=E4)
    assert [hex(v) for v in bits_np(y)] == ["0x7e", "0xfe", "0x77", "0x7e", "0x7e", "0x7e", "0x7e", "0xfe", "0x7e"]
    y = epilogue.apply(acc, torch.tensor(1.0), torch.tensor(1.0), out_dtype=E5)
    assert [hex(v) for v in bits_np(y)][:2] == ["0x70", "0xf0"] and [hex(v) for v in bits_np(y)][6:] == \
        ["0x7b", "0xfb", "0x7b"]


def test_scale_result_fp8_only_and_before_saturation():
    acc = _acc([300.0, 3.0, -1000.0])
    sr = torch.tensor(2.0)
    y = epilogue.apply(acc, torch.tensor(1.0), torch.tensor(1.0), out_dtype=E4, scale_result=sr)
    assert y.float().tolist() == [448.0, 6.0, -448.0]          # 600 saturates; scale applied before saturation
    y = epilogue.apply(_acc([1000.0]), torch.tensor(1.0), torch.tensor(1.0), out_dtype=E4,
                       scale_result=torch.tensor(0.25))
    assert y.float().tolist() == [256.0]                        # 1000 * 0.25 = 250 -> 256 (not saturated first)
    for od in (torch.float32, torch.bfloat16, torch.float16):
        y1 = epilogue.apply(acc, torch.tensor(1.0), torch.tensor(1.0), out_dtype=od, scale_result=sr)
        y0 = epilogue.apply(acc, torch.tensor(1.0), torch.tensor(1.0), out_dtype=od)
        assert_bitwise_equal(y1, y0)                            # ignored for non-fp8 outputs


def test_scale_result_is_a_second_fp32_multiply():
    g = torch.Generator().manual_seed(5)
    acc = (torch.randn(4096, generator=g) * 1000).float()
    sa, sb, sr = torch.tensor(0.0123), torch.tensor(0.731), torch.tensor(1.37)
    y = epilogue.apply(acc, sa, sb, out_dtype=E4, scale_result=sr)
    alpha = (sa.double() * sb.double()).float()
    y1 = (acc.double() * alpha.double()).float()            # RN32(acc * alpha)
    y2 = (y1.double() * sr.double()).float()                # RN32(y1 * scale_result)
    assert_bitwise_equal(y, y2.clamp(-448, 448).to(E4))


def test_out_buffer_contents_irrelevant():
    a = torch.full((16, 32), 2.0).to(E4)
    b = torch.full((16, 32), 3.0).to(E4)
    for fill in (float("nan"), -0.0, 1e30):
        out = torch.full((16, 16), fill, dtype=torch.bfloat16)
        y = scaled_mm(a, b.t(), torch.tensor(0.5), torch.tensor(-1.0), out_dtype=torch.bfloat16, out=out, splitk=1)
        assert y is out
        assert_bitwise_equal(out, torch.full((16, 16), -96.0, dtype=torch.bfloat16))


# --------------------------------------------------------------------------------------------------------------------
# truth: every epilogue.* GEMM recording and misc.api_defaults_va1_3 through scaled_mm(backend="reference")
def _epilogue_ids(loader, root):
    return [e["id"] for e in loader.list_recordings(root)
            if e["kind"] == "gemm" and (e["id"].startswith("epilogue.") or e["id"] == "misc.api_defaults_va1_3")]


@pytest.mark.truth
def test_truth_epilogue_recordings():
    loader = truth_loader()
    root = truth_dir()
    ids = _epilogue_ids(loader, root)
    assert len(ids) >= 13, ids
    totals = dict(outputs=0, values=0, errors=0, bias_rejects=0, e5m2_rejects=0, na=[], mismatches=[])
    for rid in ids:
        rec = loader.load(rid, root=root)
        for case in rec.cases:
            ins = case.inputs()
            for o in case.outputs:
                if not (o.stored or o.hw_error):
                    continue
                kind, val = replay_gemm_output(ins, o)
                if o.hw_error:
                    if kind != "error" or val != "ReferenceRejectsError":
                        totals["mismatches"].append((rid, case.id, o.name, "expected ReferenceRejectsError", kind, val))
                    else:
                        totals["errors"] += 1
                        p = o.params
                        totals["bias_rejects"] += int((p.get("bias_input") or "bias") in ins)
                        totals["e5m2_rejects"] += int(p["out_dtype"] == "float8_e5m2" and
                                                      p.get("fmt_a", "e4m3") == p.get("fmt_b", "e4m3") == "e4m3")
                    continue
                if kind == "na":
                    totals["na"].append((rid, case.id, o.name, val))
                    continue
                hw = o.bits()
                totals["outputs"] += 1
                totals["values"] += hw.size
                if kind != "bits" or not np.array_equal(val.reshape(hw.shape), hw):
                    totals["mismatches"].append((rid, case.id, o.name, kind,
                                                 val if kind != "bits" else int((val.reshape(hw.shape) != hw).sum())))
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in totals.items()}))
    assert not totals["mismatches"], totals["mismatches"][:10]
    assert totals["bias_rejects"] == 14
    assert totals["e5m2_rejects"] == totals["errors"] - 14 and totals["e5m2_rejects"] > 0
    # the only outputs outside the scope are the stored fused-bias outputs (bias raises UnsupportedFeatureError)
    assert all("UnsupportedFeatureError: bias" in r[3] for r in totals["na"]), totals["na"]
    assert len(totals["na"]) == 4, totals["na"]


# --------------------------------------------------------------------------------------------------------------------
# GPU: CPU == CUDA epilogue
def _pairs(n: int, g: torch.Generator):
    """(acc fp32 [n], alpha fp32 [n]) covering normals, subnormals, +-0, +-Inf, NaN and extreme exponents."""
    bits = torch.randint(-2 ** 31, 2 ** 31, (2, n), generator=g, dtype=torch.int64).to(torch.int32)
    x = bits.view(torch.float32)
    specials = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan"), 2.0 ** -149, -2.0 ** -149,
                             2.0 ** -126, 3.4e38, -3.4e38, 1.0, -1.0], dtype=torch.float32)
    idx = torch.randint(0, specials.numel(), (2, n // 8), generator=g)
    x[:, :n // 8] = specials[idx]
    sub = torch.randint(1, 1 << 23, (2, n // 8), generator=g, dtype=torch.int64).to(torch.int32).view(torch.float32)
    x[:, n // 8:n // 4] = sub
    return x[0].clone(), x[1].clone()


@pytest.mark.gpu
@pytest.mark.parametrize("od", [torch.float32, torch.bfloat16, torch.float16, E4, E5])
def test_epilogue_cpu_equals_cuda(od):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    g = torch.Generator().manual_seed(1234)
    acc, alpha = _pairs(1_000_000, g)
    one = torch.tensor(1.0)
    cpu = epilogue.apply(acc, alpha[:1].reshape(()), one, out_dtype=od)       # scalar alpha path
    gpu = epilogue.apply(acc.cuda(), alpha[:1].reshape(()).cuda(), one.cuda(), out_dtype=od)
    assert_bitwise_equal(gpu, cpu, msg="scalar alpha")
    # per-element alpha: the same arithmetic as apply() for each pair (fma(acc, alpha, +0) then conversion)
    for sr in (None, torch.tensor(0.75)):
        y_cpu = epilogue.to_out_dtype((acc.double() * alpha.double() + 0.0).float(), od, sr)
        y_gpu = epilogue.to_out_dtype((acc.cuda().double() * alpha.cuda().double() + 0.0).float(), od,
                                      None if sr is None else sr.cuda())
        assert_bitwise_equal(y_gpu, y_cpu, msg=f"pairs, scale_result={sr}")
