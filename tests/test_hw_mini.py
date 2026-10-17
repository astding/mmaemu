"""T1 (CPU, every push) and its GPU variant: replay the committed RTX 4060 hardware subset tests/data/hw_mini/
through mmaemu, bitwise (docs/testing.md).

- Every case compares integer bit patterns with mmaemu.testing.assert_bitwise_equal(nan="bits"): +0 != -0 and NaN
  payloads must match (mmaemu and the RTX 4060 both emit canonical NaN).
- Expected errors are matched by EXACT mmaemu class (ReferenceRejectsError: torch / cuBLAS rejected the hardware call;
  UnsupportedFeatureError: the hardware ran it, mmaemu v0.1 does not model it -- bias, row-wise scales, grad).
- The recorded split-K configuration is always passed explicitly (never splitk="auto").
- scaled_mm cases are called the way the hardware call was made (which keyword arguments were passed, bias
  positional, out= buffer prefilled), so wrong API defaults are observable.
- CPU: backend "reference" with torch.set_num_threads(4) (budget < 5 min). GPU (marker gpu): the same cases on
  cuda:0 for every backend that mmaemu reports available there; a backend that does not support a case's formats
  (BackendUnavailableError) skips that case, and every available backend must run at least one case.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hwmini import HwMini  # noqa: E402

import mmaemu  # noqa: E402
from mmaemu import errors as E  # noqa: E402
from mmaemu.testing import assert_bitwise_equal  # noqa: E402

TD = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16,
      "float8_e4m3fn": torch.float8_e4m3fn, "float8_e5m2": torch.float8_e5m2}
FP8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
IVIEW = {1: (np.uint8, torch.uint8), 2: (np.int16, torch.int16), 4: (np.int32, torch.int32)}
ERRORS = {n: getattr(E, n) for n in ("ReferenceRejectsError", "UnsupportedFeatureError", "SplitKUnknownError",
                                     "BackendUnavailableError", "MmaemuError")}
GPU_BACKENDS = ("reference", "cuda", "cuda:v1", "cuda:v4", "cuda:v9", "cuda:tc3")
ADA = "ada"  # the profile of the RTX 4060 recordings replayed here

_HM = None


def hm() -> HwMini:
    global _HM
    if _HM is None:
        _HM = HwMini()
    return _HM


def _cases():
    try:
        return hm().cases
    except Exception:  # noqa: BLE001 -- reported by test_index
        return []


@pytest.fixture(scope="module", autouse=True)
def _threads():
    n = int(os.environ.get("MMAEMU_T1_THREADS", "4"))
    old = torch.get_num_threads()
    torch.set_num_threads(n)
    yield
    torch.set_num_threads(old)


# ---------------------------------------------------------------------------------------------------------------
def bits_to_torch(a: np.ndarray, dtype: torch.dtype, device="cpu") -> torch.Tensor:
    a = np.ascontiguousarray(a)
    nv, tv = IVIEW[a.dtype.itemsize]
    return torch.from_numpy(a.view(nv).copy()).view(tv).view(dtype).to(device)


def f32(bits: int, device="cpu") -> torch.Tensor:
    """fp32 scalar tensor (shape ()) with the given IEEE bit pattern."""
    return bits_to_torch(np.array([bits], np.uint32), torch.float32, device).reshape(())


def splitk_arg(p):
    sk = p["splitk"]
    return 1 if int(sk["S"]) == 1 else (int(sk["S"]), sk["scheme"])


def _fill(v):
    return float(v) if isinstance(v, str) else v


def run_gemm(p, inputs, device, backend, label):
    """mmaemu output tensor (on CPU) of one gemm output with effective params p; raises whatever mmaemu raises."""
    od = TD[p["out_dtype"]]
    if p.get("zero_operands"):
        M, N, K = p["call_shape"]
        a = torch.zeros(M, K, dtype=torch.uint8, device=device).view(FP8[p["fmt_a"]])
        b = torch.zeros(N, K, dtype=torch.uint8, device=device).view(FP8[p["fmt_b"]])
    else:
        a = bits_to_torch(hm().array(inputs["a"]), FP8[p["fmt_a"]], device)
        b = bits_to_torch(hm().array(inputs["b"]), FP8[p["fmt_b"]], device)
        M, N = a.shape[0], b.shape[0]
    if p.get("scaling") == "rowwise":
        sa = torch.ones(M, 1, dtype=torch.float32, device=device)
        sb = torch.ones(1, N, dtype=torch.float32, device=device)
    else:
        sa, sb = f32(p["scale_a_bits"], device), f32(p["scale_b_bits"], device)
    sr = None if p.get("scale_result_bits") is None else f32(p["scale_result_bits"], device)
    if p["api"] == "accumulate":
        acc = mmaemu.accumulate(a, b, use_fast_accum=p["fast"], splitk=splitk_arg(p), backend=backend,
                               out_dtype_for_split=od, profile=ADA)
        y = mmaemu.model.epilogue.apply(acc, sa, sb, out_dtype=od, scale_result=sr)
        if p.get("out_select") == "diagonal":
            y = torch.diagonal(y).contiguous()
        return y.cpu()
    call = p.get("call", {})          # omitted keys = defaults (docs/testing.md "call spec")
    args = [a, b.t(), sa, sb]
    kw = dict(splitk=splitk_arg(p), backend=backend, profile=ADA)
    if call.get("bias"):
        bias = torch.zeros(N, dtype=TD[p.get("bias_dtype", "bfloat16")], device=device)
        if call["bias"] == "positional":
            args.append(bias)
        else:
            kw["bias"] = bias
    if sr is not None:                                   # passed iff the hardware call passed it
        kw["scale_result"] = sr
    if call.get("out_dtype", True):
        kw["out_dtype"] = od
    if call.get("use_fast_accum", True):
        kw["use_fast_accum"] = p["fast"]
    for name in call.get("requires_grad") or ():
        {"input": a, "mat2": b, "scale_a": sa, "scale_b": sb}[name].requires_grad_(True)
    out = None
    if call.get("out_fill") is not None:
        out = torch.full((M, N), _fill(call["out_fill"]), dtype=od, device=device)
        kw["out"] = out
    y = mmaemu.scaled_mm(*args, **kw)
    if out is not None:
        assert_bitwise_equal(out.cpu(), y.cpu(), msg=f"{label}: out= buffer differs from the returned tensor")
    return y.cpu()


# ---- exact calls of the R1 / R6 recordings (port of tools/record/r1_api_grid.py logical/physical/scale) ------
def _expand_call(call, diff=None):
    d = hm().index.get("call_defaults", {})
    full = {k: (dict(v) if isinstance(v, dict) else v) for k, v in d.items() if k != "mat"}
    for w in ("mat1", "mat2"):
        full[w] = dict(d.get("mat", {}))
    for k, v in call.items():
        if k in ("mat1", "mat2"):
            full[k].update(v)
        else:
            full[k] = v
    for k, v in (diff or {}).items():
        if k in ("mat1", "mat2"):
            full[k].update(v)
        else:
            full[k] = v
    return full


def _logical(bits: np.ndarray, dtype: str) -> torch.Tensor:
    if bits.dtype == np.uint16:                           # bf16 operands are stored as bf16 bits
        return torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16)
    return torch.from_numpy(np.ascontiguousarray(bits).copy()).view(getattr(torch, dtype))


def _physical(t, spec, which, device):
    rows, K = t.shape
    lay, pad, off = spec["layout"], spec["ld_pad"], spec["offset"]
    if lay == "std" and not pad and not off:
        x = t.contiguous().to(device)
        return x if which == "mat1" else x.t()
    if lay == "std":
        ld = K + pad
        buf = torch.zeros(off + rows * ld + 1, dtype=t.dtype)
        buf[off:off + rows * ld].view(rows, ld)[:, :K].copy_(t)
        phys = buf.to(device).as_strided((rows, K), (ld, 1), off)
        return phys if which == "mat1" else phys.t()
    ld = rows + pad
    buf = torch.zeros(off + K * ld + 1, dtype=t.dtype)
    buf[off:off + K * ld].view(K, ld)[:, :rows].copy_(t.t())
    phys = buf.to(device).as_strided((K, rows), (ld, 1), off)
    return phys.t() if which == "mat1" else phys


def _scale(spec, M, N, which, device):
    sh = spec["shape"]
    if spec["value"] == "rowwise":
        n = M if which == "a" else N
        vals = torch.tensor([0.25 + 0.0625 * ((i * 7) % 16) for i in range(n)], dtype=torch.float64)
        return vals.reshape(sh).to(getattr(torch, spec["dtype"])).to(device)
    return torch.full(sh, float(spec["value"]), dtype=torch.float64).to(getattr(torch, spec["dtype"])).to(device)


def run_call(case, o, device, backend):
    c = _expand_call(case["call"], o.get("call_diff"))
    M, N = c["M"], c["N"]
    mat1 = _physical(_logical(hm().array(case["inputs"]["a"]), c["mat1"]["dtype"]), c["mat1"], "mat1", device)
    mat2 = _physical(_logical(hm().array(case["inputs"]["b"]), c["mat2"]["dtype"]), c["mat2"], "mat2", device)
    sa = _scale(c["scale_a"], M, N, "a", device)
    sb = _scale(c["scale_b"], M, N, "b", device)
    kw = dict(use_fast_accum=o["fast"], backend=backend, profile=ADA,
              splitk=1 if not o.get("splitk") else (int(o["splitk"][0]), o["splitk"][1]))
    if o.get("out_dtype_passed", True):
        kw["out_dtype"] = getattr(torch, o["out_dtype"])
    bias = None
    if c["bias"]:
        bias = (torch.arange(c["bias"]["len"], dtype=torch.float64) * 0.125 - 1.0).to(
            getattr(torch, c["bias"]["dtype"])).to(device)
    if c["scale_result"]:
        kw["scale_result"] = _scale(c["scale_result"], 0, 0, "r", device)
    outb = None
    if c["out"]:
        shp = c["out"]["shape"]
        base = torch.empty(shp if c["out"]["contiguous"] else list(reversed(shp)),
                           dtype=getattr(torch, c["out"]["dtype"]), device=device)
        base.view(torch.uint8).fill_(0x7F)
        outb = base if c["out"]["contiguous"] else base.t()
    g = c["grad"]
    if g:
        for nm in g["requires_grad"]:
            {"scale_a": sa, "scale_b": sb, "mat1": mat1}[nm].requires_grad_(True)
    ctx = torch.no_grad() if (g and g["mode"] == "no_grad") else torch.enable_grad()
    with ctx:
        if c["positional_bias"]:
            y = mmaemu.scaled_mm(mat1, mat2, sa, sb, bias, **kw)
        elif outb is not None:
            y = mmaemu.scaled_mm(mat1, mat2, scale_a=sa, scale_b=sb, bias=bias, out=outb, **kw)
        else:
            y = mmaemu.scaled_mm(mat1, mat2, scale_a=sa, scale_b=sb, bias=bias, **kw)
    return y.detach().cpu().contiguous()


def run_mma(case, device):
    p = case["params"]
    a = hm().array(case["inputs"]["a"])
    b = hm().array(case["inputs"]["b"])
    c = hm().array(case["inputs"]["c"])
    K = a.shape[-1]
    if p["layout"] == "tile":          # D[t,i,j] = chain_k(a[t,i,k] * b[t,j,k]) from c[t,i,j]
        T, mi, nj = a.shape[0], a.shape[1], b.shape[1]
        a = np.broadcast_to(a[:, :, None, :], (T, mi, nj, K)).reshape(-1, K)
        b = np.broadcast_to(b[:, None, :, :], (T, mi, nj, K)).reshape(-1, K)
        c = c.reshape(-1)
    at = bits_to_torch(a, FP8[p["fmt_a"]], device)
    bt = bits_to_torch(b, FP8[p["fmt_b"]], device)
    ct = bits_to_torch(c, torch.float32, device)
    d = mmaemu.model.dot(at, bt, ct, preset=p.get("preset", "ada"))
    return d.cpu().reshape(case["expect"]["shape"])


def _expect_error(name, fn, label):
    cls = ERRORS.get(name) or {"TypeError": TypeError}[name]
    with pytest.raises(cls) as ei:
        fn()
    assert type(ei.value) is cls, f"{label}: expected exactly {cls.__name__}, got {type(ei.value).__name__}: {ei.value}"


def outputs_of(case):
    """(label, effective params, expect) of every output of a case (single-expect or `outputs` list)."""
    if "outputs" not in case:
        return [(case["id"], case.get("params", {}), case["expect"])]
    return [(f"{case['id']}/{o['name']}", {**case.get("params", {}), **o.get("params", {})}, o["expect"], o)
            for o in case["outputs"]]


def check_case(case, device="cpu", backend="reference"):
    if case["kind"] == "mma":
        want = bits_to_torch(hm().expected(case), torch.float32)
        assert_bitwise_equal(run_mma(case, device), want, nan="bits", msg=case["id"])
        return
    failures = []
    for item in outputs_of(case):
        label, p, exp = item[0], item[1], item[2]
        o = item[3] if len(item) > 3 else None
        try:
            if case["kind"] == "gemm_call":
                if isinstance(exp, str):
                    _expect_error(exp, lambda: run_call(case, o, device, backend), label)
                else:
                    want = bits_to_torch(hm().bits(exp, label), TD[o["out_dtype"]])
                    assert_bitwise_equal(run_call(case, o, device, backend), want.reshape(exp["shape"]), msg=label)
            elif "error" in exp:
                _expect_error(exp["error"], lambda: run_gemm(p, case.get("inputs", {}), device, backend, label), label)
            else:
                want = bits_to_torch(hm().bits(exp, label), TD[exp["dtype"]])
                assert_bitwise_equal(run_gemm(p, case["inputs"], device, backend, label), want, nan="bits",
                                     msg=label)
        except E.BackendUnavailableError:
            raise                                       # the GPU variant counts it as "not supported here"
        except (AssertionError, pytest.fail.Exception) as e:
            failures.append(str(e).splitlines()[0][:400])
        except Exception as e:  # noqa: BLE001 -- any unexpected exception is a failure of this output
            failures.append(f"{label}: unexpected {type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}")
    assert not failures, f"{len(failures)} output(s) failed:\n" + "\n".join(failures[:20])


# ---------------------------------------------------------------------------------------------------------------
def test_index():
    h = hm()
    assert h.index["format"] == "mmaemu-hwmini/1"
    assert h.cases, "hw_mini has no cases"
    ids = [c["id"] for c in h.cases]
    assert len(ids) == len(set(ids))
    for name in h.index["shards"]:
        h._shard(name)                                   # sha256 of every shard
    total = sum(p.stat().st_size for p in h.dir.iterdir() if p.is_file())
    assert total <= 20 * 1024 * 1024, f"hw_mini is {total} bytes > 20 MiB"
    cov = (h.dir / "coverage.json")
    assert cov.exists()
    import json
    assert json.loads(cov.read_text())["missing_in_scope"] == []
    truth = os.environ.get("MMAEMU_TRUTH_DIR")
    if truth:                                            # pinned truth manifest
        import hashlib
        got = hashlib.sha256((Path(truth) / "manifest.json").read_bytes()).hexdigest()
        assert got == h.truth_manifest_sha256, "hw_mini was exported from a different truth manifest"


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["id"])
def test_case(case):
    check_case(case)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_gpu_backend(backend):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    dev = torch.device("cuda", 0)
    st = mmaemu.backends_status(dev)
    if backend not in st:
        pytest.skip(f"backend {backend!r} not reported by mmaemu.backends_status() on this device")
    if not st[backend].available:
        pytest.skip(f"backend {backend!r} unavailable on {torch.cuda.get_device_name(dev)}: {st[backend].reason}")
    ran, routed, bad = 0, 0, []
    for case in hm().cases:
        if case["kind"] == "mma" and backend != "reference":
            continue                                     # instruction-level model: device-independent check once
        try:
            check_case(case, dev, backend)
            ran += 1
        except E.BackendUnavailableError:
            routed += 1
        except AssertionError as e:
            bad.append(str(e).splitlines()[0][:300])
    print(f"backend {backend}: {ran} cases bitwise, {routed} not supported by this backend, {len(bad)} failed")
    assert not bad, f"{len(bad)} cases failed on {backend}:\n" + "\n".join(bad[:30])
    assert ran > 0, f"backend {backend} ran no case"
