"""R1: torch._scaled_mm API parity grid on the reference GPU, and R6: zero-dimension GEMMs.

Every call records raise / no-raise (error class + first message line), the returned tensor's dtype, shape,
strides and requires_grad, its bits, and the per-call cuBLASLt config (CUBLASLT_LOG_LEVEL=2, logger callback).
Torch default workspace (CUBLASLT_WORKSPACE_SIZE unset; the logged workSpaceSizeInBytes is kept).

R1 axes (the M x K x N grid is crossed with out dtype x format pair x fast/slow; every other axis is varied one at a
time around base shapes, as in the spec's preliminary rules):
  grid     M {0,1,15,16,17} x K {0,8,15,16,17,24,32} x N {0,8,15,16,17,24,40} x out {fp32,bf16,fp16,e4m3,e5m2}
           x pair {e4m3xe4m3, e4m3xe5m2, e5m2xe4m3, e5m2xe5m2} x fast/slow, real scales (0.3, 1.7)
  dtypes   mat1 x mat2 in {e4m3fn, e5m2, e4m3fnuz, e5m2fnuz, bf16} x out {None, fp32, bf16, fp16, e4m3fn, e5m2,
           e4m3fnuz, e5m2fnuz}
  scales   dtype {fp32, bf16, fp64} x shape {(), (1,), (1,1), row-wise (M,1)/(1,N), wrong numel (2,)} per operand,
           and both row-wise
  layout   mat1 col-major / mat2 row-major / both; padded ld (K+1, K+16) per operand
  offset   storage offsets {1,2,4,8,16,32} B per operand
  out=     correct (prefilled with NaN bits), wrong dtype, wrong shape, non-contiguous, x out dtype
  bias     {bf16, fp16, fp32, wrong length, positional} x out dtype
  sresult  scale_result {1.0, 0.5, 3.0, 2^-10} x out dtype
  grad     requires_grad on scale_a / mat1 under grad mode and under no_grad (+ backward attempt)
R6: zero-dimension GEMMs M/N/K in {0, base} (every combination with a zero) x out dtype x format pair x fast/slow
with real scales (and scale_result for fp8 outputs), plus K = 0 with scale_a in {+1, -1, +0, -0, NaN, Inf}.

usage:
  python tools/record/r1_api_grid.py calls                   CPU: write the call list (stage dir)
  python tools/record/r1_api_grid.py run                     GPU (under gpu_run.py): chunked subprocesses with a
                                                             canary after every call; restart on a poisoned context
  python tools/record/r1_api_grid.py write                   CPU: recordings r1.api_grid and r6.zero_dims
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import live  # noqa: E402

STAGE = common.STAGE / "r1"
OUTS = ["float32", "bfloat16", "float16", "float8_e4m3fn", "float8_e5m2"]
PAIRS = [("float8_e4m3fn", "float8_e4m3fn"), ("float8_e4m3fn", "float8_e5m2"), ("float8_e5m2", "float8_e4m3fn"),
         ("float8_e5m2", "float8_e5m2")]
SA, SB = 0.3, 1.7
SHORT = {"float32": "f32", "bfloat16": "bf16", "float16": "f16", "float8_e4m3fn": "e4m3", "float8_e5m2": "e5m2",
         "float8_e4m3fnuz": "e4m3fnuz", "float8_e5m2fnuz": "e5m2fnuz", None: "none", "float64": "f64"}


def mat(dtype, layout="std", ld_pad=0, offset=0):
    return dict(dtype=dtype, layout=layout, ld_pad=ld_pad, offset=offset)


def call(case, name, M, N, K, m1, m2, *, out_dtype, fast, seed, sa=None, sb=None, bias=None, scale_result=None,
         out=None, grad=None, positional_bias=False, rec="r1"):
    return dict(rec=rec, case=case, name=name, M=M, N=N, K=K, mat1=m1, mat2=m2, out_dtype=out_dtype, fast=fast,
                seed=seed, scale_a=sa or dict(dtype="float32", shape=[], value=SA),
                scale_b=sb or dict(dtype="float32", shape=[], value=SB), bias=bias, scale_result=scale_result,
                out=out, grad=grad, positional_bias=positional_bias)


def build_calls() -> list[dict]:
    C = []
    seed = 100
    # -- grid ----------------------------------------------------------------------------------------------------
    for M, K, N in itertools.product([0, 1, 15, 16, 17], [0, 8, 15, 16, 17, 24, 32], [0, 8, 15, 16, 17, 24, 40]):
        for fa, fb in PAIRS:
            seed += 1
            case = f"grid_M{M}_K{K}_N{N}_{SHORT[fa]}x{SHORT[fb]}"
            for od, fast in itertools.product(OUTS, (True, False)):
                C.append(call(case, f"{SHORT[od]}_{'fast' if fast else 'slow'}", M, N, K, mat(fa), mat(fb),
                              out_dtype=od, fast=fast, seed=seed))
    # -- dtypes --------------------------------------------------------------------------------------------------
    DT = ["float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz", "bfloat16"]
    ODT = [None, "float32", "bfloat16", "float16", "float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz",
           "float8_e5m2fnuz"]
    for d1, d2 in itertools.product(DT, DT):
        seed += 1
        case = f"dtype_{SHORT[d1]}x{SHORT[d2]}"
        for od in ODT:
            C.append(call(case, f"out_{SHORT[od]}", 16, 16, 32, mat(d1), mat(d2), out_dtype=od, fast=True,
                          seed=seed))
    # -- scales --------------------------------------------------------------------------------------------------
    M, N, K = 16, 32, 64
    shapes = {"scalar": [], "1": [1], "1x1": [1, 1], "rowwise": None, "wrong2": [2]}
    for which in ("a", "b"):
        for sd in ("float32", "bfloat16", "float64"):
            for sn, sh in shapes.items():
                if sh is None:
                    sh = [M, 1] if which == "a" else [1, N]
                spec = dict(dtype=sd, shape=sh, value=0.375 if which == "a" else 1.5)
                for od in ("float32", "bfloat16"):
                    seed += 1
                    kw = {"sa": spec} if which == "a" else {"sb": spec}
                    C.append(call(f"scale_{which}_{SHORT[sd]}_{sn}_{SHORT[od]}", "out", M, N, K,
                                  mat("float8_e4m3fn"), mat("float8_e4m3fn"), out_dtype=od, fast=True, seed=seed,
                                  **kw))
    for od in OUTS:
        for fast in (True, False):
            seed += 1
            C.append(call(f"scale_rowwise_both_{SHORT[od]}_{'fast' if fast else 'slow'}", "out", M, N, K,
                          mat("float8_e4m3fn"), mat("float8_e4m3fn"), out_dtype=od, fast=fast, seed=seed,
                          sa=dict(dtype="float32", shape=[M, 1], value="rowwise"),
                          sb=dict(dtype="float32", shape=[1, N], value="rowwise")))
    # -- layouts, padded ld, storage offsets ---------------------------------------------------------------------
    for M, N, K in ((16, 32, 64), (17, 40, 48)):
        variants = {
            "mat1_col": (mat("float8_e4m3fn", "col"), mat("float8_e4m3fn")),
            "mat2_row": (mat("float8_e4m3fn"), mat("float8_e4m3fn", "row")),
            "both_swapped": (mat("float8_e4m3fn", "col"), mat("float8_e4m3fn", "row")),
            "mat1_ld+1": (mat("float8_e4m3fn", ld_pad=1), mat("float8_e4m3fn")),
            "mat1_ld+16": (mat("float8_e4m3fn", ld_pad=16), mat("float8_e4m3fn")),
            "mat2_ld+1": (mat("float8_e4m3fn"), mat("float8_e4m3fn", ld_pad=1)),
            "mat2_ld+16": (mat("float8_e4m3fn"), mat("float8_e4m3fn", ld_pad=16)),
        }
        for off in (1, 2, 4, 8, 16, 32):
            variants[f"mat1_off{off}"] = (mat("float8_e4m3fn", offset=off), mat("float8_e4m3fn"))
            variants[f"mat2_off{off}"] = (mat("float8_e4m3fn"), mat("float8_e4m3fn", offset=off))
            variants[f"both_off{off}"] = (mat("float8_e4m3fn", offset=off), mat("float8_e4m3fn", offset=off))
        for vn, (m1, m2) in variants.items():
            for od in ("float32", "bfloat16", "float8_e4m3fn"):
                seed += 1
                C.append(call(f"layout_{M}x{N}x{K}_{vn}_{SHORT[od]}", "out", M, N, K, m1, m2, out_dtype=od,
                              fast=True, seed=seed))
    # -- out= ----------------------------------------------------------------------------------------------------
    M, N, K = 16, 32, 64
    for od in OUTS:
        fa, fb = ("float8_e4m3fn", "float8_e5m2") if od == "float8_e5m2" else ("float8_e4m3fn", "float8_e4m3fn")
        other = "float32" if od != "float32" else "bfloat16"
        for kind, ospec in (("correct", dict(dtype=od, shape=[M, N], contiguous=True)),
                            ("wrong_dtype", dict(dtype=other, shape=[M, N], contiguous=True)),
                            ("wrong_shape", dict(dtype=od, shape=[M, N + 1], contiguous=True)),
                            ("wrong_shape_empty", dict(dtype=od, shape=[0], contiguous=True)),
                            ("noncontig", dict(dtype=od, shape=[M, N], contiguous=False))):
            seed += 1
            C.append(call(f"out_{kind}_{SHORT[od]}", "out", M, N, K, mat(fa), mat(fb), out_dtype=od, fast=True,
                          seed=seed, out=ospec))
    # -- bias ----------------------------------------------------------------------------------------------------
    for od in OUTS:
        fa, fb = ("float8_e4m3fn", "float8_e5m2") if od == "float8_e5m2" else ("float8_e4m3fn", "float8_e4m3fn")
        for bn, bspec in (("bf16", dict(dtype="bfloat16", len=N)), ("f16", dict(dtype="float16", len=N)),
                          ("f32", dict(dtype="float32", len=N)), ("bf16_len+1", dict(dtype="bfloat16", len=N + 1)),
                          ("f16_len+1", dict(dtype="float16", len=N + 1))):
            for fast in (True, False):
                seed += 1
                C.append(call(f"bias_{bn}_{SHORT[od]}_{'fast' if fast else 'slow'}", "out", M, N, K, mat(fa),
                              mat(fb), out_dtype=od, fast=fast, seed=seed, bias=bspec))
        seed += 1
        bd = od if od in ("bfloat16", "float16") else "bfloat16"
        C.append(call(f"bias_positional_{SHORT[bd]}_{SHORT[od]}", "out", M, N, K, mat(fa), mat(fb), out_dtype=od,
                      fast=True, seed=seed, bias=dict(dtype=bd, len=N), positional_bias=True))
    # -- scale_result --------------------------------------------------------------------------------------------
    for od in OUTS:
        fa, fb = ("float8_e4m3fn", "float8_e5m2") if od == "float8_e5m2" else ("float8_e4m3fn", "float8_e4m3fn")
        for sv in (1.0, 0.5, 3.0, 2.0 ** -10):
            for fast in (True, False):
                seed += 1
                C.append(call(f"sresult_{sv:g}_{SHORT[od]}_{'fast' if fast else 'slow'}", "out", M, N, K, mat(fa),
                              mat(fb), out_dtype=od, fast=fast, seed=seed,
                              scale_result=dict(dtype="float32", shape=[], value=sv)))
    # -- requires_grad -------------------------------------------------------------------------------------------
    for mode in ("enabled", "no_grad"):
        for rg in (["scale_a"], ["scale_a", "scale_b"], ["mat1"]):
            for od in ("float32", "bfloat16"):
                seed += 1
                C.append(call(f"grad_{mode}_{'+'.join(rg)}_{SHORT[od]}", "out", M, N, K, mat("float8_e4m3fn"),
                              mat("float8_e4m3fn"), out_dtype=od, fast=True, seed=seed,
                              grad=dict(mode=mode, requires_grad=rg)))
    # -- R6 zero dims --------------------------------------------------------------------------------------------
    for (M, N, K) in ((0, 16, 32), (16, 0, 32), (16, 16, 0), (0, 0, 32), (0, 16, 0), (16, 0, 0), (0, 0, 0)):
        for fa, fb in PAIRS:
            seed += 1
            case = f"zero_M{M}_N{N}_K{K}_{SHORT[fa]}x{SHORT[fb]}"
            for od, fast in itertools.product(OUTS, (True, False)):
                sr = dict(dtype="float32", shape=[], value=0.5) if od.startswith("float8") else None
                C.append(call(case, f"{SHORT[od]}_{'fast' if fast else 'slow'}", M, N, K, mat(fa), mat(fb),
                              out_dtype=od, fast=fast, seed=seed, scale_result=sr, rec="r6"))
    for sv, sn in ((1.0, "p1"), (-1.0, "m1"), (0.0, "p0"), (-0.0, "m0"), ("nan", "nan"), ("inf", "inf"),
                   ("-inf", "minf")):
        for od in OUTS:
            fa, fb = ("float8_e4m3fn", "float8_e5m2") if od == "float8_e5m2" else ("float8_e4m3fn", "float8_e4m3fn")
            for fast in (True, False):
                seed += 1
                C.append(call(f"zeroK_scale_{sn}_{SHORT[od]}_{'fast' if fast else 'slow'}", "out", 16, 16, 0,
                              mat(fa), mat(fb), out_dtype=od, fast=fast, seed=seed,
                              sa=dict(dtype="float32", shape=[], value=sv), rec="r6"))
    for i, c in enumerate(C):
        c["idx"] = i
    return C


# ---- tensor construction (worker side; the same logical values are regenerated on CPU by the writer) --------------
def fmt_codes(dtype):
    return "e5m2" if "e5m2" in dtype else "e4m3"


def logical(c):
    """CPU logical operands: a [M,K], b [N,K] as torch tensors of the requested dtypes (codes from the hash
    generator; bf16 operands carry the e4m3 values)."""
    import torch
    job = dict(M=c["M"], N=c["N"], K=c["K"], fa=fmt_codes(c["mat1"]["dtype"]), fb=fmt_codes(c["mat2"]["dtype"]),
               data=dict(kind="randn40", seed=c["seed"]))
    a8, b8 = live.make_inputs(job)

    def as_dtype(codes, dtype, fmt):
        if dtype.startswith("float8"):
            return codes.view(getattr(torch, dtype))
        return codes.view(getattr(torch, live.FP8[fmt])).to(getattr(torch, dtype))
    return as_dtype(a8, c["mat1"]["dtype"], job["fa"]), as_dtype(b8, c["mat2"]["dtype"], job["fb"])


def physical(t, spec, which, dev):
    """t: logical [rows, K] CPU tensor -> the operand passed to torch (mat1 = t, mat2 = t.t()) with the requested
    layout / padded ld / storage offset on dev."""
    import torch
    rows, K = t.shape
    lay, pad, off = spec["layout"], spec["ld_pad"], spec["offset"]
    if lay == "std" and not pad and not off:            # what a user passes: contiguous a, b.t()
        x = t.contiguous().to(dev)
        return x if which == "mat1" else x.t()
    if lay == "std":
        # mat1 std = row-major [M,K]; mat2 std = column-major [K,N], i.e. row-major storage of b [N,K], then .t()
        ld = K + pad
        buf = torch.zeros(off + rows * ld + 1, dtype=t.dtype)
        buf[off:off + rows * ld].view(rows, ld)[:, :K].copy_(t)
        phys = buf.to(dev).as_strided((rows, K), (ld, 1), off)
        return phys if which == "mat1" else phys.t()
    # mat1 "col" = column-major [M,K]; mat2 "row" = row-major [K,N]: row-major storage of t.t() [K, rows]
    ld = rows + pad
    buf = torch.zeros(off + K * ld + 1, dtype=t.dtype)
    buf[off:off + K * ld].view(K, ld)[:, :rows].copy_(t.t())
    phys = buf.to(dev).as_strided((K, rows), (ld, 1), off)
    return phys.t() if which == "mat1" else phys


def scale_tensor(spec, M, N, which, dev):
    import torch
    sh = spec["shape"]
    if spec["value"] == "rowwise":
        n = M if which == "a" else N
        vals = torch.tensor([0.25 + 0.0625 * ((i * 7) % 16) for i in range(n)], dtype=torch.float64)
        return vals.reshape(sh).to(getattr(torch, spec["dtype"])).to(dev)
    v = float(spec["value"])
    return torch.full(sh, v, dtype=torch.float64).to(getattr(torch, spec["dtype"])).to(dev)


def worker(start: int, end: int, out: str):
    import torch
    dev = torch.device("cuda:0")
    cap = common.LtCapture()
    calls = json.loads((STAGE / "calls.json").read_text())[start:end]
    res, arrays = [], {}
    one = torch.ones((), device=dev)
    canary_bits = None
    stopped = None
    for c in calls:
        r = dict(idx=c["idx"], setup_error=None, error=None)
        try:
            a, b = logical(c)
            mat1 = physical(a, c["mat1"], "mat1", dev)
            mat2 = physical(b, c["mat2"], "mat2", dev)
            sa = scale_tensor(c["scale_a"], c["M"], c["N"], "a", dev)
            sb = scale_tensor(c["scale_b"], c["M"], c["N"], "b", dev)
            kw = dict(scale_a=sa, scale_b=sb, use_fast_accum=c["fast"])
            if c["out_dtype"] is not None:
                kw["out_dtype"] = getattr(torch, c["out_dtype"])
            bias = None
            if c["bias"]:
                bias = (torch.arange(c["bias"]["len"], dtype=torch.float64) * 0.125 - 1.0).to(
                    getattr(torch, c["bias"]["dtype"])).to(dev)
                r["bias_bits"] = common.tbits(bias).tolist()
            if c["scale_result"]:
                kw["scale_result"] = scale_tensor(c["scale_result"], 0, 0, "r", dev)
            outb = None
            if c["out"]:
                od = getattr(torch, c["out"]["dtype"])
                shp = c["out"]["shape"]
                base = torch.empty(shp if c["out"]["contiguous"] else list(reversed(shp)), dtype=od, device=dev)
                base.view(torch.uint8).fill_(0x7F)          # prefill: every byte 0x7F (NaN for fp16/fp8 outputs)
                outb = base if c["out"]["contiguous"] else base.t()
                r["out_before"] = dict(shape=list(outb.shape), stride=list(outb.stride()))
            g = c["grad"]
            if g:
                for nm in g["requires_grad"]:
                    {"scale_a": sa, "scale_b": sb, "mat1": mat1}[nm].requires_grad_(True)
            r["mat_meta"] = dict(mat1=dict(shape=list(mat1.shape), stride=list(mat1.stride()),
                                           offset=mat1.storage_offset()),
                                 mat2=dict(shape=list(mat2.shape), stride=list(mat2.stride()),
                                           offset=mat2.storage_offset()))
        except Exception as e:  # noqa: BLE001
            r["setup_error"] = common.err_text(e)
            res.append(r)
            continue
        torch.cuda.synchronize()
        mk = cap.mark()
        y = None
        try:
            ctx = torch.no_grad() if (g and g["mode"] == "no_grad") else torch.enable_grad()
            with ctx:
                if c["positional_bias"]:
                    y = torch._scaled_mm(mat1, mat2, sa, sb, bias, **{k: v for k, v in kw.items()
                                                                     if k not in ("scale_a", "scale_b")})
                elif outb is not None:
                    y = torch._scaled_mm(mat1, mat2, bias=bias, out=outb, **kw)
                else:
                    y = torch._scaled_mm(mat1, mat2, bias=bias, **kw)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            r["error"] = common.err_text(e)
        calls_lt = cap.matmuls_since(mk)
        r["n_lt_calls"] = len(calls_lt)
        r["lt"] = calls_lt[0] if calls_lt else None
        r["lt_messages"] = cap.other_since(mk)[:4]
        if y is not None:
            r["result"] = dict(dtype=str(y.dtype).replace("torch.", ""), shape=list(y.shape), stride=list(y.stride()),
                               requires_grad=bool(y.requires_grad),
                               grad_fn=type(y.grad_fn).__name__ if y.grad_fn is not None else None,
                               is_out=bool(outb is not None and y.data_ptr() == outb.data_ptr()))
            arrays[f"{c['idx']}"] = common.tbits(y.detach())
            if g and y.requires_grad:
                try:
                    y.float().sum().backward()
                    r["backward"] = "ok"
                except Exception as e:  # noqa: BLE001
                    r["backward"] = common.err_text(e)
        # canary
        try:
            cy = torch._scaled_mm(torch.ones(16, 32, device=dev).to(torch.float8_e4m3fn),
                                  torch.ones(16, 32, device=dev).to(torch.float8_e4m3fn).t(), scale_a=one,
                                  scale_b=one, out_dtype=torch.float32)
            torch.cuda.synchronize()
            cb = common.tbits(cy)
            canary_bits = cb if canary_bits is None else canary_bits
            r["canary_ok"] = bool(np.array_equal(cb, canary_bits)) and bool((cb == 0x42000000).all())
        except Exception as e:  # noqa: BLE001
            r["canary_ok"] = False
            r["canary_error"] = common.err_text(e)
        res.append(r)
        if not r["canary_ok"]:
            stopped = c["idx"]
            break
    np.savez(out, **arrays)
    Path(out).with_suffix(".json").write_text(json.dumps(dict(results=res, stopped_after=stopped,
                                                              torch=torch.__version__), indent=0))
    print(f"worker [{start}, {end}): {len(res)} calls, errors={sum(r['error'] is not None for r in res)}, "
          f"setup_errors={sum(r['setup_error'] is not None for r in res)}, stopped_after={stopped}", flush=True)
    sys.exit(3 if stopped is not None else 0)


def cmd_calls(_a):
    STAGE.mkdir(parents=True, exist_ok=True)
    C = build_calls()
    (STAGE / "calls.json").write_text(json.dumps(C))
    import collections
    print(len(C), "calls;", dict(collections.Counter(c["case"].split("_")[0] for c in C)))


def cmd_run(a):
    """Run calls [start, end) in chunked worker subprocesses (one locked GPU run per invocation; several invocations
    with disjoint ranges fill the same run dir)."""
    run = STAGE / "run"
    run.mkdir(parents=True, exist_ok=True)
    if not (run / "hardware.json").exists():
        (run / "hardware.json").write_text(json.dumps(common.hardware(), indent=1))
    n = len(json.loads((STAGE / "calls.json").read_text()))
    start, stop, chunk = a.start, min(n, a.end if a.end is not None else n), a.chunk
    t0 = time.monotonic()
    nproc = 0
    while start < stop:
        end = min(stop, start + chunk)
        outp = run / f"{a.prefix}{start:05d}.npz"
        rc = common.run_worker([__file__, "worker", "--start", str(start), "--end", str(end), "--out", str(outp)],
                               None, run / f"lt_{a.prefix}{start:05d}.log")
        meta = json.loads(outp.with_suffix(".json").read_text())
        nproc += 1
        if rc == 3:
            start = meta["stopped_after"] + 1     # poisoned context: restart after the offending call
        elif rc != 0:
            print("worker failed", rc)
            sys.exit(1)
        else:
            start = end
    print(f"R1/R6 calls [{a.start}, {stop}) in {nproc} worker processes, {time.monotonic() - t0:.1f}s")


def cmd_worker(a):
    worker(a.start, a.end, a.out)


# ---- writer -------------------------------------------------------------------------------------------------------
def cmd_write(_a):
    F = common.truth_format()
    C = json.loads((STAGE / "calls.json").read_text())
    run = STAGE / "run"
    hw = json.loads((run / "hardware.json").read_text())
    res, arrs, overridden = {}, {}, []
    for p in sorted(run.glob("part*.json")):
        m = json.loads(p.read_text())
        z = np.load(p.with_suffix(".npz"), allow_pickle=False)
        for r in m["results"]:
            if r["idx"] in res:
                overridden.append(r["idx"])
                arrs.pop(r["idx"], None)
            res[r["idx"]] = r
            if str(r["idx"]) in z.files:
                arrs[r["idx"]] = z[str(r["idx"])]
    missing = [c["idx"] for c in C if c["idx"] not in res]
    common_meta = {"level": "torch._scaled_mm", "kind": "gemm", "hardware": hw,
                   "producer": {"agent": "mmaemu WS2 (pre-swap recordings, 2026-10)",
                                "scripts": ["tools/record/r1_api_grid.py", "tools/record/live.py",
                                            "tools/record/common.py"],
                                "sources": ["stage calls.json + worker logs (CUBLASLT_LOG_LEVEL=2)"]}}
    notes = ("Each output = one torch._scaled_mm call. params.call = the exact call (dtypes, layout, padded ld, storage "
             "offset, scale dtype/shape/value, bias, scale_result, out=, grad); params.result = the returned tensor's "
             "dtype/shape/strides/requires_grad (absent when the call raised: hw_error = '<Class>: <first line>'). "
             "Inputs a [M,K], b [N,K] are the LOGICAL operands (codes; uint16 bf16 bits for bf16 operands; mat2 = "
             "b.t() logically) whatever the physical layout; params.layout describes the physical layout. "
             "params.splitk / params.cublas come from the logged cuBLASLt call ({'S': 1} with source 'no cuBLASLt "
             "matmul call logged' when torch returned without calling cuBLASLt). Scales are fp32 bit patterns in "
             "params.scale_a/scale_b when the scale is an fp32 (or exactly representable) scalar; raw scale tensors "
             "are inputs scale_a_raw / scale_b_raw. Canary after every call (a 16x32x16 ones GEMM must give 32.0); "
             f"calls without a result: {len(missing)}.")
    writers = {
        "r1": F.RecordingWriter("r1.api_parity_grid", "r1", {
            **common_meta, "title": "R1: torch._scaled_mm API parity grid on the RTX 4060 (raise / no-raise, error "
                                    "class + message, output bits, per-call cuBLASLt config)",
            "claims": ["SPEC v0.1 WS2 R1: authority for mmaemu's ReferenceRejectsError rules (section 6 validation "
                       "order step 2)", "SPEC section 2 C14 preliminary rules re-recorded"],
            "notes": (__doc__ or "") + "\n" + notes, "params": {}}, root=common.REC_DATA,
            max_file_bytes=common.MAX_FILE_BYTES),
        "r6": F.RecordingWriter("r6.zero_dims", "r6", {
            **common_meta, "title": "R6: zero-dimension torch._scaled_mm GEMMs on the RTX 4060 (M/N/K = 0 with real "
                                    "scales, every out dtype and format pair; K = 0 with special scales)",
            "claims": ["SPEC v0.1 WS2 R6; SPEC section 3.2 'Zero dimensions' (preliminary [Mk]) re-recorded"],
            "notes": notes, "params": {}}, root=common.REC_DATA, max_file_bytes=common.MAX_FILE_BYTES),
    }
    cases = {}
    for c in C:
        cases.setdefault((c["rec"], c["case"]), []).append(c)
    stats = dict(calls=len(C), raised=0, returned=0, setup_errors=0, no_result=len(missing))
    for (recname, case), cs in cases.items():
        c0 = cs[0]
        a, b = logical(c0)
        ins = {"a": common.tbits(a), "b": common.tbits(b)}
        cp = dict(fmt_a=_fmt_param(c0["mat1"]["dtype"]), fmt_b=_fmt_param(c0["mat2"]["dtype"]),
                  data=dict(kind="randn40", seed=c0["seed"]), layout=_layout_desc(c0), tags=["api_parity"],
                  call={k: c0[k] for k in CALL_KEYS if k not in ("out_dtype", "fast")})
        outs = []
        for c in cs:
            r = res.get(c["idx"])
            op = _out_params(c, r, c0)
            o = dict(name=c["name"], params=op)
            if r is None:
                o["hw_error"] = "NOT RUN (worker stopped)"
            elif r.get("setup_error"):
                o["hw_error"] = "SETUP: " + r["setup_error"]
                stats["setup_errors"] += 1
            elif r.get("error"):
                o["hw_error"] = r["error"]
                stats["raised"] += 1
            else:
                o["bits"] = arrs[c["idx"]]
                stats["returned"] += 1
            if r is not None and r.get("bias_bits") is not None and "bias" not in ins:
                bt = np.asarray(r["bias_bits"], dtype={"bfloat16": np.uint16, "float16": np.uint16,
                                                       "float32": np.uint32}[c["bias"]["dtype"]])
                ins["bias"] = bt
            if c["scale_a"]["dtype"] != "float32" or c["scale_a"]["shape"] != [] or \
                    c["scale_a"]["value"] == "rowwise":
                ins.setdefault("scale_a_raw", common.tbits(scale_tensor(c["scale_a"], c["M"], c["N"], "a", "cpu")))
            if c["scale_b"]["dtype"] != "float32" or c["scale_b"]["shape"] != [] or \
                    c["scale_b"]["value"] == "rowwise":
                ins.setdefault("scale_b_raw", common.tbits(scale_tensor(c["scale_b"], c["M"], c["N"], "b", "cpu")))
            if c["scale_a"]["value"] == "rowwise" and c["scale_b"]["value"] == "rowwise":
                ins["scale_a"] = common.tbits(scale_tensor(c["scale_a"], c["M"], c["N"], "a", "cpu").reshape(-1))
                ins["scale_b"] = common.tbits(scale_tensor(c["scale_b"], c["M"], c["N"], "b", "cpu").reshape(-1))
            outs.append(o)
        writers[recname].add_case(case, cp, ins, outs)
    for w in writers.values():
        print("wrote", w.close())
    stats["re_recorded_calls"] = len(overridden)
    summary = dict(stats=stats, missing=missing, re_recorded=sorted(overridden),
                   re_recorded_reason="layout calls re-recorded after fixing the recorder's mat2 padded-ld / "
                                      "storage-offset layout (the first run passed mat2 row-major where column-major "
                                      "was intended); results of the re-run replace the first run")
    (common.REC_ROOT / "r1_r6_summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps(stats))


def _fmt_param(dtype):
    return {"float8_e4m3fn": "e4m3", "float8_e5m2": "e5m2", "float8_e4m3fnuz": "e4m3fnuz",
            "float8_e5m2fnuz": "e5m2fnuz", "bfloat16": "bf16"}.get(dtype, dtype)


def _layout_desc(c):
    parts = []
    for w in ("mat1", "mat2"):
        s = c[w]
        if s["layout"] != "std":
            parts.append(f"{w} {'column' if w == 'mat1' else 'row'}-major")
        if s["ld_pad"]:
            parts.append(f"{w} ld = K + {s['ld_pad']}")
        if s["offset"]:
            parts.append(f"{w} storage offset {s['offset']} B")
    return "std" if not parts else "; ".join(parts)


CALL_KEYS = ("M", "N", "K", "mat1", "mat2", "out_dtype", "fast", "scale_a", "scale_b", "bias", "scale_result",
             "out", "grad", "positional_bias")


def _out_params(c, r, c0):
    """Per-output params: effective out dtype, fast flag, scales (fp32 bits), the call fields that differ from the
    case-level params.call, the logged cuBLASLt config and the returned tensor's metadata."""
    import torch
    od = c["out_dtype"] or c["mat1"]["dtype"]
    p = dict(out_dtype=od, fast_accum=bool(c["fast"]))
    for w in ("scale_a", "scale_b"):
        s = c[w]
        if s["value"] == "rowwise":
            p["scaling"] = "rowwise"
            continue
        fv = float(s["value"])
        try:
            p[w] = common.f32bits(float(torch.tensor(fv, dtype=getattr(torch, s["dtype"])).float().item()))
        except Exception:  # noqa: BLE001
            p[w] = common.f32bits(fv)
    if c["scale_result"]:
        p["scale_result"] = common.f32bits(float(c["scale_result"]["value"]))
    if c["bias"]:
        p["bias_dtype"] = c["bias"]["dtype"]
    diff = {k: c[k] for k in CALL_KEYS if c[k] != c0[k] and k not in ("out_dtype", "fast")}
    if c["out_dtype"] is None:
        p["out_dtype_arg"] = None          # the call relied on torch's default out_dtype
    if diff:
        p["call_diff"] = diff
    if r is not None:
        lt = r.get("lt")
        if lt is not None or r.get("result") is not None:
            p["splitk"] = common.splitk_of(lt)
        if lt:
            p["cublas"] = dict(algo=lt["algo"], tile=lt["tile_id"], stages=lt["stages_id"],
                               numSplitsK=lt["numSplitsK"], customOption=lt["customOption"],
                               ctaSwizzling=lt["ctaSwizzling"], workspace_bytes=lt["workspace_bytes"],
                               Cdesc_type=lt["Cdesc"]["type"], Ddesc_type=lt["Ddesc"]["type"], d_scale=lt["d_scale"])
        for k in ("result", "out_before", "backward", "lt_messages"):
            if r.get(k) not in (None, [], ""):
                p[k] = r[k]
        if c["mat1"] != c0["mat1"] or c["mat2"] != c0["mat2"] or _layout_desc(c) != "std":
            if r.get("mat_meta"):
                p["mat_meta"] = r["mat_meta"]
        if not r.get("canary_ok", True):
            p["canary_ok"] = False
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("calls")
    r = sp.add_parser("run")
    r.add_argument("--chunk", type=int, default=1000)
    r.add_argument("--start", type=int, default=0)
    r.add_argument("--end", type=int, default=None)
    r.add_argument("--prefix", default="part", help="part file prefix; later prefixes (sorted) override earlier "
                                                    "results of the same call index")
    w = sp.add_parser("worker")
    w.add_argument("--start", type=int, required=True)
    w.add_argument("--end", type=int, required=True)
    w.add_argument("--out", required=True)
    sp.add_parser("write")
    a = ap.parse_args()
    {"calls": cmd_calls, "run": cmd_run, "worker": cmd_worker, "write": cmd_write}[a.cmd](a)


if __name__ == "__main__":
    main()
