"""Signed zeros, underflow, overflow, NaN bit patterns and the E5M2 range: torch._scaled_mm on this GPU vs
tcemu.reference.scaled_mm (CPU), compared with torch.equal (what the tests use) and with raw bit views.
Each probe is a 16 x 32 @ 32 x 16 GEMM; only element [0, 0] (and [0, 1]) is constructed.
Run: /data/quant/tcemu_audit/gpu.sh python -u zero_specials.py -> results/zero_specials.json"""
import json, math
from pathlib import Path
import torch
from tools.bringup.checks import same_bits
import sys as _sys
from pathlib import Path as _P
_d = _P(__file__).resolve().parent
while not (_d / "compat.py").exists():   # tools/bringup (compat.py)
    _d = _d.parent
_sys.path.insert(0, str(_d))
from compat import R  # noqa: E402  (vendored: was `from tcemu import reference as R`)

import os as _os
from pathlib import Path as _Path
def _bringup_out():
    d = _Path(_os.environ.get("BRINGUP_OUT") or ".")
    d.mkdir(parents=True, exist_ok=True)
    return d
def _bringup_cache():
    d = _Path(_os.environ.get("BRINGUP_CACHE") or (_Path.home() / ".cache" / "mmaemu" / "bringup"))
    d.mkdir(parents=True, exist_ok=True)
    return d

RES = _bringup_out()
dev = "cuda:0"
E4, E5 = torch.float8_e4m3fn, torch.float8_e5m2
ODS = (torch.float32, torch.bfloat16, torch.float16)


def bits(t):
    t = t.contiguous()
    w = {4: torch.int32, 2: torch.int16}[t.element_size()]
    m = (1 << (8 * t.element_size())) - 1
    return [format(int(x) & m, "0%dX" % (2 * t.element_size())) for x in t.view(w).flatten().tolist()]


def mk(vals_a, vals_b, fa=E4, fb=E4, raw_a=None, raw_b=None):
    """A[0, :len], B[j, :len] (row j of B gives output column j). vals_b: list of rows."""
    A = torch.zeros(16, 32); B = torch.zeros(16, 32)
    A[0, :len(vals_a)] = torch.tensor(vals_a)
    for j, row in enumerate(vals_b):
        B[j, :len(row)] = torch.tensor(row)
    A8, B8 = A.to(fa), B.to(fb)
    if raw_a:
        v = A8.view(torch.uint8).clone()
        for (i, k), code in raw_a.items():
            v[i, k] = code
        A8 = v.view(fa)
    if raw_b:
        v = B8.view(torch.uint8).clone()
        for (i, k), code in raw_b.items():
            v[i, k] = code
        B8 = v.view(fb)
    return A8, B8


PROBES = []


def probe(name, A8, B8, sa=1.0, sb=1.0, cols=2, ods=ODS):
    PROBES.append((name, A8, B8, sa, sb, cols, ods))


# --- signed zeros
probe("Z1 zero acc (A row 0 all +0), scale_a=-1", *mk([0.0], [[1.0]]), sa=-1.0)
probe("Z1b zero acc, scale_b=-1", *mk([0.0], [[1.0]]), sb=-1.0)
probe("Z2 all products -0 (A +0, B negative), scales +1", *mk([0.0] * 32, [[-1.0] * 32, [-2.0] * 32]))
probe("Z3 A row of -0 bytes (0x80), B positive", *mk([0.0], [[1.0] * 32], raw_a={(0, k): 0x80 for k in range(32)}))
probe("Z4 exact cancellation 1*1 + 1*(-1)", *mk([1.0, 1.0], [[1.0, -1.0], [-1.0, 1.0]]))
probe("Z4b exact cancellation across blocks (k=0 and k=16)", *mk([1.0] + [0.0] * 15 + [1.0], [[1.0] + [0.0] * 15 + [-1.0]]))
probe("Z4c cancellation then negative scale", *mk([1.0, 1.0], [[1.0, -1.0]]), sa=-1.0)
# --- underflow in the epilogue (fp32 subnormal / FTZ?)
t18 = 2.0 ** -9  # E4M3 min subnormal; product 2^-18
probe("U1 acc=2^-18, alpha=2^-120 -> 2^-138 (fp32 subnormal output)", *mk([t18], [[t18], [-t18]]), sa=2.0 ** -60, sb=2.0 ** -60)
probe("U2 acc=1, alpha=fp32(2^-70*2^-60)=2^-130 (subnormal alpha)", *mk([1.0], [[1.0], [-1.0]]), sa=2.0 ** -70, sb=2.0 ** -60)
probe("U3 acc=3, alpha=2^-148 -> 3*2^-148 (subnormal, exact)", *mk([1.5], [[2.0], [-2.0]]), sa=2.0 ** -74, sb=2.0 ** -74)
probe("U4 acc=3, alpha=2^-150 -> rounds (alpha itself rounds to 0 or 2^-149)", *mk([1.5], [[2.0], [-2.0]]), sa=2.0 ** -75, sb=2.0 ** -75)
probe("U5 acc=-2^-18, alpha=2^-140 -> underflow to -0", *mk([t18], [[-t18], [t18]]), sa=2.0 ** -70, sb=2.0 ** -70)
probe("U6 acc=1, alpha=2^-20 -> fp16 subnormal output", *mk([1.0], [[1.0], [-1.0]]), sa=2.0 ** -10, sb=2.0 ** -10)
probe("U7 acc=3, alpha=2^-26 -> fp16 underflow (3*2^-26 < 2^-24)", *mk([1.5], [[2.0], [-2.0]]), sa=2.0 ** -13, sb=2.0 ** -13)
# --- overflow
probe("O1 acc=448*448*32 = 6.4e6 -> fp16 overflow", *mk([448.0] * 32, [[448.0] * 32, [-448.0] * 32]))
probe("O2 acc=448*448, alpha=2^100 -> fp32 overflow", *mk([448.0], [[448.0], [-448.0]]), sa=2.0 ** 50, sb=2.0 ** 50)
probe("O3 alpha = fp32(2^80*2^80) = inf, acc=0 and acc=1", *mk([1.0], [[0.0], [1.0]]), sa=2.0 ** 80, sb=2.0 ** 80)
# --- E5M2 range (inputs the hardware tests never reach: |x| > 448 and subnormals < 2^-14)
probe("E1 e5m2 57344 x e4m3 448 (x32)", *mk([57344.0] * 32, [[448.0] * 32, [-448.0] * 32], fa=E5))
probe("E2 e5m2 57344 x e4m3 448 mixed with small terms", *mk([57344.0, 1.0, 2.0 ** -16, 3.0], [[448.0, 0.001953125, 448.0, -0.5]], fa=E5))
probe("E3 e5m2 subnormals 2^-16, 3*2^-16 x e4m3 subnormal 2^-9", *mk([2.0 ** -16, 3 * 2.0 ** -16, -2.0 ** -15], [[2.0 ** -9, 7 * 2.0 ** -9, 2.0 ** -9]], fa=E5))
# --- NaN / Inf bit patterns
probe("N1 e4m3 NaN 0x7F in A", *mk([1.0, 1.0], [[1.0, 1.0], [2.0, 2.0]], raw_a={(0, 0): 0x7F}))
probe("N2 e4m3 NaN 0xFF (negative NaN) in A", *mk([1.0, 1.0], [[1.0, 1.0], [2.0, 2.0]], raw_a={(0, 0): 0xFF}))
probe("N3 e5m2 +inf x 0 (NaN) and +inf x 1", *mk([1.0], [[0.0], [1.0]], fa=E5, raw_a={(0, 0): 0x7C}))
probe("N4 e5m2 +inf and -inf in one dot", *mk([1.0, 1.0], [[1.0, 1.0], [1.0, -1.0]], fa=E5, raw_a={(0, 0): 0x7C, (0, 1): 0xFC}))
probe("N5 e5m2 -inf x 1, scale_a=-1", *mk([1.0], [[1.0], [2.0]], fa=E5, raw_a={(0, 0): 0xFC}), sa=-1.0)


def main():
    rows = []
    for name, A8, B8, sa, sb, cols, ods in PROBES:
        sa_t, sb_t = torch.tensor(sa, dtype=torch.float32), torch.tensor(sb, dtype=torch.float32)
        for od in ods:
            for fast in (True, False):
                try:
                    hw = torch._scaled_mm(A8.to(dev), B8.to(dev).t(), scale_a=sa_t.to(dev), scale_b=sb_t.to(dev),
                                          out_dtype=od, use_fast_accum=fast).cpu()
                except Exception as e:  # noqa: BLE001
                    rows.append(dict(probe=name, out=str(od), fast=fast, hw_error=str(e).splitlines()[0][:160])); continue
                em = R.scaled_mm(A8, B8.t(), sa_t, sb_t, out_dtype=od, use_fast_accum=fast)
                emg = R.scaled_mm(A8.to(dev), B8.to(dev).t(), sa_t.to(dev), sb_t.to(dev), out_dtype=od,
                                  use_fast_accum=fast).cpu()
                h, m, mg = hw[0, :cols], em[0, :cols], emg[0, :cols]
                nan_both = torch.isnan(h.float()) & torch.isnan(m.float())
                hb, mb, mgb = bits(h), bits(m), bits(mg)
                rows.append(dict(probe=name, out=str(od).split(".")[-1], fast=fast, hw=[str(x) for x in h.float().tolist()],
                                 hw_bits=hb, model_cpu_bits=mb, model_gpu_bits=mgb,
                                 torch_equal_full=bool(same_bits(hw, em)),
                                 nanaware_value_equal=bool(((h.float() == m.float()) | nan_both).all()),
                                 bits_equal_cpu=hb == mb, bits_equal_gpu=hb == mgb,
                                 bits_equal_nan_masked=all(x == y or bool(n) for x, y, n in zip(hb, mb, nan_both.tolist()))))
    env = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda)
    json.dump(dict(env=env, rows=rows), open(RES / "zero_specials.json", "w"), indent=1)
    print(json.dumps(env))
    for r in rows:
        if "hw_error" in r:
            print(f"{r['probe'][:60]:60s} {r['out']:9s} fast={r['fast']!s:5s} HW ERROR {r['hw_error'][:80]}")
            continue
        flag = "" if r["bits_equal_cpu"] else ("  <-- BITS DIFFER (torch.equal=%s)" % r["torch_equal_full"])
        print(f"{r['probe'][:60]:60s} {r['out']:9s} fast={r['fast']!s:5s} hw={r['hw_bits']} cpu={r['model_cpu_bits']} "
              f"gpu={r['model_gpu_bits']}{flag}")


if __name__ == "__main__":
    main()
