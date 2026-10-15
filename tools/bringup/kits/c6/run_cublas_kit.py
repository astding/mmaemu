"""GEMM-level must-measure: embed K=32 test vectors (with an accumulator input c) into torch._scaled_mm GEMMs and read
the diagonal.  Row i of A = [cx_i, 0 x31, a_i(32)], row i of B = [cy_i, 0 x31, b_i(32)]  (K = 64), so
D[i, i] = MMA(a_i, b_i, c_i) with c_i = cx_i * cy_i produced by the first 32-element K slice (exact under every
candidate model: a single FP8 product).  Measures the accumulator path cuBLAS actually uses (QMMA chain on
sm_89 / sm_120, QGMMA (wgmma) chain on sm_90), without a custom kernel.

usage: python run_cublas_kit.py --set MM_cublas64 --out result.json [--sizes 1024 4096]
Only the E4M3 set is runnable (torch._scaled_mm rejects e5m2 x e5m2).
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from tools.bringup.checks import same_bits

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

import gzip as _gzip
def _open(p, mode="r"):
    """text reader for the vendored (gzipped) sets: p or p.gz"""
    p = Path(p)
    if "w" in mode:
        return open(p, mode)
    return open(p) if p.exists() else _gzip.open(str(p) + ".gz", "rt")
def _exists(p):
    p = Path(p)
    return p.exists() or Path(str(p) + ".gz").exists()

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--set", default="MM_cublas64")
ap.add_argument("--fmt", default="E4M3")
ap.add_argument("--sizes", nargs="+", type=int, default=[1024, 4096])
ap.add_argument("--out")
args = ap.parse_args()


def read_hex(p):
    return np.array([[int(h, 16) for h in l.split()] for l in _open(p) if l.strip()], dtype=np.uint32)


def read_bits(p):
    return np.array([int(l.strip(), 2) for l in _open(p) if l.strip()], dtype=np.uint32)


def canon(u):
    f = u.view(np.float32)
    return np.where(np.isnan(f), np.uint32(0x7FC00000), u)


S, F = args.set, args.fmt
d = HERE / "sets" / S / F
a = read_hex(d / f"a_{S}_{F}.txt").view(np.float32)
b = read_hex(d / f"b_{S}_{F}.txt").view(np.float32)
c = read_bits(d / f"c_{S}_fp32.txt").view(np.float32)
cg = read_hex(d / f"cgen_{S}_{F}.txt").view(np.float32)
assert np.array_equal(cg[:, 0].astype(np.float64) * cg[:, 1], c.astype(np.float64)), "c != cx*cy"
n = len(a)
z = np.load(d / f"pred_{S}_{F}.npz")
P = canon(z["bits"]); models = json.loads(str(z["models"]))
tags = [l.strip() for l in _open(d / f"tags_{S}_{F}.txt")]
p = torch.cuda.get_device_properties(0)
res = dict(gpu=p.name, capability=list(torch.cuda.get_device_capability(0)), torch=torch.__version__,
           cuda=torch.version.cuda, cublas=getattr(torch.backends.cuda, "cublas_version", lambda: None)(), runs={})
dt = torch.float8_e4m3fn
for M in args.sizes:
    assert M >= n
    A = torch.zeros(M, 64, dtype=torch.float32); B = torch.zeros(M, 64, dtype=torch.float32)
    A[:n, 0] = torch.from_numpy(cg[:, 0].copy()); B[:n, 0] = torch.from_numpy(cg[:, 1].copy())
    A[:n, 32:] = torch.from_numpy(a.copy()); B[:n, 32:] = torch.from_numpy(b.copy())
    A8, B8 = A.to(dt).cuda(), B.to(dt).cuda()
    assert same_bits(A8.float().cpu(), A) and same_bits(B8.float().cpu(), B)
    one = torch.ones((), device="cuda")
    for fast in (True, False):
        D = torch._scaled_mm(A8, B8.t(), scale_a=one, scale_b=one, out_dtype=torch.float32, use_fast_accum=fast)
        torch.cuda.synchronize()
        meas = canon(torch.diagonal(D)[:n].cpu().numpy().view(np.uint32))
        match = (P == meas[None, :]).sum(1)
        full = [i for i in range(len(models)) if match[i] == n]
        best = int(np.argmax(match))
        key = f"M=N={M},K=64,fast_accum={fast}"
        row = dict(n=n, n_models_100pct=len(full), best_match=int(match[best]), best_model=models[best],
                   models_100pct=[models[i] for i in full][:40])
        sys.path.insert(0, str(HERE)); sys.path.insert(1, str(HERE.parent))
        import family as Fm
        for name, sp in Fm.PRESET_SPECS.items():
            k = json.dumps(sp, sort_keys=True)
            idx = [i for i, m in enumerate(models) if json.dumps(m, sort_keys=True) == k]
            if idx:
                row[f"preset_{name}"] = int(match[idx[0]])
                bad = np.nonzero(P[idx[0]] != meas)[0]
                row[f"preset_{name}_first_mismatch_tags"] = [tags[j] for j in bad[:6]]
        res["runs"][key] = row
        with open(_bringup_out() / f"d_{S}_cublas_M{M}_fast{int(fast)}_{p.name.replace(' ', '_')}.txt", "w") as fo:
            for u in meas:
                fo.write(f"{int(u):032b}\n")
        print(key, {k: v for k, v in row.items() if k != "models_100pct"}, flush=True)
if args.out:
    json.dump(res, open(args.out, "w"), indent=1)
