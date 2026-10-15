"""Run a must-measure kit on this GPU with one mma.sync.m16n8k32 per vector and report which candidate models
(and tcemu presets) predict every measured result bit for bit.

usage:
  python run_kit.py --sets MM_sm120 [MM_ada ...] --arch 12.0  [--kind] --out result_<gpu>.json
  --arch: TORCH_CUDA_ARCH_LIST value for the probe build (8.9 for Ada, 9.0a for GH200/H100, 12.0 or 12.0a for
          RTX 50xx / RTX PRO 6000).  --kind: use mma.sync.aligned.kind::f8f6f4 (needs --arch 12.0a).
  --compile-only: build the extension and exit (do this outside any GPU lock).
Writes kit/sets/<SET>/<FMT>/d_<SET>_<tag>.txt (measured fp32 bits, one per line) and a JSON summary.
Comparison is bitwise (uint32), so -0 != +0 and NaN payloads are canonicalised.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

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
ap.add_argument("--sets", nargs="+", required=True)
ap.add_argument("--arch", required=True)
ap.add_argument("--kind", action="store_true")
ap.add_argument("--compile-only", action="store_true")
ap.add_argument("--out")
args = ap.parse_args()
os.environ["TORCH_CUDA_ARCH_LIST"] = args.arch
import torch  # noqa: E402
from tools.bringup.checks import same_bits
from torch.utils.cpp_extension import load  # noqa: E402

tag = "sm" + args.arch.replace(".", "") + ("_kind" if args.kind else "")
build = _bringup_cache() / f"c6_build_{tag}"
build.mkdir(exist_ok=True)
flags = ["-O3"] + (["-DTCEMU_KIND_F8F6F4"] if args.kind else [])
ext = load(name=f"c6kit_mma_probe_{tag}", sources=[str(HERE / "mma_probe_kit.cu")], build_directory=str(build),
           extra_cuda_cflags=flags, verbose=False)
if args.compile_only:
    print("compiled", tag); sys.exit(0)


def read_hex(p):
    return np.array([[int(h, 16) for h in l.split()] for l in _open(p) if l.strip()], dtype=np.uint32)


def read_bits(p):
    return np.array([int(l.strip(), 2) for l in _open(p) if l.strip()], dtype=np.uint32)


def canon(u):
    f = u.view(np.float32)
    return np.where(np.isnan(f), np.uint32(0x7FC00000), u)


p = torch.cuda.get_device_properties(0)
res = dict(gpu=p.name, capability=list(torch.cuda.get_device_capability(0)), torch=torch.__version__,
           cuda=torch.version.cuda, probe=tag, sets={})
dt = {"E4M3": torch.float8_e4m3fn, "E5M2": torch.float8_e5m2}
for S in args.sets:
    for fdir in sorted((HERE / "sets" / S).iterdir()):
        if not fdir.is_dir():
            continue
        F = fdir.name
        a = torch.from_numpy(read_hex(fdir / f"a_{S}_{F}.txt").view(np.float32).copy())
        b = torch.from_numpy(read_hex(fdir / f"b_{S}_{F}.txt").view(np.float32).copy())
        c = torch.from_numpy(read_bits(fdir / f"c_{S}_fp32.txt").view(np.float32).copy())
        a8, b8 = a.to(dt[F]), b.to(dt[F])
        assert same_bits(a8.float(), a) and same_bits(b8.float(), b), "inputs not representable"
        d = ext.mma_k32(a8.view(torch.uint8).cuda(), b8.view(torch.uint8).cuda(), c.cuda(), F == "E5M2", F == "E5M2",
                        args.kind)
        torch.cuda.synchronize()
        meas = canon(d.cpu().numpy().view(np.uint32))
        with open(_bringup_out() / f"d_{S}_{F}_{tag}.txt", "w") as fo:
            for u in meas:
                fo.write(f"{int(u):032b}\n")
        z = np.load(fdir / f"pred_{S}_{F}.npz")
        P = canon(z["bits"])
        models = json.loads(str(z["models"]))
        match = (P == meas[None, :]).sum(1)
        full = [i for i in range(len(models)) if match[i] == len(meas)]
        tags = [l.strip() for l in _open(fdir / f"tags_{S}_{F}.txt")]
        best = int(np.argmax(match))
        row = dict(n=len(meas), n_models=len(models), n_models_100pct=len(full), best_match=int(match[best]),
                   models_100pct=[models[i] for i in full][:50], best_model=models[best])
        # tcemu presets / paper models by name, if present in the candidate list
        sys.path.insert(0, str(HERE)); sys.path.insert(1, str(HERE.parent))
        try:
            import family as Fm
            for name, sp in Fm.PRESET_SPECS.items():
                k = json.dumps(sp, sort_keys=True)
                idx = [i for i, m in enumerate(models) if json.dumps(m, sort_keys=True) == k]
                if idx:
                    row[f"preset_{name}"] = int(match[idx[0]])
                    bad = np.nonzero(P[idx[0]] != meas)[0]
                    row[f"preset_{name}_first_mismatch_tags"] = [tags[j] for j in bad[:8]]
        except ImportError:
            pass
        recf = fdir / f"d_{S}_recorded_fp32.txt"
        if _exists(recf):  # toolbox recording of the same vectors on another GPU (TBX_* sets)
            rec = canon(read_bits(recf))
            row["equals_recorded_bitwise"] = int((rec == meas).sum())
        res["sets"][f"{S}/{F}"] = row
        print(S, F, {k: v for k, v in row.items() if k not in ("models_100pct",)}, flush=True)
if args.out:
    json.dump(res, open(args.out, "w"), indent=1)
