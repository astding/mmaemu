"""M1: how the accumulator input c enters, measured through cuBLAS with the c6 MM_cublas64* vector sets (E4M3), at
the kit's K = 64 and embedded in the production shapes, so that the kernels of the real calls run (the heuristic does
not look at the data).

Embedding (as tools/bringup/kits/c6/run_cublas_kit.py, generalised to any K): row i of A = [cx_i, 0, ..., 0, a_i],
row i of B = [cy_i, 0, ..., 0, b_i] with a_i, b_i in the last 32 k-slots, so D[i, j] = chain(a_i, b_j) started from
c = cx_i * cy_j (one exact FP8 product, carried unchanged through the zero blocks; checked on CPU by cpu_checks.py).
The diagonal is the kit's vector set; the stored n x n block D[:n, :n] adds the off-diagonal (a_i, b_j, cx_i cy_j)
combinations. Rows/columns >= n are zero.
torch: fp32 out, unit scales, fast and slow. TE (--te): general_gemm fp32 out, unit scale_inv, split False / True.
usage: m1_cvec.py [--te] [--sets MM_cublas64,MM_cublas64_sm120] [--shapes 1024x1024x64,...]"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import pc

KIT = pc.SRC / "tools" / "bringup" / "kits" / "c6"
SHAPES = "1024x1024x64,4096x4096x64,2048x5760x1920,2048x12288x4096,2048x24576x8192"


def _open(p: Path):
    return open(p) if p.exists() else gzip.open(str(p) + ".gz", "rt")


def _sha_file(p: Path) -> str:
    q = p if p.exists() else Path(str(p) + ".gz")
    return hashlib.sha256(q.read_bytes()).hexdigest()


def read_hex(p):
    return np.array([[int(h, 16) for h in ln.split()] for ln in _open(p) if ln.strip()], dtype=np.uint32)


def canon(u):
    return np.where(np.isnan(u.view(np.float32)), np.uint32(0x7FC00000), u)


def load_set(S: str, F: str = "E4M3"):
    d = KIT / "sets" / S / F
    a = read_hex(d / f"a_{S}_{F}.txt").view(np.float32)
    b = read_hex(d / f"b_{S}_{F}.txt").view(np.float32)
    cg = read_hex(d / f"cgen_{S}_{F}.txt").view(np.float32)
    z = np.load(d / f"pred_{S}_{F}.npz")
    files = {n: _sha_file(d / n) for n in (f"a_{S}_{F}.txt", f"b_{S}_{F}.txt", f"cgen_{S}_{F}.txt")}
    return a, b, cg, canon(z["bits"]), json.loads(str(z["models"])), files


def summarise(diag, P, models):
    sys.path.insert(0, str(KIT))
    import family as Fm
    match = (P == diag[None, :]).sum(1)
    order = np.argsort(-match)[:5]
    out = dict(n=int(len(diag)), n_models_100pct=int((match == len(diag)).sum()),
               top=[dict(match=int(match[i]), model=models[i]) for i in order])
    for name, sp in Fm.PRESET_SPECS.items():
        k = json.dumps(sp, sort_keys=True)
        idx = [i for i, m in enumerate(models) if json.dumps(m, sort_keys=True) == k]
        if idx:
            out[f"preset_{name}"] = int(match[idx[0]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--te", action="store_true")
    ap.add_argument("--sets", default="MM_cublas64,MM_cublas64_sm120")
    ap.add_argument("--shapes", default=SHAPES)
    a = ap.parse_args()
    import torch
    rec = pc.Rec("m1_te" if a.te else "m1_torch")
    api = pc.te_api() if a.te else None
    dev = torch.device("cuda:0")
    one = torch.ones((), device=dev)
    extra = dict(path="te" if a.te else "torch", te=pc.te_meta(api) if api else None, sets={}, summaries={})
    for S in a.sets.split(","):
        av, bv, cg, P, models, files = load_set(S)
        n = len(av)
        extra["sets"][S] = dict(n=n, n_models=len(models), files_sha256=files)
        for shp in a.shapes.split(","):
            M, N, K = (int(x) for x in shp.split("x"))
            if n > min(M, N):
                continue
            A = torch.zeros(M, K, dtype=torch.float32)
            B = torch.zeros(N, K, dtype=torch.float32)
            A[:n, 0] = torch.from_numpy(cg[:, 0].copy())
            B[:n, 0] = torch.from_numpy(cg[:, 1].copy())
            A[:n, K - 32:] = torch.from_numpy(av.copy())
            B[:n, K - 32:] = torch.from_numpy(bv.copy())
            A8, B8 = A.to(torch.float8_e4m3fn), B.to(torch.float8_e4m3fn)
            assert torch.equal(A8.float(), A) and torch.equal(B8.float(), B), "inputs not representable"
            Au, Bu = A8.view(torch.uint8).to(dev), B8.view(torch.uint8).to(dev)
            Ad, Bd = Au.view(torch.float8_e4m3fn), Bu.view(torch.float8_e4m3fn)
            del A, B, A8, B8
            meta = dict(set=S, M=M, N=N, K=K, n=n, out="float32")

            def keep(bits, n=n):
                return {"block": bits[:n, :n]}
            modes = ([("fast", dict(fast=True)), ("slow", dict(fast=False))] if not a.te else
                     [("te", dict(use_split_accumulator=False)), ("te_split", dict(use_split_accumulator=True))])
            for tag, kw in modes:
                cid = f"{S}.M{M}N{N}K{K}.{tag}"
                if not a.te:
                    fn = (lambda f=kw["fast"]: torch._scaled_mm(Ad, Bd.t(), scale_a=one, scale_b=one,
                                                                out_dtype=torch.float32, use_fast_accum=f))
                else:
                    fn = (lambda s=kw["use_split_accumulator"]: pc.te_gemm(api, Bu, Au, one, one, torch.float32, s))
                bits = rec.call(cid, fn, keep, dict(meta, **kw))
                if bits is not None:
                    extra["summaries"][cid] = summarise(canon(np.diagonal(bits[:n, :n]).copy()), P, models)
                    print("   ", json.dumps({k: v for k, v in extra["summaries"][cid].items() if k != "top"}),
                          "top", extra["summaries"][cid]["top"][0]["match"], flush=True)
            del Au, Bu, Ad, Bd
            torch.cuda.empty_cache()
    rec.save(extra)


if __name__ == "__main__":
    main()
