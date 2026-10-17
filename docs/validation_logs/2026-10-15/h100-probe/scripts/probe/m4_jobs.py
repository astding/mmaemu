"""M4 job lists (CPU; built once before the rental so that the dry runs and the H100 run use the same files):
torch._scaled_mm calls in tools/record/live.py's job format, ws = -1 meaning the torch default workspace
(CUBLASLT_WORKSPACE_SIZE unset). Inputs are regenerated from (kind, seed) by live.make_inputs, so the stored row /
column subsets (full K) can be replayed offline.
  evo2    Evo 2 (K, N) = (1920, 5760), (4096, 12288), (8192, 24576) x M in {1, 16, 64, 256, 1000, 2048, 4096},
          fast / slow x bf16 / fp32, randn40; adv and ws 1 / 32 MiB at M = 2048
  c5      240 random c5-grid shapes with random out (fp32, bf16, fp16, e4m3, e5m2), formats, fast, ws, data, scales
  longk   M = N = 256, K = 32 ... 16384, fast / slow, fp32, randn40 and adv (slow-accumulation structure)
  smallk  K in {16, 32, 48, 64, 80, 96} (ragged k-blocks) on small M, N
usage: m4_jobs.py --out-dir DIR   (writes DIR/chunk_NN.json and DIR/index.json)"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

import pc  # noqa: F401  (sys.path: the source tree)
import live  # noqa: E402  (tools/record/live.py)

EVO = [(1920, 5760), (4096, 12288), (8192, 24576)]
MS = [16, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 2048]
NS = [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1920, 4096, 5760, 8192, 12288, 24576]
KS = [1024, 1920, 2048, 4096, 5760, 8192, 12288, 16384]
MiB = 1 << 20


def f32(x):
    return float(np.float32(x))


def subset(M, N, rng):
    if M * N <= 1 << 16:
        return None
    rows = set(range(min(M, 16))) | {M - 1, M - 2}
    extra = rng.choice(M, size=min(M, 64), replace=False)
    rows |= set(int(x) for x in extra[: max(0, 64 - len(rows))])
    cols = set(range(min(N, 32))) | {N - 1}
    extra = rng.choice(N, size=min(N, 256), replace=False)
    for x in extra:
        if len(cols) >= 256:
            break
        cols.add(int(x))
    return {"rows": sorted(r for r in rows if 0 <= r < M), "cols": sorted(cols)}


def job(jid, M, N, K, fa, fb, out, fast, ws, kind, seed, rng, unit=False, scale_result=None, tags=()):
    j = dict(id=jid, M=M, N=N, K=K, fa=fa, fb=fb, out=out, fast=bool(fast), ws=int(ws),
             data=dict(kind=kind, seed=int(seed)), scale_result=scale_result, tags=list(tags))
    if unit:
        j["scale_a"], j["scale_b"] = 1.0, 1.0
    else:
        sa, sb = live.default_scales(j)
        j["scale_a"], j["scale_b"] = f32(sa), f32(sb)
    j["sub"] = subset(M, N, rng)
    return j


def build(seed=20261015):
    rng = np.random.default_rng(seed)
    jobs = []
    n = itertools.count()

    def add(**kw):
        i = next(n)
        jobs.append(job(f"m4_{i:04d}", seed=1000 + i, rng=rng, **kw))
    for (K, N) in EVO:
        for M in (1, 16, 64, 256, 1000, 2048, 4096):
            for fast in (True, False):
                for out in ("bfloat16", "float32"):
                    add(M=M, N=N, K=K, fa="e4m3", fb="e4m3", out=out, fast=fast, ws=-1, kind="randn40",
                        tags=["evo2"])
        for fast in (True, False):
            for out in ("bfloat16", "float32"):
                add(M=2048, N=N, K=K, fa="e4m3", fb="e4m3", out=out, fast=fast, ws=-1, kind="adv", tags=["evo2", "adv"])
                for ws in (1 * MiB, 32 * MiB):
                    add(M=2048, N=N, K=K, fa="e4m3", fb="e4m3", out=out, fast=fast, ws=ws, kind="randn40",
                        tags=["evo2", "ws"])
    outs = ["float32", "bfloat16", "float16", "float8_e4m3fn", "float8_e5m2"]
    pairs = [("e4m3", "e4m3")] * 3 + [("e4m3", "e5m2"), ("e5m2", "e4m3")]
    wss = [-1, -1, 1 * MiB, 32 * MiB, 0]
    for _ in range(240):
        M, N, K = int(rng.choice(MS)), int(rng.choice(NS)), int(rng.choice(KS))
        out = outs[int(rng.integers(len(outs)))]
        fa, fb = pairs[int(rng.integers(len(pairs)))]
        sr = None
        if out.startswith("float8") and rng.random() < 0.5:
            sr = f32(0.37 + rng.random())
        add(M=M, N=N, K=K, fa=fa, fb=fb, out=out, fast=bool(rng.random() < 0.5), ws=int(rng.choice(wss)),
            kind="randn40" if rng.random() < 0.7 else "adv", unit=bool(rng.random() < 0.2), scale_result=sr,
            tags=["c5"])
    for K in (32, 64, 96, 128, 160, 192, 256, 384, 512, 768, 1024, 2048, 4096, 8192, 16384):
        for fast in (True, False):
            for kind in ("randn40", "adv"):
                add(M=256, N=256, K=K, fa="e4m3", fb="e4m3", out="float32", fast=fast, ws=-1, kind=kind, unit=True,
                    tags=["longk"])
    for K in (16, 32, 48, 64, 80, 96):
        for (M, N) in ((16, 16), (64, 96), (176, 224), (512, 256), (1024, 1024)):
            add(M=M, N=N, K=K, fa="e4m3", fb="e4m3", out="float32", fast=bool(K % 32), ws=-1, kind="randn40",
                unit=True, tags=["smallk"])
    return jobs


def chunks(jobs, max_jobs=80, max_cost=1e11):
    """one workspace size per chunk (one worker process each); packed in job order; cost ~ output bytes moved and
    hashed + inputs generated."""
    out = []
    for ws in sorted({j["ws"] for j in jobs}):
        cur, cost = [], 0.0
        for j in (j for j in jobs if j["ws"] == ws):
            c = j["M"] * j["N"] * 8 + (j["M"] + j["N"]) * j["K"] * 40
            if cur and (len(cur) >= max_jobs or cost + c > max_cost):
                out.append(cur)
                cur, cost = [], 0.0
            cur.append(j)
            cost += c
        if cur:
            out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args()
    jobs = build()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    cs = chunks(jobs)
    for i, c in enumerate(cs):
        (a.out_dir / f"chunk_{i:02d}.json").write_text(json.dumps(c))
    idx = dict(n_jobs=len(jobs), n_chunks=len(cs), per_chunk=[len(c) for c in cs],
               tags={t: sum(t in j["tags"] for j in jobs) for t in ("evo2", "c5", "longk", "smallk")},
               stored_values=int(sum((len(j["sub"]["rows"]) * len(j["sub"]["cols"])) if j["sub"] else j["M"] * j["N"]
                                     for j in jobs)))
    (a.out_dir / "index.json").write_text(json.dumps(idx, indent=1) + "\n")
    print(json.dumps(idx))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.exit(main())
