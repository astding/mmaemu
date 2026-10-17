"""Part 5 job selection (CPU, on the measured host, from its own p1-p3 recordings): live torch._scaled_mm calls in
tools/record/live.py's job format (as the probe's m4_jobs.py / m6_split_jobs.py), for tools/replay_h100_log.py.

split   over the split rows of p1-p3 at the profile's own alignment that fit (K <= 262,144, input size
        (M + N) * K <= 2^26, which sets the worker's time; M * N * K <= 2^36), per verification key (fast, S, out,
        fa, fb, reduction; the H100's split scheme is always contiguous), cheapest = smallest (M + N) * K, in order:
          A  one adv_split job per key on its cheapest row with K % 128 != 0, else its cheapest row;
          B  one adv_split job per key on its cheapest row with K > 16,384 (keys that have one);
          C  one randn40 job per key on the row of A;
          D  fill: one adv_split job per (key, ws, customOption) not yet covered, cheapest first,
        up to --max-split jobs and --max-volume elements of (M + N) * K in all. A is run a second time.
unsplit S = 1 at K % 128 != 0 (K in 2,064, 5,744, 8,208, 16,400, 65,552; M = 16 and 128 with the p1 N nearest to
        1,024 and 512), fast and slow, fp32 and bf16, e4m3 x e4m3 and e5m2 x e4m3, 1 MiB, randn40 (p1 rows the
        heuristic gives S = 1).
model   p3 rows at M in {1, 16, 128} of each model (K, N) with N * K <= 2^27, fast and slow, bf16, e4m3 x e4m3,
        both workspaces, randn40: the decisions Evo 2 / Llama-style decode and small prefill make.
Every job carries the heuristic row it came from (j["table"]) so the logged cuBLASLt config of the live call can be
compared with the query. Writes chunks of at most 120 jobs: OUT_DIR/a_NN.json (A), bc_NN.json (B, C),
other_NN.json (unsplit and model), fill_NN.json (D) and OUT_DIR/index.json.
usage: live_jobs.py SWEEP_OUT_ROOT OUT_DIR [--max-split 2400] [--max-volume 2.4e10]"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import sc
import live  # noqa: E402  (tools/record/live.py of the source tree)

FMTS = {0: "e4m3", 1: "e5m2"}
SEED = 20261017


def f32(x):
    return float(np.float32(x))


def subset(M, N, rng):
    """as the probe's m4_jobs.subset: store everything up to 2^16 outputs, else <= 64 rows x <= 256 columns."""
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


def job(jid, M, N, K, fa, fb, out, fast, ws, kind, seed, rng, S=None, row=None, tags=()):
    j = dict(id=jid, M=int(M), N=int(N), K=int(K), fa=fa, fb=fb, out=out, fast=bool(fast), ws=int(ws),
             data=dict(kind=kind, seed=int(seed)), scale_result=None, tags=list(tags))
    if kind == "adv_split":
        j["data"]["S"] = int(S)
    sa, sb = live.default_scales(dict(j, data=dict(kind="randn40" if kind == "randn40" else "adv")))
    j["scale_a"], j["scale_b"] = f32(sa), f32(sb)
    j["sub"] = subset(int(M), int(N), rng)
    if row is not None:
        j["table"] = row
    return j


def load(root: Path):
    parts = []
    for p in ("p1", "p2", "p3"):
        files = sc.part_files(root / p)
        for prof, (ws, al) in sc.PROFILE_WS.items():
            if f"{prof}_a{al}" in files:
                z = sc.load_cols(files[f"{prof}_a{al}"])
                parts.append(z | {"part": np.full(len(z["M"]), int(p[1]), np.int8)})
    return {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}


def row_of(c, i):
    return {k: int(c[k][i]) for k in ("M", "N", "K", "fa", "fb", "out", "fast", "ws", "align", "status", "algo",
                                      "tile", "stages", "splitK", "reduction", "swizzle", "custom", "workspace",
                                      "part")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-split", type=int, default=2400)
    ap.add_argument("--max-volume", type=float, default=2.4e10)
    a = ap.parse_args()
    c = load(a.root)
    rng = np.random.default_rng(SEED)
    M, N, K = c["M"], c["N"], c["K"]
    cost = M * N * K
    vol = (M + N) * K
    fit = (K <= 262_144) & (vol <= 1 << 26) & (cost <= 1 << 36)
    ok = c["status"] == 0
    split_all = ok & (c["splitK"] > 1)
    idx = np.nonzero(split_all & fit)[0]
    vcols = ("fast", "splitK", "out", "fa", "fb", "reduction")
    n_keys_all = len(np.unique(np.stack([c[k][split_all] for k in vcols], 1), axis=0)) if split_all.any() else 0
    order = idx[np.lexsort((cost[idx], vol[idx]))]               # fit split rows, cheapest first
    _, vid = np.unique(np.stack([c[k][order] for k in vcols], 1), axis=0, return_inverse=True)
    vid = vid.ravel()
    nv = int(vid.max()) + 1 if len(vid) else 0

    def first_per(mask):
        """{key id: row index} of the cheapest row per key among order[mask]."""
        sel = np.nonzero(mask)[0]
        u, j = np.unique(vid[sel], return_index=True)
        return dict(zip(u.tolist(), order[sel[j]].tolist()))
    cheapest = first_per(np.ones(len(order), bool))
    ragged = first_per(K[order] % 128 != 0)
    longk = first_per(K[order] > 16384)
    jobs, n = [], itertools.count()

    def add(i, kind, tag):
        m = next(n)
        S = int(c["splitK"][i])
        jobs.append(job(f"p5s_{m:04d}_{kind}", M[i], N[i], K[i], FMTS[int(c["fa"][i])], FMTS[int(c["fb"][i])],
                        sc.OUTS[int(c["out"][i])], int(c["fast"][i]), int(c["ws"][i]), kind, 50000 + m, rng, S=S,
                        row=row_of(c, i), tags=["split", tag, f"S{S}", f"red{int(c['reduction'][i])}",
                                                f"custom{int(c['custom'][i])}"]))
    a_rows = {v: ragged.get(v, cheapest[v]) for v in range(nv)}
    for v in range(nv):
        add(a_rows[v], "adv_split", "A")
    n_a = len(jobs)
    for v in range(nv):
        if v in longk and longk[v] != a_rows[v]:
            add(longk[v], "adv_split", "B")
    for v in range(nv):
        add(a_rows[v], "randn40", "C")
    total = 0
    for q, j in enumerate(jobs):                                  # the budget cuts C, then B, never A
        total += (j["M"] + j["N"]) * j["K"]
        if q >= n_a and (q >= a.max_split or total > a.max_volume):
            jobs = jobs[:q]
            break
    n_a = min(n_a, len(jobs))
    n_core = len(jobs)
    covered = {(int(c["ws"][i]), int(c["custom"][i]), v) for v, i in a_rows.items()}
    covered |= {(int(c["ws"][i]), int(c["custom"][i]), v) for v, i in longk.items()}
    _, gid = np.unique(np.stack([vid, c["ws"][order], c["custom"][order]], 1), axis=0, return_index=True)
    for r in sorted(gid.tolist()):                                # cheapest first across groups
        i = int(order[r])
        if len(jobs) >= a.max_split or total + vol[i] > a.max_volume:
            break
        total += int(vol[i])
        g = (int(c["ws"][i]), int(c["custom"][i]), int(vid[r]))
        if g not in covered:
            covered.add(g)
            add(i, "adv_split", "D")
    n_split = len(jobs)
    # S = 1 at K % 128 != 0 (rows the heuristic decided S = 1 for; checked against the query)
    for (Kx, (m_, n_), fast, out, (fa, fb)) in itertools.product((2064, 5744, 8208, 16400, 65552),
                                                                  ((16, 1024), (128, 512)), (True, False),
                                                                  ("float32", "bfloat16"),
                                                                  (("e4m3", "e4m3"), ("e5m2", "e4m3"))):
        sel = np.nonzero(ok & (M == m_) & (K == Kx) & (c["fast"] == int(fast)) & (c["out"] == sc.OUTC[out]) &
                         (c["fa"] == sc.FMTC[fa]) & (c["fb"] == sc.FMTC[fb]) & (c["ws"] == sc.MiB) &
                         (c["align"] == 0) & (c["splitK"] == 1))[0]
        if not len(sel):
            continue
        i = int(sel[np.argmin(np.abs(N[sel] - n_))])        # the recorded N nearest to the target
        m = next(n)
        jobs.append(job(f"p5u_{m:04d}", m_, N[i], Kx, fa, fb, out, fast, sc.MiB, "randn40", 50000 + m, rng,
                        row=row_of(c, i), tags=["unsplit", "K_not_128"]))
    n_unsplit = len(jobs) - n_split
    # model rows at decode / small prefill
    kn = sorted({(Kx, Nx) for layers in json.loads((sc.SHAPES / "index.json").read_text())["p3_models"].values()
                 for (_, Kx, Nx) in layers})
    for (Kx, Nx), Mx, fast, ws in itertools.product(kn, (1, 16, 128), (True, False), (sc.MiB, 32 * sc.MiB)):
        al = 0 if ws == sc.MiB else 256
        sel = np.nonzero(ok & (M == Mx) & (N == Nx) & (K == Kx) & (c["fast"] == int(fast)) &
                         (c["out"] == sc.OUTC["bfloat16"]) & (c["fa"] == 0) & (c["fb"] == 0) & (c["ws"] == ws) &
                         (c["align"] == al))[0]
        if not len(sel) or Nx * Kx > 1 << 27:
            continue
        i = int(sel[0])
        m = next(n)
        jobs.append(job(f"p5m_{m:04d}", Mx, Nx, Kx, "e4m3", "e4m3", "bfloat16", fast, ws, "randn40", 50000 + m, rng,
                        row=row_of(c, i), tags=["model", f"S{int(c['splitK'][i])}"]))
    n_model = len(jobs) - n_split - n_unsplit
    a.out.mkdir(parents=True, exist_ok=True)
    chunks = []
    for name, js in (("a", jobs[:n_a]), ("bc", jobs[n_a:n_core]), ("other", jobs[n_split:]),
                     ("fill", jobs[n_core:n_split])):
        for q, i in enumerate(range(0, len(js), 120)):
            (a.out / f"{name}_{q:02d}.json").write_text(json.dumps(js[i:i + 120]))
            chunks.append(f"{name}_{q:02d}")
    idx = dict(jobs=len(jobs), split=n_split, unsplit=n_unsplit, model=n_model, chunks=chunks,
               a=n_a, bc=n_core - n_a, fill=n_split - n_core,
               volume={k: sum((j["M"] + j["N"]) * j["K"] for j in js) for k, js in
                       (("a", jobs[:n_a]), ("bc", jobs[n_a:n_core]), ("fill", jobs[n_core:n_split]),
                        ("other", jobs[n_split:]))}, verification_keys_all=n_keys_all, verification_keys_fit=nv,
               verification_keys_jobbed=len({tuple(j["table"][k] for k in vcols) for j in jobs if "split" in j["tags"]}),
               split_K_not_128=sum(j["K"] % 128 != 0 for j in jobs[:n_split]),
               split_K_above_16384=sum(j["K"] > 16384 for j in jobs[:n_split]),
               max_cost=max((j["M"] * j["N"] * j["K"] for j in jobs), default=0),
               max_K=max((j["K"] for j in jobs), default=0), seed=SEED)
    (a.out / "index.json").write_text(json.dumps(idx, indent=1) + "\n")
    print(json.dumps(idx))


if __name__ == "__main__":
    main()
