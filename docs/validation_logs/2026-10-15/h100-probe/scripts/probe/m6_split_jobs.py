"""M6 live split rows (CPU, on the measured host): from this GPU's heuristic recording (tools/record_heuristics.py
npz), for every (ws, out, fast, fa, fb, splitK, reduction, customOption) combination that splits, take the smallest
M*N*K row torch's descriptor reaches (align none, no D scale) and write live.py jobs on adv_split data (S = splitK)
and randn40. Outputs up to 2^16 values are stored in full, larger ones as a row/column subset (full K).
usage: m6_split_jobs.py HEUR.npz OUT.json [--max-jobs 1000]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import m4_jobs
import pc  # noqa: F401

OUTS = {0: "float32", 1: "bfloat16", 2: "float16", 3: "float8_e4m3fn", 4: "float8_e5m2"}
FMTS = {0: "e4m3", 1: "e5m2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("heur", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-jobs", type=int, default=1000)
    a = ap.parse_args()
    z = np.load(a.heur)
    c = {k: z[k] for k in z.files}
    ok = (c["status"] == 0) & (c["splitK"] > 1) & (c["align"] == 0) & (c["dscale"] == 0)
    keys = np.stack([c[k] for k in ("ws", "out", "fast", "fa", "fb", "splitK", "reduction", "custom")], 1)
    cost = c["M"] * c["N"] * c["K"]
    rng = np.random.default_rng(6)
    combos = []
    for k in np.unique(keys[ok], axis=0):
        sel = np.nonzero(ok & np.all(keys == k, 1))[0]
        i = sel[np.argmin(cost[sel])]
        combos.append((int(cost[i]), i, k))
    # round robin over (ws, splitK, reduction, customOption, fast, out) groups, cheapest first within a group, so the
    # job cap keeps every kind of split
    groups: dict = {}
    for t in sorted(combos, key=lambda t: t[0]):
        k = t[2]
        groups.setdefault((int(k[0]), int(k[5]), int(k[6]), int(k[7]), int(k[2]), int(k[1])), []).append(t)
    # first: the Evo 2 (K, N) rows that split (smallest M per setting and split kind)
    evo = np.zeros(len(cost), bool)
    for (K, N) in ((1920, 5760), (4096, 12288), (8192, 24576)):
        evo |= (c["K"] == K) & (c["N"] == N)
    ekeys = np.concatenate([keys, c["K"][:, None], c["N"][:, None]], 1)
    combos = []
    for k in np.unique(ekeys[ok & evo], axis=0):
        sel = np.nonzero(ok & evo & np.all(ekeys == k, 1))[0]
        i = sel[np.argmin(cost[sel])]
        combos.append((int(cost[i]), i, k[:8]))
    n_evo = len(combos)
    for rnd in range(max(len(v) for v in groups.values()) if groups else 0):
        combos += [v[rnd] for _, v in sorted(groups.items()) if rnd < len(v)]
    jobs = []
    for n, (_, i, k) in enumerate(combos):
        ws, out, fast, fa, fb, S, red, cust = (int(x) for x in k)
        M, N, K = int(c["M"][i]), int(c["N"][i]), int(c["K"][i])
        for kind in ("adv_split", "randn40"):
            if len(jobs) >= a.max_jobs:
                break
            j = m4_jobs.job(f"m6_{n:03d}_{kind}", M, N, K, FMTS[fa], FMTS[fb], OUTS[out], bool(fast), ws,
                            "randn40" if kind == "randn40" else "adv", 6000 + n, rng,
                            tags=["m6", f"S{S}", f"red{red}", f"custom{cust}"])
            if kind == "adv_split":
                j["data"] = dict(kind="adv_split", seed=6000 + n, S=S)
            j["table"] = dict(splitK=S, reduction=red, custom=cust, algo=int(c["algo"][i]), tile=int(c["tile"][i]),
                              stages=int(c["stages"][i]), workspace=int(c["workspace"][i]))
            jobs.append(j)
    a.out.write_text(json.dumps(jobs))
    print(f"{len(combos)} split combinations ({n_evo} Evo 2) in {len(groups)} groups -> {len(jobs)} jobs; ws "
          f"{sorted({j['ws'] for j in jobs})}")


if __name__ == "__main__":
    main()
