"""Job lists for the live recorders R2-verify, R4 and R5 (CPU; run before the locked GPU run).

R4  For each (fast, S, scheme, out, fmt pair) split combination of the truth table that has no hardware replay
    (156, incl. the 3 S=29 ones) take the smallest-M*N*K row that torch's own descriptor reaches (align none; the
    union table supplies (ws, align none) rows for mixed formats, see R2x) and record it on randn*40 and on
    adversarial data. R4b: the same for the combinations that only the R2 / R2x rows (fp8 outputs, mixed formats) add
    (adversarial data only, to keep the recording small).
R5  300 random in-envelope table shapes at the torch default workspace (1 MiB, align none), mixed out/fast/formats:
    sampled uniformly within each (out, fast, fmt pair) setting's split envelope (M*N <= max split M*N, K >= min
    split K), about 2/3 of them rows that split.
R2v 100 random R2 rows (fp8 outputs; torch-reachable: align none; dscale 1 <=> scale_result passed) + two extra
    shapes (e4m3 out at 224x176x80 and 176x224x80).

Stored subsets: the GPU call uses the full operands; the recording keeps a row/column subset with full K
(hw_shape + sub; the split partition depends on K only). The subset is chosen on CPU (cpu_model) so that every
alternative plan (other scheme, S-1, S+1, contiguous tiles, unsplit, slow restart every 64) changes at least one
stored output bit; jobs where that is impossible are flagged (params.discrimination).

usage: python tools/record/jobs.py {r4,r5,r2v} --out JOBS.json"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # tiny CPU-model calls: OpenMP start-up would dominate

import argparse  # noqa: E402
import json  # noqa: E402
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coverage  # noqa: E402
import cpu_model  # noqa: E402
import live  # noqa: E402
import tables  # noqa: E402

BUDGET = 32768          # (rows + cols) * K bytes of stored operands per job (at least 2 rows + 2 cols)


def plan_alternatives(S: int, scheme: str, fast: bool) -> list[dict]:
    alts = []
    if S > 1:
        if S >= 3:
            alts.append(dict(S=S, scheme="pairwise" if scheme == "serial" else "serial"))
        alts.append(dict(S=S - 1, scheme=scheme if S - 1 > 1 else "none"))
        alts.append(dict(S=S + 1, scheme=scheme))
        alts.append(dict(S=S, scheme=scheme, contiguous=True))
        if S - 1 > 1:
            alts.append(dict(S=1, scheme="none"))
    else:
        alts.append(dict(S=2, scheme="serial"))
        alts.append(dict(S=3, scheme="serial"))
    if not fast:
        alts.append(dict(S=S, scheme=scheme, slow_restart=64))
    return alts


def _out_bits(A, B, job, plan):
    acc = cpu_model.acc(A, B, job["fa"], job["fb"], fast=job["fast"], S=plan["S"], scheme=plan["scheme"],
                        contiguous=plan.get("contiguous", False), slow_restart=plan.get("slow_restart", 128))
    return cpu_model.epilogue_bits(acc, job["scale_a"], job["scale_b"], job["out"], job.get("scale_result"))


def choose_subset(job: dict, S: int, scheme: str, rng: np.random.Generator, discriminate: bool = True) -> dict:
    """Pick sub rows/cols within the byte budget; when discriminate, cover every alternative plan."""
    M, N, K = job["M"], job["N"], job["K"]
    tot = max(4, BUDGET // max(K, 1))
    r = min(M, max(2, min(8, tot // 2)))
    c = min(N, max(2, min(8, tot - r)))
    if not discriminate or K == 0:
        rows = sorted(rng.choice(M, r, replace=False).tolist())
        cols = sorted(rng.choice(N, c, replace=False).tolist())
        return dict(rows=rows, cols=cols, discrimination=None)
    R0 = sorted(rng.choice(M, min(M, 8), replace=False).tolist())
    C0 = sorted(rng.choice(N, min(N, 8), replace=False).tolist())
    A, B = (t.numpy() for t in live.make_inputs(job, R0, C0))
    if job["data"]["kind"] == "adv_split":
        # scale the (small, cancellation-dominated) accumulators into the out dtype's range
        acc = cpu_model.acc(A, B, job["fa"], job["fb"], fast=job["fast"], S=S, scheme=scheme)
        med = float(np.median(np.abs(acc[acc != 0]))) if (acc != 0).any() else 1.0
        target = {"float8_e4m3fn": 24.0, "float8_e5m2": 1500.0}.get(job["out"], 1500.0)
        al = target / med
        job["scale_a"] = float(np.float32(math.sqrt(al) * 1.0123))
        job["scale_b"] = float(np.float32(math.sqrt(al) / 1.0071))
    true = _out_bits(A, B, job, dict(S=S, scheme=scheme))
    alts = plan_alternatives(S, scheme, job["fast"])
    D = np.stack([(_out_bits(A, B, job, p) != true) for p in alts])        # [n_alt, R0, C0]
    best = None
    for i1 in range(len(R0)):
        for i2 in range(i1 + 1, len(R0)) if len(R0) > 1 else [i1]:
            for j1 in range(len(C0)):
                for j2 in range(j1 + 1, len(C0)) if len(C0) > 1 else [j1]:
                    sub = D[:, [i1, i2]][:, :, [j1, j2]]
                    cov = int(sub.any(axis=(1, 2)).sum())
                    score = (cov, int(sub.sum()))
                    if best is None or score > best[0]:
                        best = (score, [i1, i2], [j1, j2])
    (cov, _), ri, ci = best
    ri, ci = sorted(set(ri)), sorted(set(ci))
    extra_r = [i for i in range(len(R0)) if i not in ri][: max(0, r - len(ri))]
    extra_c = [j for j in range(len(C0)) if j not in ci][: max(0, c - len(ci))]
    ri, ci = sorted(ri + extra_r), sorted(ci + extra_c)
    sub = D[:, ri][:, :, ci]
    covered = sub.any(axis=(1, 2))
    return dict(rows=[R0[i] for i in ri], cols=[C0[j] for j in ci],
                discrimination=dict(alternatives=alts, distinguished=[bool(x) for x in covered],
                                    all_distinguished=bool(covered.all()),
                                    n_differing_outputs=[int(x) for x in sub.reshape(len(alts), -1).sum(1)]))


def _job(i, u, rid_prefix, kind, seed, scale_result=None):
    j = dict(id=f"{rid_prefix}{i:04d}_{kind}", M=int(u["M"][i]), N=int(u["N"][i]), K=int(u["K"][i]),
             fa=tables.FMT_NAMES[int(u["fa"][i])], fb=tables.FMT_NAMES[int(u["fb"][i])],
             out=tables.OUT_NAMES[int(u["out"][i])], fast=bool(u["fast"][i]), ws=int(u["ws"][i]),
             data=dict(kind=kind, seed=int(seed)), scale_result=scale_result,
             table=dict(source=tables.SOURCES[int(u["src"][i])][0], align=int(u["align"][i]),
                        **{k: int(u[k][i]) for k in ("algo", "tile", "stages", "splitK", "reduction", "swizzle",
                                                     "custom", "workspace", "status")}))
    j["scale_a"], j["scale_b"] = live.default_scales(j)
    return j


def _with_subset(j, rng, discriminate=True, tries=3):
    S = j["table"]["splitK"]
    scheme = "none" if S <= 1 else ("pairwise" if j["table"]["custom"] == 1 else "serial")
    best = None
    for t in range(tries):
        if t:
            j["data"]["seed"] += 7919
        s = choose_subset(j, S, scheme, rng, discriminate)
        if s["discrimination"] is None or s["discrimination"]["all_distinguished"]:
            best = (s, dict(j["data"]))
            break
        if best is None or sum(s["discrimination"]["distinguished"]) > sum(best[0]["discrimination"]["distinguished"]):
            best = (s, dict(j["data"]))
    s, data = best
    j["data"] = data
    j["sub"] = dict(rows=s["rows"], cols=s["cols"])
    j["discrimination"] = s["discrimination"]
    return j


def build_r4(seed=4):
    rng = np.random.default_rng(seed)
    u = tables.load_union()
    hw = coverage.hw_combos()
    torch_ok = (u["align"] == 0) & (u["splitK"] > 1) & (u["status"] == 0)
    idx = np.nonzero(torch_ok)[0]
    mnk = u["M"][idx] * u["N"][idx] * u["K"][idx]
    best = {}
    for i, v in zip(idx, mnk):
        k = tables.combo_key(u, i)
        if k in hw:
            continue
        cur = best.get(k)
        cand = (int(v), int(u["ws"][i]), int(u["src"][i]), int(i))
        if cur is None or cand < cur:
            best[k] = cand
    truth_unv = {k for k in coverage.table_combos(u, u["src"] == 0) if k not in hw}
    missing_torch = sorted(k for k in truth_unv if k not in best)
    jobs = []
    for n, (k, (_, _, _, i)) in enumerate(sorted(best.items())):
        part = "r4" if k in truth_unv else "r4b"
        kinds = ("randn40", "adv_split") if part == "r4" else ("adv_split",)
        for kind in kinds:
            j = _job(i, u, f"{part}_", kind, seed=1000 + n)
            if kind == "adv_split":
                j["data"]["S"] = int(u["splitK"][i])
            j["id"] = f"{part}_{n:03d}_{kind}"
            j["combo"] = list(k)
            j["part"] = part
            jobs.append(_with_subset(j, rng))
    return jobs, dict(n_combos=len(best), n_truth_unverified=len(truth_unv), missing_torch_reachable=missing_torch)


def envelopes(u, mask):
    """{(ws, out, fast, fa, fb): (max split M*N, min split K)} over rows in mask."""
    env = {}
    idx = np.nonzero(mask & (u["splitK"] > 1))[0]
    for i in idx:
        k = (int(u["ws"][i]), int(u["out"][i]), int(u["fast"][i]), int(u["fa"][i]), int(u["fb"][i]))
        mn, K = int(u["M"][i] * u["N"][i]), int(u["K"][i])
        e = env.get(k)
        env[k] = (mn, K) if e is None else (max(e[0], mn), min(e[1], K))
    return env


def build_r5(n_total=300, seed=5):
    rng = np.random.default_rng(seed)
    u = tables.load_union()
    base = (u["ws"] == 1 << 20) & (u["align"] == 0) & (u["status"] == 0)
    env = envelopes(u, base)
    settings = sorted(env)
    per = -(-n_total // len(settings))
    jobs = []
    for sk in settings:
        ws, o, f, fa, fb = sk
        mx, mk = env[sk]
        sel = base & (u["out"] == o) & (u["fast"] == f) & (u["fa"] == fa) & (u["fb"] == fb) & \
            (u["M"] * u["N"] <= mx) & (u["K"] >= mk)
        idx = np.nonzero(sel)[0]
        sp = idx[u["splitK"][idx] > 1]
        un = idx[u["splitK"][idx] <= 1]
        n_sp = min(len(sp), -(-2 * per // 3))
        pick = list(rng.choice(sp, n_sp, replace=False)) + list(rng.choice(un, min(len(un), per - n_sp),
                                                                          replace=False))
        for i in pick:
            jobs.append((sk, int(i)))
    rng.shuffle(jobs)
    jobs = jobs[:n_total]
    out = []
    for n, (sk, i) in enumerate(sorted(jobs, key=lambda t: t[1])):
        S = int(u["splitK"][i])
        kind = ("adv_split" if S > 1 else "adv") if n % 2 else "randn40"
        j = _job(i, u, "r5_", kind, seed=5000 + n)
        if kind == "adv_split":
            j["data"]["S"] = S
        j["id"] = f"r5_{n:03d}_{kind}"
        j["envelope"] = dict(max_split_MN=env[sk][0], min_split_K=env[sk][1])
        out.append(_with_subset(j, rng, discriminate=j["table"]["splitK"] > 1, tries=1))
    return out, dict(n_settings=len(settings), per_setting=per,
                     envelopes={"|".join(map(str, k)): v for k, v in env.items()})


def build_r2v(n=100, seed=2):
    rng = np.random.default_rng(seed)
    u = tables.load_union(sources=(1,))
    ok = (u["align"] == 0) & (u["status"] == 0)
    idx = np.nonzero(ok)[0]
    sp = idx[u["splitK"][idx] > 1]
    un = idx[u["splitK"][idx] <= 1]
    pick = list(rng.choice(sp, n // 2, replace=False)) + list(rng.choice(un, n - n // 2, replace=False))
    jobs = []
    for m, i in enumerate(sorted(int(x) for x in pick)):
        S = int(u["splitK"][i])
        kind = ("adv_split" if S > 1 else "adv") if m % 2 else "randn40"
        sr = float(np.float32(0.8125 if m % 3 == 0 else 1.3)) if m % 2 == 0 else None
        j = _job(i, u, "r2v_", kind, seed=2000 + m, scale_result=sr)
        if kind == "adv_split":
            j["data"]["S"] = S
        j["id"] = f"r2v_{m:03d}_{kind}"
        jobs.append(_with_subset(j, rng, discriminate=j["table"]["splitK"] > 1, tries=1))
    gap = (u["K"] == 80) & (u["out"] == 3) & (u["align"] == 0) & (u["ws"] == 1 << 20) & \
        (((u["M"] == 224) & (u["N"] == 176)) | ((u["M"] == 176) & (u["N"] == 224)))
    for m, i in enumerate(np.nonzero(gap)[0]):
        j = _job(int(i), u, "gap17_", "randn40", seed=2500 + m)
        j["id"] = f"gap17_{m:02d}"
        j["sub"] = None          # small: store everything
        j["discrimination"] = None
        jobs.append(j)
    return jobs, dict(n_random=n, n_gap17=int(gap.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("item", choices=["r4", "r5", "r2v"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs, meta = {"r4": build_r4, "r5": build_r5, "r2v": build_r2v}[a.item]()
    for j in jobs:
        j["est_bytes"] = (len(j["sub"]["rows"]) + len(j["sub"]["cols"])) * j["K"] if j.get("sub") else \
            (j["M"] + j["N"]) * j["K"]
    meta["n_jobs"] = len(jobs)
    meta["est_operand_bytes"] = int(sum(j["est_bytes"] for j in jobs))
    meta["not_fully_discriminating"] = [j["id"] for j in jobs if j.get("discrimination")
                                        and not j["discrimination"]["all_distinguished"]]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(jobs, indent=0))
    Path(a.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({k: (v if not isinstance(v, (list, dict)) or len(str(v)) < 400 else f"<{len(v)} items>")
                      for k, v in meta.items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
