"""Open the sealed held-out part p4 and test the committed envelope (results/analysis/envelope_rule.json, fitted by
fit_envelope.py on p0-p3) against it (plan section 3). Run once; the result is final: a failing envelope is shipped
as unvalidated (with a warning) or not at all, never refitted on p4.

Refuses to run unless
  - fit_envelope.py, this script, the rule, the training data (p1-p3, the probe's p0) and the p4 shape generator
    (scripts/sweep/make_shapes.py, scripts/shapes/index.json) are committed and unmodified (git);
  - the rule is what fit_envelope.py fits from that data (the --check of fit_envelope.py);
  - the p4 files have the sha256 recorded when they were sealed (plan section 9);
  - their rows are make_shapes.p4()'s, row for row (which also tells a random row from a model row);
  - results/analysis/eval_sealed.json does not exist yet.
Criteria:
  - envelope (profiles hopper-h100pcie-torch, 1 MiB, and hopper-h100pcie-te, 32 MiB): every row with status 0,
    K_MIN <= K <= K_MAX and M * N > B(setting, K) has numSplitsK = 1. Rows the heuristic rejects (status != 0)
    are not counted;
  - workspace 0 (hopper-h100pcie-ws0): every status-0 row has numSplitsK = 1.
Strata (rows, tested rows, splits among them): profile x K band x (K % 128 == 0) x source (random rows, model rows).
The random rows start at M*N = R / 4 = 30,720; between the floor (15,360) and 30,720 only model rows test the rule.
usage: eval_sealed.py P4_DIR      (the p4_sealed directory of the sweep's output; writes eval_sealed.json)"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sweep"))
import fit_envelope as F  # noqa: E402
import make_shapes  # noqa: E402

OUT = F.RES / "analysis" / "eval_sealed.json"
FORMAT = "mmaemu-h100-splitk-envelope-eval/1"
SEAL = {   # plan section 9: sha256 and rows of the sealed files
    "torch_a0_c00.npz": ("d58775569db947f02388b5986c85f381470a306059b4eac0d9d848d908cfb3ca", 223_940, 1 << 20),
    "te_a256_c00.npz": ("b3471ea5fabedba5e857ef89a8ab4a22fed091d3591dcf9afdaf9c3373ca0c24", 223_940, 32 << 20),
    "ws0_a0_c00.npz": ("c95cc465cea87426d9ec99c7f0f068ac811566d9180169218575c44822a9a5fa", 50_000, 0),
}
N_RANDOM = 450_000                       # the first rows of make_shapes.p4() are random, the rest model rows
LOW = F.R // 4                           # the random rows' smallest M*N
BANDS = [(1024, 5120), (5121, 16384), (16385, 65536), (65537, 4194304)]
COLS = ("M", "N", "K", "ws", "align", "out", "fa", "fb", "fast")
LIST = 50                                # rows listed per failure


def git(*a) -> str:
    return subprocess.run(["git", "-C", str(F.REPO), *a], check=True, capture_output=True, text=True).stdout


def provenance() -> dict:
    paths = [HERE / "fit_envelope.py", HERE / "eval_sealed.py", F.RULE, F.P0, HERE.parent / "sweep" / "make_shapes.py",
             HERE.parent / "shapes" / "index.json"] + [f for p in ("p1", "p2", "p3") for f in F.inputs()[p]]
    rel = [str(p.relative_to(F.REPO)) for p in paths]
    tracked = set(git("ls-files", "--", *rel).split("\n"))
    if missing := [r for r in rel if r not in tracked]:
        raise SystemExit(f"not committed: {missing}")
    if dirty := git("status", "--porcelain", "--", *rel).strip():
        raise SystemExit(f"modified since the commit:\n{dirty}")
    if F.dump(F.fit()) != F.RULE.read_text():
        raise SystemExit(f"{F.RULE}: not what fit_envelope.py fits from the committed data")
    return dict(commit=git("rev-parse", "HEAD").strip(), rule_sha256=F.sha256(F.RULE),
                scripts={r: F.sha256(p) for r, p in zip(rel[:2], paths[:2])})


def expected_rows():
    """make_shapes.p4() as the sealed shape list (sha256 of index.json), with a random / model flag"""
    rows = make_shapes.p4()
    sha = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
    want = json.loads((HERE.parent / "shapes" / "index.json").read_text())["parts"]["p4"]["sha256"]
    if sha != want:
        raise SystemExit(f"make_shapes.p4() sha256 {sha} is not the sealed list's {want}")
    return rows


def load(p4: Path, rows) -> dict:
    out = {}
    for name, (sha, n, ws) in SEAL.items():
        f = p4 / name
        if F.sha256(f) != sha:
            raise SystemExit(f"{f}: sha256 is not the sealed one")
        z = np.load(f)
        cols = {k: z[k].astype(np.int64) for k in z.files}
        mine = [(i, r) for i, r in enumerate(rows) if r[3] == ws]
        if len(cols["M"]) != n or len(mine) != n:
            raise SystemExit(f"{f}: {len(cols['M'])} rows, the seal says {n}, the shape list {len(mine)}")
        want = {"M": [r[0] for _, r in mine], "N": [r[1] for _, r in mine], "K": [r[2] for _, r in mine],
                "ws": [r[3] for _, r in mine], "align": [r[4] for _, r in mine],
                "out": [F.OUTS.index(r[5]) for _, r in mine], "fa": [F.FMTS.index(r[6]) for _, r in mine],
                "fb": [F.FMTS.index(r[7]) for _, r in mine], "fast": [r[8] for _, r in mine]}
        for c in COLS:
            if not np.array_equal(cols[c], np.asarray(want[c], dtype=np.int64)):
                raise SystemExit(f"{f}: column {c} is not the sealed shape list's")
        cols["random"] = np.asarray([i < N_RANDOM for i, _ in mine])
        out[name] = cols
    return out


def row(c, i, B=None) -> dict:
    d = {k: int(c[k][i]) for k in ("M", "N", "K", "splitK", "custom", "reduction", "algo", "tile")}
    d.update(MN=d["M"] * d["N"], out=F.OUTS[int(c["out"][i])], fast=bool(c["fast"][i]),
             fmt_a=F.FMTS[int(c["fa"][i])], fmt_b=F.FMTS[int(c["fb"][i])])
    if B is not None:
        d["B"] = int(B[i])
    return d


def evaluate(cols: dict, env: F.Envelope) -> dict:
    B = env.bounds(cols)
    mn, K = cols["M"] * cols["N"], cols["K"]
    ok = cols["status"] == 0
    split = ok & (cols["splitK"] > 1)
    tested = ok & (mn > B)
    fail = tested & split
    strata = []
    for lo, hi in BANDS:
        for k128 in (True, False):
            for src in (True, False):
                m = (K >= lo) & (K <= hi) & ((K % 128 == 0) == k128) & (cols["random"] == src)
                strata.append(dict(K_band=[lo, hi], K_multiple_of_128=k128, source="random" if src else "model",
                                   rows=int(m.sum()), status_nonzero=int((m & ~ok).sum()),
                                   tested=int((m & tested).sum()), tested_MN_le_30720=int((m & tested & (mn <= LOW)).sum()),
                                   split_tested=int((m & fail).sum()), split_all=int((m & split).sum())))
    ratio = np.where(split, mn / B, 0.0)          # B = inf: no claim, ratio 0
    i = int(np.argmax(ratio)) if split.any() else None
    return dict(rows=int(len(K)), status_nonzero=int((~ok).sum()), tested=int(tested.sum()),
                untested_status0=int((ok & ~tested).sum()), split_all=int(split.sum()), split_tested=int(fail.sum()),
                passed=bool(not fail.any()),
                closest_split=None if i is None else dict(row(cols, i, B), MN_over_B=round(float(ratio[i]), 4)),
                failures=[row(cols, j, B) for j in np.flatnonzero(fail)[:LIST]], strata=strata)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("p4", type=Path)
    a = ap.parse_args()
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: the sealed evaluation runs once")
    prov = provenance()
    data = load(a.p4, expected_rows())
    env = F.Envelope(json.loads(F.RULE.read_text()))
    rep = dict(format=FORMAT, date=time.strftime("%Y-%m-%d %H:%M:%S %Z"), provenance=prov,
               p4={n: dict(sha256=s, rows=r) for n, (s, r, _w) in SEAL.items()},
               python=platform.python_version(), numpy=np.__version__)
    for name, prof in (("torch_a0_c00.npz", "hopper-h100pcie-torch"), ("te_a256_c00.npz", "hopper-h100pcie-te")):
        rep[prof] = evaluate(data[name], env)
    w = data["ws0_a0_c00.npz"]
    ok = w["status"] == 0
    sp = ok & (w["splitK"] > 1)
    rep["hopper-h100pcie-ws0"] = dict(rows=int(len(ok)), status_nonzero=int((~ok).sum()), status0=int(ok.sum()),
                                      split=int(sp.sum()), passed=bool(not sp.any()),
                                      failures=[row(w, j) for j in np.flatnonzero(sp)[:LIST]])
    rep["envelope_validated"] = rep["hopper-h100pcie-torch"]["passed"] and rep["hopper-h100pcie-te"]["passed"]
    rep["ws0_validated"] = rep["hopper-h100pcie-ws0"]["passed"]
    OUT.write_text(json.dumps(rep, indent=1) + "\n")
    for prof in ("hopper-h100pcie-torch", "hopper-h100pcie-te"):
        r = rep[prof]
        print(f"{prof}: rows {r['rows']}, status!=0 {r['status_nonzero']}, tested {r['tested']} "
              f"(untested status 0: {r['untested_status0']}), splits {r['split_all']}, splits above the bound "
              f"{r['split_tested']}: {'PASS' if r['passed'] else 'FAIL'}; closest split {r['closest_split']}")
        for s in r["strata"]:
            print(f"   K {s['K_band'][0]:>7}-{s['K_band'][1]:<7} %128==0 {s['K_multiple_of_128']!s:<5} "
                  f"{s['source']:<6} rows {s['rows']:>6} tested {s['tested']:>6} (M*N<=30720: "
                  f"{s['tested_MN_le_30720']:>4}) splits {s['split_all']:>5} above bound {s['split_tested']}")
    r = rep["hopper-h100pcie-ws0"]
    print(f"hopper-h100pcie-ws0: status 0 {r['status0']}, split {r['split']}: {'PASS' if r['passed'] else 'FAIL'}")
    print(f"envelope_validated {rep['envelope_validated']}, ws0_validated {rep['ws0_validated']}; "
          f"commit {prov['commit'][:12]}; wrote {OUT.relative_to(F.REPO)}")
    sys.exit(0 if rep["envelope_validated"] and rep["ws0_validated"] else 2)


if __name__ == "__main__":
    main()
