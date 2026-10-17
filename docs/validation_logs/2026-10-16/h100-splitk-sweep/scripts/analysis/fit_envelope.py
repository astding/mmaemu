"""Fit the off-grid split-K envelope of heur_h100pcie_lt12.8.3.14 on the sweep's training data, before the sealed part
p4 is opened (plan section 3). Training data: p1 (envelope grid), p2 (K dense), p3 (model rows) of this sweep, and
p0, the probe's grid (docs/validation_logs/2026-10-15/h100-probe/results/m6_heur/heur.npz, which this sweep
re-recorded byte for byte: results/p0/drift.json).

Rule, per setting s = (workspace, out, fast, fmt_a, fmt_b) of the two table profiles (1 MiB: hopper-h100pcie-torch,
32 MiB: hopper-h100pcie-te), R = 122,880, F1 = the p1 levels f = 2^(i/16), i = -64..80, and 48, 64, 128:
  - at each p1 grid K (K_GRID, 33 values): lo = the largest split M*N of p1 (status 0, numSplitsK > 1, 0 if none);
    basis = ceil(f * R) for the smallest f of F1 with f * R > lo: the next p1 level, every row of which is unsplit
    at this K. No level above lo: no claim from this grid K on;
  - p0, p2 and p3 add raw lower bounds: the M*N of each split row counts at K_up, the smallest grid K >= its K;
  - v(K) = max(basis, raw lo); B_s(K) = MARGIN * the largest v over the grid K <= K_up(K), at least FLOOR
    (MARGIN * R / 16, the lowest p1 level).
  A query (M, N, K) of setting s with K_MIN <= K <= K_MAX and M * N > B_s(K) is claimed unsplit (S = 1).
Assumptions (the rule rests on them; p4 tests them, nothing proves them):
  1. at one K and setting, no split far above the largest split M*N seen. The split depends on M and N
     separately, not on M*N alone, so p1's 24 M values can miss a larger split M*N: p3's dense M found one at
     1.12 times the p1 basis (M = 80, N = 2048, K = 8192, the TE fast settings; it enters the bound as raw lo).
     MARGIN is what covers such an excess. Dense M (p0, p3) exists only at K 1920, 2048, 3584, 4096, 5120, 8192,
     11008, 11264, 14336, 18944, 22528 and 28672, with few N; at other K the excess is unmeasured;
  2. the split region does not shrink as K grows: a K between grid points takes the next grid K's bound, and every
     grid K takes the largest bound at or below it;
  3. alignment does not matter (p3: no difference between the two alignments of either profile), so the bound
     has no alignment;
  4. the shapes tested: M 1..16,384, N 16..2^20 (p1), model rows to M = 16,384 (p3); a larger M or N is
     extrapolated in M*N;
  5. it holds for this H100 PCIe (114 SMs), driver 570.148.08, cuBLASLt 12.8.3.14, torch 2.7.1's descriptor.
Workspace 0 (hopper-h100pcie-ws0) is not part of the rule: there mmaemu takes S = 1 for every shape (p1: no split
at workspace 0; p4 checks it). Other workspaces: no claim.

usage: fit_envelope.py            # write results/analysis/envelope_rule.json (refuses to overwrite)
       fit_envelope.py --check    # refit and compare with the committed rule (exit 1 on a difference)"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LOG = HERE.parents[1]                        # docs/validation_logs/2026-10-16/h100-splitk-sweep
REPO = HERE.parents[5]
RES = LOG / "results"
P0 = REPO / "docs" / "validation_logs" / "2026-10-15" / "h100-probe" / "results" / "m6_heur" / "heur.npz"
P0_SHA = "4f349703def9b6f0d25cd6b41a622fcc86fad2a870d173c214d8fdea80be031d"
RULE = RES / "analysis" / "envelope_rule.json"
FORMAT = "mmaemu-h100-splitk-envelope/1"
TABLE_ID = "heur_h100pcie_lt12.8.3.14"
R = 122_880
MARGIN = 2
F1 = [2.0 ** (i / 16) for i in range(-64, 81)] + [48.0, 64.0, 128.0]
LEVELS = [math.ceil(f * R) for f in F1]
FLOOR = MARGIN * LEVELS[0]                   # 15,360
K_GRID = [1024, 1040, 1536, 2048, 2064, 3072, 3600, 4096, 5120, 5744, 6144, 8192, 8208, 10240, 12288, 13840, 14336,
          16384, 16400, 20480, 24576, 28672, 32768, 49152, 65536, 65552, 131072, 262144, 524288, 1048576, 1048592,
          2097152, 4194304]
K_MIN, K_MAX = K_GRID[0], K_GRID[-1]
PROFILES = {1 << 20: "hopper-h100pcie-torch", 32 << 20: "hopper-h100pcie-te"}
OUTS = ["float32", "bfloat16", "float16", "float8_e4m3fn", "float8_e5m2"]
FMTS = ["e4m3", "e5m2"]
KEY = ("ws", "out", "fast", "fa", "fb")
ASSUMPTIONS = re.split(r"\s+(?=\d\. )", " ".join(
    __doc__.split("nothing proves them):")[1].split("Workspace 0")[0].split()))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def inputs() -> dict:
    """{name: npz path} of the training data, p0 first"""
    out = {"p0": [P0]}
    for p in ("p1", "p2", "p3"):
        out[p] = sorted((RES / p).glob("*_a*_c*.npz"))
    return out


def code(cols: dict) -> np.ndarray:
    """one integer per setting (ws, out, fast, fa, fb)"""
    c = [np.asarray(cols[k], dtype=np.int64) for k in KEY]
    return (((c[0] * 5 + c[1]) * 2 + c[2]) * 2 + c[3]) * 2 + c[4]


def load(files) -> dict:
    zs = [np.load(f) for f in files]
    cols = {k: np.concatenate([z[k].astype(np.int64) for z in zs]) for k in ("M", "N", "K", "status", "splitK") + KEY}
    keep = np.isin(cols["ws"], list(PROFILES))
    cols = {k: v[keep] for k, v in cols.items()}
    cols["code"] = code(cols)
    return cols


def k_up(K: np.ndarray) -> np.ndarray:
    """index of the smallest grid K >= K (len(K_GRID) when K > K_MAX)"""
    return np.searchsorted(np.asarray(K_GRID), K, side="left")


def setting_name(key) -> dict:
    ws, o, f, fa, fb = (int(x) for x in key)
    return dict(profile=PROFILES[ws], ws=ws, out=OUTS[o], fast=bool(f), fmt_a=FMTS[fa], fmt_b=FMTS[fb])


def fit() -> dict:
    files = inputs()
    if sha256(P0) != P0_SHA:
        raise SystemExit(f"{P0}: sha256 is not the probe's")
    data = {p: load(fs) for p, fs in files.items()}
    p1 = data["p1"]
    keys = np.unique(np.stack([p1[k] for k in KEY], 1), axis=0)
    settings, outside = [], {}
    for key in keys:
        c = code(dict(zip(KEY, key)))
        sel = {p: d["code"] == c for p, d in data.items()}
        grid = []
        for i, Kg in enumerate(K_GRID):
            m = sel["p1"] & (p1["K"] == Kg)
            if not m.any():
                raise SystemExit(f"p1 has no row at K {Kg} for {setting_name(key)}")
            sp = m & (p1["status"] == 0) & (p1["splitK"] > 1)
            lo = int((p1["M"][sp] * p1["N"][sp]).max()) if sp.any() else 0
            j = bisect.bisect_right(LEVELS, lo)
            grid.append(dict(K=Kg, p1_rows=int(m.sum()), p1_status0=int((m & (p1["status"] == 0)).sum()),
                             p1_split=int(sp.sum()), lo=lo, basis=LEVELS[j] if j < len(LEVELS) else None,
                             raw_lo=0, raw_by_part={}))
        for p in ("p0", "p2", "p3"):
            d = data[p]
            sp = sel[p] & (d["status"] == 0) & (d["splitK"] > 1)
            mn, ku, K = d["M"][sp] * d["N"][sp], k_up(d["K"][sp]), d["K"][sp]
            bad = (K < K_MIN) | (ku >= len(K_GRID))
            if bad.any():
                outside[p] = outside.get(p, 0) + int(bad.sum())
            for i in np.unique(ku[~bad]):
                v = int(mn[~bad][ku[~bad] == i].max())
                g = grid[int(i)]
                g["raw_by_part"][p] = v
                g["raw_lo"] = max(g["raw_lo"], v)
        run = 0
        for g in grid:
            v = None if g["basis"] is None or run is None else max(g["basis"], g["raw_lo"])
            run = None if v is None else max(run, v)
            g["B"] = None if run is None else max(FLOOR, MARGIN * run)
            g["raw_above_basis"] = g["basis"] is not None and g["raw_lo"] >= g["basis"]
        settings.append(dict(setting_name(key), grid=grid))
    summary = {}
    for ws, prof in PROFILES.items():
        ss = [s for s in settings if s["ws"] == ws]
        rows = []
        for i, Kg in enumerate(K_GRID):
            bs = [s["grid"][i]["B"] for s in ss]
            rows.append(dict(K=Kg, B_min=None if None in bs else min(bs), B_max=None if None in bs else max(bs),
                             no_claim=bs.count(None), lo_max=max(s["grid"][i]["lo"] for s in ss),
                             raw_lo_max=max(s["grid"][i]["raw_lo"] for s in ss)))
        summary[prof] = rows
    return dict(
        format=FORMAT, table_id=TABLE_ID,
        rule=dict(text=__doc__.split("usage:")[0].strip(), assumptions=ASSUMPTIONS, R=R, margin=MARGIN, floor=FLOOR,
                  F1="2^(i/16) for i = -64..80, then 48, 64, 128", levels=LEVELS, K_grid=K_GRID, K_min=K_MIN,
                  K_max=K_MAX, claim="M * N > B(setting, K) and K_min <= K <= K_max: S = 1",
                  K_lookup="B at the smallest grid K >= K"),
        inputs={p: [dict(path=str(f.relative_to(REPO)), sha256=sha256(f)) for f in fs] for p, fs in files.items()},
        training_rows={p: int(len(d["M"])) for p, d in data.items()},
        split_rows_outside_K_range=outside,
        settings=settings, summary=summary)


def dump(rule: dict) -> str:
    return json.dumps(rule, indent=1) + "\n"


class Envelope:
    """B_s(K) lookups of a rule (the claim of the module docstring)"""

    def __init__(self, rule: dict):
        self.grid = np.asarray(rule["rule"]["K_grid"])
        self.kmin, self.kmax = rule["rule"]["K_min"], rule["rule"]["K_max"]
        self.B = {}
        for s in rule["settings"]:
            key = dict(ws=s["ws"], out=OUTS.index(s["out"]), fast=int(s["fast"]), fa=FMTS.index(s["fmt_a"]),
                       fb=FMTS.index(s["fmt_b"]))
            self.B[int(code(key))] = np.asarray([np.inf if g["B"] is None else g["B"] for g in s["grid"]], float)

    def bounds(self, cols: dict) -> np.ndarray:
        """B per row (inf: no claim) of the int columns K ws out fast fa fb"""
        K = np.asarray(cols["K"], dtype=np.int64)
        out = np.full(len(K), np.inf)
        inr = (K >= self.kmin) & (K <= self.kmax)
        idx = np.minimum(np.searchsorted(self.grid, K, side="left"), len(self.grid) - 1)
        c = code(cols)
        for k in np.unique(c):
            b = self.B.get(int(k))
            if b is not None:
                m = (c == k) & inr
                out[m] = b[idx[m]]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    rule = dump(fit())
    if a.check:
        ok = RULE.exists() and RULE.read_text() == rule
        print(f"{RULE.relative_to(REPO)}: {'equal to a fresh fit' if ok else 'DIFFERS from a fresh fit'}")
        sys.exit(0 if ok else 1)
    if RULE.exists():
        raise SystemExit(f"refusing to overwrite {RULE}")
    RULE.parent.mkdir(parents=True, exist_ok=True)
    RULE.write_text(rule)
    r = json.loads(rule)
    for prof, rows in r["summary"].items():
        print(prof)
        for g in rows:
            print(f"  K {g['K']:>8}: lo_max {g['lo_max']:>9}  raw_lo_max {g['raw_lo_max']:>9}  "
                  f"B {g['B_min']!s:>9} .. {g['B_max']!s:>9}  no claim {g['no_claim']}")
    print("split rows outside the K range:", r["split_rows_outside_K_range"])
    print("raw lo >= basis:", sum(g["raw_above_basis"] for s in r["settings"] for g in s["grid"]))


if __name__ == "__main__":
    main()
