"""Summary of a sweep run (CPU): drift verdict; per part and profile the queries, split rows, largest split M*N (all K
and per K band), smallest split K, S values, reduction schemes, customOptions, split rows at K % 128 != 0; how far
splits reach past the shipped table's per-setting largest split M*N; the live calls (errors, canary, CPU = GPU
inputs, logged config = queried row, pass 1 = pass 2 sha256). The sealed part p4 is only counted (rows, sha256).
usage: summarize.py SWEEP_OUT_ROOT  (prints; run_all.sh writes it to summary.txt and summary.json)"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import sc

TABLE = sc.SRC / "mmaemu" / "platform" / "data" / "heur_h100pcie_lt12.8.3.14.json"
BANDS = [(0, 16384), (16385, 65536), (65537, 1 << 23)]


def table_max():
    meta = json.loads(TABLE.read_text())
    return {(s["ws"], sc.OUTC[s["out"]], int(s["fast"]), sc.FMTC[s["fmt_a"]], sc.FMTC[s["fmt_b"]]):
            s["max_split_MN"] for s in meta["settings"]}


def part_summary(c, tmax):
    ok = c["status"] == 0
    sp = ok & (c["splitK"] > 1)
    mn = c["M"] * c["N"]
    s = dict(queries=int(len(c["M"])), status_nonzero=int((~ok).sum()), split_rows=int(sp.sum()),
             max_split_MN=int(mn[sp].max()) if sp.any() else None,
             min_split_K=int(c["K"][sp].min()) if sp.any() else None,
             S=sorted(int(x) for x in np.unique(c["splitK"][sp])),
             reduction=sorted(int(x) for x in np.unique(c["reduction"][sp])),
             custom=sorted(int(x) for x in np.unique(c["custom"][sp])),
             split_K_not_128=int((sp & (c["K"] % 128 != 0)).sum()))
    s["max_split_MN_by_K"] = {f"{lo}-{hi}": (int(mn[sp & (c["K"] >= lo) & (c["K"] <= hi)].max())
                                            if (sp & (c["K"] >= lo) & (c["K"] <= hi)).any() else None)
                              for lo, hi in BANDS}
    # factor over the shipped table's largest split M*N of the same setting (None: the setting never split there)
    keys = list(zip(c["ws"].tolist(), c["out"].tolist(), c["fast"].tolist(), c["fa"].tolist(), c["fb"].tolist()))
    idx = np.nonzero(sp)[0]
    fac, never = [], 0
    for i in idx:
        t = tmax.get(keys[i])
        if t:
            fac.append(mn[i] / t)
        else:
            never += 1
    s["largest_factor_over_table"] = round(max(fac), 3) if fac else None
    s["split_rows_above_2x_table"] = int(sum(f > 2 for f in fac))
    s["split_rows_in_settings_unsplit_in_table"] = int(never)
    return s


def live_summary(root: Path):
    out = {}
    for d in sorted((root / "p5_live").glob("*/")) if (root / "p5_live").exists() else []:
        for js in sorted(d.glob("*_ws*.json")):
            if js.name.endswith(".jobs.json"):
                continue
            meta = json.loads(js.read_text())
            jobs = {j["id"]: j for j in json.loads(js.with_name(js.name.replace(".json", ".jobs.json")).read_text())}
            res = meta["results"]
            cfg_eq = cfg_n = 0
            for r in res:
                t = jobs[r["id"]].get("table")
                if t and r.get("lt"):
                    cfg_n += 1
                    lt = r["lt"]
                    cfg_eq += (lt["algo"] == t["algo"] and lt["numSplitsK"] == t["splitK"]
                               and lt["customOption"] == t["custom"]
                               and (lt.get("tile_id") is None or lt["tile_id"] == t["tile"])
                               and (lt.get("stages_id") is None or lt["stages_id"] == t["stages"]))
            out[f"{d.name}/{js.stem}"] = dict(
                jobs=len(res), errors=sum(r["error"] is not None for r in res),
                no_log=sum(r["n_lt_calls"] == 0 for r in res), canary_fail=sum(not r.get("canary_ok") for r in res),
                inputs_cpu_ne_gpu=sum(r.get("inputs_cpu_eq_gpu") is False for r in res),
                config_equals_query=f"{cfg_eq}/{cfg_n}", seconds=round(sum(r["seconds"] for r in res), 1),
                sha={r["id"]: r.get("full_sha256") for r in res})
    # determinism across processes: p1 vs p2 by job id
    by = {}
    for k, v in out.items():
        tag = k.split("/")[1].split("_")[0]          # p1 / p2 pass
        for jid, h in v["sha"].items():
            by.setdefault(jid, {})[tag] = h
    p12 = [x for x in by.values() if "p1" in x and "p2" in x]
    det = dict(compared=len(p12), differ=sum(x["p1"] != x["p2"] for x in p12))
    for v in out.values():
        v.pop("sha")
    return dict(chunks=out, determinism=det)


def main():
    root = Path(sys.argv[1])
    tmax = table_max()
    rep = {}
    if (root / "p0" / "drift.json").exists():
        d = json.loads((root / "p0" / "drift.json").read_text())
        rep["drift"] = {k: d.get(k) for k in ("verdict", "rows", "column_diffs", "hardware_mismatch", "driver")}
    for p in ("p1", "p2", "p3"):
        for name, files in sc.part_files(root / p).items():
            rep[f"{p}/{name}"] = part_summary(sc.load_cols(files), tmax) | dict(chunks=len(files))
    for name, files in sc.part_files(root / "p4_sealed").items():
        rep[f"p4_sealed/{name}"] = dict(rows=sum(int(len(np.load(f)["M"])) for f in files),
                                        sha256={f.name: sc.sha256(f) for f in files}, note="sealed")
    if (root / "p5_jobs" / "index.json").exists():
        rep["p5_jobs"] = json.loads((root / "p5_jobs" / "index.json").read_text())
    rep["p5_live"] = live_summary(root)
    (root / "summary.json").write_text(json.dumps(rep, indent=1) + "\n")
    for k, v in rep.items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    main()
