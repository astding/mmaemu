"""Summary of a probe run directory (run on the host at the end, and again offline): step return codes and times,
per recorder the calls / errors / non-deterministic calls / calls without a cuBLASLt log, the distinct logged
configurations and kernels, run 1 vs run 2 of every live job, and the data size.
usage: summarize.py OUT_DIR [--truth TRUTH_JSON]   (--truth: compare M2 with the RTX 4060 recording, dry runs)"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def cfg_key(lt: dict) -> str:
    return (f"algo={lt['algo']} tile={lt['tile']} stages={lt['stages']} splitK={lt['numSplitsK']} "
            f"red={lt['reductionScheme']} custom={lt['customOption']} swz={lt['ctaSwizzling']} "
            f"ws={lt['workspace_bytes']} fast={int(lt['fast'])} D={lt['Ddesc']['type']}")


def rec_summary(p: Path) -> dict:
    d = json.loads(p.read_text())
    cfgs, kern = collections.Counter(), collections.Counter()
    multi_lt = 0
    for r in d["calls"]:
        r0 = r["runs"][0]
        for lt in r0["lt"]:
            cfgs[cfg_key(lt)] += 1
        multi_lt += len(r0["lt"]) > 1
        for k in r0.get("kernels", []):
            kern[k[:160]] += 1
    return dict(summary=d["summary"], gpu=d["hardware"].get("gpu"), sm_count=d["hardware"].get("sm_count"),
                calls_with_several_lt_calls=multi_lt, configs=dict(cfgs.most_common()),
                kernels=dict(kern.most_common()), extra_keys=sorted(d.get("extra", {})))


def worker_summary(files: list[Path]) -> dict:
    out = {}
    for f in files:
        d = json.loads(f.read_text())
        res = d["results"]
        cfgs = collections.Counter(cfg_key(r["lt"]) for r in res if r.get("lt"))
        out[str(f)] = dict(jobs=len(res), errors=sum(r["error"] is not None for r in res),
                           no_log=sum(r["n_lt_calls"] == 0 for r in res),
                           canary_fail=sum(not r.get("canary_ok", False) for r in res),
                           inputs_cpu_ne_gpu=sum(r.get("inputs_cpu_eq_gpu") is False for r in res),
                           splits=sum(bool(r.get("lt")) and r["lt"]["numSplitsK"] > 1 for r in res),
                           configs=len(cfgs))
    return out


def passes(o: Path) -> dict:
    """run 1 vs run 2 (separate processes) of every live job, by job id."""
    sha = {"p1": {}, "p2": {}}
    bad_in = set()                      # GPU-generated inputs != CPU regeneration in either pass: an input fault
    for f in o.rglob("p[12]_ws*.json"):
        if f.name.endswith(".jobs.json"):
            continue
        tag = f.name[:2]
        for r in json.loads(f.read_text())["results"]:
            sha[tag][r["id"]] = r.get("full_sha256")
            if r.get("inputs_cpu_eq_gpu") is False:
                bad_in.add(r["id"])
    both = set(sha["p1"]) & set(sha["p2"])
    diff = sorted(i for i in both if sha["p1"][i] != sha["p2"][i])
    return dict(p1=len(sha["p1"]), p2=len(sha["p2"]), compared=len(both), differ=len(diff), differ_ids=diff[:40],
                differ_with_input_fault=sorted(set(diff) & bad_in), differ_inputs_ok=sorted(set(diff) - bad_in)[:40],
                input_fault_ids=sorted(bad_in)[:40])


def truth_compare(o: Path, truth: Path) -> dict:
    t = json.loads(truth.read_text())
    out = {}
    for name in ("m2_torch", "m2_te"):
        p = o / name / f"{name}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        res = collections.Counter()
        for r in d["calls"]:
            call, kind = r["id"].split(".")
            case = t["cases"][int(call[4:])]
            outs = {x["name"]: x for x in case["outputs"]}
            want = {"acc_fast": outs["acc_full"]["sha256"],
                    "y_fast": outs["y_native"].get("full_sha256", outs["y_native"]["sha256"]),
                    "te_y": outs["y_native"].get("full_sha256", outs["y_native"]["sha256"])}.get(kind)
            if want is None:
                continue
            res[f"{kind}: {'equal' if r['runs'][0].get('sha256') == want else 'DIFFERENT'}"] += 1
        out[name] = dict(res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--truth", type=Path)
    a = ap.parse_args()
    o = a.out
    rep = dict(status=(o / "status.txt").read_text().splitlines() if (o / "status.txt").exists() else None)
    for p in sorted(o.glob("m[123]_*/m[123]_*.json")):
        rep[p.stem] = rec_summary(p)
    for name in ("m1_kit", "m1_mma"):
        p = o / name / "kit.json"
        if p.exists():
            rep[name] = json.loads(p.read_text())
            for v in rep[name].get("runs", {}).values():
                v.pop("models_100pct", None)
            for v in rep[name].get("sets", {}).values():
                for w in (v.values() if isinstance(v, dict) and "n" not in v else [v]):
                    if isinstance(w, dict):
                        w.pop("models_100pct", None)
    rep["live_workers"] = worker_summary(sorted(f for f in o.rglob("p[12]_ws*.json") if not f.name.endswith(".jobs.json")))
    rep["live_passes"] = passes(o)
    if (o / "m6_heur" / "heur.json").exists():
        h = json.loads((o / "m6_heur" / "heur.json").read_text())
        rep["m6_heur"] = {k: h[k] for k in ("rows", "split_rows", "status_nonzero", "seconds",
                                            "evo2_split_status_per_ws")}
    if a.truth:
        rep["vs_truth"] = truth_compare(o, a.truth)
    size = sum(f.stat().st_size for f in o.rglob("*") if f.is_file() and ".cache" not in f.parts)
    rep["bytes"] = size
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
