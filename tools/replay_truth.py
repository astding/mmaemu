#!/usr/bin/env python3
"""T2: full replay of the RTX 4060 hardware truth ($MMAEMU_TRUTH_DIR; maintainer-only data) through mmaemu, plus the
verdict against the audit's FixedModel report (docs/testing.md).

usage (the truth package is imported read-only; nothing is ever written into it):
    MMAEMU_TRUTH_DIR=<truth checkout> PYTHONPATH=<repo>:$MMAEMU_TRUTH_DIR \
    python tools/replay_truth.py --out docs/validation_logs/<date> \
        [--compare $MMAEMU_TRUTH_DIR/replay/replay_fixed.json] [--only REGEX] [--skip REGEX] [--max-cases N]
        [--threads 8] [--device cpu|cuda] [--compare-only REPORT]

What it does
1. Installs a sys.meta_path blocker for the old package `tcemu` (and `tcemu.*`; `tcemu_ada_truth` is allowed): the
   truth's env_info() catches the ImportError; any other import of tcemu fails loudly.
2. Snapshots the truth tree (path, size, mtime_ns of every file except .git/) before and after the run and pins the
   sha256 of its manifest.json.
3. Calls tcemu_ada_truth.replay.replay([MmaemuTruthModel()]) programmatically, with replay.eval_output wrapped so
   that the adapter knows which output it is evaluating (for the by-id n/a list and the emulator-group check).
4. Writes <out>/replay_mmaemu.json (the replay report + adapter provenance) and, with --compare, <out>/compare.json
   and <out>/compare.txt (per-recording delta table vs FixedModel and the verdict). Reports go to --out only, which
   must not be inside the truth tree.
Exit status 0 iff the verdict is PASS (or no --compare was given and the run finished).

--device cuda uses mmaemu's CUDA backend for large GEMMs; run it under $MMAEMU_GPU_LOCK and keep it
under 10 minutes (e.g. with --only); the default is CPU (fastacc ChainFn), which needs no GPU lock.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


class TcemuBlocker(importlib.abc.MetaPathFinder):
    """Refuse `import tcemu` / `tcemu.*` (the audited prototype); `tcemu_ada_truth` is not affected."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tcemu" or fullname.startswith("tcemu."):
            raise ImportError(f"import of {fullname!r} is blocked: the truth replay must test mmaemu, not tcemu")
        return None


def install_blocker():
    if not any(isinstance(f, TcemuBlocker) for f in sys.meta_path):
        sys.meta_path.insert(0, TcemuBlocker())
    if "tcemu" in sys.modules:
        raise SystemExit("tcemu was imported before the blocker was installed")


REPO = Path(__file__).resolve().parents[1]
ALLOWED_NEW_NA = {"bias", "row-wise", "fused-bias"}


def snapshot(root: Path) -> dict:
    out = {}
    for dp, dns, fns in os.walk(root):
        if Path(dp) == root and ".git" in dns:
            dns.remove(".git")
        for fn in fns:
            p = Path(dp) / fn
            st = p.lstat()
            out[str(p.relative_to(root))] = [st.st_size, st.st_mtime_ns]
    return out


def diff_snapshots(a: dict, b: dict) -> dict:
    return dict(added=sorted(set(b) - set(a)), removed=sorted(set(a) - set(b)),
                changed=sorted(k for k in set(a) & set(b) if a[k] != b[k]))


def git_head(repo: Path) -> str:
    r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    d = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                       capture_output=True, text=True)
    return r.stdout.strip() + ("+dirty" if d.stdout.strip() else "")


# ---------------------------------------------------------------------------------------------------------------
def run_replay(args, truth: Path):
    import torch
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(truth))
    from tcemu_ada_truth import replay as R  # noqa: WPS433
    if Path(R.__file__).resolve().parents[1] != truth:
        raise SystemExit(f"tcemu_ada_truth imported from {R.__file__}, not {truth}")
    sys.path.insert(0, str(REPO / "tools"))
    import truth_adapter as TA  # noqa: WPS433
    import mmaemu

    model = TA.MmaemuTruthModel(device=args.device, cuda_backend=args.cuda_backend)
    emu = dict(outputs=0, values=0, diff=0, diff_outside_nan_mask=0, nan_mask_not_canonical=0, rows=[])
    orig_eval = R.eval_output

    def eval_output(m, kind, case_inputs, out, ctx):
        if m is model:
            model.current = out
        md = orig_eval(m, kind, case_inputs, out, ctx)
        if m is model and out.rec.id.startswith("emulator.") and out.stored:
            check_emulator(out, case_inputs, md, emu)
        return md

    R.eval_output = eval_output
    t0 = time.time()
    reports = R.replay([model], only=args.only, skip=args.skip, hash_only=not args.no_hash_only,
                       max_cases=args.max_cases, log=lambda s: print(s, flush=True))
    R.eval_output = orig_eval
    rep = reports[model.name]
    rep["adapter"] = dict(
        mmaemu_version=getattr(mmaemu, "__version__", None), mmaemu_file=mmaemu.__file__,
        repo_commit=git_head(REPO), device=args.device, cuda_backend=args.cuda_backend, threads=args.threads,
        fastref=model.fastref, ref_limit=model.ref_limit,
        accumulator_sources={k: dict(v) for k, v in sorted(model.sources.items())},
        na_log=model.na_log, emulator_check=emu, seconds=round(time.time() - t0, 1),
        argv=sys.argv, only=args.only, skip=args.skip, max_cases=args.max_cases)
    return rep


def check_emulator(out, ins, md, emu):
    """The emulator group (old tcemu kernels on fixed inputs) must equal mmaemu except at outputs whose row of A or
    column of B holds a NaN code (the old kernels decoded NaN as finite, c3-F4); there mmaemu gives canonical NaN."""
    import numpy as np
    hw = out.bits()
    a, b = ins["a"], ins["b"]
    fa, fb = out.params.get("fmt_a", "e4m3"), out.params.get("fmt_b", "e4m3")

    def nan_codes(x, f):
        c = x.astype(np.int16) & 0x7F
        return (c == 0x7F) if f == "e4m3" else (c > 0x7C)
    rows = nan_codes(a, fa).any(1)
    cols = nan_codes(b, fb).any(1)
    mask = rows[:, None] | cols[None, :]
    md = np.asarray(md).reshape(hw.shape)
    diff = hw != md
    emu["outputs"] += 1
    emu["values"] += int(hw.size)
    emu["diff"] += int(diff.sum())
    emu["diff_outside_nan_mask"] += int((diff & ~mask).sum())
    emu["nan_mask_not_canonical"] += int((md[mask] != 0x7FFFFFFF).sum())
    emu["rows"].append(dict(output=f"{out.case.id}/{out.name}", diff=int(diff.sum()), nan_mask=int(mask.sum())))


# ---------------------------------------------------------------------------------------------------------------
def compare(mm: dict, fx: dict, partial: bool) -> dict:
    """Per-recording delta table + verdict."""
    F = {r["id"]: r for r in fx["recordings"]}
    X = {r["id"]: r for r in mm["recordings"]}
    na_by_rec = defaultdict(list)
    for e in mm["adapter"]["na_log"]:
        na_by_rec[e["id"].split("/", 1)[0]].append(e)
    rows, fails = [], []
    tot = Counter()
    for rid, x in X.items():
        f = F.get(rid)
        if f is None:
            fails.append(f"{rid}: not in the FixedModel report")
            continue
        if x.get("status") or f.get("status"):
            rows.append(dict(id=rid, kind="table", status="table"))
            continue
        new_na = [e for e in na_by_rec.get(rid, []) if e["category"] not in ("fixed-set", "rejects")]
        new_na_values = sum(e["values"] for e in new_na if not e["hash_only"] and not e["hw_error"])
        new_na_hash = sum(1 for e in new_na if e["hash_only"])
        new_na_hwerr = sum(1 for e in new_na if e["hw_error"])
        bad_na = [e for e in new_na if e["category"] not in ALLOWED_NEW_NA]
        r = dict(id=rid, kind=x["kind"], n_fixed=f["n"], n=x["n"], match=x["match"], zero_sign=x["zero_sign"],
                 nan_payload=x["nan_payload"], value=x["value"], fixed_value=f["value"],
                 hash=f"{x['hash_only_equal']}/{x['hash_only_total']}",
                 hash_fixed=f"{f['hash_only_equal']}/{f['hash_only_total']}",
                 hwerr=f"{x['hw_error_reproduced']}/{x['hw_error_total']}",
                 hwerr_fixed=f"{f['hw_error_reproduced']}/{f['hw_error_total']}",
                 n_a=x["n_a"], n_a_fixed=f["n_a"], new_na=len(new_na), errors=len(x["errors"]),
                 fastacc=f"{x['fastacc_checks']['mismatches']}/{x['fastacc_checks']['n']}",
                 sources=mm["adapter"]["accumulator_sources"].get(rid, {}), problems=[])
        P = r["problems"]
        emulator = rid.startswith("emulator.")
        if x["errors"]:
            P.append(f"{len(x['errors'])} errors: {x['errors'][:3]}")
        if x["zero_sign"] or x["nan_payload"]:
            P.append(f"zero_sign {x['zero_sign']} nan_payload {x['nan_payload']}")
        if emulator:
            fo = {(o["case"], o["output"]): o for o in f.get("outputs", [])}
            xo = {(o["case"], o["output"]): o for o in x.get("outputs", [])}
            keys = ("n", "match", "zero_sign", "nan_payload", "value")
            if set(fo) != set(xo) or any(tuple(fo[k].get(c) for c in keys) != tuple(xo[k].get(c) for c in keys)
                                         for k in fo):
                P.append("emulator group: per-output differences are not identical to FixedModel's")
            ec = mm["adapter"]["emulator_check"]
            if ec["diff_outside_nan_mask"] or ec["nan_mask_not_canonical"]:
                P.append(f"emulator group: {ec['diff_outside_nan_mask']} differences outside the NaN-decode outputs,"
                         f" {ec['nan_mask_not_canonical']} NaN-decode outputs not canonical NaN")
        else:
            if x["value"]:
                P.append(f"value mismatches {x['value']}")
            if x["match"] != x["n"]:
                P.append(f"match {x['match']} != n {x['n']}")
        if x["n"] + new_na_values != f["n"]:
            P.append(f"stored values {x['n']} + new n/a {new_na_values} != FixedModel {f['n']}")
        if x["hash_only_equal"] != x["hash_only_total"]:
            P.append(f"hash-only {x['hash_only_equal']}/{x['hash_only_total']}")
        if x["hash_only_total"] + new_na_hash != f["hash_only_total"]:
            P.append(f"hash-only outputs {x['hash_only_total']} + new n/a {new_na_hash} != {f['hash_only_total']}")
        if x["hw_error_reproduced"] < f["hw_error_reproduced"]:
            P.append(f"hw_error reproduced {x['hw_error_reproduced']} < FixedModel {f['hw_error_reproduced']}")
        if x["hw_error_total"] + new_na_hwerr != f["hw_error_total"]:
            P.append(f"hw_error outputs {x['hw_error_total']} + new n/a {new_na_hwerr} != {f['hw_error_total']}")
        if x["n_a"] != f["n_a"] + len(new_na):
            P.append(f"n/a {x['n_a']} != FixedModel {f['n_a']} + new {len(new_na)}")
        for reason, cnt in f["n_a_reasons"].items():
            if x["n_a_reasons"].get(reason, 0) != cnt:
                P.append(f"FixedModel n/a reason {reason!r}: {cnt} vs mmaemu {x['n_a_reasons'].get(reason, 0)}")
        if bad_na:
            P.append(f"{len(bad_na)} new n/a outside bias/row-wise/fused-bias: {[e['id'] for e in bad_na[:5]]}")
        if x["fastacc_checks"]["mismatches"]:
            P.append(f"fastacc cross-check mismatches {x['fastacc_checks']['mismatches']}")
        for k in ("n", "match", "zero_sign", "nan_payload", "value", "hash_only_total", "hash_only_equal",
                  "hw_error_total", "hw_error_reproduced", "n_a"):
            tot["x_" + k] += x[k]
            tot["f_" + k] += f[k]
        tot["new_na"] += len(new_na)
        tot["new_na_hash"] += new_na_hash
        tot["fastacc_n"] += x["fastacc_checks"]["n"]
        tot["fastacc_mismatches"] += x["fastacc_checks"]["mismatches"]
        rows.append(r)
        if P:
            fails.extend(f"{rid}: {p}" for p in P)
    missing = sorted(set(F) - set(X))
    g = []
    if not partial:
        if missing:
            fails.append(f"recordings not replayed: {missing}")
        if tot["x_hw_error_reproduced"] < 113:
            fails.append(f"hw_error reproduced {tot['x_hw_error_reproduced']} < 113")
        if tot["x_hash_only_equal"] < 2539 - tot["new_na_hash"]:
            fails.append(f"hash-only equal {tot['x_hash_only_equal']} < 2539 - {tot['new_na_hash']}")
        if mm["adapter"]["emulator_check"]["diff"] != 6120:
            fails.append(f"emulator group differences {mm['adapter']['emulator_check']['diff']} != 6120")
    if tot["x_zero_sign"] or tot["x_nan_payload"]:
        fails.append(f"zero_sign {tot['x_zero_sign']} nan_payload {tot['x_nan_payload']}")
    g.append(f"recordings compared {len(rows)} of {len(F)}{' (partial run)' if partial else ''}")
    new_na_list = [dict(id=e["id"], category=e["category"], reason=e["reason"]) for e in mm["adapter"]["na_log"]
                   if e["category"] not in ("fixed-set", "rejects")]
    return dict(verdict="PASS" if not fails else "FAIL", partial=partial, failures=fails, totals=dict(tot),
                general=g, rows=rows, new_na=new_na_list)


def compare_text(c: dict, extra: list[str]) -> str:
    L = [f"VERDICT: {c['verdict']}" + (" (partial run)" if c["partial"] else "")]
    L += c["general"] + extra
    t = c["totals"]
    L.append(f"totals mmaemu vs FixedModel: n {t.get('x_n', 0)} / {t.get('f_n', 0)}; match {t.get('x_match', 0)} / "
             f"{t.get('f_match', 0)}; zero_sign {t.get('x_zero_sign', 0)}; nan_payload {t.get('x_nan_payload', 0)}; "
             f"value {t.get('x_value', 0)} / {t.get('f_value', 0)}; hash-only {t.get('x_hash_only_equal', 0)}/"
             f"{t.get('x_hash_only_total', 0)} vs {t.get('f_hash_only_equal', 0)}/{t.get('f_hash_only_total', 0)}; "
             f"hw_error {t.get('x_hw_error_reproduced', 0)}/{t.get('x_hw_error_total', 0)} vs "
             f"{t.get('f_hw_error_reproduced', 0)}/{t.get('f_hw_error_total', 0)}; n/a {t.get('x_n_a', 0)} vs "
             f"{t.get('f_n_a', 0)} (new {t.get('new_na', 0)}); fastacc cross-check {t.get('fastacc_mismatches', 0)} "
             f"mismatches in {t.get('fastacc_n', 0)} values")
    L.append("")
    L.append(f"{'recording':42s} {'n(mmaemu)':>11s} {'n(fixed)':>11s} {'zs':>4s} {'nan':>4s} {'val':>6s} "
             f"{'hash':>10s} {'hwerr':>8s} {'hwerr(F)':>8s} {'n/a':>4s} {'n/a(F)':>6s} {'new':>4s}  source / problems")
    for r in c["rows"]:
        if r.get("status") == "table":
            L.append(f"{r['id']:42s} table")
            continue
        src = ",".join(f"{k}:{v}" for k, v in r["sources"].items())
        L.append(f"{r['id']:42s} {r['n']:11d} {r['n_fixed']:11d} {r['zero_sign']:4d} {r['nan_payload']:4d} "
                 f"{r['value']:6d} {r['hash']:>10s} {r['hwerr']:>8s} {r['hwerr_fixed']:>8s} {r['n_a']:4d} "
                 f"{r['n_a_fixed']:6d} {r['new_na']:4d}  {src}" + (f"  PROBLEMS: {r['problems']}" if r["problems"]
                                                                   else ""))
    if c["new_na"]:
        L.append("")
        L.append("new n/a outputs (mmaemu v0.1 scope; FixedModel modelled them):")
        for e in c["new_na"]:
            L.append(f"  {e['id']}  [{e['category']}] {e['reason'][:100]}")
    if c["failures"]:
        L.append("")
        L.append("FAILURES:")
        L.extend("  " + f for f in c["failures"])
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="report directory, e.g. docs/validation_logs/2026-10-12")
    ap.add_argument("--compare", help="FixedModel report, e.g. $MMAEMU_TRUTH_DIR/replay/replay_fixed.json")
    ap.add_argument("--compare-only", help="skip the replay; compare this replay_mmaemu.json")
    ap.add_argument("--only")
    ap.add_argument("--skip")
    ap.add_argument("--max-cases", type=int)
    ap.add_argument("--no-hash-only", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--cuda-backend", default="auto")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args(argv)

    install_blocker()
    truth = Path(os.environ.get("MMAEMU_TRUTH_DIR", "")).resolve()
    if not os.environ.get("MMAEMU_TRUTH_DIR") or not (truth / "manifest.json").exists():
        raise SystemExit("set MMAEMU_TRUTH_DIR to the tcemu_ada_truth checkout")
    out = Path(a.out).resolve()
    if out == truth or truth in out.parents:
        raise SystemExit(f"--out {out} is inside the truth tree; reports never go there")
    out.mkdir(parents=True, exist_ok=True)
    manifest_sha = hashlib.sha256((truth / "manifest.json").read_bytes()).hexdigest()
    extra = [f"truth {truth}, manifest.json sha256 {manifest_sha}"]
    hwmini_index = REPO / "tests" / "data" / "hw_mini" / "index.json"
    if hwmini_index.exists():
        pinned = json.loads(hwmini_index.read_text()).get("truth_manifest_sha256")
        extra.append(f"hw_mini pins truth manifest {pinned}: {'same' if pinned == manifest_sha else 'DIFFERENT'}")
    before = snapshot(truth)

    if a.compare_only:
        rep = json.loads(Path(a.compare_only).read_text())
    else:
        rep = run_replay(a, truth)
        rep["truth"] = dict(root=str(truth), manifest_sha256=manifest_sha)
        p = out / f"replay_mmaemu{a.suffix}.json"
        p.write_text(json.dumps(rep, indent=1))
        print("wrote", p, json.dumps(rep["summary"]), flush=True)

    after = snapshot(truth)
    d = diff_snapshots(before, after)
    tcemu_loaded = "tcemu" in sys.modules or any(m.startswith("tcemu.") for m in sys.modules)
    checks = dict(truth_unchanged=not any(d.values()), truth_diff=d, tcemu_in_sys_modules=tcemu_loaded,
                  dont_write_bytecode=sys.dont_write_bytecode, truth_files=len(after))
    extra.append(f"truth tree unchanged (files {len(after)}, .git excluded): {checks['truth_unchanged']}"
                 + ("" if checks["truth_unchanged"] else f" {d}"))
    extra.append(f"'tcemu' in sys.modules at exit: {tcemu_loaded}")
    ad = rep.get("adapter", {})
    extra.append(f"accumulator path for large GEMMs: {ad.get('device')}; fastref {ad.get('fastref')}")
    if a.compare:
        fx = json.loads(Path(a.compare).read_text())
        partial = bool(rep["adapter"].get("only") or rep["adapter"].get("skip") or rep["adapter"].get("max_cases"))
        c = compare(rep, fx, partial)
        c["checks"] = checks
        if not checks["truth_unchanged"]:
            c["failures"].append(f"truth tree changed: {d}")
        if tcemu_loaded:
            c["failures"].append("'tcemu' is in sys.modules")
        c["verdict"] = "PASS" if not c["failures"] else "FAIL"
        (out / f"compare{a.suffix}.json").write_text(json.dumps(c, indent=1))
        txt = compare_text(c, extra)
        (out / f"compare{a.suffix}.txt").write_text(txt)
        print(txt)
        print(c["verdict"])
        return 0 if c["verdict"] == "PASS" else 1
    print("\n".join(extra))
    return 0 if checks["truth_unchanged"] and not tcemu_loaded else 1


if __name__ == "__main__":
    sys.exit(main())
