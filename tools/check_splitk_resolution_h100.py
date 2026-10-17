"""Check mmaemu's split-K resolution for the H100 profiles against everything the H100 decided (CPU, no GEMM).

Every status-0 heuristic row of the 2026-10-16 split-K sweep (p1 envelope grid, p2 K dense, p3 model rows, the sealed
p4; docs/validation_logs/2026-10-16/h100-splitk-sweep/results) and every live call of the probe and of the sweep
(results/analysis/mmaemu_replay.json of docs/validation_logs/2026-10-15/h100-probe, mmaemu_replay_p5.json: the logged
numSplitsK) goes through mmaemu.platform.splitk_table.resolve_splitk with the profile of its workspace
(hopper at 1 MiB, hopper-te at 32 MiB, hopper-ws0 at 0; another workspace: the torch profile with that workspace,
as tools/replay_h100_log.py does), splitk="auto", no live query. Per part and workspace it counts the decisions by
source (table, envelope, workspace0, or SplitKUnknownError) and the decisions whose S differs from the H100's.

Expected: no difference, except where the envelope's sealed test already found one (eval_sealed.json: one TE row of
p4). The envelope counts are below eval_sealed.json's tested counts where a held-out row is also a table row (the
table comes first). Writes results/analysis/resolver_check.json of the sweep with the commit, whether mmaemu or this
script differ from it, and the sha256 of the table files.

usage: PYTHONDONTWRITEBYTECODE=1 python tools/check_splitk_resolution_h100.py [--jobs 16] [--out PATH]"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SWEEP = REPO / "docs" / "validation_logs" / "2026-10-16" / "h100-splitk-sweep"
RES = SWEEP / "results"
REPLAYS = [REPO / "docs" / "validation_logs" / "2026-10-15" / "h100-probe" / "results" / "analysis" / "mmaemu_replay.json",
           RES / "analysis" / "mmaemu_replay_p5.json"]
OUT = RES / "analysis" / "resolver_check.json"
TABLE_ID = "heur_h100pcie_lt12.8.3.14"
DEPENDS = ("mmaemu", "tools/check_splitk_resolution_h100.py")
OUTS = ["float32", "bfloat16", "float16", "float8_e4m3fn", "float8_e5m2"]
FMTS = ["e4m3", "e5m2"]
LIST = 20


def _profile(ws: int):
    from mmaemu.platform.profiles import get_profile
    if ws == 0:
        return get_profile("hopper-ws0")
    if ws == 1 << 20:
        return get_profile("hopper")
    if ws == 32 << 20:
        return get_profile("hopper-te")
    return dataclasses.replace(get_profile("hopper"), name=f"hopper-ws{ws}", workspace_bytes=ws)


def _decide(rows) -> dict:
    """rows: (M, N, K, fa, fb, out, fast, ws, S_h100, ident); counts by source and the differing decisions."""
    from mmaemu.platform import profiles, splitk_table
    profiles.Profile.matches_device = lambda self, device=None: False        # no live query
    warnings.simplefilter("ignore")
    by, diff = {}, []
    for M, N, K, fa, fb, out, fast, ws, S, ident in rows:
        try:
            d = splitk_table.resolve_splitk(M, N, K, a_dtype=fa, b_dtype=fb, out_dtype=out, use_fast_accum=fast,
                                            profile=_profile(ws))
            src = d.source
        except splitk_table.SplitKUnknownError:
            src, d = "unknown", None
        by[src] = by.get(src, 0) + 1
        if d is not None and d.S != S:
            diff.append(dict(id=ident, M=M, N=N, K=K, fmt_a=fa, fmt_b=fb, out=out, fast=fast, ws=ws, S_h100=S,
                             S_mmaemu=d.S, source=d.source, verified=d.verified))
    return dict(rows=len(rows), by_source=by, differ=len(diff), differences=diff)


def _npz_rows(path: Path):
    z = {k: v.tolist() for k, v in np.load(path).items()}
    rows, nz = [], 0
    for i in range(len(z["M"])):
        if z["status"][i] != 0:
            nz += 1
            continue
        rows.append((z["M"][i], z["N"][i], z["K"][i], FMTS[z["fa"][i]], FMTS[z["fb"][i]], OUTS[z["out"][i]],
                     bool(z["fast"][i]), z["ws"][i], z["splitK"][i], i))
    return rows, nz


def _file_job(path: str) -> tuple[str, dict]:
    p = Path(path)
    rows, nz = _npz_rows(p)
    r = _decide(rows)
    r["status_nonzero"] = nz
    r["differences"] = [dict(d, file=p.relative_to(RES).as_posix()) for d in r["differences"]]
    return path, r


def _live_rows():
    rows = []
    for rp in REPLAYS:
        for j in json.loads(rp.read_text())["live"]:
            if j["verdict"] != "equal":
                continue
            rows.append((int(j["M"]), int(j["N"]), int(j["K"]), j["fa"], j["fb"], j["out"], bool(j["fast"]),
                         int(j["ws"]), int(j["S"] or 1), f"{rp.parents[2].name}:{j['id']}"))
    return rows


def _merge(results: list[dict]) -> dict:
    by: dict = {}
    for r in results:
        for k, v in r["by_source"].items():
            by[k] = by.get(k, 0) + v
    diffs = [d for r in results for d in r["differences"]]
    return dict(rows=sum(r["rows"] for r in results), status_nonzero=sum(r.get("status_nonzero", 0) for r in results),
                by_source=dict(sorted(by.items())), differ=len(diffs), differences=diffs[:LIST])


def provenance() -> dict:
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", *DEPENDS], capture_output=True,
                           text=True).stdout.strip()
    data = REPO / "mmaemu" / "platform" / "data"
    return dict(mmaemu_commit=head, dirty=bool(dirty), dirty_scope=list(DEPENDS), python=sys.version.split()[0],
                numpy=np.__version__,
                table={n: hashlib.sha256((data / n).read_bytes()).hexdigest()
                       for n in (f"{TABLE_ID}.npz", f"{TABLE_ID}.json", f"verified_split_combos_{TABLE_ID[5:]}.json")})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    files = {p: sorted(str(f) for f in (RES / p).glob("*_a*_c*.npz")) for p in ("p1", "p2", "p3", "p4_sealed")}
    with ProcessPoolExecutor(a.jobs) as ex:
        live = ex.submit(_decide, _live_rows())
        done = dict(ex.map(_file_job, [f for fs in files.values() for f in fs]))
        live = live.result()
    rep = dict(format="mmaemu-h100-resolver-check/1", provenance=provenance(), parts={})
    for part, fs in files.items():
        groups: dict = {}
        for f in fs:
            groups.setdefault(Path(f).name.split("_a")[0], []).append(done[f])
        rep["parts"][part] = {g: _merge(rs) for g, rs in sorted(groups.items())}
    rep["parts"]["live"] = {"probe+p5": _merge([live])}
    ev = json.loads((RES / "analysis" / "eval_sealed.json").read_text())   # keyed by the profiles' names of then
    known = [dict(f, file=f"p4_sealed/{n}") for n, prof in (("te_a256_c00.npz", "hopper-h100pcie-te"),
                                                            ("torch_a0_c00.npz", "hopper-h100pcie-torch"))
             for f in ev[prof]["failures"]]
    found = [d for g in rep["parts"].values() for r in g.values() for d in r["differences"]]
    expected = {(k["M"], k["N"], k["K"], k["out"], k["fast"], k["fmt_a"], k["fmt_b"]) for k in known}
    got = {(d["M"], d["N"], d["K"], d["out"], d["fast"], d["fmt_a"], d["fmt_b"]) for d in found}
    rep["eval_sealed_failures"] = known
    total = sum(r["differ"] for g in rep["parts"].values() for r in g.values())
    rep["passed"] = total == len(known) and got == expected
    rep["envelope_vs_eval_sealed"] = {
        prof: dict(eval_tested=ev[prof]["tested"],
                   resolver_envelope=rep["parts"]["p4_sealed"][g]["by_source"].get("envelope", 0),
                   resolver_table=rep["parts"]["p4_sealed"][g]["by_source"].get("table", 0))
        for g, prof in (("torch", "hopper-h100pcie-torch"), ("te", "hopper-h100pcie-te"))}
    a.out.write_text(json.dumps(rep, indent=1) + "\n")
    for part, g in rep["parts"].items():
        for name, r in g.items():
            print(f"{part:<10} {name:<9} rows {r['rows']:>9}  {r['by_source']}  differ {r['differ']}")
    print("envelope vs eval_sealed:", rep["envelope_vs_eval_sealed"])
    print(f"passed {rep['passed']} (differences = the sealed test's failures); wrote {a.out}")
    sys.exit(0 if rep["passed"] else 1)


if __name__ == "__main__":
    main()
