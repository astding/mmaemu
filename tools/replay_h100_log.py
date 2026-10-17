"""Replay the H100 PCIe probe (docs/validation_logs/2026-10-15/h100-probe) through mmaemu and compare full-output
sha256 with the H100's.

live  every live torch._scaled_mm call of M4 and M6 (results/live_jobs.tar.gz: job specs, logged cuBLASLt config,
      full-output sha256), or of another recording in the same format (--live, e.g. the split-K sweep's p5 calls,
      docs/validation_logs/2026-10-16/h100-splitk-sweep/results/live_jobs.tar.gz): inputs regenerated from the job
      spec (tools/record/live.py, integer hashing, device independent), then mmaemu.scaled_mm on the --backend
      (default reference; auto picks the CUDA kernel of the hopper preset, h4 / h1, for e4m3 x e4m3 and the
      reference for the other pairs) with
        - the logged split passed explicitly (numSplitsK, the scheme from the H100 table's customOption rule), and
        - splitk="auto" through mmaemu.explain: a decision (table, envelope) must equal the logged one; a shape
          the profile cannot resolve may raise SplitKUnknownError, recorded as auto = {error};
      the profile is hopper with the call's logged workspace (hopper-ws0 for 0). A call the
      H100 rejected must raise ReferenceRejectsError. The judged pass is pass 1, or pass 2 where pass 1 had an input
      fault (plan section 5); the two passes' sha256 must agree.
m2    (--truth) the 21 Evo 2 1B projections of the RTX 4060 truth's w2k window: acc / y (torch, profile
      hopper) and te_acc / te_y (Transformer Engine, profile hopper-te), fast and slow,
      splitk="auto" (the table must give S = 1), against results/m2/m2_{torch,te}.json.

usage (GPU under $MMAEMU_GPU_LOCK; CPU works, slowly):
  python tools/replay_h100_log.py [--device cuda] [--backend reference|auto|cuda:h1|cuda:h4]
         [--truth $MMAEMU_TRUTH_DIR] [--parts live,m2] [--only ID,...] [--live LIVE.tar.gz]
         [--offset 0] [--limit N] [--m2-calls 0,1,...] --part PART.json          # one chunk
  python tools/replay_h100_log.py --merge PART.json ... [--live LIVE.tar.gz]
         [--out <log>/results/analysis/mmaemu_replay.json]"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import subprocess
import sys
import tarfile
import time
import warnings
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "tools" / "record"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

LOG = REPO / "docs" / "validation_logs" / "2026-10-15" / "h100-probe"
LIVE = LOG / "results" / "live_jobs.tar.gz"
OUT = LOG / "results" / "analysis" / "mmaemu_replay.json"
FORMAT = "mmaemu-h100-replay/1"
FP8 = {"e4m3": "float8_e4m3fn", "e5m2": "float8_e5m2"}
TABLE_ID = "heur_h100pcie_lt12.8.3.14"
DEPENDS = ("mmaemu", "tools/replay_h100_log.py", "tools/record")    # what the replay runs; "dirty" refers to these


def sha_bits(y) -> str:
    import common
    return hashlib.sha256(np.ascontiguousarray(common.tbits(y)).tobytes()).hexdigest()


def live_jobs(path: Path = LIVE) -> list:
    """(source, job, judged result, pass-2 result or None, judged pass) of every live call, in a fixed order."""
    with tarfile.open(path, "r:gz") as tf:
        files = {m.name: tf.extractfile(m).read() for m in tf.getmembers() if m.isfile()}
    out = []
    for name in sorted(files):
        if "/p1/" not in name or name.endswith(".jobs.json") or not name.endswith(".json"):
            continue
        src = name.split("/")[0]
        jobs = {j["id"]: j for j in json.loads(files[name.replace(".json", ".jobs.json")])}
        n2 = name.replace("/p1/", "/p2/").replace("/p1_", "/p2_")
        p2 = {r["id"]: r for r in json.loads(files[n2])["results"]} if n2 in files else {}
        for r in json.loads(files[name])["results"]:
            r2 = p2.get(r["id"])
            judged, used = r, "p1"
            if r.get("inputs_cpu_eq_gpu") is False:
                judged, used = (r2, "p2") if r2 is not None and r2.get("inputs_cpu_eq_gpu") is True else (None, None)
            out.append((src, jobs[r["id"]], judged, r2, used))
    return out


def _profiles():
    from mmaemu.platform.profiles import get_profile
    base = get_profile("hopper")
    cache = {}

    def prof(ws: int):
        if ws == 0:
            return get_profile("hopper-ws0")
        if ws == base.workspace_bytes:
            return base
        if ws not in cache:   # torch._scaled_mm with CUBLASLT_WORKSPACE_SIZE set: the same platform, another ws
            cache[ws] = dataclasses.replace(base, name=f"hopper-ws{ws}", workspace_bytes=ws)
        return cache[ws]
    return prof


def replay_live(items, dev, backend="reference") -> list:
    import torch

    import live
    import mmaemu
    from mmaemu import errors as E
    from mmaemu.platform.splitk_table import load_table
    table = load_table(TABLE_ID)
    prof = _profiles()
    res = []
    for n, (src, j, r, r2, used) in enumerate(items):
        t0 = time.monotonic()
        rec = dict(id=j["id"], src=src, M=j["M"], N=j["N"], K=j["K"], fa=j["fa"], fb=j["fb"], out=j["out"],
                   fast=bool(j["fast"]), judged=used)
        if r is None:
            rec.update(verdict="input_fault_both_passes")
            res.append(rec)
            continue
        if r2 is not None and r2.get("inputs_cpu_eq_gpu") is True and r.get("inputs_cpu_eq_gpu") is True:
            rec["passes_equal"] = r2.get("full_sha256") == r.get("full_sha256") and r2.get("error") == r.get("error")
        a8, b8 = live.make_inputs(j, device=dev)
        a = a8.view(getattr(torch, FP8[j["fa"]]))
        b = b8.view(getattr(torch, FP8[j["fb"]]))
        sa = torch.tensor(j["scale_a"], dtype=torch.float32, device=dev)
        sb = torch.tensor(j["scale_b"], dtype=torch.float32, device=dev)
        sr = None if j.get("scale_result") is None else torch.tensor(j["scale_result"], dtype=torch.float32,
                                                                     device=dev)
        od = getattr(torch, j["out"])
        kw = dict(scale_result=sr, out_dtype=od, use_fast_accum=bool(j["fast"]), backend=backend)
        if r["error"]:
            ws = j["ws"] if j["ws"] >= 0 else 1 << 20
            rec.update(S=None, ws=ws, hw_error=r["error"][:160])
            try:
                mmaemu.scaled_mm(a, b.t(), sa, sb, profile=prof(ws), splitk=1, **kw)
                rec["verdict"] = "mmaemu_ran_hw_rejected"
            except E.ReferenceRejectsError:
                rec["verdict"] = "rejected_as_hardware"
            except Exception as e:  # noqa: BLE001
                rec["verdict"] = f"mmaemu_error {type(e).__name__}: {str(e)[:160]}"
            res.append(rec)
            continue
        lt = r["lt"]
        S, custom, ws = int(lt["numSplitsK"]), int(lt["customOption"]), int(lt["workspace_bytes"])
        scheme = "none" if S == 1 else table.scheme_of(custom)[0]
        p = prof(ws)
        rec.update(S=S, custom=custom, scheme=scheme, ws=ws, profile=p.name, tile=lt["tile"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                cp = mmaemu.explain(a, b.t(), sa, sb, profile=p, splitk="auto", **kw)
                auto_error = None
            except E.SplitKUnknownError as e:            # off the table's grid, no envelope
                cp, auto_error = None, f"SplitKUnknownError: {str(e)[:120]}"
            split = 1 if S == 1 else (S, scheme)
            try:
                y = mmaemu.scaled_mm(a, b.t(), sa, sb, profile=p, splitk=split, **kw)
                rec["backend"] = mmaemu.explain(a, b.t(), sa, sb, profile=p, splitk=split, **kw).backend  # what ran
                got = sha_bits(y)
                rec["verdict"] = "equal" if got == r["full_sha256"] else "differ"
                if rec["verdict"] == "differ":
                    rec["sha256"] = got
            except Exception as e:  # noqa: BLE001
                rec["verdict"] = f"mmaemu_error {type(e).__name__}: {str(e)[:160]}"
        if cp is None:
            rec["auto"] = dict(error=auto_error)
        elif cp.validation != "ok":
            rec["auto"] = dict(error=cp.validation[0])
        else:
            d = cp.split
            rec["auto"] = dict(S=d.S, scheme=d.scheme, source=d.source, verified=d.verified)
            rec["auto_agrees"] = (d.S, d.scheme) == (S, scheme)
        rec["seconds"] = round(time.monotonic() - t0, 2)
        res.append(rec)
        print(f"{n} {j['id']} {src} {j['M']}x{j['N']}x{j['K']} {j['fa']}x{j['fb']}->{j['out']} fast={j['fast']} S={S} "
              f"{scheme} ws={ws}: {rec['verdict']} auto={rec['auto']} ({rec['seconds']}s)", flush=True)
        del a, b, a8, b8
    return res


def replay_m2(truth: Path, dev, calls=None, backend="reference") -> list:
    import torch

    import mmaemu
    case_file = truth / "data" / "evo2" / "evo2_1b_w2k.json"
    cases = json.loads(case_file.read_text())["cases"]
    hw = {}
    for name in ("m2_torch", "m2_te"):
        for c in json.loads((LOG / "results" / "m2" / f"{name}.json").read_text())["calls"]:
            hw[c["id"]] = c
    res = []
    npz = {}
    for i, case in enumerate(cases):
        if calls is not None and i not in calls:
            continue
        ins = []
        for k in ("a", "b"):
            spec = case["inputs"][k]
            if spec["file"] not in npz:
                npz[spec["file"]] = np.load(truth / "data" / "evo2" / spec["file"])
            x = np.ascontiguousarray(npz[spec["file"]][spec["key"]])
            assert hashlib.sha256(x.tobytes()).hexdigest() == spec["sha256"], (case["id"], k)
            ins.append(torch.from_numpy(x).to(dev))
        x8, w8 = ins[0].view(torch.float8_e4m3fn), ins[1].view(torch.float8_e4m3fn)
        p = case["params"]
        inv = torch.tensor([p["fp8_scale_inv_in"], p["fp8_scale_inv_w"]], dtype=torch.int32).view(torch.float32)
        inv = inv.to(dev)
        one = torch.ones((), device=dev)
        c = f"call{i:02d}"
        for fast in (True, False):
            tag = "fast" if fast else "slow"
            te = "" if fast else "_split"
            for cid, profile, sa, sb, od in ((f"{c}.acc_{tag}", "hopper", one, one, torch.float32),
                                             (f"{c}.y_{tag}", "hopper", inv[0], inv[1],
                                              torch.bfloat16),
                                             (f"{c}.te_acc{te}", "hopper-te", one, one, torch.float32),
                                             (f"{c}.te_y{te}", "hopper-te", inv[0], inv[1], torch.bfloat16)):
                t0 = time.monotonic()
                runs = hw[cid]["runs"]
                rec = dict(id=cid, profile=profile, M=int(x8.shape[0]), N=int(w8.shape[0]), K=int(x8.shape[1]),
                           fast=fast, out=str(od).split(".")[-1], runs_equal=len({r["sha256"] for r in runs}) == 1)
                with warnings.catch_warnings():
                    warnings.simplefilter("error")          # a validated profile resolving S = 1 must not warn
                    try:
                        y = mmaemu.scaled_mm(x8, w8.t(), sa, sb, out_dtype=od, use_fast_accum=fast, profile=profile,
                                             splitk="auto", backend=backend)
                        cp = mmaemu.explain(x8, w8.t(), sa, sb, out_dtype=od, use_fast_accum=fast, profile=profile,
                                            splitk="auto", backend=backend)
                        rec["backend"] = cp.backend
                        rec["auto"] = dict(S=cp.split.S, scheme=cp.split.scheme, source=cp.split.source)
                        rec["shape_equal"] = list(y.shape) == runs[0]["shape"]
                        rec["verdict"] = "equal" if sha_bits(y) == runs[0]["sha256"] else "differ"
                    except Exception as e:  # noqa: BLE001
                        rec["verdict"] = f"mmaemu_error {type(e).__name__}: {str(e)[:160]}"
                rec["seconds"] = round(time.monotonic() - t0, 2)
                res.append(rec)
                print(f"m2 {cid} {profile}: {rec['verdict']} {rec.get('auto')} ({rec['seconds']}s)", flush=True)
        del x8, w8, ins
    return res


def summarize(jobs: list, m2: list) -> dict:
    v = collections.Counter(j["verdict"] for j in jobs)
    by = collections.Counter((j["src"], "fast" if j["fast"] else "slow", "split" if (j.get("S") or 1) > 1 else
                              "unsplit", j["verdict"]) for j in jobs)
    auto = collections.Counter("rejected" if "auto" not in j else ("unknown" if "error" in j["auto"] else
                               ("agrees" if j["auto_agrees"] else "DISAGREES")) for j in jobs)
    auto_m6 = collections.Counter(("unknown" if "error" in j["auto"] else ("agrees" if j["auto_agrees"] else
                                   "DISAGREES")) for j in jobs if j["src"] == "m6_live" and "auto" in j)
    s = dict(live_jobs=len(jobs), live_verdicts=dict(v), live_by=[[*k, n] for k, n in sorted(by.items())],
             live_auto=dict(auto), live_auto_m6=dict(auto_m6),
             live_passes_equal=dict(collections.Counter(str(j.get("passes_equal")) for j in jobs)),
             live_split_equal=sum(1 for j in jobs if j["verdict"] == "equal" and j.get("S", 1) > 1),
             live_unsplit_equal=sum(1 for j in jobs if j["verdict"] == "equal" and j.get("S") == 1),
             live_backends=dict(collections.Counter(str(j.get("backend")) for j in jobs if j["verdict"] == "equal")))
    if m2:
        s.update(m2_outputs=len(m2), m2_verdicts=dict(collections.Counter(r["verdict"] for r in m2)),
                 m2_auto=dict(collections.Counter(f"S={r['auto']['S']} {r['auto']['source']}" for r in m2
                                                  if "auto" in r)),
                 m2_runs_equal=sum(r["runs_equal"] for r in m2),
                 m2_backends=dict(collections.Counter(str(r.get("backend")) for r in m2)))
    ok = (set(v) <= {"equal", "rejected_as_hardware"} and auto.get("DISAGREES", 0) == 0 and
          auto_m6.get("unknown", 0) == 0 and all(j.get("passes_equal", True) for j in jobs) and
          (not m2 or (set(r["verdict"] for r in m2) == {"equal"} and all(r["auto"]["S"] == 1 for r in m2))))
    s["PASS"] = bool(ok)
    return s


def environment(dev, backend="reference") -> dict:
    import torch
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", *DEPENDS],
                           capture_output=True, text=True).stdout.strip()
    return dict(mmaemu_commit=head, dirty=bool(dirty), dirty_scope=list(DEPENDS), torch=torch.__version__,
                device=str(dev),
                device_name=torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
                backend=backend, python=sys.version.split()[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backend", default="reference",
                    help="mmaemu backend: reference (default), auto, cuda:h1, cuda:h4")
    ap.add_argument("--truth", type=Path, default=None)
    ap.add_argument("--parts", default="live,m2")
    ap.add_argument("--only", default=None)
    ap.add_argument("--m2-calls", default=None, help="comma-separated Evo 2 call indices (default: all 21)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--part", type=Path, default=None, help="write this chunk's records here")
    ap.add_argument("--merge", type=Path, nargs="*", default=None, help="merge chunk files into --out")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--live", type=Path, default=LIVE, help="live-call recording (default: the probe's)")
    a = ap.parse_args()
    if a.merge is not None:
        parts = [json.loads(p.read_text()) for p in a.merge]
        envs = {json.dumps(p["environment"], sort_keys=True) for p in parts}
        assert len(envs) == 1, f"chunks ran in different environments: {envs}"
        jobs = sorted((j for p in parts for j in p["live"]), key=lambda j: j["id"])
        ids = [j["id"] for j in jobs]
        assert len(ids) == len(set(ids)), "a job is in two chunks"
        n_all = len(live_jobs(a.live))
        assert len(jobs) == n_all, f"{len(jobs)} of {n_all} live jobs replayed"
        m2 = sorted((r for p in parts for r in p["m2"]), key=lambda r: r["id"])
        assert len(m2) == len({r["id"] for r in m2}) and len(m2) in (0, 168), f"{len(m2)} M2 outputs"
        live_path = a.live.resolve()
        log = (live_path.parents[1] if a.live != LIVE else LOG)
        log = log.relative_to(REPO).as_posix() if log.is_relative_to(REPO) else str(log)
        rep = dict(format=FORMAT, environment=parts[0]["environment"], log=log,
                   live_sha256=hashlib.sha256(live_path.read_bytes()).hexdigest(),
                   table_id=TABLE_ID, summary=summarize(jobs, m2), live=jobs, m2=m2)
        a.out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
        print(json.dumps(rep["summary"], indent=1))
        return
    import torch
    dev = torch.device(a.device)
    parts = set(a.parts.split(","))
    t0 = time.monotonic()
    jobs, m2 = [], []
    if "live" in parts:
        items = live_jobs(a.live)
        if a.only:
            keep = set(a.only.split(","))
            items = [t for t in items if t[1]["id"] in keep]
        items = items[a.offset:] if a.limit is None else items[a.offset:a.offset + a.limit]
        jobs = replay_live(items, dev, a.backend)
    if "m2" in parts:
        if a.truth is None:
            raise SystemExit("--parts m2 needs --truth $MMAEMU_TRUTH_DIR")
        m2 = replay_m2(a.truth, dev, None if a.m2_calls is None else {int(x) for x in a.m2_calls.split(",")},
                       a.backend)
    rec = dict(environment=environment(dev, a.backend), live=jobs, m2=m2, seconds=round(time.monotonic() - t0, 1))
    if a.part:
        a.part.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(summarize(jobs, m2), indent=1))


if __name__ == "__main__":
    main()
