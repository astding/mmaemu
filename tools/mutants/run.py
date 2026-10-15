#!/usr/bin/env python3
"""T4 mutation matrix (docs/testing.md): apply each mutant of tools/mutants/catalogue.py to a TEMPORARY COPY of the
repository and run the test tier that must kill it.

usage:
    python tools/mutants/run.py --tier cpu [--jobs 4] [--only REGEX] [--out docs/validation_logs/<date>/mutants.json]
    $MMAEMU_GPU_LOCK timeout 600 python tools/mutants/run.py --tier gpu
    [--tests "tests/test_hw_mini.py ..."] [--pytest-args "..."] [--keep]

Tiers
  cpu: T0 + T1 = pytest <tests> -m "not gpu and not slow and not truth" with CUDA_VISIBLE_DEVICES='' (default <tests>:
       the whole tests/ directory of the copy).
  gpu: pytest -x -m gpu on tools/mutants/test_gpu.py (direct NVRTC v1 conformance, NaN propagation, the pinned NVRTC
       options) and the T3 files tests/test_kernels_ieee.py, tests/test_fast_kernels.py, tests/test_cuda_runtime.py,
       tests/test_hw_mini.py (where present); their fatbin tests skip in the copy. The committed fatbins are removed
       from the copy, so the (mutated) sources are compiled by NVRTC into a private MMAEMU_CACHE_DIR.
Outcome per mutant: killed (pytest exit 1; the first failing test is recorded), survived (exit 0), equivalent
(survived, and the catalogue cites evidence that the mutant is unobservable), blocked (the edit's target does not
exist in this checkout, e.g. a stripped tree), error (anything else: collection errors, timeouts,
an anchor that matches the wrong number of times). The report is JSON (default
docs/validation_logs/<date>/mutants[_gpu].json). Exit 0 iff every identity mutant survived and every other mutant of
the tier was killed or is equivalent.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from catalogue import GROUPS_MUST_KILL, MUTANTS  # noqa: E402

COPY_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "build", "dist", ".pytest_cache")
LINK_DIRS = ("tests/data", "recordings", "results")         # large read-only data: symlinked, not copied
GPU_TESTS = ("tools/mutants/test_gpu.py", "tests/test_kernels_ieee.py", "tests/test_fast_kernels.py", "tests/test_cuda_runtime.py",
             "tests/test_hw_mini.py")


class EditError(Exception):
    pass


class Blocked(Exception):
    pass


def make_copy(dst: Path, package_src: Path | None = None, tree: Path = REPO):
    for item in tree.iterdir():
        if item.name in (".git", "build", "dist", "results", "recordings") or item.name.startswith("."):
            continue
        if item.is_dir():
            shutil.copytree(item, dst / item.name, ignore=COPY_IGNORE, symlinks=True)
        else:
            shutil.copy2(item, dst / item.name)
    for rel in LINK_DIRS:
        src = tree / rel
        if src.exists():
            target = dst / rel
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(src, target)
    if package_src is not None:          # development aid: test another implementation of the frozen API
        shutil.rmtree(dst / "mmaemu", ignore_errors=True)
        shutil.copytree(package_src, dst / "mmaemu", ignore=COPY_IGNORE)
    shutil.copy2(HERE / "impl.py", dst / "mmaemu" / "_mutation_impl.py")


def apply_edits(dst: Path, name: str, edits):
    for e in edits:
        kind = e[0]
        if kind == "remove":
            p = dst / e[1]
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
            continue
        p = dst / e[1]
        if not p.exists():
            raise Blocked(f"{e[1]} does not exist in this checkout")
        s = p.read_text()
        if kind == "override":
            s += f"\n\n# ---- tools/mutants: {name} ----\n{e[2]}"
        elif kind == "replace":
            old, new, count = e[2], e[3], e[4]
            n = s.count(old)
            if n == 0 or (count is not None and n != count):
                raise EditError(f"{e[1]}: {old!r} found {n} times, expected {count}")
            s = s.replace(old, new)
        elif kind == "regex":
            pat, rep, count = e[2], e[3], e[4]
            s2, n = re.subn(pat, rep, s)
            if n == 0 or (count is not None and n != count):
                raise EditError(f"{e[1]}: /{pat}/ matched {n} times, expected {count}")
            s = s2
        else:
            raise EditError(f"unknown edit kind {kind}")
        p.write_text(s)


def run_one(m: dict, args) -> dict:
    t0 = time.monotonic()
    res = dict(name=m["name"], group=m["group"], tier=m["tier"], source=m["source"], description=m["desc"])
    tmp = Path(tempfile.mkdtemp(prefix=f"mmaemu-mut-{m['name']}-", dir=args.tmp))
    try:
        make_copy(tmp, Path(args.package_src) if args.package_src else None,
                  Path(args.source_tree) if args.source_tree else REPO)
        try:
            apply_edits(tmp, m["name"], m["edits"])
        except Blocked as e:
            res.update(status="blocked", reason=str(e), seconds=round(time.monotonic() - t0, 1))
            return res
        except EditError as e:
            res.update(status="error", reason=str(e), seconds=round(time.monotonic() - t0, 1))
            return res
        env = dict(os.environ)
        env["PYTHONPATH"] = str(tmp)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["MMAEMU_CACHE_DIR"] = str(tmp / ".cache")
        env.setdefault("MMAEMU_T1_THREADS", str(args.threads))
        env["OMP_NUM_THREADS"] = str(args.threads)        # every test file: no oversubscription across jobs
        if args.tier == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
            marker = "not gpu and not slow and not truth"
            tests = args.tests.split() if args.tests else ["tests"]
        else:
            # Never let an inherited release directory hide the mutated sources.
            env["MMAEMU_FATBIN_DIR"] = str(tmp / "missing-fatbins")
            env.pop("MMAEMU_TEST_FATBIN_DIR", None)
            env.setdefault("MMAEMU_TEST_REF_CACHE", str(REPO / "build" / "test_refs"))
            marker = "gpu"
            tests = args.tests.split() if args.tests else [t for t in GPU_TESTS if (tmp / t).exists()]
        # sanity: the mutated package imports and is the copy
        chk = subprocess.run([sys.executable, "-c", "import mmaemu, sys; print(mmaemu.__file__)"], cwd=tmp, env=env,
                             capture_output=True, text=True, timeout=300)
        if chk.returncode != 0 or str(tmp) not in chk.stdout:
            res.update(status="error", reason=f"import check failed: {chk.stdout[-300:]} {chk.stderr[-600:]}",
                       seconds=round(time.monotonic() - t0, 1))
            return res
        cmd = [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "-rf", "-m", marker,
               *tests, *args.pytest_args.split()]
        try:
            pr = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True, timeout=args.timeout)
            rc, out = pr.returncode, pr.stdout + pr.stderr
        except subprocess.TimeoutExpired as e:
            rc, out = -9, f"timeout after {args.timeout} s\n{(e.stdout or b'')[-2000:]!r}"
        failed = re.findall(r"^FAILED (\S+)", out, re.M)
        summary = [ln for ln in out.splitlines() if re.search(r"\d+ (passed|failed|error)", ln)]
        res.update(rc=rc, killed_by=failed[:3], pytest_summary=summary[-1:] if summary else [],
                   tests=tests, marker=marker)
        if rc == 1:
            res["status"] = "killed"
        elif rc == 0:
            res["status"] = "equivalent" if m.get("equivalent_evidence") else "survived"
            if m.get("equivalent_evidence"):
                res["evidence"] = m["equivalent_evidence"]
        else:
            res["status"] = "error"
            res["reason"] = f"pytest exit {rc}: {out[-1500:]}"
        res["seconds"] = round(time.monotonic() - t0, 1)
        return res
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)


def verdict(results) -> tuple[bool, list[str]]:
    problems = []
    for r in results:
        if r["group"] == "identity":
            if r["status"] != "survived":
                problems.append(f"identity mutant {r['name']}: {r['status']} (the injection itself is unsound)")
        elif r["group"] in GROUPS_MUST_KILL:
            if r["status"] not in ("killed", "equivalent"):
                problems.append(f"{r['name']} ({r['group']}): {r['status']}")
    return not problems, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=("cpu", "gpu"), required=True)
    ap.add_argument("--only")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4, help="torch threads per pytest (MMAEMU_T1_THREADS)")
    ap.add_argument("--tests", default="", help="test paths (default: tests/ for cpu, the T3 files for gpu)")
    ap.add_argument("--pytest-args", default="")
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--tmp", default=None, help="parent dir of the temporary copies")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--source-tree", default=None,
                    help="copy this tree instead of the repository (e.g. a scratch checkout of another branch)")
    ap.add_argument("--package-src", default=None,
                    help="copy the mmaemu package from this directory instead of the repo (development aid; the "
                         "report records it)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    sel = [m for m in MUTANTS if m["tier"] == a.tier and (not a.only or re.search(a.only, m["name"]))]
    if a.tier == "gpu":
        a.jobs = 1
    t0 = time.monotonic()
    results = []
    with cf.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        futs = {ex.submit(run_one, m, a): m for m in sel}
        for f in cf.as_completed(futs):
            r = f.result()
            results.append(r)
            print(f"{r['name']:34s} {r['group']:9s} {r['status']:10s} {r.get('seconds', 0):7.1f}s "
                  f"{(r.get('killed_by') or [''])[0] if r['status'] == 'killed' else r.get('reason', '')[:160]}",
                  flush=True)
    order = {m["name"]: i for i, m in enumerate(MUTANTS)}
    results.sort(key=lambda r: order[r["name"]])
    ok, problems = verdict(results)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    import mmaemu  # noqa: WPS433 -- provenance only (the package under test is the copy)
    rep = dict(format="mmaemu-mutants/1", tier=a.tier, date=time.strftime("%Y-%m-%d %H:%M:%S"),
               repo_commit=commit + ("+dirty" if dirty else ""), mmaemu_file=mmaemu.__file__,
               mmaemu_version=getattr(mmaemu, "__version__", None), python=sys.version.split()[0],
               argv=sys.argv, package_src=a.package_src, source_tree=a.source_tree, seconds=round(time.monotonic() - t0, 1), verdict="PASS" if ok else "FAIL",
               problems=problems, counts={s: sum(r["status"] == s for r in results) for s in
                                          ("killed", "survived", "equivalent", "blocked", "error")},
               mutants=results)
    out = Path(a.out) if a.out else REPO / "docs" / "validation_logs" / time.strftime("%Y-%m-%d") / (
        "mutants.json" if a.tier == "cpu" else "mutants_gpu.json")
    if out.exists() and a.only:
        old = json.loads(out.read_text())
        keep = {r["name"]: r for r in old.get("mutants", [])}
        keep.update({r["name"]: r for r in results})
        merged = sorted(keep.values(), key=lambda r: order.get(r["name"], 1e9))
        ok, problems = verdict(merged)
        rep.update(mutants=merged, verdict="PASS" if ok else "FAIL", problems=problems,
                   counts={s: sum(r["status"] == s for r in merged) for s in
                           ("killed", "survived", "equivalent", "blocked", "error")},
                   note="merged with the previous report of this tier (run with --only)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print(f"{rep['verdict']}: {rep['counts']} -> {out}")
    for p in problems:
        print("  " + p)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
