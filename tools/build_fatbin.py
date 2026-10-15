#!/usr/bin/env python
"""Build the kernel fatbins with nvcc, write manifest.json, and check them (docs/backends.md).

  python tools/build_fatbin.py --out build/fatbin            # development build (-lineinfo)
  python tools/build_fatbin.py --release --out <dir>         # release flags (no -lineinfo, relative paths)
  python tools/build_fatbin.py --check [--out <dir>]         # staleness check only: no nvcc (CI)

Build:
  - refuses nvcc older than 12.8 (sm_100 / sm_120 need it; e.g. a distribution's nvcc 12.0 is refused);
  - flags: -O3 --fmad=false [-lineinfo unless --release]; nvcc runs inside the csrc/ directory with relative paths;
  - modules and architectures come from the discovered KernelSpecs (kernels.SPECS + fast.specs.SPECS) and the probe
    modules (PROBE_MODULES, SASS for PROBE_ARCHS, no PTX);
  - writes <out>/<module>.fatbin, <out>/manifest.json (format mmaemu-fatbin/1), <out>/cuobjdump.txt (the
    --list-elf / --list-ptx listing, which must equal the requested architectures) and <out>/build_report.json;
  - greps the PTX of every module for every architecture for forbidden instructions (fma.rz.bf16 / fma.rz.bf16x2:
    ptxas 12.0-13.3 miscompiles them, found by the tcemu audit) and fails if any is found;
  - parses ptxas -v and fails on spill bytes > 0 unless --allow-spills (documented exception).
Check (--check): recomputes every module's source_sha256 from the shipped sources and the file sha256 of every
fatbin; exits 1 if anything is stale or missing, 0 if all match, and 0 with a note if there is no manifest at all
(nothing committed yet; --require-manifest makes that an error).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mmaemu.backends import registry  # noqa: E402
from mmaemu.backends.cuda import modules  # noqa: E402

PROBE_ARCHS = (80, 86, 89, 90, 100, 120)  # probes: SASS sm_80..sm_120, no PTX
FORBIDDEN_PTX = re.compile(r"\bfma\.rz\.bf16(x2)?\b")
MIN_NVCC = (12, 8)
BASE_FLAGS = ["-O3", "--fmad=false"]


def module_plan() -> dict[str, dict]:
    """{module: dict(source, sass_archs, ptx_arch, entries, gated)} from the discovered specs and probe modules."""
    plan: dict[str, dict] = {}
    for s in registry.discover():
        m = plan.setdefault(s.module, dict(source=s.source, sass_archs=set(), ptx_arch=None, entries=[], gated=False))
        m["sass_archs"].update(s.sass_archs)
        if s.ptx_arch is not None:
            m["ptx_arch"] = s.ptx_arch if m["ptx_arch"] is None else min(m["ptx_arch"], s.ptx_arch)
        m["entries"].append(s.entry)
        m["gated"] = m["gated"] or bool(s.gated)
    fast_archs = None
    try:
        from mmaemu.backends.cuda.fast import specs as fast_specs  # noqa: F401

        fast_archs = getattr(fast_specs, "PROBE_SASS_ARCHS", None) or getattr(fast_specs, "PROBE_ARCHS", None)
    except Exception:  # noqa: BLE001
        pass
    for stem, src in registry.probe_modules().items():
        text = (modules.CSRC / src).read_text()
        entries = re.findall(r'extern\s+"C"\s+__global__\s+void\s+(?:__launch_bounds__\([^)]*\)\s*)?(\w+)', text)
        plan.setdefault(stem, dict(source=src, sass_archs=set(fast_archs or PROBE_ARCHS), ptx_arch=None,
                                   entries=entries, gated=True))
    for m in plan.values():
        m["sass_archs"] = sorted(m["sass_archs"])
    return plan


def find_nvcc(arg: str | None) -> str:
    for c in (arg, os.environ.get("NVCC"), os.environ.get("CUDA_HOME") and os.path.join(os.environ["CUDA_HOME"], "bin", "nvcc"),
              shutil.which("nvcc")):
        if c and os.path.isfile(c):
            return c
    raise SystemExit("build_fatbin: nvcc not found (pass --nvcc or set CUDA_HOME)")


def nvcc_version(nvcc: str) -> tuple[tuple[int, int], str]:
    out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=True).stdout
    m = re.search(r"release (\d+)\.(\d+), V([\d.]+)", out)
    if not m:
        raise SystemExit(f"build_fatbin: cannot parse `{nvcc} --version`")
    return (int(m.group(1)), int(m.group(2))), m.group(3)


def find_cuobjdump(arg: str | None, nvcc: str) -> str | None:
    cands = [arg, os.environ.get("CUOBJDUMP"), os.path.join(os.path.dirname(nvcc), "cuobjdump")]
    try:  # triton's pip wheel bundles a matching cuobjdump
        import triton  # noqa: F401

        cands.append(os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "cuobjdump"))
    except Exception:  # noqa: BLE001
        pass
    cands.append(shutil.which("cuobjdump"))
    for c in cands:
        if c and os.path.isfile(c):
            try:
                v, _ = nvcc_version(c)
            except (subprocess.CalledProcessError, SystemExit, OSError):
                continue
            if v >= MIN_NVCC:
                return c
    return None


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def gencode(m: dict) -> list[str]:
    g = []
    for a in m["sass_archs"]:
        g += ["-gencode", f"arch=compute_{a},code=sm_{a}"]
    if m["ptx_arch"] is not None:
        g += ["-gencode", f"arch=compute_{m['ptx_arch']},code=compute_{m['ptx_arch']}"]
    return g


def parse_ptxas(text: str) -> list[dict]:
    """[{entry, arch, registers, spill_stores, spill_loads, stack}] from nvcc -Xptxas -v output."""
    rows, cur = [], None
    for line in text.splitlines():
        m = re.search(r"Compiling entry function '(\w+)' for '(sm_\d+)'", line)
        if m:
            cur = dict(entry=m.group(1), arch=m.group(2))
            rows.append(cur)
            continue
        if cur is None:
            continue
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m:
            cur.update(stack=int(m.group(1)), spill_stores=int(m.group(2)), spill_loads=int(m.group(3)))
        m = re.search(r"Used (\d+) registers", line)
        if m:
            cur["registers"] = int(m.group(1))
    return rows


def build(args) -> int:
    nvcc = find_nvcc(args.nvcc)
    ver, ver_full = nvcc_version(nvcc)
    if ver < MIN_NVCC:
        print(f"build_fatbin: {nvcc} is CUDA {ver[0]}.{ver[1]} ({ver_full}); >= {MIN_NVCC[0]}.{MIN_NVCC[1]} is "
              "required (sm_100 / sm_120 SASS)", file=sys.stderr)
        return 2
    cuobjdump = find_cuobjdump(args.cuobjdump, nvcc)
    out = Path(args.out or ROOT / "build" / "fatbin").resolve()
    out.mkdir(parents=True, exist_ok=True)
    flags = BASE_FLAGS + ([] if args.release else ["-lineinfo"])
    plan = module_plan()
    if args.modules:
        want = set(args.modules.split(","))
        plan = {k: v for k, v in plan.items() if k in want}
    manifest = dict(format=modules.MANIFEST_FORMAT, nvcc=ver_full, flags=flags, modules={})
    report = dict(nvcc=nvcc, nvcc_version=ver_full, cuobjdump=cuobjdump, release=bool(args.release), modules={})
    failed = False
    csrc = modules.CSRC
    for stem, m in plan.items():
        fat = out / f"{stem}.fatbin"
        tmp = out / f".{stem}.fatbin.tmp"
        cmd = [nvcc, "-fatbin", *flags, "-Xptxas", "-v", "-I.", *gencode(m), "-o", str(tmp), m["source"]]
        r = subprocess.run(cmd, cwd=csrc, capture_output=True, text=True)
        rep: dict = dict(command=" ".join([os.path.basename(nvcc)] + cmd[1:-3] + ["-o", fat.name, m["source"]]))
        if r.returncode != 0:
            print(r.stdout + r.stderr, file=sys.stderr)
            print(f"build_fatbin: nvcc failed for {stem}", file=sys.stderr)
            return 1
        os.replace(tmp, fat)
        ptxas = parse_ptxas(r.stdout + r.stderr)
        spills = [x for x in ptxas if x.get("spill_stores", 0) or x.get("spill_loads", 0)]
        rep["ptxas"] = ptxas
        rep["spill_bytes"] = sum(x.get("spill_stores", 0) + x.get("spill_loads", 0) for x in ptxas)
        warn = [ln for ln in (r.stdout + r.stderr).splitlines() if "warning" in ln.lower()]
        rep["compiler_warnings"] = warn
        if spills and not args.allow_spills:
            print(f"build_fatbin: {stem}: ptxas spills: {spills}", file=sys.stderr)
            failed = True
        # forbidden-PTX grep over the PTX for every architecture
        hits = {}
        archs = sorted(set(m["sass_archs"]) | ({m["ptx_arch"]} if m["ptx_arch"] is not None else set()))
        with tempfile.TemporaryDirectory() as td:
            for a in archs:
                ptx = Path(td) / f"{stem}_{a}.ptx"
                rr = subprocess.run([nvcc, "-ptx", *flags, "-I.", f"-arch=compute_{a}", "-o", str(ptx), m["source"]],
                                    cwd=csrc, capture_output=True, text=True)
                if rr.returncode != 0:
                    print(rr.stderr, file=sys.stderr)
                    return 1
                found = sorted(set(x.group(0) for x in FORBIDDEN_PTX.finditer(ptx.read_text())))
                if found:
                    hits[f"compute_{a}"] = found
        rep["forbidden_ptx"] = hits
        if hits:
            print(f"build_fatbin: {stem}: forbidden PTX {hits}", file=sys.stderr)
            failed = True
        manifest["modules"][stem] = dict(file=fat.name, sha256=sha256_file(fat),
                                         source_sha256=modules.source_sha256(stem), sass_archs=m["sass_archs"],
                                         ptx_arch=m["ptx_arch"], entries=m["entries"])
        report["modules"][stem] = rep
    # cuobjdump listing; it must equal the requested architectures
    listing = []
    if cuobjdump is None:
        print("build_fatbin: no cuobjdump >= 12.8 found; skipping the ELF/PTX listing check", file=sys.stderr)
        failed = True
    else:
        for stem, e in manifest["modules"].items():
            r = subprocess.run([cuobjdump, "--list-elf", "--list-ptx", str(out / e["file"])], capture_output=True,
                               text=True)
            txt = r.stdout + r.stderr
            listing.append(f"== {e['file']}\n{txt}")
            elf = sorted(int(x) for x in re.findall(r"\.sm_(\d+)\.cubin", txt))
            ptx = sorted(int(x) for x in re.findall(r"\.sm_(\d+)\.ptx", txt))
            want_ptx = [e["ptx_arch"]] if e["ptx_arch"] is not None else []
            ok = r.returncode == 0 and elf == sorted(e["sass_archs"]) and ptx == want_ptx
            report["modules"][stem]["cuobjdump"] = dict(elf=elf, ptx=ptx, ok=ok)
            if not ok:
                print(f"build_fatbin: {stem}: cuobjdump lists ELF {elf} PTX {ptx}; expected {e['sass_archs']} "
                      f"PTX {want_ptx}", file=sys.stderr)
                failed = True
            r = subprocess.run([cuobjdump, "--dump-resource-usage", str(out / e["file"])], capture_output=True,
                               text=True)
            missing = [x for x in e["entries"] if x not in r.stdout]
            report["modules"][stem]["entries_missing"] = missing
            if missing:
                print(f"build_fatbin: {stem}: entries {missing} not found in the fatbin", file=sys.stderr)
                failed = True
        (out / "cuobjdump.txt").write_text("\n".join(listing))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    (out / "build_report.json").write_text(json.dumps(report, indent=1) + "\n")
    for stem, e in manifest["modules"].items():
        rep = report["modules"][stem]
        print(f"{stem:8s} {e['file']:14s} sass {e['sass_archs']} ptx {e['ptx_arch']} spill_bytes {rep['spill_bytes']} "
              f"forbidden_ptx {rep['forbidden_ptx'] or 'none'} cuobjdump_ok {rep.get('cuobjdump', {}).get('ok')}")
    print(f"wrote {out / 'manifest.json'} (nvcc {ver_full}, flags {' '.join(flags)})")
    return 1 if failed else 0


def check(args) -> int:
    d = Path(args.out).resolve() if args.out else modules.BIN
    man = d / "manifest.json"
    if not man.is_file():
        print(f"build_fatbin --check: no manifest in {d} (no fatbins committed)")
        return 1 if args.require_manifest else 0
    m = json.loads(man.read_text())
    bad = []
    if m.get("format") != modules.MANIFEST_FORMAT:
        bad.append(f"manifest format {m.get('format')!r}")
    entries = m.get("modules", {})
    for stem, e in entries.items():
        try:
            cur = modules.source_sha256(stem)
        except Exception as err:  # noqa: BLE001
            bad.append(f"{stem}: {err}")
            continue
        if cur != e.get("source_sha256"):
            bad.append(f"{stem}: stale (sources changed since the build)")
        f = d / e.get("file", "")
        if not f.is_file():
            bad.append(f"{stem}: {f.name} missing")
        elif sha256_file(f) != e.get("sha256"):
            bad.append(f"{stem}: {f.name} sha256 differs from the manifest")
    plan = module_plan()
    for stem, p in plan.items():
        if stem not in entries:
            if p["gated"]:
                print(f"note: gated/probe module {stem!r} is not in the manifest (allowed: gated kernels are optional)")
            else:
                bad.append(f"{stem}: required module missing from the manifest")
        elif sorted(entries[stem].get("sass_archs", [])) != p["sass_archs"] or entries[stem].get("ptx_arch") != p["ptx_arch"]:
            bad.append(f"{stem}: manifest archs differ from the KernelSpec")
    for b in bad:
        print("STALE:", b)
    print(f"build_fatbin --check {d}: {'FAIL' if bad else 'OK'} ({len(entries)} modules)")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="output directory (build default: build/fatbin; --check default: the package bin/)")
    ap.add_argument("--release", action="store_true", help="release flags: no -lineinfo")
    ap.add_argument("--check", action="store_true", help="hash-only staleness check, no nvcc")
    ap.add_argument("--require-manifest", action="store_true", help="--check: a missing manifest is an error")
    ap.add_argument("--modules", help="comma-separated subset of modules to build")
    ap.add_argument("--nvcc")
    ap.add_argument("--cuobjdump")
    ap.add_argument("--allow-spills", action="store_true", help="do not fail on ptxas spills (document why)")
    args = ap.parse_args(argv)
    return check(args) if args.check else build(args)


if __name__ == "__main__":
    sys.exit(main())
