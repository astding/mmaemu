#!/usr/bin/env python3
"""Check the contents of a built mmaemu wheel and sdist (docs/release.md).

Usage
  python tools/check_dist.py dist/*                          # strict: the wheel must carry the release fatbins
  python tools/check_dist.py dist/* --allow-missing-fatbins  # before the release-candidate freeze (no bin/ yet)
  python tools/check_dist.py dist/* --require-fatbins --tag v0.1.0
  python tools/check_dist.py --verify-manifest               # committed fatbins vs bin/manifest.json (no build)
  python tools/check_dist.py sdist-wheel/*.whl dist/*.tar.gz --from-sdist   # a wheel built from the sdist

What is checked
  wheel  mmaemu-<ver>-py3-none-any.whl; only mmaemu/ and the dist-info; every file of the source tree's mmaemu/
         (git-tracked when the tree is a git checkout) and nothing else; every manifest-listed fatbin with a
         matching sha256 and no unlisted fatbin; a source for every fatbin module; csrc, platform data and py.typed;
         LICENSE, NOTICE and LICENSES/* in dist-info/licenses (byte-equal to the tree); PEP 639 metadata with
         License-Expression Apache-2.0; version == mmaemu/_version.py (== tag); size <= --max-wheel-mb.
  sdist  mmaemu-<ver>.tar.gz; mmaemu/ without bin/; tests/ (incl. the toolbox licence); the licence and metadata
         files; no fatbins, recordings/, results/, docs/, tools/ or bench/.
Exit status 0 when every check passes, 1 otherwise, 2 on usage errors.
"""
from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PKG = "mmaemu"
BIN = "mmaemu/backends/cuda/bin/"
CSRC = "mmaemu/backends/cuda/csrc/"
PDATA = "mmaemu/platform/data/"
MANIFEST = BIN + "manifest.json"
MANIFEST_FORMAT = "mmaemu-fatbin/1"
LICENCE_FILES = ("LICENSE", "NOTICE", "LICENSES/BSD-2-Clause-MATLAB-tensor-core.txt")
SDIST_TOP_FILES = ("pyproject.toml", "README.md", "CHANGELOG.md", "CITATION.cff", "PKG-INFO") + LICENCE_FILES
TOOLBOX_LICENCES = ("tests/data/toolbox/LICENCE.MATLAB-tensor-core", "tests/data/LICENCE.MATLAB-tensor-core")
SDIST_FORBIDDEN_DIRS = ("mmaemu/backends/cuda/bin/", "recordings/", "results/", "docs/", "tools/", "bench/")
JUNK = re.compile(r"(^|/)__pycache__/|\.py[co]$")
VERSION_RE = re.compile(r"""^__version__\s*(?::\s*str\s*)?=\s*["']([^"']+)["']""", re.M)


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))
        print(f"{status:4s} {name}" + (f": {detail}" if detail else ""), flush=True)

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        self.add("PASS" if ok else "FAIL", name, detail)
        return ok

    @property
    def failed(self) -> int:
        return sum(1 for s, _, _ in self.rows if s == "FAIL")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def norm_version(v: str) -> str:
    try:
        from packaging.version import Version
        return str(Version(v))
    except Exception:  # packaging missing or invalid version: compare literally
        return v


def tree_files(prefix: str) -> set[str] | None:
    """Files of the source tree under `prefix` (posix, repo-relative): git-tracked if ROOT is a git checkout."""
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--", prefix], check=True,
                             capture_output=True).stdout
        files = {f for f in out.decode().split("\0") if f}
        # A tracked file deleted in the working tree is not part of a build from this tree.
        return {f for f in files if (ROOT / f).is_file()}
    except (OSError, subprocess.CalledProcessError):
        base = ROOT / prefix
        if not base.exists():
            return None
        return {p.relative_to(ROOT).as_posix() for p in base.rglob("*") if p.is_file()
                and not JUNK.search(p.relative_to(ROOT).as_posix())}


def tree_version() -> str | None:
    p = ROOT / PKG / "_version.py"
    if not p.is_file():
        return None
    m = VERSION_RE.search(p.read_text())
    return m.group(1) if m else None


def parse_metadata(text: str):
    return email.parser.Parser().parsestr(text)


def check_metadata(rep: Report, label: str, meta, version: str) -> None:
    mv = meta.get("Metadata-Version", "0")
    rep.check(tuple(int(x) for x in mv.split(".")) >= (2, 4), f"{label} Metadata-Version >= 2.4 (PEP 639)", mv)
    rep.check(meta.get("Name") == PKG, f"{label} Name", str(meta.get("Name")))
    rep.check(norm_version(meta.get("Version", "")) == norm_version(version), f"{label} Version", meta.get("Version"))
    rep.check(meta.get("License-Expression") == "Apache-2.0", f"{label} License-Expression",
              str(meta.get("License-Expression")))
    lf = set(meta.get_all("License-File") or [])
    rep.check(set(LICENCE_FILES) <= lf, f"{label} License-File entries", ", ".join(sorted(lf)))
    lic_cls = [c for c in (meta.get_all("Classifier") or []) if c.startswith("License ::")]
    rep.check(not lic_cls, f"{label} no 'License ::' classifier next to the expression", ", ".join(lic_cls))
    rp = meta.get("Requires-Python", "")
    rep.check(rp.replace(" ", "") == ">=3.10", f"{label} Requires-Python", rp)
    raw = meta.get_all("Requires-Dist") or []
    reqs = [r.split(";")[0].strip() for r in raw if "extra ==" not in r]
    base = {re.split(r"[<>=!~ \[(]", r, maxsplit=1)[0].lower() for r in reqs}
    rep.check({"torch", "numpy"} <= base, f"{label} Requires-Dist torch, numpy", "; ".join(reqs))


def check_wheel(rep: Report, path: Path, args) -> str | None:
    m = re.fullmatch(r"mmaemu-([^-]+)-py3-none-any\.whl", path.name)
    if not rep.check(m is not None, "wheel file name mmaemu-<ver>-py3-none-any.whl", path.name):
        return None
    ver = m.group(1)
    size = path.stat().st_size
    rep.check(size <= args.max_wheel_mb * 1024 * 1024, f"wheel size <= {args.max_wheel_mb} MB",
              f"{size / 1048576:.2f} MB")
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        read = z.read
        distinfo = f"mmaemu-{ver}.dist-info/"
        outside = [n for n in names if not (n.startswith(PKG + "/") or n.startswith(distinfo))]
        rep.check(not outside, "wheel holds only mmaemu/ and its dist-info", ", ".join(outside[:10]))
        junk = [n for n in names if JUNK.search(n) or n.endswith((".so", ".pyd", ".dll"))]
        rep.check(not junk, "wheel has no bytecode or compiled extensions", ", ".join(junk[:10]))
        pkg = {n for n in names if n.startswith(PKG + "/")}
        for req in ("mmaemu/__init__.py", "mmaemu/_version.py", "mmaemu/__main__.py", "mmaemu/diagnostics.py",
                    "mmaemu/py.typed"):
            rep.check(req in pkg, f"wheel has {req}")
        # version: file name == METADATA == mmaemu/_version.py (== tag)
        wv = None
        if "mmaemu/_version.py" in pkg:
            mv = VERSION_RE.search(read("mmaemu/_version.py").decode())
            wv = mv.group(1) if mv else None
        rep.check(wv is not None and norm_version(wv) == norm_version(ver), "wheel version == mmaemu/_version.py",
                  f"file name {ver}, _version.py {wv}")
        if distinfo + "METADATA" in names:
            check_metadata(rep, "wheel", parse_metadata(read(distinfo + "METADATA").decode()), ver)
        else:
            rep.check(False, "wheel METADATA present")
        if distinfo + "WHEEL" in names:
            wheel_meta = parse_metadata(read(distinfo + "WHEEL").decode())
            rep.check(wheel_meta.get("Root-Is-Purelib") == "true" and wheel_meta.get("Tag") == "py3-none-any",
                      "wheel is pure (Root-Is-Purelib, py3-none-any)",
                      f"{wheel_meta.get('Root-Is-Purelib')} {wheel_meta.get('Tag')}")
        # licence files in dist-info/licenses, byte-equal to the source tree
        for lf in LICENCE_FILES:
            n = distinfo + "licenses/" + lf
            ok = n in names
            detail = n
            if ok and (ROOT / lf).is_file():
                ok = read(n) == (ROOT / lf).read_bytes()
                detail += " (== source tree)" if ok else " (DIFFERS from the source tree)"
            rep.check(ok, f"wheel dist-info has licence file {lf}", detail)
        # mmaemu/ == the source tree's mmaemu/ (fatbins handled below)
        src = tree_files(PKG)
        if src is None:
            rep.add("FAIL", "source tree mmaemu/ found", str(ROOT))
        else:
            src_nobin = {f for f in src if not f.startswith(BIN)}
            whl_nobin = {f for f in pkg if not f.startswith(BIN)}
            missing = sorted(src_nobin - whl_nobin)
            extra = sorted(whl_nobin - src_nobin)
            rep.check(not missing, "wheel has every source-tree file of mmaemu/", ", ".join(missing[:10]))
            rep.check(not extra, "wheel has no mmaemu/ file absent from the source tree", ", ".join(extra[:10]))
            for prefix, what in ((CSRC, "kernel sources (csrc)"), (PDATA, "platform data")):
                s = sorted(f for f in src if f.startswith(prefix))
                w = sorted(f for f in pkg if f.startswith(prefix))
                if args.require_fatbins:
                    rep.check(bool(w) and w == s, f"wheel has the {what}", f"{len(w)} files")
                elif s:
                    rep.check(w == s, f"wheel has the {what}", f"{len(w)} files")
                else:
                    rep.add("NOTE", f"{what}: none in the source tree yet", prefix)
        check_fatbins(rep, "wheel", {n: (lambda n=n: read(n)) for n in pkg if n.startswith(BIN)}, pkg,
                      src, args)
    return ver


def check_fatbins(rep: Report, label: str, bin_files: dict, pkg: set[str], src: set[str] | None, args) -> None:
    if args.from_sdist:
        rep.check(not bin_files, f"{label} built from the sdist has no bin/ (sources only)",
                  ", ".join(sorted(bin_files)[:10]))
        return
    has_manifest = MANIFEST in bin_files
    src_manifest = (ROOT / MANIFEST).is_file()
    if not has_manifest:
        if src_manifest:
            rep.check(False, f"{label} carries the committed fatbins",
                      "the source tree has bin/manifest.json but the wheel does not (wheel built from the sdist? "
                      "use `python -m build --sdist --wheel`)")
        elif args.require_fatbins or not args.allow_missing_fatbins:
            rep.check(False, f"{label} carries release fatbins (bin/manifest.json)",
                      "missing; pass --allow-missing-fatbins only before the release-candidate freeze")
        else:
            rep.add("NOTE", f"{label} has no fatbins (allowed by --allow-missing-fatbins)", "NVRTC fallback only")
        stray = sorted(n for n in bin_files if n != MANIFEST)
        rep.check(not stray, f"{label} bin/ has no files without a manifest", ", ".join(stray[:10]))
        return
    raw = bin_files[MANIFEST]()
    if src_manifest:
        rep.check(raw == (ROOT / MANIFEST).read_bytes(), f"{label} manifest.json == committed manifest.json")
    verify_manifest_obj(rep, label, raw, lambda f: bin_files[BIN + f]() if BIN + f in bin_files else None,
                        sorted(n[len(BIN):] for n in bin_files), pkg, require_modules=args.require_fatbins)


def verify_manifest_obj(rep: Report, label: str, raw: bytes, get, present: list[str], pkg: set[str],
                        require_modules: bool) -> None:
    try:
        man = json.loads(raw)
    except ValueError as e:
        rep.check(False, f"{label} manifest.json parses", str(e))
        return
    rep.check(man.get("format") == MANIFEST_FORMAT, f"{label} manifest format {MANIFEST_FORMAT}", str(man.get("format")))
    mods = man.get("modules") or {}
    if require_modules:
        rep.check(bool(mods), f"{label} manifest lists at least one module", ", ".join(sorted(mods)))
    listed = {"manifest.json"}
    for stem, info in sorted(mods.items()):
        f = info.get("file", f"{stem}.fatbin")
        listed.add(f)
        data = get(f)
        if data is None:
            rep.check(False, f"{label} fatbin {f} present")
            continue
        got = sha256(data)
        rep.check(got == info.get("sha256"), f"{label} fatbin {f} sha256 matches the manifest",
                  f"{got[:16]}... vs {str(info.get('sha256'))[:16]}...")
        srcs = [n for n in pkg if n.startswith(CSRC) and PurePosixPath(n).stem == stem]
        rep.check(bool(srcs), f"{label} has the source of module {stem} (NVRTC fallback)", ", ".join(srcs))
    unlisted = sorted(set(present) - listed)
    rep.check(not unlisted, f"{label} bin/ has no file missing from the manifest", ", ".join(unlisted[:10]))


def check_sdist(rep: Report, path: Path, args) -> str | None:
    m = re.fullmatch(r"mmaemu-([^-]+)\.tar\.gz", path.name)
    if not rep.check(m is not None, "sdist file name mmaemu-<ver>.tar.gz", path.name):
        return None
    ver = m.group(1)
    top = f"mmaemu-{ver}/"
    with tarfile.open(path, "r:gz") as t:
        members = [x for x in t.getmembers() if x.isfile()]
        names = [x.name for x in members]
        outside = [n for n in names if not n.startswith(top)]
        rep.check(not outside, f"sdist entries are under {top}", ", ".join(outside[:10]))
        rel = {n[len(top):]: x for n, x in zip(names, members) if n.startswith(top)}

        def read(n: str) -> bytes:
            f = t.extractfile(rel[n])
            return f.read() if f else b""

        for req in SDIST_TOP_FILES:
            rep.check(req in rel, f"sdist has {req}")
        for lf in LICENCE_FILES:
            if lf in rel and (ROOT / lf).is_file():
                rep.check(read(lf) == (ROOT / lf).read_bytes(), f"sdist {lf} == source tree")
        if "PKG-INFO" in rel:
            check_metadata(rep, "sdist", parse_metadata(read("PKG-INFO").decode()), ver)
        sv = None
        if "mmaemu/_version.py" in rel:
            mv = VERSION_RE.search(read("mmaemu/_version.py").decode())
            sv = mv.group(1) if mv else None
        rep.check(sv is not None and norm_version(sv) == norm_version(ver), "sdist version == mmaemu/_version.py",
                  f"file name {ver}, _version.py {sv}")
        fat = sorted(n for n in rel if n.endswith((".fatbin", ".cubin")))
        rep.check(not fat, "sdist has no fatbins/cubins", ", ".join(fat[:10]))
        bad = sorted(n for n in rel if n.startswith(SDIST_FORBIDDEN_DIRS))
        rep.check(not bad, "sdist has no bin/, recordings/, results/, docs/, tools/ or bench/", ", ".join(bad[:10]))
        junk = sorted(n for n in rel if JUNK.search(n))
        rep.check(not junk, "sdist has no bytecode", ", ".join(junk[:10]))
        for prefix, label in ((PKG, "mmaemu/ (without bin/)"), ("tests", "tests/")):
            src = tree_files(prefix)
            if src is None:
                rep.check(False, f"source tree {prefix}/ found")
                continue
            src = {f for f in src if not f.startswith(BIN)}
            got = {n for n in rel if n.startswith(prefix + "/")}
            missing = sorted(src - got)
            extra = sorted(got - src)
            rep.check(bool(got) and not missing, f"sdist has every source-tree file of {label}",
                      f"{len(got)} files" + (f"; missing {', '.join(missing[:10])}" if missing else ""))
            rep.check(not extra, f"sdist has no {label} file absent from the source tree", ", ".join(extra[:10]))
        csrc = sorted(n for n in rel if n.startswith(CSRC))
        if args.require_fatbins:
            rep.check(bool(csrc), "sdist has the kernel sources (csrc)", f"{len(csrc)} files")
        tl = [p for p in TOOLBOX_LICENCES if p in rel]
        rep.check(bool(tl), "sdist has the toolbox licence next to the test vectors", ", ".join(tl) or
                  " or ".join(TOOLBOX_LICENCES))
    return ver


def verify_committed_manifest(rep: Report, manifest: Path) -> None:
    if not rep.check(manifest.is_file(), "committed fatbin manifest present", str(manifest)):
        return
    bindir = manifest.parent
    present = sorted(p.name for p in bindir.iterdir() if p.is_file())
    pkg = tree_files(PKG) or set()

    def get(f: str):
        p = bindir / f
        return p.read_bytes() if p.is_file() else None

    verify_manifest_obj(rep, "committed", manifest.read_bytes(), get, present, pkg, require_modules=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dists", nargs="*", type=Path, help="the wheel and the sdist (e.g. dist/*)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--allow-missing-fatbins", action="store_true",
                   help="accept a wheel without bin/ (only before the release-candidate freeze)")
    g.add_argument("--require-fatbins", action="store_true",
                   help="release mode: fatbins, csrc and platform data must all be present")
    g.add_argument("--from-sdist", action="store_true",
                   help="the wheel was built from the sdist: it must equal the tree without bin/ and carry no fatbins")
    ap.add_argument("--tag", help="git tag (e.g. v0.1.0); the distributions' version must equal it without the 'v'")
    ap.add_argument("--max-wheel-mb", type=float, default=25.0)
    ap.add_argument("--verify-manifest", nargs="?", const=str(ROOT / MANIFEST), metavar="MANIFEST",
                    help="check the committed fatbins against their manifest sha256 (no build needed)")
    args = ap.parse_args(argv)
    rep = Report()
    if args.verify_manifest:
        verify_committed_manifest(rep, Path(args.verify_manifest))
    if not args.dists and not args.verify_manifest:
        ap.error("give the wheel and the sdist, or --verify-manifest")
    if args.dists:
        wheels = [p for p in args.dists if p.name.endswith(".whl")]
        sdists = [p for p in args.dists if p.name.endswith(".tar.gz")]
        other = [p for p in args.dists if p not in wheels and p not in sdists]
        rep.check(len(wheels) == 1 and len(sdists) == 1 and not other,
                  "exactly one wheel and one sdist", ", ".join(p.name for p in args.dists))
        versions = set()
        for w in wheels:
            v = check_wheel(rep, w, args)
            if v:
                versions.add(norm_version(v))
        for s in sdists:
            v = check_sdist(rep, s, args)
            if v:
                versions.add(norm_version(v))
        tv = tree_version()
        if tv:
            rep.check(versions == {norm_version(tv)}, "distribution version == source tree mmaemu/_version.py",
                      f"{sorted(versions)} vs {tv}")
        if args.tag:
            want = args.tag[1:] if args.tag.startswith("v") else None
            rep.check(want is not None and versions == {norm_version(want)}, "version == tag",
                      f"tag {args.tag}, versions {sorted(versions)}")
    print(f"\ncheck_dist: {len(rep.rows)} checks, {rep.failed} failed", flush=True)
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
