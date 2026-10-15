"""Vendor the tcemu audit's hardware probe kits into tools/bringup/kits/ (run once, CPU; re-runnable).

Maintainer host only: vendoring reads the (unpublished) audit work directory $MMAEMU_AUDIT_DIR; --check does not.

Each kit is copied with its dependency closure (imports, CUDA sources, the data it reads), then adapted minimally:
- `from tcemu import reference as R`  ->  `from compat import R`   (tools/bringup/compat.py: mmaemu, never tcemu);
- every hard-coded audit path for outputs / logs / results  ->  $BRINGUP_OUT (the run directory);
- torch JIT build directories  ->  $BRINGUP_CACHE (an arch-tagged cache dir, never the repo or the audit tree);
- c6 text sets are gzipped and read through a transparent reader; pickled inputs are converted to npz.
Nothing in tools/bringup reads the audit directories at run time. PROVENANCE.json lists, per vendored file: origin
path, origin sha256, vendored sha256, modified flag and the list of changes. Build dirs, __pycache__ and pickles are
not vendored.

usage: python tools/bringup/vendor_kits.py [--check]     (--check: verify PROVENANCE.json against the files)"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KITS = HERE / "kits"
# The audit's work directory. Its scripts hard-code their own absolute location, so the patterns below that rewrite
# those paths are spelled with AUD_TXT (the same directory).
AUD = Path(os.environ.get("MMAEMU_AUDIT_DIR") or "MMAEMU_AUDIT_DIR-is-not-set")
AUD_TXT = str(AUD)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


OUT_HELPER = '''
import os as _os
from pathlib import Path as _Path
def _bringup_out():
    d = _Path(_os.environ.get("BRINGUP_OUT") or ".")
    d.mkdir(parents=True, exist_ok=True)
    return d
def _bringup_cache():
    d = _Path(_os.environ.get("BRINGUP_CACHE") or (_Path.home() / ".cache" / "mmaemu" / "bringup"))
    d.mkdir(parents=True, exist_ok=True)
    return d
'''
GZ_HELPER = '''
import gzip as _gzip
def _open(p, mode="r"):
    """text reader for the vendored (gzipped) sets: p or p.gz"""
    p = Path(p)
    if "w" in mode:
        return open(p, mode)
    return open(p) if p.exists() else _gzip.open(str(p) + ".gz", "rt")
def _exists(p):
    p = Path(p)
    return p.exists() or Path(str(p) + ".gz").exists()
'''
COMPAT_IMPORT = ('import sys as _sys\nfrom pathlib import Path as _P\n'
                 '_d = _P(__file__).resolve().parent\n'
                 'while not (_d / "compat.py").exists():   # tools/bringup (compat.py)\n'
                 '    _d = _d.parent\n'
                 '_sys.path.insert(0, str(_d))\n'
                 'from compat import R  # noqa: E402  (vendored: was `from tcemu import reference as R`)\n')

# (kit, origin dir, files, patches {file: [(old, new, why)]})
KIT_SOURCES = [
    ("c6", AUD / "c6-paper-identifiability" / "kit",
     ["run_kit.py", "run_cublas_kit.py", "family.py", "mma_probe_kit.cu", "README.md"],
     {"run_kit.py": [
         ("import numpy as np\n", "import numpy as np\n" + OUT_HELPER + GZ_HELPER, "output/cache dirs + gz reader"),
         ('build = HERE / f"build_{tag}"', 'build = _bringup_cache() / f"c6_build_{tag}"', "build dir -> cache"),
         ("for l in open(p) if l.strip()], dtype=np.uint32)\n\n\ndef read_bits",
          "for l in _open(p) if l.strip()], dtype=np.uint32)\n\n\ndef read_bits", "gz reader"),
         ("return np.array([int(l.strip(), 2) for l in open(p) if l.strip()], dtype=np.uint32)",
          "return np.array([int(l.strip(), 2) for l in _open(p) if l.strip()], dtype=np.uint32)", "gz reader"),
         ('with open(fdir / f"d_{S}_{tag}.txt", "w") as fo:',
          'with open(_bringup_out() / f"d_{S}_{F}_{tag}.txt", "w") as fo:', "measured bits -> $BRINGUP_OUT"),
         ('tags = [l.strip() for l in open(fdir / f"tags_{S}_{F}.txt")]',
          'tags = [l.strip() for l in _open(fdir / f"tags_{S}_{F}.txt")]', "gz reader"),
         ("if recf.exists():", "if _exists(recf):", "gz reader"),
     ],
      "run_cublas_kit.py": [
         ("import torch\n", "import torch\n" + OUT_HELPER + GZ_HELPER, "output dir + gz reader"),
         ("for l in open(p) if l.strip()], dtype=np.uint32)\n\n\ndef read_bits",
          "for l in _open(p) if l.strip()], dtype=np.uint32)\n\n\ndef read_bits", "gz reader"),
         ("return np.array([int(l.strip(), 2) for l in open(p) if l.strip()], dtype=np.uint32)",
          "return np.array([int(l.strip(), 2) for l in _open(p) if l.strip()], dtype=np.uint32)", "gz reader"),
         ('tags = [l.strip() for l in open(d / f"tags_{S}_{F}.txt")]',
          'tags = [l.strip() for l in _open(d / f"tags_{S}_{F}.txt")]', "gz reader"),
         ('with open(d / f"d_{S}_cublas_M{M}_fast{int(fast)}_{p.name.replace(\' \', \'_\')}.txt", "w") as fo:',
          'with open(_bringup_out() / f"d_{S}_cublas_M{M}_fast{int(fast)}_{p.name.replace(\' \', \'_\')}.txt", "w") '
          'as fo:', "measured bits -> $BRINGUP_OUT"),
     ]}),
    ("interleave", AUD / "verify-c6-paper-identifiability-1" / "models",
     ["interleave_probe.py", "models.py", "interleave_probe.txt"],
     {"interleave_probe.py": [
         ("from tcemu import reference as R\n", COMPAT_IMPORT, "tcemu -> compat (mmaemu)"),
         ('"tcemu blackwell": R.dot(', '"mmaemu b200_mma_sync (alias blackwell)": R.dot(', "label"),
         ('fmt=fmt, **R.PRESETS["blackwell"])', 'fmt=fmt, **R.PRESETS["b200_mma_sync"])', "canonical preset name"),
     ]}),
    ("window", AUD / "verify-p2-int-mixed-3",
     ["hw_collect.py", "analyse.py", "build.py", "vectors.py", "tc_models.py", "test_prims.py", "test_prims8.py",
      "epass_raw.cu", "prims.cu", "prims8.cu", "v2_common.cuh", "results/analyse_hw.json"],
     {"build.py": [
         ("from torch.utils.cpp_extension import load\n", "from torch.utils.cpp_extension import load\n" + OUT_HELPER,
          "cache helper"),
         ('bd = W / "build" / name', 'bd = _bringup_cache() / ("window_build_" + name)', "build dir -> cache"),
     ],
      "hw_collect.py": [
         ("import torch\n", "import torch\n" + OUT_HELPER, "output helper"),
         ('torch.save(out, W / "results" / "hw_epass_raw.pt")', 'torch.save(out, _bringup_out() / "hw_epass_raw.pt")',
          "results -> $BRINGUP_OUT"),
         ('torch.save(res, W / "results" / "hw_bf16_raw.pt")', 'torch.save(res, _bringup_out() / "hw_bf16_raw.pt")',
          "results -> $BRINGUP_OUT"),
     ],
      "analyse.py": [
         ("import torch\n", "import torch\n" + OUT_HELPER, "output helper"),
         ('torch.load(W / "results" / "hw_epass_raw.pt")', 'torch.load(_bringup_out() / "hw_epass_raw.pt")',
          "results <- $BRINGUP_OUT"),
         ('torch.load(W / "results" / "hw_bf16_raw.pt")', 'torch.load(_bringup_out() / "hw_bf16_raw.pt")',
          "results <- $BRINGUP_OUT"),
         ('open(W / "results" / "analyse_hw.json", "w")', 'open(_bringup_out() / "analyse_hw.json", "w")',
          "results -> $BRINGUP_OUT"),
         ('open(W / "results" / "analyse_sweep.json", "w")', 'open(_bringup_out() / "analyse_sweep.json", "w")',
          "results -> $BRINGUP_OUT"),
     ],
      "test_prims.py": [
         ('P = build.get("prims")', 'P = None', "defer CUDA build while importing CPU vector generators"),
         ('if __name__ == "__main__":\n', 'if __name__ == "__main__":\n    P = build.get("prims")\n',
          "build only for probe execution"),
         ("import torch\n", "import torch\n" + OUT_HELPER, "output helper"),
         ('    (W / "results").mkdir(exist_ok=True)\n    json.dump(res, open(W / "results" / "prims.json", "w"), indent=1)',
          '    json.dump(res, open(_bringup_out() / "prims.json", "w"), indent=1)', "results -> $BRINGUP_OUT"),
     ],
      "test_prims8.py": [
         ('P = build.get("prims8")', 'P = None', "defer CUDA build while importing CPU vector generators"),
         ('def main():\n', 'def main():\n    global P\n    P = build.get("prims8")\n', "build only for probe execution"),
         ("import torch\n", "import torch\n" + OUT_HELPER, "output helper"),
         ('    (W / "results").mkdir(exist_ok=True)\n    json.dump(res, open(W / "results" / "prims8.json", "w"), indent=1)',
          '    json.dump(res, open(_bringup_out() / "prims8.json", "w"), indent=1)', "results -> $BRINGUP_OUT"),
     ]}),
    ("b3soft", AUD / "verify-p3-tensorcore-exact-0", ["scripts/b3soft.py", "csrc/b3soft.cu"],
     {"scripts/b3soft.py": [
         (f'W = "{AUD_TXT}/verify-p3-tensorcore-exact-0"\n'
          f'P3 = "{AUD_TXT}/p3-tensorcore-exact"\n' 'sys.path.insert(0, f"{P3}/scripts")\n',
          'from pathlib import Path as _P\nW = str(_P(__file__).resolve().parents[1])   # the vendored kit dir (csrc/)\n'
          + OUT_HELPER, "audit paths -> kit dir / $BRINGUP_OUT / $BRINGUP_CACHE; p3 scripts not vendored"),
         ('bd = f"{W}/build/b3soft_sm{arch}"', 'bd = str(_bringup_cache() / f"b3soft_build_sm{arch}")',
          "build dir -> cache"),
         ('json.dump(res, open(f"{W}/logs/b3soft_probe.json", "w"), indent=1)',
          'json.dump(res, open(_bringup_out() / "b3soft_probe.json", "w"), indent=1)', "logs -> $BRINGUP_OUT"),
         ("def gemm(modes, cfgs, filt=None):\n",
          "def gemm(modes, cfgs, filt=None):\n    raise SystemExit(\"b3soft gemm mode is not vendored: it reads p3's "
          "pickled verify cases (13 MB) and tcemu.reference; use the probe mode (informational in G8)\")\n",
          "gemm mode disabled"),
     ]}),
    ("zero_specials", AUD / "verify-a3-integration-review-7", ["scripts/zero_specials.py", "results/zero_specials.json"],
     {"scripts/zero_specials.py": [
         ("from tcemu import reference as R\n", COMPAT_IMPORT + OUT_HELPER, "tcemu -> compat (mmaemu)"),
         ('RES = Path(__file__).resolve().parent.parent / "results"', 'RES = _bringup_out()', "results -> $BRINGUP_OUT"),
     ]}),
    ("epilogue_ties", AUD / "verify-c1-claims-5" / "fix", ["record_epilogue_ties.py", "test_epilogue_ties.py"],
     {"record_epilogue_ties.py": [
         (f'OUT = sys.argv[1] if len(sys.argv) > 1 else "{AUD_TXT}/verify-c1-claims-5/fix/data/'
          'epilogue_ties_rtx4060.pt"',
          'OUT = sys.argv[1]   # vendored: the output path is required (never the audit dir)', "required output path"),
     ],
      "test_epilogue_ties.py": [
         ("from tcemu import reference as R\n", COMPAT_IMPORT, "tcemu -> compat (mmaemu)"),
     ]}),
]


def patch(text: str, pats) -> tuple[str, list[str]]:
    why = []
    for old, new, reason in pats:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"patch anchor found {n} times: {old[:80]!r}")
        text = text.replace(old, new)
        why.append(reason)
    value_comparison = "torch." + "equal("
    if value_comparison in text:
        text = text.replace(value_comparison, "same_bits(")
        text = text.replace("import torch", "from tools.bringup.checks import same_bits\nimport torch", 1)
        why.append("compare raw bits, including signed zero and NaN payloads")
    if 'def ext(arch="89"):' in text:
        text = text.replace('def ext(arch="89"):', 'def ext(arch=None):')
        anchor = '    bd = str(_bringup_cache() / f"b3soft_build_sm{arch}")'
        text = text.replace(anchor, '    if arch is None:\n        cc = torch.cuda.get_device_capability(0)\n'
                            '        arch = f"{cc[0]}{cc[1]}"\n' + anchor)
        why.append("detect the active device architecture for b3soft probes")
    return text, why


def vendor() -> dict:
    prov = {"format": "mmaemu-bringup-provenance/1",
            "note": "vendored for docs/runbooks/sm120_bringup.md by tools/bringup/vendor_kits.py; origin files are "
                    "read-only audit artefacts", "files": []}
    for kit, src, files, patches in KIT_SOURCES:
        dst = KITS / kit
        for f in files:
            raw = (src / f).read_bytes()
            out_name = Path(f).name if f.startswith("results/") else f
            if f.startswith("results/"):
                out_name = "expected_sm89/" + Path(f).name
            text_mod, why = raw, []
            if f in patches:
                t, why = patch(raw.decode(), patches[f])
                text_mod = t.encode()
            p = dst / out_name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(text_mod)
            prov["files"].append(dict(path=str(p.relative_to(HERE)), origin=str(src / f), origin_sha256=sha(raw),
                                      vendored_sha256=sha(text_mod), modified=bool(why), changes=why))
    # c6 sets: the sm_120-relevant sets (+ MM_ada for the sm_89 validation), text gzipped, big JSON gzipped
    src = AUD / "c6-paper-identifiability" / "kit" / "sets"
    for S in ("MM_sm120", "MM_ada", "TBX_B200", "TBX_H100c0", "MM_cublas64_sm120"):
        for p in sorted((src / S).rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(src)
            raw = p.read_bytes()
            gz = p.suffix == ".txt" or (p.suffix == ".json" and len(raw) > 50_000)
            q = KITS / "c6" / "sets" / (str(rel) + (".gz" if gz else ""))
            q.parent.mkdir(parents=True, exist_ok=True)
            data = gzip.compress(raw, compresslevel=9, mtime=0) if gz else raw
            q.write_bytes(data)
            prov["files"].append(dict(path=str(q.relative_to(HERE)), origin=str(p), origin_sha256=sha(raw),
                                      vendored_sha256=sha(data), modified=gz,
                                      changes=["gzip (mtime 0); read through _open()"] if gz else []))
    # Ada epilogue-tie recording: pickle -> npz (+ json) so G7 can list differences on another GPU
    import numpy as np
    import torch
    srcp = AUD / "verify-c1-claims-5" / "fix" / "data" / "epilogue_ties_rtx4060.pt"
    d = torch.load(srcp, weights_only=False)
    arr = {"b": d["b"].numpy()}
    cases = []
    for i, c in enumerate(d["cases"]):
        arr[f"c{i}.rows"] = c["rows"].numpy()
        arr[f"c{i}.a"] = c["a"].numpy()
        o = c["out"].contiguous()
        arr[f"c{i}.out"] = o.view(torch.int16).numpy().view(np.uint16)
        cases.append(dict(kind=c["kind"], out_dtype=c["out_dtype"], fast=bool(c["fast"]),
                          sa_bits=int(c["sa"].reshape(()).view(torch.int32).item()) & 0xFFFFFFFF,
                          sb_bits=int(c["sb"].reshape(()).view(torch.int32).item()) & 0xFFFFFFFF))
    q = KITS / "epilogue_ties" / "ada_rtx4060.npz"
    np.savez_compressed(q, **arr)
    (KITS / "epilogue_ties" / "ada_rtx4060.json").write_text(json.dumps(dict(env=d["env"], cases=cases), indent=1,
                                                                          default=str) + "\n")
    for f in (q, KITS / "epilogue_ties" / "ada_rtx4060.json"):
        prov["files"].append(dict(path=str(f.relative_to(HERE)), origin=str(srcp), origin_sha256=sha(srcp.read_bytes()),
                                  vendored_sha256=sha(f.read_bytes()), modified=True,
                                  changes=["converted from the pickled .pt (uint bit patterns, no pickle)"]))
    return prov


def check() -> int:
    """Every file under kits/ is listed in PROVENANCE.json with its current sha256 (kits/ holds vendored files only;
    the mmaemu-side code lives next to it in tools/bringup)."""
    prov = json.loads((KITS / "PROVENANCE.json").read_text())
    bad = [f["path"] for f in prov["files"] if sha((HERE / f["path"]).read_bytes()) != f["vendored_sha256"]]
    listed = {f["path"] for f in prov["files"]}
    unlisted = [str(p.relative_to(HERE)) for p in KITS.rglob("*") if p.is_file() and p.name != "PROVENANCE.json"
                and "__pycache__" not in p.parts and str(p.relative_to(HERE)) not in listed]
    print(f"{len(prov['files'])} vendored files; sha256 mismatches: {bad}; files not in PROVENANCE: {unlisted}")
    return 1 if bad or unlisted else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        sys.exit(check())
    if not AUD.is_dir():
        raise SystemExit("vendoring needs the tcemu audit work directory: set MMAEMU_AUDIT_DIR")
    for kit, *_ in KIT_SOURCES:
        if (KITS / kit).exists():
            shutil.rmtree(KITS / kit)
    prov = vendor()
    (KITS / "PROVENANCE.json").write_text(json.dumps(prov, indent=1) + "\n")
    print(f"vendored {len(prov['files'])} files; {sum(f['modified'] for f in prov['files'])} modified")


if __name__ == "__main__":
    main()
