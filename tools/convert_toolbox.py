#!/usr/bin/env python
"""Convert test data into pickle-free, deduplicated npz + json. Run from the repo root.

  python tools/convert_toolbox.py toolbox [--src tests/data] [--out tests/data/toolbox]
      MATLAB Tensor Core toolbox validation vectors (text: hex fp32 inputs, binary-string fp32 c/d) ->
      vectors.npz (uint32 bit patterns, one array per distinct source file, keyed by its sha256) + SOURCES.json
      (original path, sha256, array key, shape of every text file; and the 10 GPU x format sets). The text files
      can then be deleted (git rm); the BSD-2 licence of the toolbox stays next to the vectors.

  python tools/convert_toolbox.py epilogue-ties [--pt PATH] [--out tests/data/epilogue]
      The tcemu audit's verify-c1-claims-5/fix/data/epilogue_ties_rtx4060.pt (torch pickle, RTX 4060 hardware
      outputs of discriminating double-rounding / operand-order tie cases; default: under $MMAEMU_AUDIT_DIR) ->
      epilogue_ties_rtx4060.npz + .json (source path and sha256 recorded). Maintainer-only input.

  python tools/convert_toolbox.py epilogue-hw [--truth $MMAEMU_TRUTH_DIR] [--out tests/data/epilogue]
      Small hardware epilogue recordings of the truth dataset (zero sign under underflow / zero scales, NaN / Inf
      payloads) -> epilogue_hw_rtx4060.npz + .json, so that the CPU tests can kill the fp32 "acc*alpha + 0"
      epilogue without the truth dataset. Read-only use of the truth package's loader (PYTHONDONTWRITEBYTECODE=1).

Every output is written with np.savez_compressed (no pickle) and is reproducible byte for byte from its inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

GPUS = ("Ada", "L40S", "H100", "H200", "B200")
FMTS = ("E4M3", "E5M2")
TOOLBOX_ORIGIN = "https://github.com/north-numerical-computing/MATLAB-tensor-core (validation vectors)"
# The toolbox commit the vectors come from, and how the paths in 'sources' map to it (reviewed 2026-10-14).
TOOLBOX_UPSTREAM = {
    "commit": "79143bd29a4a71dc7c2ed47741b63572498b17f6",
    "paths": "model_validation/<GPU>/<FMT>/<file>; the prototype's copy, named in 'sources', was "
             "tests/data/<GPU>/<FMT>/<file>",
    "checked": "the sha256 of model_validation/Ada/E4M3/a_Ada_E4M3.txt and d_Ada_fp32.txt at that commit equal the "
               "ones in 'sources' (2026-10-14)",
}
TIES_PT = "verify-c1-claims-5/fix/data/epilogue_ties_rtx4060.pt"     # relative to $MMAEMU_AUDIT_DIR
EPILOGUE_HW_IDS = ("epilogue.c1_4_hw_underflow", "epilogue.zero_nan_v2_a1_5", "epilogue.c1_4_hw_zero_scale")


def portable_path(p: Path, var: str) -> str:
    """`$VAR/<relative path>` for a path under the directory named by the environment variable `var`, so that no
    host path enters the committed test data; the path itself otherwise."""
    root = os.environ.get(var)
    try:
        rel = Path(p).resolve().relative_to(Path(root).resolve()) if root else None
    except ValueError:
        rel = None
    if rel is None:
        return str(p)
    return f"${var}" if rel == Path(".") else f"${var}/{rel.as_posix()}"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _save_npz(path: Path, arrays: dict) -> None:
    """Deterministic npz: fixed key order, fixed zip timestamps (np.savez_compressed writes the current time)."""
    import zipfile
    import io
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for k in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[k]), allow_pickle=False)
            info = zipfile.ZipInfo(k + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, buf.getvalue())


# ---------------------------------------------------------------------------------------------------------------
def read_hex(path: Path) -> np.ndarray:
    rows = [line.split() for line in open(path) if line.strip()]
    return np.array([[int(h, 16) for h in r] for r in rows], dtype=np.uint32)


def read_bits(path: Path) -> np.ndarray:
    return np.array([int(line.strip(), 2) for line in open(path) if line.strip()], dtype=np.uint32)


def cmd_toolbox(args) -> int:
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arrays, sources, sets = {}, [], []
    for gpu in GPUS:
        for fmt in FMTS:
            entry = {"gpu": gpu, "fmt": fmt.lower()}
            for role, fn in (("a", f"a_{gpu}_{fmt}.txt"), ("b", f"b_{gpu}_{fmt}.txt"), ("c", f"c_{gpu}_fp32.txt"),
                             ("d", f"d_{gpu}_fp32.txt")):
                p = src / gpu / fmt / fn
                digest = sha256_file(p)
                key = "s" + digest[:16]
                arr = read_hex(p) if role in ("a", "b") else read_bits(p)
                if key in arrays:
                    if not np.array_equal(arrays[key], arr):
                        raise SystemExit(f"sha256 prefix collision for {p}")
                else:
                    arrays[key] = arr
                rel = f"tests/data/{gpu}/{fmt}/{fn}"
                sources.append({"path": rel, "sha256": digest, "key": key, "role": role, "shape": list(arr.shape),
                                "format": "hex fp32 per element" if role in ("a", "b") else "binary-string fp32"})
                entry[role] = key
            sets.append(entry)
    _save_npz(out / "vectors.npz", arrays)
    meta = {
        "format": "mmaemu-toolbox-vectors/1",
        "origin": TOOLBOX_ORIGIN,
        "upstream": TOOLBOX_UPSTREAM,
        "licence": "BSD-2-Clause, Copyright (c) 2025 Faizan A. Khattak and Mantas Mikaitis "
                   "(LICENCE.MATLAB-tensor-core)",
        "layout": "a, b: uint32 fp32 bit patterns [5000, 32] holding FP8 values; c, d: uint32 fp32 bits [5000]; "
                  "d = MMA(a, b) + c as recorded by the toolbox authors on each GPU (H100/H200 recorded with c = 0)",
        "npz_sha256": sha256_file(out / "vectors.npz"),
        "arrays": {k: {"shape": list(v.shape), "sha256": sha256_array(v)} for k, v in sorted(arrays.items())},
        "sets": sets,
        "sources": sources,
    }
    (out / "SOURCES.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{len(sources)} text files -> {len(arrays)} distinct arrays in {out / 'vectors.npz'} "
          f"({(out / 'vectors.npz').stat().st_size} bytes)")
    return 0


# ---------------------------------------------------------------------------------------------------------------
def _bits(t) -> np.ndarray:
    import torch
    t = t.detach().cpu().contiguous()
    it = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()]
    nt = {1: np.uint8, 2: np.uint16, 4: np.uint32}[t.element_size()]
    return t.view(it).numpy().view(nt)


def cmd_epilogue_ties(args) -> int:
    import torch
    if not args.pt:
        raise SystemExit("epilogue-ties: give --pt or set MMAEMU_AUDIT_DIR (maintainer-only input)")
    pt, out = Path(args.pt), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = torch.load(pt, weights_only=False)        # trusted audit artefact; converted once, never shipped
    arrays = {"b": _bits(rec["b"])}
    cases = []
    for i, c in enumerate(rec["cases"]):
        arrays[f"a_{i:03d}"] = _bits(c["a"])
        arrays[f"out_{i:03d}"] = _bits(c["out"])
        arrays[f"rows_{i:03d}"] = c["rows"].numpy().astype(np.int64)
        cases.append({"kind": c["kind"], "out_dtype": c["out_dtype"], "fast": bool(c["fast"]),
                      "scale_a_bits": int(_bits(c["sa"].reshape(1))[0]),
                      "scale_b_bits": int(_bits(c["sb"].reshape(1))[0]),
                      "a": f"a_{i:03d}", "out": f"out_{i:03d}", "rows": f"rows_{i:03d}"})
    _save_npz(out / "epilogue_ties_rtx4060.npz", arrays)
    meta = {
        "format": "mmaemu-epilogue-ties/1",
        "source": portable_path(pt, "MMAEMU_AUDIT_DIR"), "source_sha256": sha256_file(pt),
        # the recorder next to the audit's data/
        "producer": portable_path(pt.parent.parent / "record_epilogue_ties.py", "MMAEMU_AUDIT_DIR"),
        "env": rec["env"],
        "layout": "b: uint8 E4M3 codes [768, 1920] (column operand, torch._scaled_mm(a, b.t(), ...)); per case: a "
                  "uint8 E4M3 codes [rows, 1920] (the selected rows of the 16384-row hardware call), out = hardware "
                  "bits (uint16 bf16/fp16) [rows, 768]; scales as fp32 bit patterns; splitk S = 1 (no split kernel "
                  "in the profile, see env.kernels). kind 'single': single rounding to bf16/fp16 differs from the "
                  "hardware on >= 1 output; kind 'order': (acc*sa)*sb differs on >= 1 output.",
        "npz_sha256": sha256_file(out / "epilogue_ties_rtx4060.npz"),
        "cases": cases,
    }
    (out / "epilogue_ties_rtx4060.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{len(cases)} cases -> {out / 'epilogue_ties_rtx4060.npz'}")
    return 0


def cmd_epilogue_hw(args) -> int:
    if not args.truth:
        raise SystemExit("epilogue-hw: give --truth or set MMAEMU_TRUTH_DIR (the unpublished hardware ground truth)")
    sys.dont_write_bytecode = True
    truth = Path(args.truth)
    sys.path.insert(0, str(truth))
    from tcemu_ada_truth import loader
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arrays, cases = {}, []
    for rid in EPILOGUE_HW_IDS:
        rec = loader.load(rid, root=truth)
        for ci, case in enumerate(rec.cases):
            ins = case.inputs()
            ck = f"{rid.split('.', 1)[1]}_{ci:03d}"
            arrays[ck + "_a"], arrays[ck + "_b"] = ins["a"], ins["b"]
            outs = []
            for oi, o in enumerate(case.outputs):
                if not o.stored:
                    continue
                p = o.params
                if p.get("scaling", "tensorwise") != "tensorwise" or "bias" in ins:
                    continue
                sk = p.get("splitk")
                if not isinstance(sk, dict):
                    raise SystemExit(f"{rid}/{case.id}/{o.name}: split not recorded")
                key = f"{ck}_o{oi:03d}"
                arrays[key] = o.bits()
                outs.append({"name": o.name, "bits": key, "out_dtype": p["out_dtype"],
                             "fast_accum": bool(p.get("fast_accum", True)),
                             "scale_a_bits": p.get("scale_a"), "scale_b_bits": p.get("scale_b"),
                             "scale_result_bits": p.get("scale_result"),
                             "splitk": {"S": int(sk.get("S", 1)), "scheme": sk.get("scheme", "none")}})
            cases.append({"truth_id": rid, "case": case.id, "fmt_a": case.merged_params.get("fmt_a", "e4m3"),
                          "fmt_b": case.merged_params.get("fmt_b", "e4m3"), "a": ck + "_a", "b": ck + "_b",
                          "outputs": outs})
    _save_npz(out / "epilogue_hw_rtx4060.npz", arrays)
    man = json.loads((truth / "manifest.json").read_text())
    rec_meta = {e["id"]: e for e in man["recordings"]}
    meta = {
        "format": "mmaemu-epilogue-hw/1",
        "truth_dir": portable_path(truth, "MMAEMU_TRUTH_DIR"), "truth_manifest_sha256": sha256_file(truth / "manifest.json"),
        "truth_ids": {rid: {"json": rec_meta[rid]["json"],
                            "json_sha256": sha256_file(truth / rec_meta[rid]["json"])} for rid in EPILOGUE_HW_IDS},
        "hardware": "NVIDIA GeForce RTX 4060 (sm_89, 24 SMs), cuBLASLt 12.8.3.14, torch 2.7.1+cu128",
        "layout": "a uint8 [M, K] (mat1), b uint8 [N, K] (mat2 transposed: the call was torch._scaled_mm(a, b.t(), "
                  "...)); outputs: hardware bit patterns; scales as fp32 bit patterns; splitk as captured.",
        "npz_sha256": sha256_file(out / "epilogue_hw_rtx4060.npz"),
        "cases": cases,
    }
    (out / "epilogue_hw_rtx4060.json").write_text(json.dumps(meta, indent=1) + "\n")
    n = sum(len(c["outputs"]) for c in cases)
    print(f"{n} outputs of {len(cases)} cases -> {out / 'epilogue_hw_rtx4060.npz'} "
          f"({(out / 'epilogue_hw_rtx4060.npz').stat().st_size} bytes)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("toolbox")
    t.add_argument("--src", default="tests/data")
    t.add_argument("--out", default="tests/data/toolbox")
    e = sub.add_parser("epilogue-ties")
    e.add_argument("--pt", default=str(Path(os.environ["MMAEMU_AUDIT_DIR"]) / TIES_PT)
                   if os.environ.get("MMAEMU_AUDIT_DIR") else "")
    e.add_argument("--out", default="tests/data/epilogue")
    h = sub.add_parser("epilogue-hw")
    h.add_argument("--truth", default=os.environ.get("MMAEMU_TRUTH_DIR", ""))
    h.add_argument("--out", default="tests/data/epilogue")
    args = ap.parse_args(argv)
    return {"toolbox": cmd_toolbox, "epilogue-ties": cmd_epilogue_ties, "epilogue-hw": cmd_epilogue_hw}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
