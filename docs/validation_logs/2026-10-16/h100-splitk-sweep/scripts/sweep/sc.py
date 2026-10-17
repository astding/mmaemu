"""Shared definitions of the H100 split-K sweep (/data/quant/mmaemu_work/h100_splitk_sweep_plan.md).

Every query is mmaemu.platform.cublaslt.heuristic, the call of tools/record_heuristics.py: torch 2.7.1's _scaled_mm
descriptor, requestedAlgoCount 1, no kernel runs. Rows are stored in record_heuristics' columns (COLS), with
compact dtypes. Env: SWEEP_SRC = the mmaemu source tree (default: ../src), SWEEP_OUT = this step's output directory.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = Path(os.environ.get("SWEEP_SRC") or HERE.parent / "src").resolve()
for _p in (SRC / "tools" / "record", SRC / "tools", SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
OUT = Path(os.environ.get("SWEEP_OUT") or ".").resolve()
SHAPES = HERE.parent / "shapes"

COLS = ["M", "N", "K", "fa", "fb", "out", "fast", "ws", "align", "dscale", "lib", "status", "algo", "tile",
        "stages", "splitK", "reduction", "swizzle", "custom", "workspace", "waves_x1000"]
DT = dict(M=np.int64, N=np.int64, K=np.int64, fa=np.uint8, fb=np.uint8, out=np.uint8, fast=np.uint8, ws=np.int64,
          align=np.int16, dscale=np.uint8, lib=np.int32, status=np.int16, algo=np.int16, tile=np.int16,
          stages=np.int16, splitK=np.int16, reduction=np.int16, swizzle=np.int16, custom=np.int16,
          workspace=np.int64, waves_x1000=np.int32)
OUTS = ["float32", "bfloat16", "float16", "float8_e4m3fn", "float8_e5m2"]
OUTC = {o: i for i, o in enumerate(OUTS)}
PAIRS = [("e4m3", "e4m3"), ("e4m3", "e5m2"), ("e5m2", "e4m3")]
FMTC = {"e4m3": 0, "e5m2": 1}
MiB = 1 << 20
# the two table profiles: hopper-h100pcie-torch (1 MiB, no alignment), hopper-h100pcie-te (32 MiB, 256 B); ws0:
# hopper-h100pcie-ws0 (no workspace: S = 1 for every shape, step 2 of the resolution, which this checks off the grid)
PROFILE_WS = {"torch": (1 * MiB, 0), "te": (32 * MiB, 256), "ws0": (0, 0)}
CHUNK = 500_000      # queries per saved file (a timeout loses at most one chunk)
R = 122_880          # largest split M*N of heur_h100pcie_lt12.8.3.14 at 1 MiB (the TE settings reach 245,760)


def settings(ws: int, align: int) -> list[tuple]:
    """(ws, align, out, fa, fb, fast) of one (workspace, alignment): 5 outs x 3 pairs x fast/slow = 30, including
    e4m3 x e4m3 -> e5m2 (rejected with status 15 on the grid)."""
    return [(ws, align, o, fa, fb, f) for o in OUTS for (fa, fb) in PAIRS for f in (1, 0)]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hardware(device: int = 0) -> dict:
    import record_heuristics
    return record_heuristics.hardware(device)


def libcode() -> int:
    from mmaemu.platform import cublaslt as L
    vt = L.version_tuple()
    return vt[0] * 10000 + vt[1] * 100 + vt[2]


def query(pairs, device: int = 0, progress: str = "") -> dict:
    """Run the heuristic for an iterable of ((M, N, K), (ws, align, out, fa, fb, fast)); returns the COLS arrays."""
    from mmaemu.platform import cublaslt as L
    pairs = list(pairs)
    n = len(pairs)
    cols = {c: np.empty(n, DT[c]) for c in COLS}
    lib = libcode()
    t0 = last = time.monotonic()
    for i, ((M, N, K), (ws, al, o, fa, fb, f)) in enumerate(pairs):
        r = L.heuristic(M, N, K, fmt_a=fa, fmt_b=fb, out_dtype=o, fast=bool(f), workspace_bytes=ws,
                        align=al or None, d_scale=False, device=device)
        for c, v in (("M", M), ("N", N), ("K", K), ("fa", FMTC[fa]), ("fb", FMTC[fb]), ("out", OUTC[o]),
                     ("fast", f), ("ws", ws), ("align", al), ("dscale", 0), ("lib", lib)):
            cols[c][i] = v
        for c in ("status", "algo", "tile", "stages", "splitK", "reduction", "swizzle", "custom", "workspace",
                  "waves_x1000"):
            cols[c][i] = r[c]
        now = time.monotonic()
        if progress and now - last > 60:
            print(f"{progress}: {i + 1}/{n} queries, {(i + 1) / (now - t0):.0f}/s", flush=True)
            last = now
    return cols


def save(cols: dict, path: Path, meta: dict) -> None:
    """npz of the columns + json sidecar (counts, seconds, hardware); never overwrites."""
    if path.exists() or path.with_suffix(".json").exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **cols)
    ok = cols["status"] == 0
    meta = dict(meta, format="mmaemu-h100-sweep/1", columns=COLS, rows=int(len(cols["M"])),
                status_nonzero=int((~ok).sum()), npz_sha256=sha256(path))
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")


def split_summary(cols: dict) -> dict:
    """Per (ws, out, fast, fa, fb): rows, split rows, largest split M*N, smallest split K, S values, reductions,
    customOptions, split rows with K % 128 != 0."""
    c = cols
    ok = c["status"] == 0
    sp = ok & (c["splitK"] > 1)
    mn = c["M"] * c["N"]
    keys = np.stack([c["ws"], c["out"], c["fast"], c["fa"], c["fb"]], 1)
    out = []
    for k in np.unique(keys, axis=0):
        sel = np.all(keys == k, 1)
        s = sel & sp
        out.append(dict(ws=int(k[0]), out=OUTS[int(k[1])], fast=int(k[2]), fa=int(k[3]), fb=int(k[4]),
                        rows=int(sel.sum()), status_nonzero=int((sel & ~ok).sum()), split_rows=int(s.sum()),
                        max_split_MN=int(mn[s].max()) if s.any() else None,
                        min_split_K=int(c["K"][s].min()) if s.any() else None,
                        S=sorted(int(x) for x in np.unique(c["splitK"][s])),
                        reduction=sorted(int(x) for x in np.unique(c["reduction"][s])),
                        custom=sorted(int(x) for x in np.unique(c["custom"][s])),
                        split_K_not_128=int((s & (c["K"] % 128 != 0)).sum())))
    return dict(settings=out, split_rows=int(sp.sum()), max_split_MN=int(mn[sp].max()) if sp.any() else None,
                S_values=sorted(int(x) for x in np.unique(c["splitK"][sp])))


def part_files(d: Path) -> dict:
    """{"<profile>_a<align>": [chunk files in order]} of a part directory."""
    out: dict = {}
    for f in sorted(Path(d).glob("*_a*_c*.npz")):
        out.setdefault(f.stem.rsplit("_c", 1)[0], []).append(f)
    return out


def load_cols(files) -> dict:
    zs = [np.load(f) for f in files]
    return {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files}


def cross(shapes, sts):
    return ((s, st) for st in sts for s in shapes)


def load_shapes(name: str):
    return json.loads((SHAPES / f"{name}.json").read_text())


def product(*a):
    return itertools.product(*a)
