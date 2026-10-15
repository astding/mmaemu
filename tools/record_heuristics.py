"""Record cuBLASLt heuristic choices for torch._scaled_mm's descriptor on the current GPU (heuristic queries only, no
kernels run), in the column format of the reference tables (heur.lt_heuristic_12_8_3_14 / recording R2), plus a JSON
sidecar with the hardware metadata. Used to build a candidate split-K table for a new GPU (runbook gate G6) and to
add shapes for the reference GPU (the way out named by SplitKUnknownError).

usage (GPU: run under $MMAEMU_GPU_LOCK timeout 600):
  python tools/record_heuristics.py --grid c5,evo2,evo2_small_m --ws 0,1,4,32 --out OUT.npz
        [--outs float32,bfloat16,float16,float8_e4m3fn,float8_e5m2] [--pairs e4m3xe4m3,e4m3xe5m2,e5m2xe4m3]
        [--align none,256] [--dscale 0] [--shapes shapes.json] [--device 0] [--max-queries N]
Grids: c5 (11 M x 15 N x 8 K = 1,320 shapes), evo2 ((K,N) in {(1920,5760),(4096,12288),(8192,24576)} x M = 16..8192
step 16), evo2_small_m (the same (K,N) x M = 1..256), smallk (M,N in {16..1024} x K in {16..768}, incl. 224x176x80
and 176x224x80); --shapes adds a JSON list of [M, N, K]. --ws in MiB.
Output: OUT.npz (int64 columns M N K fa fb out fast ws align dscale lib status algo tile stages splitK reduction
swizzle custom workspace waves_x1000; fa/fb 0 e4m3 / 1 e5m2; out 0 fp32, 1 bf16, 2 fp16, 3 e4m3, 4 e5m2; -1 where
cuBLASLt returned no algorithm) and OUT.json (hardware, library, grid, counts, per-setting split envelopes, Evo 2 split
status per workspace, a candidate profile name). An existing OUT is never overwritten (exit 2)."""
from __future__ import annotations

import argparse
import itertools
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

COLS = ["M", "N", "K", "fa", "fb", "out", "fast", "ws", "align", "dscale", "lib", "status", "algo", "tile",
        "stages", "splitK", "reduction", "swizzle", "custom", "workspace", "waves_x1000"]
OUTC = {"float32": 0, "bfloat16": 1, "float16": 2, "float8_e4m3fn": 3, "float8_e5m2": 4}
FMTC = {"e4m3": 0, "e5m2": 1}
EVO = [(1920, 5760), (4096, 12288), (8192, 24576)]


def grid(name: str) -> list[tuple[int, int, int]]:
    if name == "c5":
        Ms = [16, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 2048]
        Ns = [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1920, 4096, 5760, 8192, 12288, 24576]
        Ks = [1024, 1920, 2048, 4096, 5760, 8192, 12288, 16384]
        return list(itertools.product(Ms, Ns, Ks))
    if name == "evo2":
        return [(M, N, K) for (K, N) in EVO for M in range(16, 8193, 16)]
    if name == "evo2_small_m":
        return [(M, N, K) for (K, N) in EVO for M in range(1, 257)]
    if name == "smallk":
        s = set(itertools.product([16, 32, 48, 64, 96, 128, 176, 224, 256, 512],
                                  [16, 32, 64, 96, 128, 176, 224, 256, 512, 1024],
                                  [16, 32, 48, 64, 80, 96, 128, 192, 256, 384, 512, 768]))
        return sorted(s | {(224, 176, 80), (176, 224, 80)})
    raise SystemExit(f"unknown grid {name!r} (c5, evo2, evo2_small_m, smallk)")


def hardware(device: int) -> dict:
    from mmaemu.platform import cublaslt as L
    hw = dict(gpu=L.device_name(device), capability=list(L.device_capability(device)),
              sm_count=L.device_sm_count(device), cublaslt=L.version(), cublaslt_lib=L.library_path(),
              python=sys.version.split()[0], os=f"{platform.system()} {platform.release()}",
              date=time.strftime("%Y-%m-%d %H:%M:%S"))
    for smi in ("/usr/lib/wsl/lib/nvidia-smi", "nvidia-smi"):
        try:
            q = subprocess.run([smi, "--query-gpu=name,driver_version", "--format=csv,noheader", f"-i={device}"],
                               capture_output=True, text=True, timeout=30)
            if q.returncode == 0 and q.stdout.strip():
                hw["driver"] = q.stdout.strip().split(",")[-1].strip()
                break
        except Exception:  # noqa: BLE001
            continue
    try:
        import torch
        hw["torch"] = torch.__version__
    except Exception:  # noqa: BLE001
        pass
    return hw


def slug(hw: dict) -> str:
    g = re.sub(r"^nvidia\s+(geforce\s+)?", "", hw["gpu"].lower())
    g = re.sub(r"[^a-z0-9]+", "", g)
    return f"{g}-{hw['sm_count']}sm-lt{hw['cublaslt']}"


def envelopes(arr: np.ndarray) -> list[dict]:
    c = {n: arr[:, i] for i, n in enumerate(COLS)}
    ok = c["status"] == 0
    out = []
    keys = np.stack([c["ws"], c["out"], c["fast"], c["fa"], c["fb"]], 1)
    for k in np.unique(keys[ok], axis=0):
        sel = ok & np.all(keys == k, 1)
        sp = sel & (c["splitK"] > 1)
        mn = c["M"] * c["N"]
        out.append(dict(ws=int(k[0]), out=int(k[1]), fast=int(k[2]), fa=int(k[3]), fb=int(k[4]), rows=int(sel.sum()),
                        split_rows=int(sp.sum()), max_split_MN=int(mn[sp].max()) if sp.any() else None,
                        min_split_K=int(c["K"][sp].min()) if sp.any() else None,
                        S_values=sorted(int(x) for x in np.unique(c["splitK"][sp]))))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default="c5,evo2,evo2_small_m")
    ap.add_argument("--shapes")
    ap.add_argument("--ws", default="0,1,4,32", help="workspace sizes in MiB")
    ap.add_argument("--outs", default="float32,bfloat16,float16,float8_e4m3fn,float8_e5m2")
    ap.add_argument("--pairs", default="e4m3xe4m3,e4m3xe5m2,e5m2xe4m3")
    ap.add_argument("--align", default="none,256")
    ap.add_argument("--dscale", default="0")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--max-queries", type=int, default=6_000_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() or out.with_suffix(".json").exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    from mmaemu.platform import cublaslt as L
    shapes = set()
    for g in a.grid.split(","):
        if g:
            shapes |= set(grid(g))
    if a.shapes:
        shapes |= {tuple(int(x) for x in s) for s in json.loads(Path(a.shapes).read_text())}
    shapes = sorted(shapes)
    ws = [int(float(x) * (1 << 20)) for x in a.ws.split(",")]
    outs = a.outs.split(",")
    pairs = [tuple(p.split("x")) for p in a.pairs.split(",")]
    aligns = [0 if x == "none" else int(x) for x in a.align.split(",")]
    dsc = [int(x) for x in a.dscale.split(",")]
    settings = [(w, o, fa, fb, f, al, ds) for w in ws for o in outs for (fa, fb) in pairs for f in (1, 0)
                for al in aligns for ds in dsc]
    nq = len(settings) * len(shapes)
    if nq > a.max_queries:
        raise SystemExit(f"{nq} queries > --max-queries {a.max_queries}")
    hw = hardware(a.device)
    vt = L.version_tuple()
    libv = vt[0] * 10000 + vt[1] * 100 + vt[2]
    t0 = time.monotonic()
    rows = np.empty((nq, len(COLS)), np.int64)
    i = 0
    for (w, o, fa, fb, f, al, ds) in settings:
        for (M, N, K) in shapes:
            r = L.heuristic(M, N, K, fmt_a=fa, fmt_b=fb, out_dtype=o, fast=bool(f), workspace_bytes=w,
                            align=al or None, d_scale=bool(ds), device=a.device)
            rows[i] = [M, N, K, FMTC[fa], FMTC[fb], OUTC[o], f, w, al, ds, libv, r["status"], r["algo"], r["tile"],
                       r["stages"], r["splitK"], r["reduction"], r["swizzle"], r["custom"], r["workspace"],
                       r["waves_x1000"]]
            i += 1
    sec = time.monotonic() - t0
    c = {n: rows[:, k] for k, n in enumerate(COLS)}
    evo = np.zeros(len(rows), bool)
    for (K, N) in EVO:
        evo |= (c["K"] == K) & (c["N"] == N)
    evo_status = {str(w): dict(rows=int((evo & (c["ws"] == w)).sum()),
                               split_rows=int((evo & (c["ws"] == w) & (c["splitK"] > 1)).sum()))
                  for w in ws}
    meta = dict(format="mmaemu-heuristic-recording/1", columns=COLS, codes=dict(fa_fb=FMTC, out=OUTC),
                hardware=hw, library_version_code=libv, grid=a.grid, extra_shapes=a.shapes, n_shapes=len(shapes),
                settings=[dict(ws=w, out=o, fmt_a=fa, fmt_b=fb, fast=bool(f), align=al or None, dscale=ds)
                          for (w, o, fa, fb, f, al, ds) in settings],
                rows=int(len(rows)), split_rows=int((c["splitK"] > 1).sum()),
                status_nonzero=int((c["status"] != 0).sum()), seconds=round(sec, 1),
                envelopes=envelopes(rows), evo2_split_status_per_ws=evo_status,
                candidate_profile=dict(name=slug(hw), maturity="experimental",
                                       note="candidate data only; not shipped until reviewed (runbook gate G6)"),
                descriptor="torch 2.7.1 scaled_gemm (mmaemu.platform.cublaslt.heuristic), requestedAlgoCount 1")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{n: rows[:, k] for k, n in enumerate(COLS)})
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{hw['gpu']} ({hw['sm_count']} SMs, cc {hw['capability']}, cuBLASLt {hw['cublaslt']}): {len(rows)} queries "
          f"in {sec:.1f}s, split rows {meta['split_rows']}, status!=0 {meta['status_nonzero']}; Evo 2 split rows per "
          f"ws: {json.dumps({k: v['split_rows'] for k, v in evo_status.items()})}; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
