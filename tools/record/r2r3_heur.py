"""R2 / R2x / R3: cuBLASLt heuristic rows recorded on the reference GPU (heuristic queries only, no kernels).

R2   fp8 outputs (D = e4m3 / e5m2, C = fp16 as torch 2.7.1 passes it), lt descriptor of torch._scaled_mm, with and
     without the D scale pointer (torch sets it iff scale_result is given):
       ws {0, 1, 4, 32 MiB} x fast/slow x align {none, 256} x d_scale {0, 1} x (out, fmt pair) in
       {e4m3 out: e4m3xe4m3, e4m3xe5m2, e5m2xe4m3; e5m2 out: e4m3xe5m2, e5m2xe4m3, e4m3xe4m3 (expected status 15)}
     shapes: c5 grid + small-K set + Evo 2 grid (M = 16..8192 step 16) + Evo 2 small M (M = 1..256).
     Includes critic gap 17 (e4m3 out at 224x176x80 and 176x224x80).
R2x  completion of the non-fp8 table: mixed formats (e4m3xe5m2, e5m2xe4m3) at every ws x align x fp32/bf16/fp16 x
     fast/slow on the same shapes (the truth table has mixed formats only at (1 MiB, none) and (4 MiB, 256), so align
     invariance was never measured for them), and e4m3xe4m3 at the 48 main settings on the small-K set.
R3   Evo 2 (K, N) in {(1920,5760), (4096,12288), (8192,24576)} x M = 1..256 (every integer) x the 72 settings of the
     truth table (48 main e4m3xe4m3 + 24 mixed-format settings).

usage:
  python tools/record/r2r3_heur.py query --out STAGE.npz        (GPU: under gpu.sh; ~1-2 min)
  python tools/record/r2r3_heur.py write --stage STAGE.npz       (CPU: table recordings into recordings/rtx4060_2026-10)
  python tools/record/r2r3_heur.py count                          (CPU: number of queries per part)
Columns (int64, no floats): M N K fa fb out fast ws align dscale lib status algo tile stages splitK reduction swizzle
custom workspace waves_x1000; fa/fb 0 = e4m3, 1 = e5m2; out 0 fp32, 1 bf16, 2 fp16, 3 e4m3, 4 e5m2; -1 = no
algorithm returned (status != 0)."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

TRUTH_TABLE = common.TRUTH / "data" / "heur" / "lt_heuristic_12_8_3_14.npz"
COLS = ["M", "N", "K", "fa", "fb", "out", "fast", "ws", "align", "dscale", "lib", "status", "algo", "tile",
        "stages", "splitK", "reduction", "swizzle", "custom", "workspace", "waves_x1000"]
OUTC = {"float32": 0, "bfloat16": 1, "float16": 2, "float8_e4m3fn": 3, "float8_e5m2": 4}
FMTC = {"e4m3": 0, "e5m2": 1}
WS = (0, 1 << 20, 4 << 20, 32 << 20)
EVO = [(1920, 5760), (4096, 12288), (8192, 24576)]


def shapes():
    Ms = [16, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 2048]
    Ns = [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1920, 4096, 5760, 8192, 12288, 24576]
    Ks = [1024, 1920, 2048, 4096, 5760, 8192, 12288, 16384]
    c5 = set(itertools.product(Ms, Ns, Ks))
    z = np.load(TRUTH_TABLE, allow_pickle=False)
    tk = z["K"]
    old_small = {(int(m), int(n), int(k)) for m, n, k in zip(z["M"][tk < 1024], z["N"][tk < 1024], tk[tk < 1024])}
    grid_small = set(itertools.product([16, 32, 48, 64, 96, 128, 176, 224, 256, 512],
                                       [16, 32, 64, 96, 128, 176, 224, 256, 512, 1024],
                                       [16, 32, 48, 64, 80, 96, 128, 192, 256, 384, 512, 768]))
    small_k = old_small | grid_small | {(224, 176, 80), (176, 224, 80)}
    evo = {(M, N, K) for (K, N) in EVO for M in range(16, 8193, 16)}
    evo_small = {(M, N, K) for (K, N) in EVO for M in range(1, 257)}
    return dict(c5=sorted(c5), small_k=sorted(small_k), evo=sorted(evo), evo_small=sorted(evo_small))


def truth_settings():
    """The 72 (ws, out, fast, align, fa, fb) settings of the truth table, read from the npz itself."""
    z = np.load(TRUTH_TABLE, allow_pickle=False)
    key = np.stack([z["ws"], z["out"], z["fast"], z["align"], z["fa"], z["fb"]], 1)
    return [tuple(int(x) for x in r) for r in np.unique(key, axis=0)]


def jobs():
    sh = shapes()
    r2_shapes = sorted(set(sh["c5"]) | set(sh["small_k"]) | set(sh["evo"]) | set(sh["evo_small"]))
    parts = {}
    # R2: fp8 outputs
    pairs = [("float8_e4m3fn", "e4m3", "e4m3"), ("float8_e4m3fn", "e4m3", "e5m2"), ("float8_e4m3fn", "e5m2", "e4m3"),
             ("float8_e5m2", "e4m3", "e5m2"), ("float8_e5m2", "e5m2", "e4m3"), ("float8_e5m2", "e4m3", "e4m3")]
    st = [(ws, od, fa, fb, fast, al, ds) for ws in WS for (od, fa, fb) in pairs for fast in (1, 0)
          for al in (0, 256) for ds in (0, 1)]
    parts["r2"] = (r2_shapes, st)
    # R2x: mixed formats, non-fp8 outputs, every ws x align
    st = [(ws, od, fa, fb, fast, al, 0) for ws in WS for od in ("float32", "bfloat16", "float16")
          for (fa, fb) in (("e4m3", "e5m2"), ("e5m2", "e4m3")) for fast in (1, 0) for al in (0, 256)]
    parts["r2x_mixed"] = (r2_shapes, st)
    st = [(ws, od, "e4m3", "e4m3", fast, al, 0) for ws in WS for od in ("float32", "bfloat16", "float16")
          for fast in (1, 0) for al in (0, 256)]
    parts["r2x_smallk"] = (sh["small_k"], st)
    # R3: Evo 2 small M x the 72 truth settings
    inv = {v: k for k, v in OUTC.items()}
    fmt = {0: "e4m3", 1: "e5m2"}
    st = [(ws, inv[o], fmt[fa], fmt[fb], fast, al, 0) for (ws, o, fast, al, fa, fb) in truth_settings()]
    assert len(st) == 72, len(st)
    parts["r3"] = (sh["evo_small"], st)
    return parts


def cmd_count(_a):
    for name, (shp, st) in jobs().items():
        print(f"{name}: {len(shp)} shapes x {len(st)} settings = {len(shp) * len(st)} queries")


def cmd_query(a):
    from mmaemu.platform import cublaslt as L
    lib = L.version_tuple()
    libv = lib[0] * 10000 + lib[1] * 100 + lib[2]
    hw = common.hardware()
    out, meta = {}, {"hardware": hw, "parts": {}}
    want = a.parts.split(",") if a.parts else None
    for name, (shp, st) in jobs().items():
        if want and name not in want:
            continue
        t0 = time.monotonic()
        rows = np.empty((len(shp) * len(st), len(COLS)), np.int64)
        i = 0
        for (ws, od, fa, fb, fast, al, ds) in st:
            for (M, N, K) in shp:
                r = L.heuristic(M, N, K, fmt_a=fa, fmt_b=fb, out_dtype=od, fast=bool(fast), workspace_bytes=ws,
                                align=al or None, d_scale=bool(ds))
                rows[i] = [M, N, K, FMTC[fa], FMTC[fb], OUTC[od], fast, ws, al, ds, libv, r["status"], r["algo"],
                           r["tile"], r["stages"], r["splitK"], r["reduction"], r["swizzle"], r["custom"],
                           r["workspace"], r["waves_x1000"]]
                i += 1
        out[name] = rows
        sec = time.monotonic() - t0
        meta["parts"][name] = dict(n=int(len(rows)), seconds=round(sec, 1), n_shapes=len(shp), n_settings=len(st),
                                   status_nonzero=int((rows[:, COLS.index("status")] != 0).sum()),
                                   split_rows=int((rows[:, COLS.index("splitK")] > 1).sum()))
        print(name, meta["parts"][name], flush=True)
    meta["lib"] = L.library_path()
    meta["cublasLt_version"] = libv
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print("wrote", a.out)


TITLES = {
    "r2": ("r2.lt_heur_fp8out_12_8_3_14",
           "cuBLASLt 12.8.3.14 heuristic choices for fp8-output torch._scaled_mm descriptors on the RTX 4060 (R2)"),
    "r2x_mixed": ("r2.lt_heur_mixed_allsettings_12_8_3_14",
                  "cuBLASLt 12.8.3.14 heuristic choices, mixed e4m3/e5m2 inputs with fp32/bf16/fp16 outputs at every "
                  "workspace x alignment setting (R2x)"),
    "r2x_smallk": ("r2.lt_heur_smallk_12_8_3_14",
                   "cuBLASLt 12.8.3.14 heuristic choices, e4m3xe4m3 small-K shape set at the 48 main settings (R2x)"),
    "r3": ("r3.lt_heur_evo2_small_m_12_8_3_14",
           "cuBLASLt 12.8.3.14 heuristic choices for the Evo 2 projection shapes at M = 1..256 (every integer) x "
           "the 72 settings of heur.lt_heuristic_12_8_3_14 (R3)"),
}


def cmd_write(a):
    F = common.truth_format()
    for stage in a.stage:
        z = np.load(stage, allow_pickle=False)
        meta = json.loads(Path(stage).with_suffix(".json").read_text())
        hw = meta["hardware"]
        for name in z.files:
            rid, title = TITLES[name]
            group, base = rid.split(".", 1)
            d = common.REC_DATA / group
            d.mkdir(parents=True, exist_ok=True)
            arr = z[name]
            order = np.lexsort(arr[:, :10].T[::-1])          # deterministic order: M, N, K, fa, fb, out, fast, ws, ...
            arr = arr[order]
            np.savez_compressed(d / f"{base}.npz", **{c: arr[:, i] for i, c in enumerate(COLS)})
            p = meta["parts"][name]
            F.write_table(rid, group, {
                "title": title, "level": "cublasLt.heuristic", "hardware": hw,
                "producer": {"agent": "mmaemu WS2 (pre-swap recordings, 2026-10)",
                             "scripts": ["tools/record/r2r3_heur.py", "mmaemu/platform/cublaslt.py",
                                         "tools/record/common.py"],
                             "sources": [f"{os.environ.get('MMAEMU_AUDIT_DIR') or '<tcemu audit>'}"
                                         "/verify-c5-splitk-4/lt_ctypes.py (original tool)",
                                         str(TRUTH_TABLE)]},
                "claims": ["SPEC v0.1 WS2 " + name.split("_")[0].upper() + "; split-K table completion for "
                           "mmaemu.platform.splitk_table"],
                "notes": (__doc__ or "") + f"\nquery stats: {json.dumps(p)}; library {meta['lib']} "
                         f"(cublasLtGetVersion-style {meta['cublasLt_version']})",
                "params": {"columns": COLS, "requestedAlgoCount": 1,
                           "codes": {"fa/fb": FMTC, "out": OUTC, "align": "0 = no alignment preference (torch), 256 = "
                                     "MIN_ALIGNMENT_A/B/C/D 256 B (Transformer Engine)",
                                     "dscale": "1 = D scale pointer set (torch passes scale_result)"},
                           "descriptor": "torch 2.7.1 scaled_gemm: m=N, n=M, k=K, A=mat2 (fp8, transa=T), B=mat1, "
                                         "D=out dtype, C=out dtype (fp16 for fp8 outputs), FAST_ACCUM flag, A/B scale "
                                         "pointers (+ D scale pointer when dscale=1), MAX_WORKSPACE_BYTES (+ "
                                         "MIN_ALIGNMENT_A/B/C/D when align>0); no bias, no SM target"},
            }, root=common.REC_DATA, payload_files={"table": f"{base}.npz"})
            print("wrote", rid, arr.shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("count")
    q = sp.add_parser("query")
    q.add_argument("--out", default=str(common.STAGE / "heur" / "r2r3.npz"))
    q.add_argument("--parts", default="", help="comma list of r2, r2x_mixed, r2x_smallk, r3 (default: all)")
    w = sp.add_parser("write")
    w.add_argument("--stage", action="append", required=True)
    a = ap.parse_args()
    {"count": cmd_count, "query": cmd_query, "write": cmd_write}[a.cmd](a)


if __name__ == "__main__":
    main()
