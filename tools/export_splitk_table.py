"""Export the reference-GPU split-K table into the package data of mmaemu.platform.splitk_table.

Inputs (read-only):
  - the truth table $MMAEMU_TRUTH_DIR/data/heur/lt_heuristic_12_8_3_14.npz (its sha256 must equal the truth
    manifest entry);
  - the pre-swap recordings recordings/rtx4060_2026-10: R2 fp8-output rows, R2x mixed-format / small-K completion,
    R3 Evo 2 small-M rows (table recordings; dscale invariance asserted);
  - for verified_split_combos.json: every gemm output of the truth (excluding the 3 + 46 outputs whose split was
    filled from the heuristic table by apply_ltconfig.py) and of the pre-swap recordings with a logged split config.

Steps:
  1. assert align invariance: every (align none, align 256) row pair with the same key has identical (status, algo,
     tile, stages, splitK, reduction, swizzle, custom); the count is printed (truth table: 0 of 419,424);
  2. drop align (keep one row per key; duplicate keys across sources are asserted identical); drop rows with
     status != 0 (counted: e4m3 x e4m3 -> e5m2, CUBLAS_STATUS_NOT_SUPPORTED);
  3. write mmaemu/platform/data/heur_rtx4060_lt12.8.3.14.npz (sorted; byte-deterministic zip: fixed member order,
     timestamps and compression) and .json (sources + sha256, columns, codes, per-setting envelopes, counts);
  4. write mmaemu/platform/data/verified_split_combos.json.
Running it twice gives byte-identical files.

usage: PYTHONDONTWRITEBYTECODE=1 python tools/export_splitk_table.py [--check]   (--check: compare, do not write)"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "record"))
import common  # noqa: E402
import coverage  # noqa: E402
import tables  # noqa: E402

TABLE_ID = "heur_rtx4060_lt12.8.3.14"
DATA = common.REPO / "mmaemu" / "platform" / "data"
WS_CODES = [0, 1 << 20, 4 << 20, 32 << 20]
PKG_COLS = [("M", np.int32), ("N", np.int32), ("K", np.int32), ("fa", np.uint8), ("fb", np.uint8),
            ("out", np.uint8), ("fast", np.uint8), ("ws", np.uint8), ("splitK", np.uint8), ("custom", np.uint8),
            ("algo", np.int16), ("tile", np.int16), ("stages", np.int16), ("reduction", np.uint8),
            ("swizzle", np.uint8)]
INV_VALS = ["status", "algo", "tile", "stages", "splitK", "reduction", "swizzle", "custom"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def deterministic_npz(arrays: dict) -> bytes:
    """npz bytes with fixed member order, timestamps and compression (np.savez_compressed embeds the current time)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(arrays):
            b = io.BytesIO()
            np.lib.format.write_array(b, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            zi = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, b.getvalue(), compresslevel=9)
    return buf.getvalue()


def _rel(p: Path) -> str:
    """repo-relative for files in the repo (stable across worktrees), absolute otherwise (the truth dir)."""
    try:
        return str(Path(p).resolve().relative_to(common.REPO))
    except ValueError:
        return str(p)


def truth_manifest_sha256() -> str:
    m = json.loads((common.TRUTH / "manifest.json").read_text())
    e = next(r for r in m["recordings"] if r["id"] == "heur.lt_heuristic_12_8_3_14")
    return next(f["sha256"] for f in e["files"] if f["path"].endswith("lt_heuristic_12_8_3_14.npz"))


def align_invariance(raw: np.ndarray) -> dict:
    """raw: rows of tables.COLS + src. Per source, every key (M, N, K, fa, fb, out, fast, ws) present with both
    align 0 and align 256 is a pair; it differs when any of INV_VALS differs."""
    ci = {c: i for i, c in enumerate(tables.COLS)}
    key = [ci[c] for c in ("M", "N", "K", "fa", "fb", "out", "fast", "ws")]
    val = [ci[c] for c in INV_VALS]
    out = {}
    for src in np.unique(raw[:, -1]):
        r = raw[raw[:, -1] == src]
        r = r[np.lexsort(np.concatenate([r[:, key], r[:, [ci["align"]]]], 1).T[::-1])]
        k, al, v = r[:, key], r[:, ci["align"]], r[:, val]
        pair = np.all(k[1:] == k[:-1], 1) & (al[:-1] == 0) & (al[1:] == 256)
        differ = int(np.any(v[1:][pair] != v[:-1][pair], 1).sum())
        out[tables.SOURCES[int(src)][0]] = dict(pairs=int(pair.sum()), differ=differ)
    return out


def load_raw() -> np.ndarray:
    return np.concatenate([tables._load(s) for s in tables.SOURCES])


def build():
    src_sha = tables.source_sha256()
    truth_sha = src_sha["heur.lt_heuristic_12_8_3_14"]
    man_sha = truth_manifest_sha256()
    assert truth_sha == man_sha, f"truth table sha256 {truth_sha} != manifest {man_sha}"
    raw = load_raw()
    inv = align_invariance(raw)
    n_pairs = sum(v["pairs"] for v in inv.values())
    n_diff = sum(v["differ"] for v in inv.values())
    assert n_diff == 0, f"align changes the heuristic in {n_diff} of {n_pairs} pairs: {inv}"
    u = tables.load_union()            # duplicate keys across sources asserted identical
    ci = {c: u[c] for c in tables.COLS}
    # drop align: the union is keyed with align; keep align-none rows, plus align-256 rows whose key has no
    # align-none row (invariance asserted above, so their values are the decision for any alignment)
    key8 = np.stack([ci[c] for c in ("M", "N", "K", "fa", "fb", "out", "fast", "ws")], 1)
    order = np.lexsort(np.concatenate([key8, ci["align"][:, None]], 1).T[::-1])   # per key: align 0 first
    k_sorted = key8[order]
    first = np.concatenate([[True], np.any(k_sorted[1:] != k_sorted[:-1], 1)])
    keep = order[first]
    status_nonzero = int((ci["status"][keep] != 0).sum())
    keep = keep[ci["status"][keep] == 0]
    keep = keep[np.lexsort(key8[keep].T[::-1])]
    ws_code = {w: i for i, w in enumerate(WS_CODES)}
    cols = {}
    for name, dt in PKG_COLS:
        v = ci[name][keep] if name != "ws" else np.array([ws_code[int(w)] for w in ci["ws"][keep]])
        assert v.min() >= np.iinfo(dt).min and v.max() <= np.iinfo(dt).max, name
        cols[name] = v.astype(dt)
    # per-setting envelopes
    settings = []
    skey = np.stack([cols[c].astype(np.int64) for c in ("ws", "out", "fast", "fa", "fb")], 1)
    uniq, inv_idx = np.unique(skey, axis=0, return_inverse=True)
    MN = cols["M"].astype(np.int64) * cols["N"].astype(np.int64)
    sp = cols["splitK"] > 1
    for i, s in enumerate(uniq):
        sel = inv_idx.ravel() == i
        ss = sel & sp
        settings.append(dict(ws=WS_CODES[int(s[0])], out=tables.OUT_NAMES[int(s[1])], fast=bool(s[2]),
                             fmt_a=tables.FMT_NAMES[int(s[3])], fmt_b=tables.FMT_NAMES[int(s[4])],
                             rows=int(sel.sum()), split_rows=int(ss.sum()),
                             max_split_MN=int(MN[ss].max()) if ss.any() else None,
                             min_split_K=int(cols["K"][ss].min()) if ss.any() else None,
                             S_values=sorted(int(x) for x in np.unique(cols["splitK"][ss])) if ss.any() else []))
    meta = {
        "format": "mmaemu-splitk-table/1",
        "table_id": TABLE_ID,
        "profile_device": {"gpu": "NVIDIA GeForce RTX 4060", "capability": [8, 9], "sm_count": 24,
                           "cublaslt": "12.8.3.14", "torch": "2.7.1+cu128", "driver": "572.16"},
        "descriptor": "torch 2.7.1 scaled_gemm (m=N, n=M, k=K; A=mat2 transa=T; B=mat1; C=out dtype, fp16 for fp8 "
                      "outputs; D=out dtype; A/B scale pointers; FAST_ACCUM; MAX_WORKSPACE_BYTES), requestedAlgoCount 1",
        "key": ["M", "N", "K", "fa", "fb", "out", "fast", "ws"],
        "columns": {n: np.dtype(dt).name for n, dt in PKG_COLS},
        "codes": {"fa/fb": tables.FMT_NAMES, "out": tables.OUT_NAMES, "ws": WS_CODES,
                  "custom -> scheme": {"0": "serial", "1": "pairwise", "2": "serial"},
                  "tile/stages/reduction": "cublasLt.h enum values (CUDA 12.8)"},
        "sources": [{"id": tables.SOURCES[s][0], "path": _rel(tables.SOURCES[s][1]),
                     "sha256": src_sha[tables.SOURCES[s][0]]} for s in tables.SOURCES],
        "truth_manifest_sha256": man_sha,
        "align_invariance": {"pairs": n_pairs, "differ": n_diff, "per_source": inv},
        "dscale_invariance": "asserted per R2 row pair by tools/record/tables.py (0 differ)",
        "rows": int(len(keep)), "split_rows": int(sp.sum()),
        "dropped_status_nonzero": status_nonzero,
        "duplicate_keys_across_sources_checked_equal": u["n_duplicates_checked"],
        "settings": settings,
    }
    return cols, meta


def verified_combos(cols: dict) -> dict:
    hw = coverage.hw_combos()
    ws2 = coverage.hw_combos(root=common.REC_ROOT)
    allk = {}
    for k, v in hw.items():
        allk.setdefault(k, []).extend(v)
    for k, v in ws2.items():
        allk.setdefault(k, []).extend(f"rtx4060_2026-10:{x}" for x in v)
    tab = set()
    idx = np.nonzero(cols["splitK"] > 1)[0]
    keys = np.stack([cols[c][idx].astype(np.int64) for c in ("fast", "splitK", "custom", "out", "fa", "fb")], 1)
    for f, S, c, o, fa, fb in np.unique(keys, axis=0):
        tab.add((bool(f), int(S), "pairwise" if c == 1 else "serial", tables.OUT_NAMES[int(o)],
                 tables.FMT_NAMES[int(fa)] + "x" + tables.FMT_NAMES[int(fb)]))
    combos = []
    for k in sorted(tab | set(allk)):
        combos.append(dict(fast=k[0], S=k[1], scheme=k[2], out=k[3], fmt_pair=k[4], in_table=k in tab,
                           verified=k in allk, recordings=sorted(set(allk.get(k, [])))))
    return {"format": "mmaemu-verified-split-combos/1", "table_id": TABLE_ID,
            "definition": "a (fast, S, scheme, out dtype, fmt pair) combination is verified when a hardware gemm "
                          "output with that LOGGED / captured split config exists (truth recordings, excluding the 3 + "
                          "46 outputs whose split was filled from the heuristic table; WS2 recordings "
                          "rtx4060_2026-10:*) and the audit FixedModel replays it bitwise (see "
                          "recordings/rtx4060_2026-10/README.txt)",
            "n_table_combos": len(tab), "n_verified_table_combos": sum(1 for c in combos if c["in_table"] and
                                                                         c["verified"]),
            "unverified_table_combos": [c for c in combos if c["in_table"] and not c["verified"]],
            "combos": combos}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare with the committed files; write nothing")
    a = ap.parse_args()
    cols, meta = build()
    npz = deterministic_npz(cols)
    meta["npz_sha256"] = hashlib.sha256(npz).hexdigest()
    meta["npz_bytes"] = len(npz)
    js = (json.dumps(meta, indent=1, sort_keys=True) + "\n").encode()
    vc = (json.dumps(verified_combos(cols), indent=1, sort_keys=True) + "\n").encode()
    files = {f"{TABLE_ID}.npz": npz, f"{TABLE_ID}.json": js, "verified_split_combos.json": vc}
    if a.check:
        bad = [n for n, b in files.items() if not (DATA / n).exists() or (DATA / n).read_bytes() != b]
        print("check:", "OK" if not bad else f"DIFFERS {bad}")
        sys.exit(1 if bad else 0)
    DATA.mkdir(parents=True, exist_ok=True)
    for n, b in files.items():
        (DATA / n).write_bytes(b)
        print(f"wrote {DATA / n} ({len(b)} bytes, sha256 {hashlib.sha256(b).hexdigest()})")
    print(json.dumps({k: meta[k] for k in ("rows", "split_rows", "dropped_status_nonzero", "align_invariance",
                                           "truth_manifest_sha256")}, indent=1))


if __name__ == "__main__":
    main()
