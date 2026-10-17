"""Export the H100 PCIe split-K table into the package data of mmaemu.platform.splitk_table.

Inputs (read-only, all in the repository):
  - the heuristic recording of the H100 probe, docs/validation_logs/2026-10-15/h100-probe/results/m6_heur/heur.npz
    (tools/record_heuristics.py on an H100 PCIe, 114 SMs, cuBLASLt 12.8.3.14; grids c5, evo2, evo2_small_m;
    workspaces 0 / 1 / 4 / 32 MiB; align none and 256); its sha256 must equal the entry in results/SHA256SUMS;
  - the model rows (p3) of the split-K sweep, docs/validation_logs/2026-10-16/h100-splitk-sweep/results/p3 (same
    instance type, driver and library; the sweep re-recorded the probe's grid byte for byte, results/p0/drift.json):
    the (K, N) of 9 models x M from 1 to 16,384, workspaces 1 and 32 MiB, both alignments;
  - the sweep's envelope: results/analysis/envelope_rule.json (fitted on p0-p3) and eval_sealed.json (its one test
    on the sealed p4); the rule's sha256 must be the one eval_sealed.json tested, and the bounds are copied unchanged;
  - for the verified combinations and the split layouts: results/analysis/mmaemu_replay.json of the probe and
    mmaemu_replay_p5.json of the sweep, mmaemu's replays of every live call against its full-output sha256
    (tools/replay_h100_log.py; every job must be equal or rejected as on the hardware).

Steps:
  1. assert the recordings' hardware (GPU name, SM count, cuBLASLt version, library version code of every row);
  2. assert align invariance per recording: every (align none, align 256) row pair with the same key has identical
     (status, algo, tile, stages, splitK, reduction, swizzle, custom); keep the align-none row;
  3. derive the rejections: every row with status != 0 must be e4m3 x e4m3 -> e5m2 (status 15,
     CUBLAS_STATUS_NOT_SUPPORTED) and every e4m3 x e4m3 -> e5m2 row must have status != 0; drop those rows;
  4. merge the probe and p3 rows: a key in both must have identical values, except the customOption of an unsplit
     row (S = 1 in both; mmaemu ignores it there): such keys keep the probe's row and are listed in the json
     (28 keys: M 272 and 254, N 12288, K 4096, customOption 5 in the probe, 7 in p3);
  5. assert what the split rule rests on: every split row has K % 128 == 0, no split without k-tiles, reduction
     scheme COMPUTE_TYPE, workspace > 0; workspace 0 never splits (probe, sweep p1 and sealed p4);
  6. write mmaemu/platform/data/heur_h100pcie_lt12.8.3.14.npz (the columns of the RTX 4060 table, byte-deterministic)
     and .json (sources + sha256, codes incl. the customOption -> scheme rule of the customOptions seen on replayed
     live split calls, the split layouts the replays cover, the K-dependent envelope with its held-out result per
     workspace, per-setting counts), and verified_split_combos_h100pcie_lt12.8.3.14.json.
Running it twice gives byte-identical files.

usage: PYTHONDONTWRITEBYTECODE=1 python tools/export_splitk_table_h100.py [--check]   (--check: compare, do not write)"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from export_splitk_table import PKG_COLS, WS_CODES, deterministic_npz  # noqa: E402

REPO = HERE.parent
LOG = REPO / "docs" / "validation_logs" / "2026-10-15" / "h100-probe"
HEUR = LOG / "results" / "m6_heur" / "heur.npz"
REPLAY = LOG / "results" / "analysis" / "mmaemu_replay.json"
SWEEP = REPO / "docs" / "validation_logs" / "2026-10-16" / "h100-splitk-sweep"
P3 = SWEEP / "results" / "p3"
P1_WS0 = SWEEP / "results" / "p1"
P4_WS0 = SWEEP / "results" / "p4_sealed" / "ws0_a0_c00.npz"
RULE = SWEEP / "results" / "analysis" / "envelope_rule.json"
EVAL = SWEEP / "results" / "analysis" / "eval_sealed.json"
REPLAY_P5 = SWEEP / "results" / "analysis" / "mmaemu_replay_p5.json"
DATA = REPO / "mmaemu" / "platform" / "data"
TABLE_ID = "heur_h100pcie_lt12.8.3.14"
VERIFIED = "verified_split_combos_h100pcie_lt12.8.3.14.json"
HARDWARE = {"gpu": "NVIDIA H100 PCIe", "capability": [9, 0], "sm_count": 114, "cublaslt": "12.8.3.14"}
LIB_CODE = 120803
FMT_NAMES = {0: "e4m3", 1: "e5m2"}
OUT_NAMES = {0: "float32", 1: "bfloat16", 2: "float16", 3: "float8_e4m3fn", 4: "float8_e5m2"}
KEY = ["M", "N", "K", "fa", "fb", "out", "fast", "ws"]
INV_VALS = ["status", "algo", "tile", "stages", "splitK", "reduction", "swizzle", "custom"]
REDUCTION_COMPUTE_TYPE = 2       # CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE
TILE_K = 128
# the envelope's workspaces and their key in eval_sealed.json: the name of the profile when the sweep was analysed
# (since renamed hopper and hopper-te); the table records it as eval_profile
EVAL_PROFILES = {1 << 20: "hopper-h100pcie-torch", 32 << 20: "hopper-h100pcie-te"}


def _rel(p: Path) -> str:
    return Path(p).resolve().relative_to(REPO).as_posix()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sums_sha256(path: Path) -> str:
    """The results/SHA256SUMS entry of a file of the probe's output directory."""
    rel = "./" + path.relative_to(LOG / "results").as_posix()
    for ln in (LOG / "results" / "SHA256SUMS").read_text().splitlines():
        h, _, name = ln.partition("  ")
        if name == rel:
            return h
    raise KeyError(f"{rel} not in SHA256SUMS")


def _check_hardware(hw: dict, what: str) -> None:
    for k, v in HARDWARE.items():
        assert hw[k] == v, f"{what}: hardware {k} = {hw[k]!r}, expected {v!r}"


def _status0_rows(c: dict, what: str) -> tuple[np.ndarray, dict]:
    """Steps 2 and 3 for one recording: indices of the align-none, status-0 rows (sorted by key) and counts."""
    n_all = len(c["M"])
    assert (c["lib"] == LIB_CODE).all(), f"{what}: a row of another cuBLASLt"
    assert set(np.unique(c["dscale"]).tolist()) == {0}, f"{what}: dscale other than 0 was not recorded"
    assert set(np.unique(c["align"]).tolist()) == {0, 256}, f"{what}: both alignments must be recorded"
    key = np.stack([c[k] for k in KEY], 1)
    val = np.stack([c[k] for k in INV_VALS], 1)
    o = np.lexsort(np.concatenate([key, c["align"][:, None]], 1).T[::-1])          # per key: align 0 first
    ks, al, vs = key[o], c["align"][o], val[o]
    pair = np.all(ks[1:] == ks[:-1], 1) & (al[:-1] == 0) & (al[1:] == 256)
    n_pairs, n_diff = int(pair.sum()), int(np.any(vs[1:][pair] != vs[:-1][pair], 1).sum())
    assert n_diff == 0, f"{what}: align changes the heuristic in {n_diff} of {n_pairs} pairs"
    first = np.concatenate([[True], np.any(ks[1:] != ks[:-1], 1)])
    keep = o[first]
    assert len(keep) == n_pairs == n_all // 2, f"{what}: every key must be recorded with both alignments"
    st = c["status"][keep]
    rej = (c["fa"][keep] == 0) & (c["fb"][keep] == 0) & (c["out"][keep] == 4)
    assert ((st != 0) == rej).all(), f"{what}: status != 0 must be exactly e4m3 x e4m3 -> e5m2"
    rej_status = sorted(set(st[st != 0].tolist()))
    assert rej_status == [15], (what, rej_status)
    keep = keep[st == 0]
    keep = keep[np.lexsort(key[keep].T[::-1])]
    return keep, dict(pairs=n_pairs, differ=n_diff, status_nonzero=int((st != 0).sum()))


def _pkg(c: dict, idx: np.ndarray) -> dict:
    ws_code = {w: i for i, w in enumerate(WS_CODES)}
    cols = {}
    for name, dt in PKG_COLS:
        v = c[name][idx] if name != "ws" else np.array([ws_code[int(w)] for w in c["ws"][idx]], dtype=np.int64)
        assert v.min() >= np.iinfo(dt).min and v.max() <= np.iinfo(dt).max, name
        cols[name] = v.astype(dt)
    return cols


def _load(files) -> dict:
    zs = [np.load(f) for f in files]
    return {k: np.concatenate([z[k].astype(np.int64) for z in zs]) for k in zs[0].files}


def replays() -> list[tuple[Path, dict]]:
    out = []
    for p in (REPLAY, REPLAY_P5):
        r = json.loads(p.read_text())
        assert r["format"] == "mmaemu-h100-replay/1"
        bad = [j["id"] for j in r["live"] if j["verdict"] not in ("equal", "rejected_as_hardware")]
        assert not bad, f"{p}: jobs that do not reproduce the hardware: {bad[:10]}"
        out.append((p, r))
    return out


def _live_splits(reps) -> list[dict]:
    return [j for _p, r in reps for j in r["live"] if j["verdict"] == "equal" and (j["S"] or 1) > 1]


def split_layouts(reps) -> dict:
    """What the replayed live split calls cover of the contiguous rule's layouts (mmaemu.platform.splitk_table)."""
    ragged = empty = 0
    for j in _live_splits(reps):
        assert j["scheme"] == "contiguous", j["id"]
        K, S = int(j["K"]), int(j["S"])
        T = -(-K // TILE_K)
        ragged += K % TILE_K != 0
        empty += (S - 1) * -(-T // S) >= T
    return {"ragged_K": {"verified": ragged > 0, "live_split_calls": ragged,
                         "what": f"K % {TILE_K} != 0: the last k-tile is zero padded"},
            "empty_split": {"verified": empty > 0, "live_split_calls": empty,
                            "what": "(S - 1) * ceil(T / S) >= T: a split owns no k-tile"},
            "replays": [_rel(p) for p, _r in reps]}


def envelope_meta() -> dict:
    """The sweep's K-dependent envelope, as tested on the sealed p4 (bounds copied, never refitted)."""
    raw = RULE.read_bytes()
    rule, ev = json.loads(raw), json.loads(EVAL.read_text())
    assert rule["format"] == "mmaemu-h100-splitk-envelope/1" and rule["table_id"] == TABLE_ID
    assert ev["format"] == "mmaemu-h100-splitk-envelope-eval/1"
    assert hashlib.sha256(raw).hexdigest() == ev["provenance"]["rule_sha256"], "the rule is not the one tested on p4"
    r = rule["rule"]
    bounds = []
    for s in rule["settings"]:
        assert s["ws"] in EVAL_PROFILES and [g["K"] for g in s["grid"]] == r["K_grid"], s
        bounds.append(dict(ws=s["ws"], out=s["out"], fast=s["fast"], fmt_a=s["fmt_a"], fmt_b=s["fmt_b"],
                           B=[g["B"] for g in s["grid"]]))
    validation = {}
    for ws, prof in EVAL_PROFILES.items():
        e = ev[prof]
        validation[str(ws)] = dict(status="validated" if e["passed"] else "unvalidated", eval_profile=prof,
                                   rows=e["rows"], tested=e["tested"], split_above_bound=e["split_tested"],
                                   failures=e["failures"], closest_split=e["closest_split"])
    said = "; ".join(f"{int(ws) >> 20} MiB {v['status']} ({v['tested']:,} held-out shapes above the bound, "
                     f"{v['split_above_bound']} split)" for ws, v in validation.items())
    return {"inference": True, "kind": "k_grid", "margin": r["margin"], "floor": r["floor"], "K_grid": r["K_grid"],
            "K_min": r["K_min"], "K_max": r["K_max"], "claim": r["claim"], "K_lookup": r["K_lookup"],
            "assumptions": r["assumptions"], "bounds": bounds, "validation": validation,
            "rule": _rel(RULE), "rule_sha256": hashlib.sha256(raw).hexdigest(), "eval": _rel(EVAL),
            "eval_sha256": _sha(EVAL), "eval_commit": ev["provenance"]["commit"],
            "note": f"K-dependent bound of the 2026-10-16 split-K sweep ({_rel(SWEEP)}): S = 1 for M*N above "
                    f"B(setting, K), K rounded up to the next of {len(r['K_grid'])} grid values, "
                    f"{r['K_min']:,} <= K <= {r['K_max']:,}; sealed held-out test: {said}"}


def build():
    raw = HEUR.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    want = sums_sha256(HEUR)
    assert sha == want, f"{HEUR}: sha256 {sha} != SHA256SUMS {want}"
    side = json.loads(HEUR.with_suffix(".json").read_text())
    hw = side["hardware"]
    _check_hardware(hw, "probe")
    assert side["library_version_code"] == LIB_CODE
    z = np.load(HEUR)
    c = {k: z[k].astype(np.int64) for k in z.files}
    keep, probe_counts = _status0_rows(c, "probe")
    probe = _pkg(c, keep)
    ws0_q = c["ws"] == 0
    assert (c["splitK"][ws0_q] <= 1).all(), "a workspace-0 query of the probe split"
    # p3: the sweep's model rows
    p3_files = sorted(P3.glob("*_a*_c*.npz"))
    for f in p3_files:
        m = json.loads(f.with_suffix(".json").read_text())
        _check_hardware(m["hardware"], f.name)
        assert m["npz_sha256"] == _sha(f), f"{f}: sha256 is not its sidecar's"
    c3 = _load(p3_files)
    assert set(np.unique(c3["ws"]).tolist()) == set(EVAL_PROFILES)
    keep3, p3_counts = _status0_rows(c3, "p3")
    p3 = _pkg(c3, keep3)
    # 4. merge
    both = {k: np.concatenate([probe[k], p3[k]]) for k in probe}
    key = np.stack([both[k].astype(np.int64) for k in KEY], 1)
    o = np.lexsort(key.T[::-1])
    ks = key[o]
    dup = np.concatenate([[False], np.all(ks[1:] == ks[:-1], 1)])
    names = [n for n, _ in PKG_COLS]
    vals = np.stack([both[n].astype(np.int64) for n in names], 1)[o]
    d = np.flatnonzero(dup)
    differ = np.any(vals[d] != vals[d - 1], 1)
    ci, si = names.index("custom"), names.index("splitK")
    only_custom = np.all(np.delete(vals[d] == vals[d - 1], ci, 1), 1) & (vals[d, si] == 1) & (vals[d - 1, si] == 1)
    assert not (differ & ~only_custom).any(), "a key of both recordings with different values"
    custom_differs = [dict(zip(KEY, (int(x) for x in ks[i])), probe=int(vals[i - 1, ci]), p3=int(vals[i, ci]))
                      for i in d[differ]]                  # unsplit rows only; the probe's (first) row is kept
    for r in custom_differs:
        r["ws"] = WS_CODES[r["ws"]]
    cols = {k: v[o][~dup] for k, v in both.items()}
    # 5. what the split rule rests on
    sp = cols["splitK"] > 1
    Ks, Ss = cols["K"][sp].astype(np.int64), cols["splitK"][sp].astype(np.int64)
    T = -(-Ks // TILE_K)
    assert (Ks % TILE_K == 0).all(), "a split row with K % 128 != 0"
    assert ((Ss - 1) * -(-T // Ss) < T).all(), "a split row with a split that owns no k-tile"
    assert (cols["reduction"][sp] == REDUCTION_COMPUTE_TYPE).all()
    assert (cols["ws"][sp] > 0).all(), "a split row without a workspace"
    split_customs = sorted(int(x) for x in np.unique(cols["custom"][sp]))
    ws0 = cols["ws"] == 0
    p1w = _load(sorted(P1_WS0.glob("ws0_a*_c*.npz")))
    p4w = _load([P4_WS0])
    for name, w in (("sweep p1", p1w), ("sealed p4", p4w)):
        assert (w["ws"] == 0).all() and not ((w["status"] == 0) & (w["splitK"] > 1)).any(), f"{name}: ws0 split"
    reps = replays()
    live_customs = sorted({int(j["custom"]) for j in _live_splits(reps)})
    # per-setting counts (reported)
    settings = []
    skey = np.stack([cols[k].astype(np.int64) for k in ("ws", "out", "fast", "fa", "fb")], 1)
    uniq, inv_idx = np.unique(skey, axis=0, return_inverse=True)
    MN = cols["M"].astype(np.int64) * cols["N"].astype(np.int64)
    for i, s in enumerate(uniq):
        sel = inv_idx.ravel() == i
        ss = sel & sp
        settings.append(dict(ws=WS_CODES[int(s[0])], out=OUT_NAMES[int(s[1])], fast=bool(s[2]),
                             fmt_a=FMT_NAMES[int(s[3])], fmt_b=FMT_NAMES[int(s[4])],
                             rows=int(sel.sum()), split_rows=int(ss.sum()),
                             max_split_MN=int(MN[ss].max()) if ss.any() else None,
                             min_split_K=int(cols["K"][ss].min()) if ss.any() else None,
                             S_values=sorted(int(x) for x in np.unique(cols["splitK"][ss])) if ss.any() else []))
    k_grid = sorted(int(x) for x in np.unique(cols["K"]))
    n_live = len(_live_splits(reps))
    meta = {
        "format": "mmaemu-splitk-table/1",
        "table_id": TABLE_ID,
        "profile_device": dict(HARDWARE, torch=hw["torch"], driver=hw["driver"]),
        "descriptor": "torch 2.7.1 scaled_gemm (m=N, n=M, k=K; A=mat2 transa=T; B=mat1; C=out dtype, fp16 for fp8 "
                      "outputs; D=out dtype; A/B scale pointers; FAST_ACCUM; MAX_WORKSPACE_BYTES), requestedAlgoCount 1",
        "key": KEY,
        "columns": {n: np.dtype(dt).name for n, dt in PKG_COLS},
        "codes": {"fa/fb": {str(k): v for k, v in FMT_NAMES.items()}, "out": {str(k): v for k, v in OUT_NAMES.items()},
                  "ws": WS_CODES,
                  "custom -> scheme": {str(x): "contiguous" for x in live_customs},
                  "custom -> scheme, unlisted": "contiguous",
                  "tile/stages/reduction": "cublasLt.h enum values (CUDA 12.8)"},
        "split_rule": "contiguous: 128-wide k-tiles, split s owns tiles [s*q, (s+1)*q) with q = ceil(T / S); fast one "
                      "chain per split, slow every tile its own chain added serially from +0; partials added "
                      "serially; the customOption does not change the result (found after the data came in on the "
                      "1,049 live split calls of docs/validation_logs/2026-10-15/h100-probe; the 2,416 live split "
                      "calls of docs/validation_logs/2026-10-16/h100-splitk-sweep, run after it was fixed, replay "
                      f"equal; {n_live:,} live split calls in all, customOption {', '.join(map(str, live_customs))})",
        "split_layouts": split_layouts(reps),
        "envelope": envelope_meta(),
        "workspace0_note": f"cuBLASLt never splits without a workspace (0 split rows among the {int(ws0_q.sum()):,} "
                           f"workspace-0 queries of the probe, the {len(p1w['M']):,} of the sweep's p1 and the "
                           f"{len(p4w['M']):,} of its sealed p4; {int(ws0.sum()):,} table rows)",
        "verified_combos": VERIFIED,
        "rejects": [{"fmt_a": "e4m3", "fmt_b": "e4m3", "out": "float8_e5m2", "status": [15],
                     "rows": probe_counts["status_nonzero"] + p3_counts["status_nonzero"]}],
        "sources": [{"id": "h100-probe.m6_heur", "path": _rel(HEUR), "sha256": sha,
                     "sidecar": _rel(HEUR.with_suffix(".json")), "sidecar_sha256": _sha(HEUR.with_suffix(".json")),
                     "rows": int(len(probe["M"]))}]
                   + [{"id": f"h100-splitk-sweep.p3.{f.stem}", "path": _rel(f), "sha256": _sha(f),
                       "sidecar": _rel(f.with_suffix(".json")), "sidecar_sha256": _sha(f.with_suffix(".json"))}
                      for f in p3_files],
        "p3_rows": int(len(p3["M"])), "rows_in_both": int(dup.sum()),
        "unsplit_customOption_differs": {
            "note": "keys in both recordings whose unsplit rows (S = 1, same algo / tile / stages / reduction / swizzle) "
                    "differ in customOption only; the probe's row is kept. cuBLASLt's heuristic gave another "
                    "customOption for the same key in the sweep's p3 run (its p0 re-run of the probe's grid was "
                    "identical), so the customOption of an unsplit row is not a function of the key alone",
            "rows": len(custom_differs), "keys": custom_differs},
        "align_invariance": {"probe": {k: probe_counts[k] for k in ("pairs", "differ")},
                             "p3": {k: p3_counts[k] for k in ("pairs", "differ")}},
        "dscale_invariance": "not recorded (dscale 0 only)",
        "K_grid": k_grid,
        "rows": int(len(cols["M"])), "split_rows": int(sp.sum()),
        "split_customOptions": split_customs,
        "dropped_status_nonzero": probe_counts["status_nonzero"] + p3_counts["status_nonzero"],
        "settings": settings,
    }
    return cols, meta, reps


def verified_combos(cols: dict, reps) -> dict:
    tab = set()
    idx = np.nonzero(cols["splitK"] > 1)[0]
    keys = np.stack([cols[k][idx].astype(np.int64) for k in ("fast", "splitK", "out", "fa", "fb")], 1)
    for f, S, o, fa, fb in np.unique(keys, axis=0):
        tab.add((bool(f), int(S), "contiguous", OUT_NAMES[int(o)], FMT_NAMES[int(fa)] + "x" + FMT_NAMES[int(fb)]))
    hw: dict = {}
    for j in _live_splits(reps):
        k = (bool(j["fast"]), int(j["S"]), j["scheme"], j["out"], f"{j['fa']}x{j['fb']}")
        hw.setdefault(k, []).append(j["id"])
    combos = []
    for k in sorted(tab | set(hw)):
        combos.append(dict(fast=k[0], S=k[1], scheme=k[2], out=k[3], fmt_pair=k[4], in_table=k in tab,
                           verified=k in hw, recordings=sorted(hw.get(k, []))[:8], n_recordings=len(hw.get(k, []))))
    return {"format": "mmaemu-verified-split-combos/1", "table_id": TABLE_ID,
            "definition": "a (fast, S, scheme, out dtype, fmt pair) combination is verified when a live H100 call with "
                          "that LOGGED split config (numSplitsK > 1; the scheme from the table's customOption rule) "
                          "exists in docs/validation_logs/2026-10-15/h100-probe or docs/validation_logs/2026-10-16/"
                          "h100-splitk-sweep and mmaemu reproduces its full output bit for bit (sha256; "
                          "tools/replay_h100_log.py -> results/analysis/mmaemu_replay.json, mmaemu_replay_p5.json)",
            "replays": [dict(path=_rel(p), sha256=_sha(p)) for p, _r in reps],
            "n_table_combos": len(tab), "n_verified_table_combos": sum(1 for c in combos if c["in_table"] and
                                                                         c["verified"]),
            "n_combos": len(combos), "n_verified_combos": sum(1 for c in combos if c["verified"]),
            "unverified_table_combos": [c for c in combos if c["in_table"] and not c["verified"]],
            "combos": combos}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare with the committed files; write nothing")
    a = ap.parse_args()
    cols, meta, reps = build()
    npz = deterministic_npz(cols)
    meta["npz_sha256"] = hashlib.sha256(npz).hexdigest()
    meta["npz_bytes"] = len(npz)
    js = (json.dumps(meta, indent=1, sort_keys=True) + "\n").encode()
    vc = (json.dumps(verified_combos(cols, reps), indent=1, sort_keys=True) + "\n").encode()
    files = {f"{TABLE_ID}.npz": npz, f"{TABLE_ID}.json": js, VERIFIED: vc}
    if a.check:
        bad = [n for n, b in files.items() if not (DATA / n).exists() or (DATA / n).read_bytes() != b]
        print("check:", "OK" if not bad else f"DIFFERS {bad}")
        sys.exit(1 if bad else 0)
    DATA.mkdir(parents=True, exist_ok=True)
    for n, b in files.items():
        (DATA / n).write_bytes(b)
        print(f"wrote {DATA / n} ({len(b)} bytes, sha256 {hashlib.sha256(b).hexdigest()})")
    print(json.dumps({k: meta[k] for k in ("rows", "p3_rows", "rows_in_both", "split_rows", "dropped_status_nonzero",
                                           "align_invariance", "split_customOptions", "K_grid")}, indent=1))
    print(meta["codes"]["custom -> scheme"], meta["split_layouts"], meta["workspace0_note"], sep="\n")
    print(meta["envelope"]["note"])


if __name__ == "__main__":
    main()
