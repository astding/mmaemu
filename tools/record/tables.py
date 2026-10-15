"""Union of the cuBLASLt heuristic rows recorded on the RTX 4060: the truth table heur.lt_heuristic_12_8_3_14 plus
the pre-swap rows R2 (fp8 outputs), R2x (mixed formats at every setting, small-K set) and R3 (Evo 2 small M).

Columns (int64): M N K fa fb out fast ws align status algo tile stages splitK reduction swizzle custom workspace
src (0 truth, 1 R2, 2 R2x mixed, 3 R2x small-K, 4 R3). out: 0 fp32, 1 bf16, 2 fp16, 3 e4m3, 4 e5m2.
R2 rows are taken at dscale = 0 after asserting that dscale = 1 gives identical rows (it does: 0 of 855,840 pairs
differ). Duplicate keys across sources must agree exactly (asserted)."""
from __future__ import annotations

import json

import numpy as np

import common

COLS = ["M", "N", "K", "fa", "fb", "out", "fast", "ws", "align", "status", "algo", "tile", "stages", "splitK",
        "reduction", "swizzle", "custom", "workspace"]
KEY = ["M", "N", "K", "fa", "fb", "out", "fast", "ws", "align"]
VAL = ["status", "algo", "tile", "stages", "splitK", "reduction", "swizzle", "custom", "workspace"]
OUT_NAMES = {0: "float32", 1: "bfloat16", 2: "float16", 3: "float8_e4m3fn", 4: "float8_e5m2"}
FMT_NAMES = {0: "e4m3", 1: "e5m2"}
SOURCES = {
    0: ("heur.lt_heuristic_12_8_3_14", common.TRUTH / "data" / "heur" / "lt_heuristic_12_8_3_14.npz"),
    1: ("r2.lt_heur_fp8out_12_8_3_14", common.REC_DATA / "r2" / "lt_heur_fp8out_12_8_3_14.npz"),
    2: ("r2.lt_heur_mixed_allsettings_12_8_3_14", common.REC_DATA / "r2" / "lt_heur_mixed_allsettings_12_8_3_14.npz"),
    3: ("r2.lt_heur_smallk_12_8_3_14", common.REC_DATA / "r2" / "lt_heur_smallk_12_8_3_14.npz"),
    4: ("r3.lt_heur_evo2_small_m_12_8_3_14", common.REC_DATA / "r3" / "lt_heur_evo2_small_m_12_8_3_14.npz"),
}


def _load(src: int) -> np.ndarray:
    z = np.load(SOURCES[src][1], allow_pickle=False)
    n = len(z["M"])
    if "dscale" in z.files and (z["dscale"] == 1).any():
        d = z["dscale"]
        a0 = np.stack([z[c] for c in COLS], 1)[d == 0]
        a1 = np.stack([z[c] for c in COLS], 1)[d == 1]
        o0 = np.lexsort(a0[:, :9].T[::-1])
        o1 = np.lexsort(a1[:, :9].T[::-1])
        assert np.array_equal(a0[o0], a1[o1]), f"{SOURCES[src][0]}: dscale changes the heuristic"
        arr = a0
    else:
        arr = np.stack([z[c] for c in COLS], 1)
        if "dscale" in z.files:
            assert not z["dscale"].any()
    assert arr.shape[0] in (n, n // 2)
    return np.concatenate([arr, np.full((arr.shape[0], 1), src, np.int64)], 1)


def load_union(sources=(0, 1, 2, 3, 4)) -> dict:
    """Column dict of the union; rows with an identical key in several sources are checked equal and kept once
    (first source wins)."""
    arr = np.concatenate([_load(s) for s in sources if SOURCES[s][1].exists()])
    kidx = [COLS.index(c) for c in KEY]
    vidx = [COLS.index(c) for c in VAL]
    order = np.lexsort(np.concatenate([arr[:, kidx], arr[:, -1:]], 1).T[::-1])  # primary M ... align, then src
    arr = arr[order]
    k = arr[:, kidx]
    same = np.all(k[1:] == k[:-1], 1)
    dup_v = np.any(arr[1:][:, vidx] != arr[:-1][:, vidx], 1)
    conflicts = int((same & dup_v).sum())
    assert conflicts == 0, f"{conflicts} duplicate keys with different values across sources"
    keep = np.concatenate([[True], ~same])
    arr = arr[keep]
    out = {c: arr[:, i] for i, c in enumerate(COLS)}
    out["src"] = arr[:, -1]
    out["n_duplicates_checked"] = int(same.sum())
    return out


def source_sha256() -> dict:
    F = common.truth_format()
    return {SOURCES[s][0]: F.sha256_file(p) for s, (_, p) in SOURCES.items() if p.exists()}


def combo_key(u: dict, i: int) -> tuple:
    S = int(u["splitK"][i])
    sch = "none" if S <= 1 else ("pairwise" if int(u["custom"][i]) == 1 else "serial")
    return (bool(u["fast"][i]), S, sch, OUT_NAMES[int(u["out"][i])],
            FMT_NAMES[int(u["fa"][i])] + "x" + FMT_NAMES[int(u["fb"][i])])


if __name__ == "__main__":
    u = load_union()
    print(json.dumps({"rows": int(len(u["M"])), "duplicates_checked": u["n_duplicates_checked"],
                      "split_rows": int((u["splitK"] > 1).sum()),
                      "per_source": {SOURCES[s][0]: int((u["src"] == s).sum()) for s in SOURCES}}, indent=1))
