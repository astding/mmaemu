"""Hardware coverage of split-K combinations (fast, S, scheme, out dtype, fmt pair).

A combination is hardware-replayed when some truth gemm output (no hw_error) carries params.splitk with that S > 1 and
scheme, from a captured cuBLASLt config / kernel profile; the 3 + 46 outputs whose split was filled from the
heuristic table by build/apply_ltconfig.py (sources "cuBLASLt heuristic table heur.lt_heuristic_12_8_3_14 ...") are
excluded (circular). The pre-swap recordings (R2-verify, R4, R5) add their own logged configs.

usage: python tools/record/coverage.py [--out FILE]   (prints the counts: 316 / 156 / 12,434 for the truth table)"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import tables  # noqa: E402

HEUR_SRC_PREFIX = "cuBLASLt heuristic table heur.lt_heuristic_12_8_3_14"


def hw_combos(root=None, rid_filter=None) -> dict:
    """{combo: sorted list of recording ids} over gemm recordings (truth by default; root= for other recordings)."""
    L = common.truth_loader()
    out = collections.defaultdict(set)
    kw = {} if root is None else {"root": str(root)}
    for e in L.list_recordings(**kw):
        if e["kind"] != "gemm" or (rid_filter and not rid_filter(e["id"])):
            continue
        r = L.load(e["id"], **kw)
        for c in r.cases:
            for o in c.outputs:
                if o.hw_error:
                    continue
                p = o.params
                sk = p.get("splitk")
                if not isinstance(sk, dict) or int(sk.get("S", 1)) <= 1:
                    continue
                if str(sk.get("source", "")).startswith(HEUR_SRC_PREFIX):
                    continue
                if sk.get("scheme") not in ("serial", "pairwise"):
                    continue
                key = (bool(p.get("fast_accum", True)), int(sk["S"]), sk["scheme"], p.get("out_dtype"),
                       p.get("fmt_a", "e4m3") + "x" + p.get("fmt_b", "e4m3"))
                out[key].add(e["id"])
    return {k: sorted(v) for k, v in out.items()}


def table_combos(u: dict, mask=None) -> collections.Counter:
    cnt = collections.Counter()
    idx = np.nonzero((u["splitK"] > 1) & (mask if mask is not None else True))[0]
    for i in idx:
        cnt[tables.combo_key(u, i)] += 1
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    u = tables.load_union()
    truth_tab = table_combos(u, u["src"] == 0)
    hw = hw_combos()
    unv = {k: n for k, n in truth_tab.items() if k not in hw}
    union_tab = table_combos(u)
    unv_union = {k: n for k, n in union_tab.items() if k not in hw}
    rep = dict(truth_table_combos=len(truth_tab), truth_table_split_rows=sum(truth_tab.values()),
               hw_combos=len(hw), unverified_truth_table=len(unv), unverified_rows=sum(unv.values()),
               s29=[list(k) for k in truth_tab if k[1] == 29],
               union_combos=len(union_tab), unverified_union=len(unv_union),
               unverified_union_rows=sum(unv_union.values()))
    print(json.dumps(rep, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(dict(
            report=rep,
            unverified_truth_table=[dict(combo=list(k), rows=n) for k, n in sorted(unv.items())],
            unverified_union_only=[dict(combo=list(k), rows=n) for k, n in sorted(unv_union.items()) if k not in unv],
            hardware_replayed={"|".join(map(str, k)): v for k, v in sorted(hw.items())}), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
