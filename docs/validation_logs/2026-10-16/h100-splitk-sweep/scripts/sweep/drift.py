"""Part 0 drift gate (GO / NO-GO): this GPU's re-recording of the H100 probe's heuristic grid
(tools/record_heuristics.py --grid c5,evo2,evo2_small_m --ws 0,1,4,32, 850,320 queries) must equal the recording that heur_h100pcie_lt12.8.3.14
was exported from (ref/heur.npz, sha256 checked against the probe's SHA256SUMS entry), row for row in every column,
and the hardware must be an H100 PCIe with 114 SMs, cc 9.0 and cuBLASLt 12.8.3.14. The driver is recorded, not gated.
Writes SWEEP_OUT/drift.json; exit 0 = GO, 3 = NO-GO.
usage: drift.py NEW.npz"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import sc

REF = sc.HERE.parent / "ref"
REF_SHA = "4f349703def9b6f0d25cd6b41a622fcc86fad2a870d173c214d8fdea80be031d"
HW = {"gpu": "NVIDIA H100 PCIe", "capability": [9, 0], "sm_count": 114, "cublaslt": "12.8.3.14"}


def main():
    new = Path(sys.argv[1])
    assert sc.sha256(REF / "heur.npz") == REF_SHA, "reference recording sha256"
    a, b = np.load(REF / "heur.npz"), np.load(new)
    ha = json.loads((REF / "heur.json").read_text())["hardware"]
    hb = json.loads(new.with_suffix(".json").read_text())["hardware"]
    rep = dict(reference=dict(sha256=REF_SHA, hardware=ha), new=dict(path=str(new), sha256=sc.sha256(new),
                                                                       hardware=hb))
    hw_bad = {k: [v, hb.get(k)] for k, v in HW.items() if hb.get(k) != v}
    rep["hardware_mismatch"] = hw_bad
    rep["driver"] = dict(reference=ha.get("driver"), new=hb.get("driver"))
    diffs = {}
    same_len = len(a["M"]) == len(b["M"])
    rep["rows"] = dict(reference=int(len(a["M"])), new=int(len(b["M"])))
    if same_len:
        for c in sc.COLS:
            d = int((a[c] != b[c]).sum())
            if d:
                diffs[c] = d
        bad = np.zeros(len(a["M"]), bool)
        for c in diffs:
            bad |= a[c] != b[c]
        rep["rows_differing"] = int(bad.sum())
        rep["examples"] = [{c: [int(a[c][i]), int(b[c][i])] for c in sc.COLS if a[c][i] != b[c][i]}
                           | {k: int(a[k][i]) for k in ("M", "N", "K", "fa", "fb", "out", "fast", "ws", "align")}
                           for i in np.nonzero(bad)[0][:20]]
    rep["column_diffs"] = diffs
    go = same_len and not diffs and not hw_bad
    rep["verdict"] = "GO" if go else "NO-GO"
    (sc.OUT / "drift.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(f"drift gate: {rep['verdict']} (rows {rep['rows']}, column diffs {diffs}, hardware mismatch {hw_bad}, "
          f"driver {rep['driver']})")
    sys.exit(0 if go else 3)


if __name__ == "__main__":
    main()
