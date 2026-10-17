"""evo2_offload.py's outputs here against the runs they repeat; prints FAILURES 0 when everything matches.

    python compare.py 1b    test_1b_<profile>.json: per prompt, the logits SHA-256, loss and accuracy;
                            score_1b_<profile>.tsv: the scores of the first 20 BRCA1 variants, bit for bit
    python compare.py 40b   test_40b_hopper-te.json: per prompt, the logits SHA-256, loss and accuracy
The references: for hopper-te the runs of this log (../results/1b_test.json, 1b_brca1.npz, 40b_test.json,
mode "mmaemu_h100"); for ada-te, those of the 40B log of 15 October
(../../../2026-10-15/evo2-40b-rtx-5070-ti/check_1b.json and brca1_check_1b.npz, mode "mmaemu")."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[2] / "2026-10-15" / "evo2-40b-rtx-5070-ti"
RES = HERE.parent / "results"
which = sys.argv[1]
bad = 0
for prof in ("ada-te", "hopper-te") if which == "1b" else ("hopper-te",):
    if prof == "ada-te":
        ref, mode = json.loads((OLD / "check_1b.json").read_text()), "mmaemu"
    else:
        ref, mode = json.loads((RES / f"{which}_test.json").read_text()), "mmaemu_h100"
    want = {r["window"]: r for r in ref["rows"] if r["mode"] == mode}
    got = json.loads((HERE / f"test_{which}_{prof}.json").read_text())
    for k, r in enumerate(got["sequences"]):
        w = want[f"p{k}"]
        ok = (r["logits_sha256"], r["loss"], r["accuracy"]) == (w["logits_sha256"], w["loss"], w["accuracy"])
        bad += not ok
        print(which, prof, f"p{k}", "logits sha256 + loss + accuracy equal:", ok)
    print(which, prof, "mean loss", got["mean_loss"], "profile", got.get("profile"))
    if which == "1b":
        rows = list(csv.DictReader(open(HERE / f"score_1b_{prof}.tsv"), delimiter="\t"))
        nref = sum(r["name"].startswith("ref") for r in rows)
        if prof == "ada-te":
            z = np.load(OLD / "brca1_check_1b.npz")
            exp = np.r_[z[f"{mode}/rs"], z[f"{mode}/vs"]]
        else:
            z = np.load(RES / "1b_brca1.npz")           # the 500 variants; a prefix of 20 has a prefix of the refs
            exp = np.r_[z[f"{mode}/rs"][:nref], z[f"{mode}/vs"][:len(rows) - nref]]
        exp = exp.astype(np.float32)
        s = np.array([np.float32(float(r["score"])) for r in rows], dtype=np.float32)
        ok = s.shape == exp.shape and np.array_equal(s.view(np.int32), exp.view(np.int32))
        bad += not ok
        print(which, prof, f"{len(s)} BRCA1 scores bit for bit equal:", ok)
print("FAILURES", bad)
