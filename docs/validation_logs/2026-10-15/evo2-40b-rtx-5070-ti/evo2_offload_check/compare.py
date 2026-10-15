"""evo2_offload.py's outputs here against the scripts of this log; prints FAILURES 0 when everything matches.

    python compare.py 1b    test_1b_<mode>.json: per prompt, the logits SHA-256, loss and accuracy of ../check_1b.json;
                            score_1b_<mode>.tsv: the scores of ../brca1_check_1b.npz, bit for bit
    python compare.py 40b   test_40b_mmaemu.json: per prompt, the logits SHA-256, loss and accuracy of ../test_40b.json"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
which = sys.argv[1]
ref = json.loads((HERE.parent / ("check_1b.json" if which == "1b" else "test_40b.json")).read_text())
want = {(r["mode"], r["window"]): r for r in ref["rows"]}
bad = 0
for m in ("mmaemu", "native", "bf16") if which == "1b" else ("mmaemu",):
    got = json.loads((HERE / f"test_{which}_{m}.json").read_text())
    for k, r in enumerate(got["sequences"]):
        w = want[(m, f"p{k}")]
        ok = (r["logits_sha256"], r["loss"], r["accuracy"]) == (w["logits_sha256"], w["loss"], w["accuracy"])
        bad += not ok
        print(which, m, f"p{k}", "logits sha256 + loss + accuracy == this log's script:", ok)
    print(which, m, "mean loss", got["mean_loss"])
    if which == "1b":
        z = np.load(HERE.parent / "brca1_check_1b.npz")
        exp = np.r_[z[f"{m}/rs"], z[f"{m}/vs"]].astype(np.float32)
        rows = list(csv.DictReader(open(HERE / f"score_1b_{m}.tsv"), delimiter="\t"))
        s = np.array([np.float32(float(r["score"])) for r in rows], dtype=np.float32)
        ok = s.shape == exp.shape and np.array_equal(s.view(np.int32), exp.view(np.int32))
        bad += not ok
        print(which, m, f"{len(s)} BRCA1 scores bit for bit == brca1_check_1b.npz (= score_sequences):", ok)
print("FAILURES", bad)
