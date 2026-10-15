"""h100_official_te_40b.npz from the full test output of the H100 run (its bf16 logits are not committed).

    python make_reference.py <results/40b/test.npz of run_job.sh>

Keeps, for every window, `ids`, `acgt` (log p of A, C, G, T at every position) and `next` (log p of every next
token), and checks that the SHA-256 of each window's logits equals the one results/40b/test.json records."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
full = np.load(sys.argv[1])
meta = json.loads((HERE / "results" / "40b" / "test.json").read_text())
out = {}
for r in meta["sequences"]:
    w = r["name"]
    assert hashlib.sha256(np.ascontiguousarray(full[f"{w}/logits_bits"]).tobytes()).hexdigest() == r["logits_sha256"], w
    for k in ("ids", "acgt", "next"):
        out[f"{w}/{k}"] = full[f"{w}/{k}"]
np.savez_compressed(HERE / "h100_official_te_40b.npz", **out)
print(f"h100_official_te_40b.npz: {len(meta['sequences'])} windows, logits SHA-256 checked")
