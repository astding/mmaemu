"""Check that the RTX 4060 reference files shipped in the repository equal the captures of this log.

    python check_reference.py --goldens <dir with the capture npz files> [--xlsx 41586_2018_461_MOESM3_ESM.xlsx]

Checked, array by array (np.array_equal):
  examples/data/evo2_1b_ada_rtx4060_te.npz
    <w>/ids, <w>/acgt, <w>/next          == evo2_1b_base_real-te-fp8.npz                    (golden_real.py)
    brca1/rs, brca1/vs                   == evo2_1b_base_real-te-fp8_brca1-subset500_scores.npz (brca1_source_script.py)
    brca1/pos, ref, alt, lof             == the upstream notebook's table, sample(n=500, random_state=0) (with --xlsx)
  examples/data/evo2_1b_ada_rtx4060_te.json: per window, bp, mean_next_logprob and logits_sha256 == data/*.json here
  docs/validation_logs/2026-10-15/evo2-40b-rtx-5070-ti/rtx4060_official_te_40b.npz
    w2k/*, w4k/*                         == evo2_40b_oracle-te-fp8_w2k.npz, _w4k.npz       (golden_oracle.py)
and, for every capture used: the sha256 of its logits_bits equals the logits_sha256 its .json records, and the 1B
oracle (blocks one at a time) equals the real upstream model bit for bit on all six windows. Exits 1 on a mismatch."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SHIPPED_1B = ROOT / "examples" / "data" / "evo2_1b_ada_rtx4060_te"
SHIPPED_40B = ROOT / "docs" / "validation_logs" / "2026-10-15" / "evo2-40b-rtx-5070-ti" / "rtx4060_official_te_40b.npz"
REAL_1B = "evo2_1b_base_real-te-fp8"
ORACLE_1B = "evo2_1b_base_oracle-te-fp8_p0-p1-p2-p3-w8k-w2k"
BRCA1 = "evo2_1b_base_real-te-fp8_brca1-subset500"
ORACLE_40B = ("evo2_40b_oracle-te-fp8_w2k", "evo2_40b_oracle-te-fp8_w4k")

failures = []


def check(what, ok):
    print(f"{'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        failures.append(what)


def windows_of(npz):
    return sorted({k.split("/")[0] for k in npz.files})


def check_capture_sha(tag, npz):
    meta = json.loads((HERE / "data" / f"{tag}.json").read_text())
    for w in meta["windows"]:
        got = hashlib.sha256(np.ascontiguousarray(npz[f"{w['window']}/logits_bits"]).tobytes()).hexdigest()
        check(f"{tag} {w['window']}: sha256(logits_bits) == logits_sha256 of its .json", got == w["logits_sha256"])
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--goldens", type=Path, required=True)
    ap.add_argument("--xlsx", type=Path)
    a = ap.parse_args()

    real = np.load(a.goldens / f"{REAL_1B}.npz")
    oracle = np.load(a.goldens / f"{ORACLE_1B}.npz")
    real_meta = check_capture_sha(REAL_1B, real)
    check_capture_sha(ORACLE_1B, oracle)
    for w in windows_of(real):
        check(f"1B {w}: oracle logits == real upstream logits", np.array_equal(real[f"{w}/logits_bits"],
                                                                              oracle[f"{w}/logits_bits"]))

    ship = np.load(f"{SHIPPED_1B}.npz")
    ship_meta = json.loads(Path(f"{SHIPPED_1B}.json").read_text())
    check("shipped 1B windows == captured windows", sorted(ship_meta["windows"]) == windows_of(real))
    for w in windows_of(real):
        for k in ("ids", "acgt", "next"):
            check(f"shipped 1B {w}/{k} == {REAL_1B}", np.array_equal(ship[f"{w}/{k}"], real[f"{w}/{k}"]))
    for w in real_meta["windows"]:
        s = ship_meta["windows"][w["window"]]
        check(f"shipped 1B .json {w['window']}: bp, mean_next_logprob, logits_sha256 == capture",
              (s["bp"], s["mean_next_logprob"], s["logits_sha256"]) == (w["bp"], w["mean_next_logprob"],
                                                                        w["logits_sha256"]))

    scores = np.load(a.goldens / f"{BRCA1}_scores.npz")
    for k in ("rs", "vs"):
        check(f"shipped brca1/{k} == {BRCA1}_scores", np.array_equal(ship[f"brca1/{k}"], scores[k]))
    if a.xlsx:
        import pandas as pd
        df = pd.read_excel(a.xlsx, header=2)
        df = df[["position (hg19)", "reference", "alt", "func.class"]]
        df.columns = ["pos", "ref", "alt", "class"]
        df = df.sample(n=500, random_state=0).reset_index(drop=True)
        for k in ("pos", "ref", "alt"):
            check(f"shipped brca1/{k} == the notebook's sample", np.array_equal(ship[f"brca1/{k}"], df[k].values))
        check("shipped brca1/lof == the notebook's sample", np.array_equal(ship["brca1/lof"],
                                                                         (df["class"] == "LOF").values))

    ref40 = np.load(SHIPPED_40B)
    for tag in ORACLE_40B:
        g = np.load(a.goldens / f"{tag}.npz")
        check_capture_sha(tag, g)
        for w in windows_of(g):
            for k in ("acgt", "next"):
                check(f"40B log {w}/{k} == {tag}", np.array_equal(ref40[f"{w}/{k}"], g[f"{w}/{k}"]))

    print(f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
