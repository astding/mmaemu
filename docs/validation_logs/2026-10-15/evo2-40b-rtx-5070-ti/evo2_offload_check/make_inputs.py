"""The input of the evo2_offload.py check: the 40 BRCA1 windows of brca1_check_1b.json as a FASTA file.

    python make_inputs.py --data <dir with 41586_2018_461_MOESM3_ESM.xlsx and GRCh37.p13_chr17.fna.gz>

Writes brca1_40.fasta: the reference and variant windows of the first 20 variants of the notebook's subset
(ref0, ref1, ..., then var0, var1, ...), from brca1_offload.py's subset(20), in the order of the scores in
brca1_check_1b.npz."""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import brca1_offload  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, required=True)
a = ap.parse_args()
brca1_offload.DATA = a.data
brca1_offload.ADA_1B = HERE.parents[4] / "examples" / "data" / "evo2_1b_ada_rtx4060_te.npz"
ref, var, _, _, _ = brca1_offload.subset(20)
with open(HERE / "brca1_40.fasta", "w") as f:
    for i, s in enumerate(ref):
        f.write(f">ref{i}\n{s}\n")
    for i, s in enumerate(var):
        f.write(f">var{i}\n{s}\n")
print(f"brca1_40.fasta: {len(ref)} reference and {len(var)} variant windows")
