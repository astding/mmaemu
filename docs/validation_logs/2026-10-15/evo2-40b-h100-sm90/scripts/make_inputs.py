"""Inputs of the H100 40B run, built locally: the token ids of the test windows and of the BRCA1 windows.

Test windows: p0-p3 (the prompts of evo2/test/test_evo2.py), w8k / w4k / w2k (the first 8,192 / 4,096 / 2,048 bp of
their concatenation), p0_4k-p3_4k (the first 4,096 bp of each prompt), as defined by the RTX 4060 captures; the ids
are checked against those captures. BRCA1: the upstream notebook's 500-variant subset (sample(n=500, random_state=0)),
8,192-bp windows, with brca1_offload.py's subset() (the same windows as the RTX 5070 Ti run).
usage: python make_inputs.py <out.npz>"""
import csv
import sys
from importlib import resources

import numpy as np

sys.path.insert(0, "/data/quant/mmaemu_work/models/evo2_40b")
import brca1_offload  # noqa: E402

G = "/data/quant/evo2_ada_goldens/data"
out = sys.argv[1]
with resources.as_file(resources.files("evo2.test.data") / "prompts.csv") as p:
    prompts = [r[0] for r in list(csv.reader(open(p, encoding="utf-8-sig", newline="")))[1:]]
w8k = "".join(prompts)[:8192]
test = [(f"p{i}", s) for i, s in enumerate(prompts)] + [("w8k", w8k), ("w4k", w8k[:4096]), ("w2k", w8k[:2048])]
test += [(f"p{i}_4k", s[:4096]) for i, s in enumerate(prompts)]
ids = lambda s: np.frombuffer(s.encode("ascii"), dtype=np.uint8).astype(np.int16)  # noqa: E731  CharLevelTokenizer(512)

arrays, names = {}, []
real1b = np.load(f"{G}/evo2_1b_base_real-te-fp8.npz")
for name, s in test:
    arrays[f"test/{name}"] = ids(s)
    names.append(name)
    for src in [real1b] + [np.load(f"{G}/evo2_40b_oracle-te-fp8_{w}.npz") for w in ("w2k", "w4k", "p0_4k", "p1_4k",
                                                                                     "p2_4k", "p3_4k")]:
        if f"{name}/ids" in src.files:
            assert np.array_equal(src[f"{name}/ids"], arrays[f"test/{name}"]), name
            print("ids of", name, "== RTX 4060 capture")
arrays["test_names"] = np.array(names)

ref_seqs, var_seqs, ref_index, lof, ada = brca1_offload.subset(500)
assert len(ref_seqs) == 435 and len(var_seqs) == 500
for i, s in enumerate(ref_seqs):
    arrays[f"brca1/ref/{i}"] = ids(s)
for i, s in enumerate(var_seqs):
    arrays[f"brca1/var/{i}"] = ids(s)
arrays["brca1/ref_index"], arrays["brca1/lof"] = ref_index, lof
print("BRCA1:", len(ref_seqs), "reference and", len(var_seqs), "variant windows; lengths",
      sorted({len(s) for s in ref_seqs + var_seqs}), "; LOF", int(lof.sum()), "; first 100 variants use",
      int(ref_index[:100].max()) + 1, "reference windows")
np.savez_compressed(out, **arrays)
