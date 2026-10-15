"""official_blockwise.py against the real upstream model on this GPU (1B): logits and BRCA1 scores, bit for bit, and
the size of any difference. usage: python check_1b.py <official dir> <real dir> <out.json>"""
import json
import sys

import numpy as np

o, r, out = sys.argv[1:4]
ot, rt = json.load(open(f"{o}/test.json")), json.load(open(f"{r}/test.json"))
on, rn = np.load(f"{o}/test.npz"), np.load(f"{r}/test.npz")


def f32(bits):
    return (bits.astype(np.uint16).astype(np.uint32) << 16).view(np.float32)


rows = []
for x, y in zip(ot["sequences"], rt["sequences"]):
    assert x["name"] == y["name"]
    a, b = on[f"{x['name']}/logits_bits"], rn[f"{x['name']}/logits_bits"]
    rows.append(dict(name=x["name"], logits_equal=x["logits_sha256"] == y["logits_sha256"],
                     logits_differing=int((a != b).sum()), logits_max_abs_diff=float(np.abs(f32(a) - f32(b)).max())))
ob, rb = np.load(f"{o}/brca1.npz"), np.load(f"{r}/brca1.npz")
scores_equal = all(np.array_equal(ob[k].view(np.int32), rb[k].view(np.int32)) for k in ("rs", "vs"))
diff = max(float(np.abs(ob[k] - rb[k]).max()) for k in ("rs", "vs"))
res = dict(test=rows, all_logits_equal=all(x["logits_equal"] for x in rows), brca1_scores_equal=bool(scores_equal),
           brca1_max_abs_score_diff=diff, brca1_windows=int(len(rb["rs"]) + len(rb["vs"])),
           brca1_chunks_official=json.load(open(f"{o}/brca1.json"))["chunks"])
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res))
sys.exit(0 if res["all_logits_equal"] and res["brca1_scores_equal"] else 1)
