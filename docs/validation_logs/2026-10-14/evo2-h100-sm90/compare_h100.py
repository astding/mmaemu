"""Compare the H100 official-path baseline (TE FP8) with the Ada (RTX 4060) TE FP8 goldens.
usage: python compare_h100.py H100_DIR ADA_GOLDENS_DIR BRCA1_CSV > comparison.json
Window metrics are investigate.py's (diff + metrics), so they line up with the native-1 phase's vs_ada numbers."""
import csv, json, sys

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

H, G, CSV = sys.argv[1:4]
ACGT = torch.tensor([65, 67, 71, 84])


def diff(a, b):
    nd = int((a.view(torch.int16) != b.view(torch.int16)).sum())
    d = (a.float() - b.float()).abs()
    return dict(values=a.numel(), differing_values=nd, fraction=nd / a.numel(), max_abs=float(d.max()),
                mean_abs=float(d.mean()))


def metrics(lg, ref, ids):
    r = diff(lg, ref)
    lp, rp = lg.float().log_softmax(-1), ref.float().log_softmax(-1)
    dd = (lp[:, ACGT] - rp[:, ACGT]).abs()
    nxt = lambda p: p[:-1].gather(-1, ids[1:, None]).squeeze(-1)
    r.update(acgt_dlogp_mean=float(dd.mean()), acgt_dlogp_max=float(dd.max()),
             acgt_argmax_agree=float((lg[:, ACGT].argmax(-1) == ref[:, ACGT].argmax(-1)).float().mean()),
             mean_next_logprob=float(nxt(lp).mean()), mean_next_logprob_ada=float(nxt(rp).mean()))
    return r


h = np.load(f"{H}/evo2_1b_base_h100-te-fp8.npz")
a = np.load(f"{G}/evo2_1b_base_real-te-fp8.npz")
out = dict(windows={})
for w in ("p0", "p1", "p2", "p3", "w8k", "w2k"):
    assert np.array_equal(h[f"{w}/ids"], a[f"{w}/ids"])
    ids = torch.from_numpy(a[f"{w}/ids"].astype(np.int64))
    lg = torch.from_numpy(h[f"{w}/logits_bits"].copy()).view(torch.bfloat16)
    ref = torch.from_numpy(a[f"{w}/logits_bits"].copy()).view(torch.bfloat16)
    out["windows"][w] = metrics(lg, ref, ids)

rows = list(csv.DictReader(open(CSV)))
y = np.array([r["class"] == "LOF" for r in rows])
d_ada_csv = np.array([float(r["delta_te"]) for r in rows])
s = np.load(f"{H}/evo2_1b_base_h100-te-fp8_brca1-subset500_scores.npz")
s2 = np.load(f"{H}/evo2_1b_base_h100-te-fp8_brca1-subset500_scores_rerun.npz")
g = np.load(f"{G}/evo2_1b_base_real-te-fp8_brca1-subset500_scores.npz")
ri = s["ref_index"]
d_h, d_a = s["vs"] - s["rs"][ri], g["vs"] - g["rs"][ri]
assert np.allclose(d_a, d_ada_csv, rtol=0, atol=1e-6), "Ada golden deltas do not match the CSV's delta_te"
sh, sa = np.r_[s["rs"], s["vs"]], np.r_[g["rs"], g["vs"]]
out["brca1"] = dict(
    variants=len(rows), lof=int(y.sum()), unique_ref_windows=len(s["rs"]),
    auroc_h100=round(float(roc_auc_score(y, -d_h)), 4), auroc_ada=round(float(roc_auc_score(y, -d_a)), 4),
    rerun_bit_identical=bool(np.array_equal(s["rs"], s2["rs"]) and np.array_equal(s["vs"], s2["vs"])),
    seqscore_identical=int((sh == sa).sum()), seqscore_total=len(sh),
    seqscore_max_abs_diff=float(np.abs(sh - sa).max()), seqscore_mean_abs_diff=float(np.abs(sh - sa).mean()),
    seqscore_pearson=float(pearsonr(sh, sa)[0]),
    delta_pearson=float(pearsonr(d_h, d_a)[0]), delta_spearman=float(spearmanr(d_h, d_a)[0]),
    delta_sign_agree=float(np.mean(np.sign(d_h) == np.sign(d_a))),
    delta_max_abs_diff=float(np.abs(d_h - d_a).max()), delta_abs_mean_ada=float(np.abs(d_a).mean()))
print(json.dumps(out, indent=1))
