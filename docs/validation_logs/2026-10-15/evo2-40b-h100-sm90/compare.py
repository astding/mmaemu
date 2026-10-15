"""The H100 official 40B run against the RTX 4060 official run and the RTX 5070 Ti runs (mmaemu, native, bf16).

    python compare.py [--rtx4060-captures <dir>] [--h100-logits <results/40b/test.npz of run_job.sh>] [--out X.json]

From the files of this repository: the upstream test; per window, mean and max |Δ log p(A,C,G,T)| over every
position, the share of positions where the most likely of A, C, G, T agrees and the mean log p(next token), for
the RTX 5070 Ti runs (../evo2-40b-rtx-5070-ti/test_40b.npz) and the RTX 4060 (w2k, w4k:
../evo2-40b-rtx-5070-ti/rtx4060_official_te_40b.npz) against the H100 (h100_official_te_40b.npz); BRCA1: the
AUROC with a 95% bootstrap interval (2,000 resamples of the variants, seed 0) and, against the H100, Pearson and
Spearman of the variant score changes (Δ); the 1B logits on this H100 against those of 14 October (another
instance). With the RTX 4060 captures (evo2_40b_oracle-te-fp8_<w>.npz, maintainer data) and the H100's full
logits, also the share of bf16 logits that differ between the two official GPUs and the four 4,096-bp prompt
prefixes. Writes compare.json here by default."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
TI = HERE.parent / "evo2-40b-rtx-5070-ti"
ROOT = HERE.parents[3]

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--rtx4060-captures", type=Path)
ap.add_argument("--h100-logits", type=Path)
ap.add_argument("--out", type=Path, default=HERE / "compare.json")
a = ap.parse_args()


def load_json(p):
    return json.loads(Path(p).read_text())


def bf16(bits):
    return (bits.astype(np.uint16).astype(np.uint32) << 16).view(np.float32)


def window_stats(acgt, nxt, acgt_ref, nxt_ref):
    d = np.abs(acgt - acgt_ref)
    return {"mean_abs_dlogp_acgt": float(d.mean()), "max_abs_dlogp_acgt": float(d.max()),
            "acgt_argmax_agree": float((acgt.argmax(1) == acgt_ref.argmax(1)).mean()),
            "mean_next_logprob": float(nxt.mean()), "ref_mean_next_logprob": float(nxt_ref.mean())}


def auroc_ci(y, s, n=2000):
    rng, out = np.random.default_rng(0), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            out.append(roc_auc_score(y[i], s[i]))
    return [float(v) for v in np.percentile(out, [2.5, 97.5])]


H = np.load(HERE / "h100_official_te_40b.npz")
Ht = load_json(HERE / "results" / "40b" / "test.json")
res = {"upstream_test": {"h100_official": Ht["test_evo2_mean"], "expected": {"loss": 0.2159424, "accuracy": 0.91673},
                         "per_prompt": [r for r in Ht["sequences"] if r["name"] in ("p0", "p1", "p2", "p3")]}}

# the two official GPUs
G40 = np.load(TI / "rtx4060_official_te_40b.npz")
rows = []
for w in ("w2k", "w4k"):
    row = {"window": w, "length": len(H[f"{w}/ids"])}
    row.update(window_stats(H[f"{w}/acgt"], H[f"{w}/next"], G40[f"{w}/acgt"], G40[f"{w}/next"]))
    rows.append(row)
res["h100_vs_rtx4060_official"] = rows
if a.rtx4060_captures and a.h100_logits:
    full, rows = np.load(a.h100_logits), []
    for w in ("w2k", "w4k", "p0_4k", "p1_4k", "p2_4k", "p3_4k"):
        g = np.load(a.rtx4060_captures / f"evo2_40b_oracle-te-fp8_{w}.npz")
        assert np.array_equal(g[f"{w}/ids"], H[f"{w}/ids"]) and np.array_equal(full[f"{w}/acgt"], H[f"{w}/acgt"]), w
        x, y = full[f"{w}/logits_bits"], g[f"{w}/logits_bits"]
        row = {"window": w, "length": len(g[f"{w}/ids"]), "logits_differ": float((x != y).mean()),
               "logits_max_abs": float(np.abs(bf16(x) - bf16(y)).max())}
        row.update(window_stats(H[f"{w}/acgt"], H[f"{w}/next"], g[f"{w}/acgt"], g[f"{w}/next"]))
        rows.append(row)
    res["h100_vs_rtx4060_official_full"] = rows

# the RTX 5070 Ti (mmaemu, native, bf16) against the H100
R = np.load(TI / "test_40b.npz")
rows = []
for mode in ("mmaemu", "native", "bf16"):
    for w in ("w2k", "w4k", "p0", "p1", "p2", "p3"):
        row = {"mode": mode, "window": w}
        row.update(window_stats(R[f"{mode}/{w}/acgt"], R[f"{mode}/{w}/next"], H[f"{w}/acgt"], H[f"{w}/next"]))
        rows.append(row)
res["rtx5070ti_vs_h100_official"] = rows

# BRCA1
B = np.load(HERE / "results" / "40b" / "brca1.npz")
ri, lof = B["ref_index"], B["lof"]
dh = B["vs"] - B["rs"][ri]
br = {"h100_official": {"variants": len(lof), "lof": int(lof.sum()), "auroc": float(roc_auc_score(lof, -dh)),
                        "auroc_ci95": auroc_ci(lof, -dh),
                        "chunks": load_json(HERE / "results" / "40b" / "brca1.json")["chunks"]}}
L = np.load(TI / "brca1_40b.npz")
n = len(L["lof"])
assert np.array_equal(L["ref_index"], ri[:n]) and np.array_equal(L["lof"], lof[:n])
br["h100_official_first_100"] = {"variants": n, "lof": int(lof[:n].sum()), "auroc": float(roc_auc_score(lof[:n], -dh[:n])),
                                 "auroc_ci95": auroc_ci(lof[:n], -dh[:n]), "mean_abs_delta": float(np.abs(dh[:n]).mean())}
for mode in ("mmaemu", "native", "bf16"):
    d = L[f"{mode}/vs"] - L[f"{mode}/rs"][L["ref_index"]]
    s = np.r_[L[f"{mode}/rs"], L[f"{mode}/vs"]]
    sh = np.r_[B["rs"][:len(L[f"{mode}/rs"])], B["vs"][:n]]
    br[f"rtx5070ti_{mode}"] = {"variants": n, "auroc": float(roc_auc_score(L["lof"], -d)),
                               "auroc_ci95": auroc_ci(L["lof"], -d),
                               "delta_pearson_vs_h100": float(pearsonr(d, dh[:n])[0]),
                               "delta_spearman_vs_h100": float(spearmanr(d, dh[:n])[0]),
                               "delta_same_sign_vs_h100": float(np.mean(np.sign(d) == np.sign(dh[:n]))),
                               "max_abs_score_diff_vs_h100": float(np.abs(s - sh).max())}
A = np.load(ROOT / "examples" / "data" / "evo2_1b_ada_rtx4060_te.npz")   # the 1B, official path, RTX 4060
assert np.array_equal(A["brca1/lof"], lof)
d1 = A["brca1/vs"] - A["brca1/rs"][ri]
br["evo2_1b_rtx4060_official"] = {"auroc": float(roc_auc_score(lof, -d1)), "auroc_ci95": auroc_ci(lof, -d1),
                                  "auroc_first_100": float(roc_auc_score(lof[:n], -d1[:n])),
                                  "auroc_first_100_ci95": auroc_ci(lof[:n], -d1[:n])}
res["brca1"] = br

res["method_check_1b"] = load_json(HERE / "results" / "1b_check.json")
old = {r["window"]: r["logits_sha256"] for r in
       load_json(HERE.parents[1] / "2026-10-14" / "evo2-h100-sm90" / "windows.json")["windows"]}
new = {r["name"]: r["logits_sha256"] for r in load_json(HERE / "results" / "1b_official" / "test.json")["sequences"]}
res["h100_1b_logits_equal_2026_10_14"] = {w: new.get(w) == v for w, v in old.items()}
a.out.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res, indent=1)[:8000])
