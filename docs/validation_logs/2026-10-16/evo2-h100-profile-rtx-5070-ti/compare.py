"""Evo 2 with mmaemu's H100 PCIe profile (hopper-te) on the RTX 5070 Ti, against the official runs and the RTX 4060
profile (ada-te). The runs were made on fa0fd90, where these profiles were named hopper-h100pcie-te and
ada-rtx4060-te.

    python compare.py [--out X.json]

From the files of this repository: the upstream test of the 1B and the 40B; per window, mean and max
|Δ log p(A,C,G,T)| over every position, the share of positions where the most likely of A, C, G, T agrees and the
mean log p(next token) (the measures of ../../2026-10-15/evo2-40b-h100-sm90/compare.py); BRCA1: the AUROC with a
95% bootstrap interval (2,000 resamples of the variants, seed 0) and, against each other run, Pearson and Spearman
of the variant score changes (Δ), the share with the same sign, and the largest difference of a window's score.
The references:
- the H100 PCIe, official TE FP8: the 1B of 14 October (../../2026-10-14/evo2-h100-sm90/: upstream test, BRCA1
  scores) and of 15 October (../../2026-10-15/evo2-40b-h100-sm90/results/1b_official/test.json: per window the loss,
  the mean log p(next) and the logits SHA-256; its per-position log-probabilities are not in the repository), the
  40B of 15 October (h100_official_te_40b.npz, results/40b/);
- the RTX 4060, official TE FP8: the 1B (examples/data/evo2_1b_ada_rtx4060_te.npz, 2026-10-14/evo2-h100-sm90/
  upstream_test.json) and the 40B (w2k, w4k: ../../2026-10-15/evo2-40b-rtx-5070-ti/rtx4060_official_te_40b.npz);
- the RTX 5070 Ti with mmaemu's RTX 4060 profile: the 1B notebook run (../../2026-10-14/evo2-rtx-5070-ti/) and its
  offload check (../../2026-10-15/evo2-40b-rtx-5070-ti/check_1b.json), the 40B (test_40b.*, brca1_40b.*).
Writes compare.json here by default."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
LOGS = HERE.parents[1]
ROOT = HERE.parents[3]
H14 = LOGS / "2026-10-14" / "evo2-h100-sm90"
TI14 = LOGS / "2026-10-14" / "evo2-rtx-5070-ti"
H40 = LOGS / "2026-10-15" / "evo2-40b-h100-sm90"
TI40 = LOGS / "2026-10-15" / "evo2-40b-rtx-5070-ti"
MODE = "mmaemu_h100"
EXPECTED = {"evo2_1b_base": {"loss": 0.501953125, "accuracy": 0.79556},
            "evo2_40b": {"loss": 0.2159424, "accuracy": 0.91673}}   # evo2/test/test_evo2.py (evo2 0.6.0)
TESTS = ("p0", "p1", "p2", "p3")

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--out", type=Path, default=HERE / "compare.json")
a = ap.parse_args()


def load_json(p):
    return json.loads(Path(p).read_text())


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def window_stats(acgt, nxt, acgt_ref, nxt_ref):
    d = np.abs(acgt - acgt_ref)
    return {"mean_abs_dlogp_acgt": float(d.mean()), "max_abs_dlogp_acgt": float(d.max()),
            "acgt_argmax_agree": float((acgt.argmax(1) == acgt_ref.argmax(1)).mean()),
            "acgt_bit_identical": bool(np.array_equal(acgt, acgt_ref)),
            "mean_next_logprob": float(nxt.mean()), "ref_mean_next_logprob": float(nxt_ref.mean())}


def auroc_ci(y, s, n=2000):
    rng, out = np.random.default_rng(0), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            out.append(roc_auc_score(y[i], s[i]))
    return [float(v) for v in np.percentile(out, [2.5, 97.5])]


def brca1_vs(rs, vs, ri, rs_ref, vs_ref):
    """Run (rs, vs) against a reference run of the same windows: the variant score changes and the window scores."""
    d, dr = vs - rs[ri], vs_ref - rs_ref[ri]
    s, sr = np.r_[rs, vs], np.r_[rs_ref, vs_ref]
    return {"delta_pearson": float(pearsonr(d, dr)[0]), "delta_spearman": float(spearmanr(d, dr)[0]),
            "delta_same_sign": float(np.mean(np.sign(d) == np.sign(dr))),
            "max_abs_score_diff": float(np.abs(s - sr).max()), "mean_abs_score_diff": float(np.abs(s - sr).mean()),
            "scores_bit_identical": int((s == sr).sum()), "scores": len(s)}


def brca1_run(rs, vs, ri, lof):
    d = vs - rs[ri]
    return {"variants": len(lof), "lof": int(lof.sum()), "auroc": float(roc_auc_score(lof, -d)),
            "auroc_ci95": auroc_ci(lof, -d), "mean_abs_delta": float(np.abs(d).mean())}


def test_mean(rows):
    r = [x for x in rows if x["window"] in TESTS]
    assert len(r) == 4
    return {k: float(np.mean([x[k] for x in r])) for k in ("loss", "accuracy", "loss_fp32")}


res = {"inputs": {}, "evo2_1b_base": {}, "evo2_40b": {}}
for f in sorted(RES.glob("*.json")) + sorted(RES.glob("*.npz")):
    res["inputs"][f.name] = sha256(f)
A = np.load(ROOT / "examples" / "data" / "evo2_1b_ada_rtx4060_te.npz")       # the 1B, official, RTX 4060

# ---- Evo 2 1B ----
if (RES / "1b_test.json").is_file():
    T, Z = load_json(RES / "1b_test.json"), np.load(RES / "1b_test.npz")
    rows = [x for x in T["rows"] if x["mode"] == MODE]
    h15 = load_json(H40 / "results" / "1b_official" / "test.json")
    hw = {s["name"]: s for s in h15["sequences"]}
    up14 = load_json(H14 / "upstream_test.json")
    r = {"expected": EXPECTED["evo2_1b_base"], "rtx5070ti_h100_profile": test_mean(rows),
         "h100_official": h15["test_evo2_mean"],
         "rtx4060_official": {"loss_fp32": up14["RTX 4060, official TE"]["loss"],
                              "accuracy": up14["RTX 4060, official TE"]["accuracy"]},
         "rtx5070ti_rtx4060_profile": load_json(TI40 / "check_1b.json")["test_evo2_mean"]["mmaemu"]}
    r["passes_upstream_test"] = abs(r["rtx5070ti_h100_profile"]["loss"] - EXPECTED["evo2_1b_base"]["loss"]) < 1e-3
    r["resident_equal"] = T.get("resident_equal")
    res["evo2_1b_base"]["upstream_test"] = r
    # per window: against the H100 (scalars only) and the RTX 4060 official path (every position)
    win = []
    for x in rows:
        w = x["window"]
        row = {"window": w, "length": x["length"], "loss": x["loss"], "accuracy": x["accuracy"],
               "loss_fp32": x["loss_fp32"], "mean_next_logprob": x["mean_next_logprob"]}
        if w in hw:
            row.update(h100_loss_fp32=hw[w]["loss_fp32"], h100_accuracy=hw[w]["accuracy"],
                       h100_mean_next_logprob=hw[w]["mean_next_logprob"],
                       logits_equal_h100=x["logits_sha256"] == hw[w]["logits_sha256"])
        if f"{w}/acgt" in A.files:
            assert np.array_equal(Z[f"{MODE}/{w}/next"].shape, A[f"{w}/next"].shape)
            row["vs_rtx4060_official"] = window_stats(Z[f"{MODE}/{w}/acgt"], Z[f"{MODE}/{w}/next"],
                                                      A[f"{w}/acgt"], A[f"{w}/next"])
        win.append(row)
    res["evo2_1b_base"]["windows"] = win
    six = [x["vs_rtx4060_official"]["mean_abs_dlogp_acgt"] for x in win if "vs_rtx4060_official" in x]
    agree = [x["vs_rtx4060_official"]["acgt_argmax_agree"] for x in win if "vs_rtx4060_official" in x]
    old = load_json(TI14 / "results.json")["windows"]
    old_m = [x for x in old if x["mode"] == "mmaemu"]
    cmp14 = load_json(H14 / "comparison_vs_ada.json")
    h_six = [v["acgt_dlogp_mean"] for v in cmp14["windows"].values()]
    res["evo2_1b_base"]["six_windows_vs_rtx4060_official"] = {
        "rtx5070ti_h100_profile": {"mean": float(np.mean(six)), "min": float(min(six)), "max": float(max(six)),
                                   "argmax_agree_mean": float(np.mean(agree)), "windows": len(six)},
        "rtx5070ti_rtx4060_profile": {"mean": float(np.mean([x["mean_abs_dlogp_acgt"] for x in old_m])),
                                      "min": float(min(x["mean_abs_dlogp_acgt"] for x in old_m)),
                                      "max": float(max(x["mean_abs_dlogp_acgt"] for x in old_m)),
                                      "windows": len(old_m)},
        "h100_official": {"mean": float(np.mean(h_six)), "min": float(min(h_six)), "max": float(max(h_six)),
                          "windows": len(h_six)}}

if (RES / "1b_brca1.json").is_file():
    B = np.load(RES / "1b_brca1.npz")
    rs, vs, ri, lof = B[f"{MODE}/rs"], B[f"{MODE}/vs"], B["ref_index"], B["lof"]
    H = np.load(H14 / "brca1_scores.npz")
    M = np.load(TI14 / "brca1_mmaemu.npz")
    assert np.array_equal(ri, H["ref_index"]) and np.array_equal(ri, M["ref_index"])
    assert np.array_equal(lof, A["brca1/lof"])
    br = {"rtx5070ti_h100_profile": brca1_run(rs, vs, ri, lof),
          "h100_official": brca1_run(H["rs"], H["vs"], ri, lof),
          "rtx4060_official": brca1_run(A["brca1/rs"], A["brca1/vs"], ri, lof),
          "rtx5070ti_rtx4060_profile": brca1_run(M["rs"], M["vs"], ri, lof),
          "minutes": load_json(RES / "1b_brca1.json")["modes"][MODE]["minutes"]}
    br["rtx5070ti_h100_profile_vs"] = {
        "h100_official": brca1_vs(rs, vs, ri, H["rs"], H["vs"]),
        "rtx4060_official": brca1_vs(rs, vs, ri, A["brca1/rs"], A["brca1/vs"]),
        "rtx5070ti_rtx4060_profile": brca1_vs(rs, vs, ri, M["rs"], M["vs"])}
    br["for_comparison"] = {
        "rtx5070ti_rtx4060_profile_vs_h100_official": brca1_vs(M["rs"], M["vs"], ri, H["rs"], H["vs"]),
        "rtx5070ti_rtx4060_profile_vs_rtx4060_official": brca1_vs(M["rs"], M["vs"], ri, A["brca1/rs"], A["brca1/vs"]),
        "h100_official_vs_rtx4060_official": brca1_vs(H["rs"], H["vs"], ri, A["brca1/rs"], A["brca1/vs"])}
    res["evo2_1b_base"]["brca1"] = br

# ---- Evo 2 40B ----
H40Z = np.load(H40 / "h100_official_te_40b.npz")
if (RES / "40b_test.json").is_file():
    T, Z = load_json(RES / "40b_test.json"), np.load(RES / "40b_test.npz")
    rows = [x for x in T["rows"] if x["mode"] == MODE]
    Ht = load_json(H40 / "results" / "40b" / "test.json")
    old = load_json(TI40 / "test_40b.json")
    r = {"expected": EXPECTED["evo2_40b"], "rtx5070ti_h100_profile": test_mean(rows),
         "h100_official": Ht["test_evo2_mean"], "rtx5070ti_rtx4060_profile": old["test_evo2_mean"]["mmaemu"]}
    r["passes_upstream_test"] = abs(r["rtx5070ti_h100_profile"]["loss"] - EXPECTED["evo2_40b"]["loss"]) < 1e-3
    hseq = {s["name"]: s for s in Ht["sequences"]}
    r["per_prompt"] = [{"prompt": x["window"], "loss": x["loss"], "accuracy": x["accuracy"],
                        "loss_fp32": x["loss_fp32"], "h100_loss": hseq[x["window"]]["loss"],
                        "h100_accuracy": hseq[x["window"]]["accuracy"],
                        "logits_equal_h100": x["logits_sha256"] == hseq[x["window"]]["logits_sha256"]}
                       for x in rows if x["window"] in TESTS]
    res["evo2_40b"]["upstream_test"] = r
    G = np.load(TI40 / "rtx4060_official_te_40b.npz")
    O = np.load(TI40 / "test_40b.npz")
    win = []
    for x in rows:
        w = x["window"]
        assert np.array_equal(Z[f"{MODE}/{w}/next"].shape, H40Z[f"{w}/next"].shape)
        row = {"window": w, "length": x["length"],
               "vs_h100_official": window_stats(Z[f"{MODE}/{w}/acgt"], Z[f"{MODE}/{w}/next"],
                                                H40Z[f"{w}/acgt"], H40Z[f"{w}/next"])}
        if f"{w}/acgt" in G.files:
            row["vs_rtx4060_official"] = window_stats(Z[f"{MODE}/{w}/acgt"], Z[f"{MODE}/{w}/next"],
                                                      G[f"{w}/acgt"], G[f"{w}/next"])
            row["for_comparison_h100_official_vs_rtx4060_official"] = window_stats(
                H40Z[f"{w}/acgt"], H40Z[f"{w}/next"], G[f"{w}/acgt"], G[f"{w}/next"])
        if f"mmaemu/{w}/acgt" in O.files:
            row["vs_rtx5070ti_rtx4060_profile"] = window_stats(Z[f"{MODE}/{w}/acgt"], Z[f"{MODE}/{w}/next"],
                                                               O[f"mmaemu/{w}/acgt"], O[f"mmaemu/{w}/next"])
            row["for_comparison_rtx4060_profile_vs_h100_official"] = window_stats(
                O[f"mmaemu/{w}/acgt"], O[f"mmaemu/{w}/next"], H40Z[f"{w}/acgt"], H40Z[f"{w}/next"])
        win.append(row)
    res["evo2_40b"]["windows"] = win

if (RES / "40b_brca1.json").is_file():
    B = np.load(RES / "40b_brca1.npz")
    rs, vs, ri, lof = B[f"{MODE}/rs"], B[f"{MODE}/vs"], B["ref_index"], B["lof"]
    n, nr = len(lof), len(rs)
    HB = np.load(H40 / "results" / "40b" / "brca1.npz")
    L = np.load(TI40 / "brca1_40b.npz")
    assert np.array_equal(ri, HB["ref_index"][:n]) and np.array_equal(lof, HB["lof"][:n])
    assert np.array_equal(ri, L["ref_index"]) and np.array_equal(lof, L["lof"])
    hrs, hvs = HB["rs"][:nr], HB["vs"][:n]
    br = {"rtx5070ti_h100_profile": brca1_run(rs, vs, ri, lof),
          "h100_official_same_variants": brca1_run(hrs, hvs, ri, lof),
          "rtx5070ti_rtx4060_profile": brca1_run(L["mmaemu/rs"], L["mmaemu/vs"], ri, lof),
          "minutes": load_json(RES / "40b_brca1.json")["modes"][MODE]["minutes"]}
    br["rtx5070ti_h100_profile_vs"] = {
        "h100_official": brca1_vs(rs, vs, ri, hrs, hvs),
        "rtx5070ti_rtx4060_profile": brca1_vs(rs, vs, ri, L["mmaemu/rs"], L["mmaemu/vs"])}
    br["for_comparison"] = {
        "rtx5070ti_rtx4060_profile_vs_h100_official": brca1_vs(L["mmaemu/rs"], L["mmaemu/vs"], ri, hrs, hvs)}
    res["evo2_40b"]["brca1"] = br

a.out.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res, indent=1)[:12000])
