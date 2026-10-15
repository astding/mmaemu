#!/usr/bin/env python3
"""Draw the figures of README.md (docs/images/*.png) from committed logs only.

  python tools/make_figures.py            # needs matplotlib, numpy and scikit-learn (not dependencies of mmaemu)

Inputs:
  docs/validation_logs/2026-10-14/evo2-rtx-5070-ti/   results.json and brca1_<mode>.npz of examples/evo2_blackwell.ipynb
  docs/validation_logs/2026-10-14/evo2-h100-sm90/     the official-path baseline on an H100 (comparison_vs_ada.json)
  examples/data/evo2_1b_ada_rtx4060_te.npz            the official-path reference on the RTX 4060
  docs/validation_logs/2026-10-13-sm120/performance-*.json   kernel throughput on the RTX 5070 Ti
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "docs" / "validation_logs"
WINDOWS = ["p0", "p1", "p2", "p3", "w8k", "w2k"]
PROMPTS = ["p0", "p1", "p2", "p3"]
COLORS = {"RTX 4060, official TE": "#222222", "H100, official TE": "#577590", "mmaemu": "#2a9d8f",
          "native": "#e76f51", "bf16": "#f4a261"}
LABELS = {"RTX 4060, official TE": "RTX 4060 (Ada), official TE FP8 — reference",
          "H100, official TE": "H100 (Hopper), official TE FP8",
          "mmaemu": "RTX 5070 Ti + mmaemu (RTX 4060 FP8 GEMM emulated)",
          "native": "RTX 5070 Ti, native FP8 (torch._scaled_mm)",
          "bf16": "RTX 5070 Ti, bf16 projections (no FP8)"}
SHORT = {"RTX 4060, official TE": "RTX 4060\nofficial", "H100, official TE": "H100\nofficial",
         "mmaemu": "5070 Ti\n+ mmaemu", "native": "5070 Ti\nnative FP8", "bf16": "5070 Ti\nbf16"}


def style():
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
                         "legend.frameon": False})


def load(evo2_dir: Path, h100_dir: Path, ref_npz: Path):
    res = json.loads((evo2_dir / "results.json").read_text())
    h100 = json.loads((h100_dir / "comparison_vs_ada.json").read_text())
    ref = np.load(ref_npz)
    win = {(r["mode"], r["window"]): r for r in res["windows"]}
    for w in WINDOWS:  # the H100 rows of results.json carry the same numbers; take them from the H100 log itself
        h = h100["windows"][w]
        win[("H100, official TE", w)] = dict(mean_abs_dlogp_acgt=h["acgt_dlogp_mean"],
                                             acgt_argmax_agree=h["acgt_argmax_agree"],
                                             mean_next_logprob=h["mean_next_logprob"])
        win[("RTX 4060, official TE", w)] = dict(mean_next_logprob=float(ref[f"{w}/next"].mean()))
    return res, h100, ref, win


def fig_upstream_test(win, out: Path):
    """The loss of the upstream test (evo2/test/test_evo2.py): next-token cross entropy on the prompts p0..p3, here
    in fp32 from the bf16 logits (the mean of mean_next_logprob, as upstream_test.py of the H100 log computes it)."""
    modes = ["RTX 4060, official TE", "H100, official TE", "mmaemu", "bf16", "native"]
    nll = [-np.mean([win[(m, w)]["mean_next_logprob"] for w in PROMPTS]) for m in modes]
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    y = np.arange(len(modes))[::-1]
    ax.barh(y, nll, color=[COLORS[m] for m in modes], height=0.62)
    for yi, v in zip(y, nll):
        ax.text(v + 0.015, yi, f"{v:.4f}", va="center", fontsize=9)
    ax.axvline(0.502, color="#2a9d8f", ls="--", lw=1)
    ax.text(0.502, len(modes) - 0.35, " expected by the upstream test: 0.502", color="#2a9d8f", fontsize=8.5)
    for v in (1.248, 1.365):  # issue #157 (RTX 5070 Ti): TE FP8, FP8 disabled
        ax.axvline(v, color="#999999", ls=":", lw=1)
    ax.text(1.24, -0.85, "issue #157:\nTE FP8 1.248", ha="right", fontsize=7.5, color="#666666")
    ax.text(1.375, -0.85, "FP8 off\n1.365", ha="left", fontsize=7.5, color="#666666")
    ax.set_yticks(y, [LABELS[m] for m in modes], fontsize=8.5)
    ax.set_ylim(-1.1, len(modes) - 0.2)
    ax.set_xlim(0, 1.6)
    ax.set_xlabel("next-token cross entropy on prompts p0–p3 (lower is better)")
    ax.set_title("Evo 2 1B on the upstream test prompts", loc="left", fontsize=11)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_windows(win, out: Path):
    modes = ["H100, official TE", "mmaemu", "bf16", "native"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    x = np.arange(len(WINDOWS))
    width = 0.2
    for k, m in enumerate(modes):
        dl = [win[(m, w)]["mean_abs_dlogp_acgt"] for w in WINDOWS]
        ag = [win[(m, w)]["acgt_argmax_agree"] for w in WINDOWS]
        axes[0].bar(x + (k - 1.5) * width, dl, width, color=COLORS[m], label=LABELS[m])
        axes[1].bar(x + (k - 1.5) * width, ag, width, color=COLORS[m], label=LABELS[m])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("mean |Δ log p(A,C,G,T)| vs RTX 4060")
    axes[0].set_title("Distance from the official RTX 4060 output (log scale)", loc="left", fontsize=10.5)
    axes[1].set_ylabel("fraction of positions")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Most likely base agrees with the RTX 4060", loc="left", fontsize=10.5)
    for ax in axes:
        ax.set_xticks(x, WINDOWS)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def fig_brca1(res, evo2_dir: Path, h100_dir: Path, ref, out: Path):
    from sklearn.metrics import roc_auc_score

    n = res["brca1"]["n_variants"]
    lof = ref["brca1/lof"][:n].astype(bool)
    deltas = {}
    for m in ("mmaemu", "native", "bf16"):
        s = np.load(evo2_dir / f"brca1_{m}.npz")
        deltas[m] = s["vs"] - s["rs"][s["ref_index"]]
        ri = s["ref_index"]
    rs, vs = ref["brca1/rs"][:len(np.unique(ri))], ref["brca1/vs"][:n]
    deltas["RTX 4060, official TE"] = vs - rs[ri]
    if n == 500:
        s = np.load(h100_dir / "brca1_scores.npz")
        deltas["H100, official TE"] = s["vs"] - s["rs"][s["ref_index"]]
    modes = [m for m in ("RTX 4060, official TE", "H100, official TE", "mmaemu", "bf16", "native") if m in deltas]
    auc = {m: roc_auc_score(lof, -deltas[m]) for m in modes}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    ax = axes[0]
    ax.bar(range(len(modes)), [auc[m] for m in modes], color=[COLORS[m] for m in modes], width=0.65)
    for i, m in enumerate(modes):
        ax.text(i, auc[m] + 0.005, f"{auc[m]:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(modes)), [SHORT[m] for m in modes], fontsize=8)
    ax.axhline(0.5, color="#888888", ls=":", lw=0.8)
    ax.set_ylim(0.4, max(auc.values()) + 0.04)
    ax.set_ylabel("AUROC, LOF vs FUNC/INT (0.5: chance)")
    ax.set_title(f"Zero-shot BRCA1 ({n} variants)", loc="left", fontsize=10.5)
    ax.grid(axis="x", visible=False)
    base = deltas["RTX 4060, official TE"]
    lim = np.abs(base).max() * 1.1
    for ax, m in zip(axes[1:], ("mmaemu", "native")):
        r = np.corrcoef(base, deltas[m])[0, 1]
        ax.scatter(base, deltas[m], s=7, c=np.where(lof, "#c1121f", "#457b9d"), alpha=0.65, lw=0)
        ax.plot([-lim, lim], [-lim, lim], color="#888888", lw=0.8)
        ax.set_xlabel("Δ score, RTX 4060 official TE")
        ax.set_ylabel(f"Δ score, {SHORT[m].replace(chr(10), ' ')}")
        ax.set_title(f"{SHORT[m].replace(chr(10), ' ')} vs reference: Pearson {r:.3f}", loc="left", fontsize=10)
        if m == "mmaemu":
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
    axes[2].scatter([], [], c="#c1121f", s=12, label="loss of function")
    axes[2].scatter([], [], c="#457b9d", s=12, label="functional / intermediate")
    axes[2].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_throughput(out: Path):
    shapes = [("1b", "1B\nK 1920, N 5760"), ("7b", "7B\nK 4096, N 12288"), ("40b", "40B\nK 8192, N 24576")]
    kernels = ["v1", "v4", "v9"]
    colors = {"v1": "#8ecae6", "v4": "#219ebc", "v9": "#023047"}
    vals = {}
    for s, _ in shapes:
        d = json.loads((LOGS / "2026-10-13-sm120" / f"performance-{s}.json").read_text())
        for r in d["results"]:
            vals[(s, r["kernel"])] = r["tmac_s_median"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(shapes))
    for k, kern in enumerate(kernels):
        v = [vals[(s, kern)] for s, _ in shapes]
        ax.bar(x + (k - 1) * 0.26, v, 0.26, color=colors[kern], label=f"cuda:{kern}" + (" (gated)" if kern == "v9" else ""))
        for xi, vi in zip(x, v):
            ax.text(xi + (k - 1) * 0.26, vi + 0.06, f"{vi:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, [label for _, label in shapes], fontsize=8.5)
    ax.set_ylabel("TMAC/s (CUDA events, sustained)")
    ax.set_title("Emulated FP8 GEMM throughput on the RTX 5070 Ti, M = 2048", loc="left", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_ylim(0, max(vals.values()) * 1.2)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evo2-dir", type=Path, default=LOGS / "2026-10-14" / "evo2-rtx-5070-ti")
    ap.add_argument("--h100-dir", type=Path, default=LOGS / "2026-10-14" / "evo2-h100-sm90")
    ap.add_argument("--reference", type=Path, default=ROOT / "examples" / "data" / "evo2_1b_ada_rtx4060_te.npz")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "images")
    args = ap.parse_args(argv)
    style()
    args.out.mkdir(parents=True, exist_ok=True)
    res, h100, ref, win = load(args.evo2_dir, args.h100_dir, args.reference)
    fig_upstream_test(win, args.out / "evo2_upstream_test.png")
    fig_windows(win, args.out / "evo2_windows.png")
    fig_brca1(res, args.evo2_dir, args.h100_dir, ref, args.out / "evo2_brca1.png")
    fig_throughput(args.out / "kernel_throughput.png")
    for p in sorted(args.out.glob("*.png")):
        print(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
