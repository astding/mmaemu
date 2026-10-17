"""Zero-shot BRCA1 variant scoring (examples/evo2_blackwell.ipynb, section 3) with the weights in host RAM.

The same data, the same random subset of 500 variants (seed 0) and the same 8,192-bp windows as the notebook; a
prefix of N variants uses a prefix of its reference windows. Each window is scored as evo2.scoring.score_sequences
does with batch size 1 (log-softmax of the bf16 logits, log-probability of every next token, mean). The model runs
one block at a time on the GPU (run_offload.py); the activations of all windows wait in page-locked host RAM
between blocks.
The long Hyena filter of a block depends only on the window length, so it is computed once per block and length.
"""

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from run_offload import load_model, patch_projections, read_te_scales, set_mode

HERE = Path(__file__).resolve().parent
DATA = HERE / "brca1"
ADA_1B = Path("<src>/examples/data/evo2_1b_ada_rtx4060_te.npz")
WINDOW = 8192


def plain(o):
    """json default: numpy scalars as Python numbers."""
    return o.item() if hasattr(o, "item") else str(o)


def subset(n):
    """The notebook's variants, windows and labels for the first n variants of its 500."""
    A = np.load(ADA_1B)
    df = pd.read_excel(DATA / "41586_2018_461_MOESM3_ESM.xlsx", header=2)
    df = df[["chromosome", "position (hg19)", "reference", "alt", "function.score.mean", "func.class"]]
    df.columns = ["chrom", "pos", "ref", "alt", "score", "class"]
    df["class"] = df["class"].replace(["FUNC", "INT"], "FUNC/INT")
    df = df.sample(n=500, random_state=0).reset_index(drop=True).iloc[:n]
    assert (df.pos.values == A["brca1/pos"][:n]).all() and (df.alt.values == A["brca1/alt"][:n]).all()
    with gzip.open(DATA / "GRCh37.p13_chr17.fna.gz", "rt") as h:
        chr17 = str(next(SeqIO.parse(h, "fasta")).seq)
    ref_seqs, ref_of, var_seqs, ref_index = [], {}, [], []
    for row in df.itertuples():
        p = row.pos - 1
        start, end = max(0, p - WINDOW // 2), min(len(chr17), p + WINDOW // 2)
        r = chr17[start:end]
        k = min(WINDOW // 2, p)
        assert r[k] == row.ref
        v = r[:k] + row.alt + r[k + 1:]
        ref_of.setdefault(r, len(ref_seqs))
        if ref_of[r] == len(ref_seqs):
            ref_seqs.append(r)
        ref_index.append(ref_of[r])
        var_seqs.append(v)
    lof = (df["class"] == "LOF").values
    ada = dict(rs=A["brca1/rs"][:len(ref_seqs)], vs=A["brca1/vs"][:n])
    return ref_seqs, var_seqs, np.array(ref_index), lof, ada


class FilterCache:
    """Memoise compute_filter(L, device) of the long-filter (HCL) blocks while a block is on the GPU."""

    def __init__(self, block):
        self.f = getattr(block, "filter", None)
        self.on = self.f is not None and getattr(self.f, "h", 1) is None and hasattr(self.f, "compute_filter")
        if self.on:
            orig, memo = self.f.compute_filter, {}

            def cached(L, device):
                key = (L, str(device))
                if key not in memo:
                    memo[key] = orig(L, device)
                return memo[key]

            self.f.compute_filter = cached

    def close(self):
        if self.on:
            del self.f.compute_filter                   # back to the class's method


def score_offload(model, scales, mode, seqs, dev, log):
    """Scores of `seqs` (all of one length). Between blocks the activations wait in one page-locked host buffer, and
    each block's weights go to the GPU through a page-locked staging buffer: on this WSL2 host, copies between the
    GPU and freshly allocated pageable memory were the bottleneck when free memory ran low."""
    from evo2.scoring import logits_to_logprobs
    from vortex.model.tokenizer import CharLevelTokenizer

    tok = CharLevelTokenizer(512)
    ids = [torch.tensor(tok.tokenize(s), dtype=torch.long) for s in seqs]
    L = len(ids[0])
    assert all(len(i) == L for i in ids), "windows of different lengths"
    set_mode(model, scales, mode)
    model.embedding_layer.to(dev)
    model.norm.to(dev)
    tensors = lambda m: list(m.parameters()) + list(m.buffers())
    staging = torch.empty(max(sum(t.numel() * t.element_size() for t in tensors(b)) for b in model.blocks),
                          dtype=torch.uint8, pin_memory=True)
    store = torch.empty((len(ids), L, model.config.hidden_size), dtype=torch.bfloat16, pin_memory=True)
    t_copy = t_comp = 0.0
    with torch.inference_mode():
        for k, i in enumerate(ids):
            x = model.embedding_layer(i[None].to(dev))
            assert x.dtype == store.dtype and x.shape == (1, *store.shape[1:])
            store[k].copy_(x[0])
        for b, block in enumerate(model.blocks):
            t0 = time.time()
            held = []
            for t in tensors(block):
                d = t.data
                n = d.numel() * d.element_size()
                stage = staging[:n].view(d.dtype).view(d.shape)
                stage.copy_(d)
                g = torch.empty_like(d, device=dev)
                g.copy_(stage)                          # synchronous from page-locked memory
                held.append((t, d))
                t.data = g
            cache = FilterCache(block)
            torch.cuda.synchronize()
            t1 = time.time()
            for k in range(len(ids)):
                y, _ = block(store[k][None].to(dev), inference_params=None, padding_mask=None)
                assert y.dtype == store.dtype and y.shape == (1, *store.shape[1:])
                store[k].copy_(y[0])
            torch.cuda.synchronize()
            cache.close()
            for t, d in held:
                t.data = d
            del held, cache
            torch.cuda.empty_cache()
            t_copy, t_comp = t_copy + t1 - t0, t_comp + time.time() - t1
            if b % 5 == 4 or b == len(model.blocks) - 1:
                log(f"{mode}: block {b + 1}/{len(model.blocks)}, weights copy {t_copy:.0f} s, "
                    f"compute {t_comp:.0f} s, GPU peak {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB")
        scores = []
        for k in range(len(ids)):
            logits = model.unembed(model.norm(store[k][None].to(dev)))
            lp = logits_to_logprobs(logits, ids[k][None].to(dev)).float().cpu().numpy()
            scores.append(np.mean(lp[0][:len(ids[k])]))
    return np.array(scores), dict(copy_s=t_copy, compute_s=t_comp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--variants", type=int, default=100)
    ap.add_argument("--modes", default="mmaemu,native,bf16")
    ap.add_argument("--check-resident", action="store_true",
                    help="also score with evo2.scoring.score_sequences on the whole model on the GPU, compare")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    import mmaemu

    dev = torch.device("cuda:0")
    ref_seqs, var_seqs, ref_index, lof, ada = subset(args.variants)
    seqs = ref_seqs + var_seqs
    log(f"mmaemu {mmaemu.__version__} ({mmaemu.__file__}); {torch.cuda.get_device_name(dev)}; {args.variants} "
        f"variants ({lof.sum()} LOF), {len(ref_seqs)} reference + {len(var_seqs)} variant windows")
    t0 = time.time()
    model = load_model(args.model, args.checkpoint)
    scales = read_te_scales(args.checkpoint)
    patch_projections(model, scales)
    log(f"loaded {args.model} in {time.time() - t0:.0f} s; FP8 projections in {len(scales)} blocks")

    result = dict(model=args.model, variants=args.variants, lof=int(lof.sum()), windows=len(seqs),
                  device=torch.cuda.get_device_name(dev), mmaemu=mmaemu.__version__, modes={})
    ada_delta = ada["vs"] - ada["rs"][ref_index]
    result["evo2_1b_rtx4060_official_te_auroc"] = roc_auc_score(lof, -ada_delta)
    arrays = dict(ref_index=ref_index, lof=lof)
    for mode in args.modes.split(","):
        t0 = time.time()
        s, timing = score_offload(model, scales, mode, seqs, dev, log)
        rs, vs = s[:len(ref_seqs)], s[len(ref_seqs):]
        delta = vs - rs[ref_index]
        row = dict(auroc=roc_auc_score(lof, -delta), minutes=(time.time() - t0) / 60, **timing)
        if args.model == "evo2_1b_base":
            row.update(delta_pearson_vs_rtx4060=pearsonr(delta, ada_delta)[0],
                       delta_spearman_vs_rtx4060=spearmanr(delta, ada_delta)[0])
        if args.check_resident:
            from evo2.scoring import score_sequences
            from vortex.model.tokenizer import CharLevelTokenizer

            model.to(dev)
            model.block_idx_to_device = {i: str(dev) for i in model.block_idx_to_device}
            with torch.inference_mode():
                r = np.array(score_sequences(seqs, model, CharLevelTokenizer(512), batch_size=1))
            model.to("cpu")
            row["resident_bit_identical"] = bool(np.array_equal(r, s))
        result["modes"][mode] = row
        arrays[f"{mode}/rs"], arrays[f"{mode}/vs"] = rs, vs
        log(f"{mode}: AUROC {row['auroc']:.4f} in {row['minutes']:.1f} min "
            f"{json.dumps({k: v for k, v in row.items() if k not in ('auroc', 'minutes')}, default=plain)}")
        with open(args.out, "w") as f:                  # after every mode, so a stopped run keeps its results
            json.dump(result, f, indent=1, default=plain)
        np.savez(args.out.removesuffix(".json") + ".npz", **arrays)
    log(f"1B on the RTX 4060 (official TE), same variants: AUROC {result['evo2_1b_rtx4060_official_te_auroc']:.4f}")
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
