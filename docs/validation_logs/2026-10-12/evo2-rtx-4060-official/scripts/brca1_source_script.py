"""Held-out check of the FP8 bypass on a real task: BRCA1 zero-shot VEP (upstream notebook logic, 8,192-bp windows).
Same windows scored twice with evo2_1b_base on the RTX 4060 (Ada):
  (a) TE FP8 (reference; the upstream notebook reports AUROC 0.73 on the full set with this checkpoint)
  (b) FP8-free: block-0 input projection = per-token lookup table built from TE outputs, all other projections plain bf16
Reports delta-score correlation and AUROC for both on a random variant subset (seed 0)."""
import gzip, json, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from Bio import SeqIO
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

torch.cuda.set_per_process_memory_fraction(0.95)
from evo2 import Evo2
from vortex.model.layers import TELinear

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
NB = "/data/quant/stsc/research/round4/probe/evo2_src/notebooks/brca1"
OUT = "/data/quant/stsc/research/round4/probe/logs/fp8_emulation"
WINDOW = 8192

df = pd.read_excel(os.path.join(NB, "41586_2018_461_MOESM3_ESM.xlsx"), header=2)
df = df[["chromosome", "position (hg19)", "reference", "alt", "function.score.mean", "func.class"]]
df.columns = ["chrom", "pos", "ref", "alt", "score", "class"]
df["class"] = df["class"].replace(["FUNC", "INT"], "FUNC/INT")
full_n, full_lof = len(df), int((df["class"] == "LOF").sum())
df = df.sample(n=min(N, len(df)), random_state=0).reset_index(drop=True)
with gzip.open(os.path.join(NB, "GRCh37.p13_chr17.fna.gz"), "rt") as h:
    chr17 = str(next(SeqIO.parse(h, "fasta")).seq)


def parse(pos, ref, alt):
    p = pos - 1
    s, e = max(0, p - WINDOW // 2), min(len(chr17), p + WINDOW // 2)
    r = chr17[s:e]
    k = min(WINDOW // 2, p)
    v = r[:k] + alt + r[k + 1:]
    assert r[k] == ref and v[k] == alt
    return r, v


ref_seqs, idx, var_seqs, ref_index = [], {}, [], []
for _, row in df.iterrows():
    r, v = parse(row["pos"], row["ref"], row["alt"])
    if r not in idx:
        idx[r] = len(ref_seqs); ref_seqs.append(r)
    ref_index.append(idx[r]); var_seqs.append(v)
ref_index = np.array(ref_index)
print(dict(full_variants=full_n, full_lof=full_lof, subset=len(df), subset_lof=int((df["class"] == "LOF").sum()),
           unique_ref_windows=len(ref_seqs)), flush=True)

m = Evo2("evo2_1b_base")
mods = [x for x in m.model.modules() if isinstance(x, TELinear) and getattr(x, "use_fp8_input_projections", False)]


def run(tag):
    t = time.perf_counter()
    rs = np.array(m.score_sequences(ref_seqs)); vs = np.array(m.score_sequences(var_seqs))
    d = vs - rs[ref_index]
    print(f"{tag}: {len(ref_seqs) + len(var_seqs)} windows in {time.perf_counter() - t:.0f} s", flush=True)
    return rs, vs, d


_te_cache = f"{OUT}/brca1_subset{N}_te_scores.npz"
if os.path.exists(_te_cache):
    _z = np.load(_te_cache); rs_te, vs_te = _z["rs"], _z["vs"]; d_te = vs_te - rs_te[ref_index]
    print("TE_FP8: loaded cached scores", _te_cache, flush=True)
else:
    rs_te, vs_te, d_te = run("TE_FP8")
    np.savez(_te_cache, rs=rs_te, vs=vs_te)

# build the block-0 lookup table with TE FP8 (Ada), then switch everything to FP8-free
blk = [b for b in m.model.blocks if getattr(b, "projections", None) is mods[0]][0]
V = m.model.embedding_layer.weight.shape[0]
with torch.inference_mode():
    xin = blk.pad_to_multiple(blk.pre_norm(m.model.embedding_layer(torch.arange(V, device="cuda:0").unsqueeze(0))))
    o = mods[0](xin)
    lut = (o[0] if isinstance(o, tuple) else o)[0, :V].detach().clone()
for mm in mods[1:]:
    mm.use_fp8_input_projections = False
state = {}
_emb = m.model.embedding_layer.forward
def emb_fwd(ids, *a, **k):
    state["ids"] = ids
    return _emb(ids, *a, **k)
m.model.embedding_layer.forward = emb_fwd
def lut_fwd(x, mod=mods[0]):
    ids = state["ids"]; L = ids.shape[1]
    out = lut[ids.long()]
    if x.shape[1] > L:
        out = torch.cat([out, F.linear(x[:, L:], mod.weight, None).to(out.dtype)], dim=1)
    return (out, None)
mods[0].forward = lut_fwd
mods[0].use_fp8_input_projections = False

rs_lut, vs_lut, d_lut = run("LUT_bf16")
np.savez(f"{OUT}/brca1_subset{N}_lut_scores.npz", rs=rs_lut, vs=vs_lut)

y = (df["class"] == "LOF").values
res = dict(
    subset=len(df), subset_lof=int(y.sum()), windows=len(ref_seqs) + len(var_seqs),
    auroc_te=round(float(roc_auc_score(y, -d_te)), 4), auroc_lut=round(float(roc_auc_score(y, -d_lut)), 4),
    delta_pearson=round(float(pearsonr(d_te, d_lut)[0]), 5), delta_spearman=round(float(spearmanr(d_te, d_lut)[0]), 5),
    delta_sign_agree=round(float(np.mean(np.sign(d_te) == np.sign(d_lut))), 4),
    seqscore_pearson=round(float(pearsonr(np.r_[rs_te, vs_te], np.r_[rs_lut, vs_lut])[0]), 6),
    seqscore_max_abs_diff=round(float(np.max(np.abs(np.r_[rs_te, vs_te] - np.r_[rs_lut, vs_lut]))), 6),
    delta_max_abs_diff=round(float(np.max(np.abs(d_te - d_lut))), 7), delta_abs_mean_te=round(float(np.mean(np.abs(d_te))), 7),
)
print(res, flush=True)
df["delta_te"], df["delta_lut"] = d_te, d_lut
df.to_csv(f"{OUT}/brca1_subset{N}_deltas.csv", index=False)
json.dump(res, open(f"{OUT}/brca1_subset{N}_result.json", "w"), indent=1)
