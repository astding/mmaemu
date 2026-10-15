"""The official Evo 2 path (Transformer Engine FP8) for a model that does not fit on the GPU, run block by block.

As golden_oracle.py (docs/validation_logs/2026-10-12/evo2-rtx-4060-official/scripts/), which on the 1B gives the
logits of the real upstream model bit for bit: each block is built by vortex's get_block and loaded by vortex's own
StripedHyena.to_bfloat16_except_pr_lc + custom_load_state_dict through a holder module that has only that block; no
other code touches the weights. The sequences run in chunks whose hidden states fit on the GPU (each chunk streams
every block once). In inference mode Transformer Engine does not update its FP8 scales (fp8_autocast_exit updates them
only with grad enabled), so every sequence uses the checkpoint's scales, whatever the chunking or the order.

usage: python official_blockwise.py --model evo2_40b --checkpoint evo2_40b.pt --inputs inputs.npz --out DIR
           [--groups test,brca1] [--brca1-variants N] [--first-chunk-variants M] [--chunk-tokens T]
  test:  every sequence of inputs.npz's test/: the metrics of evo2/test/test_evo2.py (cross entropy of the bf16 logits
         on the GPU, accuracy), the fp32 loss, the SHA-256 of the bf16 logits, the full logits bit patterns and the
         log-probabilities of A, C, G, T and of the next token (benchmarks' summarize()) -> DIR/test.json, DIR/test.npz
  brca1: the windows of the first N BRCA1 variants of inputs.npz (unique reference windows and variant windows),
         scored as evo2.scoring.score_sequences with batch size 1 (mean log-probability of the next tokens). The
         first chunk holds the windows of the first M variants, so their AUROC is saved first; the results are saved
         after every chunk -> DIR/brca1.json, DIR/brca1.npz"""
import argparse
import hashlib
import io
import json
import logging
import os
import pkgutil
import platform
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--inputs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--groups", default="test,brca1")
ap.add_argument("--brca1-variants", type=int, default=500, help="score the windows of the first N variants")
ap.add_argument("--first-chunk-variants", type=int, default=100, help="the first chunk: the first M variants")
ap.add_argument("--chunk-tokens", type=int, default=250 * 8192, help="tokens whose hidden states share the GPU")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.cuda.set_per_process_memory_fraction(0.95)

import transformer_engine  # noqa: E402,F401  (the official path needs TE)
from evo2.scoring import logits_to_logprobs  # noqa: E402
from evo2.utils import CONFIG_MAP  # noqa: E402
from transformer_engine.common.recipe import DelayedScaling, Format, _FormatHelper  # noqa: E402
from vortex.model.layers import RMSNorm, VocabParallelEmbedding  # noqa: E402
from vortex.model.model import StripedHyena, get_block  # noqa: E402
from vortex.model.utils import Lambda, dotdict  # noqa: E402

cfg = dotdict(yaml.safe_load(pkgutil.get_data("evo2", CONFIG_MAP[a.model])))
assert cfg.get("use_fp8_input_projections", False), "this model has no FP8 projections"
torch.serialization.add_safe_globals([io.BytesIO, DelayedScaling, Format, _FormatHelper])
state = torch.load(a.checkpoint, map_location="cpu", mmap=True, weights_only=True)   # as vortex load_checkpoint
log = logging.getLogger("official_blockwise")
dev = "cuda:0"
ACGT = torch.tensor([ord(c) for c in "ACGT"], dtype=torch.long)    # CharLevelTokenizer(512): the byte value


def say(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def holder(**mods):
    h = StripedHyena.__new__(StripedHyena)
    nn.Module.__init__(h)
    h.config, h.logger = cfg, log
    for k, v in mods.items():
        setattr(h, k, v)
    if "blocks" not in mods:
        h.blocks = nn.ModuleList()
    return h


def load_upstream(h, prefix_filter):
    StripedHyena.to_bfloat16_except_pr_lc(h, to_float32=True)                 # load_checkpoint order: dtype first
    StripedHyena.custom_load_state_dict(h, {k: v for k, v in state.items() if prefix_filter(k)})


with torch.device(dev), torch.cuda.device(dev):
    emb = VocabParallelEmbedding(cfg)
    norm = RMSNorm(cfg)
load_upstream(holder(embedding_layer=emb, norm=norm), lambda k: k.startswith(("embedding_layer.", "norm.")))
assert cfg.tie_embeddings
unembed = Lambda(emb.unembed)
n_fp8 = 0


def run_chunk(ids_list, on_logits):
    """Every block once over the sequences of ids_list; on_logits(j, logits [1, L, vocab] bf16) at the end."""
    global n_fp8
    t0 = time.time()
    with torch.inference_mode():
        hs = [emb(x[None].to(dev)) for x in ids_list]
        for i in range(cfg.num_layers):
            with torch.device(dev), torch.cuda.device(dev):
                blk = get_block(cfg, i, flash_fft=None)
            hb = holder(blocks=nn.ModuleList([nn.Identity() for _ in range(i)] + [blk]))
            pre = f"blocks.{i}."
            load_upstream(hb, lambda k: k.startswith(pre))
            n_fp8 = max(n_fp8, sum(getattr(m, "use_fp8_input_projections", False) for m in blk.modules()))
            for j in range(len(hs)):                                       # in place: one extra hidden state
                hs[j] = blk(hs[j], inference_params=None, padding_mask=None)[0]
            del blk, hb
            torch.cuda.empty_cache()
            if i % 10 == 9 or i == cfg.num_layers - 1:
                say(f"  block {i + 1}/{cfg.num_layers}, {time.time() - t0:.0f} s, "
                    f"GPU peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        for j, h in enumerate(hs):
            on_logits(j, unembed(norm(h)))
            hs[j] = None
    return time.time() - t0


def chunks(lengths, budget):
    out, cur, n = [], [], 0
    for k, L in enumerate(lengths):
        if cur and n + L > budget:
            out.append(cur)
            cur, n = [], 0
        cur.append(k)
        n += L
    return out + ([cur] if cur else [])


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def env_meta():
    p = torch.cuda.get_device_properties(0)
    v = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda}
    import importlib.metadata as md
    for k in ("transformer_engine", "transformer_engine_cu12", "transformer_engine_torch", "evo2", "vtx",
              "flash_attn", "nvidia-cublas-cu12"):
        try:
            v[k] = md.version(k)
        except md.PackageNotFoundError:
            pass
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,clocks.max.sm,power.limit",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except OSError as exc:
        smi = f"nvidia-smi unavailable: {exc}"
    return dict(gpu=p.name, capability=list(torch.cuda.get_device_capability(0)), sm_count=p.multi_processor_count,
                nvidia_smi=smi, versions=v, env={k: os.environ.get(k) for k in ("NVTE_DISABLE_NVRTC",
                                                                                 "CUBLASLT_LOG_LEVEL")})


inp = np.load(a.inputs)
groups = a.groups.split(",")
say(f"{a.model} from {a.checkpoint}; {torch.cuda.get_device_name(0)}; groups {groups}")

if "test" in groups:
    names = [str(n) for n in inp["test_names"]]
    ids = [torch.from_numpy(inp[f"test/{n}"].astype(np.int64)) for n in names]
    rows, arrays = [None] * len(names), {}

    def on_test(j, lg):
        lg, x = lg[0], ids[j].to(dev)
        pred, target = lg[:-1], x[1:]
        lp = torch.log_softmax(lg.float(), dim=-1)
        arrays[f"{names[j]}/ids"] = ids[j].numpy().astype(np.int16)
        arrays[f"{names[j]}/logits_bits"] = lg.contiguous().view(torch.int16).cpu().numpy()
        arrays[f"{names[j]}/acgt"] = lp[:, ACGT.to(dev)].cpu().numpy()
        arrays[f"{names[j]}/next"] = lp[:-1].gather(-1, x[1:, None]).squeeze(-1).cpu().numpy()
        rows[j] = dict(name=names[j], length=len(x), loss=F.cross_entropy(pred, target).item(),
                       accuracy=(pred.argmax(-1) == target).float().mean().item(),
                       loss_fp32=F.cross_entropy(pred.float(), target).item(), logits_sha256=sha(lg),
                       mean_next_logprob=float(arrays[f"{names[j]}/next"].mean()))
        say(f"  {names[j]}: loss {rows[j]['loss']:.4f}, accuracy {rows[j]['accuracy']:.4f}")

    seconds = run_chunk(ids, on_test)
    np.savez_compressed(os.path.join(a.out, "test.npz"), **arrays)
    p = [r for r in rows if r["name"] in ("p0", "p1", "p2", "p3")]
    summary = dict(model=a.model, kind="official path (TE FP8), upstream blocks one at a time", seconds=seconds,
                   fp8_projection_modules_per_block_max=n_fp8,
                   test_evo2_mean=dict(loss=float(np.mean([r["loss"] for r in p])),
                                       accuracy=float(np.mean([r["accuracy"] for r in p])),
                                       loss_fp32=float(np.mean([r["loss_fp32"] for r in p]))) if len(p) == 4 else None,
                   sequences=rows, env=env_meta())
    json.dump(summary, open(os.path.join(a.out, "test.json"), "w"), indent=1)
    say(f"test done in {seconds:.0f} s: {summary['test_evo2_mean']}")

if "brca1" in groups:
    def roc_auc_score(y, s):                    # binary AUROC with ties counted half, as sklearn's
        y, s = np.asarray(y, bool), np.asarray(s, np.float64)
        pos, neg = s[y][:, None], s[~y][None, :]
        if not pos.size or not neg.size:
            return None                          # undefined without both classes
        return float(((pos > neg).sum() + 0.5 * (pos == neg).sum()) / (pos.size * neg.size))

    N, M = a.brca1_variants, min(a.first_chunk_variants, a.brca1_variants)
    ref_index, lof = inp["brca1/ref_index"][:N], inp["brca1/lof"][:N]
    n_ref = int(ref_index.max()) + 1
    seqs = [inp[f"brca1/ref/{i}"] for i in range(n_ref)] + [inp[f"brca1/var/{i}"] for i in range(N)]
    ids = [torch.from_numpy(s.astype(np.int64)) for s in seqs]
    # windows of the first M variants first (their reference windows are a prefix: in order of first use)
    m_ref = int(ref_index[:M].max()) + 1
    first = list(range(m_ref)) + [n_ref + v for v in range(M)]
    in_first = set(first)
    rest = [k for k in range(len(ids)) if k not in in_first]
    plan = [first] + [[rest[j] for j in c] for c in chunks([len(ids[k]) for k in rest], a.chunk_tokens)]
    scores = np.full(len(ids), np.nan, dtype=np.float32)
    timing, res = [], dict(model=a.model, variants=N, lof=int(lof.sum()), ref_windows=n_ref, windows=len(ids))
    for c, group in enumerate(plan):
        if not group:
            continue

        def on_score(j, lg, group=group):
            k = group[j]
            lp = logits_to_logprobs(lg, ids[k][None].to(dev)).float().cpu().numpy()
            scores[k] = np.mean(lp[0][:len(ids[k])])                 # evo2.scoring._score_sequences
        say(f"BRCA1 chunk {c + 1}/{len(plan)}: {len(group)} windows")
        timing.append(dict(windows=len(group), seconds=run_chunk([ids[k] for k in group], on_score)))
        rs, vs = scores[:n_ref], scores[n_ref:]
        res.update(windows_done=int(np.isfinite(scores).sum()), chunks=timing, env=env_meta())
        if c == 0:
            d = vs[:M] - rs[ref_index[:M]]
            res[f"auroc_first_{M}"] = roc_auc_score(lof[:M], -d)
            say(f"BRCA1 first {M} variants: AUROC {res[f'auroc_first_{M}']}")
        np.savez(os.path.join(a.out, "brca1.npz"), rs=rs, vs=vs, ref_index=ref_index, lof=lof)
        json.dump(res, open(os.path.join(a.out, "brca1.json"), "w"), indent=1)
    assert np.isfinite(scores).all()
    res["auroc"] = roc_auc_score(lof, -(vs - rs[ref_index]))
    json.dump(res, open(os.path.join(a.out, "brca1.json"), "w"), indent=1)
    say(f"BRCA1 done: AUROC {res['auroc']} ({N} variants)")
say("DONE")
