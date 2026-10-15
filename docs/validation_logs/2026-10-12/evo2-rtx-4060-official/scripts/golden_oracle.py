"""Golden for models that do not fit the GPU: the upstream model run block by block.
Every block is built with vortex's get_block() and loaded with vortex's OWN StripedHyena.to_bfloat16_except_pr_lc +
custom_load_state_dict (incl. TE fp8 recipe injection and the Wqkv column-split permutation), called through a holder
module that has only that block -- no evo2-anywhere code touches the weights. Head (embedding/norm/unembed) the same way.
Saves full logits bit patterns and a sha256 of every block's output hidden state.
usage: python golden_oracle.py --model evo2_40b --windows w2k [--official fp8|bf16]"""
import argparse, io, logging, pkgutil, time
import torch, yaml
from torch import nn
import common

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--windows", required=True)
ap.add_argument("--official", choices=["fp8", "bf16"], default="fp8"); a = ap.parse_args()
torch.cuda.set_per_process_memory_fraction(0.95)
from evo2.utils import CONFIG_MAP
from vortex.model.layers import RMSNorm, VocabParallelEmbedding
from vortex.model.model import StripedHyena, get_block
from vortex.model.utils import Lambda, dotdict
try:
    import transformer_engine  # noqa: F401
    HAS_TE = True
except ImportError:
    HAS_TE = False
assert HAS_TE == (a.official == "fp8"), "fp8 golden needs the TE env; bf16 golden needs an env WITHOUT TE (upstream fallback)"

cfg = dotdict(yaml.safe_load(pkgutil.get_data("evo2", CONFIG_MAP[a.model])))
if cfg.get("use_fp8_input_projections", False) and not HAS_TE:
    assert "7b" in a.model, "upstream refuses non-7B models without TE"
    cfg.use_fp8_input_projections = False          # exactly what evo2.Evo2 does without TE (models.py)
import os
path = {"evo2_1b_base": "evo2_1b_base/evo2_1b_base.pt", "evo2_7b": "evo2_7b/evo2_7b.pt", "evo2_40b": "evo2_40b/evo2_40b.pt"}[a.model]
ckpt = os.path.join("/data/quant/models", path)
safe = [io.BytesIO]
if HAS_TE:
    from transformer_engine.common.recipe import DelayedScaling, Format, _FormatHelper
    safe += [DelayedScaling, Format, _FormatHelper]
torch.serialization.add_safe_globals(safe)
state = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)   # as vortex load_checkpoint
log = logging.getLogger("golden_oracle"); dev = "cuda:0"


def holder(**mods):
    h = StripedHyena.__new__(StripedHyena); nn.Module.__init__(h)
    h.config, h.logger = cfg, log
    for k, v in mods.items():
        setattr(h, k, v)
    if "blocks" not in mods:
        h.blocks = nn.ModuleList()
    return h


def load_upstream(h, prefix_filter):
    StripedHyena.to_bfloat16_except_pr_lc(h, to_float32=True)                 # load_checkpoint order: dtype first
    StripedHyena.custom_load_state_dict(h, {k: v for k, v in state.items() if prefix_filter(k)})


tok = common.tokenizer()
common.bench.ACGT = torch.tensor([int(tok.tokenize(c)[0]) for c in "ACGT"], dtype=torch.long)
wins = common.windows(a.windows.split(","))
ids = [torch.tensor(tok.tokenize(s), dtype=torch.long, device=dev).unsqueeze(0) for _, s in wins]
with torch.device(dev), torch.cuda.device(dev):
    emb = VocabParallelEmbedding(cfg); norm = RMSNorm(cfg)
hh = holder(embedding_layer=emb, norm=norm)
load_upstream(hh, lambda k: k.startswith(("embedding_layer.", "norm.")))
assert cfg.tie_embeddings
unembed = Lambda(emb.unembed)
t0 = time.time(); block_hashes = {n: [] for n, _ in wins}
with torch.inference_mode():
    hs = [emb(x) for x in ids]
    for i in range(cfg.num_layers):
        with torch.device(dev), torch.cuda.device(dev):
            blk = get_block(cfg, i, flash_fft=None)
        hb = holder(blocks=nn.ModuleList([nn.Identity() for _ in range(i)] + [blk]))
        pre = f"blocks.{i}."
        load_upstream(hb, lambda k: k.startswith(pre))
        hs = [blk(h, inference_params=None, padding_mask=None)[0] for h in hs]
        for (n, _), h in zip(wins, hs):
            block_hashes[n].append(common.sha(h))
        del blk, hb; torch.cuda.empty_cache()
        print(f"block {i} done {time.time() - t0:.0f}s", flush=True)
    logits = [unembed(norm(h))[0] for h in hs]
arrays, per = {}, []
for (n, s), x, lg in zip(wins, ids, logits):
    acgt, nxt = common.bench.summarize(lg, x)
    arrays[f"{n}/ids"] = x[0].cpu().numpy().astype("int16")
    arrays[f"{n}/logits_bits"] = common.bits(lg)
    arrays[f"{n}/acgt"], arrays[f"{n}/next"] = acgt, nxt
    per.append(dict(window=n, bp=len(s), logits_dtype=str(lg.dtype), logits_shape=list(lg.shape),
                    logits_sha256=common.sha(lg), mean_next_logprob=float(nxt.mean())))
tag = f"{a.model}_oracle-{'te-fp8' if a.official == 'fp8' else 'bf16-noTE'}_{'-'.join(n for n, _ in wins)}"
common.save(tag, arrays, dict(tag=tag, kind="upstream blocks one at a time (vortex get_block + vortex loaders)",
                              model=a.model, official=a.official, windows=per, seconds=round(time.time() - t0, 1),
                              block_hashes=block_hashes, env=common.env_meta()))
