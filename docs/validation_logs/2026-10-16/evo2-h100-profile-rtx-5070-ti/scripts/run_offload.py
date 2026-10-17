"""Evo 2 forward passes with the weights in host RAM, one block at a time on the GPU.

The model is the upstream one (vortex StripedHyena, evo2's configuration, vortex's checkpoint loader), built on the
CPU. Each block is copied to the GPU, run on every (mode, prompt) activation, and its GPU copy dropped. The FP8 input
projections are replaced as in examples/evo2_blackwell.ipynb: "mmaemu" (the RTX 4060's GEMM, profile
ada-rtx4060-te), "mmaemu_h100" (the H100 PCIe's GEMM, profile hopper-h100pcie-te), "native" (torch._scaled_mm on
this GPU), or "bf16" (the plain bf16 linear layer).

Metrics are those of evo2/test/test_evo2.py: cross entropy and accuracy of the bf16 logits on its four prompts.
"""

import argparse
import contextlib
import csv
import hashlib
import io
import json
import pkgutil
import sys
import time
from importlib import resources

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import mmaemu

E4M3_MAX = 448.0
MODES = ("mmaemu", "native", "bf16")
PROFILES = {"mmaemu": "ada-rtx4060-te", "mmaemu_h100": "hopper-h100pcie-te"}   # the mmaemu modes and their profiles
EXPLAINED = set()                                   # (mode, M, K, N) already logged by mmaemu.explain


@contextlib.contextmanager
def build_on_cpu():
    """vortex puts every block on cuda:0 when CUDA is available; build on the CPU, skipping weight init."""
    saved = (torch.cuda.is_available, torch.cuda.device_count, torch.cuda.device)
    inits = ("xavier_uniform_", "xavier_normal_", "kaiming_uniform_", "kaiming_normal_", "normal_", "uniform_",
             "zeros_", "ones_", "trunc_normal_", "constant_")
    saved_inits = {n: getattr(torch.nn.init, n) for n in inits}

    class NoCudaDevice(contextlib.nullcontext):
        def __init__(self, device):
            super().__init__()

    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    torch.cuda.device = NoCudaDevice
    for n in inits:  # every parameter is overwritten by the strict load below
        setattr(torch.nn.init, n, lambda t, *a, **k: t)
    try:
        yield
    finally:
        torch.cuda.is_available, torch.cuda.device_count, torch.cuda.device = saved
        for n, f in saved_inits.items():
            setattr(torch.nn.init, n, f)


def load_model(model_name, checkpoint):
    from evo2.utils import CONFIG_MAP
    from vortex.model.model import StripedHyena
    from vortex.model.utils import dotdict, load_checkpoint

    config = dotdict(yaml.safe_load(pkgutil.get_data("evo2", CONFIG_MAP[model_name])))
    config.use_fp8_input_projections = False
    with build_on_cpu():
        model = StripedHyena(config)
    load_checkpoint(model, checkpoint)
    return model


def read_te_scales(checkpoint):
    """{block index: FP8 scale [input, weight]} from the TE `projections._extra_state` entries (as the notebook)."""
    torch.serialization.add_safe_globals([io.BytesIO])
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    scales = {}
    for key, extra in state.items():
        if not key.endswith(".projections._extra_state") or extra is None:
            continue
        if isinstance(extra, io.BytesIO):
            extra.seek(0)
            meta = torch.load(extra, map_location="cpu", weights_only=True)
        elif isinstance(extra, torch.Tensor):
            meta = torch.load(io.BytesIO(extra.numpy().tobytes()), map_location="cpu", weights_only=True)
        else:
            meta = extra
        scales[int(key.split(".")[1])] = meta["scale_fwd"][:2].float().clone()
    return scales


class Projection(torch.nn.Module):
    """A Hyena block's input projection in one of MODES; the computation of the notebook's FP8Projection.

    The E4M3 weight is derived on the device the weight is on, at every call (the same bits as precomputing it)."""

    def __init__(self, weight, scale):
        super().__init__()
        self.weight = weight                            # [N, K] bf16 Parameter, moved with the block
        self.scale = scale                              # [input, weight], fp32, CPU
        self.mode = "mmaemu"

    def forward(self, x):
        if self.mode == "bf16":                         # vortex's TELinear fallback without TE
            return F.linear(x.to(self.weight.dtype), self.weight, None), None
        scale = self.scale.to(x.device)
        scale_inv = torch.reciprocal(scale)
        wq = (self.weight.detach().float() * scale[1]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        shape, L = x.shape, x.shape[-2]
        pad = (16 - L % 16) % 16                        # vortex's TE path pads M to a multiple of 16
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        xq = (x.float() * scale[0]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).reshape(-1, x.shape[-1])
        if self.mode in PROFILES:
            key = (self.mode, *xq.shape, wq.shape[0])
            if key not in EXPLAINED:                    # the plan of each distinct call, once
                EXPLAINED.add(key)
                cp = mmaemu.explain(xq, wq.t(), scale_inv[0], scale_inv[1], out_dtype=torch.bfloat16,
                                    use_fast_accum=True, profile=PROFILES[self.mode])
                print(f"{time.strftime('%H:%M:%S')} explain {self.mode} M={key[1]} K={key[2]} N={key[3]}: {cp}",
                      flush=True)
            y = mmaemu.scaled_mm(xq, wq.t(), scale_inv[0], scale_inv[1], out_dtype=torch.bfloat16,
                                 use_fast_accum=True, profile=PROFILES[self.mode])
        elif self.mode == "native":
            y = torch._scaled_mm(xq, wq.t(), scale_a=scale_inv[0], scale_b=scale_inv[1],
                                 out_dtype=torch.bfloat16, use_fast_accum=True)
        else:
            raise ValueError(self.mode)
        return y.reshape(*shape[:-2], -1, self.weight.shape[0])[..., :L, :], None


def patch_projections(model, scales):
    for i in scales:
        original = model.blocks[i].projections
        assert getattr(original, "bias", None) is None
        model.blocks[i].projections = Projection(original.weight, scales[i])
    return len(scales)


def set_mode(model, scales, mode):
    for i in scales:
        model.blocks[i].projections.mode = mode


def prompts():
    with resources.path("evo2.test.data", "prompts.csv") as p, open(p, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))[1:]
    return [r[0] for r in rows]


def window(name, seqs):
    """p0-p3: the prompts of test_evo2.py; w2k/w4k/w8k: the first 2,048/4,096/8,192 bp of their concatenation
    (the windows of evo2-anywhere's benchmarks/bench.py)."""
    if name[0] == "p":
        return seqs[int(name[1:])]
    return "".join(seqs)[:int(name[1:-1]) * 1024]


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


ACGT = [ord(c) for c in "ACGT"]


def stats(logits, ids, ref, name):
    """test_evo2.py's loss and accuracy (bf16 logits), the fp32 cross entropy, and the log-probabilities of
    evo2-anywhere's bench.py (A, C, G, T at every position; the true next token), compared with `ref` if given."""
    pred, target = logits[0, :-1, :], ids[1:]
    lp = torch.log_softmax(logits[0].float(), dim=-1)
    acgt = lp[:, ACGT].cpu().numpy()
    nxt = lp[:-1].gather(-1, target[:, None]).squeeze(-1).cpu().numpy()
    row = dict(loss=F.cross_entropy(pred, target).item(),
               accuracy=(pred.argmax(-1) == target).float().mean().item(),
               loss_fp32=F.cross_entropy(pred.float(), target).item(),
               mean_next_logprob=float(nxt.mean()), logits_sha256=sha(logits))
    if ref is not None and f"{name}/acgt" in ref.files:
        r_acgt, r_next = ref[f"{name}/acgt"], ref[f"{name}/next"]
        d = np.abs(acgt - r_acgt)
        row.update(ref_mean_next_logprob=float(r_next.mean()), mean_abs_dlogp_acgt=float(d.mean()),
                   max_abs_dlogp_acgt=float(d.max()), acgt_argmax_agree=float((acgt.argmax(1) == r_acgt.argmax(1)).mean()),
                   acgt_bit_identical=bool(np.array_equal(acgt, r_acgt)))
    return row, acgt, nxt


def offload_forward(model, scales, modes, ids_list, dev, log):
    """Logits for every (mode, prompt), streaming each block to the GPU once."""
    model.embedding_layer.to(dev)
    if model.norm is not None:
        model.norm.to(dev)
    with torch.inference_mode():
        emb = [model.embedding_layer(ids[None].to(dev)) for ids in ids_list]
        xs = {(m, p): emb[p].clone() for m in modes for p in range(len(ids_list))}
        del emb
        for b, block in enumerate(model.blocks):
            t0 = time.time()
            held = [(t, t.data) for t in list(block.parameters()) + list(block.buffers())]
            for t, d in held:
                t.data = d.to(dev)
            torch.cuda.synchronize()
            t1 = time.time()
            for m in modes:
                if b in scales:
                    model.blocks[b].projections.mode = m
                for p in range(len(ids_list)):
                    xs[m, p], _ = block(xs[m, p], inference_params=None, padding_mask=None)
            torch.cuda.synchronize()
            for t, d in held:
                t.data = d
            del held
            log(f"block {b:2d} ({type(block).__name__}{', fp8' if b in scales else ''}): copy {t1 - t0:.1f} s, "
                f"compute {time.time() - t1:.1f} s, GPU peak {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB")
        out = {}
        for (m, p), x in xs.items():
            out[m, p] = model.unembed(model.norm(x))
    return out


def resident_forward(model, modes, scales, ids_list, dev):
    """The whole model on the GPU through vortex's own forward (for checking the offload path on small models)."""
    model.to(dev)
    model.block_idx_to_device = {i: str(dev) for i in model.block_idx_to_device}
    out = {}
    with torch.inference_mode():
        for m in modes:
            set_mode(model, scales, m)
            for p, ids in enumerate(ids_list):
                out[m, p] = model(ids[None].to(dev))[0]
    model.to("cpu")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--windows", default="p0,p1,p2,p3", help="p0-p3 (test_evo2.py's prompts), w2k, w4k, w8k")
    ap.add_argument("--ref", help="npz of reference log-probabilities (<window>/acgt, <window>/next)")
    ap.add_argument("--check-resident", action="store_true", help="also run the whole model on the GPU, compare")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    dev = torch.device("cuda:0")
    modes = args.modes.split(",")
    from vortex.model.tokenizer import CharLevelTokenizer

    tok = CharLevelTokenizer(512)
    seqs = prompts()
    sel = args.windows.split(",")
    ids_list = [torch.tensor(tok.tokenize(window(w, seqs)), dtype=torch.long) for w in sel]
    ref = np.load(args.ref) if args.ref else None
    log(f"mmaemu {mmaemu.__version__} from {mmaemu.__file__}; torch {torch.__version__}; "
        f"{torch.cuda.get_device_name(dev)}; windows {sel}, lengths {[len(x) for x in ids_list]}")

    t0 = time.time()
    model = load_model(args.model, args.checkpoint)
    scales = read_te_scales(args.checkpoint)
    n = patch_projections(model, scales)
    nparams = sum(p.numel() for p in model.parameters())
    log(f"loaded {args.model} on the CPU in {time.time() - t0:.0f} s: {nparams / 1e9:.2f} B parameters, "
        f"FP8 projections in {n} blocks {sorted(scales)}")
    shapes = sorted({tuple(model.blocks[i].projections.weight.shape) for i in scales})
    log(f"projection weight shapes [N, K]: {shapes}")

    t0 = time.time()
    out = offload_forward(model, scales, modes, ids_list, dev, log)
    log(f"offload forward, {len(modes)} modes x {len(sel)} prompts: {time.time() - t0:.0f} s")

    rows, arrays = [], {}
    for (m, p), logits in out.items():
        row, acgt, nxt = stats(logits, ids_list[p].to(dev), ref, sel[p])
        rows.append(dict(mode=m, window=sel[p], length=len(ids_list[p]), **row))
        arrays[f"{m}/{sel[p]}/acgt"], arrays[f"{m}/{sel[p]}/next"] = acgt, nxt
    result = dict(model=args.model, checkpoint=args.checkpoint, device=torch.cuda.get_device_name(dev),
                  mmaemu=mmaemu.__version__, torch=torch.__version__, rows=rows)

    if args.check_resident:
        ref = resident_forward(model, modes, scales, ids_list, dev)
        result["resident_equal"] = {f"{m}/{sel[p]}": bool(torch.equal(ref[m, p], out[m, p])) for (m, p) in out}
        log(f"offload == resident, bit for bit: {result['resident_equal']}")

    for x in rows:
        cmp = (f"; vs ref: mean |dlogp| {x['mean_abs_dlogp_acgt']:.4f}, max {x['max_abs_dlogp_acgt']:.3f}, "
               f"argmax agree {x['acgt_argmax_agree']:.2%}, ref next logprob {x['ref_mean_next_logprob']:.4f}"
               if "mean_abs_dlogp_acgt" in x else "")
        log(f"{x['mode']:7s} {x['window']:4s}: loss {x['loss']:.4f}, acc {x['accuracy']:.2%}, "
            f"next logprob {x['mean_next_logprob']:.4f}{cmp}")
    tests = [w for w in sel if w in ("p0", "p1", "p2", "p3")]
    if len(tests) == 4:
        for m in modes:
            r = [x for x in rows if x["mode"] == m and x["window"] in tests]
            mean = {k: sum(x[k] for x in r) / len(r) for k in ("loss", "accuracy", "loss_fp32")}
            result.setdefault("test_evo2_mean", {})[m] = mean
            log(f"{m:7s} test_evo2.py: mean loss {mean['loss']:.4f}, accuracy {mean['accuracy']:.3%}, "
                f"fp32 loss {mean['loss_fp32']:.4f}")
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    np.savez(args.out.removesuffix(".json") + ".npz", **arrays)
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
