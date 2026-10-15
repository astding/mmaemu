"""Run Evo 2 (including 40B) on one GPU with the weights in host RAM; the FP8 projections are computed by mmaemu.

Evo 2 1B, 20B and 40B compute their input projections as FP8 GEMMs through Transformer Engine, and give wrong
results on GPUs whose FP8 arithmetic differs from the one they were trained with (evo2 issue #157). This script
runs the upstream model (vortex's StripedHyena, evo2's configuration and checkpoint loader) without Transformer
Engine and computes each FP8 projection as Transformer Engine does on an RTX 4060, with `mmaemu.scaled_mm`
(profile "ada-rtx4060-te"), as in examples/evo2_blackwell.ipynb. The weights stay in host RAM and the model runs
one block at a time on the GPU, so Evo 2 40B (82 GB of weights) runs on a 16 GB GPU.

Measured on an RTX 5070 Ti with 180 GiB of RAM (docs/validation_logs/2026-10-15/evo2-40b-rtx-5070-ti/): Evo 2 40B
scores 0.2161 on the upstream test, where 0.2159 is expected (native FP8: 1.4023). Only the FP8 GEMMs are the
RTX 4060's; the rest of the model runs natively, so the outputs are close to the official ones, not equal to them.

Requirements: evo2 0.6.0 with its vtx and flash-attn, torch with CUDA, mmaemu; no Transformer Engine. Host RAM: the
checkpoint (82 GB for 40B, 2.7 GB for 1B) plus 2 bytes x hidden size (8,192 for 40B) x the tokens of the sequences
processed together, page-locked (--ram-gb limits the latter); the 40B test peaked at 157 GiB resident on the
180 GiB host, counting the pages of the memory-mapped checkpoint. Sequences up to 8,192 bp were run on a 16 GB GPU.

The 40B checkpoint is published in two parts; join them first:
    hf download arcinstitute/evo2_40b --local-dir evo2_40b
    cat evo2_40b/evo2_40b.pt.part0 evo2_40b/evo2_40b.pt.part1 > evo2_40b.pt

Usage:
    python evo2_offload.py test --model evo2_40b --checkpoint evo2_40b.pt
    python evo2_offload.py score --model evo2_40b --checkpoint evo2_40b.pt --input seqs.fasta --out scores.tsv
Options: --fp8 mmaemu (default) | native (torch._scaled_mm on this GPU) | bf16 (bf16 GEMMs, evo2's 7B fallback).
`score` gives each sequence's mean (or summed) log-probability of its next tokens, as evo2's `score_sequences`.
"""

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import pkgutil
import sys
import time
from importlib import resources

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import mmaemu

E4M3_MAX = 448.0  # largest finite float8_e4m3fn value; TE saturates the cast to it
# evo2/test/test_evo2.py (evo2 0.6.0)
EXPECTED = {"evo2_40b": (0.2159424, 91.673), "evo2_7b": (0.3476563, 86.346), "evo2_40b_base": (0.2149658, 91.741),
            "evo2_7b_base": (0.3520508, 85.921), "evo2_1b_base": (0.501953125, 79.556),
            "evo2_20b": (0.2166748046875, 91.666)}


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


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
    for n in inits:  # every parameter is overwritten by vortex's strict load
        setattr(torch.nn.init, n, lambda t, *a, **k: t)
    try:
        yield
    finally:
        torch.cuda.is_available, torch.cuda.device_count, torch.cuda.device = saved
        for n, f in saved_inits.items():
            setattr(torch.nn.init, n, f)


def read_te_scales(checkpoint):
    """{block index: FP8 scale [input, weight]} from the TE `projections._extra_state` entries of the checkpoint."""
    torch.serialization.add_safe_globals([io.BytesIO])
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    scales = {}
    for key, extra in state.items():
        if not key.endswith(".projections._extra_state") or extra is None:
            continue
        if isinstance(extra, io.BytesIO):
            extra.seek(0)
            meta = torch.load(extra, map_location="cpu", weights_only=True)
        elif isinstance(extra, torch.Tensor):  # newer TE versions serialise the metadata into a uint8 tensor
            meta = torch.load(io.BytesIO(extra.numpy().tobytes()), map_location="cpu", weights_only=True)
        else:
            meta = extra
        scales[int(key.split(".")[1])] = meta["scale_fwd"][:2].float().clone()
    return scales


class Projection(torch.nn.Module):
    """A Hyena block's input projection: "mmaemu" (TE's FP8 GEMM on an RTX 4060, emulated), "native"
    (torch._scaled_mm on this GPU) or "bf16" (vortex's linear layer without TE). The E4M3 weight is derived on the
    device the weight is on, at every call (the same bits as deriving it once)."""

    def __init__(self, weight, scale, mode):
        super().__init__()
        self.weight = weight                            # [N, K] bf16 Parameter, moved with its block
        self.scale = scale                              # [input, weight], fp32, CPU
        self.mode = mode

    def forward(self, x):
        if self.mode == "bf16":
            return F.linear(x.to(self.weight.dtype), self.weight, None), None
        scale = self.scale.to(x.device)
        scale_inv = torch.reciprocal(scale)             # TE scales the result by 1/scale of each operand
        wq = (self.weight.detach().float() * scale[1]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        shape, L = x.shape, x.shape[-2]
        pad = (16 - L % 16) % 16                        # vortex's TE path pads M to a multiple of 16
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        xq = (x.float() * scale[0]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).reshape(-1, x.shape[-1])
        if self.mode == "mmaemu":
            y = mmaemu.scaled_mm(xq, wq.t(), scale_inv[0], scale_inv[1], out_dtype=torch.bfloat16,
                                 use_fast_accum=True, profile="ada-rtx4060-te")
        else:
            y = torch._scaled_mm(xq, wq.t(), scale_a=scale_inv[0], scale_b=scale_inv[1],
                                 out_dtype=torch.bfloat16, use_fast_accum=True)
        return y.reshape(*shape[:-2], -1, self.weight.shape[0])[..., :L, :], None   # TELinear's (output, bias)


def load_model(model_name, checkpoint, mode):
    from evo2.utils import CONFIG_MAP
    from vortex.model.model import StripedHyena
    from vortex.model.utils import dotdict, load_checkpoint

    config = dotdict(yaml.safe_load(pkgutil.get_data("evo2", CONFIG_MAP[model_name])))
    config.use_fp8_input_projections = False
    with build_on_cpu():
        model = StripedHyena(config)
    load_checkpoint(model, checkpoint)
    scales = read_te_scales(checkpoint)
    for i in scales:
        original = model.blocks[i].projections
        assert getattr(original, "bias", None) is None, f"block {i}: Evo 2's FP8 projections have no bias"
        model.blocks[i].projections = Projection(original.weight, scales[i], mode)
    return model, scales


class FilterCache:
    """The long Hyena filter of a block depends only on the sequence length: compute it once per block and length
    (sequences are processed sorted by length, so one entry is kept)."""

    def __init__(self, block):
        self.f = getattr(block, "filter", None)
        self.on = self.f is not None and getattr(self.f, "h", 1) is None and hasattr(self.f, "compute_filter")
        if self.on:
            orig, memo = self.f.compute_filter, {}

            def cached(L, device):
                key = (L, str(device))
                if key not in memo:
                    memo.clear()
                    memo[key] = orig(L, device)
                return memo[key]

            self.f.compute_filter = cached

    def close(self):
        if self.on:
            del self.f.compute_filter


def forward_offloaded(model, ids_list, dev, on_logits):
    """Run every sequence of `ids_list` through the model, streaming each block to the GPU once; call
    on_logits(k, logits) with the bf16 logits [1, L, vocab] of sequence k. Between blocks the activations wait in one
    page-locked host buffer; the weights go to the GPU through a page-locked staging buffer."""
    order = sorted(range(len(ids_list)), key=lambda k: len(ids_list[k]))
    offsets = np.cumsum([0] + [len(ids_list[k]) for k in order])
    tensors = lambda m: list(m.parameters()) + list(m.buffers())  # noqa: E731
    staging = torch.empty(max(sum(t.numel() * t.element_size() for t in tensors(b)) for b in model.blocks),
                          dtype=torch.uint8, pin_memory=True)
    store = torch.empty((int(offsets[-1]), model.config.hidden_size), dtype=torch.bfloat16, pin_memory=True)
    view = lambda j: store[offsets[j]:offsets[j + 1]]  # noqa: E731
    model.embedding_layer.to(dev)
    model.norm.to(dev)
    t0 = time.time()
    with torch.inference_mode():
        for j, k in enumerate(order):
            x = model.embedding_layer(ids_list[k][None].to(dev))
            assert x.dtype == store.dtype
            view(j).copy_(x[0])
        for b, block in enumerate(model.blocks):
            held = []
            for t in tensors(block):
                d = t.data
                stage = staging[:d.numel() * d.element_size()].view(d.dtype).view(d.shape)
                stage.copy_(d)
                g = torch.empty_like(d, device=dev)
                g.copy_(stage)                          # synchronous from page-locked memory
                held.append((t, d))
                t.data = g
            cache = FilterCache(block)
            for j in range(len(order)):
                y, _ = block(view(j)[None].to(dev), inference_params=None, padding_mask=None)
                assert y.dtype == store.dtype and y.shape[1:] == view(j).shape
                view(j).copy_(y[0])
            torch.cuda.synchronize()
            cache.close()
            for t, d in held:
                t.data = d
            del held, cache
            torch.cuda.empty_cache()
            if b % 10 == 9 or b == len(model.blocks) - 1:
                log(f"block {b + 1}/{len(model.blocks)}, {time.time() - t0:.0f} s, "
                    f"GPU peak {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB")
        for j, k in enumerate(order):
            on_logits(k, model.unembed(model.norm(view(j)[None].to(dev))))


def chunks(ids_list, hidden, ram_gb):
    """Groups of sequence indices whose page-locked activations fit in ram_gb."""
    budget = int(ram_gb * 2**30 / (2 * hidden))
    out, cur, n = [], [], 0
    for k in sorted(range(len(ids_list)), key=lambda k: len(ids_list[k])):
        L = len(ids_list[k])
        if L > budget:
            raise SystemExit(f"a sequence of {L} tokens needs more than --ram-gb {ram_gb} of activations")
        if cur and n + L > budget:
            out.append(cur)
            cur, n = [], 0
        cur.append(k)
        n += L
    return out + ([cur] if cur else [])


def run(model, ids_list, dev, ram_gb, on_logits):
    groups = chunks(ids_list, model.config.hidden_size, ram_gb)
    for g, group in enumerate(groups):
        if len(groups) > 1:
            log(f"pass {g + 1}/{len(groups)}: {len(group)} sequences")
        forward_offloaded(model, [ids_list[k] for k in group], dev, lambda j, lg: on_logits(group[j], lg))


def read_sequences(path):
    """FASTA, or one sequence per line (names seq0, seq1, ...)."""
    names, seqs = [], []
    with open(path) as f:
        text = f.read()
    if text.lstrip().startswith(">"):
        for rec in text.split(">")[1:]:
            head, _, body = rec.partition("\n")
            names.append(head.split()[0] if head.split() else f"seq{len(names)}")
            seqs.append("".join(body.split()).upper())
    else:
        for line in text.splitlines():
            if line.strip():
                names.append(f"seq{len(names)}")
                seqs.append(line.strip().upper())
    return names, seqs


def sha256(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["test", "score"])
    ap.add_argument("--model", required=True, help="evo2 model name, e.g. evo2_40b, evo2_1b_base, evo2_20b")
    ap.add_argument("--checkpoint", required=True, help="the model's .pt file (for 40B, the two parts joined)")
    ap.add_argument("--fp8", choices=["mmaemu", "native", "bf16"], default="mmaemu")
    ap.add_argument("--input", help="score: FASTA, or one sequence per line")
    ap.add_argument("--out", help="score: TSV of name, length and score; test: JSON of the results")
    ap.add_argument("--reduce", choices=["mean", "sum"], default="mean", help="score: as evo2's score_sequences")
    ap.add_argument("--ram-gb", type=float, default=32.0,
                    help="page-locked host RAM for the activations of the sequences run together (default 32)")
    a = ap.parse_args()
    if a.command == "score" and not (a.input and a.out):
        ap.error("score needs --input and --out")

    from vortex.model.tokenizer import CharLevelTokenizer

    dev = torch.device("cuda:0")
    log(f"mmaemu {mmaemu.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(dev)}; "
        f"{a.model}, FP8 projections: {a.fp8}")
    size = os.path.getsize(a.checkpoint) / 2**30
    try:
        avail = int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) / 2**20
    except (OSError, IndexError, ValueError):
        avail = float("inf")
    if size + a.ram_gb > avail:
        log(f"warning: the checkpoint ({size:.0f} GiB) and --ram-gb {a.ram_gb:g} exceed the available RAM "
            f"({avail:.0f} GiB)")
    t0 = time.time()
    model, scales = load_model(a.model, a.checkpoint, a.fp8)
    log(f"loaded in {time.time() - t0:.0f} s: {sum(p.numel() for p in model.parameters()) / 1e9:.2f} B parameters, "
        f"FP8 projections in {len(scales)} blocks")
    tok = CharLevelTokenizer(512)

    if a.command == "test":
        with resources.as_file(resources.files("evo2.test.data") / "prompts.csv") as p, \
                open(p, encoding="utf-8-sig", newline="") as f:
            seqs = [r[0] for r in list(csv.reader(f))[1:]]
        ids_list = [torch.tensor(tok.tokenize(s), dtype=torch.long) for s in seqs]
        rows = {}

        def on_logits(k, logits):                       # the metrics of evo2/test/test_evo2.py
            pred, target = logits[0, :-1, :], ids_list[k][1:].to(dev)
            rows[k] = dict(loss=F.cross_entropy(pred, target).item(),
                           accuracy=(pred.argmax(-1) == target).float().mean().item(),
                           loss_fp32=F.cross_entropy(pred.float(), target).item(), logits_sha256=sha256(logits))

        run(model, ids_list, dev, a.ram_gb, on_logits)
        rows = [rows[k] for k in range(len(seqs))]
        for k, r in enumerate(rows):
            log(f"sequence {k + 1}: loss {r['loss']:.4f}, accuracy {r['accuracy']:.2%}")
        loss, acc = np.mean([r["loss"] for r in rows]), np.mean([r["accuracy"] for r in rows]) * 100
        exp = EXPECTED.get(a.model)
        verdict = "" if exp is None else (f"; expected {exp[0]:.4f} ({exp[1]:.3f}%): "
                                          f"{'PASS' if abs(loss - exp[0]) < 1e-3 else 'FAIL'} (tolerance 1e-3)")
        log(f"mean loss {loss:.4f}, accuracy {acc:.3f}%{verdict}")
        if a.out:
            with open(a.out, "w") as f:
                json.dump(dict(model=a.model, fp8=a.fp8, device=torch.cuda.get_device_name(dev),
                               mmaemu=mmaemu.__version__, mean_loss=float(loss), mean_accuracy=float(acc) / 100,
                               sequences=rows), f, indent=1)
        return 0 if exp is None or abs(loss - exp[0]) < 1e-3 else 1

    from evo2.scoring import logits_to_logprobs

    names, seqs = read_sequences(a.input)
    ids_list = [torch.tensor(tok.tokenize(s), dtype=torch.long) for s in seqs]
    log(f"{len(seqs)} sequences, {sum(map(len, seqs)):,} bp")
    scores = {}
    reduce = np.mean if a.reduce == "mean" else np.sum

    def on_logits(k, logits):                           # evo2.scoring._score_sequences with batch size 1
        lp = logits_to_logprobs(logits, ids_list[k][None].to(dev)).float().cpu().numpy()
        scores[k] = reduce(lp[0][:len(ids_list[k])])

    run(model, ids_list, dev, a.ram_gb, on_logits)
    with open(a.out, "w") as f:
        f.write("name\tlength\tscore\n")
        for k, name in enumerate(names):
            f.write(f"{name}\t{len(seqs[k])}\t{float(scores[k])!r}\n")   # the float32 score, exactly
    log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
