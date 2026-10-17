"""GO #1 tie (fast / c first): the 8 models that fit MM_cublas64 on the H100 differ only in zero_exp (ignore | emin),
min_exp (-132 | -133) and c_exp_sub (floor | m126). Inside a cuBLAS GEMM the accumulator c of a block is never
external: every chain starts from +0 and c is the previous block's result. This checks, with the c6 family model,
that the 8 variants give identical chains for FP8 inputs (all finite codes, zero-heavy, subnormal-heavy, cancelling
pairs; all four format pairs), fast (one chain) and slow (restart every 4 blocks, fp32 RNE adds from +0).
usage: tie_check.py [--chains 20000] [--blocks 8]"""
from __future__ import annotations

import argparse
import itertools

import torch

import pc  # noqa: F401
import family as F

FMTS = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}


def finite_codes(fmt):
    v = torch.arange(256, dtype=torch.uint8).view(FMTS[fmt]).double()
    return torch.arange(256)[torch.isfinite(v)], v


def draw(fmt, n, K, kind, g):
    codes, v = finite_codes(fmt)
    if kind == "uniform":
        c = codes[torch.randint(len(codes), (n, K), generator=g)]
    elif kind == "zeros":
        c = codes[torch.randint(len(codes), (n, K), generator=g)]
        c = torch.where(torch.rand(n, K, generator=g) < 0.9, torch.zeros_like(c), c)
    elif kind == "subnormal":
        sub = codes[(v[codes].abs() < 2.0 ** (-6 if fmt == "e4m3" else -14)) & (v[codes] != 0)]
        c = sub[torch.randint(len(sub), (n, K), generator=g)]
        c = torch.where(torch.rand(n, K, generator=g) < 0.3, codes[torch.randint(len(codes), (n, K), generator=g)], c)
    elif kind == "cancel":
        c = codes[torch.randint(len(codes), (n, K), generator=g)]
        c[:, 1::2] = c[:, 0::2] ^ 0x80                    # x, -x pairs (sign bit)
    else:
        raise KeyError(kind)
    return v[c]


def chain(a, b, s, fmt, blocks, restart):
    n = a.shape[0]
    acc = torch.zeros(n, dtype=torch.float64)
    tot = torch.zeros(n, dtype=torch.float64)
    for i in range(blocks):
        if restart and i % restart == 0 and i:
            tot = F.fp32_add(tot, acc, "rne")
            acc = torch.zeros(n, dtype=torch.float64)
        acc = F.model(a[:, 32 * i:32 * i + 32], b[:, 32 * i:32 * i + 32], acc, fmt, s)
    return F.fp32_add(tot, acc, "rne") if restart else acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=int, default=20000)
    ap.add_argument("--blocks", type=int, default=8)
    a = ap.parse_args()
    g = torch.Generator().manual_seed(1)
    variants = [F.spec(groups="flat32", zero_exp=z, min_exp=m, c_exp_sub=cs)
                for z, m, cs in itertools.product(("ignore", "emin"), (-132, -133), ("floor", "m126"))]
    total, differ = 0, 0
    for fa, fb in itertools.product(FMTS, FMTS):
        for kind in ("uniform", "zeros", "subnormal", "cancel"):
            x = draw(fa, a.chains, 32 * a.blocks, kind, g)
            y = draw(fb, a.chains, 32 * a.blocks, kind, g)
            for restart in (0, 4):
                outs = [F.bits32(chain(x, y, s, (fa, fb), a.blocks, restart)) for s in variants]
                d = sum(int((o != outs[0]).sum()) for o in outs[1:])
                total += a.chains
                differ += d
                print(f"{fa}x{fb} {kind:9s} {'slow' if restart else 'fast'}: chains {a.chains} differing outputs {d}",
                      flush=True)
    print(f"RESULT chains {total} x 8 variants: differing {differ}")


if __name__ == "__main__":
    main()
