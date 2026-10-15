"""Vectorised, exact models of a tensor-core bf16 x bf16 + fp32 -> fp32 MMA (one output = one length-16 dot + c),
written to check v9's tensor-core E-pass against every published NVIDIA tensor-core model and against a wider family
of hypothetical accumulators.

Model family (all exact integer arithmetic, torch int64):
  'global'  : the toolbox's global-alignment block FMA (Generic_BFMA_TC.m, global_alignment=1): products kept
              denormalised (significand in [1,4), exponent ea+eb); all products and c are aligned to the largest
              exponent (clamped below at min_exp_limit) with 23+neab fraction bits, bits shifted out are truncated
              (no sticky), the integer sum is normalised and rounded to fp32 with frmode ('rz' = truncation as in the
              toolbox models, or 'rne' / 'ru').  A length-16 MMA is a chain of 16/nfma such blocks in k order, the
              output of one block being c of the next (toolbox GEMM.m).
  'late_c'  : as 'global' but c is excluded from the alignment and added afterwards to the (rz-normalised) product
              sum with an fp32 RNE addition (the Hopper/Blackwell fp8-via-HMMA style, Fig. 5c of the paper).
  'seq'     : products added to c one by one with IEEE fp32 RNE (forward or reverse k order).
  'pairwise': fp32 RNE pairwise tree over the 16 products, then + c.
  'exact'   : exact sum rounded once (rz / rne / ru).
Inputs are given as integer significands S (value = S * 2^(e - 23)), exponents e and signs.
"""
from __future__ import annotations

import torch

I64 = torch.int64
F64 = torch.float64


def bitlen(s):
    """bit length of non-negative int64 tensor values (< 2^53)."""
    m, e = torch.frexp(s.to(F64))
    return torch.where(s > 0, e.to(I64), torch.zeros_like(s))


def round_to_24(s, q, mode, p=24):
    """value = s * 2^q (s >= 0 int64, q int64): round to p (default 24) significant bits, return float64 (exact)."""
    nb = bitlen(s)
    sh = (nb - p).clamp_min(0)
    base = s >> sh
    rem = s - (base << sh)
    if mode == "rz":
        r = base
    elif mode == "ru":
        r = base + (rem > 0).to(I64)
    elif mode == "rne":
        half = torch.where(sh > 0, torch.ones_like(sh) << (sh - 1).clamp_min(0), torch.zeros_like(sh))
        up = (sh > 0) & ((rem > half) | ((rem == half) & ((base & 1) == 1)))
        r = base + up.to(I64)
    else:
        raise ValueError(mode)
    return torch.ldexp(r.to(F64), (q + sh).to(F64))


def f32_parts(c):
    """fp32 values (float64 tensor holding fp32 values, >= 0 or any sign) -> (S 24-bit, e, sign)."""
    c32 = c.to(torch.float32)
    bits = c32.view(torch.int32).to(I64) & 0xFFFFFFFF
    sgn = (bits >> 31) & 1
    ef = (bits >> 23) & 0xFF
    fr = bits & 0x7FFFFF
    S = torch.where(ef > 0, fr | (1 << 23), fr)
    e = torch.where(ef > 0, ef - 127, torch.full_like(ef, -126))
    return S, e, sgn


def block_global(S, e, sg, nz, c, neab, frmode, min_exp_limit, include_c=True, p=24):
    """One global-alignment block FMA.  S,e,sg,nz: [N, n] product significands (value S*2^(e-23)), exponents, signs,
    nonzero mask; c: [N] float64 (fp32 values).  Returns [N] float64 (fp32 values)."""
    Sc, ec, sgc = f32_parts(c)
    cnz = (c != 0) & include_c
    S2 = torch.cat([S, Sc[:, None]], 1)
    e2 = torch.cat([e, ec[:, None]], 1)
    sg2 = torch.cat([sg, sgc[:, None]], 1)
    nz2 = torch.cat([nz, cnz[:, None]], 1)
    big = torch.full_like(e2, -100000)
    maxe = torch.where(nz2, e2, big).amax(1).clamp_min(min_exp_limit)
    sh = (maxe[:, None] - e2)
    Sn = S2 << neab if neab >= 0 else S2 >> (-neab)
    al = torch.where(sh <= 31, Sn >> sh.clamp(0, 62), torch.zeros_like(Sn))
    al = torch.where(nz2, al, torch.zeros_like(al))
    tot = (al * (1 - 2 * sg2)).sum(1)
    mag = tot.abs()
    v = round_to_24(mag, maxe - 23 - neab, frmode, p)
    v = torch.where(tot < 0, -v, v)
    any_nz = nz2.any(1)
    return torch.where(any_nz, v, torch.zeros_like(v))


def mma_model(S, e, sg, nz, c, model):
    """S,e,sg,nz: [N,16]; c: [N] float64 (fp32 values).  model: dict.  Returns [N] float64 (fp32 values)."""
    kind = model["kind"]
    if kind in ("global", "late_c"):
        nfma = model["nfma"]
        order = model.get("perm")  # optional k permutation (e.g. interleaved pairs)
        if order is not None:
            S, e, sg, nz = S[:, order], e[:, order], sg[:, order], nz[:, order]
        d = c.clone()
        for j in range(0, 16, nfma):
            sl = slice(j, j + nfma)
            if kind == "global":
                d = block_global(S[:, sl], e[:, sl], sg[:, sl], nz[:, sl], d, model["neab"], model["frmode"],
                                 model["min_exp_limit"], p=model.get("p", 24))
            else:
                p = block_global(S[:, sl], e[:, sl], sg[:, sl], nz[:, sl], torch.zeros_like(d), model["neab"], "rz",
                                 model["min_exp_limit"], include_c=False)
                d = (p.to(torch.float32) + d.to(torch.float32)).to(F64)  # fp32 RNE add (exact IEEE single)
        return d
    vals = torch.ldexp(S.to(F64), (e - 23).to(F64)) * (1 - 2 * sg).to(F64)
    vals = torch.where(nz, vals, torch.zeros_like(vals))
    if kind == "seq":
        d = c.to(torch.float32)
        ks = range(16) if model.get("dir", 1) == 1 else range(15, -1, -1)
        for k in ks:
            d = d + vals[:, k].to(torch.float32)  # products of bf16 are exact in fp32 (E-pass range)
        return d.to(F64)
    if kind == "pairwise":
        x = vals.to(torch.float32)
        while x.shape[1] > 1:
            x = x[:, 0::2] + x[:, 1::2]
        return (x[:, 0] + c.to(torch.float32)).to(F64)
    if kind == "exact":  # non-negative inputs only (E-pass): exact sum via integers on a common grid
        assert not bool(sg.any())
        allS = torch.cat([S, f32_parts(c)[0][:, None]], 1)
        alle = torch.cat([e, f32_parts(c)[1][:, None]], 1)
        allnz = torch.cat([nz, (c != 0)[:, None]], 1)
        # E-pass terms are S=2^23 * 2^(e-23) powers of two (c too): exact sum = sum 2^e; exponents span <= 173, so
        # keep only terms within 40 binades of the max (others are < 2^-40 relative and cannot reach a rounding bit
        # of a 24-bit result unless they are the only nonzero tail; their presence is kept as a sticky bit).
        big = torch.full_like(alle, -100000)
        maxe = torch.where(allnz, alle, big).amax(1)
        sh = maxe[:, None] - alle
        inwin = allnz & (sh <= 40)
        s = torch.where(inwin, (allS << 17) >> sh.clamp(0, 62), torch.zeros_like(allS)).sum(1)  # grid 2^(maxe-40)
        sticky = (allnz & (sh > 40)).any(1).to(I64)
        s = (s << 1) | sticky
        return round_to_24(s, maxe - 23 - 17 - 1, model["frmode"])
    raise ValueError(kind)


# Published NVIDIA bf16 -> fp32 tensor-core models (Khattak & Mikaitis arXiv:2512.07004v4; toolbox models/*.m)
PUBLISHED = {
    "A100/A2/A30/L40S/Ada (NFMA=8, neab=1, rz, emin_align=-132)": dict(kind="global", nfma=8, neab=1, frmode="rz", min_exp_limit=-132),
    "H100/H200/B200/RTX PRO 6000 (NFMA=16, neab=2, rz, emin_align=-133)": dict(kind="global", nfma=16, neab=2, frmode="rz", min_exp_limit=-133),
}


def hypothetical_models():
    M = {}
    for nfma in (1, 2, 4, 8, 16):
        for neab in (-23, -20, -10, -5, 0, 1, 2, 3, 8):
            for fr in ("rz", "rne", "ru"):
                M[f"global nfma={nfma} neab={neab} {fr}"] = dict(kind="global", nfma=nfma, neab=neab, frmode=fr, min_exp_limit=-133)
        M[f"global nfma={nfma} neab=-10 rz, 14-bit output (fp8-style)"] = dict(kind="global", nfma=nfma, neab=-10, frmode="rz", min_exp_limit=-133, p=14)
        for neab in (-10, 0, 2):
            M[f"late_c nfma={nfma} neab={neab}"] = dict(kind="late_c", nfma=nfma, neab=neab, frmode="rz", min_exp_limit=-133)
    inter = [k for q in range(4) for k in (4 * q, 4 * q + 1)] + [k for q in range(4) for k in (4 * q + 2, 4 * q + 3)]
    M["global nfma=8 neab=2 rz, interleaved pairs"] = dict(kind="global", nfma=8, neab=2, frmode="rz", min_exp_limit=-133, perm=inter)
    M["late_c nfma=8 neab=-10, interleaved pairs"] = dict(kind="late_c", nfma=8, neab=-10, frmode="rz", min_exp_limit=-133, perm=inter)
    M["seq fp32 RNE forward"] = dict(kind="seq", dir=1)
    M["seq fp32 RNE reverse"] = dict(kind="seq", dir=-1)
    M["pairwise fp32 RNE"] = dict(kind="pairwise")
    for fr in ("rz", "rne", "ru"):
        M[f"exact then {fr}"] = dict(kind="exact", frmode=fr)
    return M


def mutant_models():
    """Accumulators that VIOLATE the property v9 relies on (range-limited alignment), to show what a failure looks like
    and that the per-architecture gate detects it: the alignment exponent is clamped below at a value inside the
    E-pass range (the published A100/Hopper quirk is the same mechanism at -132/-133, i.e. 46 binades lower)."""
    M = {}
    for lim in (-60, -70, -80):
        M[f"MUTANT global nfma=16 neab=2 rz, emin_align={lim}"] = dict(kind="global", nfma=16, neab=2, frmode="rz", min_exp_limit=lim)
    return M


# ---------------------------------------------------------------- E-pass inputs and the v9 exponent mapping
def e4m3_fields(codes):
    """codes int64 E4M3 bytes -> (ea clamped to >= -6, nonzero mask)."""
    x = codes.to(torch.uint8).view(torch.float8_e4m3fn).to(F64)
    nz = x != 0
    ea = torch.where(nz, (torch.frexp(x)[1] - 1).clamp_min(-6), torch.zeros_like(codes))
    return ea.to(I64), nz


def epass_terms(A, B):
    """A,B [N,16] int64 E4M3 codes of one 16-term block -> model inputs of the v9 E-pass MMA:
    a operand bf16 2^(6ea-7), b operand 2^(6eb-7) (0 for zero bytes), c = 2^-86."""
    ea, nza = e4m3_fields(A)
    eb, nzb = e4m3_fields(B)
    nz = nza & nzb
    e = 6 * (ea + eb) - 14                     # product exponent (power of two)
    S = torch.full_like(e, 1 << 23)            # product significand 1.0
    sg = torch.zeros_like(e)
    c = torch.full((A.shape[0],), 2.0 ** -86, dtype=F64)
    E = torch.where(nz, ea + eb, torch.full_like(e, -1000)).amax(1).clamp_min(-12)
    return S, e, sg, nz, c, E


def pbits64_raw(f):  # v2_common.cuh:106   (f = fp32 bits >> 23)
    return (f * 1400832 + 907255808) & 0xFFFFFFFF


def pbits64_hi(x):   # v9.cu pbits64_hi: __umulhi(x, 717225984) + 907255808
    return (((x * 717225984) >> 32) + 907255808) & 0xFFFFFFFF


def v9_exponent_ok(d, E):
    """d: [N] float64 (fp32 values) = E-pass MMA output; E: [N] true block exponent.
    Returns (ok_raw, ok_hi, dfield): both v9 mappings must give fp32 bits (E+127)<<23 in the exponent field."""
    bits = d.to(torch.float32).view(torch.int32).to(I64) & 0xFFFFFFFF
    want = (E + 127) << 23
    ok_raw = (pbits64_raw(bits >> 23) & 0x7F800000) == want
    ok_hi = (pbits64_hi(bits) & 0x7F800000) == want
    dfield = ((bits >> 23) & 0xFF) - (6 * E + 113)   # d in [0, 4] expected
    return ok_raw, ok_hi, dfield
