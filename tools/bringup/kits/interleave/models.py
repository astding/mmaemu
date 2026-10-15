"""Independent numpy re-implementation of candidate block-FMA models for one K=32 FP8 MMA (d = a.b + c).

All arithmetic is exact in float64 (fp8/fp16 products, power-of-two scalings, truncations, sums < 2^53);
the only rounding is the final fp32 FADD, done with numpy float32 (IEEE RNE, no FTZ)."""
import numpy as np

EMIN = {"e4m3": -6, "e5m2": -14, "fp16": -14}


def fexp(x):
    """floor(log2|x|), exact (frexp); arbitrary for x == 0 (callers mask zeros)."""
    return (np.frexp(x)[1] - 1).astype(np.int64)


def trunc_to(d, frac_out):
    """truncate (toward zero) to frac_out fraction bits relative to the value's exponent, clamped at fp32 emin."""
    nz = d != 0
    ed = np.where(nz, np.maximum(fexp(np.where(nz, d, 1.0)), -126), 0)
    return np.where(nz, np.trunc(np.ldexp(d, frac_out - ed)) * np.ldexp(1.0, ed - frac_out), 0.0)


def bfma(a, b, c, frac, frac_out, min_exp, emin, c_in=True):
    """One block FMA over the last axis. a, b: [N, n] float64; c: [N] fp32 values in float64.
    Products aligned to the max exponent (input exponents clamped at emin, denormalised significands), truncated
    to `frac` fraction bits; c in the alignment iff c_in; exact sum; result truncated to `frac_out` bits."""
    p = a * b
    nzp = p != 0
    ea = np.maximum(fexp(np.where(a != 0, a, 1.0)), emin)
    eb = np.maximum(fexp(np.where(b != 0, b, 1.0)), emin)
    ep = np.where(nzp, ea + eb, -10**6)
    E = ep.max(-1)
    if c_in:
        E = np.maximum(E, np.where(c != 0, fexp(np.where(c != 0, c, 1.0)), -10**6))
    E = np.maximum(E, min_exp)
    s = np.trunc(np.ldexp(p, (frac - E)[:, None])).sum(-1)
    if c_in:
        s = s + np.trunc(np.ldexp(c, frac - E))
    return trunc_to(np.ldexp(s, E - frac), frac_out)


def fadd32(x, c):
    return (x.astype(np.float32) + c.astype(np.float32)).astype(np.float64)


# ---- candidate models of one m16n8k32 FP8 MMA (one output element, K = 32, file index = k) -----------------------
def m_single_late(a, b, c, fmt, frac=25, frac_out=23, min_exp=-133):
    """tcemu PRESETS['blackwell']: one 32-term block over products only, then fp32 RNE add of c."""
    s = bfma(a, b, np.zeros_like(c), frac, frac_out, min_exp, EMIN[fmt], c_in=False)
    return fadd32(s, c)


def m_single_cin(a, b, c, fmt, frac=25, frac_out=23, min_exp=-133):
    """toolbox B200TC.m (fp8): one 32-term block with c in the alignment, truncation (paper Fig. 5e / tcgen05)."""
    return bfma(a, b, c, frac, frac_out, min_exp, EMIN[fmt], c_in=True)


def m_hopper(a, b, c, fmt):
    """tcemu PRESETS['hopper'] / toolbox H100TC.m fp8: 32-term block, 13 fraction bits, c in alignment."""
    return bfma(a, b, c, 13, 13, -132, EMIN[fmt], c_in=True)


def groups(kind, K=32):
    k = np.arange(K)
    if kind == "pairs":      # bytes 0-1 of each 32-bit register -> first HMMA (measured SASS)
        g1 = (k % 4) < 2
    elif kind == "pairs_rev":  # bytes 2-3 first (wrong order, for identifiability)
        g1 = (k % 4) >= 2
    elif kind == "contig":   # k < 16 first (no interleave)
        g1 = k < 16
    elif kind == "oddeven":  # alternating single elements
        g1 = (k % 2) == 0
    else:
        raise ValueError(kind)
    return np.where(g1)[0], np.where(~g1)[0]


def m_hmma_chain(a, b, c, fmt, kind="pairs", in_emin="fp16", frac=25, frac_out=23, min_exp=-133, late_c=True):
    """The SASS sequence on sm_90/sm_100: F2FP (fp8 -> fp16, exact), HMMA.16816 over group 1 with accumulator 0,
    HMMA.16816 over group 2 with group-1's fp32 result as accumulator (fp16 tensor-core model: 16 products + acc
    aligned, 25 fraction bits, truncation, fp32 result), then FADD c (fp32 RNE)."""
    g1, g2 = groups(kind)
    emin = EMIN[in_emin if in_emin != "fmt" else fmt]
    z = np.zeros_like(c)
    s1 = bfma(a[:, g1], b[:, g1], z, frac, frac_out, min_exp, emin, c_in=True)
    s2 = bfma(a[:, g2], b[:, g2], s1, frac, frac_out, min_exp, emin, c_in=True)
    return fadd32(s2, c) if late_c else s2
