# Numerical model of NVIDIA FP8 tensor-core inner products (block FMA). Translated and adapted from the MATLAB Tensor
# Core toolbox (AdaTC.m, Generic_BFMA_TC.m):
#   Copyright (c) 2025, Faizan A. Khattak and Mantas Mikaitis. BSD 2-Clause License (see NOTICE and
#   LICENSES/BSD-2-Clause-MATLAB-tensor-core.txt).  https://github.com/north-numerical-computing/MATLAB-tensor-core
#   F. A. Khattak, M. Mikaitis, "Accurate Models of NVIDIA Tensor Cores", arXiv:2512.07004.
# Changes against the toolbox: exact float64 evaluation in torch, the hardware-measured +0 rule for zero block results
# (c2-M1), canonical NaN, explicit format handling, tiling.
"""Instruction-level FP8 tensor-core chain model (docs/semantics.md).

One block-FMA step (preset ``ada``: 16 products, early accumulator):

    d = sum_{i < nfma} a_i * b_i  +  c

- zero products are excluded from the maximum-exponent computation;
- input exponents are clamped at the format's emin (E4M3 -6, E5M2 -14), so subnormal inputs keep a denormalised
  significand;
- the products and c are aligned to the block's maximum exponent E (clamped below at ``min_exp``) and each term is
  truncated toward zero to ``frac`` fraction bits below 2^E;
- the terms are summed exactly and the sum is truncated to ``frac_out`` fraction bits relative to its own exponent;
- an exactly-zero or truncated-to-zero block result is +0 (RTX 4060 hardware rule c2-M1; the toolbox gives -0 in
  rare cases);
- NaN results are the canonical 0x7FFFFFFF.

A length-K dot product applies this step to consecutive blocks of ``nfma`` elements, feeding the result back as c
(the ragged tail is zero padded, which leaves the result unchanged). Presets with ``late_c`` (``b200_mma_sync``)
align the products only and add c afterwards with an fp32 round-to-nearest-even addition.

Every operation is exact in float64 (products of FP8 values, power-of-two scalings, truncations, sums of integers
< 2^53, frexp-based exponents), so the result does not depend on the device or on evaluation order: CPU and any
GPU running this code agree bit for bit.

Scope: this module is the instruction-level model. GEMM-level behaviour of cuBLAS (slow accumulation, split-K) is in
:mod:`mmaemu.model.plan`; the epilogue (alpha, output conversion) is in :mod:`mmaemu.model.epilogue`.
"""
from __future__ import annotations

import torch

from mmaemu.formats import DTYPE_OF, EMIN, decode, fmt_of, is_fp8_dtype, nonfinite_mask
from mmaemu.model.presets import Preset, get_preset

__all__ = ["dot", "chain_gemm", "bfma", "TEMP_BYTES"]

_NEG = -100000.0          # exponent of zero (excluded from every maximum)
_NEG16 = -1000            # same, in the int16 exponent tables of chain_gemm (NEG16 + NEG16 fits in int16)
TEMP_BYTES = 160 << 20    # chain_gemm's per-block temporaries stay below this (all temporaries <= 256 MB)
_CANON_NAN32 = 0x7FFFFFFF


# --------------------------------------------------------------------------------------------------------------------
# exact helpers
def _exp(x: torch.Tensor, emin: float | None = None) -> torch.Tensor:
    """floor(log2|x|) for x != 0 (optionally clamped below at emin); _NEG for x == 0.

    frexp is exact on every device (log2 is not: CUDA's float64 log2 can return 2.9999... for 8.0)."""
    e = (torch.frexp(x)[1] - 1).to(x.dtype)
    if emin is not None:
        e = e.clamp_min(emin)
    return torch.where(x != 0, e, torch.full_like(x, _NEG))


def _pow2(e: torch.Tensor) -> torch.Tensor:
    """2**e (float64) for integer-valued e in the normal range, built from the bit pattern (exact on every device)."""
    return ((e.to(torch.int64) + 1023) << 52).view(torch.float64)


def _pow2_f32(e: torch.Tensor) -> torch.Tensor:
    """2**e (float32) for integer-valued e in [-126, 127], built from the bit pattern."""
    return ((e.to(torch.int32) + 127) << 23).view(torch.float32)


def _canon_nan32(d: torch.Tensor) -> torch.Tensor:
    """fp32 tensor with every NaN replaced by the canonical 0x7FFFFFFF."""
    nan = torch.isnan(d)
    if not bool(nan.any()):
        return d
    canon = torch.tensor(_CANON_NAN32, dtype=torch.int32, device=d.device).view(torch.float32)
    return torch.where(nan, canon, d)


# --------------------------------------------------------------------------------------------------------------------
# readable reference step (port of the audit's reference_fixed.py = tcemu 5b0ac7b + M1)
def bfma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, p: Preset, fmt=("e4m3", "e4m3")) -> torch.Tensor:
    """One block-FMA step. a, b: [..., n] float64 holding FP8 values of formats ``fmt`` (str or (fmt_a, fmt_b));
    c: [...] float64 holding fp32 values. Returns float64 holding the fp32 result."""
    fa, fb = (fmt, fmt) if isinstance(fmt, str) else fmt
    frac, frac_out, min_exp = p.frac, p.frac_out, p.min_exp
    prod = a * b
    ep = torch.where(prod != 0, _exp(a, EMIN[fa]) + _exp(b, EMIN[fb]), torch.full_like(prod, _NEG))
    if p.late_c:
        E = ep.amax(-1).clamp_min(min_exp)
        q, iq = _pow2(E - frac), _pow2(frac - E)
        d = torch.trunc(prod * iq[..., None]).sum(-1) * q
        ed = _exp(d).clamp_min(-126)
        d = torch.where(d != 0, torch.trunc(d * _pow2(frac_out - ed)) * _pow2(ed - frac_out), d)
        return (d.float() + c.float()).double()  # IEEE fp32 addition (RNE): exact and device independent
    E = torch.maximum(ep.amax(-1), _exp(c)).clamp_min(min_exp)
    q, iq = _pow2(E - frac), _pow2(frac - E)
    s = torch.trunc(prod * iq[..., None]).sum(-1) + torch.trunc(c * iq)
    d = s * q
    ed = _exp(d).clamp_min(-126)  # truncate to frac_out fraction bits relative to the result's own exponent
    out = torch.where(d != 0, torch.trunc(d * _pow2(frac_out - ed)) * _pow2(ed - frac_out), d)
    return torch.where(out == 0, torch.zeros_like(out), out)  # c2-M1: zero block results are +0


def _dot_values(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor | None, p: Preset, fmt) -> torch.Tensor:
    K = a.shape[-1]
    if K % p.nfma:
        pad = p.nfma - K % p.nfma
        a = torch.nn.functional.pad(a, (0, pad))
        b = torch.nn.functional.pad(b, (0, pad))
        K += pad
    acc = torch.zeros(a.shape[:-1], dtype=torch.float64, device=a.device) if c is None else c
    for k0 in range(0, K, p.nfma):
        acc = bfma(a[..., k0:k0 + p.nfma], b[..., k0:k0 + p.nfma], acc, p, fmt)
    return acc


def _as_values(x: torch.Tensor, fmt: str | None, name: str) -> tuple:
    """(float64 values, format) of a dot() operand; validates the B13 rules."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x).__name__}")
    if is_fp8_dtype(x.dtype):
        f = fmt_of(x.dtype)
        if fmt is not None and fmt != f:
            raise ValueError(f"{name} has dtype {x.dtype} but fmt says {fmt!r}")
        return x.to(torch.float64), f
    if x.dtype == torch.float64:
        if fmt is None:
            raise TypeError(f"{name} is float64: pass fmt= ('e4m3' / 'e5m2' or a pair) so its FP8 format is explicit; "
                            "it is inferred only from float8 dtypes")
        if fmt not in DTYPE_OF:
            raise ValueError(f"unknown FP8 format {fmt!r}")
        rt = x.to(DTYPE_OF[fmt]).to(torch.float64)
        bad = ~((rt == x) | (torch.isnan(rt) & torch.isnan(x)))
        if bool(bad.any()):
            v = x[bad].flatten()[0].item()
            raise ValueError(f"{name} holds values that are not {fmt} values (e.g. {v!r})")
        return x, fmt
    raise TypeError(f"{name}: expected a float8_e4m3fn / float8_e5m2 tensor (format inferred) or a float64 tensor of "
                    f"FP8 values with fmt=, got {x.dtype}")


def dot(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor | None = None, *, preset="ada", fmt=None) -> torch.Tensor:
    """Instruction-level chain: d = chain over the last dimension of a*b, starting from c (default +0).

    a, b: same shape [..., K]; float8 tensors (format inferred from the dtype, each operand separately) or float64
    tensors holding FP8 values, which require ``fmt`` ('e4m3', 'e5m2' or a pair (fmt_a, fmt_b)). There is no
    broadcasting: a.shape must equal b.shape (ValueError otherwise; expand explicitly).
    c: None, or fp32 (or float64 holding fp32 values) of shape a.shape[:-1].
    Returns fp32 of shape a.shape[:-1] on a's device; NaN results are the canonical 0x7FFFFFFF.

    Special values follow IEEE float64 arithmetic of the model (NaN/Inf inputs or c propagate)."""
    p = get_preset(preset)
    if fmt is None or isinstance(fmt, str):
        fa_req = fb_req = fmt
    else:
        if len(fmt) != 2:
            raise ValueError("fmt must be a format name or a pair (fmt_a, fmt_b)")
        fa_req, fb_req = fmt
    av, fa = _as_values(a, fa_req, "a")
    bv, fb = _as_values(b, fb_req, "b")
    if av.device != bv.device:
        raise ValueError(f"a and b are on different devices ({av.device}, {bv.device})")
    if av.dim() < 1 or tuple(av.shape) != tuple(bv.shape):
        raise ValueError(f"a and b must have the same shape [..., K] (no broadcasting); got {tuple(a.shape)} and "
                         f"{tuple(b.shape)}")
    if c is not None:
        if not isinstance(c, torch.Tensor):
            raise TypeError("c must be a tensor or None")
        if tuple(c.shape) != tuple(av.shape[:-1]):
            raise ValueError(f"c must have shape {tuple(av.shape[:-1])}, got {tuple(c.shape)}")
        if c.dtype == torch.float32:
            c = c.to(torch.float64)
        elif c.dtype == torch.float64:
            if bool(((c.float().double() != c) & ~torch.isnan(c)).any()):
                raise ValueError("c holds values that are not fp32 values")
        else:
            raise TypeError(f"c must be float32 (or float64 holding fp32 values), got {c.dtype}")
        c = c.to(av.device)
    return _canon_nan32(_dot_values(av, bv, c, p, (fa, fb)).float())


# --------------------------------------------------------------------------------------------------------------------
# GEMM-shaped chain (the reference backend's ChainFn)
#
# chain_gemm evaluates the same block step as bfma() but restructured for speed; it is bitwise equal to dot() for
# chains that start from +0 over finite FP8 codes (tests/test_model_chain.py checks this on adversarial data):
#  - products of two FP8 values have <= 8 significant bits and lie in [2^-32, 2^32): exact in fp32;
#  - block maxima of the clamped product exponents are computed for a whole K chunk at once (int16, +127 bias);
#  - trunc(p * 2^(frac - E)) is one fused torch.div(p, 2^(E - frac), rounding_mode="trunc") (exact: division by a
#    power of two, then truncation);
#  - every accumulator value is a multiple of 2^(-28 - frac) (the finest alignment grid of any nonzero product), so
#    it is a normal fp32 number or zero. Its exponent is read from the fp32 bits, E is floored at -29 - frac (below
#    that every term is zero, so the floor cannot change a result), and truncation to frac_out fraction bits is a
#    mask on the mantissa bits; zero results become +0 (c2-M1) through "+ 0.0".
# Sums stay exact: early-c presets with frac <= 13 sum < 2^24 in fp32; otherwise sums are taken in float64.
_LUT_CACHE: dict = {}


def _luts(fmt: str, device: torch.device) -> tuple:
    """(values fp32 [256], clamped exponents int16 [256] with _NEG16 for zero codes) on device."""
    key = (fmt, str(device))
    t = _LUT_CACHE.get(key)
    if t is None:
        codes = torch.arange(256, dtype=torch.int32).to(torch.uint8)
        v64 = decode(codes, fmt)
        v64 = torch.where(torch.isfinite(v64), v64, torch.zeros_like(v64))   # non-finite codes are refused before use
        e = _exp(v64, EMIN[fmt])
        e = torch.where(v64 != 0, e, torch.full_like(e, float(_NEG16)))
        t = (v64.to(torch.float32).to(device), e.to(torch.int16).to(device))
        _LUT_CACHE[key] = t
    return t


def _tile_plan(M: int, N: int, L: int, nfma: int, device: torch.device) -> tuple:
    """(tm, tn, kc): output tile and K chunk such that the chunk temporaries stay below TEMP_BYTES."""
    if device.type == "cpu":
        target, tn_max, kc_target = 1 << 18, 1024, 64   # few, large ops: robust to OpenMP barrier costs under load
    else:
        target, tn_max, kc_target = 1 << 20, 1024, 32   # big tiles: few kernel launches
    tm = max(1, min(M, target // max(1, min(N, tn_max))))
    tn = max(1, min(N, target // tm))                    # small M: wider tiles, fewer launches
    kc = max(nfma, (max(kc_target, nfma) // nfma) * nfma)
    kc = min(kc, -(-L // nfma) * nfma)

    def nbytes(tm_, tn_, kc_):
        # P fp32 + exponent sums int16 (chunk) + block maxima + per-block quotient fp32 + [tm, tn] scalars
        return tm_ * tn_ * (6 * kc_ + 2 * (kc_ // nfma) + 4 * nfma + 64) + 14 * (tm_ + tn_) * kc_
    while nbytes(tm, tn, kc) > TEMP_BYTES and kc > nfma:
        kc -= nfma
    while nbytes(tm, tn, kc) > TEMP_BYTES and (tm > 1 or tn > 1):
        if tm >= tn:
            tm = max(1, tm // 2)
        else:
            tn = max(1, tn // 2)
    return tm, tn, kc


def _chain_chunk(va, eaB, vb, eb, p: Preset, acc: torch.Tensor, floor_b: int, f32_sums: bool) -> torch.Tensor:
    """Continue the chain over one K chunk. va [tm, L] fp32, eaB [tm, L] int16 (clamped exponent + 127), vb [tn, L],
    eb [tn, L] int16, L a multiple of nfma; acc fp32 [tm, tn]. Returns the new acc (fp32)."""
    tm, L = va.shape
    tn = vb.shape[0]
    nf, frac = p.nfma, p.frac
    nb = L // nf
    # block-major products and exponent maxima: [nb, tm, tn, nf] and [nb, tm, tn]
    va4 = va.view(tm, nb, nf).transpose(0, 1)[:, :, None, :]
    vb4 = vb.view(tn, nb, nf).transpose(0, 1)[:, None, :, :]
    P = va4 * vb4
    EPb = (eaB.view(tm, nb, nf).transpose(0, 1)[:, :, None, :] + eb.view(tn, nb, nf).transpose(0, 1)[:, None, :, :])
    # fp32 bit patterns of 2^(max(Ep, floor) - frac) for every block: [nb, tm, tn] int32
    EPS = (EPb.amax(-1).to(torch.int32).clamp_min(floor_b) - frac) << 23
    mask32 = ~((1 << (23 - p.frac_out)) - 1)
    mask64 = ~((1 << (52 - p.frac_out)) - 1)
    fbits = frac << 23
    for j in range(nb):
        if p.late_c:
            scale = EPS[j].view(torch.float32)                              # 2^(E - frac), a normal fp32
        else:
            # 2^(ec - frac) from the accumulator's exponent bits (acc = +0 gives a negative int: never the max)
            acc_s = (acc.view(torch.int32) & 0x7F800000) - fbits
            scale = torch.maximum(EPS[j], acc_s).view(torch.float32)        # 2^(E - frac), E = max(Ep, ec, floor)
        t = torch.div(P[j], scale[..., None], rounding_mode="trunc")
        if f32_sums:
            s = t.sum(-1)
            if not p.late_c:
                s = s + torch.div(acc, scale, rounding_mode="trunc")
            d = s * scale                                                    # exact (|s| < 2^24)
            d = (d.view(torch.int32) & mask32).view(torch.float32)          # truncate to frac_out fraction bits
        else:
            s = t.sum(-1, dtype=torch.float64)
            sc = scale.to(torch.float64)
            if not p.late_c:
                s = s + torch.div(acc.to(torch.float64), sc, rounding_mode="trunc")
            d = s * sc
            d = (d.view(torch.int64) & mask64).view(torch.float64).to(torch.float32)   # exact after the mask
        if p.late_c:
            acc = d + acc                                                    # fp32 round-to-nearest-even addition
        else:
            acc = d + 0.0                                                    # c2-M1: -0 -> +0
    return acc


def chain_gemm(a_codes_u8: torch.Tensor, b_codes_u8: torch.Tensor, *, fmt_a: str, fmt_b: str,
               preset="ada") -> torch.Tensor:
    """C[m, n] = chain over l of a[m, l] * b[n, l], starting from +0.

    a_codes_u8 [M, L], b_codes_u8 [N, L]: uint8 FP8 codes of formats fmt_a / fmt_b, all finite (non-finite codes are
    handled by :mod:`mmaemu.model.specials` before this is called; ValueError otherwise), L % 16 == 0.
    Returns fp32 [M, N] on the inputs' device. Temporaries stay below :data:`TEMP_BYTES` (tiling over M, N and K).
    Bitwise equal to :func:`dot` on the decoded values."""
    p = get_preset(preset)
    if a_codes_u8.dtype != torch.uint8 or b_codes_u8.dtype != torch.uint8:
        raise TypeError("chain_gemm takes uint8 FP8 codes")
    if a_codes_u8.dim() != 2 or b_codes_u8.dim() != 2 or a_codes_u8.shape[1] != b_codes_u8.shape[1]:
        raise ValueError(f"expected a [M, L] and b [N, L]; got {tuple(a_codes_u8.shape)} and {tuple(b_codes_u8.shape)}")
    if a_codes_u8.device != b_codes_u8.device:
        raise ValueError("a and b must be on the same device")
    M, L = a_codes_u8.shape
    N = b_codes_u8.shape[0]
    if L % 16:
        raise ValueError(f"L = {L} is not a multiple of 16 (the plan zero-pads; see mmaemu.model.plan)")
    dev = a_codes_u8.device
    out = torch.zeros(M, N, dtype=torch.float32, device=dev)
    if M == 0 or N == 0 or L == 0:
        return out
    if bool(nonfinite_mask(a_codes_u8, fmt_a).any()) or bool(nonfinite_mask(b_codes_u8, fmt_b).any()):
        raise ValueError("chain_gemm requires finite codes (sanitise with mmaemu.model.specials first)")
    if L % p.nfma:
        pad = p.nfma - L % p.nfma
        a_codes_u8 = torch.nn.functional.pad(a_codes_u8, (0, pad))
        b_codes_u8 = torch.nn.functional.pad(b_codes_u8, (0, pad))
        L += pad
    va_l, ea_l = _luts(fmt_a, dev)
    vb_l, eb_l = _luts(fmt_b, dev)
    eaB_l = ea_l + 127                                         # bias folded into A's exponent table
    floor_b = max(p.min_exp, -29 - p.frac) + 127                # see the comment above _LUT_CACHE
    f32_sums = (not p.late_c) and p.frac <= 13 and p.nfma <= 64
    tm, tn, kc = _tile_plan(M, N, L, p.nfma, dev)
    for m0 in range(0, M, tm):
        a_t = a_codes_u8[m0:m0 + tm]
        for n0 in range(0, N, tn):
            b_t = b_codes_u8[n0:n0 + tn]
            acc = torch.zeros(a_t.shape[0], b_t.shape[0], dtype=torch.float32, device=dev)
            for k0 in range(0, L, kc):
                ai = a_t[:, k0:k0 + kc].to(torch.int64)
                bi = b_t[:, k0:k0 + kc].to(torch.int64)
                acc = _chain_chunk(va_l[ai], eaB_l[ai], vb_l[bi], eb_l[bi], p, acc, floor_b, f32_sums)
            out[m0:m0 + tm, n0:n0 + tn] = acc
    return out
