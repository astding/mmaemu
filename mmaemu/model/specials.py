"""Non-finite FP8 inputs (docs/semantics.md).

Hardware rule (RTX 4060, cuBLASLt 12.8.3.14; truth c3.*, epilogue.zero_nan_*): wherever the IEEE value of
sum_k a[m, k] * b[n, k] is not finite, the GEMM result is that value -- NaN for any NaN operand, Inf * 0, or +Inf and
-Inf terms together, else +-Inf -- and NaN is the canonical 0x7FFFFFFF. In particular any NaN code in row m of A or
column n of B makes output (m, n) NaN. Every output whose row or column holds a non-finite code is non-finite, so
the chain itself only ever needs the finite codes: :func:`sanitize` replaces the non-finite codes by +0 (a zero
product never takes part in the block maximum) and :func:`nonfinite_patch` writes the IEEE values afterwards.

The patch counts the Inf terms exactly with 0/1 indicator matrix products (integers, exact in fp32 for K < 2^24;
float64 beyond), restricted to the K columns and output rows/columns that hold non-finite codes. It never relies on
a BLAS computing Inf * 0, which some implementations skip.
"""
from __future__ import annotations

import torch

from mmaemu.formats import decode, nonfinite_mask

__all__ = ["sanitize", "nonfinite_patch"]

_ROW_BLOCK = 4096


def sanitize(codes_u8: torch.Tensor, fmt: str) -> tuple:
    """(codes with every NaN/Inf code replaced by 0x00, any_present). Copies only if non-finite codes are present."""
    m = nonfinite_mask(codes_u8, fmt)
    if not bool(m.any()):
        return codes_u8, False
    return torch.where(m, torch.zeros_like(codes_u8), codes_u8), True


def _classes(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """IEEE class of sum_k A[m, k] * B[n, k] for float64 values A [r, k], B [c, k] that each hold at least one
    non-finite value per affected row/column: fp32 [r, c] with NaN / +Inf / -Inf (finite entries: NaN never set;
    callers only use affected positions)."""
    ct = torch.float32 if A.shape[1] < (1 << 24) else torch.float64

    def f(x):
        return x.to(ct)
    inf = float("inf")
    a_pinf, a_ninf, a_zero = A == inf, A == -inf, A == 0
    b_pinf, b_ninf, b_zero = B == inf, B == -inf, B == 0
    a_pos, a_neg = A > 0, A < 0                      # NaN compares False
    b_pos, b_neg = B > 0, B < 0
    a_pf, a_nf = a_pos & ~a_pinf, a_neg & ~a_ninf    # finite nonzero
    infzero = f(a_pinf | a_ninf) @ f(b_zero).T + f(a_zero) @ f(b_pinf | b_ninf).T
    pos = f(a_pinf) @ f(b_pos).T + f(a_ninf) @ f(b_neg).T + f(a_pf) @ f(b_pinf).T + f(a_nf) @ f(b_ninf).T
    neg = f(a_pinf) @ f(b_neg).T + f(a_ninf) @ f(b_pos).T + f(a_pf) @ f(b_ninf).T + f(a_nf) @ f(b_pinf).T
    nan = torch.isnan(A).any(1)[:, None] | torch.isnan(B).any(1)[None, :] | (infzero > 0) | ((pos > 0) & (neg > 0))
    out = torch.where(pos > 0, torch.full_like(pos, inf), torch.full_like(pos, -inf)).to(torch.float32)
    canon = torch.tensor(0x7FFFFFFF, dtype=torch.int32, device=A.device).view(torch.float32)
    return torch.where(nan, canon, out)


def nonfinite_patch(acc32: torch.Tensor, a_u8: torch.Tensor, b_u8: torch.Tensor, fmt_a: str,
                    fmt_b: str) -> torch.Tensor:
    """acc32 [M, N] with the IEEE non-finite value written wherever sum_k a[m, k] * b[n, k] is not finite (a_u8
    [M, K], b_u8 [N, K] the ORIGINAL codes, before sanitize). NaN is canonical. Returns acc32 itself if nothing is
    non-finite, else a patched copy."""
    nfa, nfb = nonfinite_mask(a_u8, fmt_a), nonfinite_mask(b_u8, fmt_b)
    rows, cols = nfa.any(1), nfb.any(1)
    if not (bool(rows.any()) or bool(cols.any())):
        return acc32
    kk = torch.nonzero(nfa.any(0) | nfb.any(0))[:, 0]           # only these K columns can hold non-finite terms
    A = decode(a_u8[:, kk], fmt_a)
    B = decode(b_u8[:, kk], fmt_b)
    out = acc32.clone()
    r_idx = torch.nonzero(rows)[:, 0]
    for i in range(0, r_idx.numel(), _ROW_BLOCK):                 # affected rows, every column
        r = r_idx[i:i + _ROW_BLOCK]
        out[r] = _classes(A[r], B)
    c_idx = torch.nonzero(cols)[:, 0]
    if c_idx.numel():
        o_idx = torch.nonzero(~rows)[:, 0]                        # remaining rows, affected columns
        Bc = B[c_idx]
        for i in range(0, o_idx.numel(), _ROW_BLOCK):
            r = o_idx[i:i + _ROW_BLOCK]
            out[r[:, None], c_idx[None, :]] = _classes(A[r], Bc)
    return out
