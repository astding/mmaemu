"""Epilogue of torch._scaled_mm with tensor-wise scales (docs/semantics.md).

Measured on the reference profile (RTX 4060, cuBLASLt 12.8.3.14, torch 2.7.1; truth epilogue.*, c3.*, c4.epi*;
the epilogue findings of the tcemu audit, whose fixed models are verify-a1-reference-review-2/scripts/
v05_fixed_scaled_mm.py and c3-specials/scripts/fixed_model.py):

1. alpha = RN32(scale_a * scale_b).
2. y = RN32(acc * alpha + (+0)): an FFMA with a +0 addend (cuBLAS beta = 0). Implemented as the exact float64
   product plus 0.0, rounded once to fp32: an exact zero product gives +0 (never -0), an underflowing nonzero product
   keeps its sign, Inf * 0 is NaN. (A plain fp32 ``acc * alpha + 0.0`` is wrong: 9,282 counterexamples in the tcemu
   audit.)
3. Output: fp32 -> y; bf16 / fp16 -> RNE(y) (double rounding, verified on constructed ties); e4m3 / e5m2 -> y times
   scale_result in a second fp32 multiply if scale_result is given, then saturation to +-448 / +-57344 (Inf
   included, NaN passes), then RNE. scale_result is ignored for non-fp8 outputs.
4. NaN is canonical: fp32 0x7FFFFFFF, bf16 / fp16 0x7FFF, fp8 0x7F.
The result never depends on the contents of an out= buffer.

Every step is exact or a single IEEE rounding, so the result is identical on CPU and every GPU (unless the CPU
flushes subnormals to zero; mmaemu.api refuses to run then).
"""
from __future__ import annotations

import torch

from mmaemu.formats import CANONICAL_NAN_BITS, FP8_MAX

__all__ = ["apply", "to_out_dtype", "OUT_DTYPES"]

OUT_DTYPES = (torch.float32, torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2)
_FP8_MAX = {torch.float8_e4m3fn: FP8_MAX["e4m3"], torch.float8_e5m2: FP8_MAX["e5m2"]}
_IV = {4: torch.int32, 2: torch.int16, 1: torch.uint8}


def _canonical_nan(out: torch.Tensor) -> torch.Tensor:
    nan = torch.isnan(out) if out.dtype in (torch.float32, torch.bfloat16, torch.float16) else torch.isnan(out.float())
    if not bool(nan.any()):
        return out
    iv = _IV[out.element_size()]
    canon = torch.tensor(CANONICAL_NAN_BITS[out.dtype], dtype=torch.int64).to(iv).to(out.device)
    return torch.where(nan, canon, out.view(iv)).view(out.dtype)


def _rn32_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """RN32(x * y) for fp32 values: the float64 product of two fp32 values is exact; one rounding to fp32."""
    return (x.to(torch.float64) * y.to(torch.float64)).to(torch.float32)


def to_out_dtype(y: torch.Tensor, out_dtype: torch.dtype, scale_result: torch.Tensor | None = None) -> torch.Tensor:
    """Step 3-4: fp32 y -> out_dtype (scale_result and saturation for fp8 outputs; canonical NaN)."""
    if out_dtype not in OUT_DTYPES:
        raise TypeError(f"unsupported out_dtype {out_dtype}")
    if out_dtype in _FP8_MAX:
        if scale_result is not None:
            y = _rn32_mul(y, scale_result.to(y.device, torch.float32).reshape(()))
        mx = _FP8_MAX[out_dtype]
        y = y.clamp(-mx, mx)                         # saturation; torch.clamp propagates NaN
        out = y.to(out_dtype)                        # RNE (exhaustively checked: tests/test_fp8_convert_exhaustive.py)
    elif out_dtype == torch.float32:
        out = y
    else:
        out = y.to(out_dtype)                        # RNE from fp32: double rounding, as the hardware
    return _canonical_nan(out)


def apply(acc32: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor, *, out_dtype: torch.dtype,
          scale_result: torch.Tensor | None = None) -> torch.Tensor:
    """torch._scaled_mm epilogue for an fp32 accumulator, computed on acc32.device.

    scale_a, scale_b: one-element fp32 tensors (tensor-wise scales); scale_result: one-element fp32 or None (used
    only for fp8 outputs). Returns a new tensor of out_dtype."""
    if acc32.dtype != torch.float32:
        raise TypeError(f"acc32 must be float32, got {acc32.dtype}")
    dev = acc32.device
    sa = scale_a.to(dev, torch.float32).reshape(())
    sb = scale_b.to(dev, torch.float32).reshape(())
    alpha = _rn32_mul(sa, sb)
    y = (acc32.to(torch.float64) * alpha.to(torch.float64) + 0.0).to(torch.float32)   # fma(acc, alpha, +0)
    return to_out_dtype(y, out_dtype, scale_result)
