"""Bitwise comparison on integer views.

Port of the tcemu audit's comparator (verify-a1-reference-review-5/scripts/v7_fixed_comparator.py). torch.equal
and == are not bit comparisons: they treat +0 and -0 as equal and NaN as never equal. Every bit claim in mmaemu goes
through ``assert_bitwise_equal``.

NaN policy:
- nan="bits" (default): NaN payloads must match exactly. mmaemu canonicalises NaN on every backend (fp32 0x7FFFFFFF,
  bf16/fp16 0x7FFF, fp8 0x7F) and so does the reference hardware.
- nan="mask": NaN positions must match, payloads are ignored. Allowed only against non-canonicalising sources
  (e.g. an IEEE float64 oracle), with an inline ``# nan=mask: <reason>`` comment at the call site.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["bits", "assert_bitwise_equal"]

_INT_VIEW = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
_MAX_REPORTED = 10


def _as_tensor(x, what: str) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(x))
    raise TypeError(f"{what} must be a torch.Tensor (or numpy.ndarray), got {type(x).__name__}")


def bits(t: torch.Tensor) -> torch.Tensor:
    """Integer view of `t` with the same element size (uint8 for 1-byte types, int16/int32/int64 otherwise); same
    device and shape, no copy. Complex tensors are refused."""
    t = _as_tensor(t, "t")
    if t.is_complex():
        raise TypeError(f"bits() does not support complex dtypes ({t.dtype})")
    try:
        iv = _INT_VIEW[t.element_size()]
    except KeyError:
        raise TypeError(f"bits(): unsupported element size {t.element_size()} ({t.dtype})") from None
    return t if t.dtype == iv else t.view(iv)


def _float32(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float32) if t.element_size() == 1 else t


def assert_bitwise_equal(actual, expected, *, nan: str = "bits", msg: str = "") -> None:
    """Assert that `actual` and `expected` have the same dtype, the same shape and identical bits.

    The comparison runs on CPU (either tensor may live on any device). On failure the AssertionError reports the
    mismatch count per class -- zero_sign (+0 vs -0), nan_payload (both NaN, different bits), value (everything
    else, including NaN vs non-NaN) -- and the first 10 mismatching elements with their values and bit patterns."""
    if nan not in ("bits", "mask"):
        raise ValueError(f"nan must be 'bits' or 'mask', got {nan!r}")
    a = _as_tensor(actual, "actual")
    e = _as_tensor(expected, "expected")
    head = f"{msg}: " if msg else ""
    if a.dtype != e.dtype:
        raise AssertionError(f"{head}dtype mismatch: actual {a.dtype}, expected {e.dtype}")
    if a.shape != e.shape:
        raise AssertionError(f"{head}shape mismatch: actual {tuple(a.shape)}, expected {tuple(e.shape)}")
    a, e = a.detach().cpu(), e.detach().cpu()
    ab, eb = bits(a), bits(e)
    diff = ab != eb
    if a.is_floating_point():
        af, ef = _float32(a), _float32(e)
        both_nan = torch.isnan(af) & torch.isnan(ef)
        if nan == "mask":
            diff &= ~both_nan
        zero_sign = diff & (af == 0) & (ef == 0)
        nan_payload = diff & both_nan
    else:
        zero_sign = nan_payload = torch.zeros_like(diff)
    n = int(diff.sum())
    if n == 0:
        return
    value = diff & ~zero_sign & ~nan_payload
    counts = dict(zero_sign=int(zero_sign.sum()), nan_payload=int(nan_payload.sum()), value=int(value.sum()))
    flat = torch.nonzero(diff.reshape(-1))[:_MAX_REPORTED, 0].tolist()
    width = 2 * a.element_size()
    mask = (1 << (8 * a.element_size())) - 1
    af_flat, ef_flat = (_float32(a).reshape(-1), _float32(e).reshape(-1)) if a.is_floating_point() \
        else (a.reshape(-1), e.reshape(-1))
    ab_flat, eb_flat = ab.reshape(-1), eb.reshape(-1)
    zs_flat, np_flat = zero_sign.reshape(-1), nan_payload.reshape(-1)
    lines = []
    for i in flat:
        idx = tuple(int(x) for x in np.unravel_index(i, tuple(a.shape))) if a.dim() else ()
        kind = "zero_sign" if zs_flat[i] else "nan_payload" if np_flat[i] else "value"
        lines.append(f"  {idx}: actual {af_flat[i].item()!r} [0x{int(ab_flat[i]) & mask:0{width}x}]  "
                     f"expected {ef_flat[i].item()!r} [0x{int(eb_flat[i]) & mask:0{width}x}]  {kind}")
    more = f"\n  ... {n - len(flat)} more" if n > len(flat) else ""
    raise AssertionError(
        f"{head}bitwise mismatch in {n} of {a.numel()} elements "
        f"(zero_sign={counts['zero_sign']}, nan_payload={counts['nan_payload']}, value={counts['value']}) "
        f"[dtype {a.dtype}, shape {tuple(a.shape)}, nan={nan!r}]; first {len(flat)}:\n" + "\n".join(lines) + more)
