"""FP8 formats: dtype <-> format names, exact decoding, special-value masks and constants.

Only the OCP formats that NVIDIA tensor cores implement are supported: E4M3 ("e4m3", ``torch.float8_e4m3fn``: no
infinities, NaN = S.1111.111) and E5M2 ("e5m2", ``torch.float8_e5m2``: IEEE-like, Inf = S.11111.00, NaN =
S.11111.{01,10,11}). The AMD "fnuz" variants are a different encoding and are rejected.
"""
from __future__ import annotations

import torch

__all__ = ["FP8_MAX", "CANONICAL_NAN_BITS", "EMIN", "MANT_BITS", "DTYPE_OF", "fmt_of", "decode", "nonfinite_mask",
           "is_fp8_dtype"]

FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}
CANONICAL_NAN_BITS = {torch.float32: 0x7FFFFFFF, torch.bfloat16: 0x7FFF, torch.float16: 0x7FFF,
                      torch.float8_e4m3fn: 0x7F, torch.float8_e5m2: 0x7F}
EMIN = {"e4m3": -6, "e5m2": -14}        # exponent of the smallest normal number; subnormals share it
MANT_BITS = {"e4m3": 3, "e5m2": 2}
DTYPE_OF = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
_FMT = {torch.float8_e4m3fn: "e4m3", torch.float8_e5m2: "e5m2"}
_FNUZ = tuple(getattr(torch, n) for n in ("float8_e4m3fnuz", "float8_e5m2fnuz") if hasattr(torch, n))


def is_fp8_dtype(dtype) -> bool:
    """True for torch.float8_e4m3fn and torch.float8_e5m2 (the formats this package models)."""
    return dtype in _FMT


def fmt_of(dtype) -> str:
    """'e4m3' for torch.float8_e4m3fn, 'e5m2' for torch.float8_e5m2; TypeError for anything else (incl. fnuz)."""
    try:
        return _FMT[dtype]
    except (KeyError, TypeError):
        pass
    if dtype in _FNUZ:
        raise TypeError(f"{dtype} (AMD fnuz encoding) is not supported; NVIDIA tensor cores use float8_e4m3fn / "
                        "float8_e5m2")
    raise TypeError(f"expected a float8_e4m3fn or float8_e5m2 dtype, got {dtype}")


def _check_fmt(fmt: str) -> str:
    if fmt not in DTYPE_OF:
        raise ValueError(f"unknown FP8 format {fmt!r}; expected 'e4m3' or 'e5m2'")
    return fmt


def decode(codes_u8: torch.Tensor, fmt: str) -> torch.Tensor:
    """FP8 codes (uint8) -> float64 values, exactly (NaN and Inf codes decode to NaN / Inf), on the codes' device."""
    _check_fmt(fmt)
    if codes_u8.dtype != torch.uint8:
        raise TypeError(f"codes must be uint8, got {codes_u8.dtype}")
    return codes_u8.view(DTYPE_OF[fmt]).to(torch.float64)


def nonfinite_mask(codes_u8: torch.Tensor, fmt: str) -> torch.Tensor:
    """Bool mask of the NaN (and, for E5M2, Inf) codes."""
    _check_fmt(fmt)
    if codes_u8.dtype != torch.uint8:
        raise TypeError(f"codes must be uint8, got {codes_u8.dtype}")
    if fmt == "e4m3":
        return (codes_u8 & 0x7F) == 0x7F
    return (codes_u8 & 0x7C) == 0x7C
