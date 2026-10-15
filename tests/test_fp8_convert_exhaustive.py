"""The fp32 -> fp8 output conversion of the epilogue (clamp to +-max, torch .to(), canonical NaN), checked on all
2^32 fp32 bit patterns against an integer round-to-nearest-even + saturation reference.

Slow (minutes): python -m pytest -q -m slow tests/test_fp8_convert_exhaustive.py"""
from __future__ import annotations

import time

import pytest
import torch

from mmaemu.model.epilogue import to_out_dtype

E4, E5 = torch.float8_e4m3fn, torch.float8_e5m2
SPEC = {E4: dict(mbits=3, bias=7, maxcode=0x7E), E5: dict(mbits=2, bias=15, maxcode=0x7B)}
CHUNK = 1 << 24


def ref_codes(bits: torch.Tensor, dtype) -> torch.Tensor:
    """fp32 bit patterns (int64 in [0, 2^32)) -> uint8 codes: RNE to the fp8 format, saturating (Inf too) to the
    largest finite code, NaN -> 0x7F. Integer arithmetic only."""
    s = SPEC[dtype]
    mb, bias, maxcode = s["mbits"], s["bias"], s["maxcode"]
    emin = 1 - bias
    sign = (bits >> 31) & 1
    e = (bits >> 23) & 0xFF
    m = bits & 0x7FFFFF
    nan = (e == 255) & (m != 0)
    inf = (e == 255) & (m == 0)
    S = torch.where(e > 0, m | 0x800000, m)                  # value = S * 2^E2
    E2 = torch.where(e > 0, e - 150, torch.full_like(e, -149))
    lg = torch.where(e > 0, e - 127, torch.full_like(e, -127))
    Q = torch.maximum(lg - mb, torch.full_like(lg, emin - mb))  # quantum exponent of the target
    shift = (Q - E2).clamp(0, 40)
    R = S >> shift
    rem = S & ((torch.ones_like(S) << shift) - 1)
    half = torch.where(shift > 0, torch.ones_like(S) << (shift - 1).clamp_min(0), torch.zeros_like(S))
    up = (shift > 0) & ((rem > half) | ((rem == half) & ((R & 1) == 1)))
    R = R + up.to(R.dtype)
    carry = R >= (1 << (mb + 1))
    R = torch.where(carry, R >> 1, R)
    Q = Q + carry.to(Q.dtype)
    normal = R >= (1 << mb)
    expf = torch.where(normal, Q + mb + bias, torch.zeros_like(Q))
    mant = torch.where(normal, R - (1 << mb), R)
    code = (expf << mb) | mant
    code = torch.where((code > maxcode) | inf, torch.full_like(code, maxcode), code)
    code = code | (sign << 7)
    code = torch.where(nan, torch.full_like(code, 0x7F), code)
    return code.to(torch.uint8)


def _spot():
    """A few hand-checked values."""
    vals = torch.tensor([0.0, -0.0, 448.0, 464.0, 480.0, 1e30, float("inf"), -float("inf"), 2.0 ** -9, 2.0 ** -10,
                         3 * 2.0 ** -10, 15.0, -0.1], dtype=torch.float32)
    return vals.view(torch.int32).to(torch.int64) & 0xFFFFFFFF


def test_reference_spot_values():
    b = _spot()
    assert ref_codes(b, E4).tolist() == [0x00, 0x80, 0x7E, 0x7E, 0x7E, 0x7E, 0x7E, 0xFE, 0x01, 0x00, 0x02, 0x57, 0x9D]
    assert ref_codes(b, E5).tolist()[:8] == [0x00, 0x80, 0x5F, 0x5F, 0x60, 0x7B, 0x7B, 0xFB]


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [E4, E5])
def test_all_fp32_patterns(dtype):
    t0 = time.monotonic()
    mismatches = 0
    first = []
    for start in range(0, 1 << 32, CHUNK):
        bits = torch.arange(start, start + CHUNK, dtype=torch.int64)
        x = bits.to(torch.int32).view(torch.float32)          # wraps to the same 32-bit patterns
        got = to_out_dtype(x, dtype).view(torch.uint8)
        want = ref_codes(bits, dtype)
        bad = got != want
        n = int(bad.sum())
        if n:
            mismatches += n
            if len(first) < 10:
                i = torch.nonzero(bad)[:10 - len(first), 0]
                first += [(hex(int(bits[j])), hex(int(got[j])), hex(int(want[j]))) for j in i]
    print(f"{dtype}: 2^32 patterns, {mismatches} mismatches, {time.monotonic() - t0:.0f} s")
    assert mismatches == 0, first
