"""Non-finite FP8 inputs: sanitize copies only when needed; nonfinite_patch equals the IEEE
float64 rule computed elementwise (no matrix product); NaN is canonical; end-to-end through scaled_mm."""
from __future__ import annotations

import pytest
import torch

from mmaemu.api import scaled_mm
from mmaemu.formats import decode, nonfinite_mask
from mmaemu.model import specials

E4, E5 = torch.float8_e4m3fn, torch.float8_e5m2


def test_nonfinite_mask_codes():
    c = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    assert torch.nonzero(nonfinite_mask(c, "e4m3"))[:, 0].tolist() == [0x7F, 0xFF]
    assert torch.nonzero(nonfinite_mask(c, "e5m2"))[:, 0].tolist() == [0x7C, 0x7D, 0x7E, 0x7F, 0xFC, 0xFD, 0xFE, 0xFF]
    v = decode(c, "e5m2")
    assert torch.isinf(v[0x7C]) and torch.isinf(v[0xFC]) and torch.isnan(v[0x7D])
    assert not torch.isinf(decode(c, "e4m3")).any()


def test_sanitize_copies_only_when_needed():
    x = torch.randint(0, 0x7E, (5, 7), dtype=torch.int32).to(torch.uint8)
    y, flag = specials.sanitize(x, "e4m3")
    assert y is x and flag is False
    x[2, 3] = 0xFF
    y, flag = specials.sanitize(x, "e4m3")
    assert flag is True and y is not x and y[2, 3].item() == 0 and x[2, 3].item() == 0xFF
    assert (y[x != 0xFF] == x[x != 0xFF]).all()


def _ieee_ref(a_u8, fa, b_u8, fb):
    """float64 IEEE value of sum_k a*b elementwise (no BLAS); finite positions -> 0 marker."""
    A, B = decode(a_u8, fa), decode(b_u8, fb)
    s = (A[:, None, :] * B[None, :, :]).sum(-1)
    return s


def _random_specials(M, N, K, fa, fb, g, p_nf=0.02, p_zero=0.2):
    a = torch.randint(0, 256, (M, K), generator=g, dtype=torch.int32).to(torch.uint8)
    b = torch.randint(0, 256, (N, K), generator=g, dtype=torch.int32).to(torch.uint8)
    for x, f in ((a, fa), (b, fb)):
        nf = nonfinite_mask(x, f)
        x[nf & (torch.rand(x.shape, generator=g) > p_nf * 10)] = 0x10     # keep only some non-finite codes
        x[torch.rand(x.shape, generator=g) < p_zero] = 0
    return a, b


@pytest.mark.parametrize("fa,fb", [("e4m3", "e4m3"), ("e5m2", "e4m3"), ("e4m3", "e5m2"), ("e5m2", "e5m2")])
def test_patch_equals_float64_rule(fa, fb):
    g = torch.Generator().manual_seed(len(fa + fb) * 7 + (fa == "e5m2") + 2 * (fb == "e5m2"))
    for trial in range(20):
        M, N, K = 13, 11, 48
        a, b = _random_specials(M, N, K, fa, fb, g)
        if trial % 4 == 0:                  # rows/cols of all-Inf with zeros elsewhere: Inf * 0 everywhere
            a[0] = 0x7C if fa == "e5m2" else 0x7F
            b[0, : K // 2] = 0
        acc = torch.full((M, N), 1.5)
        got = specials.nonfinite_patch(acc, a, b, fa, fb)
        ref = _ieee_ref(a, fa, b, fb)
        nf = ~torch.isfinite(ref)
        gb = got.view(torch.int32)
        # non-finite positions: canonical NaN or the signed Inf
        assert (torch.isnan(got) == torch.isnan(ref)).all()
        assert (gb[torch.isnan(ref)] == 0x7FFFFFFF).all()
        inf = torch.isinf(ref)
        assert (got[inf] == ref[inf].float()).all()
        # finite positions untouched
        assert (got[~nf] == 1.5).all()


def test_patch_returns_input_when_all_finite():
    a = torch.zeros(3, 16, dtype=torch.uint8)
    acc = torch.ones(3, 3)
    assert specials.nonfinite_patch(acc, a, a, "e4m3", "e4m3") is acc


def test_scaled_mm_nan_rows_and_columns():
    """Hardware rule (c3 recordings; B2): any NaN code in row m of A or column n of B makes (m, n) NaN
    (0x7FFFFFFF), even when multiplied by zero; Inf * 0 is NaN; Inf otherwise propagates with its sign."""
    a = torch.ones(16, 32).to(E5).view(torch.uint8).clone()
    b = torch.ones(16, 32).to(E4).view(torch.uint8).clone()
    a[0, 0] = 0x7C          # +Inf
    a[1, 0] = 0x7E          # NaN
    a[2, 0] = 0xFC          # -Inf
    a[3, 0], a[3, 1] = 0x7C, 0xFC   # +Inf and -Inf -> NaN
    b[5, 0] = 0x00          # column 5: Inf * 0 for rows 0, 2, 3
    b[7, 9] = 0xFF          # column 7: NaN everywhere
    one = torch.tensor(1.0)
    y = scaled_mm(a.view(E5), b.view(E4).t(), one, one, out_dtype=torch.float32, use_fast_accum=True, splitk=1)
    bits = y.view(torch.int32)
    assert y[0, 0].item() == float("inf") and y[2, 0].item() == -float("inf")
    assert (bits[1] == 0x7FFFFFFF).all() and (bits[3] == 0x7FFFFFFF).all() and (bits[:, 7] == 0x7FFFFFFF).all()
    assert bits[0, 5].item() == 0x7FFFFFFF and bits[2, 5].item() == 0x7FFFFFFF
    assert torch.isfinite(y[4:, :7]).all() and y[4, 0].item() == 32.0
    # fp8 outputs: Inf saturates, NaN is 0x7F (repro_fp8_out specials, RTX 4060)
    y8 = scaled_mm(a.view(E5), b.view(E4).t(), one, one, out_dtype=E4, use_fast_accum=True, splitk=1)
    assert y8.view(torch.uint8)[:3, 0].tolist() == [0x7E, 0x7F, 0xFE]
