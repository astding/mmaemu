"""CPU test (proposed fix for F6): recorded discriminating epilogue cases (production-shape cuBLAS kernels,
fast and slow accumulation, bf16 and fp16 outputs) replayed against tcemu.reference.scaled_mm. Each case is
self-checked to be discriminating: the alternative (single rounding, or (acc*sa)*sb) must disagree with the
hardware on at least one recorded output."""
import os
from pathlib import Path

import pytest
import torch
from tools.bringup.checks import same_bits

import sys as _sys
from pathlib import Path as _P
_d = _P(__file__).resolve().parent
while not (_d / "compat.py").exists():   # tools/bringup (compat.py)
    _d = _d.parent
_sys.path.insert(0, str(_d))
from compat import R  # noqa: E402  (vendored: was `from tcemu import reference as R`)

DATA = Path(os.environ.get("EPI_TIES", Path(__file__).parent / "data" / "epilogue_ties_rtx4060.pt"))
REC = torch.load(DATA, weights_only=False) if DATA.exists() else dict(cases=[], b=None)
PE = {torch.bfloat16: (8, -126), torch.float16: (11, -14)}


def _alt(kind, acc, sa, sb, out):
    if kind == "single":
        x = acc.double() * (sa * sb).double()
        e = (torch.frexp(x)[1] - 1).double(); p, emin = PE[out]
        q = torch.exp2(torch.clamp(e - (p - 1), min=emin - (p - 1)))
        return torch.where(x == 0, x, torch.round(x / q) * q).float().to(out)
    return ((acc * sa) * sb).to(out)


@pytest.mark.parametrize("i", range(len(REC["cases"])))
def test_epilogue_tie(i):
    c = REC["cases"][i]
    out = getattr(torch, c["out_dtype"])
    a = c["a"].view(torch.float8_e4m3fn); b = REC["b"].view(torch.float8_e4m3fn)
    em = R.scaled_mm(a, b.t(), c["sa"], c["sb"], out_dtype=out, use_fast_accum=c["fast"])
    hw = c["out"]
    acc = R.gemm_cublas(a, b, fast_accum=c["fast"]).float()
    assert bool((_alt(c["kind"], acc, c["sa"], c["sb"], out) != hw).any()), "case is not discriminating"
    assert same_bits(em.view(torch.int16), hw.view(torch.int16))
