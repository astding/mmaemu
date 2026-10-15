"""mmaemu adapter for the vendored bring-up kits (tools/bringup/kits/*).

The audit kits imported the old package (`from tcemu import reference as R`) and called R.PRESETS, R.dot,
R.scaled_mm, R.gemm_cublas and R._nonfinite. This module provides the same calls on top of mmaemu, and never imports
tcemu:
- with the v0.1 model in place (mmaemu.model.dot / presets, mmaemu.scaled_mm, mmaemu.accumulate), those public
  APIs are used (preset parameters that match no built-in preset become a custom Preset object);
- otherwise, the pre-v0.1 mmaemu.reference (the tcemu prototype renamed, same numerics, no M1 fix) is used.
`backend()` says which one is active; every kit records it in its output.

usage in a kit:   from compat import R      (tools/bringup is put on sys.path by run_bringup.py)
"""
from __future__ import annotations

import types
import warnings

import torch

PRESET_PARAMS = {
    "ada": dict(nfma=16, frac=13, frac_out=13, min_exp=-132, late_c=False),
    "hopper": dict(nfma=32, frac=13, frac_out=13, min_exp=-132, late_c=False),
    "b200_mma_sync": dict(nfma=32, frac=25, frac_out=23, min_exp=-133, late_c=True),
}
PRESET_PARAMS["blackwell"] = PRESET_PARAMS["b200_mma_sync"]      # legacy alias
FP8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}


def _new_api():
    try:
        from mmaemu.model import chain, presets  # noqa: F401  (v0.1)
        import mmaemu
        if not callable(getattr(mmaemu, "scaled_mm", None)):
            return None
        return types.SimpleNamespace(chain=chain, presets=presets, mmaemu=mmaemu)
    except Exception:  # noqa: BLE001
        return None


def backend() -> str:
    return "mmaemu.model (frozen API)" if _new_api() is not None else "mmaemu.reference (pre-split legacy module)"


def _params(nfma=16, frac=13, frac_out=None, min_exp=-132, late_c=False):
    return dict(nfma=int(nfma), frac=int(frac), frac_out=int(frac if frac_out is None else frac_out),
                min_exp=int(min_exp), late_c=bool(late_c))


def dot(a, b, c=None, fmt="e4m3", **kw):
    """R.dot(a, b, c, fmt=..., nfma=, frac=, frac_out=, min_exp=, late_c=) -> float64 tensor of fp32 values."""
    p = _params(**kw)
    api = _new_api()
    a = torch.as_tensor(a).double()
    b = torch.as_tensor(b).double()
    c = None if c is None else torch.as_tensor(c).double()
    if api is not None:
        name = next((n for n, v in PRESET_PARAMS.items() if v == p and n != "blackwell"), None)
        if name is None:
            P = api.presets.Preset
            name = P(name="bringup-hypothesis", slow_chunk=None, maturity="custom",
                     validated_scope="runbook hypothesis (not validated)", evidence=(), **p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return api.chain.dot(a, b, c, preset=name, fmt=fmt).double()
    from mmaemu import reference as legacy
    return legacy.dot(a, b, c, fmt=fmt, **p)


def scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.float32, use_fast_accum=True, **kw):
    """Run the reference on the input device, unsplit, preserving epilogue options."""
    api = _new_api()
    if api is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return api.mmaemu.scaled_mm(a, b, scale_a, scale_b, out_dtype=out_dtype,
                                        use_fast_accum=use_fast_accum, splitk=1, backend="reference", **kw)
    from mmaemu import reference as legacy
    return legacy.scaled_mm(a.cpu(), b.cpu(), scale_a.cpu(), scale_b.cpu(), out_dtype=out_dtype,
                            use_fast_accum=use_fast_accum, **kw)


def gemm_cublas(A, B, fmt="e4m3", fast_accum=True, **kw):
    """R.gemm_cublas(A [M,K], B [N,K] float64 FP8 values) -> float64 accumulator (unsplit)."""
    api = _new_api()
    fa, fb = (fmt, fmt) if isinstance(fmt, str) else tuple(fmt)
    if api is not None:
        a8 = torch.as_tensor(A).to(FP8[fa])
        b8 = torch.as_tensor(B).to(FP8[fb])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return api.mmaemu.accumulate(a8, b8, use_fast_accum=bool(fast_accum), splitk=1,
                                         backend="reference").double()
    from mmaemu import reference as legacy
    return legacy.gemm_cublas(A, B, fmt=fmt, fast_accum=fast_accum, **kw)


R = types.SimpleNamespace(PRESETS=PRESET_PARAMS, dot=dot, scaled_mm=scaled_mm, gemm_cublas=gemm_cublas,
                          backend=backend)
