"""mmaemu adapter for the vendored bring-up kits (tools/bringup/kits/*).

The audit kits imported the old package (`from tcemu import reference as R`) and called R.PRESETS, R.dot,
R.scaled_mm and R.gemm_cublas. This module provides those calls on mmaemu's v0.1 API (mmaemu.model chain / presets,
mmaemu.scaled_mm, mmaemu.accumulate; preset parameters that match no built-in preset become a custom Preset object)
and never imports tcemu. `backend()` names the API; every kit records it in its output. Without mmaemu's v0.1 API
every call raises ImportError.

usage in a kit:   from compat import R      (each vendored kit puts tools/bringup on sys.path itself, by searching
                                             its parent directories for compat.py)
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


def _api():
    api = _new_api()
    if api is None:
        raise ImportError("the bring-up kits need mmaemu's v0.1 API (mmaemu.model, mmaemu.scaled_mm)")
    return api


def backend() -> str:
    _api()
    return "mmaemu.model (frozen API)"


def _params(nfma=16, frac=13, frac_out=None, min_exp=-132, late_c=False):
    return dict(nfma=int(nfma), frac=int(frac), frac_out=int(frac if frac_out is None else frac_out),
                min_exp=int(min_exp), late_c=bool(late_c))


def dot(a, b, c=None, fmt="e4m3", **kw):
    """R.dot(a, b, c, fmt=..., nfma=, frac=, frac_out=, min_exp=, late_c=) -> float64 tensor of fp32 values."""
    p = _params(**kw)
    api = _api()
    a = torch.as_tensor(a).double()
    b = torch.as_tensor(b).double()
    c = None if c is None else torch.as_tensor(c).double()
    name = next((n for n, v in PRESET_PARAMS.items() if v == p and n != "blackwell"), None)
    if name is None:
        P = api.presets.Preset
        name = P(name="bringup-hypothesis", slow_chunk=None, maturity="custom",
                 validated_scope="runbook hypothesis (not validated)", evidence=(), **p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return api.chain.dot(a, b, c, preset=name, fmt=fmt).double()


def scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.float32, use_fast_accum=True, **kw):
    """Run the reference on the input device, unsplit, preserving epilogue options."""
    api = _api()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return api.mmaemu.scaled_mm(a, b, scale_a, scale_b, out_dtype=out_dtype,
                                    use_fast_accum=use_fast_accum, splitk=1, backend="reference", **kw)


def gemm_cublas(A, B, fmt="e4m3", fast_accum=True, **kw):
    """R.gemm_cublas(A [M,K], B [N,K] float64 FP8 values) -> float64 accumulator (unsplit)."""
    api = _api()
    fa, fb = (fmt, fmt) if isinstance(fmt, str) else tuple(fmt)
    a8 = torch.as_tensor(A).to(FP8[fa])
    b8 = torch.as_tensor(B).to(FP8[fb])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return api.mmaemu.accumulate(a8, b8, use_fast_accum=bool(fast_accum), splitk=1, backend="reference").double()


R = types.SimpleNamespace(PRESETS=PRESET_PARAMS, dot=dot, scaled_mm=scaled_mm, gemm_cublas=gemm_cublas,
                          backend=backend)
