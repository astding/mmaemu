"""CUDA backends of mmaemu: prebuilt-fatbin / NVRTC kernels loaded through the driver API (ctypes).

Importing this package loads nothing: libcuda and libnvrtc are opened on first use (driver.py, nvrtc.py).
  spec.py      KernelSpec, GateResult, GATES, register_gate, PROBE_MODULES (the kernel contract)
  driver.py    ctypes bindings to the CUDA driver API
  nvrtc.py     ctypes bindings to NVRTC
  modules.py   per-(device, context) module cache: fatbin, then NVRTC
  kernels.py   KernelSpecs v1 / v4 (preset ada), h1 / h4 (preset hopper) and the standard-ABI ChainFn wrapper
  selftest.py  per-device gates ("smoke") and the verdict cache
  fast/        gated kernels v9 / tc3 and their gates (optional)
Selection between kernels and the reference backend: mmaemu.backends.registry. See docs/backends.md.
"""
from __future__ import annotations

__all__ = ["CudaKernelBackend"]


class CudaKernelBackend:
    """Backend (mmaemu.backends.Backend protocol) running one CUDA kernel, named "cuda:<kernel>"."""

    def __init__(self, spec, source: str | None = None):
        self.spec = spec
        self.name = f"cuda:{spec.name}"
        self.source = source  # "fatbin" | "nvrtc" (how the module was loaded when selected)

    def supports(self, req) -> str | None:
        import torch

        if (req.fmt_a, req.fmt_b) not in tuple(tuple(f) for f in self.spec.formats):
            return f"format pair {req.fmt_a} x {req.fmt_b} not supported by {self.name}"
        if req.preset != self.spec.preset:
            return f"preset {req.preset!r} not implemented by {self.name}"
        if torch.device(req.device).type != "cuda":
            return f"{self.name} needs CUDA tensors"
        return None

    def chain_fn(self, req):
        from mmaemu.backends.cuda.kernels import chain_fn_for

        why = self.supports(req)
        if why is not None:
            raise ValueError(why)
        return chain_fn_for(self.spec, req)

    def __repr__(self) -> str:
        return f"CudaKernelBackend({self.name}, source={self.source})"
