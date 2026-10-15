"""The reference backend: the exact float64/fp32 torch model (mmaemu.model.chain.chain_gemm) run on the tensors'
own device (the CPU or a CUDA device torch supports). It supports every format pair and preset; it is the fallback of
every other backend and the oracle the CUDA kernels are tested against."""
from __future__ import annotations

import torch

from mmaemu.model.chain import chain_gemm

__all__ = ["ReferenceBackend", "REFERENCE"]


class ReferenceBackend:
    """Implements the mmaemu.backends.Backend protocol."""

    name = "reference"

    def supports(self, req) -> str | None:
        return None

    def chain_fn(self, req):
        fmt_a, fmt_b, preset = req.fmt_a, req.fmt_b, req.preset
        want = torch.device(req.device) if req.device is not None else None

        def fn(a_u8: torch.Tensor, b_u8: torch.Tensor) -> torch.Tensor:
            if want is not None and a_u8.device.type != want.type:
                raise ValueError(f"reference chain_fn built for {want} called with tensors on {a_u8.device}")
            return chain_gemm(a_u8, b_u8, fmt_a=fmt_a, fmt_b=fmt_b, preset=preset)
        fn.backend = self.name
        return fn

    def __repr__(self):
        return "ReferenceBackend()"


REFERENCE = ReferenceBackend()
