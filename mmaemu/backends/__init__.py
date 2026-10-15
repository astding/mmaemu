"""Backends: who computes the chains of a GEMM plan.

A backend turns a ChainRequest (device, formats, preset) into a ChainFn
``(a_u8 [M, L], b_u8 [N, L]) -> fp32 [M, N]`` (one chain per output from +0, finite codes, L % 16 == 0); the plan
executor (mmaemu.model.plan.execute_plan) composes slow accumulation, split-K, padding and specials around it.

``select``, ``status`` and ``selftest`` delegate to ``mmaemu.backends.registry`` (kernel discovery, gating rule,
fallback warnings), which provides ``select(req, backend)``, ``status(device)`` and
``selftest(device, *, kernels, force)``. Without that module (a stripped installation):
- ``select(req, "auto" | "reference")`` returns the reference backend, an explicit "cuda..." backend raises
  BackendUnavailableError, and an unknown name raises ValueError;
- ``status()`` reports the reference backend only;
- ``selftest()`` raises NotImplementedError("mmaemu.backends.registry").
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from mmaemu.errors import BackendUnavailableError

if TYPE_CHECKING:
    import torch

    from mmaemu.backends.cuda.spec import GateResult
    from mmaemu.model.plan import ChainFn

__all__ = ["ChainRequest", "Backend", "BackendStatus", "select", "status", "selftest", "BACKEND_NAMES"]

BACKEND_NAMES = ("auto", "reference", "cuda", "cuda:v1", "cuda:v4", "cuda:v9", "cuda:tc3")


@dataclass(frozen=True)
class ChainRequest:
    """What a plan needs computed: chains of fmt_a x fmt_b codes under `preset`, on `device`."""
    device: torch.device
    fmt_a: str
    fmt_b: str
    preset: str


@runtime_checkable
class Backend(Protocol):
    name: str

    def supports(self, req: ChainRequest) -> str | None:
        """None if this backend can serve `req`, else the reason it cannot."""
        ...

    def chain_fn(self, req: ChainRequest) -> ChainFn:
        """The ChainFn serving `req`."""
        ...


@dataclass(frozen=True)
class BackendStatus:
    """Availability of one backend on one device (mmaemu.backends_status)."""
    name: str
    available: bool
    source: str | None          # "fatbin" | "nvrtc" | "torch" | None
    validated_here: bool
    gates: tuple = ()           # GateResult objects of the gates run on this device
    reason: str = ""


def _registry():
    """mmaemu.backends.registry if it exists, else None. Errors raised inside an existing registry propagate."""
    try:
        return importlib.import_module("mmaemu.backends.registry")
    except ModuleNotFoundError as e:
        if e.name == "mmaemu.backends.registry":
            return None
        raise


def select(req: ChainRequest, backend: str = "auto") -> Backend:
    """The backend that serves `req` (``backend``: one of BACKEND_NAMES)."""
    if not isinstance(backend, str):
        raise TypeError(f"backend must be a str, got {type(backend).__name__}")
    reg = _registry()
    if reg is not None:
        return reg.select(req, backend)
    if backend not in BACKEND_NAMES:
        raise ValueError(f"unknown backend {backend!r}; expected one of {', '.join(BACKEND_NAMES)}")
    if backend in ("auto", "reference"):
        from .reference import REFERENCE
        return REFERENCE
    raise BackendUnavailableError(f"backend {backend!r} is not available: mmaemu.backends.registry is not installed")


def status(device=None) -> dict[str, BackendStatus]:
    """Name -> BackendStatus for every backend known on `device` (default: the current device)."""
    reg = _registry()
    if reg is not None:
        return reg.status(device)
    return {"reference": BackendStatus(name="reference", available=True, source="torch", validated_here=True)}


def selftest(device=None, *, kernels="all", force: bool = False) -> dict[str, list[GateResult]]:
    """Run the gates of every kernel, or of `kernels` (a comma-separated string or a sequence of names), on `device`;
    kernel name -> gate results."""
    reg = _registry()
    if reg is None or not hasattr(reg, "selftest"):
        raise NotImplementedError("mmaemu.backends.registry")
    return reg.selftest(device, kernels=kernels, force=force)
