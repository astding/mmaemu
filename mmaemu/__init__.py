"""mmaemu: bit-accurate emulation of NVIDIA tensor-core FP8 GEMM numerics.

The public surface below is implemented by mmaemu.api (numerics entry points), mmaemu.model (instruction model,
presets), mmaemu.platform (profiles, split-K) and mmaemu.backends (reference and CUDA backends). Importing ``mmaemu``
has no side effects: it does not import torch, does not initialise CUDA, does not load libcuda and never imports the
audited prototype package ``tcemu``. Every public name is resolved lazily on first attribute access (module
``__getattr__``); a name whose implementing module is absent from the installation resolves to a stub that raises
``NotImplementedError("<component>")`` when called.
"""
from __future__ import annotations

import importlib

from . import errors
from ._version import __version__
from .errors import (
    BackendFallbackWarning,
    BackendUnavailableError,
    ExperimentalPresetWarning,
    MmaemuError,
    ReferenceRejectsError,
    SplitKUnknownError,
    UnsupportedFeatureError,
    UnvalidatedEnvelopeWarning,
    UnverifiedSplitWarning,
)

# public name -> (module, attribute or None for the module itself, component named by the stub)
_LAZY: dict[str, tuple[str, str | None, str]] = {
    "scaled_mm": ("mmaemu.api", "scaled_mm", "mmaemu.api"),
    "accumulate": ("mmaemu.api", "accumulate", "mmaemu.api"),
    "explain": ("mmaemu.api", "explain", "mmaemu.api"),
    "CallPlan": ("mmaemu.api", "CallPlan", "mmaemu.api"),
    "resolve_splitk": ("mmaemu.platform", "resolve_splitk", "mmaemu.platform"),
    "SplitK": ("mmaemu.platform", "SplitK", "mmaemu.platform"),
    "SplitDecision": ("mmaemu.platform", "SplitDecision", "mmaemu.platform"),
    "get_profile": ("mmaemu.platform", "get_profile", "mmaemu.platform"),
    "list_profiles": ("mmaemu.platform", "list_profiles", "mmaemu.platform"),
    "Profile": ("mmaemu.platform", "Profile", "mmaemu.platform"),
    "get_preset": ("mmaemu.model", "get_preset", "mmaemu.model"),
    "list_presets": ("mmaemu.model", "list_presets", "mmaemu.model"),
    "Preset": ("mmaemu.model", "Preset", "mmaemu.model"),
    "backends_status": ("mmaemu.backends", "status", "mmaemu.backends"),
    "selftest": ("mmaemu.backends", "selftest", "mmaemu.backends"),
    # subpackages / modules
    "model": ("mmaemu.model", None, "mmaemu.model"),
    "platform": ("mmaemu.platform", None, "mmaemu.platform"),
    "backends": ("mmaemu.backends", None, "mmaemu.backends"),
    "testing": ("mmaemu.testing", None, "mmaemu.testing"),
    "formats": ("mmaemu.formats", None, "mmaemu.formats"),
    "api": ("mmaemu.api", None, "mmaemu.api"),
}

__all__ = [
    "__version__",
    "scaled_mm", "accumulate", "explain", "CallPlan",
    "resolve_splitk", "SplitK", "SplitDecision",
    "get_profile", "list_profiles", "Profile",
    "get_preset", "list_presets", "Preset",
    "backends_status", "selftest",
    "model", "testing", "errors",
    "MmaemuError", "ReferenceRejectsError", "UnsupportedFeatureError", "SplitKUnknownError",
    "BackendUnavailableError", "ExperimentalPresetWarning", "UnverifiedSplitWarning", "UnvalidatedEnvelopeWarning",
    "BackendFallbackWarning",
]


def _missing(name: str, owner: str):
    def stub(*args, **kwargs):
        raise NotImplementedError(owner)

    stub.__name__ = stub.__qualname__ = name
    stub.__doc__ = f"mmaemu.{name} is not implemented yet ({owner})."
    return stub


def __getattr__(name: str):
    try:
        modname, attr, owner = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'mmaemu' has no attribute {name!r}") from None
    try:
        mod = importlib.import_module(modname)
    except ModuleNotFoundError as e:
        if e.name != modname:  # a genuine import failure inside an existing module: never mask it
            raise
        value = _missing(name, owner)
    else:
        if attr is None:
            value = mod
        elif hasattr(mod, attr):
            value = getattr(mod, attr)
        else:
            value = _missing(name, owner)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
