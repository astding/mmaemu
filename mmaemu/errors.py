"""Exception and warning classes of mmaemu.

Exactly these classes exist. Code that maps mmaemu errors to other taxonomies (for example the truth-replay adapter)
must match the EXACT class: ReferenceRejectsError and UnsupportedFeatureError are both RuntimeError /
NotImplementedError subclasses, so a broad ``except RuntimeError`` would conflate hardware rejections with scope gaps.
"""
from __future__ import annotations

__all__ = [
    "MmaemuError",
    "ReferenceRejectsError",
    "UnsupportedFeatureError",
    "SplitKUnknownError",
    "BackendUnavailableError",
    "ExperimentalPresetWarning",
    "UnverifiedSplitWarning",
    "UnvalidatedEnvelopeWarning",
    "BackendFallbackWarning",
]


class MmaemuError(Exception):
    """Base class of every mmaemu error."""


class ReferenceRejectsError(MmaemuError, RuntimeError):
    """The reference platform (torch 2.7.1 or the profile's cuBLAS) rejects this call; mmaemu raises where it does."""


class UnsupportedFeatureError(MmaemuError, NotImplementedError):
    """The call is valid on the reference platform but outside mmaemu's modelled scope (never approximated)."""


class SplitKUnknownError(MmaemuError, LookupError):
    """The split-K decision of cuBLAS for this call is not known (no table row, no live query, inside the envelope)."""


class BackendUnavailableError(MmaemuError, RuntimeError):
    """A requested backend (or a library it needs: libcuda, libnvrtc, a fatbin) is not available here."""


class ExperimentalPresetWarning(UserWarning):
    """The selected preset is experimental (not validated against GEMM-level hardware recordings), or the profile is
    experimental although its preset is validated (hopper-experimental: a platform other than the validated one)."""


class UnverifiedSplitWarning(UserWarning):
    """The split-K decision used was never replayed on hardware (or comes from an experimental profile)."""


class UnvalidatedEnvelopeWarning(UnverifiedSplitWarning):
    """S = 1 was inferred by a split-K envelope that did not pass its held-out test (the H100 table at Transformer
    Engine's 32 MiB workspace): the reference GPU may split the shape."""


class BackendFallbackWarning(RuntimeWarning):
    """A faster backend was skipped (failed to load, failed a gate, ...) and a slower one is used instead."""
