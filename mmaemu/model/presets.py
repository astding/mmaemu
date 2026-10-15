"""Instruction-level numerics presets (docs/presets_profiles.md).

A preset fixes the block-FMA model of one FP8 tensor-core instruction family (see :mod:`mmaemu.model.chain`):
``nfma`` products per block, ``frac`` fraction bits kept below the block's alignment exponent, ``frac_out`` fraction
bits of the block result, ``min_exp`` lower clamp of the alignment exponent, ``late_c`` (the accumulator is added
after the product sum with an fp32 round-to-nearest-even addition instead of taking part in the alignment), and the
GEMM-level slow-accumulation chunk ``slow_chunk`` (None: GEMM-level behaviour is not characterised, so slow
accumulation and split-K raise :class:`~mmaemu.errors.UnsupportedFeatureError`).

Maturity follows the hardware evidence, never the paper:

========================  ============  ============================================================================
name                      maturity      evidence / scope
========================  ============  ============================================================================
``ada``                   validated     RTX 4060 (sm_89) mma.sync m16n8k32: 1.06e8 hardware outputs incl. targeted
                                        vectors, all four FP8 format pairs; toolbox Ada/L40S vectors 5000/5000;
                                        GEMM level through cuBLASLt 12.8.3.14 (see mmaemu.model.plan)
``hopper``                experimental  instruction level only: toolbox H100/H200 vectors recorded with c = 0, which
                                        cannot tell how the accumulator enters; no GEMM-level evidence
``b200_mma_sync``         experimental  fits the toolbox B200 mma.sync vectors, which ptxas compiles for sm_90/sm_100
                                        into two fp16 HMMAs plus an fp32 add (HMMA rewrite); the fitted model is not
                                        unique. NOT a model of sm_120 (RTX 50xx, whose mma.sync is a native QMMA) and
                                        NOT a model of B200 cuBLAS (tcgen05)
========================  ============  ============================================================================

``blackwell`` is accepted as an alias of ``b200_mma_sync`` and always warns. ``sm_120`` / ``rtx_blackwell`` are
refused: sm_120 is not modelled (no preset, profile or split-K table). Probes on an RTX 5070 Ti are consistent with
the B200 tensor-core numerics reported for sm_120 (arXiv:2512.07004v4), but nothing is validated at GEMM level
(docs/validation_logs/2026-10-13-sm120, docs/runbooks/sm120_bringup.md).

Custom :class:`Preset` objects (maturity ``"custom"``) can be passed wherever a preset name is accepted, e.g. to score
hypotheses for new hardware; they are never a profile default.
"""
from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass
from typing import Literal

from mmaemu.errors import ExperimentalPresetWarning

__all__ = ["Preset", "get_preset", "list_presets", "ALIASES"]


@dataclass(frozen=True)
class Preset:
    name: str
    nfma: int
    frac: int
    frac_out: int
    min_exp: int
    late_c: bool
    slow_chunk: int | None            # None -> slow accumulation (and split-K) raise UnsupportedFeatureError
    maturity: Literal["validated", "experimental", "custom"] = "custom"
    validated_scope: str = ""         # one line, used in warnings and docs
    evidence: tuple = ()              # truth ids / other evidence

    def __post_init__(self):
        if self.nfma < 1:
            raise ValueError("nfma must be positive")
        if self.slow_chunk is not None and (type(self.slow_chunk) is not int or self.slow_chunk < 1):
            raise ValueError(f"slow_chunk must be None or a positive int, got {self.slow_chunk!r}")
        if not (1 <= self.frac <= 40 and 1 <= self.frac_out <= 23):
            raise ValueError("frac must be in [1, 40] and frac_out in [1, 23]")
        if self.maturity not in ("validated", "experimental", "custom"):
            raise ValueError(f"invalid maturity {self.maturity!r}")


ADA = Preset(
    "ada", nfma=16, frac=13, frac_out=13, min_exp=-132, late_c=False, slow_chunk=128, maturity="validated",
    validated_scope="Ada (sm_89) FP8 mma.sync m16n8k32, validated bit for bit on an RTX 4060",
    evidence=("c2.*", "c6.kit_MM_ada_sm89", "c2.s0_toolbox_vectors_rerec", "toolbox Ada/L40S vectors 5000/5000",
              "tcemu audit: Ada instruction model confirmed"))
HOPPER = Preset(
    "hopper", nfma=32, frac=13, frac_out=13, min_exp=-132, late_c=False, slow_chunk=None, maturity="experimental",
    validated_scope="instruction level only, toolbox H100/H200 vectors with c = 0; how c enters and all GEMM-level "
                    "behaviour are not validated",
    evidence=("toolbox H100/H200 E4M3/E5M2 vectors (c = 0) 5000/5000", "tcemu audit: instruction level only"))
B200_MMA_SYNC = Preset(
    "b200_mma_sync", nfma=32, frac=25, frac_out=23, min_exp=-133, late_c=True, slow_chunk=None,
    maturity="experimental",
    validated_scope="fits the toolbox B200 mma.sync (HMMA-rewrite) vectors; NOT a model of sm_120 / RTX 5070 Ti; "
                    "NOT B200 cuBLAS",
    evidence=("toolbox B200 E4M3/E5M2 vectors 5000/5000 (not identifying: >= 137 models fit)",
              "tcemu audit: not a model of RTX Blackwell (docs/limitations.md)"))

_REGISTRY = {p.name: p for p in (ADA, HOPPER, B200_MMA_SYNC)}
ALIASES = {"blackwell": "b200_mma_sync"}
_REFUSED = ("sm_120", "sm120", "rtx_blackwell", "rtx-blackwell", "rtx5070ti", "rtx_5070_ti")
_ALIAS_WARNING = ("preset 'blackwell' is an alias of 'b200_mma_sync': it fits the toolbox B200 mma.sync "
                  "(HMMA-rewrite) vectors; NOT a model of sm_120 / RTX 5070 Ti; NOT B200 cuBLAS")
_warned: set = set()
_lock = threading.Lock()


def list_presets() -> list:
    """Canonical preset names (aliases excluded)."""
    return sorted(_REGISTRY)


def maturity_warning(p: Preset) -> str | None:
    """The ExperimentalPresetWarning text for a preset, or None for a validated preset."""
    if p.maturity == "validated":
        return None
    kind = "experimental" if p.maturity == "experimental" else "custom (not validated)"
    return f"preset {p.name!r} is {kind}: {p.validated_scope}"


def _lookup(name_or_preset) -> tuple:
    """(Preset, alias_used) without any warning."""
    if isinstance(name_or_preset, Preset):
        return name_or_preset, False
    if not isinstance(name_or_preset, str):
        raise TypeError(f"preset must be a name or a Preset, got {type(name_or_preset).__name__}")
    name = name_or_preset
    if name in ALIASES:
        return _REGISTRY[ALIASES[name]], True
    if name in _REGISTRY:
        return _REGISTRY[name], False
    if name.lower() in _REFUSED:
        raise KeyError(f"preset {name!r}: sm_120 (RTX 50xx) is not modelled: mmaemu has no preset, profile or "
                       "split-K table for it, and 'b200_mma_sync' (alias 'blackwell') is NOT a model of sm_120. "
                       "Probes on an RTX 5070 Ti are consistent with the B200 tensor-core numerics but are not a "
                       "GEMM-level validation; the bring-up procedure is docs/runbooks/sm120_bringup.md.")
    raise KeyError(f"unknown preset {name!r}; known: {list_presets()} (alias: 'blackwell')")


def get_preset(name, *, warn: bool = True) -> Preset:
    """Look up a preset by name (or pass a :class:`Preset` through).

    Emits :class:`~mmaemu.errors.ExperimentalPresetWarning` once per process per name for experimental / custom
    presets, and on every lookup of the alias ``blackwell``. ``warn=False`` only looks up (used by explain())."""
    p, alias = _lookup(name)
    if warn:
        if alias:
            warnings.warn(_ALIAS_WARNING, ExperimentalPresetWarning, stacklevel=2)
        else:
            msg = maturity_warning(p)
            if msg is not None:
                with _lock:
                    first = p.name not in _warned
                    _warned.add(p.name)
                if first:
                    warnings.warn(msg, ExperimentalPresetWarning, stacklevel=2)
    return p


def _reset_warnings_for_tests() -> None:
    """Forget which presets have warned (test hook: the once-per-process policy makes tests order-dependent)."""
    with _lock:
        _warned.clear()
