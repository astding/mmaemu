"""Instruction-level numerics: chain model, presets, specials, GEMM plans and the epilogue (docs/semantics.md)."""
from __future__ import annotations

from .chain import dot
from .presets import Preset, get_preset, list_presets

__all__ = ["dot", "Preset", "get_preset", "list_presets"]
