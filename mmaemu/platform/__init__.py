"""Reference-platform data: profiles and split-K resolution (docs/presets_profiles.md, docs/splitk_tables.md)."""
from __future__ import annotations

from .profiles import Profile, RejectRule, get_profile, list_profiles
from .splitk_table import SplitDecision, SplitK, resolve_splitk

__all__ = ["Profile", "RejectRule", "get_profile", "list_profiles", "SplitK", "SplitDecision", "resolve_splitk"]
