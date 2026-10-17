"""Platform profiles: which GPU / cuBLASLt / caller a split-K decision and the hardware rejections refer to.

A Profile separates the platform (GPU, SM count, cuBLASLt version, caller, workspace size, alignment preference,
split-K table, hardware rejections) from the numerics (the instruction preset, mmaemu.model.presets). Evo 2 under
Transformer Engine on an RTX 4060 therefore picks TE's 4 MiB workspace (``ada-te``) while torch._scaled_mm
uses torch's 1 MiB default (``ada``). The default profile is ``hopper``.

    get_profile(name_or_profile=None) -> Profile     # None -> $MMAEMU_PROFILE or "hopper"
    list_profiles() -> list[str]

Evidence (docs/splitk_tables.md). RTX 4060 (``ada``, ``ada-te``, ``ada-ws0``; measured on that GPU):
- reference GPU: NVIDIA GeForce RTX 4060, sm_89, 24 SMs, cuBLASLt 12.8.3.14 (torch 2.7.1+cu128 wheel), driver
  572.16, WSL2; heuristic table heur_rtx4060_lt12.8.3.14 (truth heur.lt_heuristic_12_8_3_14 + recordings R2/R2x/R3);
- torch default cuBLASLt workspace 1 MiB (logged workSpaceSizeInBytes=1048576 on every recorded call);
- TE 2.3: workspace 4 MiB and MIN_ALIGNMENT 256 B; its split decisions equal torch's descriptor + alignment on the
  61 measured shapes, alignment never changes a decision (0 of 1,422,624 table pairs), but 46 TE split outputs had
  their reduction scheme filled from the table itself (circular) -> splitk_maturity "experimental";
- workspace 0: 0 split rows (209,712 queries); 7,731 shapes measured by the audit;
- e4m3 x e4m3 -> e5m2: CUBLAS_STATUS_NOT_SUPPORTED on sm_89 / cuBLASLt 12.8.3 (truth hw_error outputs; R2 heuristic
  status 15 on every queried shape); mixed e4m3/e5m2 inputs -> e5m2 run.

H100 PCIe (``hopper``, ``hopper-te``, ``hopper-ws0``; measured on that GPU, docs/validation_logs/2026-10-15/h100-probe):
sm_90, 114 SMs, cuBLASLt 12.8.3.14 (torch 2.7.1+cu128 wheel), driver 570.148.08; table heur_h100pcie_lt12.8.3.14
(850,320 heuristic queries); torch's workspace 1 MiB and TE 2.3's 32 MiB as logged on every recorded call; the 168
logged Evo 2 calls of each caller equal the table rows (algo, tile, stages, numSplitsK, customOption); workspace 0: 0
split rows (915,980 queries); e4m3 x e4m3 -> e5m2: status 15 on every queried shape and on every live call. The split-K
sweep (docs/validation_logs/2026-10-16/h100-splitk-sweep) added the model rows of 9 models (M 1..16,384) to the table
and a K-dependent envelope (S = 1 above a bound per setting and K): validated at 1 MiB on the sealed held-out shapes
(154,191 above the bound, 0 split), unvalidated at 32 MiB (77,518, 1 split; UnvalidatedEnvelopeWarning). The split rule
was found after the probe's data came in; the sweep's 2,416 live split calls, run after it was fixed, replay equal
(1,630 at 1 MiB, covering every S and customOption of the table's 1 MiB split rows and 5 of their 6 K; K = 5,760 only in
the probe's 84 split calls, which the rule was fitted to) -> splitk_maturity "validated" at 1 MiB and workspace 0
(decided after the sweep). TE: no split call ran through TE's own GEMM -> "experimental". No Evo 2 FP8 projection shape
((K, N) = (1920, 5760), (4096, 12288), (8192, 24576)) splits.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

__all__ = ["RejectRule", "Profile", "get_profile", "list_profiles", "DEFAULT_PROFILE", "PROFILES",
           "profile_maturity_warning"]

DEFAULT_PROFILE = "hopper"
_PROFILE_DESCRIPTIONS: dict[str, str] = {}


def _dtype_name(dt) -> str | None:
    if dt is None:
        return None
    s = str(dt)
    return s.split(".")[-1] if s.startswith("torch.") else s


def _fmt_name(dt) -> str | None:
    s = _dtype_name(dt)
    return {"float8_e4m3fn": "e4m3", "float8_e5m2": "e5m2"}.get(s, s)


@dataclass(frozen=True)
class RejectRule:
    """A call the profile's cuBLASLt rejects. None fields match anything. fmt_a/fmt_b: "e4m3" | "e5m2" (formats of
    mat1 / mat2); out_dtype: torch dtype name, e.g. "float8_e5m2"."""
    fmt_a: str | None
    fmt_b: str | None
    out_dtype: str | None
    message: str

    def matches(self, fmt_a, fmt_b, out_dtype) -> bool:
        fa, fb, od = _fmt_name(fmt_a), _fmt_name(fmt_b), _dtype_name(out_dtype)
        return ((self.fmt_a is None or self.fmt_a == fa) and (self.fmt_b is None or self.fmt_b == fb)
                and (self.out_dtype is None or self.out_dtype == od))


@dataclass(frozen=True)
class Profile:
    name: str
    preset: str
    gpu_name: str | None
    capability: tuple[int, int] | None
    sm_count: int | None
    cublaslt_version: str | None
    caller: Literal["torch._scaled_mm", "transformer_engine"]
    torch_version: str | None
    workspace_bytes: int | None          # None: not applicable (profiles without a reference platform)
    align: int | None
    splitk_table: str | None
    rejects: tuple[RejectRule, ...]
    maturity: Literal["validated", "experimental"]
    splitk_maturity: Literal["validated", "experimental", "none"]
    @property
    def description(self) -> str:
        """Display text kept outside the frozen profile contract."""
        return _PROFILE_DESCRIPTIONS.get(self.name, "")

    def rejection(self, fmt_a, fmt_b, out_dtype) -> RejectRule | None:
        """The first RejectRule matching the call, or None."""
        for r in self.rejects:
            if r.matches(fmt_a, fmt_b, out_dtype):
                return r
        return None

    def matches_device(self, device=None) -> bool:
        """True iff the CUDA device is this profile's reference platform: GPU name AND SM count AND cuBLASLt version
        (the loaded libcublasLt; never the compute capability alone). False without CUDA, with
        MMAEMU_DISABLE_CUDA=1, for profiles without a platform, or on any probe error. Uses the driver API through
        ctypes (mmaemu.platform.cublaslt): no torch CUDA context is created."""
        if not self.gpu_name or self.sm_count is None or not self.cublaslt_version:
            return False
        if os.environ.get("MMAEMU_DISABLE_CUDA") == "1":
            return False
        if device is not None and getattr(device, "type", None) not in (None, "cuda"):
            return False
        if isinstance(device, str) and not device.startswith("cuda"):
            return False
        try:
            from . import cublaslt
            if cublaslt.device_name(device) != self.gpu_name:
                return False
            if cublaslt.device_sm_count(device) != self.sm_count:
                return False
            return _version_matches(cublaslt.version(), self.cublaslt_version)
        except Exception:  # noqa: BLE001
            return False


def _version_matches(found: str, want: str) -> bool:
    """found is "major.minor.patch[.build]" from the library; the build number is compared only when both have it."""
    f, w = found.split("."), want.split(".")
    if f[:3] != w[:3]:
        return False
    return len(f) < 4 or len(w) < 4 or f[3] == w[3]


_ADA_REJECTS = (
    RejectRule("e4m3", "e4m3", "float8_e5m2",
               "cuBLASLt 12.8.3 on sm_89 rejects e4m3 x e4m3 -> e5m2 output (CUBLAS_STATUS_NOT_SUPPORTED); mixed "
               "e4m3/e5m2 inputs may produce e5m2"),
)
_ADA = dict(preset="ada", gpu_name="NVIDIA GeForce RTX 4060", capability=(8, 9), sm_count=24,
            cublaslt_version="12.8.3.14", torch_version="2.7.1+cu128", splitk_table="heur_rtx4060_lt12.8.3.14",
            rejects=_ADA_REJECTS, maturity="validated")


_H100_REJECTS = (
    RejectRule("e4m3", "e4m3", "float8_e5m2",
               "cuBLASLt 12.8.3 on the H100 (sm_90) rejects e4m3 x e4m3 -> e5m2 output (CUBLAS_STATUS_NOT_SUPPORTED); "
               "mixed e4m3/e5m2 inputs may produce e5m2"),
)
_H100 = dict(preset="hopper", gpu_name="NVIDIA H100 PCIe", capability=(9, 0), sm_count=114,
             cublaslt_version="12.8.3.14", torch_version="2.7.1+cu128", splitk_table="heur_h100pcie_lt12.8.3.14",
             rejects=_H100_REJECTS, maturity="validated")


def _profile(*, description: str = "", **kwargs) -> Profile:
    p = Profile(**kwargs)
    _PROFILE_DESCRIPTIONS[p.name] = description
    return p

PROFILES: dict[str, Profile] = {p.name: p for p in (
    _profile(name="ada", caller="torch._scaled_mm", workspace_bytes=1 << 20, align=None,
            splitk_maturity="validated",
            description="torch._scaled_mm as measured on an RTX 4060 (24 SMs, cuBLASLt 12.8.3.14, torch 2.7.1+cu128): "
                        "torch's default 1 MiB cuBLASLt workspace, no alignment preference", **_ADA),
    _profile(name="ada-te", caller="transformer_engine", workspace_bytes=4 << 20, align=256,
            splitk_maturity="experimental",
            description="Transformer Engine 2.3 general_gemm as measured on an RTX 4060 (Evo 2's FP8 path): 4 MiB "
                        "workspace, 256 B alignment preference. S > 1 decisions warn (UnverifiedSplitWarning): the "
                        "reduction scheme of 46 TE split outputs was filled from the table itself; all Evo 2 shapes "
                        "are S = 1",
            **_ADA),
    _profile(name="ada-ws0", caller="torch._scaled_mm", workspace_bytes=0, align=None,
            splitk_maturity="validated",
            description="torch._scaled_mm as measured on an RTX 4060 with CUBLASLT_WORKSPACE_SIZE=0: never split (0 "
                        "split rows in 209,712 queries)", **_ADA),
    _profile(name="hopper", caller="torch._scaled_mm", workspace_bytes=1 << 20, align=None,
            splitk_maturity="validated",
            description="torch._scaled_mm as measured on an H100 PCIe (114 SMs, cuBLASLt 12.8.3.14, torch "
                        "2.7.1+cu128): torch's default 1 MiB cuBLASLt workspace. Shapes off the table resolve by its "
                        "K-dependent envelope (S = 1 above the bound; validated on 154,191 held-out shapes) or on that "
                        "GPU (live query); S > 1 decisions are validated by 1,630 live split calls replayed after the "
                        "split rule was fixed; no Evo 2 FP8 projection shape splits", **_H100),
    _profile(name="hopper-te", caller="transformer_engine", workspace_bytes=32 << 20, align=256,
            splitk_maturity="experimental",
            description="Transformer Engine 2.3 general_gemm as measured on an H100 PCIe (Evo 2's FP8 path): 32 MiB "
                        "workspace, 256 B alignment preference; the table was recorded with torch's descriptor, which "
                        "gives TE's logged decision on all 168 Evo 2 calls (all S = 1); S > 1 decisions warn (no split "
                        "call ran through TE's own GEMM), and so do the envelope's S = 1 inferences "
                        "(UnvalidatedEnvelopeWarning: its held-out test found 1 split among 77,518 shapes above the "
                        "bound)", **_H100),
    _profile(name="hopper-ws0", caller="torch._scaled_mm", workspace_bytes=0, align=None,
            splitk_maturity="validated",
            description="torch._scaled_mm as measured on an H100 PCIe with CUBLASLT_WORKSPACE_SIZE=0: never split (0 "
                        "split rows in 915,980 queries)", **_H100),
    _profile(name="hopper-experimental", preset="hopper", gpu_name=None, capability=(9, 0), sm_count=None,
            cublaslt_version=None, caller="torch._scaled_mm", torch_version=None, workspace_bytes=None, align=None,
            splitk_table=None, rejects=(), maturity="experimental", splitk_maturity="none",
            description="hopper numerics without a reference platform (H100 SXM, H200, other cuBLASLt versions): the "
                        "hopper preset is validated on an H100 PCIe only (profiles hopper, hopper-te, hopper-ws0); no "
                        "split-K table: splitk='auto' always raises SplitKUnknownError"),
    _profile(name="b200-mma-sync-experimental", preset="b200_mma_sync", gpu_name=None, capability=(10, 0),
            sm_count=None, cublaslt_version=None, caller="torch._scaled_mm", torch_version=None,
            workspace_bytes=None, align=None, splitk_table=None, rejects=(), maturity="experimental",
            splitk_maturity="none",
            description="fits the toolbox B200 mma.sync (HMMA-rewrite) vectors; NOT a model of sm_120 / RTX 5070 Ti; "
                        "NOT B200 cuBLAS; no split-K table: splitk='auto' always raises SplitKUnknownError"),
)}


def profile_maturity_warning(p: Profile, preset_validated: bool) -> str | None:
    """The ExperimentalPresetWarning text of an experimental profile whose preset is validated (an experimental preset
    warns by itself), or None."""
    if p.maturity != "experimental" or not preset_validated:
        return None
    return f"profile {p.name!r} is experimental: {p.description}"


def list_profiles() -> list[str]:
    return list(PROFILES)


def get_profile(name_or_profile=None) -> Profile:
    """Profile by name (None -> $MMAEMU_PROFILE or DEFAULT_PROFILE, "hopper"); a Profile instance is
    returned as is."""
    if isinstance(name_or_profile, Profile):
        return name_or_profile
    name = name_or_profile if name_or_profile is not None else (os.environ.get("MMAEMU_PROFILE") or DEFAULT_PROFILE)
    if not isinstance(name, str):
        raise TypeError(f"profile must be a name or a Profile, got {type(name_or_profile).__name__}")
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown profile {name!r}; known: {', '.join(PROFILES)}. A GPU without a profile needs a "
                       f"recorded table: see tools/record_heuristics.py and docs/runbooks/sm120_bringup.md") from None
