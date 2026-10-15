"""Kernel descriptions, gate registry and kernel ABI v1.

Kernel ABI v1 (every chain kernel):

    extern "C" __global__ void mmaemu_<name>_e4m3e4m3(const uint8_t* __restrict__ A,
                                                      const uint8_t* __restrict__ B,
                                                      float* __restrict__ C, int M, int N, int K);

- A: [M][K] E4M3 codes, row-major. B: [N][K], row-major. C: [M][N] fp32, row-major.
- C[m][n] = Ada fast chain over k = 0..K-1, starting from +0, with M1 (+0 for zero block results).
- Preconditions, checked by the Python wrapper and never by the kernel:
  1 <= M, N; 16 <= K < k_max; K % 16 == 0; all codes finite; pointers aligned to align_bytes;
  grid = (ceil(N/tile_n), ceil(M/tile_m), 1); block = (threads, 1, 1); dynamic smem = smem_bytes
  (see KernelSpec.launch_config).
- The kernel must not fault for any byte value. Sources include no torch headers and no host code.

Gating rule: a gated kernel is auto-selected only if (a) the device capability is in validated_on, or
MMAEMU_ALLOW_GATED_KERNELS=1, and (b) all its gates pass on that device. Non-gated kernels need only the "smoke"
gate. Gates are callables ``(KernelSpec, torch.device) -> GateResult`` registered by name with register_gate
(selftest.py registers "smoke"; mmaemu.backends.cuda.fast.gates registers "window" and "design_a").

Probe modules (device code that is not a chain kernel, e.g. the fast kernels' "probes") are listed in PROBE_MODULES
as stem -> source file under mmaemu/backends/cuda/csrc/. fast.specs adds its entries to this dict
(``PROBE_MODULES.update(...)``) when imported; build_fatbin and the runtime read it after discovery.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import torch

__all__ = ["KERNEL_ABI_VERSION", "KernelSpec", "GateResult", "GATES", "register_gate", "PROBE_MODULES"]

KERNEL_ABI_VERSION = 1


@dataclass(frozen=True)
class KernelSpec:
    """One shipped chain kernel (one config per kernel)."""
    name: str                       # "v1" | "v4" | "v9" | "tc3"
    module: str                     # fatbin and source stem
    source: str                     # file under mmaemu/backends/cuda/csrc/
    entry: str                      # "mmaemu_<name>_e4m3e4m3"
    sass_archs: tuple[int, ...]     # SASS targets of the release fatbin
    ptx_arch: int | None
    min_cc: tuple[int, int]
    gated: bool                     # True: hardware-dependent numerics (v9, tc3)
    validated_on: frozenset         # capabilities with hardware validation, e.g. {(8, 9), (12, 0)}
    gates: tuple[str, ...]          # extra gates beyond "smoke" (v9: ("window",), tc3: ("design_a",))
    priority: int                   # tc3 40, v9 30, v4 20, v1 10
    tile_m: int
    tile_n: int
    threads: int
    smem_bytes: int
    align_bytes: int                # 4 (v1/v4/v9) or 16 (tc3)
    k_max: int                      # exclusive; v4/v9: 1 << 24
    formats: tuple = (("e4m3", "e4m3"),)

    def __post_init__(self):
        def need(cond, what):
            if not cond:
                raise ValueError(f"KernelSpec {self.name!r}: {what}")

        need(isinstance(self.name, str) and self.name, "name must be a non-empty str")
        need(self.entry == f"mmaemu_{self.name}_e4m3e4m3", f"entry must be 'mmaemu_{self.name}_e4m3e4m3' (ABI v1)")
        need(isinstance(self.sass_archs, tuple) and all(isinstance(a, int) for a in self.sass_archs),
             "sass_archs must be a tuple of ints such as 89")
        need(self.ptx_arch is None or isinstance(self.ptx_arch, int), "ptx_arch must be an int or None")
        need(isinstance(self.min_cc, tuple) and len(self.min_cc) == 2, "min_cc must be a (major, minor) tuple")
        need(isinstance(self.validated_on, frozenset)
             and all(isinstance(c, tuple) and len(c) == 2 for c in self.validated_on),
             "validated_on must be a frozenset of (major, minor) tuples")
        need(isinstance(self.gates, tuple) and all(isinstance(g, str) for g in self.gates),
             "gates must be a tuple of gate names")
        need(min(self.tile_m, self.tile_n, self.threads) > 0 and self.smem_bytes >= 0, "bad launch geometry")
        need(self.align_bytes > 0 and self.align_bytes & (self.align_bytes - 1) == 0,
             "align_bytes must be a power of two")
        need(self.k_max > 16, "k_max must exceed 16")

    def launch_config(self, M: int, N: int) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
        """(grid, block, dynamic smem bytes) of the ABI v1 launch for an [M, N] output."""
        return ((N + self.tile_n - 1) // self.tile_n, (M + self.tile_m - 1) // self.tile_m, 1), \
            (self.threads, 1, 1), self.smem_bytes


@dataclass(frozen=True)
class GateResult:
    """Outcome of one gate for one kernel on one device."""
    name: str
    passed: bool
    reason: str
    details: dict = field(default_factory=dict)
    seconds: float = 0.0
    cached: bool = False


GATES: dict[str, Callable[[KernelSpec, "torch.device"], GateResult]] = {}

PROBE_MODULES: dict[str, str] = {}   # module stem -> source file; fast.specs adds "probes"


def register_gate(name: str, fn: Callable[[KernelSpec, "torch.device"], GateResult]) -> None:
    """Register gate `name`. Re-registering the same function (or a reload of it: same module and qualname) replaces
    it; registering a different function under a taken name raises ValueError."""
    if not isinstance(name, str) or not name:
        raise ValueError("gate name must be a non-empty str")
    if not callable(fn):
        raise TypeError(f"gate {name!r}: fn must be callable")
    old = GATES.get(name)
    if old is not None and old is not fn and (getattr(old, "__module__", None), getattr(old, "__qualname__", None)) \
            != (getattr(fn, "__module__", None), getattr(fn, "__qualname__", None)):
        raise ValueError(f"gate {name!r} is already registered by {old.__module__}.{old.__qualname__}")
    GATES[name] = fn
