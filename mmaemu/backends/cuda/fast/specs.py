"""KernelSpecs of the gated fast kernels and the probe module.

    v9  : p2-int-mixed v9 cfg 1 -- bf16 tensor-core block-exponent pass ("E-pass") + CUDA-core FFMA.RZ chain.
          Hardware dependence: the bf16 mma.sync.m16n8k16 must have the window property (result of a sum of <= 17
          powers of two in [max term, 17 x max term]); gate "window".
    tc3 : p3 tc_emu3 cfg 6 -- fp16 tensor-core dummy-accumulator emulation ("design A").
          Hardware dependence: fp16 mma.sync.m16n8k8 with the Ampere/Ada numerics (8-term blocks, 24 fraction bits,
          c in the alignment) and the bf16 window property; gate "design_a".  SASS only for sm_80 / sm_86 / sm_89.

Both are gated (validated_on {(8, 9)}: RTX 4060) and are not release-blocking: if they miss the release-candidate
freeze, their specs are left out of the release fatbin and the registry runs without them.
"""
from __future__ import annotations

from mmaemu.backends.cuda import spec as _spec
from mmaemu.backends.cuda.spec import KernelSpec

#: v9 cfg 1: 64 x 64 tile, 128 threads, 2 x sizeof(Tile8<64, 64>) = 2 x 18,432 B of dynamic shared memory.
V9 = KernelSpec(
    name="v9", module="v9", source="v9.cu", entry="mmaemu_v9_e4m3e4m3",
    sass_archs=(80, 86, 89, 90, 100, 120), ptx_arch=None, min_cc=(8, 0),
    gated=True, validated_on=frozenset({(8, 9)}), gates=("window",), priority=30,
    tile_m=64, tile_n=64, threads=128, smem_bytes=36864, align_bytes=4,
    k_max=1 << 24,   # keeps the block exponent <= 55, the limit of the kernel's byte encodings (LA_ZERO = 60)
)

#: tc3 cfg 6: 128 x 128 tile, 256 threads, 2 buffers x (128 + 128) rows x 40 halves x 2 (values, exponent codes)
#: x 2 B = 81,920 B of dynamic shared memory (> 48 KiB: the loader must opt in) + 1 KiB static (s_cpx).
TC3 = KernelSpec(
    name="tc3", module="tc3", source="tc3.cu", entry="mmaemu_tc3_e4m3e4m3",
    sass_archs=(80, 86, 89), ptx_arch=None, min_cc=(8, 0),
    gated=True, validated_on=frozenset({(8, 9)}), gates=("design_a",), priority=40,
    tile_m=128, tile_n=128, threads=256, smem_bytes=81920, align_bytes=16,
    k_max=1 << 24,   # justified in docs/kernels_fast.md (gate domain covers every block exponent reachable below 2^24)
)

SPECS: tuple[KernelSpec, ...] = (TC3, V9)

#: Probe module shipped in the fatbin with the kernels' flags (SASS sm_80, 86, 89, 90, 100, 120; no PTX).
PROBE_MODULES: dict[str, str] = {"probes": "probes.cu"}
PROBE_SASS_ARCHS: tuple[int, ...] = (80, 86, 89, 90, 100, 120)

for _k, _v in PROBE_MODULES.items():   # register the "probes" module (see spec.PROBE_MODULES)
    _spec.PROBE_MODULES.setdefault(_k, _v)

from . import gates  # noqa: E402,F401  (registers the "window" and "design_a" gates)
