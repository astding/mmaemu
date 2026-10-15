"""The IEEE kernels v1 / v4 (KernelSpecs) and the standard-ABI ChainFn wrapper shared by every CUDA kernel.

v1 and v4 use only IEEE-defined instructions, so they are exact by construction on every GPU they run on; they need
only the "smoke" gate (see spec.py). The gated fast kernels v9 / tc3 live in mmaemu.backends.cuda.fast and
use the same ABI and wrapper.

chain_fn_for(spec, req) returns ChainFn(a, b) -> fp32 [M, N]: the Ada chain over k = 0..L-1 starting from +0, for
a [M, L] and b [N, L] E4M3 codes (torch.uint8 or torch.float8_e4m3fn) that are finite (mmaemu.model.plan.execute_plan
sanitises non-finite codes before calling it). The wrapper
  - accepts only uint8 codes or float8_e4m3fn (TypeError otherwise: e5m2, fnuz, bf16, ... never reach a kernel, B4);
  - requires both operands on the request's device (ValueError for mixed devices);
  - makes the operands contiguous and clones them if their address is not aligned to spec.align_bytes (B12);
  - checks 16 <= L < spec.k_max and L % 16 == 0 (ValueError), handles M == 0, N == 0 (empty) and L == 0 (+0);
  - launches under torch.cuda.device(device) on the current stream, in row chunks if ceil(M / tile_m) > 65535.
"""
from __future__ import annotations

import ctypes

from mmaemu.backends.cuda.spec import KernelSpec

__all__ = ["V1", "V4", "SPECS", "chain_fn_for"]

MAX_GRID_Y = 65535

V1 = KernelSpec(
    name="v1", module="v1", source="v1.cu", entry="mmaemu_v1_e4m3e4m3",
    sass_archs=(75, 80, 86, 89, 90, 100, 120), ptx_arch=75, min_cc=(7, 5),
    gated=False, validated_on=frozenset({(8, 9), (12, 0)}), gates=(), priority=10,
    tile_m=64, tile_n=64, threads=256, smem_bytes=0, align_bytes=4,
    k_max=1 << 30,  # int K with k0 + KT never overflowing; the fp32 accumulator cannot overflow below 2^30
)

V4 = KernelSpec(
    name="v4", module="v4", source="v4.cu", entry="mmaemu_v4_e4m3e4m3",
    sass_archs=(80, 86, 89, 90, 100, 120), ptx_arch=80, min_cc=(8, 0),
    gated=False, validated_on=frozenset({(8, 9), (12, 0)}), gates=(), priority=20,
    tile_m=128, tile_n=128, threads=256, smem_bytes=36864, align_bytes=4,
    k_max=1 << 24,  # block exponents <= 55 keep the byte encodings of the exponent pass valid
)

SPECS: tuple[KernelSpec, ...] = (V1, V4)


def _codes(t, name: str):
    import torch

    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name}: expected a torch.Tensor, got {type(t).__name__}")
    if t.dtype == torch.float8_e4m3fn:
        return t.view(torch.uint8)
    if t.dtype == torch.uint8:
        return t
    raise TypeError(f"{name}: the CUDA kernels take E4M3 codes (torch.uint8 or torch.float8_e4m3fn), got {t.dtype}; "
                    "other formats run on the reference backend")


def _aligned(t, align: int):
    t = t.contiguous()
    if t.numel() and t.data_ptr() % align:
        t = t.clone()  # a fresh allocation is aligned (B12: a misaligned 32-bit load is a sticky context error)
    return t


def chain_fn_for(spec: KernelSpec, req):
    """Standard-ABI ChainFn of `spec` for `req` (mmaemu.backends.ChainRequest)."""
    import torch

    fmts = (req.fmt_a, req.fmt_b)
    if fmts not in tuple(tuple(f) for f in spec.formats):
        raise ValueError(f"kernel {spec.name} supports {spec.formats}, not {fmts}")
    if req.preset != "ada":
        raise ValueError(f"kernel {spec.name} implements preset 'ada', not {req.preset!r}")
    device = torch.device(req.device)
    if device.type != "cuda":
        raise ValueError(f"kernel {spec.name} needs a CUDA device, got {device}")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    from mmaemu.backends.cuda import modules

    def chain(a, b):
        A, B = _codes(a, "a"), _codes(b, "b")
        if A.dim() != 2 or B.dim() != 2:
            raise ValueError(f"expected a [M, L] and b [N, L], got {tuple(A.shape)} and {tuple(B.shape)}")
        if A.device != device or B.device != device:
            raise ValueError(f"mixed devices: a on {A.device}, b on {B.device}, kernel {spec.name} on {device}")
        M, L = A.shape
        N, L2 = B.shape
        if L != L2:
            raise ValueError(f"K mismatch: a has {L}, b has {L2}")
        out = torch.empty((M, N), dtype=torch.float32, device=device)
        if M == 0 or N == 0:
            return out
        if L == 0:
            return out.zero_()
        if L % 16:
            raise ValueError(f"kernel {spec.name}: L = {L} is not a multiple of 16 (the plan zero-pads K)")
        if L >= spec.k_max:
            raise ValueError(f"kernel {spec.name}: L = {L} >= k_max = {spec.k_max}")
        if M >= 2 ** 31 or N >= 2 ** 31:
            raise ValueError("M and N must be < 2^31")
        A, B = _aligned(A, spec.align_bytes), _aligned(B, spec.align_bytes)
        rows = MAX_GRID_Y * spec.tile_m
        gx = -(-N // spec.tile_n)
        for r0 in range(0, M, rows):
            m = min(rows, M - r0)
            args = [ctypes.c_void_p(A.data_ptr() + r0 * L), ctypes.c_void_p(B.data_ptr()),
                    ctypes.c_void_p(out.data_ptr() + r0 * N * 4), ctypes.c_int(m), ctypes.c_int(N), ctypes.c_int(L)]
            modules.launch(spec.module, spec.entry, (gx, -(-m // spec.tile_m), 1), (spec.threads, 1, 1),
                           spec.smem_bytes, args, device)
        return out

    chain.spec = spec  # type: ignore[attr-defined]
    return chain
