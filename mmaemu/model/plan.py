"""GEMM-level accumulation of cuBLAS FP8 GEMMs as a plan of chains (docs/semantics.md).

Measured on the RTX 4060 (24 SMs, cuBLASLt 12.8.3.14, torch 2.7.1; truth c4.*, c5.*, misc.tcemu_tests_recorded_hw;
model: the tcemu audit's c5-splitk/splitk_model.py) and, where said, on an H100 PCIe (114 SMs, same cuBLASLt;
docs/validation_logs/2026-10-15/h100-probe):

- every chain starts from +0 and runs the instruction model (:mod:`mmaemu.model.chain`) over its K indices in order;
- fast accumulation, unsplit (numSplitsK = 1): one chain over k = 0..K-1;
- slow accumulation (use_fast_accum=False), unsplit: chains over consecutive 128-element chunks, added in K order
  into an fp32 accumulator from +0 with round-to-nearest additions. This 128-element restart holds for unsplit calls
  only;
- split-K (numSplitsK = S > 1): K is cut into 64-wide k-tiles t (the ragged tail is zero padded); split s owns the
  tiles with t mod S == s in increasing t (interleaved, not contiguous slices). Fast: one chain per split. Slow: the
  chain restarts every 2 of the split's OWN tiles and the chunk results are added serially from +0 in fp32. The S
  partials are combined in fp32 (round to nearest):
    "serial"   (cuBLASLt customOption 0 or 2): ((p0 + p1) + p2) + ...
    "pairwise" (customOption 1): h = ceil(S/2); slot_i = p_i + p_(i+h) if i + h < S - 1 else p_i;
               result (((p_(S-1) + slot_0) + slot_1) + ...) + slot_(h-1); for S = 3 this is (p2 + p0) + p1.
  alpha is applied after the combination (by :mod:`mmaemu.model.epilogue`);
- split-K on the H100 ("contiguous", every customOption): K is cut into 128-wide k-tiles (the sm_90 kernels' k-stage);
  split s owns the contiguous tiles [s*q, (s+1)*q) with q = ceil(T / S). Fast: one chain per split. Slow: every tile
  is its own chain, added serially from +0 in fp32. The partials are combined as "serial". Replayed on the H100's
  live split calls, ragged last k-tiles (K % 128 != 0) included; an empty split (never recorded) gives +0;
- non-finite inputs: :mod:`mmaemu.model.specials` (sanitise before the chains, patch after the combination).

Which (S, scheme) cuBLAS picks is a heuristic decision of GPU, SM count, cuBLASLt version, workspace size, output
dtype and fast/slow; it is not a function of the shape (see mmaemu.platform.splitk_table).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch

from mmaemu.errors import UnsupportedFeatureError
from mmaemu.model import specials
from mmaemu.model.presets import get_preset, split_schemes

__all__ = ["GemmPlan", "make_plan", "execute_plan", "ChainFn", "split_index", "contiguous_index", "reduce_partials",
           "TILE_K", "CONTIG_TILE_K", "SLOW_CHUNK_TILES", "SCHEMES"]

TILE_K = 64            # k-tile of every Ada FP8 xmma kernel cuBLASLt 12.8.3 picks
SLOW_CHUNK_TILES = 2   # slow accumulation restarts every 2 of a split's own k-tiles (= 128 K elements unsplit)
CONTIG_TILE_K = 128    # k-tile of the H100's sm_90 nvjet split kernels (MATMUL_STAGES_128xAUTO)
SCHEMES = ("serial", "pairwise", "contiguous")

ChainFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""(a_u8 [M, L] contiguous finite codes, b_u8 [N, L]) -> fp32 [M, N], chain from +0 over L (L % 16 == 0)."""


@dataclass(frozen=True)
class GemmPlan:
    K: int
    fast: bool
    S: int
    scheme: Literal["none", "serial", "pairwise", "contiguous"]
    chunks: tuple       # chunks[s][j]: int64 index vector into K, len % 16 == 0, -1 = zero code; () = empty split
    identity: bool      # True iff S == 1, fast, and K % 16 == 0 (no gather, no copy)


def split_index(K: int, S: int, tile: int = TILE_K) -> torch.Tensor:
    """[S, L] K indices handled by split s in chain order (-1 = zero padding): tile t goes to split t mod S."""
    T = -(-K // tile)
    Tp = -(-T // S) * S
    k = torch.arange(Tp * tile).reshape(Tp // S, S, tile).permute(1, 0, 2).reshape(S, -1)
    return torch.where(k < K, k, torch.full_like(k, -1))


def contiguous_index(K: int, S: int, tile: int = CONTIG_TILE_K) -> list:
    """S int64 index vectors into K in chain order (-1 = zero padding): split s owns the k-tiles [s*q, (s+1)*q),
    q = ceil(T / S); the last splits may own fewer tiles, or none (an empty vector)."""
    T = -(-K // tile)
    q = -(-T // S)
    out = []
    for s in range(S):
        k = torch.arange(min(s * q, T) * tile, min((s + 1) * q, T) * tile)
        out.append(torch.where(k < K, k, torch.full_like(k, -1)))
    return out


def make_plan(K: int, *, fast: bool, S: int = 1, scheme: str = "none", preset="ada") -> GemmPlan:
    """Plan of the cuBLAS accumulation for a K-long reduction (see the module docstring).

    S > 1 requires a scheme of the preset's kernels (mmaemu.model.presets.SPLIT_SCHEMES: ada 'serial' or 'pairwise',
    hopper 'contiguous'; a custom preset any of SCHEMES); S == 1 ignores the scheme. Presets without a characterised
    GEMM level (slow_chunk None: b200_mma_sync) raise UnsupportedFeatureError for slow accumulation and split-K."""
    if isinstance(K, bool) or not isinstance(K, int) or K < 0:
        raise ValueError(f"K must be a non-negative int, got {K!r}")
    if isinstance(S, bool) or not isinstance(S, int) or S < 1:
        raise ValueError(f"S must be a positive int, got {S!r}")
    p = get_preset(preset, warn=False)
    if S == 1:
        scheme = "none"
    elif scheme not in SCHEMES:
        raise ValueError(f"S = {S} > 1 needs scheme 'serial', 'pairwise' or 'contiguous', got {scheme!r}")
    if p.slow_chunk is None and not fast:
        raise UnsupportedFeatureError(f"slow accumulation (use_fast_accum=False) is not characterised for preset "
                                      f"{p.name!r} ({p.maturity})")
    if p.slow_chunk is None and S > 1:
        raise UnsupportedFeatureError(f"split-K is not characterised for preset {p.name!r} ({p.maturity})")
    allowed = split_schemes(p)
    if S > 1 and allowed is not None and scheme not in allowed:
        raise ValueError(f"scheme {scheme!r} is not a split-K layout of preset {p.name!r}: its cuBLASLt kernels split "
                         f"as {' or '.join(repr(x) for x in allowed)} (a custom Preset accepts any scheme)")
    if S == 1 and fast:
        if K % 16 == 0:
            return GemmPlan(K, True, 1, "none", ((torch.arange(K),),), True)
        L = -(-K // 16) * 16
        idx = torch.arange(L)
        return GemmPlan(K, True, 1, "none", ((torch.where(idx < K, idx, torch.full_like(idx, -1)),),), False)
    if K == 0:
        return GemmPlan(0, bool(fast), S, scheme, tuple(() for _ in range(S)), False)
    if scheme == "contiguous":
        chunks = []
        for ix in contiguous_index(K, S):         # each a multiple of 128 long
            if fast or len(ix) == 0:
                chunks.append((ix,) if len(ix) else ())
                continue
            cl = p.slow_chunk                     # hopper: 128, every k-tile its own chain
            nc = -(-len(ix) // cl)
            ix = torch.nn.functional.pad(ix, (0, nc * cl - len(ix)), value=-1)
            chunks.append(tuple(ix[j * cl:(j + 1) * cl] for j in range(nc)))
        return GemmPlan(K, bool(fast), S, scheme, tuple(chunks), False)
    idx = split_index(K, S)                       # [S, L], L a multiple of 64
    if fast:
        chunks = tuple((idx[s],) for s in range(S))
    else:
        cl = SLOW_CHUNK_TILES * TILE_K
        if S == 1 and p.slow_chunk != cl:
            cl = p.slow_chunk                     # custom presets: unsplit restart length from the preset
        L = idx.shape[1]
        nc = -(-L // cl)
        idx = torch.nn.functional.pad(idx, (0, nc * cl - L), value=-1).reshape(S, nc, cl)
        chunks = tuple(tuple(idx[s, j] for j in range(nc)) for s in range(S))
    return GemmPlan(K, bool(fast), S, scheme, chunks, False)


def reduce_partials(parts, scheme: str) -> torch.Tensor:
    """fp32 combination of the S partials (sequence of fp32 [M, N]) in cuBLAS's order."""
    S = len(parts)
    if S == 1:
        return parts[0]
    if scheme in ("serial", "contiguous"):
        acc = parts[0]
        for s in range(1, S):
            acc = acc + parts[s]
        return acc
    if scheme == "pairwise":
        h = -(-S // 2)
        acc = parts[S - 1]
        for i in range(h):
            slot = parts[i] + parts[i + h] if i + h < S - 1 else parts[i]
            acc = acc + slot
        return acc
    raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")


def _gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x[:, idx] with -1 -> zero code; a contiguous copy."""
    idx = idx.to(x.device)
    neg = idx < 0
    if not bool(neg.any()):
        return x[:, idx].contiguous()
    g = x[:, idx.clamp_min(0)]
    return torch.where(neg[None, :], torch.zeros((), dtype=x.dtype, device=x.device), g).contiguous()


def execute_plan(plan: GemmPlan, a_u8: torch.Tensor, b_u8: torch.Tensor, *, fmt_a: str, fmt_b: str,
                 chain_fn: ChainFn) -> torch.Tensor:
    """fp32 [M, N] accumulator of a [M, K] x b [N, K] (uint8 codes) under ``plan``, before alpha.

    Steps: sanitise non-finite codes (copy only if present); gather each chunk (none for an identity plan); one
    chain_fn call per chunk (every chain starts from +0); fp32 combination (slow chunks serially from +0, splits by
    scheme); IEEE non-finite patch with canonical NaN. K == 0 gives +0; M == 0 or N == 0 an empty result."""
    if a_u8.dtype != torch.uint8 or b_u8.dtype != torch.uint8:
        raise TypeError("execute_plan takes uint8 FP8 codes")
    if a_u8.dim() != 2 or b_u8.dim() != 2 or a_u8.shape[1] != b_u8.shape[1]:
        raise ValueError(f"expected a [M, K] and b [N, K]; got {tuple(a_u8.shape)} and {tuple(b_u8.shape)}")
    if a_u8.shape[1] != plan.K:
        raise ValueError(f"the plan is for K = {plan.K}, the operands have K = {a_u8.shape[1]}")
    if a_u8.device != b_u8.device:
        raise ValueError("a and b must be on the same device")
    M, N, K = a_u8.shape[0], b_u8.shape[0], plan.K
    dev = a_u8.device
    if M == 0 or N == 0 or K == 0:
        return torch.zeros(M, N, dtype=torch.float32, device=dev)
    a_s, a_nf = specials.sanitize(a_u8, fmt_a)
    b_s, b_nf = specials.sanitize(b_u8, fmt_b)
    if plan.identity:
        acc = chain_fn(a_s.contiguous(), b_s.contiguous())
    else:
        parts = []
        for chunks in plan.chunks:
            part = None
            for idx in chunks:
                c = chain_fn(_gather(a_s, idx), _gather(b_s, idx))
                if plan.fast:
                    part = c                                    # one chain per split
                else:
                    part = (torch.zeros_like(c) if part is None else part) + c   # serial from +0
            if part is None:                                    # an empty contiguous split: +0
                part = torch.zeros(M, N, dtype=torch.float32, device=dev)
            parts.append(part)
        acc = reduce_partials(parts, plan.scheme)
    if a_nf or b_nf:
        acc = specials.nonfinite_patch(acc, a_u8, b_u8, fmt_a, fmt_b)
    return acc
