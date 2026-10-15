"""CPU model used ONLY to choose discriminating inputs/subsets for the recordings (never as ground truth).

Exact Ada chain accumulator: the truth's vendored AVX-512 oracle csrc/fastref3.c, compiled privately into
$MMAEMU_CACHE_DIR/ws2_fastref (never into the truth dir; tcemu_ada_truth.fastacc.lib() is not called).
Split-K plan per docs/semantics.md (interleaved 64-wide k-tiles t mod S; slow: restart every 2 own tiles; serial /
pairwise fp32 reduction; alpha after the reduction) and epilogue (fp32(acc*alpha) from the exact
double product, then bf16/fp16 RNE or fp8 scale_result + saturation + RNE). Non-finite codes are not supported.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np

import common

SRC = common.TRUTH / "tcemu_ada_truth" / "csrc" / "fastref3.c"
_lib = None


def lib():
    global _lib
    if _lib is None:
        d = Path(os.environ.get("MMAEMU_CACHE_DIR", Path.home() / ".cache" / "mmaemu")) / "ws2_fastref"
        d.mkdir(parents=True, exist_ok=True)
        so = d / "libfastref3.so"
        if not so.exists() or so.stat().st_mtime < SRC.stat().st_mtime:
            subprocess.check_call(["gcc", "-O3", "-march=native", "-fopenmp", "-ffp-contract=off", "-fPIC", "-shared",
                                   str(SRC), "-o", str(so)])
        _lib = ctypes.CDLL(str(so))
        _lib.chain_gemm_avx.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int] * 10 + [ctypes.c_void_p]
    return _lib


FMT = {"e4m3": 0, "e5m2": 1}


def chain(A: np.ndarray, B: np.ndarray, fa: str, fb: str, restart: int = 0) -> np.ndarray:
    A = np.ascontiguousarray(A, np.uint8)
    B = np.ascontiguousarray(B, np.uint8)
    M, K = A.shape
    N = B.shape[0]
    out = np.zeros((M, N), np.float32)
    if M and N and K:
        lib().chain_gemm_avx(A.ctypes.data, B.ctypes.data, M, N, K, FMT[fa], FMT[fb], 16, 13, 13, -132, restart,
                             out.ctypes.data)
    return out


def split_index(K: int, S: int, contiguous: bool = False) -> list[np.ndarray]:
    T = -(-K // 64)
    tiles = [list(range(s, T, S)) for s in range(S)] if not contiguous else \
        [list(range(s * T // S, (s + 1) * T // S)) for s in range(S)]
    out = []
    for ts in tiles:
        idx = np.concatenate([np.arange(t * 64, t * 64 + 64) for t in ts]) if ts else np.zeros(0, np.int64)
        out.append(np.where(idx < K, idx, -1))
    return out


def _gather(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    g = np.zeros((X.shape[0], len(idx)), np.uint8)
    ok = idx >= 0
    g[:, ok] = X[:, idx[ok]]
    return g


def acc(A, B, fa, fb, *, fast: bool, S: int = 1, scheme: str = "none", contiguous: bool = False,
        slow_restart: int = 128) -> np.ndarray:
    """fp32 accumulator [M, N] of the cuBLAS plan (finite codes only)."""
    restart = 0 if fast else slow_restart
    if S <= 1:
        return chain(A, B, fa, fb, restart)
    parts = [chain(_gather(A, ix), _gather(B, ix), fa, fb, restart) for ix in split_index(A.shape[1], S, contiguous)]
    if scheme == "serial":
        r = parts[0]
        for p in parts[1:]:
            r = (r + p).astype(np.float32)
        return r
    h = -(-S // 2)
    r = parts[S - 1]
    for i in range(h):
        slot = (parts[i] + parts[i + h]).astype(np.float32) if i + h < S - 1 else parts[i]
        r = (r + slot).astype(np.float32)
    return r


def epilogue_bits(acc32: np.ndarray, sa: float, sb: float, out: str, scale_result: float | None = None) -> np.ndarray:
    import torch
    alpha = np.float32(np.float32(sa) * np.float32(sb))
    y = torch.from_numpy((acc32.astype(np.float64) * np.float64(alpha)).astype(np.float32))
    if out.startswith("float8"):
        if scale_result is not None:
            y = y * torch.tensor(np.float32(scale_result))
        mx = 448.0 if out == "float8_e4m3fn" else 57344.0
        y = y.clamp(-mx, mx)
    t = y.to(getattr(torch, out))
    return common.tbits(t)
