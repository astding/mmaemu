"""E-pass test vectors.
  prims8_tiles(): exactly the tiles of the other agent's test_prims8.py (same seeds): At [T,16,16], Bt [T,8,16] codes.
  pairs_from_tiles(At, Bt): all (row, col) 16-term blocks of the tiles -> A [T*128,16], B [T*128,16].
  structural(): designed worst cases for every block exponent E in [-12,16] and every count n of terms attaining it
                (n = 1..16), with the other terms zero / one binade-level below (E-1) / at the floor (-12) / random
                below, the max terms placed first, last, or at random positions (30 permutations)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

W = Path(__file__).resolve().parent
sys.path.insert(0, str(W))


def prims8_tiles():
    import test_prims as tp
    tp.dev = "cpu"  # the stubbed epass calls only move tensors; keep them on the CPU
    import test_prims8 as t8  # reuses the other agent's generator (copied verbatim into this dir)
    from test_prims import VALID
    A, B = t8.rows_like_test_prims()
    R = A.shape[0]
    T = R // 8
    A, B = A[:T * 8], B[:T * 8]
    g = torch.Generator().manual_seed(23)
    extra = VALID[torch.randint(0, len(VALID), (T, 8, 16), generator=g)]
    At = torch.cat([A.reshape(T, 8, 16), extra], 1)
    Bt = B.reshape(T, 8, 16)
    return At.long(), Bt.long()


def pairs_from_tiles(At, Bt):
    T = At.shape[0]
    A = At[:, :, None, :].expand(T, 16, 8, 16).reshape(-1, 16)
    B = Bt[:, None, :, :].expand(T, 16, 8, 16).reshape(-1, 16)
    return A, B


def byte_of(ea):  # E4M3 normal byte with exponent ea (mantissa 0), ea in [-6, 8]
    return (ea + 7) << 3


def split(s):  # exponent sum s in [-12, 16] -> (ea, eb) both in [-6, 8]
    ea = max(-6, s - 8)
    return ea, s - ea


def structural(seed=5, nperm=30):
    g = torch.Generator().manual_seed(seed)
    rowsA, rowsB, meta = [], [], []
    for E in range(-12, 17):
        for n in range(1, 17):
            for pat in ("zero", "Em1", "floor", "rand"):
                if pat == "Em1" and E - 1 < -12:
                    continue
                base_s = []
                for k in range(16):
                    if k < n:
                        base_s.append(E)
                    elif pat == "zero":
                        base_s.append(None)
                    elif pat == "Em1":
                        base_s.append(E - 1)
                    elif pat == "floor":
                        base_s.append(-12 if E > -12 else None)
                    else:
                        lo = -12
                        if E - 1 < lo:
                            base_s.append(None)
                        else:
                            base_s.append(int(torch.randint(lo, E, (1,), generator=g)))
                perms = [list(range(16)), list(range(16))[::-1]]
                perms += [torch.randperm(16, generator=g).tolist() for _ in range(nperm)]
                for pm in perms:
                    a, b = [0] * 16, [0] * 16
                    for k in range(16):
                        s = base_s[pm[k]]
                        if s is not None:
                            ea, eb = split(s)
                            a[k], b[k] = byte_of(ea), byte_of(eb)
                    rowsA.append(a); rowsB.append(b); meta.append((E, n, pat))
    return torch.tensor(rowsA, dtype=torch.int64), torch.tensor(rowsB, dtype=torch.int64), meta
