"""M3 probe patterns: where the slow-accumulation chain restarts (promotion to the fp32 accumulator), and in what
order the chains are added. Shared by the GPU recorder (m3_promo.py) and the CPU checks / analysis.

Every row of A has at most three nonzero E4M3 values; all rows of B in one column group are the same vector, so every
column of a group computes the same dot product (any column-to-column difference is itself a finding: tile-dependent
K partition, e.g. stream-K).

pairs     big product L at k = 32 p (block p), small product s at k = 32 j + off. In one chain the small product
          is truncated away (the chain keeps ~13 fraction bits below the block maximum); across a restart it survives
          the fp32 addition. Column groups: g0 B = 1 (L = 2^8, s = 2^-6 / 2^-9: s/L = 2^-14 / 2^-17), g1 B = 16 at
          block starts, 2^-2 elsewhere (L = 2^12, s = 2^-8 / 2^-11: 2^-20 / 2^-23). fp32 out.
triples   L = 2^16 at block p, two products 2^-8 = ulp(2^16)/2 at blocks j1 < j2 (offset 31). Adding them to L one at a
          time ties twice (RNE keeps 2^16); adding them to each other first gives 2^16 + 2^-7. fp32 out.
tie_bf16  256 at 32 p, 1 at 32 p + 1 (257 = a bf16 tie), 2^-6 at 32 j + 31. One chain: 257 -> bf16 256; a restart
          between them: 257 + 2^-6 -> bf16 258. bf16 out (the Evo 2 output type)."""
from __future__ import annotations

import numpy as np

ROWS = 2048           # rows per GEMM (the production M); smaller M uses M rows per GEMM


def _p_full(nb: int) -> list[int]:
    return list(range(nb)) if nb <= 128 else list(range(0, nb, 4)) + [nb - 1]


def sparse_blocks(nb: int) -> list[int]:
    if nb <= 64:
        return list(range(nb))
    step = 4 if nb <= 128 else 8
    return sorted(set(range(16)) | set(range(16, nb, step)) | {nb - 1})


def groups(kind: str, K: int) -> list[tuple[str, np.ndarray]]:
    k = np.arange(K)
    start = (k % 32) == 0
    if kind == "pairs":
        return [("g0", np.ones(K)), ("g1", np.where(start, 16.0, 0.25))]
    if kind == "triples":
        return [("g2", np.where(start, 256.0, 2.0 ** -4))]
    if kind == "tie_bf16":
        return [("g0", np.ones(K))]
    raise ValueError(kind)


def patterns(kind: str, K: int) -> np.ndarray:
    """[R, 6] float64 rows (pos0, val0, pos1, val1, pos2, val2); pos -1 = unused. Rows 0 and 1 are controls (all
    zero; L alone at block 0)."""
    nb = K // 32
    rows = [(-1, 0, -1, 0, -1, 0)]
    if kind == "pairs":
        rows.append((0, 256.0, -1, 0, -1, 0))
        for p in _p_full(nb):
            for j in range(nb):
                rows.append((32 * p, 256.0, 32 * j + 31, 2.0 ** -6, -1, 0))
        for p in sorted({0, 1, nb // 2}):
            for off, t in ((16, 2.0 ** -9), (31, 2.0 ** -9)):
                for j in range(nb):
                    rows.append((32 * p, 256.0, 32 * j + off, t, -1, 0))
    elif kind == "triples":
        rows.append((0, 256.0, -1, 0, -1, 0))
        S = sparse_blocks(nb)
        for p in sorted({0, 1, nb // 2, nb - 1}):
            for x, j1 in enumerate(S):
                for j2 in S[x + 1:]:
                    rows.append((32 * p, 256.0, 32 * j1 + 31, 2.0 ** -4, 32 * j2 + 31, 2.0 ** -4))
    elif kind == "tie_bf16":
        rows.append((0, 256.0, 1, 1.0, -1, 0))
        for p in sorted({0, 1, 2, 3, nb // 2, nb - 1}):
            for j in range(nb):
                rows.append((32 * p, 256.0, 32 * p + 1, 1.0, 32 * j + 31, 2.0 ** -6))
    else:
        raise ValueError(kind)
    return np.array(rows, dtype=np.float64)


def dense_rows(spec: np.ndarray, K: int) -> np.ndarray:
    """float64 [R, K] A rows of a pattern block."""
    A = np.zeros((len(spec), K))
    for c in (0, 2, 4):
        m = spec[:, c] >= 0
        A[np.nonzero(m)[0], spec[m, c].astype(int)] = spec[m, c + 1]
    return A


SHAPES = ((2048, 5760, 1920), (2048, 12288, 4096), (2048, 24576, 8192), (1024, 1024, 4096))
KINDS = ("pairs", "triples", "tie_bf16")
