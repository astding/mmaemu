"""Choose the most discriminating output positions of a recorded split-K / slow-accumulation GEMM (used by
tools/export_hw_mini.py to keep hw_mini within its 20 MiB budget with 1x1 subsets).

For candidate output positions (m, n) of a recorded output, the hardware bits are predicted under the recorded plan
(a sanity check of the export: it must reproduce the hardware) and under rival plans (the plan mutants of the
catalogue): unsplit, serial <-> pairwise, contiguous instead of interleaved k-tiles, S - 1 / S + 1, slow restart every
1 or 4 of a split's tiles, fast restart every 2048, alpha applied before the reduction. The chosen position maximises
the number of rivals whose output differs from the hardware bits.

Self-contained (torch only; the chain model of tools/mutants/impl.py, see docs/semantics.md); it does not import
mmaemu, so it is independent of the implementation under test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import impl as M  # noqa: E402

TD8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
OD = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16,
      "float8_e4m3fn": torch.float8_e4m3fn, "float8_e5m2": torch.float8_e5m2}
TILE = 64


def _f32(bits: int) -> torch.Tensor:
    return torch.from_numpy(np.array([bits], np.uint32).view(np.int32).copy()).view(torch.float32).reshape(())


def _vals(codes: np.ndarray, fmt: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(codes, dtype=np.uint8).copy()).view(TD8[fmt]).double()


def _chunks(K, fast, S, scheme, rival):
    """Index lists (per split, per chunk) of a plan; -1 = zero padding."""
    if S == 1:
        idx = torch.arange(K)
        if fast:
            size = rival.get("fast_restart")
            return [[idx]] if not size else [[idx[j:j + size] for j in range(0, K, size)]]
        size = rival.get("slow_chunk", 128)
        return [[idx[j:j + size] for j in range(0, K, size)]]
    T = -(-K // TILE)
    Tp = -(-T // S) * S
    ar = torch.arange(Tp * TILE)
    kk = (ar.reshape(S, -1) if rival.get("contiguous") else
          ar.reshape(Tp // S, S, TILE).permute(1, 0, 2).reshape(S, -1))
    kk = torch.where(kk < K, kk, torch.full_like(kk, -1))
    out = []
    for s in range(S):
        row = kk[s]
        if fast:
            out.append([row])
        else:
            size = TILE * rival.get("slow_tiles", 2)
            out.append([row[j:j + size] for j in range(0, row.numel(), size)])
    return out


CHAIN = None     # optional fast chain on codes: fn(a_codes [P, L] uint8, b_codes [P, L], fa, fb) -> float32 [P]


def _chain_batch(A, B, idxs, fa, fb):
    """A [P, K], B [P, K] (float64 values, or uint8 codes when CHAIN is set) of P (row, col) pairs; idxs: list of
    G equal-length index vectors -> fp32 [G, P] chains from +0 over each index vector."""
    G, P = len(idxs), A.shape[0]
    if CHAIN is not None:
        res = []
        for idx in idxs:
            i = idx.numpy()
            ok = i >= 0
            a = np.zeros((P, -(-len(i) // 16) * 16), np.uint8)
            b = np.zeros_like(a)
            a[:, :len(i)][:, ok] = A[:, i[ok]]
            b[:, :len(i)][:, ok] = B[:, i[ok]]
            res.append(torch.from_numpy(np.asarray(CHAIN(a, b, fa, fb), np.float32)))
        return torch.stack(res)
    ii = torch.stack([torch.where(i >= 0, i, torch.full_like(i, A.shape[1])) for i in idxs])      # [G, L]
    Ap = torch.nn.functional.pad(A, (0, 1))[:, ii].permute(1, 0, 2).reshape(G * P, -1)
    Bp = torch.nn.functional.pad(B, (0, 1))[:, ii].permute(1, 0, 2).reshape(G * P, -1)
    L = Ap.shape[1]
    if L % 16:
        Ap = torch.nn.functional.pad(Ap, (0, 16 - L % 16))
        Bp = torch.nn.functional.pad(Bp, (0, 16 - L % 16))
    c0 = torch.zeros(G * P, dtype=torch.float64)
    return M._chain(Ap, Bp, c0, M.CHAIN_BASE, fa, fb).float().reshape(G, P)


def predict(A, B, fa, fb, fast, S, scheme, sa, sb, sr, od, rival=None):
    """A, B: [P, K] float64 values (or uint8 codes when CHAIN is set) of P (row, col) pairs."""
    """Output bits of P pairs under a plan (rival knobs: contiguous, slow_tiles, slow_chunk, fast_restart,
    swap_scheme, alpha_before)."""
    rival = rival or {}
    if rival.get("swap_scheme") and S > 1:
        scheme = {"serial": "pairwise", "pairwise": "serial"}[scheme]
    K = A.shape[1]
    ch = _chunks(K, fast, S, scheme, rival)
    parts = [None] * len(ch)
    for j in range(max(len(c) for c in ch)):
        groups = [(s, c[j]) for s, c in enumerate(ch) if j < len(c)]
        by_len = {}
        for s, idx in groups:
            by_len.setdefault(idx.numel(), []).append((s, idx))
        for _, grp in by_len.items():
            res = _chain_batch(A, B, [idx for _, idx in grp], fa, fb)
            for (s, _), r in zip(grp, res):
                parts[s] = (torch.zeros_like(r) + r) if parts[s] is None else parts[s] + r
    sa_t, sb_t = _f32(sa), _f32(sb)
    sr_t = None if sr is None else _f32(sr)
    apply = M.epilogue_override(None, {})
    if rival.get("alpha_before") and S > 1:
        alpha = (sa_t * sb_t).double()
        y = M._combine_splits([(p.double() * alpha + 0.0).float() for p in parts], scheme)
        acc = y
        out = apply(acc, torch.tensor(1.0), torch.tensor(1.0), out_dtype=OD[od], scale_result=sr_t)
    else:
        acc = M._combine_splits(parts, scheme)
        out = apply(acc, sa_t, sb_t, out_dtype=OD[od], scale_result=sr_t)
    t = out.contiguous()
    iv = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()]
    nv = {1: np.uint8, 2: np.uint16, 4: np.uint32}[t.element_size()]
    return t.view(iv).numpy().view(nv)


def rivals(fast, S, scheme):
    R = []
    if S > 1:
        R += [dict(name="unsplit", S=1), dict(name="scheme_swapped", swap_scheme=True), dict(name="contiguous", contiguous=True),
              dict(name="S-1", S=S - 1) if S > 2 else None, dict(name="S+1", S=S + 1),
              dict(name="alpha_before", alpha_before=True)]
        if not fast:
            R += [dict(name="slow_restart_1tile", slow_tiles=1), dict(name="slow_restart_4tiles", slow_tiles=4)]
    else:
        R += ([dict(name="fast_restart2048", fast_restart=2048)] if fast else
              [dict(name="slow_chunk64", slow_chunk=64), dict(name="slow_chunk256", slow_chunk=256)])
    return [r for r in R if r]


def choose(a_rows: np.ndarray, b_rows: np.ndarray, pairs, hw_bits, *, fa, fb, fast, S, scheme, sa, sb, sr, od):
    """pairs: list of (i, j) indices into a_rows / b_rows; hw_bits: the hardware bits at those pairs.
    Returns (best pair index, details) or (None, details) if the recorded plan does not reproduce the hardware."""
    finite = []
    for k, (i, j) in enumerate(pairs):
        ra = a_rows[i].astype(np.int16) & 0x7F
        rb = b_rows[j].astype(np.int16) & 0x7F
        nf_a = (ra == 0x7F) if fa == "e4m3" else (ra >= 0x7C)
        nf_b = (rb == 0x7F) if fb == "e4m3" else (rb >= 0x7C)
        if not (nf_a.any() or nf_b.any()):
            finite.append(k)
    if not finite:
        return None, dict(reason="no finite candidate")
    P = [pairs[k] for k in finite]
    A = np.stack([a_rows[i] for i, _ in P]).astype(np.uint8)
    B = np.stack([b_rows[j] for _, j in P]).astype(np.uint8)
    if CHAIN is None:
        A, B = _vals(A, fa), _vals(B, fb)
    hw = np.asarray([hw_bits[k] for k in finite])
    base = predict(A, B, fa, fb, fast, S, scheme, sa, sb, sr, od)
    ok = base == hw
    score = np.zeros(len(P), np.int64)
    names = []
    for r in rivals(fast, S, scheme):
        rS = r.get("S", S)
        pr = predict(A, B, fa, fb, fast, rS, scheme if rS > 1 else "none", sa, sb, sr, od, r)
        score += (pr != hw)
        names.append(r["name"])
    score = np.where(ok, score, -1)
    best = int(np.argmax(score))
    if score[best] < 0:
        return None, dict(reason="the recorded plan does not reproduce the hardware at any candidate",
                          candidates=len(P))
    return finite[best], dict(rivals=names, distinguished=int(score[best]), of=len(names),
                              model_reproduces=int(ok.sum()), candidates=len(P))
