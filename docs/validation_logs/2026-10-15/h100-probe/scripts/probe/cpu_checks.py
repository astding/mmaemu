"""CPU checks of the probe design (run before the rental; no GPU).

1. M1 embedding: under the ada, hopper and b200_mma_sync presets the embedded chain [cx, 0 x (K - 33), a] x
   [cy, 0, ..., b] equals the kit's definition chain(a, b, c = cx * cy) for every vector of the MM_cublas64* sets
   and every K used (the zero blocks carry c unchanged).
2. M3 patterns: under the hopper block model with slow-accumulation restarts every R elements (R = 32 ... 1024 and
   no restart = fast), chains added sequentially in fp32 RNE or pairwise, every pair of hypotheses gives a different
   prediction on the pairs (fp32), triples (fp32) and tie_bf16 (bf16) patterns of every M3 shape.
3. epilogue_ties premise: the kit's accumulator (4 nonzero products per row) is exact under the hopper preset.
usage: PROBE_SRC=<src> python cpu_checks.py"""
from __future__ import annotations

import itertools
import json
import sys
import warnings

import numpy as np

import m1_cvec
import m3_patterns as P
import pc  # noqa: F401  (sys.path: the source tree)

import torch  # noqa: E402
from mmaemu.model.chain import dot  # noqa: E402

warnings.simplefilter("ignore")
torch.set_num_threads(4)
PRESETS = ("ada", "hopper", "b200_mma_sync")
res: dict = {}


def check_m1():
    out = {}
    for S in ("MM_cublas64", "MM_cublas64_sm120"):
        av, bv, cg, _, _, _ = m1_cvec.load_set(S)
        a = torch.from_numpy(av.astype(np.float64))
        b = torch.from_numpy(bv.astype(np.float64))
        cx = torch.from_numpy(cg[:, 0].astype(np.float64))
        cy = torch.from_numpy(cg[:, 1].astype(np.float64))
        c = (cx * cy).float()
        for pre in PRESETS:
            ref = dot(a, b, c, preset=pre, fmt="e4m3").view(torch.int32)
            for K in (64, 1920, 4096, 8192):
                A = torch.zeros(len(a), K, dtype=torch.float64)
                B = torch.zeros(len(a), K, dtype=torch.float64)
                A[:, 0], B[:, 0], A[:, K - 32:], B[:, K - 32:] = cx, cy, a, b
                emb = dot(A, B, None, preset=pre, fmt="e4m3").view(torch.int32)
                out[f"{S}.{pre}.K{K}"] = int((emb != ref).sum())
    res["m1_embedding_mismatches"] = out
    return all(v == 0 for v in out.values())


def chains(a: torch.Tensor, b: torch.Tensor, R: int | None, combine: str, pre: str) -> torch.Tensor:
    """slow-accumulation hypothesis: chain restarts every R elements (None: one chain), chain results added in fp32
    RNE, sequentially in K order ('seq') or as a pairwise tree ('tree'). Returns fp32."""
    K = a.shape[-1]
    if R is None:
        return dot(a, b, None, preset=pre, fmt="e4m3")
    parts = [dot(a[:, k0:k0 + R], b[:, k0:k0 + R], None, preset=pre, fmt="e4m3") for k0 in range(0, K, R)]
    if combine == "seq":
        acc = torch.zeros_like(parts[0])
        for p in parts:
            acc = acc + p                      # fp32 RNE
        return acc
    while len(parts) > 1:
        nxt = [parts[i] + parts[i + 1] for i in range(0, len(parts) - 1, 2)]
        if len(parts) % 2:
            nxt.append(parts[-1])
        parts = nxt
    return parts[0]


def check_m3():
    hyps = [(None, "seq")] + [(R, c) for R in (32, 64, 128, 256, 512, 1024) for c in ("seq", "tree")]
    out = {}
    ok = True
    for (M, N, K) in P.SHAPES[:2] + ((256, 256, 8192),):
        for kind in P.KINDS:
            spec = P.patterns(kind, K)
            if K == 8192:
                spec = spec[::5]
            A = torch.from_numpy(P.dense_rows(spec, K))
            for gname, vec in P.groups(kind, K):
                B = torch.from_numpy(vec)[None, :].expand_as(A).contiguous()
                preds = {}
                for (R, comb) in hyps:
                    if R is not None and R >= K:
                        continue
                    y = chains(A, B, R, comb, "hopper")
                    if kind == "tie_bf16":
                        y = y.to(torch.bfloat16).float()
                    preds[(R, comb)] = y.view(torch.int32).clone()
                same = []
                for h1, h2 in itertools.combinations(preds, 2):
                    if torch.equal(preds[h1], preds[h2]):
                        # tree vs seq are the same plan when there are <= 2 chains in K
                        if h1[0] == h2[0] and h1[0] is not None and K // h1[0] <= 2:
                            continue
                        same.append(f"{h1}={h2}")
                key = f"{kind}.{gname}.K{K}"
                out[key] = dict(rows=len(spec), indistinguishable=same)
                # pairs / tie_bf16 have to separate the restart intervals; triples the combination order (seq vs
                # tree) at the same interval
                diff_r = [s for s in same if s.split("=")[0].split(",")[0] != s.split("=")[1].split(",")[0]]
                same = [s for s in same if s not in diff_r] if kind == "triples" else diff_r
                out[key]["indistinguishable"] = same
                ok &= not same
    res["m3"] = out
    return ok


def check_epilogue_premise():
    S = torch.arange(8192, 16384, dtype=torch.float64)
    v = S.long()
    digits = torch.stack([(v >> 12) * 16, (v >> 8) & 15, (v >> 4) & 15, v & 15], 1).double()
    A = torch.cat([digits, -digits])
    out = {}
    for n in range(4):
        bcol = torch.tensor([256.0, 256.0, 16.0, 1.0]) * 2.0 ** -n
        a = torch.zeros(len(A), 32, dtype=torch.float64)
        b = torch.zeros(len(A), 32, dtype=torch.float64)
        a[:, :4], b[:, :4] = A, bcol
        for pre in PRESETS:
            got = dot(a, b, None, preset=pre, fmt="e4m3")
            exact = (torch.cat([S, -S]) * 2.0 ** -n).float()
            out[f"{pre}.n{n}"] = int((got.view(torch.int32) != exact.view(torch.int32)).sum())
    res["epilogue_ties_acc_inexact"] = out
    return all(v == 0 for v in out.values())


if __name__ == "__main__":
    r = dict(m1=check_m1(), m3=check_m3(), epilogue=check_epilogue_premise())
    print(json.dumps(res, indent=1))
    print("RESULT", json.dumps(r))
    sys.exit(0 if all(r.values()) else 1)
