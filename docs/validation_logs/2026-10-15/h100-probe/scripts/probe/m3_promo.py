"""M3: slow-accumulation structure (chain restarts and their combination order) at the production shapes, with the
model-independent patterns of m3_patterns.py. Each GEMM holds up to M pattern rows; B is constant per column group.
Stored per GEMM: the value of the first column of every group ([rows, groups] bits) and, per row and group, how many
columns differ from it; rows with differing columns are stored in full (first 64).
torch: pairs / triples fp32 out, tie_bf16 bf16 out; fast and slow; unit scales.
TE (--te): the same through general_gemm with use_split_accumulator False / True.
usage: m3_promo.py [--te] [--shapes 2048x5760x1920,...] [--kinds pairs,triples,tie_bf16]"""
from __future__ import annotations

import argparse

import numpy as np

import m3_patterns as P
import pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--te", action="store_true")
    ap.add_argument("--shapes", default=",".join("x".join(map(str, s)) for s in P.SHAPES))
    ap.add_argument("--kinds", default=",".join(P.KINDS))
    a = ap.parse_args()
    import torch
    rec = pc.Rec("m3_te" if a.te else "m3_torch")
    api = pc.te_api() if a.te else None
    dev = torch.device("cuda:0")
    one = torch.ones((), device=dev)
    extra = dict(path="te" if a.te else "torch", te=pc.te_meta(api) if api else None, blocks={})
    for shp in a.shapes.split(","):
        M, N, K = (int(x) for x in shp.split("x"))
        for kind in a.kinds.split(","):
            spec = P.patterns(kind, K)
            grp = P.groups(kind, K)
            ng = len(grp)
            bounds = [(g * N) // ng for g in range(ng + 1)]
            B = torch.zeros(N, K, dtype=torch.float32)
            for g, (_, vec) in enumerate(grp):
                B[bounds[g]:bounds[g + 1]] = torch.from_numpy(vec.astype(np.float32))[None, :]
            B8 = B.to(torch.float8_e4m3fn)
            assert torch.equal(B8.float(), B)
            Bu = B8.view(torch.uint8).to(dev)
            Bd = Bu.view(torch.float8_e4m3fn)
            del B, B8
            out = torch.bfloat16 if kind == "tie_bf16" else torch.float32
            key = f"{kind}.M{M}N{N}K{K}"
            rec.arrays[f"{key}.spec"] = spec
            extra["blocks"][key] = dict(M=M, N=N, K=K, kind=kind, rows=len(spec), groups=[g for g, _ in grp],
                                        group_bounds=bounds, out=str(out).split(".")[-1], gemms=[])
            for r0 in range(0, len(spec), M):
                sub = spec[r0:r0 + M]
                A = np.zeros((M, K), dtype=np.float32)
                A[:len(sub)] = P.dense_rows(sub, K)
                At = torch.from_numpy(A)
                A8 = At.to(torch.float8_e4m3fn)
                assert torch.equal(A8.float(), At)
                Au = A8.view(torch.uint8).to(dev)
                Ad = Au.view(torch.float8_e4m3fn)
                del A, At, A8

                def keep(bits, nr=len(sub)):
                    bits = bits[:nr]
                    vals = np.stack([bits[:, bounds[g]] for g in range(ng)], 1)
                    var = np.stack([(bits[:, bounds[g]:bounds[g + 1]] != bits[:, bounds[g]:bounds[g] + 1]).sum(1)
                                    for g in range(ng)], 1).astype(np.int32)
                    d = {"vals": vals, "colvar": var}
                    bad = np.nonzero(var.any(1))[0][:64]
                    if len(bad):
                        d["bad_rows"] = bad.astype(np.int32)
                        d["bad_full"] = bits[bad]
                    return d
                modes = ([("fast", True), ("slow", False)] if not a.te else [("te", False), ("te_split", True)])
                for tag, flag in modes:
                    cid = f"{key}.r{r0}.{tag}"
                    if not a.te:
                        fn = (lambda f=flag: torch._scaled_mm(Ad, Bd.t(), scale_a=one, scale_b=one, out_dtype=out,
                                                              use_fast_accum=f))
                    else:
                        fn = (lambda s=flag: pc.te_gemm(api, Bu, Au, one, one, out, s))
                    rec.call(cid, fn, keep, dict(kind=kind, M=M, N=N, K=K, r0=r0, rows=len(sub), mode=tag))
                extra["blocks"][key]["gemms"].append(r0)
                del Au, Ad
            del Bu, Bd
            torch.cuda.empty_cache()
    rec.save(extra)


if __name__ == "__main__":
    main()
