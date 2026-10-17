"""GO #1 follow-up: the members of the MM_cublas64 / MM_cublas64_sm120 model families that share the H100's fitted
core (flat32, early chain, frac 13, frac_out 13, rz, denormal products, clamped subnormal inputs) and differ only in
the edge parameters zero_exp / min_exp / c_exp_sub, per c mode, and whether their predictions differ on the set.
usage: edge_variants.py"""
from __future__ import annotations

import numpy as np

import m1_cvec as M


def main():
    for S in ("MM_cublas64", "MM_cublas64_sm120"):
        _, _, _, P, models, _ = M.load_set(S)
        for cm in ("first", "late_rne"):
            mem = [(i, m) for i, m in enumerate(models)
                   if (m["groups"], m["chain"], m["c_mode"], m["frac"], m["frac_out"], m["rnd_out"], m["align"],
                       m["denorm_prd"], m["in_sub"]) == ("flat32", "early", cm, 13, 13, "rz", "rz", True, "clamp")]
            same = all(np.array_equal(P[mem[0][0]], P[i]) for i, _ in mem) if mem else None
            print(S, cm, "members:", len(mem), "edge variants:",
                  [(m["zero_exp"], m["min_exp"], m["c_exp_sub"]) for _, m in mem],
                  "identical predictions on every vector:", same)


if __name__ == "__main__":
    main()
