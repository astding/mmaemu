"""Live cuBLASLt split-K decisions just above mmaemu's split envelope (M*N > max split M*N of the setting).

For both table profiles (ada-rtx4060-torch, 1 MiB; ada-rtx4060-te, 4 MiB) and every setting (out dtype, fast, format
pair) whose envelope has split rows, query the heuristic at M*N = f * max_split_MN (N rounded up to a multiple of 16)
for f in 1.0..2.0 (step 0.02) and 2.25..4.0 (step 0.25), 16 values of M and 13 values of K. Writes sweep_env2.json
(one row per query) and prints the largest f that split.
"""
import itertools
import json
import time

import torch

from mmaemu.platform import cublaslt
from mmaemu.platform.profiles import get_profile
from mmaemu.platform.splitk_table import load_table

dev = torch.device("cuda", 0)
OUTS = ["float32", "bfloat16", "float16", "float8_e4m3fn", "float8_e5m2"]
PAIRS = [("e4m3", "e4m3"), ("e4m3", "e5m2"), ("e5m2", "e4m3")]
Ms = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1024, 2048, 4096, 8192)
Fs = [round(1.0 + 0.02 * i, 2) for i in range(51)] + [2.25, 2.5, 2.75, 3.0, 3.5, 4.0]
Ks = (1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 32768, 65536, 262144, 1048576, 4194304)
rows, t0 = [], time.time()
for prof_name in ("ada-rtx4060-torch", "ada-rtx4060-te"):
    p = get_profile(prof_name)
    t = load_table(p.splitk_table)
    for od, fast, (fa, fb) in itertools.product(OUTS, (True, False), PAIRS):
        if od == "float8_e5m2" and (fa, fb) == ("e4m3", "e4m3"):
            continue
        env, how = t.envelope(p.workspace_bytes, od, fast, fa, fb)
        if env is None or not env["split_rows"]:
            continue
        mx = env["max_split_MN"]
        for M, f, K in itertools.product(Ms, Fs, Ks):
            N = max(16, -(-int(mx * f / M) // 16) * 16)
            if M * N <= mx:
                N += 16
            r = cublaslt.heuristic(M, N, K, fmt_a=fa, fmt_b=fb, out_dtype=od, fast=fast,
                                   workspace_bytes=p.workspace_bytes, align=p.align, device=dev)
            rows.append([prof_name, od, fast, fa, fb, mx, M, N, K, r["status"], r["splitK"], r["custom"]])
json.dump(rows, open("sweep_env2.json", "w"))
sp = [r for r in rows if r[9] == 0 and r[10] > 1]
print("queries", len(rows), "status!=0", sum(r[9] != 0 for r in rows), "split", len(sp),
      "max f", max((r[6] * r[7] / r[5] for r in sp), default=None), "seconds", round(time.time() - t0, 1))
