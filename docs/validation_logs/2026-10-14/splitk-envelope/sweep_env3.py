"""Dense M search: live cuBLASLt split-K decisions at M*N = 1.25-2.0 x the largest split M*N of each setting.

As sweep_env2.py, with M = 16..1024 (step 16), f = 1.25..2.0 (step 0.025) and K in {3072, 4096, 6144, 8192, 16384,
32768}. Writes sweep_env3.json (one row per query) and prints the largest f that split.
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
Ms = tuple(range(16, 1025, 16))
Fs = [round(1.25 + 0.025 * i, 3) for i in range(31)]
Ks = (3072, 4096, 6144, 8192, 16384, 32768)
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
json.dump(rows, open("sweep_env3.json", "w"))
sp = [r for r in rows if r[9] == 0 and r[10] > 1]
print("queries", len(rows), "status!=0", sum(r[9] != 0 for r in rows), "split", len(sp),
      "max f", max((r[6] * r[7] / r[5] for r in sp), default=None), "seconds", round(time.time() - t0, 1))
