"""Dense live sweep of cuBLASLt's split-K boundary around mmaemu's split envelope (RTX 4060)."""
import json, itertools, time
import torch
from mmaemu.platform import cublaslt
from mmaemu.platform.profiles import get_profile
from mmaemu.platform.splitk_table import load_table

dev = torch.device("cuda", 0)
OUTS = ["float32", "bfloat16", "float16", "float8_e4m3fn", "float8_e5m2"]
PAIRS = [("e4m3", "e4m3"), ("e4m3", "e5m2"), ("e5m2", "e4m3")]
Ms = (16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096, 8192)
Fs = [round(1.0 + 0.1 * i, 2) for i in range(31)] + [5, 6, 8, 12, 16]
Ks = (2048, 16384, 65536, 262144, 1048576, 4194304)
rows, t0 = [], time.time()
for prof_name in ("ada", "ada-te"):
    p = get_profile(prof_name)
    t = load_table(p.splitk_table)
    for od, fast, (fa, fb) in itertools.product(OUTS, (True, False), PAIRS):
        if od == "float8_e5m2" and (fa, fb) == ("e4m3", "e4m3"):
            continue
        env, how = t.envelope(p.workspace_bytes, od, fast, fa, fb)
        if env is None or not env["split_rows"]:
            continue
        mx, mk = env["max_split_MN"], env["min_split_K"]
        q = lambda M, N, K: cublaslt.heuristic(M, N, K, fmt_a=fa, fmt_b=fb, out_dtype=od, fast=fast,
                                               workspace_bytes=p.workspace_bytes, align=p.align, device=dev)
        for M, f, K in itertools.product(Ms, Fs, Ks):
            N = max(16, -(-int(mx * f / M) // 16) * 16)
            if M * N <= mx:
                N += 16
            r = q(M, N, K)
            rows.append([prof_name, od, fast, fa, fb, mx, mk, M, N, K, r["status"], r["splitK"], r["custom"]])
        # K below the smallest split K, M*N inside
        for K in range(16, mk, 16):
            for M in (16, 32, 64, 128, 256):
                for N in (16, 32, 64, 128, 256, 512, 1024):
                    if M * N > mx:
                        continue
                    r = q(M, N, K)
                    rows.append([prof_name, od, fast, fa, fb, mx, mk, M, N, K, r["status"], r["splitK"], r["custom"]])
json.dump(rows, open("sweep_env.json", "w"))
print("queries", len(rows), "seconds", round(time.time() - t0, 1))
