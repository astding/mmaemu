"""Shape lists of the H100 split-K sweep (CPU, deterministic; run once before the bundle is built, so the dry runs
and the H100 use the same files). R = 122,880, the largest split M*N of heur_h100pcie_lt12.8.3.14 at 1 MiB.

p1  envelope: M in MS1 (24 values, 1..16384), M*N = f * R with f = 2^(i/16) for i = -64..80 (1/16..32) and the
    tail 48, 64, 128 (N rounded up to a multiple of 16, 16..2^20), K in KS1 (33 values, 1,024..4,194,304, 9 of them
    not a multiple of 128). Queried for all 30 settings of each table profile's (workspace, alignment).
p2  K dense: M in {1, 16, 128, 1024} x f in {0.25, 0.5, 1, 1.5, 2.5}, K = 1,024..16,384 step 16, 16,512..65,536
    step 128, then 8 values per octave up to 2^22 (multiples of 16).
p3  model rows: the (K, N) of the projections of 9 models (Evo 2 1B/7B/40B, Llama 2 7B, Llama 3 8B, Mistral 7B,
    Qwen2.5 7B, Llama 3 70B, Llama 3.2 1B) x every M up to max(256, ceil(8 * 245,760 / N)) (8 times the TE table's
    largest split M*N), then step 16 to 8,192 and step 256 to 16,384. Queried for both alignments of each workspace
    (the table key has no alignment; the export asserts invariance).
p4  sealed held-out (seed SEED): 200,000 random rows per table profile and 50,000 at workspace 0 (setting uniform
    over its 30, M log-uniform on 1..16384, f = M*N / R log-uniform on 1/4..128, K log-uniform on 1,024..4,194,304,
    half of them rounded to a multiple of 128, the rest to 16) and 24 random M (log-uniform) for every (K, N) of 10
    models that are not in p3, every setting of both table profiles. One row per (shape, setting); analysed only after the envelope fitted on p1-p3 is committed.
usage: make_shapes.py OUT_DIR   (writes p1.json .. p4.json and index.json with counts and sha256)"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

R = 122_880
SEED = 20261016
NMAX = 1 << 20
MS1 = [1, 2, 4, 8, 16, 17, 24, 32, 48, 64, 96, 100, 128, 192, 256, 384, 512, 768, 1000, 1024, 2048, 4096, 8192,
       16384]
KS1 = [1024, 1040, 1536, 2048, 2064, 3072, 3600, 4096, 5120, 5744, 6144, 8192, 8208, 10240, 12288, 13840, 14336,
       16384, 16400, 20480, 24576, 28672, 32768, 49152, 65536, 65552, 131072, 262144, 524288, 1048576, 1048592,
       2097152, 4194304]
F1 = [2.0 ** (i / 16) for i in range(-64, 81)] + [48.0, 64.0, 128.0]
# (model, layer, K, N): out = x[M, K] @ W[N, K]^T
P3_MODELS = {
    "Evo 2 1B": [("in_proj", 1920, 5760), ("out_proj", 1920, 1920), ("mlp_up", 1920, 10240), ("mlp_down", 5120, 1920)],
    "Evo 2 7B": [("in_proj", 4096, 12288), ("out_proj", 4096, 4096), ("mlp_up", 4096, 22528), ("mlp_down", 11264, 4096)],
    "Evo 2 40B": [("in_proj", 8192, 24576), ("out_proj", 8192, 8192), ("mlp_up", 8192, 45056), ("mlp_down", 22528, 8192)],
    "Llama 2 7B": [("qkv", 4096, 12288), ("o", 4096, 4096), ("gate_up", 4096, 22016), ("down", 11008, 4096)],
    "Llama 3 8B": [("qkv", 4096, 6144), ("o", 4096, 4096), ("gate_up", 4096, 28672), ("down", 14336, 4096)],
    "Mistral 7B": [("qkv", 4096, 6144), ("o", 4096, 4096), ("gate_up", 4096, 28672), ("down", 14336, 4096)],
    "Qwen2.5 7B": [("qkv", 3584, 4608), ("o", 3584, 3584), ("gate_up", 3584, 37888), ("down", 18944, 3584)],
    "Llama 3 70B": [("qkv", 8192, 10240), ("o", 8192, 8192), ("gate_up", 8192, 57344), ("down", 28672, 8192)],
    "Llama 3.2 1B": [("qkv", 2048, 3072), ("o", 2048, 2048), ("gate_up", 2048, 16384), ("down", 8192, 2048)],
}
# held-out models (none of their (K, N) is in p3)
P4_MODELS = {
    "GPT-2 XL": [("qkv", 1600, 4800), ("o", 1600, 1600), ("up", 1600, 6400), ("down", 6400, 1600)],
    "Phi-3 mini": [("qkv", 3072, 9216), ("o", 3072, 3072), ("gate_up", 3072, 16384), ("down", 8192, 3072)],
    "Gemma 2 9B": [("qkv", 3584, 8192), ("o", 4096, 3584), ("gate_up", 3584, 28672), ("down", 14336, 3584)],
    "Qwen2.5 14B": [("qkv", 5120, 7168), ("o", 5120, 5120), ("gate_up", 5120, 27648), ("down", 13824, 5120)],
    "Falcon 7B": [("qkv", 4544, 4672), ("o", 4544, 4544), ("up", 4544, 18176), ("down", 18176, 4544)],
    "Llama 3.1 405B": [("qkv", 16384, 18432), ("o", 16384, 16384), ("gate_up", 16384, 106496), ("down", 53248, 16384)],
    "DeepSeek-V3 dense": [("gate_up", 7168, 36864), ("down", 18432, 7168)],
    "DeepSeek-V3 expert": [("gate_up", 7168, 4096), ("down", 2048, 7168)],
    "Mixtral 8x22B": [("qkv", 6144, 8192), ("o", 6144, 6144), ("gate_up", 6144, 32768), ("down", 16384, 6144)],
    "BERT large": [("qkv", 1024, 3072), ("o", 1024, 1024), ("up", 1024, 4096), ("down", 4096, 1024)],
}
OUTS = ["float32", "bfloat16", "float16", "float8_e4m3fn", "float8_e5m2"]
PAIRS = [("e4m3", "e4m3"), ("e4m3", "e5m2"), ("e5m2", "e4m3")]
PROFILES = [(1 << 20, 0, 200_000), (32 << 20, 256, 200_000)]


def n_for(M: int, mn: float) -> int:
    return int(min(NMAX, max(16, -(-int(math.ceil(mn / M)) // 16) * 16)))


def p1():
    s = {(M, n_for(M, f * R), K) for M in MS1 for f in F1 for K in KS1}
    return sorted(s)


def p2_K():
    ks = list(range(1024, 16385, 16)) + list(range(16512, 65537, 128))
    ks += [int(round(65536 * 2 ** (i / 8) / 16)) * 16 for i in range(1, 49)]
    return sorted(set(ks))


def p2():
    mn = [(M, n_for(M, f * R)) for M in (1, 16, 128, 1024) for f in (0.25, 0.5, 1.0, 1.5, 2.5)]
    return sorted({(M, N, K) for (M, N) in mn for K in p2_K()})


def p3_M(N: int):
    dense = max(256, -(-8 * 245_760 // N))
    return sorted(set(range(1, dense + 1)) | set(range(16, 8193, 16)) | set(range(8448, 16385, 256)))


def p3():
    kn = sorted({(K, N) for layers in P3_MODELS.values() for (_, K, N) in layers})
    return sorted({(M, N, K) for (K, N) in kn for M in p3_M(N)})


def p4():
    """rows [M, N, K, ws, align, out, fa, fb, fast]"""
    rng = np.random.default_rng(SEED)
    rows = []
    sts = [(o, fa, fb, f) for o in OUTS for (fa, fb) in PAIRS for f in (1, 0)]
    for ws, al, n in PROFILES + [(0, 0, 50_000)]:
        for _ in range(n):
            o, fa, fb, f = sts[rng.integers(len(sts))]
            M = int(round(math.exp(rng.uniform(0, math.log(16384)))))
            fr = 2.0 ** rng.uniform(-2, 7)
            N = n_for(M, fr * R)
            k = math.exp(rng.uniform(math.log(1024), math.log(4194304)))
            q = 128 if rng.random() < 0.5 else 16
            K = int(min(4194304, max(1024, round(k / q) * q)))
            rows.append([M, N, K, ws, al, o, fa, fb, f])
    p3kn = {(K, N) for layers in P3_MODELS.values() for (_, K, N) in layers}
    kn = sorted({(K, N) for layers in P4_MODELS.values() for (_, K, N) in layers})
    assert not (set(kn) & p3kn), set(kn) & p3kn
    for (K, N) in kn:
        Ms = sorted({int(round(math.exp(rng.uniform(0, math.log(16384))))) for _ in range(24)})
        for ws, al, _ in PROFILES:
            for (o, fa, fb, f) in sts:
                rows += [[M, N, K, ws, al, o, fa, fb, f] for M in Ms]
    return rows


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    index = dict(R=R, seed=SEED, p3_models=P3_MODELS, p4_models=P4_MODELS, parts={})
    for name, fn in (("p1", p1), ("p2", p2), ("p3", p3), ("p4", p4)):
        rows = [list(r) for r in fn()]
        p = out / f"{name}.json"
        p.write_text(json.dumps(rows, separators=(",", ":")))
        index["parts"][name] = dict(rows=len(rows), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        print(name, len(rows), index["parts"][name]["sha256"][:16])
    (out / "index.json").write_text(json.dumps(index, indent=1) + "\n")


if __name__ == "__main__":
    main()
