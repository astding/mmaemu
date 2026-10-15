"""Shared helpers for the Ada (RTX 4060) Evo 2 golden capture. Only upstream code (evo2 / vortex / TE) touches the model."""
import hashlib, importlib.metadata as md, json, os, platform, subprocess, sys

import numpy as np
import torch

sys.path.insert(0, "/data/quant/evo2-anywhere/benchmarks")
import bench  # noqa: E402  (window definitions only: upstream test prompts + 8 kb / 2 kb windows)

OUT = "/data/quant/evo2_ada_goldens/data"


def env_meta():
    p = torch.cuda.get_device_properties(0)
    v = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda}
    for k in ("transformer_engine", "transformer-engine", "evo2", "vtx", "flash-attn", "nvidia-cublas-cu12", "triton"):
        try:
            v[k] = md.version(k)
        except md.PackageNotFoundError:
            pass
    smi = subprocess.run(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=driver_version,clocks.sm,power.limit",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return dict(gpu=p.name, capability=list(torch.cuda.get_device_capability(0)), sm_count=p.multi_processor_count,
                nvidia_smi=smi, versions=v,
                env={k: os.environ.get(k) for k in ("CUBLASLT_WORKSPACE_SIZE", "CUBLAS_WORKSPACE_CONFIG",
                                                     "CUBLASLT_LOG_LEVEL", "NVTE_DISABLE_NVRTC")})


def tokenizer():
    from vortex.model.tokenizer import CharLevelTokenizer
    return CharLevelTokenizer(512)


def windows(names):
    """names from bench.windows('all') (p0..p3 upstream test prompts, w8k, w2k), plus w4k = w8k[:4096] and
    '<name>_4k' = the first 4096 bp of <name> (for models whose official path does not fit 8 GB at full length)."""
    allw = dict(bench.windows("all"))
    allw["w4k"] = allw["w8k"][:4096]
    out = []
    for n in names:
        out.append((n, allw[n[:-3]][:4096] if n.endswith("_4k") else allw[n]))
    return out


def bits(t):
    """bf16/fp16 -> int16 bit pattern, fp32 -> int32."""
    return t.contiguous().view({torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}[t.dtype]).cpu().numpy()


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def save(tag, arrays, meta):
    np.savez_compressed(os.path.join(OUT, tag + ".npz"), **arrays)
    json.dump(meta, open(os.path.join(OUT, tag + ".json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("block_hashes", "env")})[:2000], flush=True)
