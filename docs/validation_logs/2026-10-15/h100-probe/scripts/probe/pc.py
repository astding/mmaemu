"""Shared helpers of the H100 bit-exactness probe (Phase A).

Every recorded call runs twice. Run 1 is under torch.profiler (CUDA kernel names); both runs log the cuBLASLt
matmul configuration through the logger callback (tools/record/common.LtCapture, CUBLASLT_LOG_LEVEL=2) and the
sha256 of the full output bits. keep(bits) chooses what of run 1 is stored (default: everything).
Env: PROBE_SRC = the mmaemu source tree, PROBE_OUT = this step's output directory."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

SRC = Path(os.environ["PROBE_SRC"]).resolve()
for _p in (SRC / "tools" / "record", SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import common  # noqa: E402  (tools/record/common.py of the source tree)

OUT = Path(os.environ.get("PROBE_OUT") or ".").resolve()
ENV_KEYS = ("CUBLASLT_WORKSPACE_SIZE", "CUBLASLT_LOG_LEVEL", "CUBLASLT_LOG_FILE", "NVTE_DISABLE_NVRTC",
            "MMAEMU_CUBLASLT_HEADER", "CUDA_VISIBLE_DEVICES")


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def f32(x: float):
    """python float -> fp32 bits and value (scales are passed as fp32 tensors)."""
    v = np.float32(x)
    return int(v.view(np.uint32)), float(v)


class Rec:
    def __init__(self, step: str):
        import torch
        import mmaemu
        self.torch = torch
        self.step = step
        OUT.mkdir(parents=True, exist_ok=True)
        torch.cuda.init()
        self.cap = common.LtCapture()
        self.rows: list[dict] = []
        self.arrays: dict = {}
        self.t0 = time.monotonic()
        self.meta = dict(step=step, argv=sys.argv, started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         hardware=common.hardware(), src=str(SRC), mmaemu_file=mmaemu.__file__,
                         env={k: os.environ.get(k) for k in ENV_KEYS})

    def call(self, cid: str, fn, keep=None, meta=None, reps: int = 2):
        torch = self.torch
        row = dict(id=cid, meta=meta or {}, runs=[])
        first = None
        for rep in range(reps):
            r: dict = {}
            torch.cuda.synchronize()
            mk = self.cap.mark()
            t0 = time.monotonic()
            try:
                if rep == 0:
                    from torch.profiler import ProfilerActivity, profile
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        y = fn()
                        torch.cuda.synchronize()
                    r["kernels"] = sorted({e.name for e in prof.events() if e.device_type.name == "CUDA"})
                else:
                    y = fn()
                    torch.cuda.synchronize()
                r["gpu_seconds"] = round(time.monotonic() - t0, 4)
                b = common.tbits(y)
                del y
                r["sha256"] = sha(b)
                r["shape"] = list(b.shape)
                r["dtype"] = str(b.dtype)
            except Exception as e:  # noqa: BLE001
                b = None
                r["error"] = common.err_text(e)
                r["traceback"] = traceback.format_exc()[-3000:]
            r["lt"] = self.cap.matmuls_since(mk)
            r["lt_other"] = self.cap.other_since(mk)[:5]
            r["seconds"] = round(time.monotonic() - t0, 4)
            row["runs"].append(r)
            if rep == 0:
                first = b
        shas = [r.get("sha256") for r in row["runs"]]
        row["deterministic"] = None if None in shas else len(set(shas)) == 1
        if first is not None:
            kept = keep(first) if keep else {"out": first}
            for k, v in kept.items():
                self.arrays[f"{cid}.{k}"] = np.ascontiguousarray(v)
            row["stored"] = sorted(kept)
        self.rows.append(row)
        lt = row["runs"][0]["lt"]
        cfg = lt[0]["lt_algo_string"] if lt else "NO-LT-LOG"
        err = row["runs"][0].get("error") or row["runs"][-1].get("error")
        print(f"{cid}: det={row['deterministic']} {cfg} {('ERROR ' + err) if err else ''}", flush=True)
        return first

    def save(self, extra: dict | None = None) -> dict:
        np.savez_compressed(OUT / f"{self.step}.npz", **self.arrays)
        summ = dict(n_calls=len(self.rows),
                    n_nondeterministic=sum(r["deterministic"] is False for r in self.rows),
                    n_errors=sum(any("error" in x for x in r["runs"]) for r in self.rows),
                    n_without_lt_log=sum(not r["runs"][0]["lt"] for r in self.rows),
                    seconds=round(time.monotonic() - self.t0, 1))
        meta = dict(self.meta, extra=extra or {}, summary=summ, calls=self.rows)
        (OUT / f"{self.step}.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
        print(f"[{self.step}] {json.dumps(summ)}", flush=True)
        return summ


def te_api():
    """Transformer Engine 2.3.0 pieces (imported only by the --te paths)."""
    import transformer_engine.pytorch as te  # noqa: F401
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm
    from transformer_engine.pytorch.module.base import get_workspace
    from transformer_engine.pytorch.tensor._internal.float8_tensor_base import Float8TensorBase
    try:
        import transformer_engine
        ver = transformer_engine.__version__
    except Exception:  # noqa: BLE001
        ver = None
    return dict(tex=tex, general_gemm=general_gemm, get_workspace=get_workspace, F8=Float8TensorBase, version=ver)


def te_f8(api, u8, sinv_f32_tensor, fmt="e4m3"):
    dt = api["tex"].DType.kFloat8E4M3 if fmt == "e4m3" else api["tex"].DType.kFloat8E5M2
    return api["F8"](data=u8, fp8_scale_inv=sinv_f32_tensor.reshape(1), fp8_dtype=dt)


def te_gemm(api, w_u8, x_u8, sinv_w, sinv_x, out_dtype, split: bool, fmt_w="e4m3", fmt_x="e4m3"):
    """Y[M, N] = X[M, K] @ W[N, K]^T the way vortex TELinear fprop calls it (layout TN, TE's own workspace)."""
    W = te_f8(api, w_u8, sinv_w, fmt_w)
    X = te_f8(api, x_u8, sinv_x, fmt_x)
    return api["general_gemm"](W, X, api["get_workspace"](), out_dtype=out_dtype, layout="TN",
                               use_split_accumulator=split)[0]


def te_meta(api) -> dict:
    ws = api["get_workspace"]()
    return dict(te=api["version"], te_workspace_bytes=int(ws.numel() * ws.element_size()))
