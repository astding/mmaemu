"""M2: the 21 Evo 2 1B projection GEMMs (real operands of one 2 kb window, fixed inputs), full output bits.

torch path (torch._scaled_mm with this process's cuBLASLt workspace: the torch default unless CUBLASLT_WORKSPACE_SIZE
is set):
  acc_fast / acc_slow  unit scales, fp32 out, use_fast_accum True / False
  y_fast / y_slow      scale_a = fp8_scale_inv[0], scale_b = fp8_scale_inv[1], bf16 out (y_fast = the Evo 2 native
                       call of evo2-anywhere's FP8Projection)
TE path (--te; Transformer Engine 2.3.0 general_gemm(W, X, TE workspace, layout TN), as vortex TELinear fprop):
  te_y / te_y_split    bf16 out with the checkpoint scale_inv, use_split_accumulator False / True
  te_acc / te_acc_split  fp32 out, unit scale_inv
Stored: full bits for calls 0, 10, 20; rows 0..255 for the others; the full sha256 of every output (both runs).
usage: m2_evo2.py --data DIR [--te] [--calls 0,1,...]
  DIR holds fp8-inputs/callNN.pt (uint8 xq [M, K], wq [N, K]) and truth/evo2_1b_w2k.json (input sha256 + scales)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import pc

FULL = (0, 10, 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--te", action="store_true")
    ap.add_argument("--calls", default=",".join(str(i) for i in range(21)))
    a = ap.parse_args()
    import torch
    rec = pc.Rec("m2_te" if a.te else "m2_torch")
    truth = json.loads((a.data / "truth" / "evo2_1b_w2k.json").read_text())
    dev = torch.device("cuda:0")
    api = pc.te_api() if a.te else None
    extra = dict(path="te" if a.te else "torch", te=pc.te_meta(api) if api else None, inputs=[])
    one = torch.ones((), device=dev)
    for i in (int(x) for x in a.calls.split(",")):
        case = truth["cases"][i]
        d = torch.load(a.data / "fp8-inputs" / f"call{i:02d}.pt", map_location="cpu", weights_only=True)
        xq, wq = d["xq"].contiguous(), d["wq"].contiguous()
        sx = hashlib.sha256(xq.numpy().tobytes()).hexdigest()
        sw = hashlib.sha256(wq.numpy().tobytes()).hexdigest()
        ok = (sx == case["inputs"]["a"]["sha256"] and sw == case["inputs"]["b"]["sha256"]
              and xq.dtype == torch.uint8 and wq.dtype == torch.uint8)
        extra["inputs"].append(dict(call=i, xq_sha256=sx, wq_sha256=sw, equal_to_truth_inputs=ok,
                                    shape_x=list(xq.shape), shape_w=list(wq.shape)))
        if not ok:
            print(f"call{i:02d}: INPUT SHA256 MISMATCH, skipped", flush=True)
            continue
        p = case["params"]
        inv = torch.tensor([p["fp8_scale_inv_in"], p["fp8_scale_inv_w"]], dtype=torch.int32).view(torch.float32)
        inv = inv.to(dev)
        xg, wg = xq.to(dev), wq.to(dev)
        x8, w8 = xg.view(torch.float8_e4m3fn), wg.view(torch.float8_e4m3fn)
        keep = None if i in FULL else (lambda b: {"rows0_256": b[:256]})
        meta = dict(call=i, M=int(xq.shape[0]), N=int(wq.shape[0]), K=int(xq.shape[1]),
                    scale_inv_bits=[p["fp8_scale_inv_in"], p["fp8_scale_inv_w"]])
        if not a.te:
            for fast in (True, False):
                tag = "fast" if fast else "slow"
                rec.call(f"call{i:02d}.acc_{tag}", lambda fast=fast: torch._scaled_mm(
                    x8, w8.t(), scale_a=one, scale_b=one, out_dtype=torch.float32, use_fast_accum=fast),
                    keep, dict(meta, out="float32", fast=fast, scales="unit"))
                rec.call(f"call{i:02d}.y_{tag}", lambda fast=fast: torch._scaled_mm(
                    x8, w8.t(), scale_a=inv[0], scale_b=inv[1], out_dtype=torch.bfloat16, use_fast_accum=fast),
                    keep, dict(meta, out="bfloat16", fast=fast, scales="fp8_scale_inv"))
        else:
            for split in (False, True):
                tag = "_split" if split else ""
                rec.call(f"call{i:02d}.te_y{tag}", lambda split=split: pc.te_gemm(
                    api, wg, xg, inv[1], inv[0], torch.bfloat16, split),
                    keep, dict(meta, out="bfloat16", use_split_accumulator=split, scales="fp8_scale_inv"))
                rec.call(f"call{i:02d}.te_acc{tag}", lambda split=split: pc.te_gemm(
                    api, wg, xg, one, one, torch.float32, split),
                    keep, dict(meta, out="float32", use_split_accumulator=split, scales="unit"))
        del xg, wg, x8, w8
    extra["n_inputs_ok"] = sum(r["equal_to_truth_inputs"] for r in extra["inputs"])
    rec.save(extra)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
