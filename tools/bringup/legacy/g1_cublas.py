"""Full native bf16 GEMM comparison, with nonunit scales and logged split-K.

Exit 1 records a numerical mismatch; exit 2 records a probe failure. Run with
CUBLASLT_LOG_LEVEL=2 under the GPU lock. Outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

import mmaemu
from mmaemu.model.presets import list_presets
from mmaemu.platform.profiles import get_profile
from mmaemu.testing import bits
from tools.record.common import LtCapture, cublas_params, hardware, splitk_of

SHAPES = {"1b": (1920, 5760), "7b": (4096, 12288), "40b": (8192, 24576)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="ada", choices=list_presets())
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="1b,7b,40b")
    ap.add_argument("--M", default="16,48,128")
    args = ap.parse_args(argv)
    path = Path(args.out)
    if path.exists():
        ap.error(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda", 0)
    capture = LtCapture()
    gen = torch.Generator(device=dev).manual_seed(157)
    # Nonunit, non-power-of-two scales exercise the bf16 epilogue.
    sa = torch.tensor(0.009513410739, device=dev)
    sb = torch.tensor(0.004164181649, device=dev)
    profile = replace(get_profile("ada-rtx4060-torch"), name=f"bringup-{args.preset}", preset=args.preset)
    report = {"env": hardware(), "preset": args.preset, "out_dtype": "bf16", "rows": []}
    for model in args.models.split(","):
        K, N = SHAPES[model]
        for M in map(int, args.M.split(",")):
            row = {"model": model, "shape": [M, N, K], "scale_a": sa.item(), "scale_b": sb.item()}
            try:
                a = (torch.randn((M, K), generator=gen, device=dev) * 40).clamp(-448, 448).to(torch.float8_e4m3fn)
                b = (torch.randn((N, K), generator=gen, device=dev) * 40).clamp(-448, 448).to(torch.float8_e4m3fn)
                mark = capture.mark()
                hw = torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb,
                                      out_dtype=torch.bfloat16, use_fast_accum=True)
                torch.cuda.synchronize()
                calls = capture.matmuls_since(mark)
                if len(calls) != 1:
                    raise RuntimeError(f"expected one logged cuBLASLt GEMM, got {len(calls)}")
                cfg = calls[0]
                sk = splitk_of(cfg)
                split = 1 if sk["S"] == 1 else (sk["S"], sk["scheme"])
                em = mmaemu.scaled_mm(a, b.t(), sa, sb, out_dtype=torch.bfloat16,
                                     use_fast_accum=True, profile=profile, splitk=split)
                count = int((bits(hw).cpu() != bits(em).cpu()).sum())
                row.update(outputs=hw.numel(), mismatches=count, bitwise_equal=count == 0,
                           cublaslt=cublas_params(cfg), splitk=sk)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb,
                                     out_dtype=torch.bfloat16, use_fast_accum=True)
                    torch.cuda.synchronize()
                row["kernel_names"] = sorted({e.name for e in prof.events() if e.device_type.name == "CUDA"})
            except Exception as exc:  # noqa: BLE001 - retain each probe failure
                row["error"] = f"{type(exc).__name__}: {exc}"
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
    report["verdict"] = ("ERROR" if any("error" in r for r in report["rows"]) else
                         "MISMATCH" if any(r["mismatches"] for r in report["rows"]) else "PASS")
    path.write_text(json.dumps(report, indent=2) + "\n")
    return {"PASS": 0, "MISMATCH": 1, "ERROR": 2}[report["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
