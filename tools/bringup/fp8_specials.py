"""Measure native FP8 saturation/scale-result and verify reference CPU/GPU twins."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import mmaemu
from mmaemu.testing import assert_bitwise_equal, bits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    path = Path(args.out)
    if path.exists():
        ap.error(f"refusing to overwrite {path}")
    rows = []
    failures = 0
    for od in (torch.float8_e4m3fn, torch.float8_e5m2):
        for alpha in (1.0, -1.0, 2.0 ** -30):
            for sr in (0.5, 2.0):
                a = torch.zeros(16, 32)
                b = torch.zeros(16, 32)
                a[0, 0] = 448
                a[1, 0] = -448
                a = a.to(torch.float8_e4m3fn)
                b[:, 0] = 448
                b = b.to(torch.float8_e5m2).t()
                sa, sb, scale_result = map(torch.tensor, (alpha, 1.0, sr))
                kw = dict(out_dtype=od, use_fast_accum=True, splitk=1, backend="reference",
                          profile="ada")
                cpu = mmaemu.scaled_mm(a, b, sa, sb, scale_result=scale_result, **kw)
                gpu = mmaemu.scaled_mm(a.cuda(), b.cuda(), sa.cuda(), sb.cuda(), scale_result=scale_result.cuda(), **kw)
                row = {"out_dtype": str(od), "alpha": alpha, "scale_result": sr}
                try:
                    assert_bitwise_equal(gpu, cpu)
                    row["cpu_gpu_bitwise_equal"] = True
                except AssertionError as exc:
                    row["cpu_gpu_bitwise_equal"] = False
                    row["emulator_error"] = str(exc)
                    failures += 1
                row["model_bits"] = bits(cpu[:, 0]).tolist()
                try:
                    hw = torch._scaled_mm(a.cuda(), b.cuda(), scale_a=sa.cuda(), scale_b=sb.cuda(),
                                          scale_result=scale_result.cuda(), out_dtype=od, use_fast_accum=True)
                    row["native_bits"] = bits(hw[:, 0]).cpu().tolist()
                    row["native_mismatches"] = int((bits(hw).cpu() != bits(cpu)).sum())
                except RuntimeError as exc:
                    row["native_error"] = str(exc)
                rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"gpu": torch.cuda.get_device_name(), "rows": rows,
                                "emulator_failures": failures}, indent=2) + "\n")
    print(f"{len(rows)} native observations; {failures} CPU/GPU twin failures")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
