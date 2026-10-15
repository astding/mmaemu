"""Compatibility entry point for the self-contained c6 instruction-vector kit."""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main():
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch")
    ap.add_argument("--sets", nargs="+", default=["MM_sm120", "MM_ada", "TBX_B200", "TBX_H100c0"])
    args = ap.parse_args()
    cc = torch.cuda.get_device_capability(0) if args.arch is None else None
    arch = args.arch or f"{cc[0]}.{cc[1]}"
    kit = Path(__file__).resolve().parents[1] / "kits" / "c6" / "run_kit.py"
    sys.argv = [str(kit), "--sets", *args.sets, "--arch", arch, "--out", args.out]
    runpy.run_path(str(kit), run_name="__main__")


if __name__ == "__main__":
    main()
