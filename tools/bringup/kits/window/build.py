"""Compile the p2-int-mixed extensions (no GPU needed). usage: python build.py [name ...]"""
import os, sys
from pathlib import Path
from torch.utils.cpp_extension import load

import os as _os
from pathlib import Path as _Path
def _bringup_out():
    d = _Path(_os.environ.get("BRINGUP_OUT") or ".")
    d.mkdir(parents=True, exist_ok=True)
    return d
def _bringup_cache():
    d = _Path(_os.environ.get("BRINGUP_CACHE") or (_Path.home() / ".cache" / "mmaemu" / "bringup"))
    d.mkdir(parents=True, exist_ok=True)
    return d

W = Path(__file__).resolve().parent
EXTS = {"v2": "v2.cu", "v3": "v3.cu", "v4": "v4.cu", "v4a": "v4a.cu", "v4t": "v4t.cu", "v5": "v5.cu", "v6": "v6.cu", "v7": "v7.cu", "v8": "v8.cu", "v9": "v9.cu", "v9p": "v9.cu", "prims8": "prims8.cu", "epass_raw": "epass_raw.cu", "v6o1": "v6.cu", "v6o2": "v6.cu", "v7o1": "v7.cu", "prims": "prims.cu", "clockprobe": "clockprobe.cu", "v1": "v1_copy.cu", "v1fmarz": "v1_copy.cu"}
EXTRA = {"v1fmarz": ["-DTCEMU_FMA_RZ=1"], "v6o1": ["-Xptxas=-O1"], "v6o2": ["-Xptxas=-O2"], "v7o1": ["-Xptxas=-O1"], "v9p": ["-DP2_PORTABLE_DECODE"]}

def get(name, verbose=False, extra=()):
    bd = _bringup_cache() / ("window_build_" + name)
    bd.mkdir(parents=True, exist_ok=True)
    return load(name=f"vp2i3_{name}", sources=[str(W / EXTS[name])], build_directory=str(bd),
                extra_cuda_cflags=["-O3", "-lineinfo", "-fmad=false", *EXTRA.get(name, []), *extra], verbose=verbose)

if __name__ == "__main__":
    for n in (sys.argv[1:] or EXTS):
        get(n, verbose=False)
        print("built", n)
