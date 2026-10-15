"""The paper's own interleave probe (arXiv:2512.07004v4, Sec. 4.1.6, reused for B200 in Sec. 4.1.7):
p1 = 1, p2 = 2^-24, then 2^-24 placed at p3..p32 one at a time, c = 0. Paper: d = 1 + 2^-23 iff the probed product
is in the same accumulation group as p1, p2; otherwise d = 1; groups alternate in pairs (Fig. 5c).
Here products are formed from FP8 inputs: E5M2 (1*1, 2^-12*2^-12) and E4M3 scaled (8*8 = 2^6, 2^-9*2^-9 = 2^-18).
CPU only."""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent))
from models import *
import sys as _sys
from pathlib import Path as _P
_d = _P(__file__).resolve().parent
while not (_d / "compat.py").exists():   # tools/bringup (compat.py)
    _d = _d.parent
_sys.path.insert(0, str(_d))
from compat import R  # noqa: E402  (vendored: was `from tcemu import reference as R`)

for fmt, big, small in (("e5m2", 1.0, 2.0 ** -12), ("e4m3", 8.0, 2.0 ** -9)):
    N = 30
    a = np.zeros((N, 32)); b = np.zeros((N, 32))
    a[:, 0] = b[:, 0] = big
    a[:, 1] = b[:, 1] = small
    for i, j in enumerate(range(2, 32)):   # 0-based position of the probed product (paper's p3..p32)
        a[i, j] = b[i, j] = small
    c = np.zeros(N)
    base = big * big
    ulp = base * 2.0 ** -23
    res = {
        "mmaemu b200_mma_sync (alias blackwell)": R.dot(torch.tensor(a), torch.tensor(b), torch.tensor(c), fmt=fmt, **R.PRESETS["b200_mma_sync"]).numpy(),
        "HMMAx2 pairs (SASS)": m_hmma_chain(a, b, c, fmt, "pairs"),
        "HMMAx2 contig": m_hmma_chain(a, b, c, fmt, "contig"),
    }
    print(f"\n# {fmt}: p1 = {base:g}, p2 = p_j = {small*small:g} (ratio 2^-24); entries: d = p1 + n*ulp(p1), n shown")
    print("paper p_j: " + " ".join(f"{j+1:2d}" for j in range(2, 32)))
    for k, v in res.items():
        n = np.round((v - base) / ulp).astype(int)
        print(f"{k:20s}: " + " ".join(f"{x:2d}" for x in n))
    paper = np.array([1 if (j % 4) < 2 else 0 for j in range(2, 32)])  # paper's description of Fig. 5c
    print(f"{'paper (pairs, Fig5c)':20s}: " + " ".join(f"{x:2d}" for x in paper))
    for k, v in res.items():
        n = np.round((v - base) / ulp).astype(int)
        print(f"  {k:20s} agrees with paper's interleave pattern at {int((n == paper).sum())}/30 positions")
