"""Exhaustive / adversarial test of v8's tensor-core E-pass (prims8.cu) on the same 16-term vectors as test_prims.py
(every byte pair alone, every byte pair x16, structured max-count rows) plus all 16x8 cross pairs of each tile.
Run under gpu.sh -> results/prims8.json"""
import json, sys, time
from pathlib import Path
import torch

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
sys.path.insert(0, str(W))
import build  # noqa: E402
from test_prims import VALID, truth_fields  # noqa: E402

P = None
dev = "cuda:0"


def rows_like_test_prims():
    import test_prims as tp
    g = torch.Generator().manual_seed(7)
    captured = {}
    orig = tp.expected_pbits
    def cap(A, B):
        captured["A"], captured["B"] = A, B
        return orig(A, B)
    tp.expected_pbits = cap
    class Stub:
        def epass(self, A, B, mode): return torch.zeros(A.shape[0], 2, dtype=torch.int32)
        def epass64(self, A, B): return torch.zeros(A.shape[0], 2, dtype=torch.int32)
    tp.P = Stub()
    tp.check_epass({})
    return captured["A"].long(), captured["B"].long()


def main():
    global P
    P = build.get("prims8")
    A, B = rows_like_test_prims()  # [R, 16] each, row r = one 16-term block (a bytes, b bytes)
    R = A.shape[0]
    T = R // 8
    A, B = A[:T * 8], B[:T * 8]
    # tile t: A rows = test rows 8t..8t+7 followed by 8 random rows; B cols = test rows' b bytes
    g = torch.Generator().manual_seed(23)
    extra = VALID[torch.randint(0, len(VALID), (T, 8, 16), generator=g)]
    At = torch.cat([A.reshape(T, 8, 16), extra], 1)            # [T, 16, 16]
    Bt = B.reshape(T, 8, 16)                                    # [T, 8, 16]
    out = P.epass_mma(At.reshape(-1, 16).to(torch.uint8).to(dev).contiguous(), Bt.reshape(-1, 16).to(torch.uint8).to(dev).contiguous()).cpu()
    # truth for all (row, col) pairs
    _, ea, _, nza = truth_fields(At.reshape(-1))
    _, eb, _, nzb = truth_fields(Bt.reshape(-1))
    ea, nza = ea.reshape(T, 16, 1, 16), nza.reshape(T, 16, 1, 16)
    eb, nzb = eb.reshape(T, 1, 8, 16), nzb.reshape(T, 1, 8, 16)
    s = torch.where(nza & nzb, ea + eb, torch.full_like(ea + eb, -1000))
    E = s.amax(-1).clamp_min(-12)
    want = ((E + 127) << 23).int()
    diag = torch.arange(8)
    res = dict(tiles=T, outputs=int(want.numel()), mismatches=int((out != want).sum()),
               designed_rows=int(T * 8), designed_row_mismatches=int((out[:, diag, diag] != want[:, diag, diag]).sum()))
    print(res, flush=True)
    json.dump(res, open(_bringup_out() / "prims8.json", "w"), indent=1)


if __name__ == "__main__":
    main()
