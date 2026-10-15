"""GPU part (run under gpu.sh): run the raw bf16-MMA probes on the 4060 and save inputs + raw fp32 outputs.
  results/hw_epass_raw.pt : v9 E-pass data path (decode_word -> bf16 2^(6e-7) -> mma m16n8k16), raw d bits, for
                            (1) the test_prims8 tiles and (2) the structural worst-case set (diagonal-tile layout).
  results/hw_bf16_raw.pt  : the same mma.sync on general bf16 inputs (5 families) for model validation.
No analysis here; analyse.py does it on the CPU."""
import sys, time
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
import vectors as V  # noqa: E402

dev = "cuda:0"
X = build.get("epass_raw")


def struct_tiles():
    A, B, meta = V.structural()
    R = A.shape[0]
    T = (R + 7) // 8
    pad = T * 8 - R
    if pad:
        A = torch.cat([A, A[:pad]]); B = torch.cat([B, B[:pad]])
    g = torch.Generator().manual_seed(29)
    from test_prims import VALID
    extra = VALID[torch.randint(0, len(VALID), (T, 8, 16), generator=g)]
    At = torch.cat([A.reshape(T, 8, 16), extra], 1)
    Bt = B.reshape(T, 8, 16)
    return At, Bt, R


def bf16_bits(sign, ef, man):
    return ((sign << 15) | (ef << 7) | man).to(torch.int16)


def general_families(T, g):
    fams = {}
    def rnd(*shape, lo, hi):
        return torch.randint(lo, hi, shape, generator=g)
    # random: exponents within +-20 of 0, random signs/mantissas, c random (25% zero)
    sA = rnd(T, 16, 16, lo=0, hi=2); sB = rnd(T, 8, 16, lo=0, hi=2)
    A = bf16_bits(sA, rnd(T, 16, 16, lo=107, hi=148), rnd(T, 16, 16, lo=0, hi=128))
    B = bf16_bits(sB, rnd(T, 8, 16, lo=107, hi=148), rnd(T, 8, 16, lo=0, hi=128))
    C = (torch.randn(T, 16, 8, generator=g) * torch.exp2(rnd(T, 16, 8, lo=-20, hi=20).float()))
    C = torch.where(torch.rand(T, 16, 8, generator=g) < 0.25, torch.zeros_like(C), C)
    fams["random"] = (A, B, C)
    # truncation: one large product at k0, the rest 2^-18..2^-34 relative, mixed signs; c small or zero
    eA = rnd(T, 16, 16, lo=127 - 17, hi=127 - 8); eA[:, :, 0] = 127
    eB = rnd(T, 8, 16, lo=127 - 17, hi=127 - 8); eB[:, :, 0] = 127
    perm = torch.argsort(torch.rand(T, 16, generator=g), 1)
    eA = eA.gather(2, perm[:, None, :].expand(T, 16, 16)); eB = eB.gather(2, perm[:, None, :].expand(T, 8, 16))
    A = bf16_bits(rnd(T, 16, 16, lo=0, hi=2), eA, rnd(T, 16, 16, lo=0, hi=128))
    B = bf16_bits(rnd(T, 8, 16, lo=0, hi=2), eB, rnd(T, 8, 16, lo=0, hi=128))
    C = torch.where(torch.rand(T, 16, 8, generator=g) < 0.5, torch.zeros(T, 16, 8),
                    torch.randn(T, 16, 8, generator=g) * 2.0 ** -20)
    fams["truncation"] = (A, B, C)
    # carry: all products of similar size (exponents 127..128), positive, plus a small c
    A = bf16_bits(torch.zeros(T, 16, 16, dtype=torch.long), rnd(T, 16, 16, lo=127, hi=129), rnd(T, 16, 16, lo=0, hi=128))
    B = bf16_bits(torch.zeros(T, 8, 16, dtype=torch.long), rnd(T, 8, 16, lo=127, hi=129), rnd(T, 8, 16, lo=0, hi=128))
    C = torch.rand(T, 16, 8, generator=g) * 2.0 ** -10
    fams["carry"] = (A, B, C)
    # chunk: big term only in the SECOND half of k (8..15), small terms in the first half (distinguishes NFMA 8 vs 16)
    eA = rnd(T, 16, 16, lo=127 - 14, hi=127 - 9); eA[:, :, 8 + rnd(1, lo=0, hi=8).item()] = 127
    eB = rnd(T, 8, 16, lo=127 - 14, hi=127 - 9); eB[:, :, 8:] = 127
    A = bf16_bits(rnd(T, 16, 16, lo=0, hi=2), eA, rnd(T, 16, 16, lo=0, hi=128))
    B = bf16_bits(torch.zeros(T, 8, 16, dtype=torch.long), eB, rnd(T, 8, 16, lo=0, hi=128))
    C = torch.where(torch.rand(T, 16, 8, generator=g) < 0.5, torch.zeros(T, 16, 8), torch.randn(T, 16, 8, generator=g))
    fams["chunk"] = (A, B, C)
    # powers of two only (E-pass-like but general exponents and signs): exponents spaced arbitrarily
    A = bf16_bits(rnd(T, 16, 16, lo=0, hi=2), rnd(T, 16, 16, lo=100, hi=150), torch.zeros(T, 16, 16, dtype=torch.long))
    B = bf16_bits(torch.zeros(T, 8, 16, dtype=torch.long), rnd(T, 8, 16, lo=100, hi=150), torch.zeros(T, 8, 16, dtype=torch.long))
    C = torch.where(torch.rand(T, 16, 8, generator=g) < 0.5, torch.zeros(T, 16, 8), torch.exp2(rnd(T, 16, 8, lo=-40, hi=40).float()))
    fams["pow2"] = (A, B, C)
    return fams


def main():
    t0 = time.time()
    out = {}
    At, Bt = V.prims8_tiles()
    r = X.epass_raw(At.reshape(-1, 16).to(torch.uint8).to(dev).contiguous(), Bt.reshape(-1, 16).to(torch.uint8).to(dev).contiguous()).cpu()
    out["prims8"] = dict(At=At.to(torch.uint8), Bt=Bt.to(torch.uint8), d=r)
    St, SBt, R = struct_tiles()
    r = X.epass_raw(St.reshape(-1, 16).to(torch.uint8).to(dev).contiguous(), SBt.reshape(-1, 16).to(torch.uint8).to(dev).contiguous()).cpu()
    out["structural"] = dict(At=St.to(torch.uint8), Bt=SBt.to(torch.uint8), d=r, rows=R)
    torch.save(out, _bringup_out() / "hw_epass_raw.pt")
    print("epass raw done", {k: tuple(v["d"].shape) for k, v in out.items()}, f"{time.time() - t0:.1f}s", flush=True)
    g = torch.Generator().manual_seed(4060)
    fams = general_families(4096, g)
    res = {}
    for name, (A, B, C) in fams.items():
        T = A.shape[0]
        d = X.mma_bf16_raw(A.reshape(-1, 16).to(dev).contiguous(), B.reshape(-1, 16).to(dev).contiguous(),
                           C.float().to(dev).contiguous()).cpu()
        res[name] = dict(A=A, B=B, C=C.float(), d=d)
    torch.save(res, _bringup_out() / "hw_bf16_raw.pt")
    print("bf16 raw done", {k: tuple(v["d"].shape) for k, v in res.items()}, f"{time.time() - t0:.1f}s",
          torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)


if __name__ == "__main__":
    main()
