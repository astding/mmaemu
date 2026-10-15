"""Kernel-level validation of design B (p3's tc_emuB3) with a software MMA (csrc/b3soft.cu).
usage:
  python b3soft.py build [arch]          compile (no GPU) for sm_89 (default) or e.g. 120
  python b3soft.py probe                 (GPU lock) instruction level: hw vs soft models on random + design-B tiles
  python b3soft.py gemm <modes> <cfgs> [case-filter]   (GPU lock) GEMM level on p3's verify cases vs tcemu.reference
modes: 0 = hardware mma.sync, 1 = soft Ada bf16 model, 2 = soft B200 bf16 model, 3 = B200 but late c, 4 = F=26."""
import json, os, sys, time
import torch
from tools.bringup.checks import same_bits

from pathlib import Path as _P
W = str(_P(__file__).resolve().parents[1])   # the vendored kit dir (csrc/)

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


def ext(arch=None):
    from torch.utils.cpp_extension import load
    if arch is None:
        cc = torch.cuda.get_device_capability(0)
        arch = f"{cc[0]}{cc[1]}"
    bd = str(_bringup_cache() / f"b3soft_build_sm{arch}")
    os.makedirs(bd, exist_ok=True)
    return load(name=f"vp3v0_b3soft_sm{arch}", sources=[f"{W}/csrc/b3soft.cu"], build_directory=bd,
                extra_cuda_cflags=["-O3", f"-arch=sm_{arch}", "-lineinfo", "-fmad=false"], verbose=False)


def exp_(x, emin=None):
    e = (torch.frexp(x)[1] - 1).double()
    if emin is not None:
        e = e.clamp_min(emin)
    return torch.where(x != 0, e, torch.full_like(x, -1e5))


def py_model(A, B, C, mode):
    """CPU float64 reference of the soft models, written independently in the style of p3's sim_designs.hmma
    (A [V,16,16], B [V,16,8], C [V,16,8]) -> D [V,16,8]."""
    a = A[:, :, None, :].expand(-1, 16, 8, 16)          # [V, m, n, k]
    b = B.transpose(1, 2)[:, None, :, :].expand(-1, 16, 8, 16)

    def blk(a, b, c, F, late):
        p = a * b
        ep = torch.where(p != 0, exp_(a, -126) + exp_(b, -126), torch.full_like(p, -1e5))
        E = ep.amax(-1)
        if not late:
            E = torch.maximum(E, exp_(c))
        E = E.clamp_min(-133)
        s = torch.trunc(p * torch.pow(2.0, F - E)[..., None]).sum(-1)
        if not late:
            s = s + torch.trunc(c * torch.pow(2.0, F - E))
        d = s * torch.pow(2.0, E - F)
        ed = exp_(d).clamp_min(-126)
        d = torch.where(d != 0, torch.trunc(d * torch.pow(2.0, 23 - ed)) * torch.pow(2.0, ed - 23), torch.zeros_like(d))
        if late:
            d = (d.float() + c.float()).double()
        return d
    if mode == 1:
        return blk(a[..., 8:], b[..., 8:], blk(a[..., :8], b[..., :8], C, 24, False), 24, False)
    if mode == 2:
        return blk(a, b, C, 25, False)
    if mode == 3:
        return blk(a, b, C, 25, True)
    return blk(a, b, C, 26, False)


def tiles(kind, V, g):
    """bf16-valued tiles A [V,16,16], B [V,16,8] (k x n), C [V,16,8] fp32-valued, all float64."""
    def e4(shape):
        return (torch.randn(shape, generator=g) * 40).clamp(-448, 448).to(torch.float8_e4m3fn).double()
    if kind == "gauss":
        A = torch.randn(V, 16, 16, generator=g).bfloat16().double(); B = torch.randn(V, 16, 8, generator=g).bfloat16().double()
        C = torch.randn(V, 16, 8, generator=g).float().double()
    elif kind == "wide":
        A = (torch.randn(V, 16, 16, generator=g) * torch.exp2(torch.randint(-30, 30, (V, 16, 16), generator=g).float())).bfloat16().double()
        B = (torch.randn(V, 16, 8, generator=g) * torch.exp2(torch.randint(-30, 30, (V, 16, 8), generator=g).float())).bfloat16().double()
        C = (torch.randn(V, 16, 8, generator=g) * torch.exp2(torch.randint(-40, 40, (V, 16, 8), generator=g).float())).float().double()
    elif kind == "e4m3":
        A, B = e4((V, 16, 16)), e4((V, 16, 8))
        C = (torch.randn(V, 16, 8, generator=g) * 1e4).float().double()
    elif kind == "designB":
        # exactly the kernel's construction: k 0-7 real E4M3 values, A[r][8+n] = -2^(E_rn+12), B[8+n'][n] = (n'==n),
        # C[r][n] = 2^(E_rn+12), E_rn = max(maxep_rn, -12) + offset (offset 0..20 covers accumulators >> products)
        A = torch.zeros(V, 16, 16, dtype=torch.float64); B = torch.zeros(V, 16, 8, dtype=torch.float64)
        A[:, :, :8] = e4((V, 16, 8)); B[:, :8, :] = e4((V, 8, 8))
        for n in range(8):
            B[:, 8 + n, n] = 1.0
        ea = torch.where(A[:, :, :8] != 0, exp_(A[:, :, :8], -6), torch.full_like(A[:, :, :8], -1e5))  # [V,16,8k]
        eb = torch.where(B[:, :8, :] != 0, exp_(B[:, :8, :], -6), torch.full_like(B[:, :8, :], -1e5))  # [V,8k,8n]
        ep = (ea[:, :, :, None] + eb[:, None, :, :]).amax(2)                                           # [V,16,8]
        E = torch.maximum(ep, torch.full_like(ep, -12)) + torch.randint(0, 21, (V, 16, 8), generator=g).double()
        # the dummy slot of (r, n) is A[r][8+n] -- but A[r][8+n] is shared by all columns n' of row r with B[8+n][n'] = 0,
        # so the dummy of (r, n) must be A[r][8+n]: one A entry per (r, n) pair -> fine (k-slot 8+n is column n's own)
        A[:, :, 8:] = -torch.pow(2.0, E + 12)
        C = torch.pow(2.0, E + 12)
    elif kind == "xenc":   # exponent-encoding vectors 2^(5 ea), 2^(5 eb), c = 0
        ea = torch.randint(-6, 9, (V, 16, 16), generator=g).double(); eb = torch.randint(-6, 9, (V, 16, 8), generator=g).double()
        A = torch.where(torch.rand(V, 16, 16, generator=g) < 0.15, 0.0, torch.pow(2.0, 5 * ea))
        B = torch.where(torch.rand(V, 16, 8, generator=g) < 0.15, 0.0, torch.pow(2.0, 5 * eb))
        C = torch.zeros(V, 16, 8, dtype=torch.float64)
    else:
        raise ValueError(kind)
    assert same_bits(A.bfloat16().double(), A) and same_bits(B.bfloat16().double(), B) and same_bits(C.float().double(), C)
    return A, B, C


def probe():
    e = ext()
    g = torch.Generator().manual_seed(2026)
    res = dict(device=torch.cuda.get_device_name(0))
    for kind in ("gauss", "wide", "e4m3", "designB", "xenc"):
        V = 2048
        A, B, C = tiles(kind, V, g)
        Ab = A.bfloat16().view(torch.int16).cuda(); Bb = B.bfloat16().view(torch.int16).cuda(); Cf = C.float().cuda()
        D = {m: e.probe(Ab, Bb, Cf, m).cpu() for m in range(5)}
        r = {"n_outputs": V * 128}
        r["hw_eq_soft_ada_bitwise"] = int((D[0].view(torch.int32) == D[1].view(torch.int32)).sum())
        r["hw_eq_soft_b200"] = int((D[0] == D[2]).sum())
        for m in (1, 2, 3, 4):
            pm = py_model(A, B, C, m)
            r[f"soft{m}_eq_python_model"] = int((D[m].double() == pm).sum())
        if kind == "designB":
            # the property the kernel relies on: D == exact truncated sum of the 8 real products on the 2^(E-13) grid
            E = torch.log2(C) - 12
            u = torch.pow(2.0, E - 13)
            p = A[:, :, None, :8] * B.transpose(1, 2)[:, None, :, :8]
            S = (torch.trunc(p / u[..., None])).sum(-1) * u
            for m, nm in ((0, "hw"), (1, "soft_ada"), (2, "soft_b200"), (3, "soft_latec"), (4, "soft_F26")):
                r[f"property_holds_{nm}"] = int((D[m].double() == S).sum())
        res[kind] = r
        print(kind, json.dumps(r), flush=True)
    json.dump(res, open(_bringup_out() / "b3soft_probe.json", "w"), indent=1)


def gemm(modes, cfgs, filt=None):
    raise SystemExit("b3soft gemm mode is not vendored: it reads p3's pickled verify cases (13 MB) and tcemu.reference; use the probe mode (informational in G8)")
    e = ext()
    cases = torch.load(f"{P3}/data/verify_cases.pt") + torch.load(f"{P3}/data/verify_cases_extra.pt")
    if filt:
        cases = [c for c in cases if any(f in c["name"] for f in filt.split(","))]
    out = {}
    for c in cases:
        a, b = c["A"].cuda(), c["B"].cuda()
        rec = {}
        hw = None
        for mode in modes:
            for cfg in cfgs:
                t = time.time()
                o = e.b3_gemm(a, b, mode, cfg); torch.cuda.synchronize()
                dt = time.time() - t
                o = o.cpu()
                nbad = int((o.view(torch.int32) != c["ref"].view(torch.int32)).sum())
                rec[f"m{mode}_c{cfg}_nbad_vs_ref"] = nbad
                if mode == 0 and cfg == cfgs[0]:
                    hw = o
                if mode == 1 and hw is not None:
                    rec[f"m1_c{cfg}_nbad_vs_hw"] = int((o.view(torch.int32) != hw.view(torch.int32)).sum())
                rec[f"m{mode}_c{cfg}_s"] = round(dt, 2)
        rec["n"] = c["ref"].numel()
        out[c["name"]] = rec
        print(c["name"], json.dumps(rec), flush=True)
    tag = "m" + "".join(map(str, modes)) + "_c" + "".join(map(str, cfgs)) + (("_" + filt.replace(",", "+")) if filt else "")
    json.dump(out, open(f"{W}/logs/b3soft_gemm_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        ext(sys.argv[2] if len(sys.argv) > 2 else None)
        print("built")
    elif sys.argv[1] == "probe":
        probe()
    else:
        gemm([int(x) for x in sys.argv[2].split(",")], [int(x) for x in sys.argv[3].split(",")],
             sys.argv[4] if len(sys.argv) > 4 else None)
