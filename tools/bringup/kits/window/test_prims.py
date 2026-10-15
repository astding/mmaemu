"""Exhaustive / adversarial tests of the v2 device primitives (v2_common.cuh) against exact float64 ground truth.
Run under gpu.sh:  python test_prims.py   -> results/prims.json"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from tools.bringup.checks import same_bits

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

P = None
dev = "cuda:0"
f64 = torch.float64
VALID = torch.tensor([c for c in range(256) if c not in (0x7F, 0xFF)], dtype=torch.int64)


def e4m3_value(codes):
    return codes.to(torch.uint8).view(torch.float8_e4m3fn).double()


def truth_fields(codes):
    x = e4m3_value(codes)
    nz = x != 0
    ea = torch.where(nz, (torch.frexp(x)[1] - 1).clamp_min(-6), torch.zeros_like(codes))
    n = (x.abs() * 512).round().long()  # |x| = n * 2^-9, n integer
    lowbit = n & (-n)
    la = torch.where(nz, torch.log2(lowbit.clamp_min(1).double()).round().long() - 9, torch.zeros_like(codes))
    return x, ea.long(), la, nz


def check_decode(res):
    t = P.decode_table().cpu()
    codes = VALID
    x, ea, la, nz = truth_fields(codes)
    tv = t[codes]
    v = tv[:, 0].view(torch.float32).double()
    ok_v = same_bits(v, x)
    o32 = torch.where(nz, ((5 * ea + 127) << 23), 0)
    o16 = torch.where(nz, ((5 * ea - 30 + 127) << 7), 0)
    lae = torch.where(nz, la + 9, 60)
    sg = ((codes >> 7) & 1) * 0x80
    res["decode"] = dict(codes=len(codes), value_ok=ok_v, o32_ok=bool((tv[:, 1] == o32).all()),
                         o16_ok=bool((tv[:, 2] == o16).all()), la_ok=bool((tv[:, 3] == lae).all()),
                         sign_ok=bool((tv[:, 4] == sg).all()), la_range=[int(lae[nz].min()), int(lae[nz].max())])
    print("decode", res["decode"], flush=True)


def expected_pbits(a, b):
    xa, ea, _, nza = truth_fields(a.long())
    xb, eb, _, nzb = truth_fields(b.long())
    s = torch.where(nza & nzb, ea + eb, torch.full_like(ea, -1000))
    E = s.amax(-1).clamp_min(-12)
    return ((E + 127) << 23).int(), E


def check_epass(res):
    g = torch.Generator().manual_seed(7)
    rows_a, rows_b = [], []
    # (i) every byte pair as the only nonzero term (15 zero terms), (ii) every byte pair repeated 16 times
    pa, pb = torch.meshgrid(VALID, VALID, indexing="ij")
    pa, pb = pa.reshape(-1), pb.reshape(-1)
    z = torch.zeros(len(pa), 15, dtype=torch.int64)
    rows_a.append(torch.cat([pa[:, None], z], 1)); rows_b.append(torch.cat([pb[:, None], z], 1))
    rows_a.append(pa[:, None].expand(-1, 16)); rows_b.append(pb[:, None].expand(-1, 16))
    # (iii) for each target E and count c of terms attaining it: c terms at E, rest random below (or zero)
    xa, ea, _, nza = truth_fields(pa)
    xb, eb, _, nzb = truth_fields(pb)
    nzp = nza & nzb
    ep = (ea + eb)[nzp]; qa = pa[nzp]; qb = pb[nzp]
    order = torch.argsort(ep); ep, qa, qb = ep[order], qa[order], qb[order]
    lo = -12
    cnt = torch.bincount(ep - lo, minlength=40); off = torch.cumsum(cnt, 0) - cnt
    Es = torch.arange(-12, 17); Cs = torch.arange(1, 17); R = 60
    E = Es[:, None, None].expand(-1, len(Cs), R).reshape(-1)
    C = Cs[None, :, None].expand(len(Es), -1, R).reshape(-1)
    k = torch.arange(16)[None, :]
    tgt = torch.where(k < C[:, None], E[:, None], E[:, None] - 1 - torch.randint(0, 6, (len(E), 16), generator=g))
    zero = (k >= C[:, None]) & ((torch.rand(len(E), 16, generator=g) < 0.15) | (tgt < lo))
    t = tgt.clamp_min(lo) - lo
    pick = off[t] + (torch.rand(len(E), 16, generator=g) * cnt[t]).long()
    A3 = torch.where(zero, torch.zeros_like(pick), qa[pick]); B3 = torch.where(zero, VALID[torch.randint(0, len(VALID), (len(E), 16), generator=g)], qb[pick])
    perm = torch.argsort(torch.rand(len(E), 16, generator=g), 1)
    rows_a.append(A3.gather(1, perm)); rows_b.append(B3.gather(1, perm))
    A = torch.cat(rows_a).to(torch.uint8); B = torch.cat(rows_b).to(torch.uint8)
    exp, _ = expected_pbits(A, B)
    out = {}
    for mode in (0, 1):
        r = P.epass(A.to(dev).contiguous(), B.to(dev).contiguous(), mode).cpu()
        out[f"mode{mode}"] = dict(rows=len(A), mismatches=int((r != exp[:, None]).any(1).sum()))
    r = P.epass64(A.to(dev).contiguous(), B.to(dev).contiguous()).cpu()
    out["base64_v3"] = dict(rows=len(A), mismatches=int((r != exp[:, None]).any(1).sum()))
    res["epass"] = out
    print("epass", out, flush=True)


def check_rz(res):
    pa, pb = torch.meshgrid(VALID, VALID, indexing="ij")
    pa, pb = pa.reshape(-1), pb.reshape(-1)
    xa, ea, _, nza = truth_fields(pa)
    xb, eb, _, nzb = truth_fields(pb)
    p = xa * xb
    ep = torch.where(nza & nzb, ea + eb, torch.full_like(ea, -12))
    offs = torch.tensor([-(2 ** 21) + 1, -(2 ** 19), -12345, -1, 0, 1, 777, 2 ** 19, 2 ** 21 - 1], dtype=f64)
    n = bad = 0
    for E in range(-12, 56):
        m = ep <= E
        pp = p[m]
        g = 2.0 ** (E - 13)
        Mm = 1.5 * 2.0 ** (E + 10)
        S = (Mm + offs[None, :] * g).expand(len(pp), -1).reshape(-1)
        a = xa[m][:, None].expand(-1, len(offs)).reshape(-1)
        b = xb[m][:, None].expand(-1, len(offs)).reshape(-1)
        want = S + torch.floor((a * b) / g) * g
        got = P.rz(a.float().to(dev), b.float().to(dev), S.float().to(dev)).cpu().double()
        n += len(want); bad += int((got != want).sum())
    res["rz_accumulate"] = dict(cases=n, mismatches=bad)
    print("rz", res["rz_accumulate"], flush=True)


def check_init(res):
    g = torch.Generator().manual_seed(11)
    n = 2_000_000
    eV = torch.randint(-25, 55, (n,), generator=g)
    sig = torch.randint(-(2 ** 21) + 1, 2 ** 21, (n,), generator=g).double()
    sig = torch.where(sig.abs() < 2 ** 20, sig + torch.sign(sig + 0.5) * 2 ** 20, sig)  # 21 significant bits
    V = sig * torch.exp2((eV - 20).double())
    V[:2000] = 0.0
    eVt = torch.where(V != 0, torch.frexp(V)[1] - 1, torch.full_like(eV, -200))
    Ep = torch.randint(-12, 56, (n,), generator=g)
    # bias towards E == exponent(V) and small gaps
    Ep = torch.where(torch.rand(n, generator=g) < 0.5, torch.clamp(eVt - torch.randint(0, 3, (n,), generator=g), -12, 55), Ep)
    E = torch.maximum(Ep, eVt)
    keep = E <= 55
    V, Ep, E = V[keep], Ep[keep], E[keep]
    gr = torch.exp2((E - 13).double())
    want = 1.5 * torch.exp2((E + 10).double()) + torch.trunc(V / gr) * gr
    got = P.init(V.float().to(dev), ((Ep + 127) << 23).int().to(dev)).cpu().double()
    got3 = P.init3(V.float().to(dev), ((Ep + 127) << 23).int().to(dev)).cpu().double()
    res["block_init"] = dict(cases=len(V), mismatches=int((got != want).sum()), mismatches_init3=int((got3 != want).sum()),
                             V_exact_in_fp32=bool((V.float().double() == V).all()))
    print("init", res["block_init"], flush=True)


def check_finish(res):
    codes = list(range(18)) + [60]
    combos = torch.tensor([(la, lb, sa, sb) for la in codes for lb in codes for sa in (0, 1) for sb in (0, 1)])
    g = torch.Generator().manual_seed(13)
    n = bad = 0
    for E in range(-12, 56):
        # each test packs 16 random combos in its 4 words x 4 bytes; every combo appears in every byte lane
        reps = 8
        idx = torch.cat([torch.randperm(len(combos), generator=g) for _ in range(16 * reps)])
        T = len(idx) // 16
        cb = combos[idx[:T * 16]].reshape(T, 4, 4, 4)  # [test, group, byte, (la, lb, sa, sb)]
        sh = (torch.arange(4) * 8)[None, None, :]
        la = (cb[..., 0] << sh).sum(-1); lb = (cb[..., 1] << sh).sum(-1)
        sa = ((cb[..., 2] * 0x80) << sh).sum(-1); sb = ((cb[..., 3] * 0x80) << sh).sum(-1)
        w = torch.cat([la, sa, lb, sb], 1).to(torch.int64)
        w = torch.where(w >= 2 ** 31, w - 2 ** 32, w).int()
        nzz = (cb[..., 0] != 60) & (cb[..., 1] != 60)
        inexact = (cb[..., 0] - 9) + (cb[..., 1] - 9) < E - 13
        cnt = (nzz & inexact & (cb[..., 2] != cb[..., 3])).sum((1, 2))
        gr = 2.0 ** (E - 13)
        Mm = 1.5 * 2.0 ** (E + 10)
        j = torch.randint(-(2 ** 20), 2 ** 20, (T,), generator=g).double()
        S = Mm + j * gr
        want = S + cnt.double() * gr - Mm
        got = P.finish(S.float().to(dev), w.to(dev).contiguous()).cpu().double()
        n += T; bad += int((got != want).sum())
    res["block_finish"] = dict(tests=n, combos_per_E=len(combos), mismatches=bad)
    print("finish", res["block_finish"], flush=True)


def check_decode_word(res):
    # every valid byte in every lane: words b0 | b1<<8 | b2<<16 | b3<<24 with lane q cycling through all codes
    codes = VALID
    n = len(codes)
    g = torch.Generator().manual_seed(19)
    lanes = [codes[torch.randperm(n, generator=g)] for _ in range(4)]
    W4 = torch.stack(lanes, 1)  # [n, 4]
    W4 = torch.cat([W4, codes[torch.randint(0, n, (200000, 4), generator=g)]])
    w = (W4[:, 0] | (W4[:, 1] << 8) | (W4[:, 2] << 16) | (W4[:, 3] << 24))
    w = torch.where(w >= 2 ** 31, w - 2 ** 32, w).int()
    out = P.decode_word(w.to(dev)).cpu().long()
    x, ea, la, nz = truth_fields(W4.reshape(-1))
    x, ea, la, nz = x.reshape(-1, 4), ea.reshape(-1, 4), la.reshape(-1, 4), nz.reshape(-1, 4)
    sh = torch.arange(4) * 8
    sg_w = ((((W4 >> 7) & 1) * 0x80) << sh).sum(1)
    la_w = (torch.where(nz, la + 9, 60) << sh).sum(1)
    x_w = (torch.where(nz, 3 * ea + 60, 0) << sh).sum(1)
    m32 = 0xFFFFFFFF
    vals = out[:, 3:7].int().view(torch.float32).double()
    res["decode_word"] = dict(words=len(w), sg_ok=bool(((out[:, 0] & m32) == sg_w).all()), la_ok=bool(((out[:, 1] & m32) == la_w).all()),
                              x_ok=bool(((out[:, 2] & m32) == x_w).all()), v_ok=bool(same_bits(vals, x)))
    print("decode_word", res["decode_word"], flush=True)


def check_trunc(res):
    g = torch.Generator().manual_seed(17)
    n = 1_000_000
    e = torch.randint(-25, 56, (n,), generator=g)
    sig = torch.randint(2 ** 23, 2 ** 24, (n,), generator=g).double() * torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0)
    V = sig * torch.exp2((e - 23).double())
    V[:100] = 0.0
    ee = torch.where(V != 0, torch.frexp(V)[1] - 1, torch.zeros_like(e))
    want = torch.trunc(V * torch.exp2((13 - ee).double())) * torch.exp2((ee - 13).double())
    got = P.trunc14(V.float().to(dev)).cpu().double()
    res["final_trunc"] = dict(cases=n, mismatches=int((got != want).sum()))
    print("trunc", res["final_trunc"], flush=True)


if __name__ == "__main__":
    P = build.get("prims")
    res = {}
    t = time.time()
    for f in (check_decode, check_decode_word, check_epass, check_rz, check_init, check_finish, check_trunc):
        f(res)
    res["seconds"] = time.time() - t
    json.dump(res, open(_bringup_out() / "prims.json", "w"), indent=1)
