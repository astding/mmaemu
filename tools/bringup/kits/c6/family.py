"""Parameterised family of FP8 block-FMA inner-product models (exact, vectorised, float64 torch).

One K=32 MMA (d = sum_k a_k b_k + c) is modelled as:
  terms -> groups (hardware block FMAs) -> per-group alignment sum -> normalisation -> chaining -> c handling.

Spec keys (defaults reproduce tcemu's 'ada' preset):
  groups     'seq16' | 'seq16_rev' | 'flat32' | 'il_pairs' | 'il_pairs_rev' | 'il_single' | 'seq8'
  chain      'early'     : the previous group's (normalised) result enters the next group's alignment as c
             'add_rne' / 'add_rz': groups computed with c = 0, results added with one fp32 addition each
  c_mode     'first'     : external c enters the alignment of the first group (Ada, toolbox early c)
             'late_rne'  : normalised group result + c with an fp32 RNE addition (tcemu 'blackwell')
             'late_rz'   : same with fp32 round-toward-zero addition
             'late_exact_rne': c added to the *unnormalised* aligned sum, one RNE rounding to fp32
             'late_tbx'  : toolbox Generic_BFMA_TC late_partial_sum (AddTwoNonNormSums, 8 extra bits, rz)
  frac       fraction bits kept below the alignment exponent E (13 Ada/Hopper, 25 B200 = 23 + neab)
  frac_out   fraction bits kept at normalisation (13 Ada/Hopper; 23 B200)
  rnd_out    'rz' | 'rd' | 'rne'   rounding at normalisation
  align      'rz' (sign-magnitude truncation) | 'rd' (two's complement truncation = floor)
  denorm_prd True: product exponent = ea + eb, significand in [0, 4); False: normalised products (exp + 1 if >= 2)
  in_sub     'clamp': subnormal inputs keep exponent emin (denormalised significand)
             'fp16': inputs converted to fp16 first (E4M3 subnormals become normal fp16 numbers)
  zero_exp   'ignore': zero products do not take part in max-exponent; 'emin': they count with exponent 2*emin
  min_exp    clamp of E from below
  c_exp_sub  'floor': exponent of a subnormal c = floor(log2|c|) (tcemu); 'm126': -126 (toolbox, fp32 field)
All arithmetic is exact in float64 (inputs FP8, c fp32); fp32 additions are rounded exactly via TwoSum.
"""
from __future__ import annotations

import torch

EMIN = {"e4m3": -6, "e5m2": -14}
NEG = -100000.0
DEFAULT = dict(groups="seq16", chain="early", c_mode="first", frac=13, frac_out=13, rnd_out="rz", align="rz",
               denorm_prd=True, in_sub="clamp", zero_exp="ignore", min_exp=-132, c_exp_sub="floor")


def spec(**kw):
    s = dict(DEFAULT); s.update(kw); return s


PRESET_SPECS = {
    "ada": spec(),
    "hopper": spec(groups="flat32"),
    "blackwell": spec(groups="flat32", frac=25, frac_out=23, c_mode="late_rne", min_exp=-133),
    # toolbox B200TC.m / Fig. 5e (tcgen05; paper: also RTX PRO 6000 mma.sync)
    "tbx_B200TC": spec(groups="flat32", frac=25, frac_out=23, min_exp=-133),
    "tbx_B200TCRN": spec(groups="flat32", frac=25, frac_out=23, rnd_out="rne", min_exp=-133),
    # paper Fig. 5c (mma.sync on H100/H200/B200 -> fp16 HMMA, interleaved pairs, c late RNE)
    "fig5c": spec(groups="il_pairs", frac=25, frac_out=23, c_mode="late_rne", min_exp=-133, in_sub="fp16"),
}


def _exp(x):
    e = (torch.frexp(x)[1] - 1).to(x.dtype)
    return torch.where(x != 0, e, torch.full_like(x, NEG))


def _pow2(e):
    return ((e.to(torch.int64) + 1023) << 52).view(torch.float64)


def group_index(groups: str, K: int = 32):
    idx = list(range(K))
    if groups == "seq16":
        return [idx[i:i + 16] for i in range(0, K, 16)]
    if groups == "seq16_rev":
        return [idx[i:i + 16] for i in range(0, K, 16)][::-1]
    if groups == "seq8":
        return [idx[i:i + 8] for i in range(0, K, 8)]
    if groups == "flat32":
        return [idx]
    if groups in ("il_pairs", "il_pairs_rev"):
        g1 = [k for k in idx if (k // 2) % 2 == 0]
        g2 = [k for k in idx if (k // 2) % 2 == 1]
        return [g1, g2] if groups == "il_pairs" else [g2, g1]
    if groups == "il_single":
        return [idx[0::2], idx[1::2]]
    raise ValueError(groups)


def products(a, b, fmt, s):
    fa, fb = (fmt, fmt) if isinstance(fmt, str) else fmt
    p = a * b

    def ex(x, f):
        e = _exp(x)
        emin = -14 if s["in_sub"] == "fp16" else EMIN[f]
        return e.clamp_min(emin)
    ep = ex(a, fa) + ex(b, fb)
    if not s["denorm_prd"]:
        ep = ep + (p.abs() >= _pow2(ep + 1)).to(ep.dtype)
    if s["zero_exp"] == "ignore":
        ep = torch.where(p != 0, ep, torch.full_like(p, NEG))
    else:
        ep = torch.where(p != 0, ep, torch.full_like(p, float(EMIN[fa] + EMIN[fb])))
    return p, ep


def _q(x, mode):
    if mode == "rz":
        return torch.trunc(x)
    if mode == "rd":
        return torch.floor(x)
    if mode == "rne":
        return torch.round(x)  # half to even
    raise ValueError(mode)


def c_exponent(c, s):
    e = _exp(c)
    if s["c_exp_sub"] == "m126":
        e = torch.where(c != 0, e.clamp_min(-126), e)
    return e


def align_sum(p, ep, c, s, frac):
    """Aligned integer sum (units of 2^(E-frac)) and E. c may be None (no c term)."""
    Ep = ep.amax(-1)
    E = Ep if c is None else torch.maximum(Ep, c_exponent(c, s))
    E = E.clamp_min(s["min_exp"])
    iq = _pow2(frac - E)
    S = _q(p * iq[..., None], s["align"]).sum(-1)
    if c is not None:
        S = S + _q(c * iq, s["align"])
    return S, E


def normalise(S, E, frac, frac_out, mode):
    d = S * _pow2(E - frac)
    ed = _exp(d).clamp_min(-126)
    r = _q(d * _pow2(frac_out - ed), mode) * _pow2(ed - frac_out)
    return torch.where(d != 0, r, d)


def _two_sum(x, y):
    s = x + y
    bp = s - x
    err = (x - (s - bp)) + (y - bp)
    return s, err


def fp32_round_exact(s, err, mode):
    """Round the exact value s + err (s = fl64(x + y), err = exact error) to fp32 (normal range assumed)."""
    es = _exp(s).clamp_min(-126)
    u = _pow2(es - 23)  # fp32 ulp in s's binade
    t = s / u          # exact (power-of-two scaling)
    if mode == "rne":
        r = torch.round(t)
        tie = (t - torch.floor(t)) == 0.5
        up = torch.floor(t) + 1
        dn = torch.floor(t)
        r = torch.where(tie & (err > 0), up, torch.where(tie & (err < 0), dn, r))
        return torch.where(s == 0, s, r * u)
    if mode == "rz":
        r = torch.trunc(t)
        exact = r == t
        # s on the fp32 grid but the true value is smaller in magnitude -> one step toward zero
        step = exact & (err != 0) & ((err > 0) != (s > 0))
        mag = r.abs()
        # stepping below a power of two uses the finer ulp of the lower binade
        is_pow2 = (mag == 2.0 ** 23) & (es > -126)
        newv = torch.where(is_pow2, (mag * 2 - 1) * (u / 2), (mag - 1) * u) * torch.sign(s)
        return torch.where(step, newv, r * u)
    raise ValueError(mode)


def fp32_add(x, y, mode):
    s, err = _two_sum(x, y)
    return fp32_round_exact(s, err, mode)


def late_tbx(S, E, frac, c):
    """Toolbox AddTwoNonNormSums(prod_max_exp, c_exp, ..., armode='rz', eab_1=8, eab_2=0) + norm_helper (rz),
    for frac = 23 + neab and fp32 output (frac_out = 23)."""
    neab = frac - 23
    ce = torch.where(c != 0, _exp(c).clamp_min(-126), torch.full_like(c, -1024.0))
    cs = torch.where(c != 0, c.abs() * _pow2(23 + neab - ce), torch.zeros_like(c))  # exact integer
    ps = S.abs(); psign = torch.sign(S); csign = torch.sign(c)
    noprod = E <= NEG / 2
    E2 = torch.where(noprod, ce, E)
    mx = torch.maximum(E2, ce)
    sh = (E2 - ce).abs()
    sh = torch.where(sh > 512, torch.zeros_like(sh), sh)
    # case 1: product sum smaller -> both << 8, product sum >> shift (truncate); neab += 8
    c1 = (E2 < mx)
    ps1 = torch.floor(ps * 256 * _pow2(-sh)); cs1 = cs * 256
    # case 2: c smaller -> c >> shift
    c2 = (ce < mx) & ~c1
    cs2 = torch.floor(cs * _pow2(-sh))
    psv = torch.where(c1, ps1, ps); csv = torch.where(c1, cs1, torch.where(c2, cs2, cs))
    nb = torch.where(c1, torch.full_like(S, neab + 8.0), torch.full_like(S, float(neab)))
    U = psv * psign + csv * csign
    d = U * _pow2(mx - 23 - nb)  # value of the unnormalised sum
    return normalise(d, torch.zeros_like(d), 0, 23, "rz")


def model(a, b, c, fmt="e4m3", s=None, K=32):
    """d for a, b: [N, K] float64 FP8 values; c: [N] float64 fp32 values."""
    s = DEFAULT if s is None else s
    a, b, c = a.double(), b.double(), c.double()
    p, ep = products(a, b, fmt, s)
    gidx = group_index(s["groups"], K)
    frac, fo, ro = s["frac"], s["frac_out"], s["rnd_out"]
    cm = s["c_mode"]
    acc = None
    ng = len(gidx)
    for gi, g in enumerate(gidx):
        gi_t = torch.tensor(g)
        pg, eg = p[:, gi_t], ep[:, gi_t]
        cin = None
        if s["chain"] == "early" and acc is not None:
            cin = acc
        if cm == "first" and gi == 0:
            cin = c
        if cm in ("late_exact_rne", "late_tbx") and gi == ng - 1:
            assert s["chain"] == "early"
            S, E = align_sum(pg, eg, cin, s, frac)
            if cm == "late_tbx":
                acc_g = late_tbx(S, E, frac, c)
            else:
                v = S * _pow2(E - frac)
                acc_g = fp32_add(v, c, "rne")
            return acc_g
        S, E = align_sum(pg, eg, cin, s, frac)
        r = normalise(S, E, frac, fo, ro)
        if s["chain"] == "early" or acc is None:
            acc = r
        else:
            acc = fp32_add(acc, r, s["chain"][4:])
    if cm in ("late_rne", "late_rz"):
        acc = fp32_add(acc, c, cm[5:])
    return acc


def bits32(x: torch.Tensor) -> torch.Tensor:
    return x.float().view(torch.int32)
