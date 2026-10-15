"""Mutant implementations injected into a TEMPORARY COPY of the mmaemu package by tools/mutants/run.py.

run.py copies this file to <tmp>/mmaemu/_mutation_impl.py and appends two lines to the module that defines the
public function being mutated, e.g. at the end of mmaemu/model/epilogue.py:

    from mmaemu._mutation_impl import epilogue_override as _mut
    apply = _mut(apply, {"epi": "sa_sb"})

Because the lines run at the end of the defining module, every later `from .epilogue import apply`, every lazy
re-export and every call through the module attribute sees the mutant. Each override family is a self-contained
re-implementation of the frozen contract with one modelling decision changed; with the empty knob set ({}) it must
reproduce mmaemu bit for bit (the "identity" mutants, which must SURVIVE the test suite -- that is what shows the
injection itself is sound).

This module must not import mmaemu.model.* at import time (it is imported while those modules execute).
"""
from __future__ import annotations

import functools

import torch

EMIN = {"e4m3": -6, "e5m2": -14}
_NEG = -100000.0
_FP8 = {torch.float8_e4m3fn: "e4m3", torch.float8_e5m2: "e5m2"}
_TD8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
STATE = {}                         # cross-override state (alpha-before-reduction)


# ---------------------------------------------------------------------------------------------------------------
# instruction model (mmaemu.model.chain.dot / chain_gemm)
CHAIN_BASE = dict(nfma=16, frac=13, frac_out=13, min_exp=-132, align="trunc", norm="trunc", emin_clamp=True,
                  c_trunc=True, ed_clamp=-126, zero_products_in_max=False, allzero_c_pass=False, m1=True,
                  ieee_zero=False, model="chain")


def _exp(x, emin=None):
    e = (torch.frexp(x)[1] - 1).to(x.dtype)
    if emin is not None:
        e = e.clamp_min(emin)
    return torch.where(x != 0, e, torch.full_like(x, _NEG))


def _pow2(e):
    return ((e.to(torch.int64) + 1023) << 52).view(torch.float64)


def _rint(x, mode):
    return torch.trunc(x) if mode == "trunc" else torch.round(x)


def _bfma(a, b, c, k, fa, fb):
    frac, frac_out = k["frac"], k["frac_out"]
    p = a * b
    ea = _exp(a, EMIN[fa] if k["emin_clamp"] else None)
    eb = _exp(b, EMIN[fb] if k["emin_clamp"] else None)
    if k["zero_products_in_max"]:        # zero products (from a zero operand) still set the block maximum
        ep = ea.where(a != 0, torch.full_like(a, float(EMIN[fa]))) + eb.where(b != 0, torch.full_like(b, float(EMIN[fb])))
    else:
        ep = torch.where(p != 0, ea + eb, torch.full_like(p, _NEG))
    E = torch.maximum(ep.amax(-1), _exp(c)).clamp_min(k["min_exp"])
    q, iq = _pow2(E - frac), _pow2(frac - E)
    cc = _rint(c * iq, k["align"]) if k["c_trunc"] else c * iq
    s = _rint(p * iq[..., None], k["align"]).sum(-1) + cc
    d = s * q
    ed = _exp(d)
    if k["ed_clamp"] is not None:
        ed = ed.clamp_min(k["ed_clamp"])
    out = torch.where(d != 0, _rint(d * _pow2(frac_out - ed), k["norm"]) * _pow2(ed - frac_out), d)
    if k["allzero_c_pass"]:              # a block whose products are all zero returns c untruncated
        out = torch.where((p == 0).all(-1), c.float().double(), out)
    if k["ieee_zero"]:                   # exact-zero results follow IEEE sign rules (-0 when all terms are -0)
        negz = (torch.signbit(p) & (p == 0)).all(-1) & (c == 0) & torch.signbit(c)
        return torch.where(out == 0, torch.where(negz, torch.full_like(out, -0.0), torch.zeros_like(out)), out)
    if k["m1"]:                          # c2-M1: an exactly-zero / truncated-to-zero block result is +0
        out = torch.where(out == 0, torch.zeros_like(out), out)
    return out


def _chain(a64, b64, c64, k, fa, fb):
    if k["model"] == "exact_rn":
        return ((a64 * b64).sum(-1) + c64).float().double()
    if k["model"] == "fp32_fma_seq":
        acc = c64.float()
        for i in range(a64.shape[-1]):
            acc = (acc.double() + a64[..., i] * b64[..., i]).float()
        return acc.double()
    K = a64.shape[-1]
    n = k["nfma"]
    if K % n:
        pad = n - K % n
        a64 = torch.nn.functional.pad(a64, (0, pad))
        b64 = torch.nn.functional.pad(b64, (0, pad))
        K += pad
    acc = c64
    for k0 in range(0, K, n):
        acc = _bfma(a64[..., k0:k0 + n], b64[..., k0:k0 + n], acc, k, fa, fb)
    return acc


def _canon_nan32(d):
    nan = torch.tensor(0x7FFFFFFF, dtype=torch.int32, device=d.device).view(torch.float32)
    return torch.where(torch.isnan(d), nan, d)


def chain_overrides(orig_dot, orig_chain_gemm, knobs):
    k = dict(CHAIN_BASE, **knobs)

    @functools.wraps(orig_dot)
    def dot(a, b, c=None, *, preset="ada", fmt=None):
        from mmaemu.model.presets import get_preset

        if get_preset(preset, warn=False).name != "ada":
            return orig_dot(a, b, c, preset=preset, fmt=fmt)
        if fmt is None or isinstance(fmt, str):
            fa_req = fb_req = fmt
        else:
            if len(fmt) != 2:
                return orig_dot(a, b, c, preset=preset, fmt=fmt)
            fa_req, fb_req = fmt
        as_values = orig_dot.__globals__["_as_values"]
        a64, fa = as_values(a, fa_req, "a")
        b64, fb = as_values(b, fb_req, "b")
        if a64.device != b64.device or a64.dim() < 1 or a64.shape != b64.shape:
            return orig_dot(a, b, c, preset=preset, fmt=fmt)
        if c is not None:
            if (not isinstance(c, torch.Tensor) or c.shape != a64.shape[:-1]
                    or c.dtype not in (torch.float32, torch.float64)):
                return orig_dot(a, b, c, preset=preset, fmt=fmt)
            if c.dtype == torch.float64 and bool(((c.float().double() != c) & ~torch.isnan(c)).any()):
                return orig_dot(a, b, c, preset=preset, fmt=fmt)
        c64 = (torch.zeros(a64.shape[:-1], dtype=torch.float64, device=a64.device) if c is None
               else c.to(device=a64.device, dtype=torch.float64))
        return _canon_nan32(_chain(a64, b64, c64, k, fa, fb).float())

    @functools.wraps(orig_chain_gemm)
    def chain_gemm(a_codes_u8, b_codes_u8, *, fmt_a, fmt_b, preset="ada"):
        from mmaemu.formats import nonfinite_mask
        from mmaemu.model.presets import get_preset

        invalid = (a_codes_u8.dtype != torch.uint8 or b_codes_u8.dtype != torch.uint8
                   or a_codes_u8.dim() != 2 or b_codes_u8.dim() != 2
                   or a_codes_u8.device != b_codes_u8.device)
        if not invalid:
            invalid = (a_codes_u8.shape[1] != b_codes_u8.shape[1] or a_codes_u8.shape[1] % 16
                       or not a_codes_u8.numel() or not b_codes_u8.numel()
                       or bool(nonfinite_mask(a_codes_u8, fmt_a).any())
                       or bool(nonfinite_mask(b_codes_u8, fmt_b).any()))
        if get_preset(preset, warn=False).name != "ada" or invalid:
            return orig_chain_gemm(a_codes_u8, b_codes_u8, fmt_a=fmt_a, fmt_b=fmt_b, preset=preset)
        A = a_codes_u8.contiguous().view(_TD8[fmt_a]).double()
        B = b_codes_u8.contiguous().view(_TD8[fmt_b]).double()
        M, N = A.shape[0], B.shape[0]
        out = torch.empty(M, N, dtype=torch.float32, device=A.device)
        t = 64
        for m0 in range(0, M, t):
            a = A[m0:m0 + t][:, None, :]
            for n0 in range(0, N, t):
                bb = B[n0:n0 + t][None, :, :]
                aa, bx = a.expand(-1, bb.shape[1], -1), bb.expand(a.shape[0], -1, -1)
                c0 = torch.zeros(aa.shape[:-1], dtype=torch.float64, device=A.device)
                out[m0:m0 + t, n0:n0 + t] = _chain(aa, bx, c0, k, fmt_a, fmt_b).float()
        return out

    return dot, chain_gemm


# ---------------------------------------------------------------------------------------------------------------
# GEMM plan (mmaemu.model.plan.make_plan / execute_plan)
PLAN_BASE = dict(split="interleaved", slow_chunk=128, slow_tail="tail", split_slow_tiles=2, split_slow_global=False,
                 fast_restart=None, swap_scheme=False, reverse_chunks=False)
EXEC_BASE = dict(chunk_sum="fp32_seq", combine="cuBLAS", alpha_before_reduction=False)
TILE_K = 64


def _pad16(idx):
    r = (-idx.numel()) % 16
    return torch.cat([idx, torch.full((r,), -1, dtype=torch.int64)]) if r else idx


def _chunks_1d(idx, size, tail="tail"):
    L = idx.numel()
    if tail == "head" and L % size:
        r = L % size
        bounds = [(0, r)] + [(j, j + size) for j in range(r, L, size)]
    else:
        bounds = [(j, min(L, j + size)) for j in range(0, L, size)]
    return tuple(_pad16(idx[j0:j1]) for j0, j1 in bounds)


def plan_overrides(orig_make_plan, GemmPlan, knobs):
    k = dict(PLAN_BASE, **knobs)

    @functools.wraps(orig_make_plan)
    def make_plan(K, *, fast, S=1, scheme="none", preset="ada"):
        from mmaemu.model.presets import get_preset

        baseline = orig_make_plan(K, fast=fast, S=S, scheme=scheme, preset=preset)
        if get_preset(preset, warn=False).name != "ada" or K == 0:
            return baseline
        scheme = baseline.scheme
        if k["swap_scheme"] and S > 1:
            scheme = {"serial": "pairwise", "pairwise": "serial"}[scheme]
        if S == 1:
            idx = torch.arange(K)
            if fast and k["fast_restart"] is None:
                return GemmPlan(K=K, fast=True, S=1, scheme="none", chunks=((_pad16(idx),),),
                                identity=(K % 16 == 0))
            size = k["fast_restart"] if fast else k["slow_chunk"]
            if not fast and k["slow_tail"] == "tail":
                pad = (-idx.numel()) % size
                idx = torch.cat([idx, torch.full((pad,), -1, dtype=torch.int64)])
            ch = _chunks_1d(idx, size, "tail" if fast else k["slow_tail"])
            if k["reverse_chunks"]:
                ch = ch[::-1]
            return GemmPlan(K=K, fast=fast, S=1, scheme="none", chunks=(ch,), identity=False)
        T = -(-K // TILE_K)
        Tp = -(-T // S) * S
        ar = torch.arange(Tp * TILE_K)
        if k["split"] == "interleaved":
            kk = ar.reshape(Tp // S, S, TILE_K).permute(1, 0, 2).reshape(S, -1)
        else:                                     # contiguous slices of ceil(T/S) tiles per split
            kk = ar.reshape(S, -1)
        kk = torch.where(kk < K, kk, torch.full_like(kk, -1))
        out = []
        for s in range(S):
            row = kk[s]
            if fast:
                out.append((row,))
                continue
            if k["split_slow_global"]:            # restart on global 128-element boundaries (= every own tile)
                size = TILE_K
            else:
                size = k["split_slow_tiles"] * TILE_K * k["slow_chunk"] // 128
            ch = tuple(row[j:j + size] for j in range(0, row.numel(), size))
            if k["reverse_chunks"]:
                ch = ch[::-1]
            out.append(ch)
        return GemmPlan(K=K, fast=fast, S=S, scheme=scheme, chunks=tuple(out), identity=False)

    return make_plan


def _gather(x, idx):
    xp = torch.nn.functional.pad(x, (0, 1))
    ii = torch.where(idx >= 0, idx, torch.full_like(idx, x.shape[1])).to(x.device)
    return xp[:, ii].contiguous()


def _combine_chunks(parts, mode):
    if mode == "fp64":
        return sum(p.double() for p in parts).float()
    if mode == "pairwise":
        ps = list(parts)
        while len(ps) > 1:
            ps = [ps[i] + ps[i + 1] if i + 1 < len(ps) else ps[i] for i in range(0, len(ps), 2)]
        return torch.zeros_like(ps[0]) + ps[0]
    seq = parts[::-1] if mode == "reverse" else parts
    acc = torch.zeros_like(seq[0])
    for p in seq:
        acc = acc + p
    return acc


def _combine_splits(P, scheme):
    S = len(P)
    if S == 1:
        return P[0]
    if scheme == "serial":
        acc = P[0]
        for s in range(1, S):
            acc = acc + P[s]
        return acc
    h = -(-S // 2)
    acc = P[S - 1]
    for i in range(h):
        acc = acc + (P[i] + P[i + h] if i + h < S - 1 else P[i])
    return acc


def execute_override(orig_execute_plan, knobs):
    k = dict(EXEC_BASE, **knobs)

    @functools.wraps(orig_execute_plan)
    def execute_plan(plan, a_u8, b_u8, *, fmt_a, fmt_b, chain_fn):
        from mmaemu.model import specials          # lazy: imported after mmaemu.model is initialised
        invalid = (a_u8.dtype != torch.uint8 or b_u8.dtype != torch.uint8 or a_u8.dim() != 2 or b_u8.dim() != 2
                   or a_u8.device != b_u8.device)
        if not invalid:
            invalid = a_u8.shape[1] != b_u8.shape[1] or a_u8.shape[1] != plan.K
        if invalid:
            return orig_execute_plan(plan, a_u8, b_u8, fmt_a=fmt_a, fmt_b=fmt_b, chain_fn=chain_fn)
        M, N = a_u8.shape[0], b_u8.shape[0]
        if M == 0 or N == 0 or plan.K == 0:
            return orig_execute_plan(plan, a_u8, b_u8, fmt_a=fmt_a, fmt_b=fmt_b, chain_fn=chain_fn)
        a_s, fa = specials.sanitize(a_u8, fmt_a)
        b_s, fb = specials.sanitize(b_u8, fmt_b)
        parts = []
        for chunks in plan.chunks:
            res = [chain_fn(a_s.contiguous(), b_s.contiguous()) if plan.identity else
                   chain_fn(_gather(a_s, idx), _gather(b_s, idx)) for idx in chunks]
            parts.append(res[0] if plan.fast and k["chunk_sum"] == "fp32_seq"
                         else _combine_chunks(res, k["chunk_sum"]))
        acc = _combine_splits(parts, plan.scheme)
        if fa or fb:
            acc = specials.nonfinite_patch(acc, a_u8, b_u8, fmt_a, fmt_b)
        if k["alpha_before_reduction"] and plan.S > 1:
            STATE[id(acc)] = (acc, parts, plan.scheme)
        return acc

    return execute_plan


# ---------------------------------------------------------------------------------------------------------------
# epilogue (mmaemu.model.epilogue.apply)
EPI_BASE = dict(epi="fma64", saturate=True, sr_nonfp8=False, sr_after_sat=False, nan="canonical", zero="asis",
                sign_flip_nonzero=False)
_MAX = {torch.float8_e4m3fn: 448.0, torch.float8_e5m2: 57344.0}
_CANON = {torch.float32: 0x7FFFFFFF, torch.bfloat16: 0x7FFF, torch.float16: 0x7FFF, torch.float8_e4m3fn: 0x7F,
          torch.float8_e5m2: 0x7F}
_IV = {1: torch.uint8, 2: torch.int16, 4: torch.int32}


def _round_once(x, dtype):
    """float64 -> bf16/fp16 with ONE round-to-nearest-even (torch's cast goes through fp32)."""
    p, emin = {torch.bfloat16: (8, -126), torch.float16: (11, -14)}[dtype]
    fin = torch.isfinite(x) & (x != 0)
    e = _exp(torch.where(fin, x, torch.ones_like(x))).clamp_min(emin)
    ulp = _pow2(e - (p - 1))
    r = torch.where(fin, torch.round(x / ulp) * ulp, x)
    return r.to(dtype)


def epilogue_override(orig_apply, knobs):
    k = dict(EPI_BASE, **knobs)

    @functools.wraps(orig_apply)
    def apply(acc32, scale_a, scale_b, *, out_dtype, scale_result=None):
        dev = acc32.device
        sa = scale_a.float().reshape(()).to(dev)
        sb = scale_b.float().reshape(()).to(dev)
        alpha = sa * sb
        st = STATE.pop(id(acc32), None)
        if st is not None and st[0] is acc32:           # alpha applied to each split partial, then reduced
            _, parts, scheme = st
            y = _combine_splits([(p.double() * alpha.double() + 0.0).float() for p in parts], scheme)
            y = torch.where(torch.isnan(acc32) | torch.isinf(acc32), (acc32.double() * alpha.double()).float(), y)
        elif k["epi"] == "fma64":
            y = (acc32.double() * alpha.double() + 0.0).float()
        elif k["epi"] == "sa_sb":
            y = ((acc32 * sa) * sb)
        elif k["epi"] == "fp32_mul_add0":
            y = (acc32 * alpha) + 0.0
        elif k["epi"] == "single":
            y = None
        else:
            raise ValueError(k["epi"])
        fp8 = out_dtype in _MAX
        if y is None:                                   # single rounding from the exact product
            ex = acc32.double() * alpha.double() + 0.0
            if out_dtype in (torch.bfloat16, torch.float16):
                out = _round_once(ex, out_dtype)
            else:
                y = ex.float()
        if y is not None:
            if scale_result is not None and (fp8 or k["sr_nonfp8"]) and not k["sr_after_sat"]:
                y = y * scale_result.float().reshape(()).to(dev)
            if fp8 and k["saturate"]:
                y = y.clamp(-_MAX[out_dtype], _MAX[out_dtype])
            if scale_result is not None and fp8 and k["sr_after_sat"]:
                y = y * scale_result.float().reshape(()).to(dev)
            if k["sign_flip_nonzero"]:                  # control mutant
                y = torch.where(y != 0, -y, y)
            out = y.to(out_dtype)
        if k["zero"] == "plus":
            out = torch.where(out == 0, torch.zeros_like(out), out)
        elif k["zero"] == "flip":
            out = torch.where(out == 0, -out, out)
        nan = torch.isnan(out.float())
        if bool(nan.any()) and k["nan"] != "keep":
            iv = _IV[out.element_size()]
            ob = out.view(iv).clone()
            val = _CANON[out_dtype] if k["nan"] == "canonical" else -1
            ob[nan] = torch.tensor(val, dtype=torch.int64).to(iv)
            out = ob.view(out_dtype)
        return out

    return apply


# ---------------------------------------------------------------------------------------------------------------
# public API (mmaemu.api.scaled_mm)
def api_override(orig_scaled_mm, knobs):
    mode = knobs.get("mode", "identity")

    if mode == "default_fast":
        def scaled_mm(input, mat2, scale_a, scale_b, bias=None, scale_result=None, out_dtype=None,
                      use_fast_accum=True, **kw):
            return orig_scaled_mm(input, mat2, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum, **kw)
    elif mode == "default_fp32":
        def scaled_mm(input, mat2, scale_a, scale_b, bias=None, scale_result=None, out_dtype=None,
                      use_fast_accum=False, **kw):
            od = torch.float32 if out_dtype is None else out_dtype
            return orig_scaled_mm(input, mat2, scale_a, scale_b, bias, scale_result, od, use_fast_accum, **kw)
    elif mode == "drop_positional_bias":
        def scaled_mm(input, mat2, scale_a, scale_b, *args, **kw):
            if args:
                args = (None,) + tuple(args[1:])
            return orig_scaled_mm(input, mat2, scale_a, scale_b, *args, **kw)
    elif mode == "allow_e4m3_e5m2_out":
        def scaled_mm(input, mat2, scale_a, scale_b, bias=None, scale_result=None, out_dtype=None,
                      use_fast_accum=False, **kw):
            e4, e5 = torch.float8_e4m3fn, torch.float8_e5m2
            if out_dtype is e5 and input.dtype is e4 and mat2.dtype is e4 and bias is None:
                import mmaemu
                from mmaemu.model import epilogue
                acc = mmaemu.accumulate(input, mat2.t(), use_fast_accum=use_fast_accum,
                                        splitk=kw.get("splitk", "auto"), backend=kw.get("backend", "auto"))
                y = epilogue.apply(acc, scale_a, scale_b, out_dtype=e5, scale_result=scale_result)
                if kw.get("out") is not None:
                    kw["out"].copy_(y)
                    return kw["out"]
                return y
            return orig_scaled_mm(input, mat2, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum, **kw)
    else:
        def scaled_mm(*args, **kw):
            return orig_scaled_mm(*args, **kw)
    functools.update_wrapper(scaled_mm, orig_scaled_mm, assigned=("__module__", "__name__", "__qualname__", "__doc__"))
    return scaled_mm
