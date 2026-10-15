"""Per-device gates of the gated fast kernels (registered with mmaemu.backends.cuda.spec.register_gate).

The fast kernels borrow the rounding behaviour of the fp16 / bf16 tensor cores, which is not specified by PTX and
differs between GPU generations.  Each gate first probes exactly the instruction property its kernel relies on, with
the same instructions, data path and build flags (probes.cu ships in the fatbin), and then checks the kernel itself at
GEMM level against reference results.  A gate never approximates: any difference rejects the kernel.

"design_a" (tc3, p3 tc_emu3 cfg 6)
    1. property: full-tile two-sign dummy-accumulator check of mma.sync.m16n8k8 f16 -> f32 (probes.cu
       mmaemu_probe_hmma_f16_k8).  Every one of the 128 outputs of a 16 x 8 tile carries its own 8-term vector; each
       vector runs with c = +2^(E+11) and with c = -2^(E+11) and the gate tests exactly what tc3 computes from the two
       results (tc3.cu: s = rp < cp ? rp - cp : rn + cp) against the exact half sum S*u, u = 2^(E-13),
       S = sum_k trunc(a_k b_k / u).  This is the sharpened check of the tcemu audit's
       verify-p3-tensorcore-exact-1 (scripts/sim_gate_v2.py, logs/gpu_run2.json), which replaces p3
       scripts/selftest.py:58 (one random sign per vector, same-sign half compared with the Ada truncation model):
       that check falsely rejects hardware whose only difference is an RNE final normalisation, under which tc3 is
       still exact (sim_gate_v2.log: "ada_final_RNE" instr_A_v1 3705/4096 -> v2 4096/4096, truth_A True).
       Domain: 4 data classes x block-exponent offsets E - maxep in [0, 55] (covers every E tc3 can produce for
       K < 2^24, see docs/kernels_fast.md) + all-zero-product outputs with tiny c; 16,384 vectors.
    2. xpass: the base-32 exponent pass of tc3 (bf16 m16n8k16 on 2^(5 ea) x 2^(5 eb), c = +0, probes.cu
       mmaemu_probe_mma_bf16_raw) over the structural rows of the window gate: tc3's table lookup
       s_cpx[exponent field of X] must give 2^(maxep + 11) for every output (window [1, 16] x 2^(5 maxep)).
    3. gemm: tc3 on 48x40x256, 17x9x80 and 130x136x1936 (8 cases) bitwise against stored reference results.

"window" (v9, p2 v9 cfg 1)
    1. epass: the v9 E-pass probe (probes.cu mmaemu_probe_epass_raw: decode_word -> byte-placed bf16 2^(6ea-7) ->
       ldmatrix -> mma.sync.m16n8k16 bf16, c = 2^-86 -- v9's own data path) over the structural rows (block exponent
       E in [-12, 16] x 1..16 terms at E x 4 patterns of the other terms x 32 placements: 58,880 rows, all 128 row x
       column pairs of each tile evaluated) plus 256 random tiles.  Each output d must satisfy
       d / 2^(6E-14) in [1, 17] ("window.ratio"), exponent offset d_field - (6E + 113) in {0..4} ("window.offset") and
       v9's mapping pbits64_raw(d) == 2^E ("window.mapping").  Port of
       the tcemu audit's verify-p2-int-mixed-3/analyse.py (hw part) with vectors.py structural().
    2. gemm: v9 on the same 8 GEMM cases.

Reference results of the GEMM checks are stored in data/gate_gemm_v1.npz (uint32 bit patterns, no pickle), so a gate
does not run the CPU reference.  Their inputs come
from a self-contained integer generator (splitmix64), so they do not depend on torch's RNG; the file records the
sha256 of every input, the test suite recomputes every expected matrix with the mmaemu reference
(tests/test_gates_hmma.py) and ``python -m mmaemu.backends.cuda.fast.gates --write-gemm-data`` regenerates it.

Gate functions follow the frozen signature fn(spec, device) -> GateResult; the optional keyword ``runner`` replaces
the device (used by the tests to feed CPU simulations of other HMMA designs).  Reasons start with a stable code:
"design_a.property", "design_a.xpass", "design_a.gemm", "window.ratio", "window.offset", "window.mapping",
"window.gemm", "<gate>.error" (probe / kernel unavailable or failed) or "forced" (MMAEMU_TEST_FORCE_FAIL).
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import time
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from mmaemu.backends.cuda.spec import GateResult, register_gate

GATE_VERSION = "1"
GEMM_DATA = Path(__file__).resolve().parent / "data" / "gate_gemm_v1.npz"

# ------------------------------------------------------------------------------------------------ deterministic data
_GOLD, _C1, _C2 = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


def _u32(tag: str, n: int) -> np.ndarray:
    """n uint32 values of splitmix64 over a counter seeded by crc32(tag): bit-identical everywhere."""
    seed = np.uint64(zlib.crc32(tag.encode()))
    with np.errstate(over="ignore"):
        z = (np.arange(n, dtype=np.uint64) + (seed << np.uint64(32))) * _GOLD
        z = (z ^ (z >> np.uint64(30))) * _C1
        z = (z ^ (z >> np.uint64(27))) * _C2
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(32)).astype(np.uint32)


def _e4m3_tables():
    val, ea, nz = np.zeros(256), np.zeros(256, np.int64), np.zeros(256, bool)
    for c in range(256):
        e, m = (c >> 3) & 15, c & 7
        mag = m * 2.0 ** -9 if e == 0 else (8 + m) * 2.0 ** (e - 10)
        val[c] = -mag if c & 0x80 else mag
        nz[c] = (c & 0x7F) != 0
        ea[c] = (max(e, 1) - 7) if nz[c] else 0        # exponent clamped at emin = -6
    return val, ea, nz


E4M3_VAL, E4M3_EA, E4M3_NZ = _e4m3_tables()
KINDS = ("bell", "allcodes", "tiny", "sat", "blockscale", "decay")


def codes(kind: str, shape: tuple, tag: str) -> np.ndarray:
    """E4M3 codes (uint8, never a NaN code) of one data class.
      bell       exponent field 6..14 (triangular, mode 10: magnitudes ~ 2^3, like randn*40), 2% zeros
      allcodes   uniform over the 254 finite codes (subnormals, +-0, +-448)
      tiny       exponent field 0..2 (subnormal-heavy), 20% zeros
      sat        +-448
      blockscale one random exponent field 0..14 per (row, 16-block): block maxima of very different size
      decay      exponent field decreasing along K (14 -> 0): later blocks are dominated by the accumulator
      mixed      each row drawn from one of the classes above"""
    rows, K = shape
    n = rows * K
    r = _u32(tag, n).reshape(rows, K)
    s = ((r >> 19) & 1).astype(np.uint8) << 7
    m = ((r >> 16) & 7).astype(np.uint8)
    if kind == "mixed":
        pick = _u32(tag + "/rows", rows) % len(KINDS)
        out = np.empty(shape, np.uint8)
        for i, k in enumerate(KINDS):
            sel = np.nonzero(pick == i)[0]
            if len(sel):
                out[sel] = codes(k, (len(sel), K), f"{tag}/{k}")
        return out
    if kind == "allcodes":
        v = (r % 254).astype(np.int64)
        return (v + (v >= 0x7F)).astype(np.uint8)
    if kind == "bell":
        e = (6 + r % 5 + (r >> 8) % 5).astype(np.uint8)
        c = s | (e << 3) | m
        return np.where((r >> 21) % 50 == 0, np.uint8(0), c).astype(np.uint8)
    if kind == "tiny":
        e = (r % 3).astype(np.uint8)
        c = s | (e << 3) | m
        return np.where((r >> 21) % 5 == 0, np.uint8(0), c).astype(np.uint8)
    if kind == "sat":
        return (s | 0x7E).astype(np.uint8)
    if kind == "blockscale":
        nb = (K + 15) // 16
        eb = (_u32(tag + "/blk", rows * nb).reshape(rows, nb) % 15).astype(np.uint8)
        e = np.repeat(eb, 16, axis=1)[:, :K]
        return (s | (e << 3) | m).astype(np.uint8)
    if kind == "decay":
        k = np.arange(K)[None, :]
        e = np.clip(14 - k // 96 - (r % 3).astype(np.int64), 0, 14).astype(np.uint8)
        return (s | (e << 3) | m).astype(np.uint8)
    raise ValueError(kind)


#: (case name, data class, M, N, K) of the GEMM-level checks (48x40x256, 17x9x80, 130x136x1936).
GEMM_CASES = (
    ("bell_48x40x256", "bell", 48, 40, 256),
    ("allcodes_48x40x256", "allcodes", 48, 40, 256),
    ("tiny_48x40x256", "tiny", 48, 40, 256),
    ("bell_17x9x80", "bell", 17, 9, 80),
    ("allcodes_17x9x80", "allcodes", 17, 9, 80),
    ("tiny_17x9x80", "tiny", 17, 9, 80),
    ("bell_130x136x1936", "bell", 130, 136, 1936),
    ("mixed_130x136x1936", "mixed", 130, 136, 1936),
)


def gemm_inputs(name: str):
    case = next(c for c in GEMM_CASES if c[0] == name)
    _, kind, M, N, K = case
    return codes(kind, (M, K), f"gemm/{name}/a"), codes(kind, (N, K), f"gemm/{name}/b")


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


@lru_cache(None)
def gemm_expected() -> dict:
    """{case: uint32 [M, N]} from data/gate_gemm_v1.npz; the input sha256 recorded there must match the generator."""
    z = np.load(GEMM_DATA, allow_pickle=False)
    meta = json.loads(bytes(z["meta"]).decode())
    out = {}
    for name, *_ in GEMM_CASES:
        a, b = gemm_inputs(name)
        if meta["inputs"][name] != [_sha(a), _sha(b)]:
            raise RuntimeError(f"gate data: generated inputs of {name} differ from {GEMM_DATA.name}")
        out[name] = z[name]
    return out


def write_gemm_data(path: Path = GEMM_DATA, device: str = "cpu") -> dict:
    """Recompute every expected matrix with the mmaemu reference (model.chain.chain_gemm) and write the npz."""
    from mmaemu.model.chain import chain_gemm
    arrays, meta = {}, {"format": "mmaemu-ws4-gate-gemm/1", "reference": "mmaemu.model.chain.chain_gemm (ada)",
                        "cases": [list(c) for c in GEMM_CASES], "inputs": {}}
    for name, *_ in GEMM_CASES:
        a, b = gemm_inputs(name)
        ref = chain_gemm(torch.from_numpy(a).to(device), torch.from_numpy(b).to(device), fmt_a="e4m3", fmt_b="e4m3")
        arrays[name] = ref.cpu().contiguous().view(torch.int32).numpy().view(np.uint32)
        meta["inputs"][name] = [_sha(a), _sha(b)]
    arrays["meta"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    gemm_expected.cache_clear()
    return meta


# ---------------------------------------------------------------------------------------------- probe vector sets
_NONE = -999


@lru_cache(None)
def structural_tiles():
    """Port of verify-p2-int-mixed-3 vectors.structural() + hw_collect.struct_tiles(), with the integer generator.

    Rows: for every block exponent E in [-12, 16], every count n = 1..16 of terms with exponent sum E, and the other
    terms zero / one binade below (E-1) / at the floor (-12) / random below, placed first, last or at 30 random
    positions: 1,840 x 32 = 58,880 (A row, B row) pairs of E4M3 codes (powers of two, exponent sum split over a and b).
    Tiles: rows 0-7 of each A tile and the 8 B rows are structural pairs (diagonal (i, i)); A rows 8-15 are random
    codes; all 128 (row, column) pairs are evaluated.  256 random tiles (4 classes) are appended.
    Returns At [T, 16, 16] and Bt [T, 8, 16] uint8 codes."""
    combos = []
    for E in range(-12, 17):
        for n in range(1, 17):
            for pat in ("zero", "Em1", "floor", "rand"):
                if pat == "Em1" and E - 1 < -12:
                    continue
                combos.append((E, n, pat))
    C = len(combos)
    base = np.full((C, 16), _NONE, np.int64)
    rnd = _u32("struct/rand", C * 16).reshape(C, 16)
    for i, (E, n, pat) in enumerate(combos):
        base[i, :n] = E
        if pat == "Em1":
            base[i, n:] = E - 1
        elif pat == "floor" and E > -12:
            base[i, n:] = -12
        elif pat == "rand" and E - 1 >= -12:
            base[i, n:] = -12 + (rnd[i, n:] % (E + 12)).astype(np.int64)
    keys = _u32("struct/perm", C * 30 * 16).reshape(C, 30, 16)
    perms = np.concatenate([np.broadcast_to(np.arange(16), (C, 1, 16)), np.broadcast_to(np.arange(16)[::-1], (C, 1, 16)),
                            np.argsort(keys, axis=-1, kind="stable")], axis=1)            # [C, 32, 16]
    s = np.take_along_axis(np.broadcast_to(base[:, None, :], perms.shape), perms, axis=-1).reshape(-1, 16)
    ea = np.maximum(-6, s - 8)
    eb = s - ea
    none = s == _NONE
    A = np.where(none, 0, (ea + 7) << 3).astype(np.uint8)
    B = np.where(none, 0, (eb + 7) << 3).astype(np.uint8)
    T = A.shape[0] // 8
    At = np.concatenate([A[:T * 8].reshape(T, 8, 16), codes("allcodes", (T * 8, 16), "struct/extra").reshape(T, 8, 16)], 1)
    Bt = B[:T * 8].reshape(T, 8, 16)
    ra, rb = [], []
    for i, kind in enumerate(("allcodes", "bell", "tiny", "blockscale")):
        ra.append(codes(kind, (64 * 16, 16), f"struct/rnd{i}/a").reshape(64, 16, 16))
        rb.append(codes(kind, (64 * 8, 16), f"struct/rnd{i}/b").reshape(64, 8, 16))
    return np.concatenate([At] + ra), np.concatenate([Bt] + rb)


def _pair_maxexp(At: np.ndarray, Bt: np.ndarray):
    """max over k of ea + eb for the nonzero products of every (row i, column j) pair: ([T,16,8] int, any-nonzero)."""
    ea, eb = E4M3_EA[At].astype(np.int16), E4M3_EA[Bt].astype(np.int16)
    nz = E4M3_NZ[At][:, :, None, :] & E4M3_NZ[Bt][:, None, :, :]
    s = np.where(nz, ea[:, :, None, :] + eb[:, None, :, :], np.int16(-1000))
    return s.max(-1).astype(np.int64), nz.any(-1)


@lru_cache(None)
def design_a_vectors(tiles_per_class: int = 32):
    """Full-tile design-A probe vectors: A [T,16,8] and B [T,8 (n),8 (k)] E4M3 values (float64), the positive dummy
    accumulator cp = 2^(E+11) [T,16,8] and the exact half sum S*u [T,16,8] (u = 2^(E-13)).
    E = min(max(maxep, -12) + offset, 43), offset in [0, 7] for half of the outputs and in [0, 55] for the other half;
    outputs whose 8 products are all zero get E in [-30, 41] or E = -127 (tc3's c for acc == +0)."""
    A_l, B_l = [], []
    for kind in ("bell", "allcodes", "tiny", "blockscale"):
        A_l.append(codes(kind, (tiles_per_class * 16, 8), f"design_a/{kind}/a").reshape(-1, 16, 8))
        B_l.append(codes(kind, (tiles_per_class * 8, 8), f"design_a/{kind}/b").reshape(-1, 8, 8))
    Ac, Bc = np.concatenate(A_l), np.concatenate(B_l)
    T = Ac.shape[0]
    # a few zero rows / columns so that some outputs have no nonzero product
    zr = _u32("design_a/zero_rows", T * 16).reshape(T, 16) % 16 == 0
    Ac[zr] = 0
    a, b = E4M3_VAL[Ac], E4M3_VAL[Bc]
    maxep, anynz = _pair_maxexp(Ac, Bc)
    r = _u32("design_a/offset", T * 128 * 2).reshape(2, T, 16, 8).astype(np.int64)
    off = np.where(r[0] % 2 == 0, r[1] % 8, r[1] % 56).astype(np.int64)
    zE = np.where(r[1] % 7 == 0, -127, -30 + (r[0] >> 8) % 72).astype(np.int64)
    E = np.where(anynz, np.minimum(np.maximum(maxep, -12) + off, 43), zE)   # 43 >= every E reachable for K < 2^24
    u = np.ldexp(1.0, E - 13)
    p = a[:, :, None, :] * b[:, None, :, :]                                   # [T,16,8,8] exact in float64
    S = np.trunc(p / u[..., None]).sum(-1)                                   # exact (|S| < 2^18)
    # + 0.0: an all-truncated half (S = -0.0 in float64) is +0 in tc3's combine (rn + cp = -cp + cp = +0, RN)
    return dict(a=a, b=b, cp=np.ldexp(1.0, E + 11), want=S * u + 0.0, E=E, codes_a=Ac, codes_b=Bc)


def _f32bits(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, np.float32).view(np.uint32)


# --------------------------------------------------------------------------------------------- property evaluation
def eval_design_a(vec: dict, rp: np.ndarray, rn: np.ndarray) -> tuple[bool, str, dict]:
    """tc3's combine of the two HMMA results (tc3.cu: s = rp < cp ? rp - cp : rn + cp, fp32 RN) == S*u, bitwise."""
    cp = vec["cp"].astype(np.float32)
    rp32, rn32 = np.asarray(rp, np.float32), np.asarray(rn, np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        sh = np.where(rp32 < cp, (rp32.astype(np.float64) - cp).astype(np.float32),
                      (rn32.astype(np.float64) + cp).astype(np.float32))
    bad = _f32bits(sh) != _f32bits(vec["want"])
    n, tot = int(bad.sum()), bad.size
    det = dict(outputs=tot, signs=2, mismatches=n)
    if n:
        t, i, j = (int(x[0]) for x in np.nonzero(bad))
        det["first"] = dict(tile=t, row=i, col=j, E=int(vec["E"][t, i, j]), a=vec["a"][t, i].tolist(),
                            b=vec["b"][t, j].tolist(), rp=float(rp32[t, i, j]), rn=float(rn32[t, i, j]),
                            got=float(sh[t, i, j]), want=float(vec["want"][t, i, j]))
        return False, f"design_a.property: {n}/{tot} probe outputs violate the dummy-accumulator property " \
                      f"(the fp16 HMMA is not an 8-term F=24 design-A device; tc3 would be inexact)", det
    return True, "", det


def _cpx_table() -> np.ndarray:
    """tc3.cu s_cpx: biased exponent of X -> fp32 bits of 2^(maxep + 11) (0 outside [-12, 16])."""
    t = np.zeros(256, np.int64)
    for i in range(256):
        ex = i - 127
        mx = (ex + 60) // 5 - 12 if ex + 60 >= 0 else -1000
        t[i] = 0 if (i == 0 or mx < -12 or mx > 16) else (mx + 138) << 23
    return t


_CPX = _cpx_table()


def enc1(c: np.ndarray) -> np.ndarray:
    """tc3.cu enc1: bf16 bits of 2^(5 ea) (ea clamped at -6), 0 for +-0."""
    e = (c.astype(np.int64) >> 3) & 15
    return np.where((c & 0x7F) != 0, (5 * np.maximum(e, 1) + 92) << 7, 0).astype(np.uint16)


def eval_xpass(At: np.ndarray, Bt: np.ndarray, d_bits: np.ndarray) -> tuple[bool, str, dict]:
    maxep, anynz = _pair_maxexp(At, Bt)
    db = np.asarray(d_bits).astype(np.int64) & 0xFFFFFFFF
    got = _CPX[np.minimum(db >> 23, 255)]          # tc3 indexes s_cpx[bits >> 23]; X must be >= +0 (sign bit clear)
    want = np.where(anynz, (maxep + 138) << 23, 0)
    bad = (got != want) | ((db >> 31) != 0)
    n = int(bad.sum())
    det = dict(outputs=int(bad.size), mismatches=n)
    if n:
        t, i, j = (int(x[0]) for x in np.nonzero(bad))
        det["first"] = dict(tile=t, row=i, col=j, maxep=int(maxep[t, i, j]), d_bits=hex(int(db[t, i, j])))
        return False, f"design_a.xpass: {n}/{bad.size} exponent-pass outputs map to the wrong block exponent " \
                      f"(bf16 HMMA window property violated)", det
    return True, "", det


def eval_window(At: np.ndarray, Bt: np.ndarray, d_bits: np.ndarray) -> tuple[bool, str, dict]:
    """analyse.py (hw part): ratio d / 2^(6E-14) in [1, 17], offset in {0..4}, v9's pbits64_raw mapping == 2^E."""
    maxep, anynz = _pair_maxexp(At, Bt)
    E = np.maximum(np.where(anynz, maxep, -12), -12)
    db = np.asarray(d_bits).astype(np.int64) & 0xFFFFFFFF
    d = db.astype(np.uint32).view(np.float32).astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        ratio = d / np.ldexp(1.0, 6 * E - 14)
    off = ((db >> 23) & 0xFF) - (6 * E + 113)
    mapped = (((db >> 23) * 1400832 + 907255808) & 0xFFFFFFFF) & 0x7F800000
    checks = (("window.ratio", ~((ratio >= 1.0) & (ratio <= 17.0))),
              ("window.offset", ~((off >= 0) & (off <= 4))),
              ("window.mapping", mapped != (E + 127) << 23))
    det = dict(outputs=int(db.size), ratio_min=float(np.nanmin(ratio)), ratio_max=float(np.nanmax(ratio)),
               offset_hist={int(k): int(v) for k, v in zip(*np.unique(off, return_counts=True))})
    for code, bad in checks:
        n = int(bad.sum())
        det[code.split(".")[1] + "_violations"] = n
    for code, bad in checks:
        n = int(bad.sum())
        if n:
            t, i, j = (int(x[0]) for x in np.nonzero(bad))
            det["first"] = dict(tile=t, row=i, col=j, E=int(E[t, i, j]), d=float(d[t, i, j]), offset=int(off[t, i, j]))
            return False, f"{code}: {n}/{db.size} E-pass outputs outside the window (v9 would be inexact)", det
    return True, "", det


def eval_gemm(kernel: str, run) -> tuple[bool, str, dict]:
    exp = gemm_expected()
    det, tot_bad, first = {}, 0, None
    for name, kind, M, N, K in GEMM_CASES:
        a, b = gemm_inputs(name)
        out = run(torch.from_numpy(a), torch.from_numpy(b))
        got = out.detach().cpu().contiguous().view(torch.int32).numpy().view(np.uint32)
        nb = int((got != exp[name]).sum())
        det[name] = nb
        tot_bad += nb
        if nb and first is None:
            first = name
    if tot_bad:
        return False, f"{kernel}.gemm: {tot_bad} bit mismatches against the reference (first case {first})", det
    return True, "", det


# -------------------------------------------------------------------------------------------------- device runner
def _ptr(t: torch.Tensor):
    return ctypes.c_void_p(t.data_ptr())


class DeviceRunner:
    """Runs the probes (module "probes") and the kernel under test on a CUDA device through the CUDA runtime
    (mmaemu.backends.cuda.modules.launch / kernels.chain_fn_for).  All inputs and outputs are CPU tensors/arrays."""

    def __init__(self, device):
        self.device = torch.device(device)

    def _launch(self, entry, T, args):
        from mmaemu.backends.cuda import modules
        modules.launch("probes", entry, (T, 1, 1), (32, 1, 1), 0, args, self.device)

    def hmma_f16_k8(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        T = a.shape[0]
        A = torch.from_numpy(a).to(torch.float16).contiguous().to(self.device)
        B = torch.from_numpy(b).to(torch.float16).contiguous().to(self.device)
        C = torch.from_numpy(c).to(torch.float32).contiguous().to(self.device)
        D = torch.empty(T, 16, 8, dtype=torch.float32, device=self.device)
        self._launch("mmaemu_probe_hmma_f16_k8", T, [_ptr(A), _ptr(B), _ptr(C), _ptr(D)])
        return D.cpu().numpy()

    def epass_raw(self, At: np.ndarray, Bt: np.ndarray) -> np.ndarray:
        T = At.shape[0]
        A = torch.from_numpy(np.ascontiguousarray(At.reshape(T * 16, 16))).to(self.device)
        B = torch.from_numpy(np.ascontiguousarray(Bt.reshape(T * 8, 16))).to(self.device)
        D = torch.empty(T, 16, 8, dtype=torch.int32, device=self.device)
        self._launch("mmaemu_probe_epass_raw", T, [_ptr(A), _ptr(B), _ptr(D)])
        return D.cpu().numpy()

    def mma_bf16_raw(self, a16: np.ndarray, b16: np.ndarray, c: np.ndarray) -> np.ndarray:
        T = a16.shape[0]
        A = torch.from_numpy(np.ascontiguousarray(a16.reshape(T * 16, 16)).view(np.int16)).to(self.device)
        B = torch.from_numpy(np.ascontiguousarray(b16.reshape(T * 8, 16)).view(np.int16)).to(self.device)
        C = torch.from_numpy(np.ascontiguousarray(c, dtype=np.float32)).to(self.device)
        D = torch.empty(T, 16, 8, dtype=torch.int32, device=self.device)
        self._launch("mmaemu_probe_mma_bf16_raw", T, [_ptr(A), _ptr(B), _ptr(C), _ptr(D)])
        return D.cpu().numpy()

    def gemm(self, spec, a_u8: torch.Tensor, b_u8: torch.Tensor) -> torch.Tensor:
        from mmaemu.backends import ChainRequest
        from mmaemu.backends.cuda import kernels
        fn = kernels.chain_fn_for(spec, ChainRequest(self.device, "e4m3", "e4m3", "ada"))
        out = fn(a_u8.to(self.device), b_u8.to(self.device))
        return out.cpu()


# ------------------------------------------------------------------------------------------------------------ gates
def _forced(name: str) -> bool:
    return name in [x.strip() for x in os.environ.get("MMAEMU_TEST_FORCE_FAIL", "").split(",") if x.strip()]


def _device_info(device) -> dict:
    try:
        d = torch.device(device)
        if d.type == "cuda":
            return dict(device=torch.cuda.get_device_name(d), cc=list(torch.cuda.get_device_capability(d)))
    except Exception:  # noqa: BLE001  (informational only)
        pass
    return dict(device=str(device))


def _run_stages(name, spec, device, runner, stages) -> GateResult:
    t0 = time.perf_counter()
    details = dict(_device_info(device), gate_version=GATE_VERSION, kernel=getattr(spec, "name", None))
    if _forced(name):
        return GateResult(name, False, "forced: MMAEMU_TEST_FORCE_FAIL", details, time.perf_counter() - t0, False)
    runner = runner if runner is not None else DeviceRunner(device)
    for stage, fn in stages:
        t1 = time.perf_counter()
        try:
            ok, reason, det = fn(runner)
        except Exception as e:  # noqa: BLE001  (a gate fails closed: an unavailable probe / kernel rejects)
            ok, reason, det = False, f"{name}.error: {stage}: {type(e).__name__}: {str(e).splitlines()[0][:300]}", {}
        det["seconds"] = round(time.perf_counter() - t1, 4)
        details[stage] = det
        if not ok:
            return GateResult(name, False, reason, details, time.perf_counter() - t0, False)
    return GateResult(name, True, "", details, time.perf_counter() - t0, False)


def _stage_design_a_property(runner):
    v = design_a_vectors()
    T = v["a"].shape[0]
    a16, b16 = v["a"], v["b"]
    rp = runner.hmma_f16_k8(a16, b16, v["cp"])
    rn = runner.hmma_f16_k8(a16, b16, -v["cp"])
    ok, reason, det = eval_design_a(v, rp, rn)
    det["tiles"] = T
    return ok, reason, det


def _stage_xpass(runner):
    At, Bt = structural_tiles()
    T = At.shape[0]
    d = runner.mma_bf16_raw(enc1(At), enc1(Bt), np.zeros((T, 16, 8), np.float32))
    return eval_xpass(At, Bt, d)


def _stage_epass(runner):
    At, Bt = structural_tiles()
    return eval_window(At, Bt, runner.epass_raw(At, Bt))


def design_a(spec, device, *, runner=None) -> GateResult:
    """Gate of tc3 (see the module docstring)."""
    return _run_stages("design_a", spec, device, runner, (
        ("property", _stage_design_a_property),
        ("xpass", _stage_xpass),
        ("gemm", lambda r: eval_gemm("design_a", lambda a, b: r.gemm(spec, a, b))),
    ))


def window(spec, device, *, runner=None) -> GateResult:
    """Gate of v9 (see the module docstring)."""
    return _run_stages("window", spec, device, runner, (
        ("epass", _stage_epass),
        ("gemm", lambda r: eval_gemm("window", lambda a, b: r.gemm(spec, a, b))),
    ))


design_a.version = GATE_VERSION
window.version = GATE_VERSION
register_gate("design_a", design_a)
register_gate("window", window)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="gate data tools of the fast kernels")
    ap.add_argument("--write-gemm-data", action="store_true", help="recompute data/gate_gemm_v1.npz with the reference")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.write_gemm_data:
        print(json.dumps(write_gemm_data(device=args.device), indent=1))
