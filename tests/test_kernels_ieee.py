"""Conformance of the IEEE kernels v1 / v4 (preset ada) and h1 / h4 (preset hopper). GPU; run under the lock:

    $GPU timeout 600 python -m pytest -q tests/test_kernels_ieee.py

Port of the tcemu audit's verify-a3-integration-review-5/test_kernel.py: every kernel, loaded both from
the fatbin in build/fatbin ($MMAEMU_TEST_FATBIN_DIR overrides; build it with tools/build_fatbin.py --out
build/fatbin) and through NVRTC, is compared BIT FOR BIT (int32 views, +0 != -0, NaN payloads) with the CPU float64
reference (mmaemu.model.chain.chain_gemm) under the kernel's preset (KernelSpec.preset) on 8 data classes x
M {1, 65, 129, 200} x N {1, 63, 127, 257} x K {16, 48, 1920} (for h1 and h4, K = 16 and 48 end in a half block),
and NVRTC output must equal fatbin output. Then: NaN codes in rows / columns (canonical NaN exactly at the affected outputs,
through the raw v1 and h1 kernels and through the plan), and plan compositions (slow accumulation, split-K with the
preset's schemes: serial and pairwise for ada, contiguous for hopper; K % 16 != 0 through accumulate) against the
reference path.

The CPU references of the 384 conformance cases per preset take ~25 s each; precompute them outside the GPU lock with
    python tests/test_kernels_ieee.py --precompute
(cached as .npy under $MMAEMU_TEST_REF_CACHE or build/test_refs, keyed by the case and the reference sources).
"""
from __future__ import annotations

import hashlib
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
E4M3 = torch.float8_e4m3fn
U8 = torch.uint8
GENS = ["random", "allcodes", "tiny", "sat", "spread", "blockscale", "cancel", "zeros"]
SHAPES = [(M, N, K) for K in (16, 48, 1920) for M in (1, 65, 129, 200) for N in (1, 63, 127, 257)]
CASES = [(g, M, N, K) for g in GENS for (M, N, K) in SHAPES]
FINITE = torch.tensor([c for c in range(256) if c & 0x7F != 0x7F], dtype=U8)
SUBN = torch.tensor([c | s for c in range(1, 8) for s in (0, 0x80)], dtype=U8)


# ----------------------------------------------------------------------------------------------------------------
# data (CPU, deterministic; identical generators to the audit's test_kernel.py)

def codes(kind, shape, g):
    rows, K = shape
    if kind == "random":
        return (torch.randn(shape, generator=g) * 40).clamp(-448, 448).to(E4M3).view(U8)
    if kind == "allcodes":
        return FINITE[torch.randint(0, len(FINITE), shape, generator=g)]
    if kind == "tiny":
        c = SUBN[torch.randint(0, len(SUBN), shape, generator=g)]
        z = torch.rand(shape, generator=g) < 0.1
        return torch.where(z, torch.where(torch.rand(shape, generator=g) < 0.5, 0x00, 0x80).to(U8), c)
    if kind == "sat":
        return torch.where(torch.rand(shape, generator=g) < 0.5, 0x7E, 0xFE).to(U8)
    sign = (torch.rand(shape, generator=g) < 0.5).to(U8) << 7
    man = torch.randint(0, 8, shape, generator=g, dtype=U8)
    if kind == "spread":
        big = torch.rand(shape, generator=g) < 0.5
        e = torch.where(big, torch.randint(13, 16, shape, generator=g), torch.zeros(shape, dtype=torch.long))
        man = torch.where(big | (man != 0), man, torch.ones_like(man))
        c = sign | (e.to(U8) << 3) | man
        return torch.where((c & 0x7F) == 0x7F, c - 1, c)
    if kind == "blockscale":
        eb = torch.randint(0, 16, (rows, (K + 15) // 16), generator=g).repeat_interleave(16, 1)[:, :K]
        c = sign | (eb.to(U8) << 3) | man
        return torch.where((c & 0x7F) == 0x7F, c - 1, c)
    raise ValueError(kind)


def make(kind, M, N, K):
    """(a_u8 [M, K], b_u8 [N, K]) on CPU."""
    g = torch.Generator().manual_seed(zlib.crc32(f"{kind}-{M}-{N}-{K}".encode()))
    if kind == "cancel":
        a = codes("random", (M, K), g).view(E4M3).float()
        b = codes("random", (N, K), g).view(E4M3).float()
        a[:, 1::2] = -a[:, 0::2]
        bb = b[:, 0::2].to(E4M3).view(U8).clone()
        step = torch.randint(-1, 2, bb.shape, generator=g, dtype=torch.int16)
        step = torch.where(torch.rand(bb.shape, generator=g) < 0.25, step, torch.zeros_like(step))
        mag = ((bb & 0x7F).to(torch.int16) + step).clamp(0, 0x7E).to(U8)
        b[:, 1::2] = ((bb & 0x80) | mag).view(E4M3).float()
        return a.to(E4M3).view(U8), b.to(E4M3).view(U8)
    if kind == "zeros":
        a = codes("random", (M, K), g)
        b = codes("random", (N, K), g)
        a[torch.rand(M, generator=g) < 0.3] = 0
        b[torch.rand(N, generator=g) < 0.3] = 0
        a[torch.rand((M, K), generator=g) < 0.2] = 0x80
        b[torch.rand((N, K), generator=g) < 0.2] = 0x80
        return a, b
    return codes(kind, (M, K), g), codes(kind, (N, K), g)


# ----------------------------------------------------------------------------------------------------------------
# CPU reference (cached)

def _ref_identity() -> str:
    import mmaemu.model.chain as chain

    h = hashlib.sha256(Path(chain.__file__).read_bytes())
    old = Path(chain.__file__).resolve().parents[1] / "reference.py"  # the pre-v0.1 model, if chain wraps it
    if old.is_file():
        h.update(old.read_bytes())
    return h.hexdigest()[:16]


def ref_dir() -> Path:
    return Path(os.environ.get("MMAEMU_TEST_REF_CACHE", ROOT / "build" / "test_refs"))


def reference(a_u8, b_u8, preset="ada"):
    from mmaemu.model.chain import chain_gemm

    return chain_gemm(a_u8.cpu(), b_u8.cpu(), fmt_a="e4m3", fmt_b="e4m3", preset=preset).float()


def cached_reference(case, preset="ada") -> torch.Tensor:
    gen, M, N, K = case
    p = ref_dir() / _ref_identity() / f"{gen}-{M}-{N}-{K}{'' if preset == 'ada' else '-' + preset}.npy"
    if p.is_file():
        return torch.from_numpy(np.load(p, allow_pickle=False)).view(torch.float32)
    a, b = make(gen, M, N, K)
    r = reference(a, b, preset)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.tmp.npy")
        np.save(tmp, r.view(torch.int32).numpy(), allow_pickle=False)
        os.replace(tmp, p)
    except OSError:
        pass
    return r


# ----------------------------------------------------------------------------------------------------------------
# fixtures

def _fatbin_dir() -> Path:
    return Path(os.environ.get("MMAEMU_TEST_FATBIN_DIR", ROOT / "build" / "fatbin")).resolve()


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
pytestmark = [pytest.mark.gpu, requires_cuda]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Module-wide environment: fatbins from build/, a private cache dir, a fresh loader and registry."""
    if not (_fatbin_dir() / "manifest.json").is_file():
        pytest.skip(f"no fatbin manifest in {_fatbin_dir()}: run tools/build_fatbin.py --out build/fatbin")
    mp = pytest.MonkeyPatch()
    mp.setenv("MMAEMU_FATBIN_DIR", str(_fatbin_dir()))
    mp.setenv("MMAEMU_CACHE_DIR", str(tmp_path_factory.mktemp("mmaemu_cache")))
    mp.delenv("MMAEMU_TEST_FORCE_FAIL", raising=False)
    from mmaemu.backends import registry
    from mmaemu.backends.cuda import modules

    modules.clear()
    registry.clear()
    yield torch.device("cuda", torch.cuda.current_device())
    modules.clear()
    registry.clear()
    mp.undo()


def _specs():
    from mmaemu.backends.cuda.kernels import SPECS

    return SPECS


KERNELS = ["v1", "v4", "h1", "h4"]
PRESETS = ("ada", "hopper")


def _spec(kernel):
    return {s.name: s for s in _specs()}[kernel]


def _chain(spec, dev):
    from mmaemu.backends import ChainRequest
    from mmaemu.backends.cuda.kernels import chain_fn_for

    return chain_fn_for(spec, ChainRequest(device=dev, fmt_a="e4m3", fmt_b="e4m3", preset=spec.preset))


def _bits(t):
    return t.detach().cpu().contiguous().view(torch.int32)


@pytest.fixture(scope="module")
def outputs(env):
    """{(kernel, source): {case: int32 bits}} for every conformance case, fatbin first then NVRTC."""
    from mmaemu.backends.cuda import modules

    dev = env
    res: dict = {}
    for source in ("fatbin", "nvrtc"):
        modules.clear()
        os.environ["MMAEMU_TEST_FORCE_FAIL"] = "nvrtc" if source == "fatbin" else "fatbin"
        try:
            for spec in _specs():
                fn = _chain(spec, dev)
                got = modules.module_info(spec.module, dev).source
                assert got == source, f"{spec.name}: wanted the {source} module, loaded {got}"
                d = res.setdefault((spec.name, source), {})
                for case in CASES:
                    a, b = make(*case)
                    d[case] = _bits(fn(a.to(dev), b.to(dev)))
                torch.cuda.synchronize(dev)
        finally:
            os.environ.pop("MMAEMU_TEST_FORCE_FAIL", None)
    modules.clear()
    return res


@pytest.fixture(scope="module")
def refs():
    return {(preset, case): cached_reference(case, preset).view(torch.int32) for preset in PRESETS for case in CASES}


# ----------------------------------------------------------------------------------------------------------------
# conformance and NVRTC == fatbin

@pytest.mark.parametrize("source", ["fatbin", "nvrtc"])
@pytest.mark.parametrize("kernel", KERNELS)
def test_conformance(outputs, refs, kernel, source):
    """All 384 cases (8 data classes x 48 shapes), bit for bit against the CPU reference of the kernel's preset."""
    from mmaemu.testing import assert_bitwise_equal

    preset = _spec(kernel).preset
    got = outputs[(kernel, source)]
    bad, total = [], 0
    for case in CASES:
        want = refs[(preset, case)]
        total += want.numel()
        n = int((got[case] != want).sum())
        if n:
            bad.append((case, n))
    print(f"{kernel}/{source} ({preset}): {len(CASES)} cases, {total} outputs, {sum(n for _, n in bad)} bit mismatches")
    if bad:
        case = bad[0][0]
        assert_bitwise_equal(got[case].view(torch.float32), refs[(preset, case)].view(torch.float32),
                             msg=f"{kernel}/{source} {case}; all failing cases: {bad[:20]}")


def test_presets_differ_on_the_conformance_data(refs):
    """The conformance data tells the presets apart, so h1 cannot pass by computing ada's chain (or v1 hopper's):
    for K = 48 and 1920, the ada and hopper references differ at thousands of outputs in six of the eight data
    classes. They agree on "sat" (every product the same size) and "tiny" (exact either way), and at K = 16 (one
    block in both presets)."""
    diff = {}
    for case in CASES:
        k = (case[0], case[3])
        diff[k] = diff.get(k, 0) + int((refs[("ada", case)] != refs[("hopper", case)]).sum())
    for gen in GENS:
        assert diff[(gen, 16)] == 0
        if gen in ("sat", "tiny"):
            assert diff[(gen, 48)] == diff[(gen, 1920)] == 0
        else:
            assert diff[(gen, 48)] > 1000 and diff[(gen, 1920)] > 1000, (gen, diff[(gen, 48)], diff[(gen, 1920)])


@pytest.mark.parametrize("kernel", KERNELS)
def test_nvrtc_equals_fatbin(outputs, kernel):
    from mmaemu.testing import assert_bitwise_equal

    f, n = outputs[(kernel, "fatbin")], outputs[(kernel, "nvrtc")]
    for case in CASES:
        assert_bitwise_equal(n[case].view(torch.float32), f[case].view(torch.float32), msg=f"{kernel} {case}")


# ----------------------------------------------------------------------------------------------------------------
# NaN codes

def _nan_data():
    g = torch.Generator().manual_seed(157)
    M, N, K = 65, 63, 256
    a = codes("random", (M, K), g)
    b = codes("random", (N, K), g)
    a[3, 17] = 0x7F
    a[40, 200] = 0xFF
    b[5, 0] = 0xFF
    b[60, 255] = 0x7F
    b[:, 100] = 0x00          # column k = 100 is zero in every row of B ...
    a[10, 100] = 0x7F         # ... so row 10 of A only has NaN * 0 products from its NaN code
    rows = torch.zeros(M, dtype=torch.bool)
    rows[[3, 40, 10]] = True
    cols = torch.zeros(N, dtype=torch.bool)
    cols[[5, 60]] = True
    affected = rows[:, None] | cols[None, :]
    return a, b, affected


@pytest.mark.parametrize("kernel", ["v1", "h1"])
def test_nan_raw_kernel(env, kernel):
    """The v1 and h1 kernels themselves decode NaN codes to NaN (the prototype's v1 decoded them as 0): canonical NaN
    exactly at the affected outputs."""
    spec = _spec(kernel)
    a, b, affected = _nan_data()
    out = _chain(spec, env)(a.to(env), b.to(env)).cpu()
    bits = out.view(torch.int32)
    assert bool((bits[affected] == 0x7FFFFFFF).all()), "affected outputs must be canonical NaN 0x7FFFFFFF"
    assert not bool(torch.isnan(out[~affected]).any())
    clean = torch.where((a & 0x7F) == 0x7F, 0, a), torch.where((b & 0x7F) == 0x7F, 0, b)
    ref = reference(*clean, spec.preset).view(torch.int32)
    assert int((bits[~affected] != ref[~affected]).sum()) == 0


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("fast", [True, False])
def test_nan_through_plan(env, kernel, fast):
    """execute_plan (sanitise, chain, patch) with the kernel equals the reference path bit for bit; NaN canonical."""
    from mmaemu.backends.reference import REFERENCE
    from mmaemu.backends import ChainRequest
    from mmaemu.model.plan import execute_plan, make_plan
    from mmaemu.testing import assert_bitwise_equal

    spec = _spec(kernel)
    a, b, affected = _nan_data()
    plan = make_plan(a.shape[1], fast=fast, preset=spec.preset)
    got = execute_plan(plan, a.to(env), b.to(env), fmt_a="e4m3", fmt_b="e4m3", chain_fn=_chain(spec, env)).cpu()
    ref_fn = REFERENCE.chain_fn(ChainRequest(device=torch.device("cpu"), fmt_a="e4m3", fmt_b="e4m3",
                                             preset=spec.preset))
    want = execute_plan(plan, a, b, fmt_a="e4m3", fmt_b="e4m3", chain_fn=ref_fn)
    assert_bitwise_equal(got, want, msg=f"{kernel} fast={fast}")
    assert bool((got.view(torch.int32)[affected] == 0x7FFFFFFF).all())
    assert not bool(torch.isnan(got[~affected]).any())


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("fill", ["0xFF", "0x7F", "0x80", "uniform256"])
def test_any_byte_value_never_faults(env, kernel, fill):
    """ABI v1 (spec.py): the kernel must not fault for any byte value, NaN codes included (the wrapper never feeds
    them, but a fault would kill the CUDA context). v1 and h1 also give canonical NaN wherever a NaN code is
    involved."""
    spec = _spec(kernel)
    M, N, K = 200, 257, 1920
    if fill == "uniform256":
        g = torch.Generator().manual_seed(256)
        a = torch.randint(0, 256, (M, K), generator=g, dtype=U8)
        b = torch.randint(0, 256, (N, K), generator=g, dtype=U8)
    else:
        a = torch.full((M, K), int(fill, 16), dtype=U8)
        b = torch.full((N, K), int(fill, 16), dtype=U8)
    out = _chain(spec, env)(a.to(env), b.to(env))
    torch.cuda.synchronize(env)  # an illegal address or misaligned access would raise here
    assert out.shape == (M, N)
    if kernel in ("v1", "h1") and fill in ("0xFF", "0x7F", "uniform256"):
        assert bool((out.cpu().view(torch.int32) == 0x7FFFFFFF).all())  # every output involves a NaN code
    if fill == "0x80":  # -0 * -0: an exactly-zero chain is +0 (M1)
        assert int(out.cpu().view(torch.int32).abs().sum()) == 0


# ----------------------------------------------------------------------------------------------------------------
# plan compositions

def _plan_case(M, N, K, seed):
    g = torch.Generator().manual_seed(seed)
    return codes("random", (M, K), g), codes("allcodes", (N, K), g)


def _compare_plan(env, kernel, plan, a, b):
    from mmaemu.backends import ChainRequest
    from mmaemu.backends.reference import REFERENCE
    from mmaemu.model.plan import execute_plan
    from mmaemu.testing import assert_bitwise_equal

    spec = _spec(kernel)
    got = execute_plan(plan, a.to(env), b.to(env), fmt_a="e4m3", fmt_b="e4m3", chain_fn=_chain(spec, env)).cpu()
    ref_fn = REFERENCE.chain_fn(ChainRequest(device=torch.device("cpu"), fmt_a="e4m3", fmt_b="e4m3",
                                             preset=spec.preset))
    want = execute_plan(plan, a, b, fmt_a="e4m3", fmt_b="e4m3", chain_fn=ref_fn)
    assert_bitwise_equal(got, want, msg=f"{kernel} plan K={plan.K} fast={plan.fast} S={plan.S} {plan.scheme}")
    return got


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("K", [1920, 8192])
def test_slow_accumulation(env, kernel, K):
    from mmaemu.model.plan import make_plan

    a, b = _plan_case(24, 40, K, K)
    _compare_plan(env, kernel, make_plan(K, fast=False, preset=_spec(kernel).preset), a, b)


SPLITS = [(S, scheme, fast) for S in (2, 3, 8, 12, 29) for scheme in ("serial", "pairwise") for fast in (True, False)]
SPLITS_H = [(S, K, fast) for S in (2, 3, 4, 5, 6, 8) for K in (128 * 4 * S, 128 * 4 * S + 1000) for fast in (True, False)]
PROFILE = {"ada": "ada", "hopper": "hopper"}


@pytest.mark.parametrize("kernel", ["v1", "v4"])
@pytest.mark.parametrize("S,scheme,fast", SPLITS, ids=[f"S{s}-{c}-{'fast' if f else 'slow'}" for s, c, f in SPLITS])
def test_split_k(env, kernel, S, scheme, fast):
    from mmaemu.model.plan import make_plan

    K = max(1000, 64 * S * 2 + 40)  # ragged last tile; every split owns >= 2 tiles
    a, b = _plan_case(33, 47, K, 1000 + S)
    _compare_plan(env, kernel, make_plan(K, fast=fast, S=S, scheme=scheme), a, b)


@pytest.mark.parametrize("kernel", ["h1", "h4"])
@pytest.mark.parametrize("S,K,fast", SPLITS_H, ids=[f"S{s}-K{k}-{'fast' if f else 'slow'}" for s, k, f in SPLITS_H])
def test_split_k_contiguous_hopper(env, kernel, S, K, fast):
    """h1 and h4 under the H100's contiguous split-K (128-wide k-tiles; K a multiple of 128, and a ragged K)."""
    from mmaemu.model.plan import make_plan

    a, b = _plan_case(33, 47, K, 2000 + S)
    _compare_plan(env, kernel, make_plan(K, fast=fast, S=S, scheme="contiguous", preset="hopper"), a, b)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("K", [1, 15, 100, 1000])
@pytest.mark.parametrize("fast", [True, False])
def test_k_not_multiple_of_16_via_accumulate(env, kernel, K, fast):
    import mmaemu.api as api
    from mmaemu.testing import assert_bitwise_equal

    prof = PROFILE[_spec(kernel).preset]
    a, b = _plan_case(17, 29, K, 7 + K)
    a8, b8 = a.view(E4M3), b.view(E4M3)
    got = api.accumulate(a8.to(env), b8.to(env), use_fast_accum=fast, splitk=1, profile=prof,
                         backend=f"cuda:{kernel}").cpu()
    want = api.accumulate(a8, b8, use_fast_accum=fast, splitk=1, profile=prof, backend="reference")
    assert_bitwise_equal(got, want, msg=f"{kernel} K={K} fast={fast}")


# ----------------------------------------------------------------------------------------------------------------

if __name__ == "__main__" and "--precompute" in sys.argv:
    import concurrent.futures as cf
    import time

    sys.path.insert(0, str(ROOT))
    t0 = time.monotonic()
    torch.set_num_threads(2)
    with cf.ProcessPoolExecutor(max_workers=int(os.environ.get("JOBS", "8"))) as ex:
        work = sorted(((c, p) for p in PRESETS for c in CASES), key=lambda t: -t[0][1] * t[0][2] * t[0][3])
        list(ex.map(cached_reference, [c for c, _ in work], [p for _, p in work]))
    print(f"{len(CASES)} x {len(PRESETS)} references in {ref_dir() / _ref_identity()} ({time.monotonic() - t0:.1f} s)")
