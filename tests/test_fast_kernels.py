"""Bit-exactness of the gated fast kernels v9 and tc3 against the mmaemu reference.

Every comparison is bitwise (mmaemu.testing.assert_bitwise_equal on int32 views: +0 != -0).  The reference is the
float64 chain model run on the GPU (mmaemu.model.chain.chain_gemm on the inputs' device; exact and device
independent by construction), or, for the 77 cases of the tcemu audit's
vperf-p3-tensorcore-exact-1/scripts/verify2.py, the reference results that script cached (tcemu.reference.gemm on the
GPU, data/ref2/*.pt; each cached matrix is spot-checked against mmaemu's reference here).

Sets (gpu marker):
  - the IEEE kernels' conformance matrix (port of the audit's verify-a3-integration-review-5/test_kernel.py):
    8 data classes x M {1, 65, 129, 200} x N {1, 63, 127, 257} x K {16, 48, 1920};
  - the 77 verify2 cases (core / extra / big / zero / big40; 37.7 M outputs; audit data under $MMAEMU_AUDIT_DIR,
    skipped if absent);
  - p2 check (data/cases.pt, 22 cases), fuzz (seed 202, 60 random cases) and cases8k (6 cases at K = 8192) of
    the audit's p2-int-mixed (v9; tc3 is run on them too; the 22 p2 cases need $MMAEMU_AUDIT_DIR);
  - K = 0 (wrapper and raw kernel launch) and all-0xFF (NaN-code) inputs: no fault, context still usable.
truth marker (MMAEMU_TRUTH_DIR): the 21 Evo 2 1B projections of truth evo2.evo2_1b_w2k: full fp32 accumulator
sha256 == recorded acc_full for 21/21, and the bf16 outputs through scaled_mm(profile "ada-rtx4060-te") equal the
recorded y_native rows (and the full-output sha256 where only a hash is stored), for v9 and for tc3.

Capability handling: a kernel without a binary for the device is skipped; on a capability outside
validated_on the kernel's gates run first and, if they fail, the test asserts the kernel is not auto-selected and
xfails with the gate's reason.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch

from mmaemu.backends.cuda.fast import specs

pytestmark = pytest.mark.gpu
E4M3 = torch.float8_e4m3fn
KERNELS = {"v9": specs.V9, "tc3": specs.TC3}
# Work directory of the (unpublished) tcemu audit, on the maintainer host only; its sets are skipped without it.
AUDIT = Path(os.environ["MMAEMU_AUDIT_DIR"]) if os.environ.get("MMAEMU_AUDIT_DIR") else None
VERIFY2_REF = AUDIT / "vperf-p3-tensorcore-exact-1" / "data" / "ref2" if AUDIT else None
P2_CASES = AUDIT / "p2-int-mixed" / "data" / "cases.pt" if AUDIT else None


@pytest.fixture(scope="module")
def dev():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    return torch.device("cuda", torch.cuda.current_device())


_READY: dict = {}


def kernel(name, dev):
    """ChainFn of kernel `name` on `dev` (standard-ABI wrapper of kernels.py), after the capability policy above."""
    if name in _READY:
        return _READY[name]
    from mmaemu.backends import ChainRequest
    from mmaemu.backends.cuda import kernels
    sp = KERNELS[name]
    cc = torch.cuda.get_device_capability(dev)
    arch = cc[0] * 10 + cc[1]
    if arch not in sp.sass_archs:
        pytest.skip(f"no {name} binary for sm_{arch} (SASS {sp.sass_archs})")
    if tuple(cc) not in sp.validated_on:
        from mmaemu.backends import select
        from mmaemu.backends.cuda import selftest
        res = selftest.run_gates(sp, dev)
        bad = [r for r in res if not r.passed]
        if bad:
            chosen = select(ChainRequest(dev, "e4m3", "e4m3", "ada")).name
            assert chosen != f"cuda:{name}"
            pytest.xfail(f"{name} gate {bad[0].name} failed on sm_{arch}: {bad[0].reason}")
    _READY[name] = kernels.chain_fn_for(sp, ChainRequest(dev, "e4m3", "e4m3", "ada"))
    return _READY[name]


def reference(a_u8, b_u8, dev):
    from mmaemu.model.chain import chain_gemm
    return chain_gemm(a_u8.to(dev), b_u8.to(dev), fmt_a="e4m3", fmt_b="e4m3").cpu()


def check(name, dev, a_u8, b_u8, ref, msg=""):
    from mmaemu.testing import assert_bitwise_equal
    out = kernel(name, dev)(a_u8.to(dev), b_u8.to(dev))
    torch.cuda.synchronize(dev)
    assert_bitwise_equal(out.cpu(), ref, msg=f"{name} {msg}")


# ------------------------------------------------------------- conformance matrix (port of a3-5 test_kernel.py)
FINITE = torch.tensor([c for c in range(256) if c & 0x7F != 0x7F], dtype=torch.uint8)
SUBN = torch.tensor([c | s for c in range(1, 8) for s in (0, 0x80)], dtype=torch.uint8)
GENS = ["random", "allcodes", "tiny", "sat", "spread", "blockscale", "cancel", "zeros"]
SHAPES = [(M, N, K) for K in (16, 48, 1920) for M in (1, 65, 129, 200) for N in (1, 63, 127, 257)]


def _codes(kind, shape, g):
    rows, K = shape
    if kind == "random":
        return (torch.randn(shape, generator=g) * 40).clamp(-448, 448).to(E4M3).view(torch.uint8)
    if kind == "allcodes":
        return FINITE[torch.randint(0, len(FINITE), shape, generator=g)]
    if kind == "tiny":
        c = SUBN[torch.randint(0, len(SUBN), shape, generator=g)]
        z = torch.rand(shape, generator=g) < 0.1
        return torch.where(z, torch.where(torch.rand(shape, generator=g) < 0.5, 0x00, 0x80).to(torch.uint8), c)
    if kind == "sat":
        return torch.where(torch.rand(shape, generator=g) < 0.5, 0x7E, 0xFE).to(torch.uint8)
    sign = (torch.rand(shape, generator=g) < 0.5).to(torch.uint8) << 7
    man = torch.randint(0, 8, shape, generator=g, dtype=torch.uint8)
    if kind == "spread":
        big = torch.rand(shape, generator=g) < 0.5
        e = torch.where(big, torch.randint(13, 16, shape, generator=g), torch.zeros(shape, dtype=torch.long))
        man = torch.where(big | (man != 0), man, torch.ones_like(man))
        c = sign | (e.to(torch.uint8) << 3) | man
        return torch.where((c & 0x7F) == 0x7F, c - 1, c)
    if kind == "blockscale":
        eb = torch.randint(0, 16, (rows, (K + 15) // 16), generator=g).repeat_interleave(16, 1)[:, :K]
        c = sign | (eb.to(torch.uint8) << 3) | man
        return torch.where((c & 0x7F) == 0x7F, c - 1, c)
    raise ValueError(kind)


def make_conformance(kind, M, N, K):
    g = torch.Generator().manual_seed(zlib.crc32(f"{kind}-{M}-{N}-{K}".encode()))
    if kind == "cancel":
        a = _codes("random", (M, K), g).view(E4M3).float()
        b = _codes("random", (N, K), g).view(E4M3).float()
        a[:, 1::2] = -a[:, 0::2]
        bb = b[:, 0::2].to(E4M3).view(torch.uint8).clone()
        step = torch.randint(-1, 2, bb.shape, generator=g, dtype=torch.int16)
        step = torch.where(torch.rand(bb.shape, generator=g) < 0.25, step, torch.zeros_like(step))
        mag = ((bb & 0x7F).to(torch.int16) + step).clamp(0, 0x7E).to(torch.uint8)
        b[:, 1::2] = ((bb & 0x80) | mag).view(E4M3).float()
        return a.to(E4M3).view(torch.uint8), b.to(E4M3).view(torch.uint8)
    if kind == "zeros":
        a = _codes("random", (M, K), g)
        b = _codes("random", (N, K), g)
        a[torch.rand(M, generator=g) < 0.3] = 0
        b[torch.rand(N, generator=g) < 0.3] = 0
        a[torch.rand((M, K), generator=g) < 0.2] = 0x80
        b[torch.rand((N, K), generator=g) < 0.2] = 0x80
        return a, b
    return _codes(kind, (M, K), g), _codes(kind, (N, K), g)


@pytest.mark.parametrize("gen", GENS)
@pytest.mark.parametrize("name", tuple(KERNELS))
def test_conformance_matrix(dev, gen, name):
    kernel(name, dev)
    n = 0
    for M, N, K in SHAPES:
        a, b = make_conformance(gen, M, N, K)
        ref = reference(a, b, dev)
        check(name, dev, a, b, ref, f"{gen} {M}x{N}x{K}")
        n += M * N
    print(f"conformance {name} {gen}: {len(SHAPES)} shapes, {n} outputs, 0 mismatches")


# ------------------------------------------------------------------------------- the 77 cases of verify2.py (p3)
def _verify2_files():
    return sorted(VERIFY2_REF.glob("*.pt")) if VERIFY2_REF is not None and VERIFY2_REF.is_dir() else []


@pytest.mark.skipif(not _verify2_files(),
                    reason="audit data absent (set MMAEMU_AUDIT_DIR; needs vperf-p3-tensorcore-exact-1/data/ref2)")
@pytest.mark.parametrize("path", _verify2_files(), ids=lambda p: p.stem)
@pytest.mark.parametrize("name", tuple(KERNELS))
def test_verify2_77_cases(dev, path, name):
    kernel(name, dev)
    assert len(_verify2_files()) == 77
    d = torch.load(path, weights_only=True)
    A, B, ref = d["A"], d["B"], d["ref"]
    assert A.dtype == torch.uint8 and ref.dtype == torch.float32 and d["ref_fp32_ok"]
    # tie the cached tcemu reference to mmaemu's reference: 2 rows x 32 columns spread over the matrix
    M, N = ref.shape
    rows = torch.tensor(sorted({0, M - 1, M // 2})[:2])
    cols = torch.linspace(0, N - 1, min(N, 32)).round().long().unique()
    from mmaemu.testing import assert_bitwise_equal
    assert_bitwise_equal(reference(A[rows], B[cols], dev), ref[rows][:, cols].contiguous(), msg=f"{path.stem} cache")
    check(name, dev, A, B, ref, path.stem)


# ---------------------------------------------------------------- p2 sets: check (cases.pt), fuzz, cases8k (v9)
@pytest.mark.skipif(P2_CASES is None or not P2_CASES.exists(),
                    reason="audit data absent (set MMAEMU_AUDIT_DIR; needs p2-int-mixed/data/cases.pt)")
@pytest.mark.parametrize("name", tuple(KERNELS))
def test_p2_check_cases(dev, name):
    kernel(name, dev)
    cases = torch.load(P2_CASES, weights_only=True)
    assert len(cases) == 22
    tot = 0
    for c in cases:
        check(name, dev, c["a"], c["b"], c["ref"], c["name"])
        tot += c["ref"].numel()
    assert tot == 231254


F8 = torch.float8_e4m3fn


def _td_gauss40(shape, g):
    return (torch.randn(shape, generator=g) * 40).clamp(-448, 448).to(F8)


def _td_adversarial(shape, g):
    x = torch.randint(0, 254, shape, generator=g, dtype=torch.int32)
    x = x + (x >= 0x7F).int()
    return x.to(torch.uint8).view(F8)


def _td_tiny(shape, g):
    e = torch.randint(0, 3, shape, generator=g)
    m = torch.randint(0, 8, shape, generator=g)
    s = torch.randint(0, 2, shape, generator=g)
    x = (s << 7) | (e << 3) | m
    x = torch.where(torch.rand(shape, generator=g) < 0.2, torch.zeros_like(x), x)
    return x.to(torch.uint8).view(F8)


def _td_logu(shape, g):
    mag = torch.exp2(torch.rand(shape, generator=g) * 17.8 - 9.0)
    s = torch.where(torch.rand(shape, generator=g) < 0.5, -1.0, 1.0)
    return (mag * s).clamp(-448, 448).to(F8)


def _td_decay(shape, g):
    K = shape[-1]
    sc = torch.exp2(-torch.arange(K) / 48.0)
    return (torch.randn(shape, generator=g) * 200 * sc).clamp(-448, 448).to(F8)


def _fz_sparse_blocks(shape, g):
    x = _td_gauss40(shape, g).float()
    K = shape[-1]
    keep = torch.rand(shape[0], (K + 15) // 16, generator=g) < 0.4
    return (x * keep.repeat_interleave(16, 1)[:, :K]).to(F8)


def _fz_one_binade(shape, g):
    m = torch.randint(0, 8, shape, generator=g)
    s = torch.randint(0, 2, shape, generator=g)
    return ((s << 7) | (13 << 3) | m).to(torch.uint8).view(F8)


def _fz_max_signs(shape, g):
    return torch.where(torch.rand(shape, generator=g) < 0.5, -448.0, 448.0).to(F8)


FUZZ_FAMILIES = [_td_gauss40, _td_adversarial, _td_tiny, _td_logu, _td_decay, _fz_sparse_blocks, _fz_one_binade,
                 _fz_max_signs]


@pytest.mark.parametrize("name", tuple(KERNELS))
def test_p2_fuzz(dev, name):
    """port of p2-int-mixed/fuzz.py with the seed and case count of the verify-p2-int-mixed-3 repro (202, 60)."""
    kernel(name, dev)
    g = torch.Generator().manual_seed(202)
    tot = 0
    for ci in range(60):
        M = int(torch.randint(1, 301, (1,), generator=g))
        N = int(torch.randint(1, 301, (1,), generator=g))
        K = 16 * int(torch.randint(1, 257, (1,), generator=g))
        fa = FUZZ_FAMILIES[int(torch.randint(0, len(FUZZ_FAMILIES), (1,), generator=g))]
        fb = FUZZ_FAMILIES[int(torch.randint(0, len(FUZZ_FAMILIES), (1,), generator=g))]
        a, b = fa((M, K), g).view(torch.uint8), fb((N, K), g).view(torch.uint8)
        ref = reference(a, b, dev)
        check(name, dev, a, b, ref, f"fuzz{ci} {M}x{N}x{K} {fa.__name__}/{fb.__name__}")
        tot += M * N
    assert tot == 1425596      # the audit's fuzz_202 output count: the same 60 cases


@pytest.mark.parametrize("name", tuple(KERNELS))
def test_p2_cases8k(dev, name):
    """port of p2-int-mixed/cases8k.py: Evo 2's largest K (8192) and worst-case accumulator growth."""
    kernel(name, dev)
    g = torch.Generator().manual_seed(8192)
    K = 8192
    cases = [
        ("adversarial_64x256x8192", _td_adversarial((64, K), g), _td_adversarial((256, K), g)),
        ("all448_32x64x8192", torch.full((32, K), 448.0).to(F8), torch.full((64, K), 448.0).to(F8)),
        ("all448_signs_32x64x8192", torch.full((32, K), 448.0).to(F8),
         torch.where(torch.rand(64, K, generator=g) < 0.5, -448.0, 448.0).to(F8)),
        ("decay_64x128x8192", _td_decay((64, K), g), _td_decay((128, K), g)),
        ("logu_64x128x8192", _td_logu((64, K), g), _td_logu((128, K), g)),
        ("gauss40_x_tiny_48x96x8192", _td_gauss40((48, K), g), _td_tiny((96, K), g)),
    ]
    for cname, a, b in cases:
        a, b = a.view(torch.uint8), b.view(torch.uint8)
        ref = reference(a, b, dev)
        check(name, dev, a, b, ref, cname)


# ---------------------------------------------------------------------------------------- K = 0, NaN-code bytes
def _raw_launch(sp, dev, A, B, M, N, K):
    from mmaemu.backends.cuda import modules
    C = torch.full((M, N), float("nan"), dtype=torch.float32, device=dev)
    grid = ((N + sp.tile_n - 1) // sp.tile_n, (M + sp.tile_m - 1) // sp.tile_m, 1)
    args = [ctypes.c_void_p(A.data_ptr()), ctypes.c_void_p(B.data_ptr()), ctypes.c_void_p(C.data_ptr()),
            ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)]
    modules.launch(sp.module, sp.entry, grid, (sp.threads, 1, 1), sp.smem_bytes, args, dev)
    torch.cuda.synchronize(dev)
    return C


@pytest.mark.parametrize("name", list(KERNELS))
def test_k0_is_safe(dev, name):
    fn = kernel(name, dev)
    sp = KERNELS[name]
    out = fn(torch.empty(37, 0, dtype=torch.uint8, device=dev), torch.empty(19, 0, dtype=torch.uint8, device=dev))
    assert out.shape == (37, 19) and bool((out.view(torch.int32) == 0).all())          # +0 everywhere
    # the kernel itself (bypassing the wrapper's K == 0 path): K = 0 writes +0 and never reads A / B
    empty = torch.empty(0, dtype=torch.uint8, device=dev)
    C = _raw_launch(sp, dev, empty, empty, 37, 19, 0)
    assert bool((C.view(torch.int32) == 0).all())
    C = _raw_launch(sp, dev, empty, empty, 200, 300, 0)
    assert bool((C.view(torch.int32) == 0).all())


@pytest.mark.parametrize("name", list(KERNELS))
def test_all_0xff_inputs_do_not_fault(dev, name):
    """NaN codes never reach a kernel through mmaemu (execute_plan sanitises them), but the kernels must not fault on
    any byte value: launch them directly on all-0xFF (and all-0x7F) operands, then run a normal GEMM."""
    sp = KERNELS[name]
    kernel(name, dev)
    for byte in (0xFF, 0x7F):
        for M, N, K in ((130, 136, 1936), (1, 1, 16), (64, 64, 8192)):
            A = torch.full((M, K), byte, dtype=torch.uint8, device=dev)
            B = torch.full((N, K), byte, dtype=torch.uint8, device=dev)
            _raw_launch(sp, dev, A, B, M, N, K)                                  # no CUDA error
    a, b = make_conformance("allcodes", 65, 63, 48)
    check(name, dev, a, b, reference(a, b, dev), "after 0xFF launches")


# ------------------------------------------------------------------------------- truth: Evo 2 1B, 21 projections
def _truth():
    root = os.environ.get("MMAEMU_TRUTH_DIR")
    if not root:
        pytest.skip("MMAEMU_TRUTH_DIR not set")
    sys.dont_write_bytecode = True               # never write into the truth tree
    if root not in sys.path:
        sys.path.insert(0, root)
    from tcemu_ada_truth import loader
    return loader


def _scaled_mm_bf16(name, dev, a, b, acc, sa_bits, sb_bits):
    """bf16 output through mmaemu.scaled_mm(profile "ada-rtx4060-te", backend "cuda:<name>"); if scaled_mm is a stub,
    through the shared epilogue (mmaemu.model.epilogue.apply) on the kernel's accumulator."""
    sa = torch.tensor(np.uint32(sa_bits).view(np.float32).item(), dtype=torch.float32, device=dev)
    sb = torch.tensor(np.uint32(sb_bits).view(np.float32).item(), dtype=torch.float32, device=dev)
    import mmaemu
    fn = getattr(mmaemu, "scaled_mm", None)       # absent only in a stripped tree
    if fn is not None:
        try:
            y = fn(a.view(E4M3), b.view(E4M3).t(), sa, sb, out_dtype=torch.bfloat16, use_fast_accum=True,
                   profile="ada-rtx4060-te", backend=f"cuda:{name}")
            return y, "scaled_mm"
        except NotImplementedError as e:          # mmaemu stub (module absent); any other error propagates
            if not str(e).startswith("mmaemu."):
                raise
    from mmaemu.model.epilogue import apply
    return apply(acc, sa, sb, out_dtype=torch.bfloat16), "epilogue.apply"


@pytest.mark.truth
@pytest.mark.parametrize("name", list(KERNELS))
def test_truth_evo2_1b_w2k(dev, name):
    loader = _truth()
    fn = kernel(name, dev)
    rec = loader.load("evo2.evo2_1b_w2k")
    assert len(rec.cases) == 21
    ok_acc = ok_rows = ok_full = n_full = 0
    paths = set()
    for case in rec.cases:
        ins = case.inputs()
        a = torch.from_numpy(ins["a"]).to(dev)
        b = torch.from_numpy(ins["b"]).to(dev)
        acc = fn(a, b)
        torch.cuda.synchronize(dev)
        outs = {o.name: o for o in case.outputs}
        bits = acc.cpu().view(torch.int32).numpy().view(np.uint32)
        if hashlib.sha256(np.ascontiguousarray(bits).tobytes()).hexdigest() == outs["acc_full"].sha256:
            ok_acc += 1
        y, path = _scaled_mm_bf16(name, dev, a, b, acc, outs["y_native"].params["scale_a"],
                                  outs["y_native"].params["scale_b"])
        paths.add(path)
        yb = y.cpu().contiguous().view(torch.int16).numpy().view(np.uint16)
        yn = outs["y_native"]
        r0, r1 = yn.rows if yn.rows else (0, yb.shape[0])
        if np.array_equal(yb[r0:r1], yn.bits()):
            ok_rows += 1
        if "y_native_full" in outs:
            n_full += 1
            ok_full += hashlib.sha256(np.ascontiguousarray(yb).tobytes()).hexdigest() == outs["y_native_full"].sha256
        case.drop_inputs()
    msg = (f"{name}: acc sha256 {ok_acc}/21, y_native rows {ok_rows}/21, y_native full sha256 {ok_full}/{n_full}, "
           f"bf16 via {sorted(paths)}")
    print(msg)
    assert ok_acc == 21 and ok_rows == 21 and ok_full == n_full, msg
    if os.environ.get("MMAEMU_REQUIRE_SCALED_MM") == "1":   # the bf16 rows must come from scaled_mm
        assert paths == {"scaled_mm"}, msg
