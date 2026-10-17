"""T3p: live torch._scaled_mm on the profile's reference GPU vs mmaemu.scaled_mm.

Runs only where get_profile("ada").matches_device() (RTX 4060, 24 SMs, cuBLASLt 12.8.3.14), with
torch's default cuBLASLt workspace (CUBLASLT_WORKSPACE_SIZE unset); elsewhere it is skipped with "profile device
unavailable". Run: $MMAEMU_GPU_LOCK timeout 600 python -m pytest -q tests/test_live_cublas.py

Covers fast and slow accumulation; fp32 / bf16 / fp16 / e4m3 / e5m2 outputs (where allowed); the three FP8 format
pairs; three scale sets (incl. underflowing alpha and negative scales); unsplit shapes (M*N = 2^20, run with
splitk=1) and shapes whose split-K decision was recorded on this GPU (S = 3, 8, 12); NaN/Inf inputs;
K = 0 / M = 0 / N = 0; and raise / no-raise parity of the argument checks. test_live_out_and_operand_layouts adds
column-major and padded out= on shapes whose transposed problem splits differently, an out= without a unit stride
(torch leaves it unchanged), a mat1 that torch copies (expanded, step 2), and zero sizes with out dtypes cuBLASLt
rejects or with fnuz / e8m0 inputs. Each comparison is bitwise
(mmaemu.testing.assert_bitwise_equal, NaN payloads included). mmaemu runs its reference backend on the same GPU.
"""
from __future__ import annotations

import os

import pytest
import torch

from mmaemu import api
from mmaemu.errors import ReferenceRejectsError
from mmaemu.testing import assert_bitwise_equal

pytestmark = [pytest.mark.gpu, pytest.mark.profile_device]
PROFILE = "ada"  # the live GPU's profile; every mmaemu call here runs under it


def scaled_mm(*args, **kw):
    return api.scaled_mm(*args, profile=PROFILE, **kw)


E4, E5 = torch.float8_e4m3fn, torch.float8_e5m2
OUTS = [torch.float32, torch.bfloat16, torch.float16, E4, E5]
PAIRS = [(E4, E4), (E4, E5), (E5, E4)]
SCALES = [(0.0123, 0.000731), (-1.7, 0.0031), (2.0 ** -60, 2.0 ** -61)]
# split-K decisions recorded on this GPU/profile (torch default workspace):
#   (M, N, K), fmt pair, out, fast -> (S, scheme)
#   misc.tcemu_tests_recorded_hw (16 x 128 x 8192, e4m3 x e4m3 and e5m2 x e4m3)
#   epilogue.c1_4_hw_underflow (32 x 32 x 32 slow fp32: S = 3 pairwise; fast: S = 1)
KNOWN_SPLITS = []
for _fa in (E4, E5):
    KNOWN_SPLITS += [((16, 128, 8192), (_fa, E4), torch.float32, True, (8, "pairwise")),
                     ((16, 128, 8192), (_fa, E4), torch.bfloat16, True, (8, "pairwise")),
                     ((16, 128, 8192), (_fa, E4), torch.float32, False, (8, "pairwise")),
                     ((16, 128, 8192), (_fa, E4), torch.bfloat16, False, (12, "pairwise"))]
KNOWN_SPLITS += [((32, 32, 32), (E5, E4), torch.float32, False, (3, "pairwise")),
                 ((32, 32, 32), (E4, E4), torch.float32, False, (3, "pairwise")),
                 ((32, 32, 32), (E4, E4), torch.float32, True, 1),
                 ((32, 32, 32), (E5, E4), torch.bfloat16, True, 1)]


def _device_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("profile device unavailable (no CUDA)")
    from mmaemu.platform.profiles import get_profile
    if not get_profile(PROFILE).matches_device():
        pytest.skip("profile device unavailable")
    if os.environ.get("CUBLASLT_WORKSPACE_SIZE"):
        pytest.skip("CUBLASLT_WORKSPACE_SIZE is set: not the profile's workspace")
    return torch.device("cuda", torch.cuda.current_device())


class Tally:
    def __init__(self):
        self.calls = 0
        self.failures = []

    def compare(self, label, hw_fn, em_fn):
        self.calls += 1
        try:
            hw = hw_fn()
            hw_err = None
        except TypeError:
            hw, hw_err = None, "TypeError"
        except RuntimeError:
            hw, hw_err = None, "ReferenceRejectsError"
        try:
            em = em_fn()
            em_err = None
        except TypeError:
            em, em_err = None, "TypeError"
        except ReferenceRejectsError:
            em, em_err = None, "ReferenceRejectsError"
        except Exception as e:  # noqa: BLE001
            self.failures.append((label, f"mmaemu raised {type(e).__name__}: {e}"))
            return
        if hw_err or em_err:
            if hw_err != em_err:
                self.failures.append((label, f"torch: {hw_err or 'ok'}, mmaemu: {em_err or 'ok'}"))
            return
        try:
            assert_bitwise_equal(em.cpu(), hw.cpu(), msg=label)
        except AssertionError as e:
            self.failures.append((label, str(e).splitlines()[0]))


def _data(M, K, N, fa, fb, g, dev, specials=False):
    a = (torch.randn(M, K, generator=g) * 40).clamp(-448, 448).to(fa)
    b = (torch.randn(N, K, generator=g) * 40).clamp(-448, 448).to(fb)
    if specials and M > 2 and N > 2:
        av = a.view(torch.uint8)
        av[1, 3] = 0x7F if fa == E4 else 0x7E              # NaN row
        if fa == E5:
            av[2, 5] = 0x7C                                 # +Inf row
    return a.to(dev), b.to(dev).t()


def test_live_parity():
    dev = _device_or_skip()
    T = Tally()
    g = torch.Generator().manual_seed(2026)

    def run(label, a, b, sa, sb, od, fast, splitk, **kw):
        T.compare(label,
                  lambda: torch._scaled_mm(a, b, sa, sb, out_dtype=od, use_fast_accum=fast, **kw),
                  lambda: scaled_mm(a, b, sa, sb, out_dtype=od, use_fast_accum=fast, splitk=splitk,
                                    backend="reference", **kw))

    # unsplit grid, run with splitk=1: M*N = 2^20 (the largest split M*N of any recorded setting is 983,040)
    for (M, K, N) in [(1024, 256, 1024), (512, 1920, 2048)]:
        for fa, fb in PAIRS:
            a, b = _data(M, K, N, fa, fb, g, dev, specials=(K == 256))
            for sa_, sb_ in SCALES:
                sa, sb = torch.tensor(sa_, device=dev), torch.tensor(sb_, device=dev)
                for fast in (True, False):
                    for od in OUTS:
                        run(f"{M}x{K}x{N} {fa}x{fb}->{od} fast={fast} s=({sa_:g},{sb_:g})", a, b, sa, sb, od, fast, 1)
    # scale_result (fp8 outputs: second multiply before saturation; ignored otherwise)
    a, b = _data(1024, 256, 1024, E4, E4, g, dev)
    for od in OUTS:
        for sr in (0.37, 3.0e5):
            T.compare(f"scale_result {sr} -> {od}",
                      lambda: torch._scaled_mm(a, b, torch.tensor(0.5, device=dev), torch.tensor(0.25, device=dev),
                                               scale_result=torch.tensor(sr, device=dev), out_dtype=od),
                      lambda: scaled_mm(a, b, torch.tensor(0.5, device=dev), torch.tensor(0.25, device=dev),
                                        scale_result=torch.tensor(sr, device=dev), out_dtype=od, splitk=1,
                                        backend="reference"))
    # shapes whose split-K decision was recorded on this GPU
    for (M, N, K), (fa, fb), od, fast, sk in KNOWN_SPLITS:
        for rep in range(3):
            a, b = _data(M, K, N, fa, fb, g, dev)
            run(f"split {M}x{N}x{K} {fa}x{fb}->{od} fast={fast} S={sk} rep{rep}", a, b,
                torch.tensor(0.0123, device=dev), torch.tensor(0.000731, device=dev), od, fast, sk)
    # zero sizes (torch: K = 0 -> zeros regardless of the scales)
    for od in (torch.float32, torch.bfloat16, torch.float16, E4):
        for s in (1.0, -1.0, float("nan"), float("inf")):
            a = torch.zeros(16, 0, device=dev).to(E4)
            b = torch.zeros(16, 0, device=dev).to(E4).t()
            run(f"K=0 {od} sa={s}", a, b, torch.tensor(s, device=dev), torch.tensor(1.0, device=dev), od, True, 1)
    for M, N in ((0, 16), (16, 0)):
        a = torch.zeros(M, 32, device=dev).to(E4)
        b = torch.zeros(N, 32, device=dev).to(E4).t()
        run(f"M={M} N={N}", a, b, torch.tensor(1.0, device=dev), torch.tensor(1.0, device=dev), torch.float32, True, 1)
    # argument-check parity (raise / no-raise and exception class)
    _arg_checks(T, dev, g)
    print(f"live calls: {T.calls}, failures: {len(T.failures)}")
    for f in T.failures[:20]:
        print("  FAIL", f)
    assert T.calls >= 200
    assert not T.failures, T.failures[:10]


# (M, N, K, fmt a, fmt b, out, fast): table rows whose transposed problem (N, M, K) splits differently, so a
# column-major out= changes cuBLAS's split-K decision (picked from the 406 such rows probed on 2026-10-14)
OUT_LAYOUT_CASES = [
    (16, 48, 8208, E4, E4, torch.float32, False), (96, 64, 12288, E4, E5, torch.float32, False),
    (32, 48, 1856, E4, E4, torch.float32, True), (64, 96, 1920, E5, E4, torch.float32, True),
    (16, 48, 8192, E4, E4, torch.bfloat16, False), (64, 192, 2048, E5, E4, torch.bfloat16, False),
    (16, 112, 1904, E4, E4, torch.bfloat16, True), (256, 96, 4096, E4, E5, torch.bfloat16, True),
    (64, 16, 4096, E4, E4, torch.float16, False), (192, 64, 8192, E4, E5, torch.float16, False),
    (16, 32, 12288, E4, E4, torch.float16, True), (96, 128, 4096, E4, E5, torch.float16, True),
    (64, 192, 1920, E4, E4, E4, False), (128, 96, 2048, E4, E5, E4, False),
    (96, 128, 4096, E4, E4, E4, True), (64, 96, 8192, E5, E4, E4, True),
    (64, 96, 1920, E4, E5, E5, False), (96, 128, 4096, E5, E4, E5, True),
]


def test_live_out_and_operand_layouts():
    """Column-major and padded out=, an out= without a unit stride (torch leaves it unchanged), a mat1 torch copies
    (expand, step 2), and zero sizes with any out dtype or any float8 input dtype."""
    dev = _device_or_skip()
    T = Tally()
    g = torch.Generator().manual_seed(2027)
    s = torch.tensor(1.0, device=dev)

    def outs(M, N, od):
        return {"row": None,
                "col": lambda: torch.empty(N, M, dtype=od, device=dev).t(),
                "row ld+16": lambda: torch.empty(M, N + 16, dtype=od, device=dev)[:, :N],
                "col ld+16": lambda: torch.empty(N, M + 16, dtype=od, device=dev)[:, :M].t(),
                "step 2 (NaN)": lambda: torch.full((M, 2 * N), float("nan"), device=dev).to(od)[:, ::2]}

    for M, N, K, fa, fb, od, fast in OUT_LAYOUT_CASES:
        a = (torch.randn(M, K, generator=g) * (0.5 if od in (E4, E5) else 4.0)).clamp(-448, 448).to(fa).to(dev)
        b = (torch.randn(N, K, generator=g) * (0.5 if od in (E4, E5) else 4.0)).clamp(-448, 448).to(fb).to(dev).t()
        for name, mk in outs(M, N, od).items():
            kw = dict(out_dtype=od, use_fast_accum=fast)
            T.compare(f"out {name} {M}x{N}x{K} {fa}x{fb}->{od} fast={fast}",
                      lambda: torch._scaled_mm(a, b, s, s, **kw, **({"out": mk()} if mk else {})).contiguous(),
                      lambda: scaled_mm(a, b, s, s, **kw, backend="reference",
                                        **({"out": mk()} if mk else {})).contiguous())
    x = (torch.randn(1, 256, generator=g) * 4).to(E4).to(dev)
    big = (torch.randn(64, 512, generator=g) * 4).to(E4).to(dev)
    w = (torch.randn(48, 256, generator=g) * 4).to(E4).to(dev).t()
    for name, mat1 in (("expand", x.expand(64, 256)), ("step 2", big[:, ::2])):
        T.compare(f"mat1 {name}", lambda: torch._scaled_mm(mat1, w, s, s, out_dtype=torch.bfloat16),
                  lambda: scaled_mm(mat1, w, s, s, out_dtype=torch.bfloat16, backend="reference"))
    for od in (torch.float64, torch.int32):
        for M, K, N in ((0, 256, 48), (64, 256, 0), (64, 0, 48), (64, 256, 48)):
            a, b = big[:M, :K], w[:K, :N]
            T.compare(f"M={M} K={K} N={N} -> {od}", lambda: torch._scaled_mm(a, b, s, s, out_dtype=od),
                      lambda: scaled_mm(a, b, s, s, out_dtype=od, backend="reference"))
    for dt in ("float8_e4m3fnuz", "float8_e5m2fnuz", "float8_e8m0fnu"):
        if not hasattr(torch, dt):                      # float8_e8m0fnu is new in torch 2.7
            continue
        for M, K, N in ((0, 256, 48), (64, 256, 0), (64, 0, 48), (64, 256, 48)):
            a, b = torch.ones(M, K, device=dev).to(getattr(torch, dt)), w[:K, :N]
            T.compare(f"{dt} M={M} K={K} N={N}", lambda: torch._scaled_mm(a, b, s, s, out_dtype=torch.float32),
                      lambda: scaled_mm(a, b, s, s, out_dtype=torch.float32, backend="reference"))
    print(f"live calls: {T.calls}, failures: {len(T.failures)}")
    for f in T.failures[:20]:
        print("  FAIL", f)
    n_fp8 = sum(hasattr(torch, dt) for dt in ("float8_e4m3fnuz", "float8_e5m2fnuz", "float8_e8m0fnu"))
    assert T.calls == 5 * len(OUT_LAYOUT_CASES) + 2 + 8 + 4 * n_fp8
    assert not T.failures, T.failures[:10]


def _arg_checks(T: Tally, dev, g):
    one = torch.ones((), device=dev)
    a, b = _data(32, 64, 32, E4, E4, g, dev)

    def offset(x, off):
        base = torch.zeros(x.numel() + off, device=dev, dtype=torch.uint8)
        v = base[off:].view(x.shape)
        v.copy_(x.contiguous().view(torch.uint8))
        return v.view(x.dtype)

    def ld(x, n):
        big = torch.zeros(x.shape[0], n, device=dev, dtype=torch.uint8)
        big[:, : x.shape[1]] = x.contiguous().view(torch.uint8)
        return big[:, : x.shape[1]].view(x.dtype)

    bt = b.t()
    cases = []
    for off in (1, 2, 4, 8, 16, 32):
        cases.append((f"a offset {off}", dict(input=offset(a, off), mat2=b)))
        cases.append((f"b offset {off}", dict(input=a, mat2=offset(bt, off).t())))
    for n in (72, 80, 96):
        cases.append((f"a ld {n}", dict(input=ld(a, n), mat2=b)))
        cases.append((f"b ld {n}", dict(input=a, mat2=ld(bt, n).t())))
    a5, b5 = _data(32, 64, 32, E5, E5, g, dev)
    cases += [
        ("a col-major", dict(input=a.t().contiguous().t(), mat2=b)),
        ("b row-major", dict(input=a, mat2=b.contiguous())),
        ("scale bf16", dict(input=a, mat2=b, scale_a=one.bfloat16())),
        ("scale (1,1)", dict(input=a, mat2=b, scale_a=torch.ones(1, 1, device=dev))),
        ("scale (2,)", dict(input=a, mat2=b, scale_a=torch.ones(2, device=dev))),
        ("scale on cpu", dict(input=a, mat2=b, scale_a=torch.ones(()))),
        ("e5m2 x e5m2", dict(input=a5, mat2=b5)),
        ("e4m3 x e4m3 -> e5m2", dict(input=a, mat2=b, out_dtype=E5)),
        ("out_dtype int32", dict(input=a, mat2=b, out_dtype=torch.int32)),
        ("bias fp32 -> bf16", dict(input=a, mat2=b, out_dtype=torch.bfloat16,
                                   bias=torch.zeros(32, device=dev))),
        ("bias bf16 -> fp32", dict(input=a, mat2=b, out_dtype=torch.float32,
                                   bias=torch.zeros(32, device=dev, dtype=torch.bfloat16))),
        ("out dtype mismatch", dict(input=a, mat2=b, out_dtype=torch.float32,
                                    out=torch.empty(32, 32, device=dev, dtype=torch.bfloat16))),
        ("scale_result (2,)", dict(input=a, mat2=b, out_dtype=E4, scale_result=torch.ones(2, device=dev))),
        ("K % 16", dict(input=_data(32, 24, 32, E4, E4, g, dev)[0], mat2=_data(32, 24, 32, E4, E4, g, dev)[1])),
        ("N % 16", dict(input=a, mat2=_data(32, 64, 24, E4, E4, g, dev)[1])),
        ("M = 17", dict(input=_data(17, 64, 32, E4, E4, g, dev)[0], mat2=b)),
    ]
    for label, kw in cases:
        kw.setdefault("scale_a", one)
        kw.setdefault("scale_b", one)
        kw.setdefault("out_dtype", torch.float32)
        T.compare(f"args: {label}", lambda kw=kw: torch._scaled_mm(**kw),      # torch 2.7.1 accepts input=
                  lambda kw=kw: scaled_mm(**kw, splitk=1, backend="reference"))
