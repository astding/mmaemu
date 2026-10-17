"""Public API: torch 2.7.1 schema parity, validation order, preliminary rejection rules (measured on the
RTX 4060 with torch 2.7.1 / cuBLASLt 12.8.3.14 on 2026-10-12; the R1 recordings of recordings/rtx4060_2026-10
supersede them), scope errors, grad guard, zero sizes, out=, accumulate(), explain(), and the tcemu audit's repros
B6 / B13."""
from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest
import torch

from mmaemu.api import CallPlan, accumulate, explain, scaled_mm
from mmaemu.errors import (ExperimentalPresetWarning, MmaemuError, ReferenceRejectsError, SplitKUnknownError,
                           UnsupportedFeatureError)
from mmaemu.model import presets as P
from mmaemu.testing import assert_bitwise_equal

E4, E5 = torch.float8_e4m3fn, torch.float8_e5m2
ONE = torch.tensor(1.0)

# aten::_scaled_mm(Tensor self, Tensor mat2, Tensor scale_a, Tensor scale_b, Tensor? bias=None,
#                  Tensor? scale_result=None, ScalarType? out_dtype=None, bool use_fast_accum=False) -> Tensor
SCHEMA_2_7_1 = [("input", inspect.Parameter.empty), ("mat2", inspect.Parameter.empty),
                ("scale_a", inspect.Parameter.empty), ("scale_b", inspect.Parameter.empty), ("bias", None),
                ("scale_result", None), ("out_dtype", None), ("use_fast_accum", False)]


def _mats(M=32, K=64, N=32, fa=E4, fb=E4, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = (torch.randn(M, K, generator=g) * 40).clamp(-448, 448).to(fa)
    b = (torch.randn(N, K, generator=g) * 40).clamp(-448, 448).to(fb)
    return a, b.t()                        # mat2 column-major [K, N]


# --------------------------------------------------------------------------------------------------------------------
# signature
def test_signature_matches_torch_2_7_1_schema():
    sig = inspect.signature(scaled_mm)
    ps = list(sig.parameters.values())
    assert [(p.name, p.default) for p in ps[:8]] == SCHEMA_2_7_1
    assert all(p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for p in ps[:8])
    rest = {p.name: p for p in ps[8:]}
    assert list(rest) == ["out", "profile", "splitk", "backend"]
    assert all(p.kind == inspect.Parameter.KEYWORD_ONLY for p in rest.values())
    assert rest["out"].default is None and rest["splitk"].default == "auto" and rest["backend"].default == "auto"
    assert list(inspect.signature(explain).parameters) == list(sig.parameters)


@pytest.mark.skipif(not torch.__version__.startswith("2.7."), reason="live schema compared on torch 2.7.x only")
def test_signature_matches_live_schema():
    schema = torch.ops.aten._scaled_mm.default._schema
    live = [(a.name, a.default_value if a.has_default_value() else inspect.Parameter.empty) for a in schema.arguments]
    live[0] = ("input", live[0][1])          # `self` is exposed as `input`
    assert live == SCHEMA_2_7_1
    out_schema = torch.ops.aten._scaled_mm.out._schema
    assert [a.name for a in out_schema.arguments if a.kwarg_only] == ["out"]


def test_accumulate_signature():
    sig = inspect.signature(accumulate)
    p = sig.parameters["use_fast_accum"]
    assert p.kind == inspect.Parameter.KEYWORD_ONLY and p.default is inspect.Parameter.empty
    a, b = _mats()
    with pytest.raises(TypeError):
        accumulate(a, b.t())                               # use_fast_accum is required (tcemu audit repro B6)


def test_compiler_disable_marker():
    assert getattr(scaled_mm, "_torchdynamo_disable", False) or hasattr(scaled_mm, "__wrapped__")


# --------------------------------------------------------------------------------------------------------------------
# step 1: argument types (TypeError, as torch's argument parser)
def test_type_errors():
    a, b = _mats()
    with pytest.raises(TypeError):
        scaled_mm(a, b, 1.0, ONE)                          # python float scale
    with pytest.raises(TypeError):
        scaled_mm(a, b, ONE, ONE, torch.bfloat16)          # positional 5th argument is bias, not out_dtype
    with pytest.raises(TypeError):
        scaled_mm(a, b, ONE, ONE, use_fast_accum=1)
    with pytest.raises(TypeError):
        scaled_mm(a, b, ONE, ONE, use_fast_accum=np.bool_(True))      # torch 2.7.1 rejects numpy.bool_ too
    with pytest.raises(TypeError):
        scaled_mm(a, b, ONE, ONE, out_dtype="bfloat16")
    with pytest.raises(TypeError):
        scaled_mm([[1]], b, ONE, ONE)


# --------------------------------------------------------------------------------------------------------------------
# step 2: torch 2.7.1 checks -> ReferenceRejectsError (each measured on the RTX 4060, 2026-10-12)
def _offset_view(x: torch.Tensor, off: int) -> torch.Tensor:
    """Same values, storage offset `off` bytes."""
    base = torch.zeros(x.numel() + off, dtype=torch.uint8)
    v = base[off:].view(x.shape)
    v.copy_(x.contiguous().view(torch.uint8))
    return v.view(x.dtype)


def _padded_ld(x: torch.Tensor, ld: int) -> torch.Tensor:
    big = torch.zeros(x.shape[0], ld, dtype=torch.uint8)
    big[:, : x.shape[1]] = x.contiguous().view(torch.uint8)
    return big[:, : x.shape[1]].view(x.dtype)


def _reject_cases():
    a, b = _mats()
    bt = b.t()                                               # [N, K] row-major
    a5, b5 = _mats(fa=E5, fb=E5)
    c = []
    for off in (1, 2, 4, 8):
        c.append((f"a offset {off}B", lambda o=off: scaled_mm(_offset_view(a, o), b, ONE, ONE,
                                                                out_dtype=torch.float32)))
        c.append((f"b offset {off}B", lambda o=off: scaled_mm(a, _offset_view(bt, o).t(), ONE, ONE,
                                                                out_dtype=torch.float32)))
    c += [
        ("a ld 72", lambda: scaled_mm(_padded_ld(a, 72), b, ONE, ONE, out_dtype=torch.float32)),
        ("b ld 72", lambda: scaled_mm(a, _padded_ld(bt, 72).t(), ONE, ONE, out_dtype=torch.float32)),
        ("a col-major", lambda: scaled_mm(a.t().contiguous().t(), b, ONE, ONE, out_dtype=torch.float32)),
        ("b row-major", lambda: scaled_mm(a, b.contiguous(), ONE, ONE, out_dtype=torch.float32)),
        ("K % 16", lambda: scaled_mm(*_mats(32, 24, 32), ONE, ONE, out_dtype=torch.float32)),
        ("N % 16", lambda: scaled_mm(*_mats(32, 64, 24), ONE, ONE, out_dtype=torch.float32)),
        ("scale bf16", lambda: scaled_mm(a, b, ONE.bfloat16(), ONE, out_dtype=torch.float32)),
        ("scale fp64", lambda: scaled_mm(a, b, ONE.double(), ONE, out_dtype=torch.float32)),
        ("scale (2,)", lambda: scaled_mm(a, b, torch.ones(2), ONE, out_dtype=torch.float32)),
        ("scale (M,1) with scalar", lambda: scaled_mm(a, b, torch.ones(32, 1), ONE, out_dtype=torch.float32)),
        ("rowwise fp32 out", lambda: scaled_mm(a, b, torch.ones(32, 1), torch.ones(1, 32), out_dtype=torch.float32)),
        ("scale_result bf16", lambda: scaled_mm(a, b, ONE, ONE, scale_result=ONE.bfloat16(), out_dtype=E4)),
        ("scale_result (2,)", lambda: scaled_mm(a, b, ONE, ONE, scale_result=torch.ones(2), out_dtype=torch.float32)),
        ("bias bf16 -> fp16", lambda: scaled_mm(a, b, ONE, ONE, bias=torch.zeros(32, dtype=torch.bfloat16),
                                                out_dtype=torch.float16)),
        ("bias bf16 -> fp32", lambda: scaled_mm(a, b, ONE, ONE, bias=torch.zeros(32, dtype=torch.bfloat16),
                                                out_dtype=torch.float32)),
        ("bias fp16 -> bf16", lambda: scaled_mm(a, b, ONE, ONE, bias=torch.zeros(32, dtype=torch.float16),
                                                out_dtype=torch.bfloat16)),
        ("bias fp32 -> e4m3", lambda: scaled_mm(a, b, ONE, ONE, bias=torch.zeros(32), out_dtype=E4)),
        ("bias wrong length", lambda: scaled_mm(a, b, ONE, ONE, bias=torch.zeros(16, dtype=torch.bfloat16),
                                                out_dtype=torch.bfloat16)),
        ("out dtype mismatch", lambda: scaled_mm(a, b, ONE, ONE, out_dtype=torch.float32,
                                                 out=torch.empty(32, 32, dtype=torch.bfloat16))),
        ("fnuz input", lambda: scaled_mm(a.float().to(torch.float8_e4m3fnuz), b, ONE, ONE, out_dtype=torch.float32)),
        ("bf16 input", lambda: scaled_mm(a.float().bfloat16(), b, ONE, ONE, out_dtype=torch.float32)),
        ("e5m2 x e5m2", lambda: scaled_mm(a5, b5, ONE, ONE, out_dtype=torch.float32)),
        ("e4m3 x e4m3 -> e5m2 (profile)", lambda: scaled_mm(a, b, ONE, ONE, out_dtype=E5)),
        ("out_dtype int32", lambda: scaled_mm(a, b, ONE, ONE, out_dtype=torch.int32)),
        ("out_dtype fp64", lambda: scaled_mm(a, b, ONE, ONE, out_dtype=torch.float64)),
        ("3-d input", lambda: scaled_mm(a[None], b, ONE, ONE, out_dtype=torch.float32)),
        ("K mismatch", lambda: scaled_mm(a, _mats(32, 48, 32)[1], ONE, ONE, out_dtype=torch.float32)),
    ]
    return c


@pytest.mark.parametrize("name,fn", _reject_cases(), ids=lambda x: x if isinstance(x, str) else "")
def test_rejections(name, fn):
    with pytest.raises(ReferenceRejectsError):
        fn()


def test_accepted_layouts_and_shapes():
    a, b = _mats()
    bt = b.t()
    ref = scaled_mm(a, b, ONE, ONE, out_dtype=torch.float32, splitk=1)
    for off in (16, 32):
        assert_bitwise_equal(scaled_mm(_offset_view(a, off), b, ONE, ONE, out_dtype=torch.float32, splitk=1), ref)
        assert_bitwise_equal(scaled_mm(a, _offset_view(bt, off).t(), ONE, ONE, out_dtype=torch.float32, splitk=1), ref)
    for ld in (80, 96, 128):
        assert_bitwise_equal(scaled_mm(_padded_ld(a, ld), b, ONE, ONE, out_dtype=torch.float32, splitk=1), ref)
        assert_bitwise_equal(scaled_mm(a, _padded_ld(bt, ld).t(), ONE, ONE, out_dtype=torch.float32, splitk=1), ref)
    for M in (1, 17):                                        # M needs no alignment
        am, bm = _mats(M, 64, 32)
        assert scaled_mm(am, bm, ONE, ONE, out_dtype=torch.float32, splitk=1).shape == (M, 32)
    for s in (torch.ones(1), torch.ones(1, 1)):              # one-element scales of any shape are tensor-wise
        assert_bitwise_equal(scaled_mm(a, b, s, ONE, out_dtype=torch.float32, splitk=1), ref)
    a45, b45 = _mats(fa=E4, fb=E5)
    assert scaled_mm(a45, b45, ONE, ONE, out_dtype=E5, splitk=1).dtype == E5      # mixed -> e5m2 is allowed
    assert scaled_mm(a45, b45, ONE, ONE, splitk=1).dtype == E4                    # default out dtype = input dtype
    a54, b54 = _mats(fa=E5, fb=E4)
    assert scaled_mm(a54, b54, ONE, ONE, splitk=1).dtype == E5


# --------------------------------------------------------------------------------------------------------------------
# validation order: torch checks (2) < profile rejects (3) < scope (4)
def test_order_torch_checks_before_scope():
    a, b = _mats()
    bias = torch.zeros(32, dtype=torch.bfloat16)
    with pytest.raises(ReferenceRejectsError):              # bias with fp32 out: torch rejects first
        scaled_mm(a, b, ONE, ONE, bias, out_dtype=torch.float32)
    with pytest.raises(ReferenceRejectsError):              # profile rejects e4m3 x e4m3 -> e5m2 before scope
        scaled_mm(a, b, ONE, ONE, bias, out_dtype=E5)


def test_positional_bias_raises_never_dropped():
    a, b = _mats()
    bias = torch.zeros(32, dtype=torch.bfloat16)
    with pytest.raises(UnsupportedFeatureError, match="bias"):
        scaled_mm(a, b, ONE, ONE, bias, None, torch.bfloat16, True)       # torch-style full positional call
    with pytest.raises(UnsupportedFeatureError, match="bias"):
        scaled_mm(a, b, ONE, ONE, bias=bias, out_dtype=torch.bfloat16)
    with pytest.raises(UnsupportedFeatureError, match="bias"):
        scaled_mm(a, b, ONE, ONE, torch.zeros(32, dtype=torch.float16), out_dtype=E4)


def test_rowwise_scales_unsupported():
    a, b = _mats()
    with pytest.raises(UnsupportedFeatureError, match="row-wise"):
        scaled_mm(a, b, torch.ones(32, 1), torch.ones(1, 32), out_dtype=torch.bfloat16)


def test_grad_guard():
    a, b = _mats()
    sg = torch.tensor(1.0, requires_grad=True)
    with pytest.raises(UnsupportedFeatureError, match="inference-only"):
        scaled_mm(a, b, sg, ONE, out_dtype=torch.float32, splitk=1)
    ag = a.float().to(E4).requires_grad_(True)
    with pytest.raises(UnsupportedFeatureError):
        scaled_mm(ag, b, ONE, ONE, out_dtype=torch.float32, splitk=1)
    with pytest.raises(UnsupportedFeatureError):
        accumulate(ag, b.t(), use_fast_accum=True, splitk=1)
    with torch.no_grad():
        y = scaled_mm(a, b, sg, ONE, out_dtype=torch.float32, splitk=1)
    assert not y.requires_grad


def test_experimental_profile_warns_before_split_resolution():
    a, b = _mats()
    P._reset_warnings_for_tests()
    with pytest.warns(ExperimentalPresetWarning, match="profile 'hopper-experimental' is experimental"):
        scaled_mm(a, b, ONE, ONE, out_dtype=torch.float32, use_fast_accum=True, profile="hopper-experimental",
                  splitk=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                # once per process per profile
        scaled_mm(a, b, ONE, ONE, out_dtype=torch.float32, use_fast_accum=True, profile="hopper-experimental",
                  splitk=1)
    P._reset_warnings_for_tests()
    with pytest.warns(ExperimentalPresetWarning, match="profile 'hopper-experimental' is experimental"):
        accumulate(a, b.t(), use_fast_accum=False, profile="hopper-experimental", splitk=1)   # slow is modelled


# --------------------------------------------------------------------------------------------------------------------
# zero sizes (RTX 4060, torch 2.7.1: K = 0 -> all +0 for EVERY scale, incl. NaN/Inf; M or N = 0 -> empty)
@pytest.mark.parametrize("od", [torch.float32, torch.bfloat16, torch.float16, E4])
@pytest.mark.parametrize("sa", [1.0, -1.0, 0.0, -0.0, float("nan"), float("inf")])
def test_k0_is_plus_zero_for_every_scale(od, sa):
    a = torch.zeros(16, 0).to(E4)
    b = torch.zeros(16, 0).to(E4).t()
    for fast in (True, False):
        y = scaled_mm(a, b, torch.tensor(sa), ONE, out_dtype=od, use_fast_accum=fast)
        assert y.shape == (16, 16) and y.dtype == od
        assert int(y.contiguous().view({4: torch.int32, 2: torch.int16, 1: torch.uint8}[y.element_size()])
                   .abs().sum()) == 0


def test_m0_n0_empty():
    for M, N in ((0, 16), (16, 0)):
        a = torch.zeros(M, 32).to(E4)
        b = torch.zeros(N, 32).to(E4).t()
        y = scaled_mm(a, b, ONE, ONE, out_dtype=torch.bfloat16)
        assert y.shape == (M, N) and y.dtype == torch.bfloat16
    assert accumulate(torch.zeros(0, 5).to(E4), torch.zeros(3, 5).to(E4), use_fast_accum=True).shape == (0, 3)
    z = accumulate(torch.zeros(2, 0).to(E4), torch.zeros(3, 0).to(E4), use_fast_accum=False)
    assert z.shape == (2, 3) and int(z.view(torch.int32).abs().sum()) == 0


# --------------------------------------------------------------------------------------------------------------------
# out=
def test_out_semantics():
    a, b = _mats()
    ref = scaled_mm(a, b, ONE, ONE, out_dtype=torch.bfloat16, splitk=1)
    out = torch.full((32, 32), float("nan"), dtype=torch.bfloat16)
    y = scaled_mm(a, b, ONE, ONE, out=out, splitk=1)        # out_dtype None -> out's dtype
    assert y is out
    assert_bitwise_equal(out, ref)
    with pytest.warns(UserWarning, match="resized"):
        o2 = scaled_mm(a, b, ONE, ONE, out_dtype=torch.bfloat16, out=torch.empty(16, 16, dtype=torch.bfloat16),
                       splitk=1)
    assert_bitwise_equal(o2, ref)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        o3 = scaled_mm(a, b, ONE, ONE, out_dtype=torch.bfloat16, out=torch.empty(0, dtype=torch.bfloat16), splitk=1)
    assert_bitwise_equal(o3, ref)
    nc = torch.empty(32, 32, dtype=torch.bfloat16).t()
    scaled_mm(a, b, ONE, ONE, out_dtype=torch.bfloat16, out=nc, splitk=1)
    assert_bitwise_equal(nc.contiguous(), ref)


def test_out_without_unit_stride_is_returned_unchanged():
    # torch 2.7.1 computes into a contiguous copy of such an out= and never copies it back (RTX 4060: an out of
    # strides (64, 2) prefilled with NaN stays NaN)
    a, b = _mats(16, 64, 32)
    o = torch.full((16, 64), float("nan"), dtype=torch.bfloat16)[:, ::2]
    assert any("returned unchanged" in w for w in explain(a, b, ONE, ONE, out_dtype=torch.bfloat16, out=o).warnings)
    with pytest.warns(UserWarning, match="returned unchanged"):
        y = scaled_mm(a, b, ONE, ONE, out_dtype=torch.bfloat16, out=o, splitk=1)
    assert y is o and o.isnan().all()
    # the result is discarded, so a shape whose split-K decision is unknown does not raise either
    a2, b2 = _mats(17, 4112, 112)
    with pytest.raises(SplitKUnknownError):
        scaled_mm(a2, b2, ONE, ONE, out_dtype=torch.float32)
    o2 = torch.full((17, 224), float("nan"))[:, ::2]
    with pytest.warns(UserWarning, match="returned unchanged"):
        assert scaled_mm(a2, b2, ONE, ONE, out_dtype=torch.float32, out=o2) is o2
    assert o2.isnan().all()
    assert explain(a2, b2, ONE, ONE, out_dtype=torch.float32, out=o2).backend == "none (out= is returned unchanged)"
    o0 = torch.full((16, 64), float("nan"), dtype=torch.bfloat16)[:, ::2]
    scaled_mm(torch.zeros(16, 0).to(E4), torch.zeros(32, 0).to(E4).t(), ONE, ONE, out=o0)
    assert not o0.isnan().any() and not o0.any()           # K = 0: torch zeroes out= itself, whatever its strides


def test_column_major_out_uses_the_swapped_split():
    # torch 2.7.1 computes a column-major out= as mat1 x mat2, so cuBLASLt's heuristic sees the row-major problem
    # (N, M, K) with the formats swapped (RTX 4060: 406 of 406 such calls; T3p checks it live)
    a, b = _mats(16, 8192, 48)
    bf = dict(out_dtype=torch.bfloat16, profile="ada")
    col = torch.empty(48, 16, dtype=torch.bfloat16).t()
    row = explain(a, b, ONE, ONE, **bf).split
    swapped = explain(b.t(), a.t(), ONE, ONE, **bf).split
    assert (row.S, row.scheme, swapped.S, swapped.scheme) == (8, "pairwise", 12, "pairwise")
    d = explain(a, b, ONE, ONE, **bf, out=col).split
    assert (d.S, d.scheme, d.source) == (12, "pairwise", "table")
    y = scaled_mm(a, b, ONE, ONE, **bf, out=col)
    assert_bitwise_equal(y.contiguous(), scaled_mm(a, b, ONE, ONE, **bf, splitk=(12, "pairwise")))
    # a padded column-major out is still column-major; a wrong-shape out is resized to row-major first
    padded = torch.empty(48, 32, dtype=torch.bfloat16)[:, :16].t()
    assert explain(a, b, ONE, ONE, **bf, out=padded).split.S == 12
    resized = torch.empty(16, 48, dtype=torch.bfloat16).t()
    assert explain(a, b, ONE, ONE, **bf, out=resized).split.S == 8


def test_mat1_neither_row_nor_column_major_is_copied():
    # torch 2.7.1 (prepare_matrix_for_cublas) copies such a mat1 to a contiguous one; measured on the RTX 4060
    g = torch.Generator().manual_seed(7)
    x = (torch.randn(1, 64, generator=g) * 40).clamp(-448, 448).to(E4)
    big = (torch.randn(32, 128, generator=g) * 40).clamp(-448, 448).to(E4)
    _, b = _mats()
    for mat1 in (x.expand(32, 64), big[:, ::2]):
        assert_bitwise_equal(scaled_mm(mat1, b, ONE, ONE, splitk=1), scaled_mm(mat1.contiguous(), b, ONE, ONE,
                                                                                splitk=1))
    with pytest.raises(ReferenceRejectsError, match="row-major and column-major"):
        scaled_mm(big[:, :64].t().contiguous().t(), b, ONE, ONE, splitk=1)        # column-major mat1


@pytest.mark.parametrize("od", [torch.float64, torch.int32])
def test_zero_size_returns_before_the_out_dtype_check(od):
    # torch returns early for M, N or K = 0 whatever the out dtype; cuBLASLt rejects the dtype only for real GEMMs
    for M, K, N in ((0, 32, 16), (16, 32, 0), (16, 0, 16)):
        a, b = torch.zeros(M, K).to(E4), torch.zeros(N, K).to(E4).t()
        y = scaled_mm(a, b, ONE, ONE, out_dtype=od)
        assert y.shape == (M, N) and y.dtype == od and not y.any()
    a, b = _mats()
    with pytest.raises(ReferenceRejectsError, match="not supported by cuBLASLt"):
        scaled_mm(a, b, ONE, ONE, out_dtype=od)


@pytest.mark.parametrize("dt", ["float8_e4m3fnuz", "float8_e5m2fnuz", "float8_e8m0fnu"])
def test_zero_size_accepts_every_torch_float8_dtype(dt):
    # torch's isFloat8Type admits fnuz and e8m0: zero sizes return (RTX 4060), real GEMMs fail on CUDA
    if not hasattr(torch, dt):
        pytest.skip(f"torch {torch.__version__} has no {dt}")
    D = getattr(torch, dt)
    for M, K, N in ((0, 32, 16), (16, 0, 16), (16, 32, 0)):
        y = scaled_mm(torch.ones(M, K).to(D), torch.zeros(N, K).to(E4).t(), ONE, ONE, out_dtype=torch.float32)
        assert y.shape == (M, N) and not y.any()
    with pytest.raises(ReferenceRejectsError, match="other than float8_e4m3fn"):
        scaled_mm(torch.ones(16, 32).to(D), torch.zeros(16, 32).to(E4).t(), ONE, ONE, out_dtype=torch.float32)


# --------------------------------------------------------------------------------------------------------------------
# accumulate: any K, alignment, layout; dtype and K checks
def test_accumulate_any_k_alignment_layout():
    g = torch.Generator().manual_seed(4)
    a = (torch.randn(9, 37, generator=g) * 40).clamp(-448, 448).to(E4)
    b = (torch.randn(7, 37, generator=g) * 40).clamp(-448, 448).to(E4)
    ref = accumulate(a, b, use_fast_accum=True, splitk=1)
    assert ref.dtype == torch.float32 and ref.shape == (9, 7)
    pad = torch.zeros(9, 11).to(E4), torch.zeros(7, 11).to(E4)
    assert_bitwise_equal(accumulate(torch.cat([a, pad[0]], 1), torch.cat([b, pad[1]], 1), use_fast_accum=True,
                                    splitk=1), ref)
    assert_bitwise_equal(accumulate(_offset_view(a, 3), _offset_view(b, 5), use_fast_accum=True, splitk=1), ref)
    assert_bitwise_equal(accumulate(a.t().contiguous().t(), b.t().contiguous().t(), use_fast_accum=True, splitk=1),
                         ref)


def test_accumulate_checks():
    a = torch.zeros(4, 32).to(E4)
    with pytest.raises(TypeError):
        accumulate(a.float(), a, use_fast_accum=True)
    with pytest.raises(TypeError):
        accumulate(a.float().to(torch.float8_e4m3fnuz), a, use_fast_accum=True)
    with pytest.raises(ValueError, match="K mismatch"):
        accumulate(a, torch.zeros(4, 48).to(E4), use_fast_accum=True)
    with pytest.raises(TypeError):
        accumulate(a, a, use_fast_accum="yes")
    assert accumulate(a, a, use_fast_accum=np.bool_(True), splitk=1).shape == (4, 4)    # accumulate is not torch


# --------------------------------------------------------------------------------------------------------------------
# explain
def test_explain():
    a, b = _mats()
    cp = explain(a, b, ONE, ONE, out_dtype=torch.float32, splitk=1, profile="ada")
    assert isinstance(cp, CallPlan) and cp.validation == "ok"
    assert cp.profile == "ada" and cp.preset == "ada" and cp.preset_maturity == "validated"
    assert cp.split.S == 1 and cp.split.source == "explicit" and cp.backend == "reference"
    assert cp.fallbacks == ["cuda: tensors on cpu"]           # the selection's own reasons for this call
    cp = explain(a, b, ONE, ONE, out_dtype=E5)
    assert cp.validation[0] == "ReferenceRejectsError"
    cp = explain(a, b, ONE, ONE, torch.zeros(32, dtype=torch.bfloat16), out_dtype=torch.bfloat16, splitk=1)
    assert cp.validation[0] == "UnsupportedFeatureError"
    for prof, split in (("hopper", (2, "serial")), ("ada", (2, "contiguous"))):
        cp = explain(a, b, ONE, ONE, out_dtype=torch.float32, use_fast_accum=True, splitk=split, profile=prof)
        assert cp.validation[0] == "ValueError" and "is not a split-K layout of preset" in cp.validation[1]
        with pytest.raises(ValueError, match="is not a split-K layout of preset"):    # what explain reports
            scaled_mm(a, b, ONE, ONE, out_dtype=torch.float32, use_fast_accum=True, splitk=split, profile=prof)
    P._reset_warnings_for_tests()
    with warnings.catch_warnings():
        warnings.simplefilter("error")              # explain never emits the warning itself, it reports it
        cp = explain(a, b, ONE, ONE, out_dtype=torch.float32, use_fast_accum=True, profile="hopper-experimental",
                     splitk=1)
    assert cp.preset_maturity == "validated"
    assert any("profile 'hopper-experimental' is experimental" in w for w in cp.warnings)


# --------------------------------------------------------------------------------------------------------------------
# FTZ canary
def test_flush_denormal_canary():
    a, b = _mats()
    if not torch.set_flush_denormal(True):
        pytest.skip("this CPU cannot flush denormals")
    try:
        with pytest.raises(MmaemuError, match="flush"):
            scaled_mm(a, b, ONE, ONE, out_dtype=torch.float32, splitk=1)
        with pytest.raises(MmaemuError, match="flush"):
            accumulate(a, b.t(), use_fast_accum=True, splitk=1)
    finally:
        torch.set_flush_denormal(False)
    scaled_mm(a, b, ONE, ONE, out_dtype=torch.float32, splitk=1)


# --------------------------------------------------------------------------------------------------------------------
# audit repros
def test_b6_repro_api_defaults_behaviour():
    """verify-c4-cublas-fuzz-5/repro_api_defaults.py (RTX 4060): torch's default out dtype is the input dtype and
    use_fast_accum defaults to False; a positional 5th dtype is a TypeError; a positional bias tensor is accepted by
    torch (mmaemu raises instead of dropping it); 8 positional arguments are accepted. Bits: -m truth below."""
    a, b = _mats(64, 1920, 256, seed=1)
    sa, sb = torch.tensor(0.0123), torch.tensor(0.000731)
    assert scaled_mm(a, b, sa, sb, splitk=1).dtype == E4
    for od in (torch.float32, torch.bfloat16):
        default = scaled_mm(a, b, sa, sb, out_dtype=od, splitk=1)
        assert_bitwise_equal(default, scaled_mm(a, b, sa, sb, out_dtype=od, use_fast_accum=False, splitk=1))
    fast = scaled_mm(a, b, sa, sb, out_dtype=torch.float32, use_fast_accum=True, splitk=1)
    slow = scaled_mm(a, b, sa, sb, out_dtype=torch.float32, splitk=1)
    assert (fast.view(torch.int32) != slow.view(torch.int32)).float().mean().item() > 0.9   # 99.56% on hardware
    y = scaled_mm(a, b, sa, sb, None, None, torch.bfloat16, True, splitk=1)
    assert y.dtype == torch.bfloat16
    y = scaled_mm(a, b, sa, sb, scale_result=torch.tensor(2.0), out_dtype=E4, splitk=1)
    assert y.dtype == E4


def _fp8_out_inputs(fa):
    """verify-c4-cublas-fuzz-5/repro_fp8_out.py inputs (deterministic), M, K, N = 16, 32, 16."""
    a = torch.full((16, 32), 16.0).to(fa)
    b = torch.full((16, 32), 16.0).to(E4).t()
    a[1] = -16.0
    a[2] = 0.0
    a[2, 0] = 15.0
    a[3] = 0.0
    a[3, 0] = 30.0
    a[4] = 0.0
    a[4, 0] = 29.0 if fa == E4 else 28.0
    return a, b


# hardware bytes of rows 0..4, column 0 (repro_fp8_out.out, RTX 4060, torch 2.7.1)
FP8_OUT_HW = {(E4, E4): [0x7E, 0xFE, 0x77, 0x7E, 0x7E], (E5, E4): [0x7E, 0xFE, 0x78, 0x7E, 0x7E],
              (E5, E5): [0x70, 0xF0, 0x5C, 0x60, 0x5F]}


@pytest.mark.parametrize("fa,od", [(E4, E4), (E5, E4), (E5, E5), (E4, E5)])
def test_b6_repro_fp8_out(fa, od):
    a, b = _fp8_out_inputs(fa)
    if (fa, od) == (E4, E5):
        with pytest.raises(ReferenceRejectsError):   # CUBLAS_STATUS_NOT_SUPPORTED on the RTX 4060
            scaled_mm(a, b, ONE, ONE, out_dtype=od, use_fast_accum=True, splitk=1)
        return
    y = scaled_mm(a, b, ONE, ONE, out_dtype=od, use_fast_accum=True, splitk=1)
    assert y.view(torch.uint8)[:5, 0].tolist() == FP8_OUT_HW[(fa, od)]
    assert scaled_mm(a, b, ONE, ONE, use_fast_accum=True, splitk=1).dtype == fa


def test_b6_repro_fp8_out_specials():
    a = torch.zeros(16, 32).to(E5)
    av = a.view(torch.uint8)
    av[0, 0], av[1, 0], av[2, 0] = 0x7C, 0x7E, 0xFC                 # +inf, nan, -inf
    b = torch.ones(16, 32).to(E4).t()
    hw = {E4: [0x7E, 0x7F, 0xFE], E5: [0x7B, 0x7F, 0xFB]}
    for od, want in hw.items():
        y = scaled_mm(a, b, ONE, ONE, out_dtype=od, use_fast_accum=True, splitk=1)
        assert y.view(torch.uint8)[:3, 0].tolist() == want
    y = scaled_mm(a, b, ONE, ONE, out_dtype=torch.float32, use_fast_accum=True, splitk=1)
    assert y[0, 0].item() == float("inf") and y[2, 0].item() == -float("inf")
    assert y.view(torch.int32)[1, 0].item() == 0x7FFFFFFF


def test_b13_repro_v05_minimal():
    """verify-a1-reference-review-4/scripts/v05_minimal.py: M1 through scaled_mm (E5M2 A, E4M3 B) equals the RTX
    4060 (0x1.001p-9); M2 K mismatch raises as torch did; M3 is covered in test_model_chain."""
    a = torch.zeros(16, 32).to(E5)
    b = torch.zeros(16, 32).to(E4)
    a[0, 0], a[0, 1] = 2.0 ** -9, 2.0 ** -12
    b[0, 0], b[0, 1] = 1.0, 2.0 ** -9
    y = scaled_mm(a, b.t(), ONE, ONE, out_dtype=torch.float32, use_fast_accum=True, splitk=1)
    assert y[0, 0].item() == 2.0 ** -9 + 2.0 ** -21
    assert int(y.view(torch.int32).flatten()[1:].abs().sum()) == 0
    with pytest.raises(ReferenceRejectsError, match="cannot be multiplied"):
        scaled_mm(torch.ones(16, 32).to(E4), torch.ones(48, 16).to(E4), ONE, ONE, out_dtype=torch.float32)


@pytest.mark.truth
def test_truth_api_defaults():
    """misc.api_defaults_va1_3: the hardware outputs of calls that omit out_dtype / use_fast_accum (call_variant),
    replayed with the same omissions."""
    from test_epilogue import replay_gemm_output, truth_dir, truth_loader
    loader = truth_loader()
    rec = loader.load("misc.api_defaults_va1_3", root=truth_dir())
    variants = set()
    for case in rec.cases:
        ins = case.inputs()
        for o in case.outputs:
            kind, val = replay_gemm_output(ins, o, honour_call_variant=True)
            assert kind == "bits", (case.id, o.name, kind, val)
            hw = o.bits()
            assert np.array_equal(val.reshape(hw.shape), hw), (case.id, o.name)
            variants.add(o.params.get("call_variant"))
    assert {"default_all", "default_fa", "default_fa_bf16"} <= variants
