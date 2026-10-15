"""GEMM-level plans: hand-written split-K index vectors, reduction orders, execute_plan
composition, identity plans, and (-m truth) the recorded project cases of misc.tcemu_tests_recorded_hw at S = 8/8/8/12.

The index vectors below are written out literally (not via split_index) so the test is not circular."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mmaemu.api import accumulate
from mmaemu.backends.reference import REFERENCE
from mmaemu.model.chain import dot
from mmaemu.model.plan import execute_plan, make_plan, reduce_partials, split_index
from mmaemu.testing import assert_bitwise_equal

E4 = torch.float8_e4m3fn


def R(a, b):
    return list(range(a, b))


PAD = -1


def idx(*parts):
    out = []
    for p in parts:
        out += p
    return out


# literal expectations: split s owns 64-wide tiles t with t % S == s, in increasing t; -1 = zero padding
LITERAL = {
    (80, 3): [R(0, 64), idx(R(64, 80), [PAD] * 48), [PAD] * 64],
    (200, 3): [idx(R(0, 64), R(192, 200), [PAD] * 56), idx(R(64, 128), [PAD] * 64), idx(R(128, 192), [PAD] * 64)],
    (64, 2): [R(0, 64), [PAD] * 64],
    (65, 2): [R(0, 64), idx([64], [PAD] * 63)],
    (128, 2): [R(0, 64), R(64, 128)],
}


@pytest.mark.parametrize("K,S", sorted(LITERAL))
def test_split_plan_literal_fast(K, S):
    plan = make_plan(K, fast=True, S=S, scheme="serial")
    assert plan.S == S and not plan.identity and plan.scheme == "serial"
    assert [[c.tolist() for c in chunks] for chunks in plan.chunks] == [[v] for v in LITERAL[(K, S)]]


def test_split_plan_literal_slow():
    # S = 2, K = 512: 8 tiles; split 0 owns tiles 0, 2, 4, 6 -> chunks (0, 2) and (4, 6) (restart every 2 OWN tiles)
    plan = make_plan(512, fast=False, S=2, scheme="pairwise")
    got = [[c.tolist() for c in chunks] for chunks in plan.chunks]
    assert got == [[idx(R(0, 64), R(128, 192)), idx(R(256, 320), R(384, 448))],
                   [idx(R(64, 128), R(192, 256)), idx(R(320, 384), R(448, 512))]]
    # S = 3, K = 200 slow: each split has 2 tiles -> one chunk each (the padding stays in place)
    plan = make_plan(200, fast=False, S=3, scheme="serial")
    assert [[c.tolist() for c in ch] for ch in plan.chunks] == [[v] for v in LITERAL[(200, 3)]]


def test_slow_unsplit_chunks_of_128():
    plan = make_plan(300, fast=False)
    got = [c.tolist() for c in plan.chunks[0]]
    assert got == [R(0, 128), R(128, 256), idx(R(256, 300), [PAD] * 84)]
    assert plan.S == 1 and plan.scheme == "none" and not plan.identity


def test_identity_and_padded_fast_plans():
    p = make_plan(1920, fast=True)
    assert p.identity and p.chunks[0][0].tolist() == R(0, 1920)
    p = make_plan(40, fast=True)
    assert not p.identity and p.chunks[0][0].tolist() == idx(R(0, 40), [PAD] * 8)
    p = make_plan(64, fast=True, S=1, scheme="pairwise")
    assert p.identity and p.scheme == "none"


def test_make_plan_rejects():
    with pytest.raises(ValueError):
        make_plan(64, fast=True, S=2)                     # S > 1 needs a scheme
    with pytest.raises(ValueError):
        make_plan(64, fast=True, S=0)
    with pytest.raises(ValueError):
        make_plan(-1, fast=True)


def test_split_index_matches_literal():
    for (K, S), v in LITERAL.items():
        assert split_index(K, S).tolist() == v


# --------------------------------------------------------------------------------------------------------------------
# reduction orders
def _p(*vals):
    return [torch.tensor([v], dtype=torch.float32) for v in vals]


def test_pairwise_s3_is_p2_plus_p0_then_p1():
    p0, p1, p2 = 1.0, 2.0 ** -24, -1.0
    pw = reduce_partials(_p(p0, p1, p2), "pairwise")
    se = reduce_partials(_p(p0, p1, p2), "serial")
    assert pw.item() == 2.0 ** -24          # (p2 + p0) + p1 = 0 + 2^-24
    assert se.item() == 0.0                 # (p0 + p1) + p2 = 1 + (-1): 1 + 2^-24 rounds to 1 (tie to even)


def test_pairwise_s5_slots():
    # h = 3: slot0 = p0 + p3, slot1 = p1 (1 + 3 = 4 = S - 1), slot2 = p2; result ((p4 + slot0) + slot1) + slot2
    v = [2.0 ** 24, 1.0, 1.0, 1.0, -(2.0 ** 24)]
    got = reduce_partials(_p(*v), "pairwise").item()
    f = np.float32
    want = float(((f(v[4]) + (f(v[0]) + f(v[3]))) + f(v[1])) + f(v[2]))
    assert got == want == 2.0
    assert reduce_partials(_p(*v), "serial").item() == 0.0     # (((2^24 + 1) + 1) + 1) - 2^24: 2^24+1 rounds down


def test_unknown_scheme():
    with pytest.raises(ValueError):
        reduce_partials(_p(1.0, 2.0), "tree")


# --------------------------------------------------------------------------------------------------------------------
# execute_plan composition against an independent construction from dot()
def _gauss(shape, g, scale=40.0):
    return (torch.randn(shape, generator=g) * scale).clamp(-448, 448).to(E4)


def _cols(x8, ix):
    """x8[:, ix] with -1 -> zero, as float8."""
    v = x8.float()
    out = torch.zeros(x8.shape[0], len(ix))
    for j, k in enumerate(ix):
        if k >= 0:
            out[:, j] = v[:, k]
    return out.to(E4)


def _chain(a8, b8, ix):
    A, B = _cols(a8, ix), _cols(b8, ix)
    M, N = A.shape[0], B.shape[0]
    return dot(A[:, None, :].expand(M, N, -1), B[None, :, :].expand(M, N, -1))


@pytest.mark.parametrize("K,S", sorted(LITERAL))
@pytest.mark.parametrize("scheme", ["serial", "pairwise"])
def test_execute_split_matches_hand_composition(K, S, scheme):
    g = torch.Generator().manual_seed(K * 10 + S)
    a8, b8 = _gauss((6, K), g), _gauss((5, K), g)
    parts = [_chain(a8, b8, v) for v in LITERAL[(K, S)]]
    want = reduce_partials(parts, scheme)
    plan = make_plan(K, fast=True, S=S, scheme=scheme)
    fn = REFERENCE.chain_fn(_req())
    got = execute_plan(plan, a8.view(torch.uint8), b8.view(torch.uint8), fmt_a="e4m3", fmt_b="e4m3", chain_fn=fn)
    assert_bitwise_equal(got, want)


def test_execute_slow_split_matches_hand_composition():
    g = torch.Generator().manual_seed(77)
    K = 512
    a8, b8 = _gauss((4, K), g), _gauss((3, K), g)
    chunks = {0: [idx(R(0, 64), R(128, 192)), idx(R(256, 320), R(384, 448))],
              1: [idx(R(64, 128), R(192, 256)), idx(R(320, 384), R(448, 512))]}
    parts = []
    for s in (0, 1):
        acc = torch.zeros(4, 3)
        for v in chunks[s]:
            acc = acc + _chain(a8, b8, v)         # fp32, serially from +0
        parts.append(acc)
    want = reduce_partials(parts, "pairwise")
    got = accumulate(a8, b8, use_fast_accum=False, splitk=(2, "pairwise"), backend="reference")
    assert_bitwise_equal(got, want)


def _req():
    from mmaemu.backends import ChainRequest
    return ChainRequest(device=torch.device("cpu"), fmt_a="e4m3", fmt_b="e4m3", preset="ada")


def test_identity_plan_does_not_copy():
    g = torch.Generator().manual_seed(5)
    a = _gauss((8, 64), g).view(torch.uint8)
    b = _gauss((4, 64), g).view(torch.uint8)
    seen = []

    def spy(x, y):
        seen.append((x.data_ptr(), y.data_ptr()))
        return torch.zeros(x.shape[0], y.shape[0])
    execute_plan(make_plan(64, fast=True), a, b, fmt_a="e4m3", fmt_b="e4m3", chain_fn=spy)
    assert seen == [(a.data_ptr(), b.data_ptr())]


def test_every_chain_starts_from_plus_zero_and_slow_adds_serially():
    """execute_plan calls chain_fn once per chunk and combines exactly as documented (spy chain_fn returning
    constants): fast S=3 pairwise -> (p2 + p0) + p1; slow unsplit K=256 -> (+0 + c0) + c1."""
    a = torch.zeros(1, 200, dtype=torch.uint8)
    vals = iter([1.0, 2.0 ** -24, -1.0])
    calls = []

    def spy(x, y):
        calls.append(x.shape[1])
        return torch.full((1, 1), next(vals))
    out = execute_plan(make_plan(200, fast=True, S=3, scheme="pairwise"), a, a, fmt_a="e4m3", fmt_b="e4m3",
                       chain_fn=spy)
    assert calls == [128, 128, 128] and out.item() == 2.0 ** -24
    vals2 = iter([-0.0, -0.0])
    out = execute_plan(make_plan(256, fast=False), torch.zeros(1, 256, dtype=torch.uint8),
                       torch.zeros(1, 256, dtype=torch.uint8), fmt_a="e4m3", fmt_b="e4m3",
                       chain_fn=lambda x, y: torch.full((1, 1), next(vals2)))
    assert out.view(torch.int32).item() == 0          # +0 + (-0) + (-0) = +0


def test_execute_edge_cases():
    fn = REFERENCE.chain_fn(_req())
    z = execute_plan(make_plan(0, fast=False), torch.zeros(3, 0, dtype=torch.uint8),
                     torch.zeros(2, 0, dtype=torch.uint8), fmt_a="e4m3", fmt_b="e4m3", chain_fn=fn)
    assert z.shape == (3, 2) and z.view(torch.int32).abs().sum().item() == 0
    e = execute_plan(make_plan(32, fast=True), torch.zeros(0, 32, dtype=torch.uint8),
                     torch.zeros(2, 32, dtype=torch.uint8), fmt_a="e4m3", fmt_b="e4m3", chain_fn=fn)
    assert e.shape == (0, 2)
    with pytest.raises(ValueError, match="plan is for K"):
        execute_plan(make_plan(32, fast=True), torch.zeros(1, 48, dtype=torch.uint8),
                     torch.zeros(1, 48, dtype=torch.uint8), fmt_a="e4m3", fmt_b="e4m3", chain_fn=fn)


def test_nonfinite_patch_after_combination():
    g = torch.Generator().manual_seed(9)
    a = _gauss((4, 200), g).view(torch.uint8).clone()
    b = _gauss((3, 200), g).view(torch.uint8).clone()
    a[1, 150] = 0x7F                                     # NaN in row 1 (lands in split 2's second tile)
    out = execute_plan(make_plan(200, fast=True, S=3, scheme="pairwise"), a, b, fmt_a="e4m3", fmt_b="e4m3",
                       chain_fn=REFERENCE.chain_fn(_req()))
    bits = out.view(torch.int32)
    assert (bits[1] == 0x7FFFFFFF).all() and torch.isfinite(out[[0, 2, 3]]).all()
    clean = a.clone()
    clean[1, 150] = 0
    ref = execute_plan(make_plan(200, fast=True, S=3, scheme="pairwise"), clean, b, fmt_a="e4m3", fmt_b="e4m3",
                       chain_fn=REFERENCE.chain_fn(_req()))
    assert_bitwise_equal(out[[0, 2, 3]], ref[[0, 2, 3]])


# --------------------------------------------------------------------------------------------------------------------
# truth: the recorded project cases (formerly a strict xfail), 16 outputs incl. the 8 split ones at S = 8/8/8/12
@pytest.mark.truth
def test_truth_recorded_project_cases():
    from test_epilogue import replay_gemm_output, truth_dir, truth_loader
    loader = truth_loader()
    rec = loader.load("misc.tcemu_tests_recorded_hw", root=truth_dir())
    seen_S, n = [], 0
    for case in rec.cases:
        ins = case.inputs()
        for o in case.outputs:
            kind, val = replay_gemm_output(ins, o)
            assert kind == "bits", (case.id, o.name, kind, val)
            hw = o.bits()
            mism = int((val.reshape(hw.shape) != hw).sum())
            assert mism == 0, (case.id, o.name, o.params.get("splitk"), mism)
            sk = o.params["splitk"]
            seen_S.append(int(sk.get("S", 1)))
            n += hw.size
    assert sorted(s for s in seen_S if s > 1) == [8, 8, 8, 8, 8, 8, 12, 12]
    print(f"misc.tcemu_tests_recorded_hw: {len(seen_S)} outputs, {n} values bitwise")
