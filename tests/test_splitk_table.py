"""Split-K resolution (docs/splitk_tables.md) against the exported reference-GPU table.

Markers: truth (needs MMAEMU_TRUTH_DIR: hardware-captured splits of the truth gemm outputs, export reproducibility),
profile_device (needs the RTX 4060 profile device: live heuristic re-query of 500 random rows)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from mmaemu.platform import profiles, splitk_table
from mmaemu.platform.profiles import get_profile
from mmaemu.platform.splitk_table import (SplitDecision, SplitK, SplitKUnknownError, UnverifiedSplitWarning,
                                          load_table, resolve_splitk)

REPO = Path(__file__).resolve().parents[1]
DATA = Path(splitk_table.__file__).resolve().parent / "data"
TID = "heur_rtx4060_lt12.8.3.14"
E4, E5 = "float8_e4m3fn", "float8_e5m2"
EVO = [(1920, 5760), (4096, 12288), (8192, 24576)]


@pytest.fixture(autouse=True)
def no_live_query(monkeypatch):
    """Resolution steps 3/5/6 are tested without the live query (step 4), which would fire on the RTX 4060."""
    monkeypatch.setattr(profiles.Profile, "matches_device", lambda self, device=None: False)
    monkeypatch.delenv("MMAEMU_SPLITK_STRICT", raising=False)


def rs(M, N, K, out="float32", fast=True, fa=E4, fb=E4, **kw):
    return resolve_splitk(M, N, K, a_dtype=fa, b_dtype=fb, out_dtype=out, use_fast_accum=fast, **kw)


# ---- package data ------------------------------------------------------------------------------------------------
def test_table_is_sha256_checked_and_small(tmp_path):
    meta = json.loads((DATA / f"{TID}.json").read_text())
    assert hashlib.sha256((DATA / f"{TID}.npz").read_bytes()).hexdigest() == meta["npz_sha256"]
    total = sum(p.stat().st_size for p in DATA.iterdir())
    assert total <= 3 * 2 ** 20, total
    for p in DATA.iterdir():
        shutil.copy(p, tmp_path / p.name)
    b = bytearray((tmp_path / f"{TID}.npz").read_bytes())
    b[-100] ^= 1
    (tmp_path / f"{TID}.npz").write_bytes(bytes(b))
    with pytest.raises(RuntimeError, match="sha256"):
        splitk_table.SplitTable(TID, tmp_path)


def test_table_facts_and_align_invariance():
    meta = json.loads((DATA / f"{TID}.json").read_text())
    ai = meta["align_invariance"]
    assert ai["differ"] == 0 and ai["pairs"] == 1422624
    assert ai["per_source"]["heur.lt_heuristic_12_8_3_14"] == {"pairs": 419424, "differ": 0}
    src = {s["id"]: s["sha256"] for s in meta["sources"]}
    assert src["heur.lt_heuristic_12_8_3_14"] == meta["truth_manifest_sha256"] == \
        "992aa95cd5f0d351809128e928047fddbde8c188686b64d67636fc492f76c3c9"
    assert meta["rows"] == 1228448 and "align" not in meta["columns"]
    t = load_table(TID)
    # workspace 0 never splits; S values of the truth table (+ R2/R2x)
    for st in meta["settings"]:
        if st["ws"] == 0:
            assert st["split_rows"] == 0
    S = sorted({s for st in meta["settings"] for s in st["S_values"]})
    assert {2, 3, 8, 12, 24, 29} <= set(S)
    assert t.lookup(16, 128, 8192, "e4m3", "e4m3", "float32", True, 1 << 20)["splitK"] == 8


def test_every_table_combo_is_hardware_verified():
    vc = json.loads((DATA / "verified_split_combos.json").read_text())
    assert vc["n_table_combos"] == 486 and vc["n_verified_table_combos"] == 486
    assert vc["unverified_table_combos"] == []
    s29 = [c for c in vc["combos"] if c["S"] == 29]
    assert len(s29) == 3 and all(c["verified"] for c in s29)
    assert all(any(r.startswith("rtx4060_2026-10:r4.") for r in c["recordings"]) for c in s29)


# ---- resolution order --------------------------------------------------------------------------------------------
def test_explicit_forms():
    d = rs(16, 128, 8192, splitk=1)
    assert (d.S, d.scheme, d.source, d.verified) == (1, "none", "explicit", True)
    d = rs(16, 128, 8192, splitk=SplitK(8, "pairwise"))
    assert (d.S, d.scheme, d.source, d.verified) == (8, "pairwise", "explicit", True)
    d = rs(16, 128, 8192, out="bfloat16", fast=False, splitk=(12, "pairwise"))
    assert (d.S, d.scheme, d.source) == (12, "pairwise", "explicit") and d.splitk == SplitK(12, "pairwise")
    with pytest.raises(ValueError, match="scheme"):
        rs(16, 128, 8192, splitk=8)
    with pytest.raises(ValueError):
        SplitK(2, "none")
    with pytest.raises(ValueError):
        SplitK(1, "serial")
    with pytest.raises(TypeError):
        rs(16, 128, 8192, splitk="8")


def test_project_cases_8_8_8_12():
    """The 4 recorded split keys of tcemu tests/data/hw (truth misc.tcemu_tests_recorded_hw), 16x128x8192."""
    got = [(d.S, d.scheme, d.source) for d in (rs(16, 128, 8192, "float32", True), rs(16, 128, 8192, "bfloat16", True),
                                               rs(16, 128, 8192, "float32", False),
                                               rs(16, 128, 8192, "bfloat16", False))]
    assert got == [(8, "pairwise", "table")] * 3 + [(12, "pairwise", "table")]
    for fa in (E4, E5):        # the e5m2 x e4m3 variant of the same recording splits identically
        assert rs(16, 128, 8192, "bfloat16", False, fa=fa).S == 12


def test_workspace0_never_splits():
    d = rs(16, 128, 8192, profile="ada-rtx4060-ws0")
    assert (d.S, d.scheme, d.source, d.verified) == (1, "none", "workspace0", True)
    d = rs(17, 100, 4096, profile="ada-rtx4060-ws0")            # not even in the table
    assert d.source == "workspace0"


@pytest.mark.parametrize("profile", ["ada-rtx4060-torch", "ada-rtx4060-te"])
def test_evo2_shapes_are_unsplit_and_never_raise(profile):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnverifiedSplitWarning)
        for K, N in EVO:
            for fast in (True, False):
                for M in range(16, 16385, 16):
                    d = rs(M, N, K, "bfloat16", fast, profile=profile, strict=False)
                    assert d.S == 1 and d.source in ("table", "envelope"), (M, N, K, d)
                    if M <= 8192:
                        assert d.source == "table"
                for M in range(1, 257):                          # R3: every integer M from the table
                    d = rs(M, N, K, "bfloat16", fast, profile=profile)
                    assert (d.S, d.source) == (1, "table"), (M, N, K, d)


def test_evo2_small_m_finding_is_in_the_table():
    """R3 finding: (8192, 24576) splits at M = 65..74 with fp16 or fp8 output, fast accumulation, 32 MiB workspace;
    never with fp32 / bf16 output or at 1 / 4 MiB."""
    p32 = dataclasses.replace(get_profile(), name="test-32mib", workspace_bytes=32 << 20)
    for M in range(65, 75):
        d = rs(M, 24576, 8192, "float16", True, profile=p32)
        assert (d.S, d.scheme, d.source) == (14, "serial", "table")
        assert rs(M, 24576, 8192, E4, True, profile=p32).S == 14
        assert rs(M, 24576, 8192, E5, True, fa=E4, fb=E5, profile=p32).S == 14
        assert rs(M, 24576, 8192, "float32", True, profile=p32).S == 1
        assert rs(M, 24576, 8192, "float16", True, profile="ada-rtx4060-te").S == 1
    assert rs(64, 24576, 8192, "float16", True, profile=p32).S == 1
    assert rs(70, 24576, 8192, "bfloat16", True, profile=p32).S == 1


def test_unknown_in_envelope_shape_raises_with_three_ways_out():
    # 1 MiB, fp32 out, fast, e4m3 x e4m3: envelope max split M*N 100,352, min split K 1,824; N = 100 is not a grid N
    assert load_table(TID).lookup(17, 100, 4096, "e4m3", "e4m3", "float32", True, 1 << 20) is None
    with pytest.raises(SplitKUnknownError) as ei:
        rs(17, 100, 4096)
    msg = str(ei.value)
    assert "splitk=1" in msg and "splitk=(S" in msg and "tools/record_heuristics.py" in msg
    assert "100352" in msg and "1824" in msg
    assert isinstance(ei.value, LookupError)


def test_envelope_inference_and_strict_mode(monkeypatch):
    # outside the envelope (M*N too large) and not in the table
    assert load_table(TID).lookup(4000, 4000, 4096, "e4m3", "e4m3", "float32", True, 1 << 20) is None
    d = rs(4000, 4000, 4096)
    assert (d.S, d.source) == (1, "envelope") and "inference" in d.note
    d = rs(17, 100, 1000)                                       # K below the smallest split K (1,824)
    assert (d.S, d.source) == (1, "envelope")
    with pytest.raises(SplitKUnknownError, match="strict"):
        rs(4000, 4000, 4096, strict=True)
    monkeypatch.setenv("MMAEMU_SPLITK_STRICT", "1")
    with pytest.raises(SplitKUnknownError):
        rs(4000, 4000, 4096)
    assert rs(16, 128, 8192).source == "table"                  # strict never blocks a table hit


def test_profiles_without_table_always_raise_on_auto():
    for name in ("hopper-experimental", "b200-mma-sync-experimental"):
        for (M, N, K) in ((16, 128, 8192), (16, 5760, 1920), (1, 16, 16), (4096, 4096, 4096)):
            with pytest.raises(SplitKUnknownError, match="no split-K table"):
                rs(M, N, K, profile=name)
            assert rs(M, N, K, profile=name, splitk=1).source == "explicit"


def test_zero_size_and_default_out_dtype():
    d = rs(0, 128, 8192)
    assert (d.S, d.source) == (1, "envelope") and "zero-size" in d.note
    assert rs(16, 16, 0).S == 1
    a = resolve_splitk(16, 128, 8192, a_dtype=E4, b_dtype=E4, out_dtype=None, use_fast_accum=True)
    b = rs(16, 128, 8192, out=E4)
    assert (a.S, a.scheme) == (b.S, b.scheme)
    with pytest.raises(TypeError):
        rs(16, 128, 8192, fa="bfloat16")


def test_fp8_output_rows_from_r2():
    """R2 rows: fp8 outputs resolve from the table (the 224x176x80 / 176x224x80 e4m3-output shapes are unsplit)."""
    for M, N in ((224, 176), (176, 224)):
        for fa, fb in ((E4, E4), (E4, E5), (E5, E4)):
            assert rs(M, N, 80, E4, True, fa=fa, fb=fb).source == "table"
            assert rs(M, N, 80, E4, True, fa=fa, fb=fb).S == 1
    assert rs(16, 128, 8192, E5, True, fa=E4, fb=E5).source == "table"


def test_unverified_split_warning_on_the_te_profile():
    t = load_table(TID)
    hit = None
    for M in (16, 32, 48, 64):
        for N in (64, 96, 128, 192, 256, 384, 512):
            r = t.lookup(M, N, 8192, "e4m3", "e4m3", "bfloat16", True, 4 << 20)
            if r and r["splitK"] > 1:
                hit = (M, N)
                break
        if hit:
            break
    assert hit is not None
    splitk_table._WARNED.clear()
    with pytest.warns(UnverifiedSplitWarning, match="experimental"):
        d = rs(hit[0], hit[1], 8192, "bfloat16", True, profile="ada-rtx4060-te")
    assert d.S > 1 and d.verified
    with warnings.catch_warnings():                              # once per combination
        warnings.simplefilter("error", UnverifiedSplitWarning)
        rs(hit[0], hit[1], 8192, "bfloat16", True, profile="ada-rtx4060-te")
        d = rs(16, 128, 8192)                                    # validated profile, verified combination
        assert d.S == 8


def test_decisions_are_frozen_dataclasses():
    d = rs(16, 128, 8192)
    assert isinstance(d, SplitDecision) and d.table_id == TID
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.S = 2


# ---- truth: hardware-captured splits ------------------------------------------------------------------------------
HEUR_SRC = "cuBLASLt heuristic table heur.lt_heuristic_12_8_3_14"


def _is_equivalence(sk: dict) -> bool:
    """An S = 1 asserted by the audit for a single k-tile (K <= 64: the split is unobservable), not a captured
    config (same rule as the truth's build/apply_ltconfig.py is_equivalence)."""
    s = json.dumps(sk)
    return bool(sk.get("equivalent_for_K_le_64")) or "single k-tile" in s or '"K=32"' in s or "K <= 64" in s


def _truth_loader():
    td = os.environ.get("MMAEMU_TRUTH_DIR")
    if not td:
        pytest.skip("MMAEMU_TRUTH_DIR not set")
    sys.dont_write_bytecode = True
    if td not in sys.path:
        sys.path.append(td)
    from tcemu_ada_truth import loader
    return loader


@pytest.mark.truth
def test_resolve_equals_hardware_captured_split_for_every_in_table_truth_output():
    """Every truth gemm output (torch._scaled_mm or TE level, tensor-wise, std layout, no bias / SM target) whose
    split was captured on hardware (trace / profiler / recaptured config; not the 3 + 46 outputs filled from the
    table, not an S = 1 asserted for a single k-tile) and whose key is in the table: resolve_splitk(strict) gives the
    captured (S, scheme). GPU call shape = hw_shape, or call_M rows for the stacked c5 probes. Workspace = the
    recorded one (custom Profile objects for workspaces without a built-in profile)."""
    L = _truth_loader()
    base = get_profile("ada-rtx4060-torch")
    profs = {}
    n = agree = excluded_filled = not_in_table = skipped = equivalence = 0
    eq_cases = []
    bad = []
    for e in L.list_recordings():
        if e["kind"] != "gemm":
            continue
        rec = L.load(e["id"])
        for c in rec.cases:
            ins = c.spec["inputs"]
            for o in c.outputs:
                p = o.params
                if o.hw_error:
                    continue
                sk = p.get("splitk")
                level = p.get("level", rec.level)
                if not isinstance(sk, dict) or "S" not in sk or level not in ("torch._scaled_mm",
                                                                              "transformer_engine"):
                    skipped += 1
                    continue
                if str(sk.get("source", "")).startswith(HEUR_SRC):
                    excluded_filled += 1
                    continue
                if int(sk["S"]) > 1 and sk.get("scheme") not in ("serial", "pairwise"):
                    skipped += 1
                    continue
                if _is_equivalence(sk):
                    equivalence += 1
                    eq_cases.append((e["id"], c.id, o.name, p, ins))
                    continue
                if (p.get("layout", "std") not in ("std", None) or p.get("scaling", "tensorwise") != "tensorwise"
                        or (p.get("bias_input") or "bias") in ins or p.get("bias_dtype") or p.get("sm_target")):
                    skipped += 1
                    continue
                hs = p.get("hw_shape") or p.get("hw_call_shape")
                if isinstance(hs, dict):
                    M, N, K = hs["M"], hs["N"], hs["K"]
                elif hs:
                    M, N, K = hs
                else:
                    M, K = ins["a"]["shape"][-2:]
                    N = ins["b"]["shape"][0]
                if p.get("stacked_calls"):           # A stacks several GPU calls of call_M rows each (c5 probes)
                    M = int(p["call_M"])
                cb = p.get("cublas") or {}
                ws = next((v for v in (cb.get("workspace_bytes"), p.get("workspace_bytes"), p.get("ws"))
                           if v is not None), 4 << 20 if level == "transformer_engine" else 1 << 20)
                if ws not in profs:
                    profs[ws] = dataclasses.replace(base, name=f"test-ws{ws}", workspace_bytes=int(ws),
                                                    splitk_maturity="validated")
                try:
                    d = resolve_splitk(int(M), int(N), int(K), a_dtype=p.get("fmt_a", "e4m3"),
                                       b_dtype=p.get("fmt_b", "e4m3"), out_dtype=p["out_dtype"],
                                       use_fast_accum=bool(p.get("fast_accum", True)), profile=profs[ws], strict=True)
                except SplitKUnknownError:
                    not_in_table += 1
                    continue
                if d.source != "table":
                    not_in_table += 1
                    continue
                n += 1
                hw = (int(sk["S"]), sk.get("scheme") if int(sk["S"]) > 1 else "none")
                if (d.S, d.scheme) == hw:
                    agree += 1
                else:
                    bad.append((e["id"], c.id, o.name, (M, N, K), hw, (d.S, d.scheme)))
    # the single-k-tile S = 1 assertions, for the record: how many are in the table and how many the table splits
    eq_in, eq_split = 0, []
    for rid, cid, oname, p, ins in eq_cases:
        M, K = ins["a"]["shape"][-2:]
        N = ins["b"]["shape"][0]
        r = load_table(TID).lookup(int(M), int(N), int(K), p.get("fmt_a", "e4m3"), p.get("fmt_b", "e4m3"),
                                   p["out_dtype"], bool(p.get("fast_accum", True)), 1 << 20)
        if r is not None:
            eq_in += 1
            if r["splitK"] > 1:
                eq_split.append((rid, cid, oname, (M, N, K), r["splitK"]))
    print(f"\nS = 1 asserted for a single k-tile (K <= 64, split unobservable): {equivalence} outputs, {eq_in} in "
          f"the table, {len(eq_split)} where the table splits: {sorted({(x[0], x[3], x[4]) for x in eq_split})}")
    assert all(x[3][2] <= 64 for x in eq_split)
    print(f"truth gemm outputs in the table: {n}, agree {agree}; excluded (split filled from the table by "
          f"apply_ltconfig): {excluded_filled}; S = 1 asserted for a single k-tile (not captured): {equivalence}; "
          f"not in the table: {not_in_table}; other skipped (bias, row-wise, layout, SM target, no split info): "
          f"{skipped}")
    assert excluded_filled == 49
    assert n > 10000 and not bad, bad[:10]


@pytest.mark.truth
def test_export_is_reproducible_and_matches_the_truth_manifest():
    _truth_loader()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, str(REPO / "tools" / "export_splitk_table.py"), "--check"], env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "check: OK" in r.stdout


# ---- profile device: live re-query ---------------------------------------------------------------------------------
def _require_profile_device_run(request):
    """profile_device tests touch the GPU (driver + cuBLASLt): they run only when selected explicitly with
    -m profile_device (under $MMAEMU_GPU_LOCK), never as part of a plain CPU run."""
    if "profile_device" not in (request.config.getoption("markexpr") or ""):
        pytest.skip("GPU test: select with -m profile_device and run under gpu.sh")


@pytest.mark.profile_device
def test_live_requery_of_500_random_rows(monkeypatch, request):
    _require_profile_device_run(request)
    monkeypatch.undo()                                   # use the real matches_device
    p = get_profile("ada-rtx4060-torch")
    if not p.matches_device():
        pytest.skip("profile device unavailable")
    from mmaemu.platform import cublaslt
    t = load_table(TID)
    z = t._cols
    rng = np.random.default_rng(20261012)
    split = np.nonzero(z["splitK"] > 1)[0]
    idx = np.concatenate([rng.choice(len(z["M"]), 400, replace=False), rng.choice(split, 100, replace=False)])
    fmts, outs = {0: "e4m3", 1: "e5m2"}, {0: "float32", 1: "bfloat16", 2: "float16", 3: E4, 4: E5}
    bad = []
    for i in idx:
        r = cublaslt.heuristic(int(z["M"][i]), int(z["N"][i]), int(z["K"][i]), fmt_a=fmts[int(z["fa"][i])],
                               fmt_b=fmts[int(z["fb"][i])], out_dtype=outs[int(z["out"][i])], fast=bool(z["fast"][i]),
                               workspace_bytes=t.ws_values[int(z["ws"][i])])
        want = (int(z["algo"][i]), int(z["tile"][i]), int(z["splitK"][i]), int(z["custom"][i]))
        got = (r["algo"], r["tile"], r["splitK"], r["custom"])
        if got != want:
            bad.append((int(i), want, got))
    print(f"\n{len(idx)} rows re-queried live, {len(bad)} differ")
    assert not bad, bad[:10]


@pytest.mark.profile_device
def test_live_query_step_resolves_unknown_shapes(monkeypatch, request):
    _require_profile_device_run(request)
    monkeypatch.undo()
    p = get_profile("ada-rtx4060-torch")
    if not p.matches_device():
        pytest.skip("profile device unavailable")
    d = rs(17, 100, 4096)                                 # in the envelope, not in the table
    assert d.source == "live-query" and d.S >= 1
