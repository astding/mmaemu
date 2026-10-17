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
from mmaemu.platform.splitk_table import (SplitDecision, SplitK, SplitKUnknownError, UnvalidatedEnvelopeWarning,
                                          UnverifiedSplitWarning, load_table, resolve_splitk)

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


ADA = "ada"  # most cases here are the RTX 4060's (its table, its live splits)


def rs(M, N, K, out="float32", fast=True, fa=E4, fb=E4, profile=ADA, **kw):
    return resolve_splitk(M, N, K, a_dtype=fa, b_dtype=fb, out_dtype=out, use_fast_accum=fast, profile=profile, **kw)


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
    for bad in ((2.7, "serial"), (True, "none")):                # not truncated to an int
        with pytest.raises(ValueError):
            rs(16, 128, 8192, splitk=bad)


def test_project_cases_8_8_8_12():
    """The 4 recorded split keys of tcemu tests/data/hw (truth misc.tcemu_tests_recorded_hw), 16x128x8192."""
    got = [(d.S, d.scheme, d.source) for d in (rs(16, 128, 8192, "float32", True), rs(16, 128, 8192, "bfloat16", True),
                                               rs(16, 128, 8192, "float32", False),
                                               rs(16, 128, 8192, "bfloat16", False))]
    assert got == [(8, "pairwise", "table")] * 3 + [(12, "pairwise", "table")]
    for fa in (E4, E5):        # the e5m2 x e4m3 variant of the same recording splits identically
        assert rs(16, 128, 8192, "bfloat16", False, fa=fa).S == 12


def test_workspace0_never_splits():
    d = rs(16, 128, 8192, profile="ada-ws0")
    assert (d.S, d.scheme, d.source, d.verified) == (1, "none", "workspace0", True)
    d = rs(17, 100, 4096, profile="ada-ws0")            # not even in the table
    assert d.source == "workspace0"


@pytest.mark.parametrize("profile", ["ada", "ada-te"])
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
    p32 = dataclasses.replace(get_profile(ADA), name="test-32mib", workspace_bytes=32 << 20)
    for M in range(65, 75):
        d = rs(M, 24576, 8192, "float16", True, profile=p32)
        assert (d.S, d.scheme, d.source) == (14, "serial", "table")
        assert rs(M, 24576, 8192, E4, True, profile=p32).S == 14
        assert rs(M, 24576, 8192, E5, True, fa=E4, fb=E5, profile=p32).S == 14
        assert rs(M, 24576, 8192, "float32", True, profile=p32).S == 1
        assert rs(M, 24576, 8192, "float16", True, profile="ada-te").S == 1
    assert rs(64, 24576, 8192, "float16", True, profile=p32).S == 1
    assert rs(70, 24576, 8192, "bfloat16", True, profile=p32).S == 1


def test_unknown_in_envelope_shape_raises_with_three_ways_out():
    # 1 MiB, fp32 out, fast, e4m3 x e4m3: largest split M*N 100,352, inference bound 200,704; N = 100 is not a grid N
    assert load_table(TID).lookup(17, 100, 4096, "e4m3", "e4m3", "float32", True, 1 << 20) is None
    with pytest.raises(SplitKUnknownError) as ei:
        rs(17, 100, 4096)
    msg = str(ei.value)
    assert "splitk=1" in msg and "splitk=(S" in msg and "tools/record_heuristics.py" in msg
    assert "100352" in msg and "200704" in msg
    assert isinstance(ei.value, LookupError)


def test_envelope_inference_and_strict_mode(monkeypatch):
    # M*N above twice the largest split M*N and not in the table
    assert load_table(TID).lookup(4000, 4000, 4096, "e4m3", "e4m3", "float32", True, 1 << 20) is None
    d = rs(4000, 4000, 4096)
    assert (d.S, d.source) == (1, "envelope") and "inference" in d.note and "200704" in d.note
    with pytest.raises(SplitKUnknownError, match="not above the inference bound"):
        rs(17, 100, 1000)                                       # a small K does not make S = 1
    with pytest.raises(SplitKUnknownError, match="strict"):
        rs(4000, 4000, 4096, strict=True)
    monkeypatch.setenv("MMAEMU_SPLITK_STRICT", "1")
    with pytest.raises(SplitKUnknownError):
        rs(4000, 4000, 4096)
    assert rs(16, 128, 8192).source == "table"                  # strict never blocks a table hit


# Shapes absent from the table that cuBLASLt splits on the RTX 4060 (live heuristic, sweeps in
# docs/validation_logs/2026-10-14/splitk-envelope): M*N up to 1.27 x the largest split M*N of the setting, or K below
# the smallest split K of the table. (profile, M, N, K, out, fast, fa, fb, live numSplitsK).
LIVE_SPLITS_OFF_TABLE = [
    ("ada", 128, 976, 4096, E5, False, E4, E5, 2),           # M*N 1.27 x 98,304
    ("ada", 64, 1552, 16384, "float32", True, E4, E5, 2),    # M*N 1.01 x 98,304
    ("ada", 128, 128, 2176, "float16", True, E4, E5, 3),     # K 2,176 < table's smallest split K 4,096
    ("ada-te", 1024, 960, 6144, "float32", True, E4, E5, 5),       # M*N 1.25 x 786,432
    ("ada-te", 256, 1024, 1760, "bfloat16", True, E4, E5, 2),      # K 1,760 < 1,920
]


@pytest.mark.parametrize("case", LIVE_SPLITS_OFF_TABLE, ids=lambda c: f"{c[0][12:]}-{c[1]}x{c[2]}x{c[3]}")
def test_shapes_that_split_live_are_not_inferred_unsplit(case):
    prof, M, N, K, out, fast, fa, fb, _ = case
    ws = get_profile(prof).workspace_bytes
    assert load_table(TID).lookup(M, N, K, splitk_table.FMTS[fa], splitk_table.FMTS[fb], out, fast, ws) is None
    with pytest.raises(SplitKUnknownError, match="not above the inference bound"):
        rs(M, N, K, out, fast, fa=fa, fb=fb, profile=prof)


def test_envelope_bound_is_twice_the_largest_split_mn():
    # 1 MiB, fp32 out, fast, e4m3 x e5m2: largest split M*N 98,304; K = 20,000 is not a grid K
    assert splitk_table.ENVELOPE_MARGIN == 2
    d = rs(64, 3088, 20000, fa=E4, fb=E5)                       # 197,632 > 196,608
    assert (d.S, d.source) == (1, "envelope")
    with pytest.raises(SplitKUnknownError):
        rs(64, 3072, 20000, fa=E4, fb=E5)                       # 196,608 = 2 x 98,304


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
    a = resolve_splitk(16, 128, 8192, a_dtype=E4, b_dtype=E4, out_dtype=None, use_fast_accum=True, profile=ADA)
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
        d = rs(hit[0], hit[1], 8192, "bfloat16", True, profile="ada-te")
    assert d.S > 1 and d.verified
    with warnings.catch_warnings():                              # once per combination
        warnings.simplefilter("error", UnverifiedSplitWarning)
        rs(hit[0], hit[1], 8192, "bfloat16", True, profile="ada-te")
        d = rs(16, 128, 8192)                                    # validated profile, verified combination
        assert d.S == 8


def test_dry_resolution_does_not_use_up_the_warning():
    """mmaemu.explain resolves with dry=True: it reports the warning every time, and the next real call still warns."""
    assert rs(16, 128, 8192, "bfloat16", True, profile="ada-te").S > 1
    splitk_table._WARNED.clear()
    for _ in range(2):
        with pytest.warns(UnverifiedSplitWarning, match="experimental"):
            rs(16, 128, 8192, "bfloat16", True, profile="ada-te", dry=True)
    with pytest.warns(UnverifiedSplitWarning, match="experimental"):
        rs(16, 128, 8192, "bfloat16", True, profile="ada-te")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnverifiedSplitWarning)
        rs(16, 128, 8192, "bfloat16", True, profile="ada-te")


def test_decisions_are_frozen_dataclasses():
    d = rs(16, 128, 8192)
    assert isinstance(d, SplitDecision) and d.table_id == TID
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.S = 2


# ---- the H100 PCIe table (docs/validation_logs/2026-10-15/h100-probe, 2026-10-16/h100-splitk-sweep) -------------
HID = "heur_h100pcie_lt12.8.3.14"
H = "hopper"
HTE = "hopper-te"
SWEEP = REPO / "docs/validation_logs/2026-10-16/h100-splitk-sweep"


def test_h100_table_facts():
    meta = json.loads((DATA / f"{HID}.json").read_text())
    assert hashlib.sha256((DATA / f"{HID}.npz").read_bytes()).hexdigest() == meta["npz_sha256"]
    assert meta["align_invariance"] == {"probe": {"pairs": 425160, "differ": 0}, "p3": {"pairs": 1605840, "differ": 0}}
    # the probe's grid (396,816 rows) and the sweep's model rows (p3: 1,498,784), 128,184 keys in both; 28 of those
    # are unsplit rows whose customOption differs (5 / 7), the probe's is kept
    assert (meta["rows"], meta["split_rows"], meta["dropped_status_nonzero"]) == (1767416, 36457, 135400)
    assert (meta["p3_rows"], meta["rows_in_both"]) == (1498784, 128184)
    assert meta["unsplit_customOption_differs"]["rows"] == 28
    assert all(k["probe"] == 5 and k["p3"] == 7 for k in meta["unsplit_customOption_differs"]["keys"])
    # the scheme rule lists the customOptions of replayed live split calls; p3 has split rows with 4 and 5
    assert meta["codes"]["custom -> scheme"] == {str(c): "contiguous" for c in range(4)}
    assert meta["split_customOptions"] == [0, 1, 2, 3, 4, 5]
    lay = meta["split_layouts"]
    assert (lay["ragged_K"]["verified"], lay["ragged_K"]["live_split_calls"]) == (True, 1040)
    assert (lay["empty_split"]["verified"], lay["empty_split"]["live_split_calls"]) == (False, 0)
    env = meta["envelope"]
    assert (env["inference"], env["kind"], env["K_min"], env["K_max"], env["margin"]) == (True, "k_grid", 1024,
                                                                                           4194304, 2)
    assert {w: v["status"] for w, v in env["validation"].items()} == {"1048576": "validated",
                                                                       "33554432": "unvalidated"}
    assert len(env["bounds"]) == 60 and {b["ws"] for b in env["bounds"]} == {1 << 20, 32 << 20}
    assert meta["verified_combos"].endswith(f"{HID[5:]}.json")
    z = load_table(HID)._cols
    sp = z["splitK"] > 1
    assert (z["K"][sp] % 128 == 0).all() and int(z["K"][sp].min()) == 5760
    assert set(np.unique(z["splitK"][sp]).tolist()) == {2, 3, 4, 5, 6} and (z["ws"][sp] > 0).all()


def test_scheme_rules_of_the_two_tables():
    t = load_table(HID)
    assert [t.scheme_of(c) for c in range(4)] == [("contiguous", None)] * 4
    s, cav = t.scheme_of(5)                                      # customOption 4..7 only on unsplit rows
    assert s == "contiguous" and "customOption 5 was never recorded" in cav
    a = load_table(TID)                                          # RTX 4060: 1 -> pairwise, every other -> serial
    assert [a.scheme_of(c)[0] for c in range(5)] == ["serial", "pairwise", "serial", "serial", "serial"]
    assert all(a.scheme_of(c)[1] is None for c in range(5))


def test_h100_table_has_no_split_row_for_an_evo2_fp8_projection():
    """Evo 2's FP8 GEMMs are its input projections (docs/evo2.md): no row of the H100 table splits them, at any M,
    workspace or setting (the probe's grid and the sweep's dense model rows, M 1..16,384)."""
    z = load_table(HID)._cols
    for K, N in EVO:
        m = (z["K"] == K) & (z["N"] == N)
        assert m.sum() > 10_000 and not (z["splitK"][m] > 1).any(), (K, N)


@pytest.mark.parametrize("profile", ["hopper", "hopper-te", "hopper-ws0"])
def test_h100_evo2_shapes_are_unsplit(profile):
    for K, N in EVO:
        for M in (1, 16, 255, 2048, 8192):
            for out in ("float32", "bfloat16"):
                for fast in (True, False):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        d = rs(M, N, K, out, fast, profile=profile)
                    assert (d.S, d.scheme, d.verified) == (1, "none", True)
                    assert d.source == ("workspace0" if profile.endswith("ws0") else "table")


def test_h100_split_rows_resolve_contiguous_silent_at_1mib_and_warn_on_te():
    """S > 1 is validated on the torch profile (no warning) and experimental on the TE profile."""
    t = load_table(HID)
    z = t._cols
    fmt = {0: E4, 1: E5}
    outs = {0: "float32", 1: "bfloat16", 2: "float16", 3: E4, 4: E5}
    splitk_table._WARNED.clear()
    for code, prof in ((1, H), (3, HTE)):
        # a customOption the scheme rule lists (0..3), so that only the profile's maturity can warn
        i = int(np.nonzero((z["splitK"] > 1) & (z["ws"] == code) & (z["fast"] == 1) & (z["custom"] <= 3))[0][0])
        row = {k: int(v[i]) for k, v in z.items()}
        args = (row["M"], row["N"], row["K"], outs[row["out"]], True)
        if prof == H:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                d = rs(*args, fa=fmt[row["fa"]], fb=fmt[row["fb"]], profile=prof)
        else:
            with pytest.warns(UnverifiedSplitWarning, match="are experimental"):
                d = rs(*args, fa=fmt[row["fa"]], fb=fmt[row["fb"]], profile=prof)
        assert (d.S, d.scheme, d.source, d.table_id, d.verified) == (row["splitK"], "contiguous", "table", HID, True)


def _fit_envelope():
    import importlib.util
    path = SWEEP / "scripts/analysis/fit_envelope.py"
    if not path.exists():
        pytest.skip("the sweep's validation log is not in this checkout")
    spec = importlib.util.spec_from_file_location("fit_envelope_h100", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_h100_envelope_is_the_rule_tested_on_p4():
    """The table's bounds are the sweep's envelope_rule.json byte for byte (the rule eval_sealed.json opened p4 with),
    and the resolver's lookup is the evaluation's: the smallest grid K >= K, 1,024 <= K <= 4,194,304."""
    F = _fit_envelope()
    env = json.loads((DATA / f"{HID}.json").read_text())["envelope"]
    raw = (REPO / env["rule"]).read_bytes()
    ev = json.loads((REPO / env["eval"]).read_text())
    assert hashlib.sha256(raw).hexdigest() == env["rule_sha256"] == ev["provenance"]["rule_sha256"]
    rule = json.loads(raw)
    assert [[g["B"] for g in s["grid"]] for s in rule["settings"]] == [b["B"] for b in env["bounds"]]
    assert env["K_grid"] == rule["rule"]["K_grid"] == F.K_GRID
    t, E = load_table(HID), F.Envelope(rule)
    g = np.asarray(F.K_GRID)
    Ks = np.unique(np.concatenate([g, g - 1, g + 1, [1, 1023, 4194305, 8 << 20],
                                   np.random.default_rng(0).integers(1, 5 << 20, 500)]))
    for b in env["bounds"]:
        cols = dict(K=Ks, ws=np.full(len(Ks), b["ws"]), out=np.full(len(Ks), F.OUTS.index(b["out"])),
                    fast=np.full(len(Ks), int(b["fast"])), fa=np.full(len(Ks), F.FMTS.index(b["fmt_a"])),
                    fb=np.full(len(Ks), F.FMTS.index(b["fmt_b"])))
        got = [t.k_bound(int(K), b["ws"], b["out"], b["fast"], b["fmt_a"], b["fmt_b"])[0] for K in Ks]
        assert [np.inf if x is None else x for x in got] == E.bounds(cols).tolist(), b


def test_h100_envelope_validated_at_1mib_and_warns_at_32mib():
    splitk_table._WARNED.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = rs(2048, 11008, 4096, "bfloat16", profile=H)                # not a table row
    assert (d.S, d.scheme, d.source, d.verified) == (1, "none", "envelope", True)
    assert "validated on held-out shapes (154,191 above the bound, 0 split" in d.note
    with pytest.warns(UnvalidatedEnvelopeWarning, match="did not pass its held-out test") as w:
        d = rs(2048, 11008, 4096, "bfloat16", profile=HTE)
    assert "1 of 77,518 held-out shapes" in str(w[0].message) and issubclass(UnvalidatedEnvelopeWarning,
                                                                             UnverifiedSplitWarning)
    assert (d.S, d.source, d.verified) == (1, "envelope", False) and "unvalidated" in d.note
    with warnings.catch_warnings():                                     # once per setting
        warnings.simplefilter("error")
        rs(1000, 5760, 1920, "bfloat16", profile=HTE)
    with pytest.warns(UnvalidatedEnvelopeWarning):                      # dry (explain) warns every time
        rs(1000, 5760, 1920, "bfloat16", profile=HTE, dry=True)
    # the validation goes by workspace, not by profile name: torch's profile at 32 MiB is unvalidated too
    p32 = dataclasses.replace(get_profile(H), name="h100-torch-ws32", workspace_bytes=32 << 20)
    with pytest.warns(UnvalidatedEnvelopeWarning):
        assert not rs(2048, 11008, 4096, "bfloat16", profile=p32).verified


def test_h100_envelope_claims_nothing_outside_its_bounds():
    t = load_table(HID)
    B, _ = t.k_bound(8192, 1 << 20, "float32", True, "e4m3", "e4m3")
    assert B == 256642                                                  # 2 x the p1 level above 122,880
    with pytest.raises(SplitKUnknownError, match=f"not above the envelope bound {B:,} at grid K = 8,192"):
        rs(1, B, 8192, profile=H)                                       # M*N = B: strictly above only
    assert rs(1, B + 1, 8192, profile=H).source == "envelope"
    assert t.k_bound(8000, 1 << 20, "float32", True, "e4m3", "e4m3")[0] == B     # K rounds up to the grid
    for M, N, K, match in ((8192, 8192, 1000, "outside the envelope's K range"),
                           (8192, 8192, 4194305, "outside the envelope's K range"),
                           (100, 100, 5000, "not above the envelope bound 15,360 at grid K = 5,120")):
        with pytest.raises(SplitKUnknownError, match=match) as e:
            rs(M, N, K, profile=H)
        assert "(S, 'contiguous')" in str(e.value) and "NVIDIA H100 PCIe" in str(e.value)
    with pytest.raises(SplitKUnknownError, match="strict mode"):
        rs(2048, 11008, 4096, "bfloat16", profile=H, strict=True)
    p4m = dataclasses.replace(get_profile(H), name="h100-torch-ws4", workspace_bytes=4 << 20)
    with pytest.raises(SplitKUnknownError, match="no bound at workspace 4194304 B"):
        rs(2048, 11008, 4096, "bfloat16", profile=p4m)                  # table rows at 4 MiB, no envelope
    assert rs(8192, 8192, 1000, profile="hopper-ws0").S == 1


def test_h100_te_envelope_failure_of_the_sealed_test():
    """The one held-out shape above the 32 MiB bound that split on the H100 (S = 2): the envelope says S = 1, warns
    and marks it unverified."""
    val = json.loads((DATA / f"{HID}.json").read_text())["envelope"]["validation"]
    assert val["1048576"]["failures"] == [] and val["1048576"]["tested"] == 154191
    (f,) = val["33554432"]["failures"]
    assert (f["M"], f["N"], f["K"], f["splitK"], f["custom"], f["out"], f["fast"]) == (69, 29456, 19872, 2, 5,
                                                                                       "float32", False)
    splitk_table._WARNED.clear()
    with pytest.warns(UnvalidatedEnvelopeWarning):
        d = rs(f["M"], f["N"], f["K"], f["out"], f["fast"], fa=f["fmt_a"], fb=f["fmt_b"], profile=HTE)
    assert (d.S, d.source, d.verified) == (1, "envelope", False)


def test_h100_decode_and_model_shapes_from_the_p3_rows():
    """Small M (decode) of the sweep's 9 models resolve from the table, split or not (no envelope there)."""
    splits = set()
    for M in (1, 2, 7, 16, 100, 256):
        for K, N in ((4096, 4096), (11008, 4096), (4096, 22016), (28672, 8192)):
            with warnings.catch_warnings():
                warnings.simplefilter("error")                                   # S > 1 is validated at 1 MiB
                d = rs(M, N, K, "bfloat16", profile=H)
            assert d.source == "table", (M, N, K, d)
            if d.S > 1:
                splits.add((M, N, K, d.S))
    assert splits and all(K == 28672 for _M, _N, K, _S in splits), splits


def test_h100_explicit_splits_and_the_replayed_layouts():
    splitk_table._WARNED.clear()
    # a ragged last k-tile: 1,040 replayed live split calls of the sweep cover it
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = rs(64, 64, 6000, profile=H, splitk=(2, "contiguous"))
    assert d.verified and "ragged" not in d.note
    with pytest.warns(UnverifiedSplitWarning, match="are experimental"):          # the TE profile's S > 1
        d = rs(64, 64, 6000, profile=HTE, splitk=(2, "contiguous"))
    assert d.verified and "ragged" not in d.note
    with pytest.warns(UnverifiedSplitWarning, match="not a multiple of 128"):    # no table, no replayed layouts
        d = rs(64, 64, 6000, profile="hopper-experimental", splitk=(2, "contiguous"))
    assert not d.verified and "ragged" in d.note
    with pytest.warns(UnverifiedSplitWarning, match="without k-tiles"):
        d = rs(64, 64, 768, profile=H, splitk=(4, "contiguous"))
    assert not d.verified
    with pytest.raises(ValueError, match=r"\(4, 'contiguous'\)"):
        rs(64, 64, 6000, profile=H, splitk=4)
    with pytest.raises(ValueError, match="'serial', 'pairwise' or 'contiguous'"):
        SplitK(2, "tree")


def test_verified_combos_file_must_belong_to_the_table(tmp_path):
    for p in DATA.iterdir():
        shutil.copy(p, tmp_path / p.name)
    meta = json.loads((tmp_path / f"{HID}.json").read_text())
    meta["verified_combos"] = "verified_split_combos.json"      # the RTX 4060 table's
    (tmp_path / f"{HID}.json").write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="is for table"):
        splitk_table.SplitTable(HID, tmp_path)


def test_h100_verified_combos_come_from_the_replays():
    vc = json.loads((DATA / f"verified_split_combos_{HID[5:]}.json").read_text())
    assert [Path(r["path"]).name for r in vc["replays"]] == ["mmaemu_replay.json", "mmaemu_replay_p5.json"]
    for r in vc["replays"]:
        if (REPO / r["path"]).exists():
            assert r["sha256"] == hashlib.sha256((REPO / r["path"]).read_bytes()).hexdigest()
    # every split combination of the table has a live split call that mmaemu reproduced (the probe's 1,049 and the
    # sweep's 2,416 split calls); 444 combinations in all
    assert (vc["n_table_combos"], vc["n_verified_table_combos"], vc["n_combos"], vc["n_verified_combos"]) == (
        140, 140, 444, 444)
    assert all(c["scheme"] == "contiguous" and c["verified"] == (c["n_recordings"] > 0) for c in vc["combos"])
    # so no split row of the torch profile warns: every 1 MiB split row has a replayed combination and a listed
    # customOption (0..3)
    t = load_table(HID)
    z = t._cols
    outs = {0: "float32", 1: "bfloat16", 2: "float16", 3: E4, 4: E5}
    rows = np.nonzero((z["splitK"] > 1) & (z["ws"] == 1))[0]
    assert set(z["custom"][rows].tolist()) <= {0, 1, 2, 3}
    splitk_table._WARNED.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for i in rows[::50]:
            row = {k: int(v[i]) for k, v in z.items()}
            d = rs(row["M"], row["N"], row["K"], outs[row["out"]], bool(row["fast"]), fa=outs[3 + row["fa"]],
                   fb=outs[3 + row["fb"]], profile=H)
            assert (d.S, d.scheme, d.verified) == (row["splitK"], "contiguous", True)
    assert (True, 32, "contiguous", "float32", "e4m3xe4m3") not in t.verified           # live splits: S <= 24
    with pytest.warns(UnverifiedSplitWarning, match="never replayed on hardware"):
        d = rs(64, 64, 65536, profile=H, splitk=(32, "contiguous"))
    assert not d.verified


def test_h100_resolver_check_is_of_the_shipped_table():
    """results/analysis/resolver_check.json of the sweep (tools/check_splitk_resolution_h100.py) was made with the
    package data of this checkout and found only the sealed test's failure."""
    p = SWEEP / "results/analysis/resolver_check.json"
    if not p.exists():
        pytest.skip("results/analysis/resolver_check.json of the sweep is not in this checkout")
    r = json.loads(p.read_text())
    for n, sha in r["provenance"]["table"].items():
        assert hashlib.sha256((DATA / n).read_bytes()).hexdigest() == sha, n
    assert r["passed"] and not r["provenance"]["dirty"]
    assert sum(g["differ"] for part in r["parts"].values() for g in part.values()) == 1


@pytest.mark.slow
def test_h100_resolution_agrees_with_every_h100_decision(tmp_path):
    """resolve_splitk over every status-0 row of the sweep (p1-p4) and every live call of the probe and the sweep:
    only the sealed test's failure differs from the H100."""
    if not SWEEP.exists():
        pytest.skip("the sweep's validation log is not in this checkout")
    out = tmp_path / "check.json"
    r = subprocess.run([sys.executable, str(REPO / "tools/check_splitk_resolution_h100.py"), "--jobs",
                        str(min(16, os.cpu_count() or 1)), "--out", str(out)], capture_output=True, text=True,
                       timeout=3600, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    rep = json.loads(out.read_text())
    assert rep["passed"] and rep["parts"]["live"]["probe+p5"]["differ"] == 0


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
    base = get_profile("ada")
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
    p = get_profile("ada")
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
    p = get_profile("ada")
    if not p.matches_device():
        pytest.skip("profile device unavailable")
    d = rs(17, 100, 4096)                                 # in the envelope, not in the table
    assert d.source == "live-query" and d.S >= 1


@pytest.mark.profile_device
def test_live_heuristic_never_splits_where_the_envelope_infers_s1(monkeypatch, request):
    """Step 5 (S = 1 above twice the largest split M*N) against the live heuristic: shapes with M*N = 2.0-4.0 x the
    largest split M*N of every setting of both table profiles, K = 1,024 to 4,194,304 (the RTX 4060 splits up to
    1.27 x)."""
    _require_profile_device_run(request)
    monkeypatch.undo()
    from mmaemu.platform import cublaslt
    if not get_profile("ada").matches_device():
        pytest.skip("profile device unavailable")
    t, bad, n = load_table(TID), [], 0
    for prof in ("ada", "ada-te"):
        p = get_profile(prof)
        for st in json.loads((DATA / f"{TID}.json").read_text())["settings"]:
            if st["ws"] != p.workspace_bytes or not st["split_rows"]:
                continue
            mx = st["max_split_MN"]
            for M in (1, 16, 64, 128, 256, 1024, 4096):
                for f in (2.0, 2.25, 2.5, 3.0, 4.0):
                    N = max(16, -(-int(mx * f / M) // 16) * 16)
                    if M * N <= 2 * mx:
                        N += 16
                    outside, _, _ = splitk_table._envelope(t, M, N, p.workspace_bytes, st["out"], st["fast"],
                                                           st["fmt_a"], st["fmt_b"])
                    assert outside
                    for K in (1024, 2048, 4096, 16384, 262144, 4194304):
                        r = cublaslt.heuristic(M, N, K, fmt_a=st["fmt_a"], fmt_b=st["fmt_b"], out_dtype=st["out"],
                                               fast=st["fast"], workspace_bytes=p.workspace_bytes, align=p.align)
                        n += 1
                        if r["status"] != 0 or r["splitK"] != 1:
                            bad.append((prof, st["out"], st["fast"], st["fmt_a"], st["fmt_b"], M, N, K, r))
    print(f"\n{n} live queries above the inference bound, {len(bad)} not S = 1")
    assert n > 10000 and not bad, bad[:10]
