"""Preset registry: maturity, ExperimentalPresetWarning policy, the 'blackwell' alias, refused names,
uncharacterised GEMM-level features, custom presets, corrected wording (docs/limitations.md)."""
from __future__ import annotations

import dataclasses
import re
import warnings
from pathlib import Path

import pytest
import torch

from mmaemu.errors import ExperimentalPresetWarning, UnsupportedFeatureError
from mmaemu.model import presets as P
from mmaemu.model.chain import dot
from mmaemu.model.plan import make_plan

PKG = Path(P.__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _fresh_warning_state():
    P._reset_warnings_for_tests()
    yield
    P._reset_warnings_for_tests()


def test_registry_contents():
    assert P.list_presets() == ["ada", "b200_mma_sync", "hopper"]
    ada, hop, b2 = (P.get_preset(n, warn=False) for n in ("ada", "hopper", "b200_mma_sync"))
    assert (ada.nfma, ada.frac, ada.frac_out, ada.min_exp, ada.late_c, ada.slow_chunk, ada.maturity) == \
        (16, 13, 13, -132, False, 128, "validated")
    assert (hop.nfma, hop.frac, hop.frac_out, hop.min_exp, hop.late_c, hop.slow_chunk, hop.maturity) == \
        (32, 13, 13, -132, False, 128, "validated")
    assert (b2.nfma, b2.frac, b2.frac_out, b2.min_exp, b2.late_c, b2.slow_chunk, b2.maturity) == \
        (32, 25, 23, -133, True, None, "experimental")
    for p in (ada, hop, b2):
        assert p.validated_scope and p.evidence
    assert "NOT a model of sm_120" in b2.validated_scope and "NOT B200 cuBLAS" in b2.validated_scope
    assert "H100 PCIe" in hop.validated_scope and "not a model of the sm_90 mma.sync" in hop.validated_scope
    assert P.split_schemes(ada) == ("serial", "pairwise") and P.split_schemes(hop) == ("contiguous",)
    assert P.split_schemes(b2) == () and P.split_schemes(dataclasses.replace(hop, maturity="custom")) is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        ada.frac = 14


@pytest.mark.parametrize("name", ["ada", "hopper"])
def test_validated_preset_never_warns(name):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        P.get_preset(name)
        P.get_preset(name)


@pytest.mark.parametrize("name", ["b200_mma_sync"])
def test_experimental_warns_once_per_process(name):
    with pytest.warns(ExperimentalPresetWarning, match=f"'{name}' is experimental"):
        P.get_preset(name)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        P.get_preset(name)                      # second lookup: silent


def test_blackwell_alias_warns_every_time():
    for _ in range(3):
        with pytest.warns(ExperimentalPresetWarning) as rec:
            p = P.get_preset("blackwell")
        assert p is P.get_preset("b200_mma_sync", warn=False)
        msg = str(rec[0].message)
        assert "NOT a model of sm_120" in msg and "NOT B200 cuBLAS" in msg and "b200_mma_sync" in msg


@pytest.mark.parametrize("name", ["sm_120", "rtx_blackwell"])
def test_sm120_names_are_refused(name):
    with pytest.raises(KeyError, match="sm120_bringup"):
        P.get_preset(name)


@pytest.mark.parametrize("bad", [0, -128, 64.0, True, "128"])
def test_slow_chunk_must_be_a_positive_int(bad):
    with pytest.raises(ValueError, match="slow_chunk"):
        P.Preset("x", nfma=16, frac=13, frac_out=13, min_exp=-132, late_c=False, slow_chunk=bad)


def test_unknown_preset():
    with pytest.raises(KeyError, match="unknown preset"):
        P.get_preset("volta")
    with pytest.raises(TypeError):
        P.get_preset(3)


@pytest.mark.parametrize("name", ["b200_mma_sync"])
def test_uncharacterised_gemm_level_raises(name):
    with pytest.raises(UnsupportedFeatureError, match="slow accumulation"):
        make_plan(256, fast=False, preset=name)
    with pytest.raises(UnsupportedFeatureError, match="split-K"):
        make_plan(256, fast=True, S=2, scheme="serial", preset=name)
    plan = make_plan(256, fast=True, preset=name)     # fast unsplit is allowed (experimental)
    assert plan.identity


def test_uncharacterised_slow_accumulation_raises_through_the_api():
    from mmaemu.api import accumulate, scaled_mm
    a = torch.ones(16, 32).to(torch.float8_e4m3fn)
    with pytest.raises(UnsupportedFeatureError):
        with pytest.warns(ExperimentalPresetWarning):
            accumulate(a, a, use_fast_accum=False, profile="b200-mma-sync-experimental", splitk=1)
    P._reset_warnings_for_tests()
    with pytest.raises(UnsupportedFeatureError):
        with pytest.warns(ExperimentalPresetWarning):
            scaled_mm(a, a.t(), torch.tensor(1.0), torch.tensor(1.0), out_dtype=torch.float32,
                      profile="b200-mma-sync-experimental", splitk=1)


def test_custom_preset():
    sm120_paper = P.Preset("sm120_paper_hypothesis", nfma=32, frac=25, frac_out=23, min_exp=-133, late_c=False,
                           slow_chunk=None, maturity="custom", validated_scope="paper model, no hardware evidence")
    a = torch.full((4, 64), 3.0).to(torch.float8_e4m3fn)
    with pytest.warns(ExperimentalPresetWarning, match="custom"):
        d = dot(a, a, preset=sm120_paper)
    assert d.dtype == torch.float32 and d.tolist() == [576.0] * 4
    assert P.get_preset(sm120_paper) is sm120_paper
    with pytest.raises(ValueError):
        P.Preset("bad", nfma=0, frac=13, frac_out=13, min_exp=-132, late_c=False, slow_chunk=None)


FORBIDDEN = [r"B200\s*/\s*RTX[- ]Blackwell", r"restarts every 64", r"bit-identical to torch\._scaled_mm",
             r"no split-K, as in cuBLAS", r"\bany GPU\b", r"\bguaranteed\b"]
NUMERICS_FILES = ["formats.py", "model/presets.py", "model/chain.py", "model/specials.py", "model/plan.py",
                  "model/epilogue.py", "api.py", "backends/reference.py"]


@pytest.mark.parametrize("rel", NUMERICS_FILES)
def test_corrected_wording(rel):
    text = (PKG / rel).read_text()
    for pat in FORBIDDEN:
        assert not re.search(pat, text), (rel, pat)
