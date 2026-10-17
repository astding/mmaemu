"""MATLAB Tensor Core toolbox validation vectors (BSD-2, Khattak & Mikaitis; tests/data/toolbox): 10 sets (5 GPUs x
E4M3/E5M2), 5000 recorded MMA outputs each, bitwise through mmaemu.model.dot.

Scope of what these vectors show (a tcemu audit finding): only ~30,000 distinct inputs (Ada and L40S files are byte
identical; every GPU shares a and b), |a|, |b| <= 4, c in (0, 1). Hopper vectors were recorded with c = 0, so they
cannot tell how the accumulator enters; the B200 vectors are mma.sync compiled into an HMMA rewrite and fit >= 137
different models. Passing them is necessary, not sufficient: the Ada model is validated by the RTX 4060 recordings
(tests/test_model_chain.py -m truth), the other presets stay experimental.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from mmaemu.errors import ExperimentalPresetWarning
from mmaemu.model import presets as P
from mmaemu.model.chain import dot
from mmaemu.testing import assert_bitwise_equal

DATA = Path(__file__).parent / "data" / "toolbox"
META = json.loads((DATA / "SOURCES.json").read_text())
Z = np.load(DATA / "vectors.npz", allow_pickle=False)
PRESET = {"Ada": "ada", "L40S": "ada", "H100": "hopper", "H200": "hopper", "B200": "b200_mma_sync"}


def _f32(key: str) -> torch.Tensor:
    return torch.from_numpy(Z[key].view(np.int32).copy()).view(torch.float32)


def _set(s: dict):
    a, b, c, d = (_f32(s[r]) for r in ("a", "b", "c", "d"))
    if s["gpu"] in ("H100", "H200"):
        c = torch.zeros_like(c)          # the toolbox validates Hopper FP8 (wgmma) with D = A*B (c = 0)
    return a.double(), b.double(), c, d


def test_data_integrity():
    assert META["format"] == "mmaemu-toolbox-vectors/1"
    assert hashlib.sha256((DATA / "vectors.npz").read_bytes()).hexdigest() == META["npz_sha256"]
    for k, v in META["arrays"].items():
        assert hashlib.sha256(np.ascontiguousarray(Z[k]).tobytes()).hexdigest() == v["sha256"], k
    assert len(META["sets"]) == 10 and len(META["sources"]) == 40
    assert {(s["gpu"], s["fmt"]) for s in META["sets"]} == {(g, f) for g in PRESET for f in ("e4m3", "e5m2")}
    assert (DATA / "LICENCE.MATLAB-tensor-core").read_text().startswith("BSD 2-Clause License")


@pytest.mark.parametrize("s", META["sets"], ids=lambda s: f"{s['gpu']}-{s['fmt']}")
def test_toolbox_set(s):
    a, b, c, d = _set(s)
    preset = PRESET[s["gpu"]]
    P._reset_warnings_for_tests()
    if preset in ("ada", "hopper"):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ExperimentalPresetWarning)
            got = dot(a, b, c, preset=preset, fmt=s["fmt"])
    else:
        with pytest.warns(ExperimentalPresetWarning, match="experimental"):
            got = dot(a, b, c, preset=preset, fmt=s["fmt"])
    assert got.shape == (5000,)
    assert_bitwise_equal(got, d, msg=f"{s['gpu']} {s['fmt']} with preset {preset}")


def test_float8_inputs_infer_the_format():
    s = next(x for x in META["sets"] if x["gpu"] == "Ada" and x["fmt"] == "e5m2")
    a, b, c, d = _set(s)
    got = dot(a.to(torch.float8_e5m2), b.to(torch.float8_e5m2), c)
    assert_bitwise_equal(got, d)


def test_blackwell_alias_always_warns():
    s = next(x for x in META["sets"] if x["gpu"] == "B200" and x["fmt"] == "e4m3")
    a, b, c, d = _set(s)
    for _ in range(2):
        with pytest.warns(ExperimentalPresetWarning, match="NOT a model of sm_120"):
            got = dot(a, b, c, preset="blackwell", fmt="e4m3")
        assert_bitwise_equal(got, d)


def test_presets_are_not_interchangeable():
    """Sanity: the vectors discriminate the three presets (each fails the other GPUs' sets)."""
    for s in META["sets"]:
        a, b, c, d = _set(s)
        for other in {"ada", "hopper", "b200_mma_sync"} - {PRESET[s["gpu"]]}:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ExperimentalPresetWarning)
                got = dot(a, b, c, preset=other, fmt=s["fmt"])
            assert bool((got.view(torch.int32) != d.view(torch.int32)).any()), (s["gpu"], s["fmt"], other)
