"""Profiles: built-ins, default/env selection, rejects, matches_device (never cc alone)."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from mmaemu.platform import cublaslt, profiles
from mmaemu.platform.profiles import Profile, RejectRule, get_profile, list_profiles

DATA = Path(profiles.__file__).resolve().parent / "data"


def test_builtin_names_and_default(monkeypatch):
    assert list_profiles() == ["ada-rtx4060-torch", "ada-rtx4060-te", "ada-rtx4060-ws0", "hopper-experimental",
                               "b200-mma-sync-experimental"]
    monkeypatch.delenv("MMAEMU_PROFILE", raising=False)
    assert get_profile().name == "ada-rtx4060-torch"
    monkeypatch.setenv("MMAEMU_PROFILE", "ada-rtx4060-te")
    assert get_profile().name == "ada-rtx4060-te"
    p = get_profile("ada-rtx4060-ws0")
    assert get_profile(p) is p


def test_unknown_profile_names_the_runbook():
    with pytest.raises(KeyError, match="sm120_bringup"):
        get_profile("rtx5070ti")
    with pytest.raises(TypeError):
        get_profile(3)


def test_table_of_section_4_2():
    t, te, w0 = (get_profile(n) for n in ("ada-rtx4060-torch", "ada-rtx4060-te", "ada-rtx4060-ws0"))
    for p in (t, te, w0):
        assert (p.preset, p.gpu_name, p.capability, p.sm_count) == ("ada", "NVIDIA GeForce RTX 4060", (8, 9), 24)
        assert (p.cublaslt_version, p.torch_version) == ("12.8.3.14", "2.7.1+cu128")
        assert p.splitk_table == "heur_rtx4060_lt12.8.3.14" and p.maturity == "validated"
    assert (t.workspace_bytes, t.align, t.caller, t.splitk_maturity) == (1 << 20, None, "torch._scaled_mm",
                                                                        "validated")
    assert (te.workspace_bytes, te.align, te.caller, te.splitk_maturity) == (4 << 20, 256, "transformer_engine",
                                                                            "experimental")
    assert (w0.workspace_bytes, w0.splitk_maturity) == (0, "validated")
    for n, preset in (("hopper-experimental", "hopper"), ("b200-mma-sync-experimental", "b200_mma_sync")):
        p = get_profile(n)
        assert (p.preset, p.maturity, p.splitk_maturity, p.splitk_table) == (preset, "experimental", "none", None)
        assert p.matches_device() is False
    assert "NOT a model of sm_120" in get_profile("b200-mma-sync-experimental").description


def test_profiles_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        get_profile().workspace_bytes = 0


def test_rejects_e4m3_e4m3_to_e5m2_only():
    import torch
    p = get_profile()
    e4, e5 = torch.float8_e4m3fn, torch.float8_e5m2
    r = p.rejection(e4, e4, e5)
    assert isinstance(r, RejectRule) and "CUBLAS_STATUS_NOT_SUPPORTED" in r.message
    assert p.rejection("e4m3", "e4m3", "float8_e5m2") is r
    assert p.rejection(e4, e5, e5) is None and p.rejection(e5, e4, e5) is None
    for od in (torch.float32, torch.bfloat16, torch.float16, e4):
        assert p.rejection(e4, e4, od) is None
    assert get_profile("hopper-experimental").rejection(e4, e4, e5) is None


def test_reject_rule_matches_the_recorded_heuristic_status():
    """The exported table dropped exactly the status != 0 rows, all of them e4m3 x e4m3 -> e5m2 (R2: status 15)."""
    meta = json.loads((DATA / "heur_rtx4060_lt12.8.3.14.json").read_text())
    assert meta["dropped_status_nonzero"] == 71320            # 8,915 shapes x 4 workspaces x fast/slow
    z = np.load(DATA / "heur_rtx4060_lt12.8.3.14.npz", allow_pickle=False)
    bad = (z["fa"] == 0) & (z["fb"] == 0) & (z["out"] == 4)
    assert int(bad.sum()) == 0


@pytest.fixture
def fake_device(monkeypatch):
    state = dict(name="NVIDIA GeForce RTX 4060", sm=24, ver="12.8.3.14", raise_=False)

    def name(device=None):
        if state["raise_"]:
            raise cublaslt.CublasLtError("no driver")
        return state["name"]
    monkeypatch.setattr(cublaslt, "device_name", name)
    monkeypatch.setattr(cublaslt, "device_sm_count", lambda device=None: state["sm"])
    monkeypatch.setattr(cublaslt, "version", lambda: state["ver"])
    monkeypatch.setattr(cublaslt, "device_capability", lambda device=None: (8, 9))
    monkeypatch.delenv("MMAEMU_DISABLE_CUDA", raising=False)
    return state


def test_matches_device_needs_name_sm_count_and_cublaslt(fake_device, monkeypatch):
    p = get_profile()
    assert p.matches_device() is True
    assert p.matches_device(0) is True
    fake_device["ver"] = "12.8.3"                    # no build number from the library: first three must match
    assert p.matches_device() is True
    fake_device["ver"] = "12.8.4.1"
    assert p.matches_device() is False
    fake_device["ver"] = "12.8.3.14"
    fake_device["sm"] = 70                           # same name, other SM count -> no
    assert p.matches_device() is False
    fake_device["sm"] = 24
    fake_device["name"] = "NVIDIA GeForce RTX 4060 Ti"   # cc (8, 9) alone never matches
    assert p.matches_device() is False
    fake_device["name"] = "NVIDIA GeForce RTX 4060"
    monkeypatch.setenv("MMAEMU_DISABLE_CUDA", "1")
    assert p.matches_device() is False
    monkeypatch.delenv("MMAEMU_DISABLE_CUDA")
    fake_device["raise_"] = True
    assert p.matches_device() is False
    fake_device["raise_"] = False
    import torch
    assert p.matches_device(torch.device("cpu")) is False


def test_custom_profile_objects_are_accepted():
    p = dataclasses.replace(get_profile(), name="custom-32mib", workspace_bytes=32 << 20)
    assert get_profile(p) is p and isinstance(p, Profile)


@pytest.mark.profile_device
def test_matches_the_reference_gpu_live(request):
    """Runs only on the profile device (RTX 4060, 24 SMs, cuBLASLt 12.8.3.14), and only when selected with
    -m profile_device (it touches the GPU; run it under gpu.sh)."""
    if "profile_device" not in (request.config.getoption("markexpr") or ""):
        pytest.skip("GPU test: select with -m profile_device and run under gpu.sh")
    p = get_profile("ada-rtx4060-torch")
    if not p.matches_device():
        pytest.skip("profile device unavailable")
    assert get_profile("ada-rtx4060-te").matches_device() and get_profile("ada-rtx4060-ws0").matches_device()
    assert cublaslt.device_capability() == (8, 9)
    assert cublaslt.version() == "12.8.3.14"
