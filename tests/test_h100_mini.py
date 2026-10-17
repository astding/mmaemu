"""T1 (CPU, every push) and its GPU variant: replay the committed H100 PCIe live calls tests/data/h100_mini/jobs.json
through mmaemu, bitwise (tools/export_h100_mini.py; docs/validation_logs/2026-10-15/h100-probe).

- Inputs are regenerated from each job's spec by tools/record/live.py (integer hashing, the same bytes on CPU and GPU);
  the expected result is the sha256 of the H100's full output bits (or the error class of a call the H100 rejected).
- The logged split is passed explicitly; splitk="auto" must give the same decision wherever the profile's table has
  the shape (every M6 job).
- Profiles: hopper with the call's logged workspace (hopper-ws0 for 0).
- CPU: backend "reference", torch.set_num_threads(4). GPU (marker gpu): the same cases on cuda:0 with the reference
  backend, with backend "auto", which runs the hopper preset's CUDA kernel h4 for e4m3 x e4m3 (h1 below sm_80; the
  other format pairs stay on the reference), and with "cuda:h1" on the e4m3 x e4m3 cases; the tests check which
  backend ran."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if not (REPO / "tools" / "record" / "live.py").is_file():
    pytest.skip("tools/ not in this tree (sdist)", allow_module_level=True)
sys.path.insert(0, str(REPO / "tools" / "record"))
import live  # noqa: E402

import mmaemu  # noqa: E402
from mmaemu import errors as E  # noqa: E402
from mmaemu.platform.profiles import get_profile  # noqa: E402

DATA = json.loads((REPO / "tests" / "data" / "h100_mini" / "jobs.json").read_text())
JOBS = DATA["jobs"]
FP8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}


def _profile(ws: int):
    if ws == 0:
        return get_profile("hopper-ws0")
    p = get_profile("hopper")
    return p if ws == p.workspace_bytes else dataclasses.replace(p, name=f"{p.name}-ws{ws}", workspace_bytes=ws)


def _sha(y: torch.Tensor) -> str:
    t = y.detach().cpu().contiguous()
    it = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()]
    return hashlib.sha256(np.ascontiguousarray(t.view(it).numpy()).tobytes()).hexdigest()


def _run(j: dict, dev: torch.device, backend: str = "reference"):
    a8, b8 = live.make_inputs(j, device=dev)
    a, b = a8.view(FP8[j["fa"]]), b8.view(FP8[j["fb"]])
    sa = torch.tensor(j["scale_a"], dtype=torch.float32, device=dev)
    sb = torch.tensor(j["scale_b"], dtype=torch.float32, device=dev)
    sr = None if j["scale_result"] is None else torch.tensor(j["scale_result"], dtype=torch.float32, device=dev)
    kw = dict(scale_result=sr, out_dtype=getattr(torch, j["out"]), use_fast_accum=bool(j["fast"]),
              profile=_profile(j["ws"]), backend=backend)
    if "error" in j["expect"]:
        with pytest.raises(getattr(E, j["expect"]["error"])) as ei:
            mmaemu.scaled_mm(a, b.t(), sa, sb, splitk=1, **kw)
        assert type(ei.value) is getattr(E, j["expect"]["error"])           # the exact class
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", E.UnverifiedSplitWarning)          # S > 1 on the TE profile warns
        split = 1 if j["S"] == 1 else (j["S"], j["scheme"])
        y = mmaemu.scaled_mm(a, b.t(), sa, sb, splitk=split, **kw)
        ran = mmaemu.explain(a, b.t(), sa, sb, splitk=split, **kw)
        cp = mmaemu.explain(a, b.t(), sa, sb, splitk="auto", **kw)
    assert _sha(y) == j["expect"]["sha256"], f"{j['id']}: {j['why']}"
    assert ran.validation == "ok", ran.validation
    if backend != "reference" and dev.type == "cuda":
        e4m3 = (j["fa"], j["fb"]) == ("e4m3", "e4m3")
        want = backend if backend != "auto" else ("cuda:h4" if torch.cuda.get_device_capability(dev) >= (8, 0)
                                                  else "cuda:h1")
        assert ran.backend == (want if e4m3 else "reference"), ran.backend
    if j["src"] == "m6_live" or cp.validation == "ok":
        assert cp.validation == "ok", cp.validation
        assert (cp.split.S, cp.split.scheme) == (j["S"], j["scheme"] if j["S"] > 1 else "none")


def test_fixture_covers_what_it_claims():
    assert DATA["format"] == "mmaemu-h100-mini/1" and DATA["n_jobs"] == len(JOBS)
    split = [j for j in JOBS if j.get("S") and j["S"] > 1]
    assert {j["S"] for j in split} == {2, 3, 4, 5, 6} and {j["custom"] for j in split} == {0, 1, 2, 3}
    assert {j["scheme"] for j in split} == {"contiguous"} and {j["fast"] for j in split} == {True, False}
    assert {j["out"] for j in JOBS} == {"float32", "bfloat16", "float16", "float8_e4m3fn", "float8_e5m2"}
    assert {(j["fa"], j["fb"]) for j in JOBS} == {("e4m3", "e4m3"), ("e4m3", "e5m2"), ("e5m2", "e4m3")}
    assert any("error" in j["expect"] for j in JOBS) and any(j["K"] < 128 for j in JOBS)
    assert {j["ws"] for j in JOBS} >= {0, 1 << 20, 4 << 20, 32 << 20}


@pytest.mark.parametrize("j", JOBS, ids=lambda j: j["id"])
def test_case(j):
    torch.set_num_threads(4)
    _run(j, torch.device("cpu"))


# auto (h4 on sm_80+) on every job, and h1 explicitly on the E4M3 x E4M3 jobs (the only pair a kernel serves)
GPU_CASES = [(j, be) for be in ("reference", "auto", "cuda:h1") for j in JOBS
             if not be.startswith("cuda:") or (j["fa"], j["fb"]) == ("e4m3", "e4m3")]


@pytest.mark.gpu
@pytest.mark.parametrize("j,backend", GPU_CASES, ids=[f"{j['id']}-{be}" for j, be in GPU_CASES])
def test_case_gpu(j, backend):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    _run(j, torch.device("cuda:0"), backend)
