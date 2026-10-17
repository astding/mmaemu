"""CLI smoke tests: ``python -m mmaemu info | selftest | explain``.

Unit tests drive mmaemu.diagnostics / mmaemu.__main__ with a fake public API, so the exit-code and classification
rules are tested without a GPU and independently of the backends. Subprocess tests run the real CLI on CPU.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import sys
import types

import pytest

import mmaemu.diagnostics as D
from mmaemu import __main__ as cli


# ----------------------------------------------------------------------------------------------- fakes
@dataclasses.dataclass(frozen=True)
class FakePreset:
    name: str
    maturity: str
    validated_scope: str


@dataclasses.dataclass(frozen=True)
class FakeProfile:
    name: str
    preset: str = "ada"
    gpu_name: str = "NVIDIA GeForce RTX 4060"
    capability: tuple = (8, 9)
    sm_count: int = 24
    cublaslt_version: str = "12.8.3.14"
    caller: str = "torch._scaled_mm"
    torch_version: str = "2.7.1"
    workspace_bytes: int = 1 << 20
    align: int | None = None
    splitk_table: str | None = "heur_rtx4060_lt12.8.3.14"
    maturity: str = "validated"
    splitk_maturity: str = "validated"
    match: bool = False

    def matches_device(self, device=None):
        return self.match


@dataclasses.dataclass(frozen=True)
class FakeStatus:
    name: str
    available: bool
    source: str | None
    validated_here: bool
    gates: tuple = ()
    reason: str = ""


@dataclasses.dataclass(frozen=True)
class FakeGate:
    name: str
    passed: bool
    reason: str = ""
    details: dict = dataclasses.field(default_factory=dict)
    seconds: float = 0.01
    cached: bool = False


@dataclasses.dataclass(frozen=True)
class FakeSpec:
    name: str
    gated: bool
    validated_on: frozenset
    min_cc: tuple = (7, 5)
    sass_archs: tuple = (89,)
    ptx_arch: int | None = None
    preset: str = "ada"


SPECS = {"v1": FakeSpec("v1", False, frozenset({(8, 9)})), "v4": FakeSpec("v4", False, frozenset({(8, 9)}), (8, 0)),
         "v9": FakeSpec("v9", True, frozenset({(8, 9)}), (8, 0)), "tc3": FakeSpec("tc3", True, frozenset({(8, 9)}), (8, 0))}


def not_impl(component):
    def f(*a, **k):
        raise NotImplementedError(component)
    return f


def fake_api(*, selftest=None, explain=None, profiles=None):
    presets = {"ada": FakePreset("ada", "validated", "RTX 4060 hardware"),
               "hopper": FakePreset("hopper", "experimental", "instruction level, c=0 vectors only"),
               "b200_mma_sync": FakePreset("b200_mma_sync", "experimental", "toolbox B200 mma.sync vectors; not sm_120")}
    profs = profiles or {"ada": FakeProfile("ada", match=True),
                         "hopper-experimental": FakeProfile("hopper-experimental", preset="hopper", splitk_table=None,
                                                            maturity="experimental", splitk_maturity="none")}

    def get_preset(name):
        if name == "blackwell":
            return presets["b200_mma_sync"]
        return presets[name]

    def get_profile(name=None):
        return profs[name or "ada"]

    return types.SimpleNamespace(
        __version__="0.1.0.dev0",
        list_presets=lambda: list(presets), get_preset=get_preset,
        list_profiles=lambda: list(profs), get_profile=get_profile,
        backends_status=lambda device=None: {
            "cuda:v1": FakeStatus("cuda:v1", True, "fatbin", True, ("smoke",)),
            "cuda:tc3": FakeStatus("cuda:tc3", False, None, False, (), "no binary for sm_120"),
            "reference": FakeStatus("reference", True, "torch", True)},
        selftest=selftest or not_impl("mmaemu.backends.registry"), explain=explain or not_impl("mmaemu.api"))


GPU = {"torch": "2.7.1", "cuda_available": True, "device_index": 0, "name": "Fake GPU", "capability": [8, 9],
       "sm_count": 24, "total_memory_mib": 8188}


@pytest.fixture
def fake(monkeypatch):
    def install(api, device=None, specs=SPECS):
        monkeypatch.setattr(D, "_api", lambda: api)
        monkeypatch.setattr(D, "device_info", lambda idx=0: dict(device or {"torch": "2.7.1", "cuda_available": False,
                                                                           "note": "no CUDA device"}))
        monkeypatch.setattr(D, "_torch_device", lambda info: "cuda:0" if info.get("cuda_available") else None)
        monkeypatch.setattr(D, "kernel_specs", lambda: (dict(specs), []))
    return install


# ----------------------------------------------------------------------------------------------- info
def test_info_all_components_unavailable_exits_0(fake, capsys):
    names = ("list_presets", "get_preset", "list_profiles", "get_profile", "backends_status")
    api = types.SimpleNamespace(**{n: not_impl("mmaemu.platform") for n in names})
    fake(api)
    assert cli.main(["info"]) == 0
    out = capsys.readouterr().out
    assert "Unavailable components" in out and "not implemented yet (mmaemu.platform)" in out


def test_info_reports_profiles_presets_backends(fake, capsys, tmp_path):
    fake(fake_api(), device=GPU)
    path = tmp_path / "info.json"
    assert cli.main(["info", "--json", str(path)]) == 0
    out = capsys.readouterr().out
    assert re.search(r"^  ada: preset ada, .*MATCHES this device", out, re.M)
    assert "b200_mma_sync (alias: blackwell): experimental" in out
    assert "cuda:tc3 (preset ada): unavailable - no binary for sm_120" in out
    assert "cuda:v1 (preset ada): available via fatbin, validated here" in out
    rep = json.loads(path.read_text())
    assert rep["format"] == D.INFO_FORMAT and rep["default_profile"] == "ada"
    assert {p["name"]: p["matches_device"] for p in rep["profiles"]} == {"ada": True,
                                                                       "hopper-experimental": False}
    assert rep["backends"]["cuda:v1"]["source"] == "fatbin" and rep["kernels"]["v1"]["preset"] == "ada"
    assert {p["name"]: p["maturity"] for p in rep["presets"]}["hopper"] == "experimental"


def test_info_json_stdout_is_pure_json(fake, capsys):
    fake(fake_api())
    assert cli.main(["info", "--json", "-"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["mmaemu"]["version"] == "0.1.0.dev0"


# ----------------------------------------------------------------------------------------------- selftest
def gates(**passed):
    return {k: [FakeGate("smoke", v)] for k, v in passed.items()}


@pytest.mark.parametrize("cc,result,code,verdict", [
    ([8, 9], dict(v1=True, v4=True, v9=True, tc3=True), 0, "PASS"),
    ([8, 9], dict(v1=True, v4=True, v9=False, tc3=True), 1, "FAIL"),     # gated kernel fails where it is validated
    ([12, 0], dict(v1=True, v4=True, v9=False, tc3=False), 0, "PASS"),   # rejected on unvalidated hardware: optional
    ([12, 0], dict(v1=False, v4=True, v9=True, tc3=False), 1, "FAIL"),   # IEEE kernel must pass everywhere
    ([7, 5], dict(v1=True, v4=False), 0, "PASS"),                        # v4 below its min_cc: not required
])
def test_selftest_classification(fake, capsys, tmp_path, cc, result, code, verdict):
    res = gates(**result)
    fake(fake_api(selftest=lambda device=None, *, kernels="all", force=False: res), device=dict(GPU, capability=cc))
    path = tmp_path / "st.json"
    assert cli.main(["selftest", "--force", "--json", str(path)]) == code
    rep = json.loads(path.read_text())
    assert rep["verdict"] == verdict and rep["force"] is True
    assert set(rep["kernels"]) == set(result)


def test_selftest_without_device_is_unavailable_not_success(fake):
    fake(fake_api(selftest=lambda *a, **k: gates(v1=True)))
    assert cli.main(["selftest"]) == D.EXIT_UNAVAILABLE


def test_selftest_not_implemented_is_unavailable(fake, capsys):
    fake(fake_api(), device=GPU)
    assert cli.main(["selftest"]) == D.EXIT_UNAVAILABLE
    assert "not implemented yet (mmaemu.backends.registry)" in capsys.readouterr().out


def test_selftest_empty_result_is_unavailable(fake):
    fake(fake_api(selftest=lambda *a, **k: {}), device=GPU)
    assert cli.main(["selftest"]) == D.EXIT_UNAVAILABLE


def test_selftest_prefixed_names_and_kernel_list(fake, tmp_path):
    """Results keyed "cuda:v9" find the v9 spec; --kernels v1,v4 reaches mmaemu.selftest as a tuple."""
    seen = {}

    def selftest(device=None, *, kernels="all", force=False):
        seen["kernels"] = kernels
        return {"cuda:v1": [FakeGate("smoke", True)], "cuda:v9": [FakeGate("window", False)]}

    fake(fake_api(selftest=selftest), device=dict(GPU, capability=[12, 0]))
    assert cli.main(["selftest", "--kernels", "v1, v9", "--json", str(tmp_path / "s.json")]) == 0
    assert seen["kernels"] == ("v1", "v9")
    rep = json.loads((tmp_path / "s.json").read_text())
    assert rep["kernels"]["cuda:v9"]["required"] is False and rep["kernels"]["cuda:v1"]["required"] is True


def test_unknown_kernel_without_spec_is_required(fake):
    fake(fake_api(selftest=lambda *a, **k: gates(v1=True, vX=False)), device=GPU)
    assert cli.main(["selftest"]) == D.EXIT_FAIL


# ----------------------------------------------------------------------------------------------- explain
@dataclasses.dataclass
class FakePlan:
    validation: object
    profile: str = "ada"
    preset: str = "ada"
    preset_maturity: str = "validated"
    split: object = None
    backend: str = "reference"
    fallbacks: list = dataclasses.field(default_factory=list)
    warnings: list = dataclasses.field(default_factory=list)


def test_explain_ok_and_rejected(fake, capsys):
    seen = {}

    def explain(input, mat2, scale_a, scale_b, bias=None, scale_result=None, out_dtype=None, use_fast_accum=False,
                **kw):
        seen.update(shape=(tuple(input.shape), tuple(mat2.shape)), stride=mat2.stride(), out=out_dtype,
                    fast=use_fast_accum, **kw)
        return FakePlan("ok" if input.shape[1] % 16 == 0 else ("ReferenceRejectsError", "K % 16 != 0"))

    fake(fake_api(explain=explain))
    assert cli.main(["explain", "16", "8", "32", "--out", "bf16", "--fast", "--splitk", "8,serial",
                     "--device", "cpu"]) == 0
    assert seen["shape"] == ((16, 32), (32, 8)) and seen["stride"] == (1, 32)  # mat2 column-major
    assert seen["splitk"] == (8, "serial") and seen["fast"] is True and str(seen["out"]) == "torch.bfloat16"
    assert "validation       ok" in capsys.readouterr().out
    assert cli.main(["explain", "16", "8", "24", "--device", "cpu"]) == 1


def test_explain_unavailable(fake):
    fake(fake_api())
    assert cli.main(["explain", "16", "16", "32", "--device", "cpu"]) == D.EXIT_UNAVAILABLE


@pytest.mark.parametrize("argv", [["explain", "16"], ["explain", "16", "-8", "32"], ["explain", "16", "8", "x"],
                                  ["explain", "16", "16", "32", "--splitk", "two"],
                                  ["explain", "16", "16", "32", "--splitk", "2.5,serial"]])
def test_usage_error_exit_2(argv):
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 2


def test_explain_unusable_device_exits_4(capsys):
    assert cli.main(["explain", "16", "16", "32", "--device", "nonsense"]) == D.EXIT_UNAVAILABLE
    assert "device 'nonsense'" in capsys.readouterr().err


# ----------------------------------------------------------------------------------------------- real CLI
def _run(*args, env_extra=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-m", "mmaemu", *args], capture_output=True, text=True, env=env,
                          timeout=300)


def test_real_info_cpu_exits_0():
    r = _run("info")
    assert r.returncode == 0, r.stderr
    assert "mmaemu" in r.stdout and "Device" in r.stdout


def test_real_info_json(tmp_path):
    r = _run("info", "--json", str(tmp_path / "i.json"))
    assert r.returncode == 0, r.stderr
    rep = json.loads((tmp_path / "i.json").read_text())
    assert rep["format"] == D.INFO_FORMAT and rep["device"]["cuda_available"] is False


def test_real_selftest_cpu_is_unavailable():
    r = _run("selftest")
    assert r.returncode == D.EXIT_UNAVAILABLE, r.stdout + r.stderr
    assert "UNAVAILABLE" in r.stdout


def test_real_explain_cpu():
    r = _run("explain", "16", "5760", "1920", "--out", "bf16", "--fast", "--device", "cpu")
    if r.returncode == D.EXIT_UNAVAILABLE and ("not implemented yet" in r.stderr or "has no attribute" in r.stderr):
        pytest.skip(f"explain not implemented in this tree: {r.stderr.strip()}")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "validation" in r.stdout and "backend" in r.stdout


def test_info_respects_disable_cuda():
    """With MMAEMU_DISABLE_CUDA=1 the info report does not initialise CUDA and maps no CUDA library beyond those
    that `import torch` itself maps (a CUDA build of torch may map libcuda.so at import, e.g. on WSL2)."""
    code = ("import re, torch; maps = lambda: set(re.findall(r'\\S*libcuda\\.so\\S*', open('/proc/self/maps').read())); "
            "before = maps(); import mmaemu.diagnostics as d; r = d.collect_info(); "
            "print(torch.cuda.is_initialized(), sorted(maps() - before), r['device']['cuda_available'])")
    env = dict(os.environ, MMAEMU_DISABLE_CUDA="1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False [] False", r.stdout
