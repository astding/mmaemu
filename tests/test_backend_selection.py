"""Backend routing, the gating rule, environment overrides and fallback warnings.
CPU only: kernels are fake KernelSpecs, the device capability and the load/gate outcome are monkeypatched, and the
CUDA driver must never be touched."""
from __future__ import annotations

import json
import warnings

import pytest
import torch

from mmaemu.backends import ChainRequest, registry
from mmaemu.backends.cuda import driver, modules, selftest
from mmaemu.backends.cuda.spec import GATES, GateResult, KernelSpec
from mmaemu.errors import BackendFallbackWarning, BackendUnavailableError


def fake_spec(name, priority, *, gated=False, validated=((8, 9),), gates=(), min_cc=(7, 5), smem=0, preset="ada"):
    return KernelSpec(name=name, module=f"fake_{name}", source=f"fake_{name}.cu", entry=f"mmaemu_{name}_e4m3e4m3",
                      sass_archs=(75, 80, 89, 120), ptx_arch=None, min_cc=min_cc, gated=gated,
                      validated_on=frozenset(validated), gates=gates, priority=priority, tile_m=64, tile_n=64,
                      threads=256, smem_bytes=smem, align_bytes=4, k_max=1 << 24, preset=preset)


TC3 = fake_spec("tc3", 40, gated=True, gates=("design_a",), min_cc=(8, 0))
V9 = fake_spec("v9", 30, gated=True, gates=("window",), min_cc=(8, 0))
V4 = fake_spec("v4", 20, min_cc=(8, 0))
V1 = fake_spec("v1", 10)
REQ = ChainRequest(device=torch.device("cuda", 0), fmt_a="e4m3", fmt_b="e4m3", preset="ada")
ENV = ("MMAEMU_BACKEND", "MMAEMU_DISABLE_CUDA", "MMAEMU_ALLOW_GATED_KERNELS", "MMAEMU_TEST_FORCE_FAIL",
       "MMAEMU_FATBIN_DIR")


class World:
    """Programmable outcome of loading and gating each fake kernel."""

    def __init__(self, mp, cc=(8, 9), specs=(TC3, V9, V4, V1)):
        self.cc = cc
        self.specs = list(specs)
        self.outcome = {s.name: (True, "", "fatbin", None) for s in specs}  # ok, reason, source, fatbin_error
        self.tried: list[str] = []
        mp.setattr(registry, "discover", lambda: sorted(self.specs, key=lambda s: -s.priority))
        mp.setattr(torch.cuda, "is_available", lambda: True)
        mp.setattr(torch.cuda, "get_device_capability", lambda d=None: self.cc)
        mp.setattr(torch.cuda, "current_device", lambda: 0)
        mp.setattr(registry, "_try_kernel", self._try)

        def no_driver(*a, **k):
            raise AssertionError("the CUDA driver must not be touched in this test")
        mp.setattr(driver, "libcuda", no_driver)
        for k in ENV:
            mp.delenv(k, raising=False)
        registry.clear()

    def _try(self, spec, dev):
        self.tried.append(spec.name)
        ok, reason, source, fe = self.outcome[spec.name]
        return ok, reason, source, (), fe


@pytest.fixture
def world(monkeypatch):
    w = World(monkeypatch)
    yield w
    registry.clear()


def select(req=REQ, backend="auto"):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        be = registry.select(req, backend)
    return be, [w for w in rec if issubclass(w.category, BackendFallbackWarning)]


# ---------------------------------------------------------------------------------------------------- routing

def test_priority_order_on_validated_device(world):
    be, w = select()
    assert be.name == "cuda:tc3" and not w and world.tried == ["tc3"]


def test_cpu_tensors_use_reference_without_cuda(world, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: (_ for _ in ()).throw(AssertionError("no CUDA query")))
    be, w = select(ChainRequest(device=torch.device("cpu"), fmt_a="e4m3", fmt_b="e4m3", preset="ada"))
    assert be.name == "reference" and not w and world.tried == []


def test_unsupported_formats_and_presets_route_to_reference(world):
    for fa, fb, preset in (("e5m2", "e5m2", "ada"), ("e4m3", "e5m2", "ada"), ("e4m3", "e4m3", "hopper")):
        be, w = select(ChainRequest(device=REQ.device, fmt_a=fa, fmt_b=fb, preset=preset))
        assert be.name == "reference" and not w
    assert world.tried == []


def test_kernels_serve_only_their_preset(monkeypatch):
    """A kernel is used only for its own preset (KernelSpec.preset): hopper requests reach h1 and never the ada
    kernels, ada requests never reach h1; formats a kernel lacks still go to the reference."""
    h1 = fake_spec("h1", 10, preset="hopper")
    w = World(monkeypatch, cc=(12, 0), specs=(TC3, V9, V4, V1, h1))
    hop = ChainRequest(device=REQ.device, fmt_a="e4m3", fmt_b="e4m3", preset="hopper")
    be, warns = select(hop)
    assert be.name == "cuda:h1" and not warns and w.tried == ["h1"]
    w.tried.clear()
    be, _ = select(REQ)
    assert be.name == "cuda:v4" and "h1" not in w.tried
    be, _ = select(ChainRequest(device=REQ.device, fmt_a="e4m3", fmt_b="e5m2", preset="hopper"))
    assert be.name == "reference"
    _, skipped = registry.explain_selection(hop)
    assert any(r.startswith("cuda:v4: implements preset 'ada', not 'hopper'") for r in skipped), skipped
    with pytest.raises(BackendUnavailableError, match="implements preset 'hopper'"):
        registry.select(REQ, "cuda:h1")
    st = registry.status(torch.device("cuda", 0))
    assert st["cuda:h1"].available and st["cuda:v1"].available
    registry.clear()


def test_hopper_prefers_h4_and_falls_back_to_h1(monkeypatch):
    """Within the hopper preset the priority order holds as for ada: h4 (20, sm_80+) before h1 (10); below sm_80, or
    when h4 fails to load, h1 serves (a load failure warns once)."""
    h1 = fake_spec("h1", 10, preset="hopper")
    h4 = fake_spec("h4", 20, min_cc=(8, 0), preset="hopper")
    hop = ChainRequest(device=REQ.device, fmt_a="e4m3", fmt_b="e4m3", preset="hopper")
    w = World(monkeypatch, cc=(12, 0), specs=(V4, V1, h1, h4))
    be, warns = select(hop)
    assert be.name == "cuda:h4" and not warns and w.tried == ["h4"]
    w = World(monkeypatch, cc=(7, 5), specs=(V4, V1, h1, h4))
    be, warns = select(hop)
    assert be.name == "cuda:h1" and not warns and w.tried == ["h1"]
    w = World(monkeypatch, cc=(12, 0), specs=(V4, V1, h1, h4))
    w.outcome["h4"] = (False, "no binary", None, None)
    be, warns = select(hop)
    assert be.name == "cuda:h1" and w.tried == ["h4", "h1"] and len(warns) == 1
    registry.clear()


def test_explicit_backends(world):
    assert select(backend="reference")[0].name == "reference"
    assert select(backend="cuda:v1")[0].name == "cuda:v1"
    assert select(backend="cuda")[0].name == "cuda:tc3"
    with pytest.raises(ValueError):
        registry.select(REQ, "gpu")
    with pytest.raises(BackendUnavailableError):
        registry.select(REQ, "cuda:v7")
    world.outcome["v4"] = (False, "cannot load: boom", None, None)
    with pytest.raises(BackendUnavailableError):
        registry.select(REQ, "cuda:v4")


def test_env_backend_and_disable(world, monkeypatch):
    monkeypatch.setenv("MMAEMU_BACKEND", "cuda:v4")
    assert select()[0].name == "cuda:v4"
    assert select(backend="cuda:v1")[0].name == "cuda:v1"  # an explicit argument wins over the environment
    monkeypatch.setenv("MMAEMU_BACKEND", "reference")
    assert select()[0].name == "reference"
    monkeypatch.delenv("MMAEMU_BACKEND")
    monkeypatch.setenv("MMAEMU_DISABLE_CUDA", "1")
    be, w = select()
    assert be.name == "reference" and not w
    with pytest.raises(BackendUnavailableError):
        registry.select(REQ, "cuda:v1")


def test_no_cuda_device(world, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    be, w = select()
    assert be.name == "reference" and not w and world.tried == []
    with pytest.raises(BackendUnavailableError):
        registry.select(REQ, "cuda")


def test_cuda_device_without_index_when_cuda_is_unusable(world, monkeypatch):
    # "cuda" (no index) must not ask torch for the current device when CUDA is unavailable or disabled
    def no_current():
        raise AssertionError("Torch not compiled with CUDA enabled")
    monkeypatch.setattr(torch.cuda, "current_device", no_current)
    req = ChainRequest(device=torch.device("cuda"), fmt_a="e4m3", fmt_b="e4m3", preset="ada")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select(req)[0].name == "reference"
    registry.clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("MMAEMU_DISABLE_CUDA", "1")
    assert select(req)[0].name == "reference" and world.tried == []


def test_min_cc(world):
    world.cc = (7, 5)
    be, w = select()
    assert be.name == "cuda:v1" and not w and world.tried == ["v1"]


# ---------------------------------------------------------------------------------------------------- gating rule

def test_gated_kernels_need_validated_capability(world, monkeypatch):
    world.cc = (12, 0)
    be, w = select()
    assert be.name == "cuda:v4" and not w and world.tried == ["v4"]
    st = registry.explain_selection(REQ)[1]
    assert any(s.startswith("cuda:tc3: gated kernel not validated on sm_120") for s in st)
    monkeypatch.setenv("MMAEMU_ALLOW_GATED_KERNELS", "1")
    world.tried.clear()
    assert select()[0].name == "cuda:tc3" and world.tried == ["tc3"]


def test_explicit_gated_kernel_still_runs_its_gates(world):
    world.cc = (12, 0)
    world.outcome["v9"] = (False, "gate 'window' failed: no window property", "fatbin", None)
    with pytest.raises(BackendUnavailableError, match="window"):
        registry.select(REQ, "cuda:v9")


def test_gate_failure_falls_back_with_one_warning(world):
    world.outcome["tc3"] = (False, "gate 'design_a' failed: F=25 HMMA", "fatbin", None)
    be, w = select()
    assert be.name == "cuda:v9" and len(w) == 1 and "design_a" in str(w[0].message)
    be, w = select()  # same device, same reason: no second warning
    assert be.name == "cuda:v9" and not w


def test_all_kernels_fail_gives_reference_and_one_warning(world):
    for k in world.outcome:
        world.outcome[k] = (False, "cannot load: fatbin: forced; nvrtc: forced", None, None)
    be, w = select()
    assert be.name == "reference" and len(w) == 1 and "reference" in str(w[0].message)
    assert select()[1] == []


def test_nvrtc_fallback_warns_once_per_device(world, monkeypatch):
    world.outcome["tc3"] = (True, "", "nvrtc", "no fatbin manifest in /x")
    be, w = select()
    assert be.name == "cuda:tc3" and len(w) == 1 and "NVRTC" in str(w[0].message)
    assert select()[1] == []
    be, w = select(ChainRequest(device=torch.device("cuda", 1), fmt_a="e4m3", fmt_b="e4m3", preset="ada"))
    assert len(w) == 1  # another device: its own warning
    registry.reset_warnings()
    assert len(select()[1]) == 1


def test_fast_package_absent_is_tolerated(monkeypatch):
    registry.clear()
    specs = registry.discover()  # real discovery: v1 and v4 at least, plus the fast kernels
    names = [s.name for s in specs]
    assert names.index("v4") < names.index("v1")
    assert [s.priority for s in specs] == sorted((s.priority for s in specs), reverse=True)


# ---------------------------------------------------------------------------------------------------- status

def test_status_reasons(world):
    world.cc = (12, 0)
    world.outcome["v4"] = (False, "gate 'smoke' failed: 3 of 4352 outputs differ", "fatbin", None)
    st = registry.status(torch.device("cuda", 0))
    assert list(st) == ["cuda:tc3", "cuda:v9", "cuda:v4", "cuda:v1", "reference"]
    assert not st["cuda:tc3"].available and "not validated" in st["cuda:tc3"].reason
    assert not st["cuda:v4"].available and "smoke" in st["cuda:v4"].reason
    assert st["cuda:v1"].available and st["cuda:v1"].source == "fatbin" and not st["cuda:v1"].validated_here
    assert st["reference"].available


def test_status_without_cuda_never_touches_the_driver(world, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    st = registry.status()
    assert all(not v.available for k, v in st.items() if k != "reference") and st["reference"].available
    monkeypatch.setenv("MMAEMU_DISABLE_CUDA", "1")
    st = registry.status(torch.device("cuda", 0))
    assert st["cuda:v1"].reason == "MMAEMU_DISABLE_CUDA=1"


# ---------------------------------------------------------------------------------------------------- selftest cache

@pytest.fixture
def gate_world(monkeypatch, tmp_path):
    monkeypatch.setenv("MMAEMU_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("MMAEMU_TEST_FORCE_FAIL", raising=False)
    monkeypatch.setattr(modules, "get_function", lambda m, e, d: (1, "fatbin"))

    class LM:
        sha256 = "ab" * 32
    monkeypatch.setattr(modules, "module_info", lambda m, d: LM)
    monkeypatch.setattr(driver, "device_uuid", lambda i: "GPU-fake")
    monkeypatch.setattr(driver, "driver_version", lambda: 12080)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: (8, 9))
    calls = []

    def gate(spec, device):
        calls.append(spec.name)
        return GateResult(name="smoke", passed=True, reason="", details={"n": 1}, seconds=0.01, cached=False)
    monkeypatch.setitem(GATES, "smoke", gate)
    selftest.clear_memory_cache()
    yield calls, tmp_path
    selftest.clear_memory_cache()


def test_verdict_cache_memory_disk_and_corruption(gate_world):
    calls, tmp = gate_world
    dev = torch.device("cuda", 0)
    r1 = selftest.run_gates(V1, dev)
    assert r1[0].passed and not r1[0].cached and calls == ["v1"]
    r2 = selftest.run_gates(V1, dev)
    assert r2[0].cached and calls == ["v1"]
    selftest.clear_memory_cache()  # a new process: the disk verdict is used
    r3 = selftest.run_gates(V1, dev)
    assert r3[0].cached and calls == ["v1"]
    files = list((tmp / "verdicts").glob("*.json"))
    assert len(files) == 1
    key = json.loads(files[0].read_text())["key"]
    assert {"gpu_uuid", "cc", "driver", "mmaemu", "module_sha256", "gate", "gate_version"} <= set(key)
    files[0].write_text("{not json")
    selftest.clear_memory_cache()
    r4 = selftest.run_gates(V1, dev)
    assert not r4[0].cached and calls == ["v1", "v1"]
    assert selftest.run_gates(V1, dev, force=True)[0].cached is False and calls == ["v1"] * 3


def test_forced_gate_failure_is_not_cached(gate_world, monkeypatch):
    calls, tmp = gate_world
    monkeypatch.setenv("MMAEMU_TEST_FORCE_FAIL", "smoke")
    r = selftest.run_gates(V1, torch.device("cuda", 0))
    assert not r[0].passed and "forced" in r[0].reason and calls == []
    assert not (tmp / "verdicts").exists()
    monkeypatch.delenv("MMAEMU_TEST_FORCE_FAIL")
    assert selftest.run_gates(V1, torch.device("cuda", 0))[0].passed and calls == ["v1"]


def test_unregistered_and_crashing_gates(gate_world, monkeypatch):
    calls, _ = gate_world
    r = selftest.run_gates(fake_spec("x", 5, gates=("nope",)), torch.device("cuda", 0))
    assert [g.name for g in r] == ["smoke", "nope"] and r[0].passed and not r[1].passed
    monkeypatch.setitem(GATES, "boom", lambda s, d: 1 / 0)
    r = selftest.run_gates(fake_spec("y", 5, gates=("boom",)), torch.device("cuda", 0))
    assert not r[-1].passed and "ZeroDivisionError" in r[-1].reason


def test_read_only_cache_keeps_verdicts_in_memory(gate_world, monkeypatch, tmp_path):
    calls, _ = gate_world
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o555)
    monkeypatch.setenv("MMAEMU_CACHE_DIR", str(ro / "cache"))
    try:
        assert selftest.run_gates(V1, torch.device("cuda", 0))[0].passed
        assert selftest.run_gates(V1, torch.device("cuda", 0))[0].cached
        assert not (ro / "cache").exists()
    finally:
        ro.chmod(0o755)


# ---------------------------------------------------------------------------------------------------- fatbin manifest

def test_fatbin_manifest_checks(monkeypatch, tmp_path):
    """The loader uses a fatbin only if the manifest format, file sha256, source sha256 and archs all match."""
    import hashlib

    loaded = []
    monkeypatch.setattr(driver, "module_load", lambda img: loaded.append(img) or 123)
    monkeypatch.setenv("MMAEMU_FATBIN_DIR", str(tmp_path))
    monkeypatch.delenv("MMAEMU_TEST_FORCE_FAIL", raising=False)
    modules.clear()
    assert "no fatbin manifest" in modules._try_fatbin("v1", 89)
    img = b"\x50\xed\x55\xba" + b"x" * 100
    (tmp_path / "v1.fatbin").write_bytes(img)
    entry = dict(file="v1.fatbin", sha256=hashlib.sha256(img).hexdigest(), source_sha256=modules.source_sha256("v1"),
                 sass_archs=[75, 80, 86, 89, 90, 100, 120], ptx_arch=75, entries=["mmaemu_v1_e4m3e4m3"])

    def write(**kw):
        m = dict(format="mmaemu-fatbin/1", nvcc="12.8.61", flags=["-O3", "--fmad=false"],
                 modules={"v1": dict(entry, **kw)})
        (tmp_path / "manifest.json").write_text(json.dumps(m))
        modules.clear()
    write()
    assert modules._try_fatbin("v1", 89) == (123, img) and modules._try_fatbin("v1", 87) == (123, img)
    assert "not in" in modules._try_fatbin("v4", 89)
    write(sha256="0" * 64)
    assert "does not match" in modules._try_fatbin("v1", 89)
    write(source_sha256="0" * 64)
    assert "stale" in modules._try_fatbin("v1", 89)
    write(sass_archs=[89], ptx_arch=None)
    assert "no SASS or PTX" in modules._try_fatbin("v1", 120)
    write()
    monkeypatch.setenv("MMAEMU_TEST_FORCE_FAIL", "fatbin")
    assert "forced" in modules._try_fatbin("v1", 89)
    modules.clear()


def test_nvrtc_does_not_extend_kernel_architectures(monkeypatch):
    """An opt-in or compiler fallback must not run the Ada-only tc3 chain on sm_120."""
    from mmaemu.backends.cuda.fast.specs import TC3, V9

    monkeypatch.setattr(registry, "discover", lambda: [TC3, V9])
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (12, 0))
    tried = []
    monkeypatch.setattr(modules, "_try_fatbin", lambda *args: tried.append(args) or (123, b"image"))
    monkeypatch.setattr(modules, "_try_nvrtc", lambda *args: pytest.fail("unsupported kernel compiled"))
    with pytest.raises(BackendUnavailableError, match="no binary for sm_120"):
        modules._load("tc3", 0, 1)
    assert tried == []
    monkeypatch.setenv("MMAEMU_ALLOW_GATED_KERNELS", "1")
    assert "no binary" in registry._design_skip(TC3, REQ, REQ.device, True)
    assert modules._load("v9", 0, 1).source == "fatbin"
    assert tried == [("v9", 120)]
# ---------------------------------------------------------------------------------------------------- selftest()

def test_registry_selftest(world, monkeypatch):
    calls = []

    def fake_run_gates(spec, dev, force=False):
        calls.append((spec.name, force))
        return [GateResult(name="smoke", passed=spec.name != "v4", reason="" if spec.name != "v4" else "differs")]
    monkeypatch.setattr(selftest, "run_gates", fake_run_gates)
    world.cc = (12, 0)
    r = registry.selftest(torch.device("cuda", 0))
    assert list(r) == ["tc3", "v9", "v4", "v1"]  # gated kernels are tested even where not validated
    assert r["v1"][0].passed and not r["v4"][0].passed
    r = registry.selftest(torch.device("cuda", 0), kernels="v1", force=True)
    assert list(r) == ["v1"] and calls[-1] == ("v1", True)
    # the CLI (diagnostics.run_selftest) passes a tuple: on the H100 run "v9" became ["('v9'", ")"]
    assert list(registry.selftest(torch.device("cuda", 0), kernels=("v9",))) == ["v9"]
    assert list(registry.selftest(torch.device("cuda", 0), kernels=("v1", " v4"))) == ["v4", "v1"]
    with pytest.raises(ValueError):
        registry.selftest(torch.device("cuda", 0), kernels="v7")
    world.cc = (7, 0)
    assert registry.selftest(torch.device("cuda", 0), kernels="v1")["v1"][0].name == "availability"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    r = registry.selftest()
    assert all(v[0].name == "availability" and not v[0].passed for v in r.values())


def test_frozen_entry_points_delegate_to_registry(world):
    import mmaemu.backends as B

    assert B.select(REQ, "cuda:v1").name == "cuda:v1"
    assert set(B.status(torch.device("cuda", 0))) >= {"cuda:v1", "reference"}
