"""Scaffold tests: import side effects, the frozen public surface, the comparator, the tcemu blocker, the backend
selector, the kernel spec registry and the driver/NVRTC bindings. They never require a stub to stay a stub."""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

import mmaemu
from mmaemu import errors
from mmaemu.backends import BackendStatus, ChainRequest
from mmaemu.backends.cuda import driver, nvrtc, spec
from mmaemu.testing import assert_bitwise_equal, bits

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
PKG_PARENT = Path(mmaemu.__file__).resolve().parents[1]
COMPONENT = re.compile(r"mmaemu(\.\w+)+")
K_SRC = 'extern "C" __global__ void k(float* x){ x[0] = 1.f; }'


def _run(code: str, *, env_extra: dict | None = None, pythonpath: list | None = None, cwd=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in (pythonpath or [PKG_PARENT]))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=cwd,
                          timeout=600)


def _fake_packages(root: Path) -> Path:
    for name in ("tcemu", "tcemu_ada_truth"):
        (root / name).mkdir(parents=True, exist_ok=True)
        (root / name / "__init__.py").write_text(f"NAME = {name!r}\n")
    (root / "tcemu" / "reference.py").write_text("X = 1\n")
    return root


# --- A2: import side effects -------------------------------------------------------------------------------------

def test_a2_literal_command():
    r = _run("import mmaemu,torch,sys; mmaemu.scaled_mm; print(torch.cuda.is_initialized(), 'tcemu' in sys.modules)")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False False"


def test_import_has_no_side_effects(tmp_path):
    """`import mmaemu` alone: no torch, no libcuda mapping, no tcemu. With torch imported (torch 2.7.1+cu128 maps
    libcuda itself, via libcublasLt): touching every public name makes mmaemu dlopen nothing and does not initialise
    CUDA. A fake importable `tcemu` is on the path, so an accidental import would be detected."""
    code = textwrap.dedent("""
        import json, os, sys
        events = []
        def hook(event, args):
            if event == "ctypes.dlopen":
                f, mods = sys._getframe(1), []
                while f is not None:
                    mods.append(f.f_globals.get("__name__", "")); f = f.f_back
                events.append([str(args[0]), [m for m in mods if m == "mmaemu" or m.startswith("mmaemu.")]])
        sys.addaudithook(hook)
        def libcuda_mapped():
            return any(os.path.basename(l.split()[-1]).startswith("libcuda.so") for l in open("/proc/self/maps")
                       if len(l.split()) >= 6)
        import mmaemu
        out = dict(step1=dict(torch="torch" in sys.modules, libcuda=libcuda_mapped(), tcemu="tcemu" in sys.modules))
        import torch
        for name in mmaemu.__all__:
            getattr(mmaemu, name)
        import mmaemu.backends.cuda.spec, mmaemu.backends.cuda.driver, mmaemu.backends.cuda.nvrtc
        import mmaemu.backends.reference, mmaemu.platform.cublaslt, mmaemu.model.plan, mmaemu.model.epilogue
        mmaemu.model.dot; mmaemu.testing.bits
        out["step2"] = dict(cuda_initialized=torch.cuda.is_initialized(), tcemu="tcemu" in sys.modules,
                            mmaemu_dlopens=[e for e in events if e[1]])
        try:
            mmaemu.backends.cuda.nvrtc.libnvrtc()
            out["control"] = [e[0] for e in events if e[1]]
        except mmaemu.BackendUnavailableError:
            out["control"] = None
        print(json.dumps(out))
    """)
    fake = _fake_packages(tmp_path / "fake")
    r = _run(code, pythonpath=[PKG_PARENT, fake])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["step1"] == dict(torch=False, libcuda=False, tcemu=False)
    assert out["step2"] == dict(cuda_initialized=False, tcemu=False, mmaemu_dlopens=[])
    if out["control"] is not None:  # positive control: the hook does see mmaemu's own dlopen
        assert any("nvrtc" in p for p in out["control"]), out["control"]


# --- public surface and frozen signatures ----------------------------------------------------------------------------

def _params(fn):
    return [(p.name, p.kind.name, p.default) for p in inspect.signature(fn).parameters.values()]


E = inspect.Parameter.empty
TORCH_271_SCHEMA = [("input", E), ("mat2", E), ("scale_a", E), ("scale_b", E), ("bias", None),
                    ("scale_result", None), ("out_dtype", None), ("use_fast_accum", False)]
SCALED_MM_KWONLY = [("out", None), ("profile", None), ("splitk", "auto"), ("backend", "auto")]


@pytest.mark.parametrize("name", ["scaled_mm", "explain"])
def test_scaled_mm_signature(name):
    ps = _params(getattr(mmaemu, name))
    assert [(n, d) for n, k, d in ps[:8]] == TORCH_271_SCHEMA
    assert all(k == "POSITIONAL_OR_KEYWORD" for n, k, d in ps[:8])
    assert [(n, d) for n, k, d in ps[8:]] == SCALED_MM_KWONLY
    assert all(k == "KEYWORD_ONLY" for n, k, d in ps[8:])


FROZEN_SIGNATURES = {
    "mmaemu.accumulate": [("a", "POSITIONAL_OR_KEYWORD", E), ("b", "POSITIONAL_OR_KEYWORD", E),
                          ("use_fast_accum", "KEYWORD_ONLY", E), ("profile", "KEYWORD_ONLY", None),
                          ("splitk", "KEYWORD_ONLY", "auto"), ("backend", "KEYWORD_ONLY", "auto"),
                          ("out_dtype_for_split", "KEYWORD_ONLY", torch.float32)],
    "mmaemu.resolve_splitk": [("M", "POSITIONAL_OR_KEYWORD", E), ("N", "POSITIONAL_OR_KEYWORD", E),
                              ("K", "POSITIONAL_OR_KEYWORD", E), ("a_dtype", "KEYWORD_ONLY", E),
                              ("b_dtype", "KEYWORD_ONLY", E), ("out_dtype", "KEYWORD_ONLY", E),
                              ("use_fast_accum", "KEYWORD_ONLY", E), ("profile", "KEYWORD_ONLY", None),
                              ("splitk", "KEYWORD_ONLY", "auto"), ("layout_std", "KEYWORD_ONLY", True),
                              ("strict", "KEYWORD_ONLY", None)],
    "mmaemu.get_profile": [("name_or_profile", "POSITIONAL_OR_KEYWORD", None)],
    "mmaemu.list_profiles": [],
    "mmaemu.list_presets": [],
    "mmaemu.backends_status": [("device", "POSITIONAL_OR_KEYWORD", None)],
    "mmaemu.selftest": [("device", "POSITIONAL_OR_KEYWORD", None), ("kernels", "KEYWORD_ONLY", "all"),
                        ("force", "KEYWORD_ONLY", False)],
    "mmaemu.model.dot": [("a", "POSITIONAL_OR_KEYWORD", E), ("b", "POSITIONAL_OR_KEYWORD", E),
                         ("c", "POSITIONAL_OR_KEYWORD", None), ("preset", "KEYWORD_ONLY", "ada"),
                         ("fmt", "KEYWORD_ONLY", None)],
    "mmaemu.testing.assert_bitwise_equal": [("actual", "POSITIONAL_OR_KEYWORD", E),
                                            ("expected", "POSITIONAL_OR_KEYWORD", E),
                                            ("nan", "KEYWORD_ONLY", "bits"), ("msg", "KEYWORD_ONLY", "")],
    "mmaemu.testing.bits": [("t", "POSITIONAL_OR_KEYWORD", E)],
    "mmaemu.formats.fmt_of": [("dtype", "POSITIONAL_OR_KEYWORD", E)],
    "mmaemu.formats.decode": [("codes_u8", "POSITIONAL_OR_KEYWORD", E), ("fmt", "POSITIONAL_OR_KEYWORD", E)],
    "mmaemu.formats.nonfinite_mask": [("codes_u8", "POSITIONAL_OR_KEYWORD", E), ("fmt", "POSITIONAL_OR_KEYWORD", E)],
    "mmaemu.model.chain.chain_gemm": [("a_codes_u8", "POSITIONAL_OR_KEYWORD", E),
                                      ("b_codes_u8", "POSITIONAL_OR_KEYWORD", E), ("fmt_a", "KEYWORD_ONLY", E),
                                      ("fmt_b", "KEYWORD_ONLY", E), ("preset", "KEYWORD_ONLY", "ada")],
    "mmaemu.model.plan.make_plan": [("K", "POSITIONAL_OR_KEYWORD", E), ("fast", "KEYWORD_ONLY", E),
                                    ("S", "KEYWORD_ONLY", 1), ("scheme", "KEYWORD_ONLY", "none"),
                                    ("preset", "KEYWORD_ONLY", "ada")],
    "mmaemu.model.plan.execute_plan": [("plan", "POSITIONAL_OR_KEYWORD", E), ("a_u8", "POSITIONAL_OR_KEYWORD", E),
                                       ("b_u8", "POSITIONAL_OR_KEYWORD", E), ("fmt_a", "KEYWORD_ONLY", E),
                                       ("fmt_b", "KEYWORD_ONLY", E), ("chain_fn", "KEYWORD_ONLY", E)],
    "mmaemu.model.specials.sanitize": [("codes_u8", "POSITIONAL_OR_KEYWORD", E), ("fmt", "POSITIONAL_OR_KEYWORD", E)],
    "mmaemu.model.specials.nonfinite_patch": [(n, "POSITIONAL_OR_KEYWORD", E)
                                              for n in ("acc32", "a_u8", "b_u8", "fmt_a", "fmt_b")],
    "mmaemu.model.epilogue.apply": [("acc32", "POSITIONAL_OR_KEYWORD", E), ("scale_a", "POSITIONAL_OR_KEYWORD", E),
                                    ("scale_b", "POSITIONAL_OR_KEYWORD", E), ("out_dtype", "KEYWORD_ONLY", E),
                                    ("scale_result", "KEYWORD_ONLY", None)],
    "mmaemu.backends.select": [("req", "POSITIONAL_OR_KEYWORD", E), ("backend", "POSITIONAL_OR_KEYWORD", "auto")],
    "mmaemu.backends.status": [("device", "POSITIONAL_OR_KEYWORD", None)],
    "mmaemu.backends.cuda.spec.register_gate": [("name", "POSITIONAL_OR_KEYWORD", E),
                                                ("fn", "POSITIONAL_OR_KEYWORD", E)],
    "mmaemu.backends.cuda.driver.check": [("res", "POSITIONAL_OR_KEYWORD", E), ("call", "POSITIONAL_OR_KEYWORD", E)],
    "mmaemu.backends.cuda.driver.launch": [(n, "POSITIONAL_OR_KEYWORD", E)
                                           for n in ("fn", "grid", "block", "smem", "stream", "args")],
    "mmaemu.backends.cuda.driver.module_load": [("image", "POSITIONAL_OR_KEYWORD", E)],
    "mmaemu.backends.cuda.driver.get_function": [("module", "POSITIONAL_OR_KEYWORD", E),
                                                 ("name", "POSITIONAL_OR_KEYWORD", E)],
    "mmaemu.backends.cuda.driver.set_max_dynamic_smem": [("fn", "POSITIONAL_OR_KEYWORD", E),
                                                         ("nbytes", "POSITIONAL_OR_KEYWORD", E)],
    "mmaemu.backends.cuda.driver.device_attribute": [("ordinal", "POSITIONAL_OR_KEYWORD", E),
                                                     ("attr", "POSITIONAL_OR_KEYWORD", E)],
    "mmaemu.backends.cuda.driver.device_uuid": [("ordinal", "POSITIONAL_OR_KEYWORD", E)],
    "mmaemu.backends.cuda.driver.driver_version": [],
    "mmaemu.backends.cuda.driver.current_context": [],
    "mmaemu.backends.cuda.driver.libcuda": [],
    "mmaemu.backends.cuda.nvrtc.libnvrtc": [],
    "mmaemu.backends.cuda.nvrtc.version": [],
}


def _resolve(dotted: str):
    mod, _, attr = dotted.rpartition(".")
    return getattr(importlib.import_module(mod), attr)


@pytest.mark.parametrize("dotted", sorted(FROZEN_SIGNATURES))
def test_frozen_signature(dotted):
    assert _params(_resolve(dotted)) == FROZEN_SIGNATURES[dotted]


def test_nvrtc_compile_signature():
    ps = _params(nvrtc.compile)
    assert [(n, k) for n, k, d in ps] == [("source", "POSITIONAL_OR_KEYWORD"), ("name", "KEYWORD_ONLY"),
                                          ("arch", "KEYWORD_ONLY"), ("options", "KEYWORD_ONLY"),
                                          ("headers", "KEYWORD_ONLY")]
    assert ps[1][2] is E and ps[2][2] is E and tuple(ps[3][2]) == () and dict(ps[4][2]) == {}


FROZEN_FIELDS = {
    "mmaemu.SplitK": ["S", "scheme"],
    "mmaemu.SplitDecision": ["S", "scheme", "source", "verified", "table_id", "note"],
    "mmaemu.Profile": ["name", "preset", "gpu_name", "capability", "sm_count", "cublaslt_version", "caller",
                       "torch_version", "workspace_bytes", "align", "splitk_table", "rejects", "maturity",
                       "splitk_maturity"],
    "mmaemu.platform.RejectRule": ["fmt_a", "fmt_b", "out_dtype", "message"],
    "mmaemu.Preset": ["name", "nfma", "frac", "frac_out", "min_exp", "late_c", "slow_chunk", "maturity",
                      "validated_scope", "evidence"],
    "mmaemu.CallPlan": ["validation", "profile", "preset", "preset_maturity", "split", "backend", "fallbacks",
                        "warnings"],
    "mmaemu.backends.ChainRequest": ["device", "fmt_a", "fmt_b", "preset"],
    "mmaemu.backends.BackendStatus": ["name", "available", "source", "validated_here", "gates", "reason"],
    "mmaemu.model.plan.GemmPlan": ["K", "fast", "S", "scheme", "chunks", "identity"],
    "mmaemu.backends.cuda.spec.KernelSpec": ["name", "module", "source", "entry", "sass_archs", "ptx_arch", "min_cc",
                                             "gated", "validated_on", "gates", "priority", "tile_m", "tile_n",
                                             "threads", "smem_bytes", "align_bytes", "k_max", "formats"],
    "mmaemu.backends.cuda.spec.GateResult": ["name", "passed", "reason", "details", "seconds", "cached"],
}


# The classes the v0.1 interface contract declares `@dataclass(frozen=True)`. CallPlan is absent on purpose: the
# contract lists its fields but does not say frozen, so its frozenness is an implementation choice and is not pinned.
CONTRACT_FROZEN = {"SplitK", "SplitDecision", "Profile", "RejectRule", "Preset", "ChainRequest", "BackendStatus",
               "GemmPlan", "KernelSpec", "GateResult"}


def _check_dataclass_contract(dotted: str, cls) -> None:
    import dataclasses

    assert dataclasses.is_dataclass(cls), dotted
    assert [f.name for f in dataclasses.fields(cls)] == FROZEN_FIELDS[dotted]
    if dotted.rsplit(".", 1)[1] in CONTRACT_FROZEN:
        assert cls.__dataclass_params__.frozen, f"{dotted}: the contract declares @dataclass(frozen=True)"


@pytest.mark.parametrize("dotted", sorted(FROZEN_FIELDS))
def test_frozen_dataclass_fields(dotted):
    _check_dataclass_contract(dotted, _resolve(dotted))


def test_frozen_flags_follow_contract():
    """Regression: frozen=True is pinned exactly where the contract says so, and nowhere else."""
    import dataclasses

    names = {d.rsplit(".", 1)[1] for d in FROZEN_FIELDS}
    assert CONTRACT_FROZEN <= names and "CallPlan" not in CONTRACT_FROZEN
    # a non-frozen CallPlan with the contract's fields satisfies the contract (what mmaemu.api ships) ...
    plan = dataclasses.make_dataclass("CallPlan", FROZEN_FIELDS["mmaemu.CallPlan"])
    _check_dataclass_contract("mmaemu.CallPlan", plan)
    frozen_plan = dataclasses.make_dataclass("CallPlan", FROZEN_FIELDS["mmaemu.CallPlan"], frozen=True)
    _check_dataclass_contract("mmaemu.CallPlan", frozen_plan)
    # ... a non-frozen Profile does not, and neither does a Profile with a field the contract does not list
    fields = FROZEN_FIELDS["mmaemu.Profile"]
    with pytest.raises(AssertionError, match="frozen"):
        _check_dataclass_contract("mmaemu.Profile", dataclasses.make_dataclass("Profile", fields))
    with pytest.raises(AssertionError):
        _check_dataclass_contract("mmaemu.Profile",
                                  dataclasses.make_dataclass("Profile", [*fields, "description"], frozen=True))


def test_public_names_resolve():
    for name in mmaemu.__all__:
        assert getattr(mmaemu, name) is not None, name
    assert mmaemu.backends_status is mmaemu.backends.status
    assert mmaemu.model.dot is importlib.import_module("mmaemu.model.chain").dot
    assert isinstance(mmaemu.__version__, str) and mmaemu.__version__.startswith("0.1")
    with pytest.raises(AttributeError):
        mmaemu.does_not_exist  # noqa: B018
    from mmaemu import model, testing  # noqa: F401  (module __getattr__ must not break from-imports)


def test_constants():
    from mmaemu import formats

    assert formats.FP8_MAX == {"e4m3": 448.0, "e5m2": 57344.0}
    assert formats.CANONICAL_NAN_BITS == {torch.float32: 0x7FFFFFFF, torch.bfloat16: 0x7FFF, torch.float16: 0x7FFF,
                                          torch.float8_e4m3fn: 0x7F, torch.float8_e5m2: 0x7F}
    assert spec.KERNEL_ABI_VERSION == 1


def _raises_owner_or_works(fn, *args, **kwargs):
    """A stub raises NotImplementedError('<component>'); an implementation may return or raise anything else."""
    try:
        fn(*args, **kwargs)
    except NotImplementedError as e:
        if type(e) is NotImplementedError:
            assert COMPONENT.fullmatch(str(e)), f"stub message must name the missing component, got {e!r}"
    except Exception:
        pass


def test_stubs_name_their_component():
    z = torch.zeros(16, 16, dtype=torch.float8_e4m3fn)
    s = torch.ones(())
    _raises_owner_or_works(mmaemu.scaled_mm, z, z.t(), s, s)
    _raises_owner_or_works(mmaemu.accumulate, z, z, use_fast_accum=True)
    _raises_owner_or_works(mmaemu.explain, z, z.t(), s, s)
    _raises_owner_or_works(mmaemu.get_preset, "ada")
    _raises_owner_or_works(mmaemu.list_profiles)


def test_missing_module_falls_back_to_stub(monkeypatch, tmp_path):
    monkeypatch.setitem(mmaemu._LAZY, "zz_missing", ("mmaemu.zz_does_not_exist", "f", "mmaemu.zz"))
    try:
        with pytest.raises(NotImplementedError, match=r"^mmaemu\.zz$"):
            mmaemu.zz_missing()
    finally:
        mmaemu.__dict__.pop("zz_missing", None)
    # an ImportError raised inside an existing module is never masked
    pkg = tmp_path / "zzpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("import zz_no_such_dependency\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(mmaemu._LAZY, "zz_broken", ("zzpkg", None, "mmaemu.zz"))
    try:
        with pytest.raises(ModuleNotFoundError):
            mmaemu.zz_broken  # noqa: B018
    finally:
        mmaemu.__dict__.pop("zz_broken", None)
        sys.modules.pop("zzpkg", None)


def test_legacy_modules_removed():
    for name in ("mmaemu.reference", "mmaemu.vectors"):
        assert importlib.util.find_spec(name) is None, name


# --- errors.py ---------------------------------------------------------------------------------------------------

def test_errors_exactly_as_spec():
    expected = {
        "MmaemuError": (Exception,),
        "ReferenceRejectsError": (errors.MmaemuError, RuntimeError),
        "UnsupportedFeatureError": (errors.MmaemuError, NotImplementedError),
        "SplitKUnknownError": (errors.MmaemuError, LookupError),
        "BackendUnavailableError": (errors.MmaemuError, RuntimeError),
        "ExperimentalPresetWarning": (UserWarning,),
        "UnverifiedSplitWarning": (UserWarning,),
        "BackendFallbackWarning": (RuntimeWarning,),
    }
    classes = {n for n, v in vars(errors).items() if isinstance(v, type) and v.__module__ == errors.__name__}
    assert classes == set(expected) == set(errors.__all__)
    for name, bases in expected.items():
        assert getattr(errors, name).__bases__ == bases, name
        assert getattr(mmaemu, name) is getattr(errors, name)


# --- A3: comparator ----------------------------------------------------------------------------------------------

def _from_bits(values, dtype):
    iv = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[torch.empty((), dtype=dtype).element_size()]
    return torch.tensor(np.array(values, dtype=np.uint64).astype(np.int64)).to(iv).view(dtype)


@pytest.mark.parametrize("dtype,iv", [(torch.float8_e4m3fn, torch.uint8), (torch.float8_e5m2, torch.uint8),
                                      (torch.bfloat16, torch.int16), (torch.float16, torch.int16),
                                      (torch.float32, torch.int32), (torch.float64, torch.int64),
                                      (torch.bool, torch.uint8), (torch.int32, torch.int32),
                                      (torch.uint8, torch.uint8)])
def test_bits_view(dtype, iv):
    t = torch.zeros(3, 2, dtype=dtype)
    b = bits(t)
    assert b.dtype == iv and b.shape == t.shape and b.data_ptr() == t.data_ptr()


def test_bits_rejects_complex():
    with pytest.raises(TypeError):
        bits(torch.zeros(2, dtype=torch.complex64))


ZERO_BITS = {torch.float32: (0x00000000, 0x80000000), torch.bfloat16: (0x0000, 0x8000),
             torch.float16: (0x0000, 0x8000), torch.float8_e4m3fn: (0x00, 0x80), torch.float8_e5m2: (0x00, 0x80),
             torch.float64: (0, 0x8000000000000000)}
NAN_BITS = {torch.float32: (0x7FFFFFFF, 0x7FC00000), torch.bfloat16: (0x7FFF, 0x7FC0),
            torch.float16: (0x7FFF, 0x7E00), torch.float8_e4m3fn: (0x7F, 0xFF), torch.float8_e5m2: (0x7F, 0x7E),
            torch.float64: (0x7FFFFFFFFFFFFFFF, 0x7FF8000000000000)}


@pytest.mark.parametrize("dtype", list(ZERO_BITS), ids=str)
def test_comparator_zero_sign(dtype):
    pz, nz = ZERO_BITS[dtype]
    a, e = _from_bits([pz, pz], dtype), _from_bits([pz, nz], dtype)
    assert_bitwise_equal(a, a.clone())
    for nan in ("bits", "mask"):
        with pytest.raises(AssertionError, match=r"zero_sign=1, nan_payload=0, value=0"):
            assert_bitwise_equal(a, e, nan=nan)


@pytest.mark.parametrize("dtype", list(NAN_BITS), ids=str)
def test_comparator_nan_payload(dtype):
    n1, n2 = NAN_BITS[dtype]
    a, e = _from_bits([n1, 0], dtype), _from_bits([n2, 0], dtype)
    assert bool(torch.isnan(a.float())[0]) and bool(torch.isnan(e.float())[0])
    with pytest.raises(AssertionError, match=r"zero_sign=0, nan_payload=1, value=0"):
        assert_bitwise_equal(a, e)  # default nan="bits"
    assert_bitwise_equal(a, e, nan="mask")  # nan=mask: test of the mask mode itself
    assert_bitwise_equal(a, a.clone(), nan="bits")


def test_comparator_value_and_nan_position():
    a = torch.tensor([1.0, float("nan"), 3.0])
    e = torch.tensor([1.0, 2.0, 3.0])
    e[2] = torch.nextafter(e[2], torch.tensor(4.0))  # one fp32 ulp above 3.0
    for nan in ("bits", "mask"):
        with pytest.raises(AssertionError, match=r"zero_sign=0, nan_payload=0, value=2"):
            assert_bitwise_equal(a, e, nan=nan)
    assert_bitwise_equal(torch.tensor([float("inf"), -float("inf")]), torch.tensor([float("inf"), -float("inf")]))


def test_comparator_report_format():
    a = torch.zeros(4, 8)
    e = a.clone()
    e.view(-1)[:13] = -0.0
    e[3, 7] = 5.0
    with pytest.raises(AssertionError) as ei:
        assert_bitwise_equal(a, e, msg="ctx")
    text = str(ei.value)
    assert text.startswith("ctx: bitwise mismatch in 14 of 32 elements (zero_sign=13, nan_payload=0, value=1)")
    assert text.count("\n  (") == 10 and "... 4 more" in text
    assert "(0, 0): actual 0.0 [0x00000000]  expected -0.0 [0x80000000]  zero_sign" in text


def test_comparator_dtype_shape_and_args():
    with pytest.raises(AssertionError, match="dtype mismatch"):
        assert_bitwise_equal(torch.zeros(2), torch.zeros(2, dtype=torch.float64))
    with pytest.raises(AssertionError, match="shape mismatch"):
        assert_bitwise_equal(torch.zeros(2), torch.zeros(1, 2))
    with pytest.raises(ValueError):
        assert_bitwise_equal(torch.zeros(2), torch.zeros(2), nan="ignore")
    with pytest.raises(TypeError):
        assert_bitwise_equal([0.0], [0.0])
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert_bitwise_equal(x.t(), x.t().contiguous())  # strided views
    assert_bitwise_equal(np.float32([1, -0.0]), torch.tensor([1, -0.0]))
    assert_bitwise_equal(torch.zeros(0, 3), torch.zeros(0, 3))
    with pytest.raises(AssertionError):
        assert_bitwise_equal(torch.tensor([1, 2], dtype=torch.int32), torch.tensor([1, 3], dtype=torch.int32))


@pytest.mark.gpu
def test_comparator_cuda_vs_cpu(cuda_device):
    x = torch.tensor([0.0, -0.0, 1.5], device=cuda_device)
    assert_bitwise_equal(x, x.cpu())
    with pytest.raises(AssertionError, match="zero_sign=1"):
        assert_bitwise_equal(x, torch.tensor([0.0, 0.0, 1.5]))


# --- A3: tcemu blocker -------------------------------------------------------------------------------------------

def test_tcemu_blocked_in_process():
    for name in ("tcemu", "tcemu.reference", "tcemu._ext"):
        with pytest.raises(ModuleNotFoundError, match="blocked"):
            importlib.import_module(name)
    assert not [m for m in sys.modules if m == "tcemu" or m.startswith("tcemu.")]


def test_tcemu_blocked_in_subprocess(tmp_path):
    fake = _fake_packages(tmp_path / "fake")
    probe = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(TESTS)!r})
        if sys.argv[1] == "block":
            import conftest
            conftest.install_tcemu_blocker()
        res = []
        for name in ("tcemu", "tcemu.reference", "tcemu_ada_truth"):
            try:
                __import__(name); res.append(name + ":ok")
            except ImportError as e:
                res.append(name + ":" + type(e).__name__ + ("(blocked)" if "blocked" in str(e) else ""))
        print(" ".join(res))
    """)
    env = dict(os.environ, PYTHONPATH=str(fake), PYTHONDONTWRITEBYTECODE="1")
    run = lambda mode: subprocess.run([sys.executable, "-c", probe, mode], capture_output=True, text=True, env=env,
                                      timeout=300)
    control, blocked = run("control"), run("block")
    assert control.returncode == 0 and blocked.returncode == 0, control.stderr + blocked.stderr
    assert control.stdout.split() == ["tcemu:ok", "tcemu.reference:ok", "tcemu_ada_truth:ok"]
    assert blocked.stdout.split() == ["tcemu:ModuleNotFoundError(blocked)",
                                      "tcemu.reference:ModuleNotFoundError(blocked)", "tcemu_ada_truth:ok"]


@pytest.mark.parametrize("leak", [False, True])
def test_session_fails_if_tcemu_was_imported(tmp_path, leak):
    """The session-end check turns a passing session into a failure if 'tcemu' got into sys.modules."""
    shutil.copy(TESTS / "conftest.py", tmp_path / "conftest.py")
    body = "import sys, types\n"
    body += "def test_x():\n"
    body += "    sys.modules['tcemu'] = types.ModuleType('tcemu')\n" if leak else "    pass\n"
    (tmp_path / "test_leak.py").write_text(body)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
                       capture_output=True, text=True, env=env, cwd=tmp_path, timeout=300)
    if leak:
        assert r.returncode == 1 and "1 passed" in r.stdout and "'tcemu' was imported" in r.stdout + r.stderr, \
            r.stdout + r.stderr
    else:
        assert r.returncode == 0, r.stdout + r.stderr


LEGACY_BODY = "from mmaemu.api import _legacy_reference as R\n\n\ndef test_x():\n    assert R is not None\n"


@pytest.mark.parametrize("api, name, expect", [
    ("X = 1\n", "test_recorded_hw.py", "skip"),                         # the shim is gone
    ("X = 1\n", "test_reference.py", "skip"),
    ("_legacy_reference = object()\n", "test_recorded_hw.py", "pass"),  # the shim exists
    ("import no_such_module_mmaemu_xyz\n", "test_recorded_hw.py", "error"),   # a broken api is not hidden
    ("X = 1\n", "test_other.py", "error"),                              # only the legacy file names
], ids=["shim-gone", "shim-gone-other-file", "shim-present", "api-broken", "not-legacy"])
def test_superseded_legacy_tests_are_counted_skips(tmp_path, api, name, expect):
    """Regression: without the bootstrap shim, the legacy tests that import
    mmaemu.api._legacy_reference are one visible, counted skip each (not a collection error that stops the suite);
    while the shim exists they run; anything else still errors."""
    pkg = tmp_path / "pkg" / "mmaemu"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "api.py").write_text(api)
    tests = tmp_path / "tests"
    tests.mkdir()
    shutil.copy(TESTS / "conftest.py", tests / "conftest.py")
    (tests / name).write_text(LEGACY_BODY)
    (tests / "test_ok.py").write_text("def test_ok():\n    pass\n")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(tmp_path / "pkg"))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", "tests"],
                       capture_output=True, text=True, env=env, cwd=tmp_path, timeout=300)
    out = r.stdout + r.stderr
    if expect == "skip":
        assert r.returncode == 0 and "1 passed, 1 skipped" in out and "superseded legacy test" in out, out
        assert f"{name} is a pre-v0.1 test" in out and "error" not in out.lower(), out
    elif expect == "pass":
        assert r.returncode == 0 and "2 passed" in out and "skipped" not in out, out
    else:
        assert r.returncode == 2 and "1 error" in out and "superseded legacy test:" not in out, out


# --- backends: selector, status, reference ------------------------------------------------------------------------

def _registry_present() -> bool:
    return importlib.util.find_spec("mmaemu.backends.registry") is not None


def test_select_returns_reference_on_cpu():
    from mmaemu.backends import Backend, select
    from mmaemu.backends.reference import REFERENCE, ReferenceBackend

    req = ChainRequest(device=torch.device("cpu"), fmt_a="e4m3", fmt_b="e4m3", preset="ada")
    for name in ("auto", "reference"):
        be = select(req, name)
        assert be.name == "reference" and isinstance(be, Backend) and be.supports(req) is None
    assert isinstance(REFERENCE, ReferenceBackend)
    if not _registry_present():
        assert select(req) is REFERENCE
    fn = REFERENCE.chain_fn(req)
    _raises_owner_or_works(fn, torch.zeros(1, 16, dtype=torch.uint8), torch.zeros(1, 16, dtype=torch.uint8))
    with pytest.raises(ValueError if not _registry_present() else (ValueError, errors.BackendUnavailableError)):
        select(req, "cuda:v7")
    with pytest.raises(TypeError):
        select(req, 3)


@pytest.mark.skipif(_registry_present(), reason="the registry decides when it is installed")
def test_select_explicit_cuda_unavailable_without_registry():
    from mmaemu.backends import select

    req = ChainRequest(device=torch.device("cpu"), fmt_a="e4m3", fmt_b="e4m3", preset="ada")
    for name in ("cuda", "cuda:v1", "cuda:tc3"):
        with pytest.raises(errors.BackendUnavailableError):
            select(req, name)
    with pytest.raises(NotImplementedError, match=r"^mmaemu\.backends\.registry$"):
        mmaemu.selftest()


def test_status_reports_reference():
    st = mmaemu.backends_status(torch.device("cpu"))
    assert isinstance(st, dict) and isinstance(st["reference"], BackendStatus) and st["reference"].available


# --- kernel spec registry ----------------------------------------------------------------------------------------

def _v1_like(**kw):
    base = dict(name="v1", module="v1", source="v1.cu", entry="mmaemu_v1_e4m3e4m3", sass_archs=(75, 80, 86, 89),
                ptx_arch=75, min_cc=(7, 5), gated=False, validated_on=frozenset({(8, 9)}), gates=(), priority=10,
                tile_m=64, tile_n=64, threads=256, smem_bytes=0, align_bytes=4, k_max=1 << 31)
    base.update(kw)
    return spec.KernelSpec(**base)


def test_kernel_spec_validation_and_launch_config():
    s = _v1_like()
    assert s.formats == (("e4m3", "e4m3"),)
    assert s.launch_config(M=65, N=129) == ((3, 2, 1), (256, 1, 1), 0)
    for bad in (dict(entry="v1"), dict(validated_on={(8, 9)}), dict(sass_archs=[89]), dict(align_bytes=3),
                dict(gates=["window"]), dict(k_max=16), dict(min_cc=(8,))):
        with pytest.raises(ValueError):
            _v1_like(**bad)


def test_register_gate(monkeypatch):
    monkeypatch.setattr(spec, "GATES", {})

    def g1(s, d):
        return spec.GateResult("g", True, "", {}, 0.0, False)

    spec.register_gate("g", g1)
    spec.register_gate("g", g1)  # idempotent
    assert spec.GATES["g"] is g1

    def other(s, d):
        return g1(s, d)

    with pytest.raises(ValueError, match="already registered"):
        spec.register_gate("g", other)
    with pytest.raises(TypeError):
        spec.register_gate("h", 42)
    assert isinstance(spec.PROBE_MODULES, dict)


# --- A5: NVRTC and driver bindings -------------------------------------------------------------------------------

def _need_nvrtc():
    try:
        nvrtc.libnvrtc()
    except errors.BackendUnavailableError as e:
        pytest.skip(f"libnvrtc unavailable: {e}")


@pytest.mark.parametrize("arch", [89, 120])
def test_nvrtc_compile(arch):
    _need_nvrtc()
    cubin = nvrtc.compile(K_SRC, name="k.cu", arch=arch)
    assert isinstance(cubin, bytes) and len(cubin) > 0 and cubin[:4] == b"\x7fELF"


def test_nvrtc_fixed_options_headers_and_errors(monkeypatch):
    _need_nvrtc()
    major, minor = nvrtc.version()
    assert major >= 11 and minor >= 0
    real = nvrtc.libnvrtc()
    seen = []

    class Spy:
        def __getattr__(self, name):
            return getattr(real, name)

        def nvrtcCompileProgram(self, prog, n, arr):
            seen.append([arr[i].decode() for i in range(n)])
            return real.nvrtcCompileProgram(prog, n, arr)

    monkeypatch.setattr(nvrtc, "libnvrtc", lambda: Spy())
    src = '#include "val.cuh"\nextern "C" __global__ void k(float* x){ x[0] = VAL * SCALE; }'
    cubin = nvrtc.compile(src, name="k.cu", arch=89, options=["-DSCALE=2.f"], headers={"val.cuh": "#define VAL 1.f"})
    assert cubin[:4] == b"\x7fELF"
    fixed = {"--gpu-architecture=sm_89", "--fmad=false", "-default-device", "--std=c++17"}
    assert fixed | {"-DSCALE=2.f"} <= set(seen[-1]), seen[-1]
    for bad in ("--fmad=true", "-fmad=true", "--use_fast_math", "--ftz=true", "--prec-div=false", "-arch=sm_80",
                "--gpu-architecture=sm_80"):
        with pytest.raises(ValueError):  # arithmetic / architecture flags cannot be overridden
            nvrtc.compile(K_SRC, name="k.cu", arch=89, options=[bad])
    with pytest.raises(TypeError):
        nvrtc.compile(K_SRC, name="k.cu", arch=89, options="-DX=1")
    with pytest.raises(ValueError):
        nvrtc.compile(K_SRC, name="k.cu", arch="sm_89")
    with pytest.raises(errors.BackendUnavailableError) as ei:
        nvrtc.compile("this is not CUDA", name="bad.cu", arch=89)
    assert "bad.cu" in str(ei.value) and "error" in str(ei.value).lower()  # the compile log is carried


@pytest.mark.parametrize("env,call", [("MMAEMU_NVRTC_PATH", "nvrtc.libnvrtc()"),
                                      ("MMAEMU_LIBCUDA_PATH", "driver.libcuda()")])
def test_explicit_library_path_is_not_bypassed(env, call):
    """An explicitly configured but unloadable library raises; the search order is not silently used instead."""
    code = ("from mmaemu.backends.cuda import driver, nvrtc\nfrom mmaemu.errors import BackendUnavailableError\n"
            f"try:\n    {call}\nexcept BackendUnavailableError:\n    print('unavailable')\n"
            "else:\n    print('loaded')\n")
    r = _run(code, env_extra={env: "/nonexistent/lib.so", "MMAEMU_DISABLE_CUDA": ""})
    assert r.returncode == 0 and r.stdout.strip() == "unavailable", r.stdout + r.stderr


def test_driver_cpu_side_checks():
    """Argument errors are raised before the driver is touched (this test runs with CUDA hidden too)."""
    with pytest.raises(TypeError):  # a bare int: 4-byte int or 8-byte pointer? never guessed
        driver.launch(0, (1, 1, 1), (1, 1, 1), 0, 0, [5])
    with pytest.raises((ValueError, errors.BackendUnavailableError)):
        driver.launch(0, (1, 0, 1), (1, 1, 1), 0, 0, [])
    driver.check(0, "cuNothing")
    with pytest.raises(errors.BackendUnavailableError) as ei:
        driver.check(1, "cuFoo")
    assert "cuFoo" in str(ei.value)


@pytest.mark.gpu
def test_driver_launch(cuda_device):
    """Optional A5 GPU part: an NVRTC cubin for this device's capability, loaded and launched through driver.py in
    torch's context on torch's current stream (and on a side stream), writes 1.0."""
    _need_nvrtc()
    major, minor = torch.cuda.get_device_capability(cuda_device)
    arch = 10 * major + minor
    x = torch.zeros(2, device=cuda_device)  # makes torch's primary context current on this thread
    ordinal = cuda_device.index
    assert driver.current_context() != 0
    assert driver.driver_version() >= 12000
    assert driver.device_attribute(ordinal, driver.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) == major
    assert driver.device_attribute(ordinal, driver.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR) == minor
    props = torch.cuda.get_device_properties(cuda_device)
    sms = driver.device_attribute(ordinal, driver.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    assert sms == props.multi_processor_count
    assert driver.device_uuid(ordinal) == "GPU-" + str(props.uuid)
    mod = driver.module_load(nvrtc.compile(K_SRC, name="k.cu", arch=arch))
    fn = driver.get_function(mod, "k")
    driver.set_max_dynamic_smem(fn, 0)
    with torch.cuda.device(cuda_device):
        driver.launch(fn, (1, 1, 1), (1, 1, 1), 0, torch.cuda.current_stream().cuda_stream, [x])
        side = torch.cuda.Stream()
        y = torch.zeros(1, device=cuda_device)
        side.wait_stream(torch.cuda.current_stream())
        driver.launch(fn, (1, 1, 1), (1, 1, 1), 0, side, [y])
        torch.cuda.synchronize()
    assert_bitwise_equal(x.cpu(), torch.tensor([1.0, 0.0]))
    assert_bitwise_equal(y.cpu(), torch.tensor([1.0]))
    with pytest.raises(errors.BackendUnavailableError, match="cuModuleGetFunction"):
        driver.get_function(mod, "no_such_kernel")
    driver.module_unload(mod)
