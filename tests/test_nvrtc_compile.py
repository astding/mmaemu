"""Every discovered kernel source and probe module compiles through NVRTC for every architecture in its list
(CPU only). No GPU, no nvcc: only libnvrtc (torch's nvidia-cuda-nvrtc-cu12 wheel)."""
from __future__ import annotations

import re

import pytest

from mmaemu.backends import registry
from mmaemu.backends.cuda import modules, nvrtc

try:
    nvrtc.libnvrtc()
    _NO_NVRTC = None
except Exception as e:  # noqa: BLE001
    _NO_NVRTC = f"NVRTC unavailable: {e}"

pytestmark = pytest.mark.skipif(_NO_NVRTC is not None, reason=_NO_NVRTC or "")

PROBE_ARCHS = (80, 86, 89, 90, 100, 120)


def _cases():
    out = []
    for s in registry.discover():
        archs = sorted(set(s.sass_archs) | ({s.ptx_arch} if s.ptx_arch is not None else set()))
        out += [(s.module, a) for a in archs]
    try:
        from mmaemu.backends.cuda.fast import specs as fast_specs

        probe_archs = tuple(getattr(fast_specs, "PROBE_SASS_ARCHS", None) or getattr(fast_specs, "PROBE_ARCHS", PROBE_ARCHS))
    except Exception:  # noqa: BLE001
        probe_archs = PROBE_ARCHS
    for stem in registry.probe_modules():
        out += [(stem, a) for a in probe_archs]
    return out


CASES = _cases()


def test_ieee_kernels_are_discovered():
    names = [s.name for s in registry.discover()]
    assert "v1" in names and "v4" in names
    assert ("v1", 75) in CASES and ("v1", 120) in CASES and ("v4", 80) in CASES and ("v4", 120) in CASES


@pytest.mark.parametrize("module,arch", CASES, ids=[f"{m}-sm_{a}" for m, a in CASES])
def test_compiles(module, arch):
    if arch not in nvrtc.supported_archs():
        pytest.skip(f"NVRTC {nvrtc.version()} cannot target sm_{arch}")
    src, files = modules.source_files(module)
    headers = {n: t for n, t in files.items() if n != src}
    cubin = nvrtc.compile(files[src], name=src, arch=arch, headers=headers)
    assert len(cubin) > 1000 and cubin[:4] == b"\x7fELF"
    info = nvrtc.last_compile
    assert info["name"] == src and info["arch"] == arch
    assert "--fmad=false" in info["options"]
    assert f"--gpu-architecture=sm_{arch}" in info["options"]
    assert "warning" not in info["log"].lower(), info["log"]


def test_sources_have_no_torch_or_host_runtime_headers():
    for s in registry.discover():
        src, files = modules.source_files(s.module)
        for name, text in files.items():
            code = re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)  # comments may name what was removed
            includes = re.findall(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', code, re.M)
            for inc in includes:
                assert inc in files or inc in ("stdint.h", "stddef.h"), f"{name} includes {inc}"
            assert "PYBIND11_MODULE" not in code and "TORCH_CHECK" not in code and "at::" not in code, name
            if name == src:
                assert 'extern "C" __global__' in text and s.entry in text


def test_arithmetic_options_cannot_be_overridden():
    for bad in ("--fmad=true", "-use_fast_math", "--ftz=true", "-arch=sm_80"):
        with pytest.raises(ValueError):
            nvrtc.build_options(89, [bad])
    opts = nvrtc.build_options(89)
    assert opts == ["--gpu-architecture=sm_89", "--fmad=false", "-default-device", "--std=c++17"]
