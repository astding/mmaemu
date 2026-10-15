"""Minimal ctypes binding of NVRTC: compile a CUDA C++ source to a cubin at run time (a port of the tcemu audit's
p4-system-backends/d_pack/rtc.py).

Nothing happens at import; libnvrtc is loaded on the first call. Search order:
  1. $MMAEMU_NVRTC_PATH (if set, only that path);
  2. the nvidia.cuda_nvrtc pip package (a dependency of torch's CUDA wheels): <pkg>/lib/libnvrtc.so.<major>;
  3. $CONDA_PREFIX/lib;  4. $CUDA_HOME/lib64;  5. ctypes.util.find_library("nvrtc").
Every failure (library missing, compile error with its log) raises mmaemu.errors.BackendUnavailableError.

Runtime helpers: build_options() (the exact option list compile() passes; part of the runtime's cubin cache key),
library_path(), and last_compile (diagnostics of the latest compile: name, arch, options, log, seconds).

compile() always uses the arithmetic flags of the shipped fatbins (--fmad=false, no fast math) plus
-default-device --std=c++17 --gpu-architecture=sm_<arch>; options that would change floating-point code generation
or the target architecture are refused (ValueError), so an NVRTC build can never differ numerically by flags.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import importlib.util
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from mmaemu.errors import BackendUnavailableError

__all__ = ["libnvrtc", "version", "compile", "supported_archs", "search_paths", "FIXED_OPTIONS", "build_options",
           "library_path", "last_compile"]

FIXED_OPTIONS = ("--fmad=false", "-default-device", "--std=c++17")
_FORBIDDEN = ("--fmad", "-fmad", "--use_fast_math", "-use_fast_math", "--ftz", "-ftz", "--prec-div", "-prec-div",
              "--prec-sqrt", "-prec-sqrt", "--gpu-architecture", "-arch")
_SONAME = re.compile(r"^libnvrtc\.so(\.(\d+))?$")
last_compile: dict = {}  # diagnostics of the latest compile(): name, arch, options, log, seconds

_c_size_p = ctypes.POINTER(ctypes.c_size_t)
_SIGNATURES = {
    "nvrtcVersion": (ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)),
    "nvrtcCreateProgram": (ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
                           ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p)),
    "nvrtcCompileProgram": (ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)),
    "nvrtcGetProgramLogSize": (ctypes.c_void_p, _c_size_p),
    "nvrtcGetProgramLog": (ctypes.c_void_p, ctypes.c_char_p),
    "nvrtcGetCUBINSize": (ctypes.c_void_p, _c_size_p),
    "nvrtcGetCUBIN": (ctypes.c_void_p, ctypes.c_char_p),
    "nvrtcDestroyProgram": (ctypes.POINTER(ctypes.c_void_p),),
    "?nvrtcGetNumSupportedArchs": (ctypes.POINTER(ctypes.c_int),),
    "?nvrtcGetSupportedArchs": (ctypes.POINTER(ctypes.c_int),),
}


def _dir_candidates(d: str | os.PathLike | None) -> list[str]:
    if not d or not Path(d).is_dir():
        return []
    found = []
    for p in Path(d).iterdir():
        m = _SONAME.match(p.name)
        if m:
            found.append((int(m.group(2) or -1), str(p)))
    return [p for _, p in sorted(found, reverse=True)]  # highest major first, unversioned last


def _pip_package_dir() -> str | None:
    try:
        spec = importlib.util.find_spec("nvidia.cuda_nvrtc")
    except (ImportError, ValueError):
        return None
    locs = list(spec.submodule_search_locations or []) if spec else []
    return str(Path(locs[0]) / "lib") if locs else None


def search_paths() -> list[str]:
    """The candidate library paths in search order (without $MMAEMU_NVRTC_PATH)."""
    out = _dir_candidates(_pip_package_dir())
    if os.environ.get("CONDA_PREFIX"):
        out += _dir_candidates(Path(os.environ["CONDA_PREFIX"]) / "lib")
    if os.environ.get("CUDA_HOME"):
        out += _dir_candidates(Path(os.environ["CUDA_HOME"]) / "lib64")
    name = ctypes.util.find_library("nvrtc")
    if name:
        out.append(name)
    return list(dict.fromkeys(out))


def _open(path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(path)
    for name, argtypes in _SIGNATURES.items():
        optional = name.startswith("?")
        name = name.lstrip("?")
        try:
            fn = getattr(lib, name)
        except AttributeError:
            if optional:
                continue
            raise BackendUnavailableError(f"{path}: symbol {name} not found") from None
        fn.argtypes = argtypes
        fn.restype = ctypes.c_int
    lib.nvrtcGetErrorString.argtypes = (ctypes.c_int,)
    lib.nvrtcGetErrorString.restype = ctypes.c_char_p
    lib._mmaemu_path = path
    return lib


@lru_cache(maxsize=None)
def _load() -> ctypes.CDLL:
    env = os.environ.get("MMAEMU_NVRTC_PATH")
    candidates = [env] if env else search_paths()
    errors = []
    for path in candidates:
        try:
            return _open(path)
        except OSError as e:
            errors.append(f"{path}: {e}")
    where = "MMAEMU_NVRTC_PATH" if env else "search order: MMAEMU_NVRTC_PATH, nvidia.cuda_nvrtc, $CONDA_PREFIX/lib, " \
                                            "$CUDA_HOME/lib64, find_library('nvrtc')"
    raise BackendUnavailableError(f"libnvrtc not found ({where})" + (": " + "; ".join(errors) if errors else ""))


def libnvrtc() -> ctypes.CDLL:
    """The loaded libnvrtc (cached after the first success; failures are not cached)."""
    return _load()


libnvrtc.cache_clear = _load.cache_clear  # for tests that change MMAEMU_NVRTC_PATH


def library_path() -> str | None:
    """Path of the loaded libnvrtc (None if not loaded yet)."""
    return getattr(_load(), "_mmaemu_path", None) if _load.cache_info().currsize else None


def _check(lib, res: int, call: str, log: str = "") -> None:
    if res != 0:
        msg = (lib.nvrtcGetErrorString(res) or b"?").decode()
        raise BackendUnavailableError(f"{call} failed: {msg} ({res})" + (f"\n{log}" if log else ""))


def version() -> tuple[int, int]:
    """(major, minor) of the loaded libnvrtc."""
    lib = libnvrtc()
    a, b = ctypes.c_int(), ctypes.c_int()
    _check(lib, lib.nvrtcVersion(ctypes.byref(a), ctypes.byref(b)), "nvrtcVersion")
    return a.value, b.value


def supported_archs() -> tuple[int, ...]:
    """SASS architectures this libnvrtc can target, e.g. (50, 52, ..., 89, 90, 100, 120)."""
    lib = libnvrtc()
    if not hasattr(lib, "nvrtcGetSupportedArchs"):
        raise BackendUnavailableError("this libnvrtc does not report its supported architectures")
    n = ctypes.c_int()
    _check(lib, lib.nvrtcGetNumSupportedArchs(ctypes.byref(n)), "nvrtcGetNumSupportedArchs")
    arr = (ctypes.c_int * n.value)()
    _check(lib, lib.nvrtcGetSupportedArchs(arr), "nvrtcGetSupportedArchs")
    return tuple(arr)


def _check_options(options: Sequence[str]) -> list[bytes]:
    if isinstance(options, (str, bytes)):
        raise TypeError("options must be a sequence of str, not a single string")
    out = []
    for o in options:
        if not isinstance(o, str):
            raise TypeError(f"option {o!r} is not a str")
        if o.startswith(_FORBIDDEN):
            raise ValueError(f"option {o!r} is refused: arithmetic and architecture flags are fixed "
                             f"({' '.join(FIXED_OPTIONS)} --gpu-architecture=sm_<arch>)")
        out.append(o.encode())
    return out


def build_options(arch: int, options: Sequence[str] = ()) -> list[str]:
    """The full option list compile() passes for sm_<arch>: --gpu-architecture=sm_<arch>, FIXED_OPTIONS, then
    `options` (refused if they touch arithmetic or architecture flags)."""
    return [f"--gpu-architecture=sm_{int(arch)}", *FIXED_OPTIONS] + [o.decode() for o in _check_options(options)]


def compile(source: str, *, name: str, arch: int, options: Sequence[str] = (),
            headers: Mapping[str, str] = MappingProxyType({})) -> bytes:
    """Compile `source` (file name `name`, used in diagnostics and #line) for sm_<arch> and return the cubin.

    headers: include name -> source text, resolvable by #include "<name>". The options always start with
    --gpu-architecture=sm_<arch> --fmad=false -default-device --std=c++17; extra `options` (e.g. "-DX=1") follow."""
    if isinstance(arch, bool) or not isinstance(arch, int) or arch < 30:
        raise ValueError(f"arch must be an int such as 89 or 120, got {arch!r}")
    user = _check_options(options)
    lib = libnvrtc()
    hnames = [k.encode() for k in headers]
    hsrcs = [headers[k].encode() for k in headers]
    n = len(hnames)
    prog = ctypes.c_void_p()
    t0 = time.monotonic()
    _check(lib, lib.nvrtcCreateProgram(ctypes.byref(prog), source.encode(), name.encode(), n,
                                       (ctypes.c_char_p * max(n, 1))(*hsrcs), (ctypes.c_char_p * max(n, 1))(*hnames)),
           "nvrtcCreateProgram")
    try:
        opts = [f"--gpu-architecture=sm_{arch}".encode()] + [o.encode() for o in FIXED_OPTIONS] + user
        res = lib.nvrtcCompileProgram(prog, len(opts), (ctypes.c_char_p * len(opts))(*opts))
        size = ctypes.c_size_t()
        log = ""
        if lib.nvrtcGetProgramLogSize(prog, ctypes.byref(size)) == 0 and size.value > 1:
            buf = ctypes.create_string_buffer(size.value)
            if lib.nvrtcGetProgramLog(prog, buf) == 0:
                log = buf.value.decode(errors="replace")
        last_compile.clear()
        last_compile.update(name=name, arch=arch, options=[o.decode() for o in opts], log=log, ok=res == 0)
        _check(lib, res, f"nvrtcCompileProgram({name}, sm_{arch})", log)
        _check(lib, lib.nvrtcGetCUBINSize(prog, ctypes.byref(size)), "nvrtcGetCUBINSize")
        buf = ctypes.create_string_buffer(size.value)
        _check(lib, lib.nvrtcGetCUBIN(prog, buf), "nvrtcGetCUBIN")
        last_compile["seconds"] = time.monotonic() - t0
        return buf.raw
    finally:
        lib.nvrtcDestroyProgram(ctypes.byref(prog))
