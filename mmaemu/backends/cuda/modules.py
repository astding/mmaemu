"""Kernel module loader and cache: prebuilt fatbin first, NVRTC second.

Modules are cached per (device ordinal, CUDA context): a module, and its compile-time __constant__ tables, belongs to
the context it was loaded in, so a second context (or a second device) loads its own copy (the tcemu prototype gave
zeros there).

Loader ladder for module `m` on a device with compute capability sm_XY:
  1. fatbin: <fatbin dir>/manifest.json lists m with its file, sha256 and source_sha256. The file's sha256 must match
     the manifest, and source_sha256 must match the kernel sources shipped next to this file (a fatbin built from other
     sources is stale and not used). The driver picks the SASS for the device (or JIT-compiles the PTX).
  2. NVRTC: the shipped source is compiled for sm_XY with the fixed arithmetic options of nvrtc.py. The cubin is cached
     in $MMAEMU_CACHE_DIR/nvrtc/<sha256>.cubin (atomic write), keyed by sha256(source + headers + options + arch +
     NVRTC version). If the cache directory cannot be written, the cubin is kept in memory only.
  3. Otherwise BackendUnavailableError naming both reasons (the registry then falls back to the next backend).

The fatbin directory is the package's bin/ (release fatbins, committed by the maintainer at the release-candidate
freeze).
Development and tests may point MMAEMU_FATBIN_DIR at a local build (tools/build_fatbin.py --out build/fatbin).
MMAEMU_TEST_FORCE_FAIL=fatbin and/or nvrtc make the corresponding rung fail (test-only).
Nothing here initialises CUDA at import time.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from mmaemu.backends.cuda import driver
from mmaemu.errors import BackendUnavailableError

__all__ = ["get_function", "launch", "module_info", "source_files", "source_sha256", "cache_dir", "fatbin_dir",
           "forced_failures", "stats", "clear", "MANIFEST_FORMAT", "LoadedModule"]

HERE = Path(__file__).resolve().parent
CSRC = HERE / "csrc"
BIN = HERE / "bin"
MANIFEST_FORMAT = "mmaemu-fatbin/1"
DEFAULT_DYNAMIC_SMEM = 48 * 1024

_LOCK = threading.RLock()
_STATS = {"fatbin_loads": 0, "nvrtc_compiles": 0, "nvrtc_disk_hits": 0, "nvrtc_memory_hits": 0, "module_loads": 0,
          "cache_write_errors": 0}
_NVRTC_MEM: dict[str, bytes] = {}
_MANIFESTS: dict[tuple, dict] = {}


@dataclass
class LoadedModule:
    """A module loaded in one (device, context)."""
    module: str
    ordinal: int
    context: int
    handle: int
    source: str                 # "fatbin" | "nvrtc"
    sha256: str                 # of the loaded image (fatbin file or cubin)
    arch: int                   # sm number of the device
    fatbin_error: str | None    # why the fatbin was not used (None if it was)
    functions: dict = field(default_factory=dict)
    smem_configured: dict = field(default_factory=dict)


_MODULES: dict[tuple[int, int, str], LoadedModule] = {}


# ----------------------------------------------------------------------------------------------------------------
# environment

def cache_dir() -> Path:
    """$MMAEMU_CACHE_DIR, default ${XDG_CACHE_HOME:-~/.cache}/mmaemu (read at call time)."""
    env = os.environ.get("MMAEMU_CACHE_DIR")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".cache") / "mmaemu"


def fatbin_dir() -> Path:
    """The fatbin directory: $MMAEMU_FATBIN_DIR (development/tests) or the package's bin/."""
    env = os.environ.get("MMAEMU_FATBIN_DIR")
    return Path(env) if env else BIN


def forced_failures() -> frozenset[str]:
    """Test-only: the comma-separated items of $MMAEMU_TEST_FORCE_FAIL ("fatbin", "nvrtc", gate names)."""
    return frozenset(x.strip() for x in os.environ.get("MMAEMU_TEST_FORCE_FAIL", "").split(",") if x.strip())


def stats() -> dict:
    """Counters of this process: fatbin_loads, nvrtc_compiles, nvrtc_disk_hits, nvrtc_memory_hits, module_loads,
    cache_write_errors."""
    with _LOCK:
        return dict(_STATS)


def clear() -> None:
    """Forget every cached module handle and in-memory cubin (tests; the modules stay loaded in their contexts)."""
    with _LOCK:
        _MODULES.clear()
        _NVRTC_MEM.clear()
        _MANIFESTS.clear()


# ----------------------------------------------------------------------------------------------------------------
# sources

def _module_source(module: str) -> str:
    """Source file (under csrc/) of a module stem, from the discovered KernelSpecs and probe modules."""
    try:
        from mmaemu.backends import registry

        for spec in registry.discover():
            if spec.module == module:
                return spec.source
        probes = registry.probe_modules()
        if module in probes:
            return probes[module]
    except ImportError:
        pass
    return f"{module}.cu"


_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.M)


def source_files(module: str, csrc: Path | None = None) -> tuple[str, dict[str, str]]:
    """(source file name, {file name: text}) for the module's source and every header it includes with
    #include "..." (resolved in csrc/, recursively). The first item of the dict is the source itself."""
    root = Path(csrc) if csrc is not None else CSRC
    src = _module_source(module)
    files: dict[str, str] = {}
    todo = [src]
    while todo:
        name = todo.pop(0)
        if name in files:
            continue
        p = root / name
        if not p.is_file():
            raise BackendUnavailableError(f"kernel source {p} not found")
        text = p.read_text()
        files[name] = text
        todo += [h for h in _INCLUDE.findall(text) if h not in files]
    return src, files


def source_sha256(module: str, csrc: Path | None = None) -> str:
    """sha256 over the module's source and included headers (name, length, bytes; source first, headers sorted).
    tools/build_fatbin.py writes it into the manifest; the loader refuses a fatbin whose value differs."""
    src, files = source_files(module, csrc)
    h = hashlib.sha256(b"mmaemu-source/1\0")
    for name in [src] + sorted(n for n in files if n != src):
        data = files[name].encode()
        h.update(name.encode() + b"\0" + str(len(data)).encode() + b"\0" + data)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------------------------
# loading

def _cuda_device(device):
    import torch

    d = torch.device(device) if not isinstance(device, torch.device) else device
    if d.type != "cuda":
        raise ValueError(f"not a CUDA device: {d}")
    if d.index is None:
        d = torch.device("cuda", torch.cuda.current_device())
    return d


def _manifest(d: Path) -> dict:
    p = d / "manifest.json"
    st = p.stat()
    key = (str(p), st.st_mtime_ns, st.st_size)
    m = _MANIFESTS.get(key)
    if m is None:
        m = json.loads(p.read_text())
        _MANIFESTS[key] = m
    return m


def _arch_ok(sm: int, sass: list, ptx) -> bool:
    if any(a // 10 == sm // 10 and a <= sm for a in sass):  # SASS is binary compatible within a major version
        return True
    return ptx is not None and int(ptx) <= sm


def _try_fatbin(module: str, sm: int) -> tuple[int, bytes] | str:
    """(handle, image) or the reason the fatbin cannot be used."""
    if "fatbin" in forced_failures():
        return "forced by MMAEMU_TEST_FORCE_FAIL"
    d = fatbin_dir()
    if not (d / "manifest.json").is_file():
        return f"no fatbin manifest in {d}"
    try:
        m = _manifest(d)
    except (OSError, ValueError) as e:
        return f"unreadable manifest {d / 'manifest.json'}: {e}"
    if m.get("format") != MANIFEST_FORMAT:
        return f"manifest format {m.get('format')!r} is not {MANIFEST_FORMAT!r}"
    e = (m.get("modules") or {}).get(module)
    if not e:
        return f"module {module!r} not in {d / 'manifest.json'}"
    f = d / e["file"]
    try:
        image = f.read_bytes()
    except OSError as err:
        return f"cannot read {f}: {err}"
    if hashlib.sha256(image).hexdigest() != e.get("sha256"):
        return f"{f} does not match its manifest sha256"
    try:
        cur = source_sha256(module)
    except BackendUnavailableError as err:
        return str(err)
    if e.get("source_sha256") != cur:
        return f"{f} is stale (built from other kernel sources)"
    if not _arch_ok(sm, list(e.get("sass_archs") or []), e.get("ptx_arch")):
        return f"{f} has no SASS or PTX usable on sm_{sm}"
    try:
        return driver.module_load(image), image
    except BackendUnavailableError as err:
        return f"loading {f} failed: {err}"


def _nvrtc_arch(sm: int) -> int:
    from mmaemu.backends.cuda import nvrtc

    try:
        sup = nvrtc.supported_archs()
    except BackendUnavailableError:
        sup = ()
    if not sup or sm in sup:
        return sm
    same = [a for a in sup if a // 10 == sm // 10 and a <= sm]
    if not same:
        raise BackendUnavailableError(f"NVRTC {nvrtc.version()} cannot target sm_{sm} (supports {sup})")
    return max(same)


def _atomic_write(path: Path, data: bytes) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
        return True
    except OSError:
        try:
            tmp.unlink()  # type: ignore[possibly-undefined]
        except Exception:  # noqa: BLE001
            pass
        _STATS["cache_write_errors"] += 1
        return False


def _try_nvrtc(module: str, sm: int) -> tuple[int, bytes] | str:
    if "nvrtc" in forced_failures():
        return "forced by MMAEMU_TEST_FORCE_FAIL"
    try:
        from mmaemu.backends.cuda import nvrtc

        arch = _nvrtc_arch(sm)
        src, files = source_files(module)
        opts = nvrtc.build_options(arch)
        key = hashlib.sha256(json.dumps(["mmaemu-nvrtc/1", module, src, sorted(files.items()), opts,
                                         list(nvrtc.version())]).encode()).hexdigest()
        image = _NVRTC_MEM.get(key)
        if image is not None:
            _STATS["nvrtc_memory_hits"] += 1
        else:
            path = cache_dir() / "nvrtc" / f"{key}.cubin"
            try:
                image = path.read_bytes() if path.is_file() else None
            except OSError:
                image = None
            if image:
                try:
                    handle = driver.module_load(image)
                    _STATS["nvrtc_disk_hits"] += 1
                    _NVRTC_MEM[key] = image
                    return handle, image
                except BackendUnavailableError:
                    image = None  # corrupt cache entry: recompile and overwrite
            headers = {n: t for n, t in files.items() if n != src}
            image = nvrtc.compile(files[src], name=src, arch=arch, headers=headers)
            _STATS["nvrtc_compiles"] += 1
            _atomic_write(path, image)
            _NVRTC_MEM[key] = image
        return driver.module_load(image), image
    except BackendUnavailableError as err:
        return str(err)
    except OSError as err:
        return f"{type(err).__name__}: {err}"


def _load(module: str, ordinal: int, ctx: int) -> LoadedModule:
    import torch

    cc = torch.cuda.get_device_capability(ordinal)
    sm = cc[0] * 10 + cc[1]
    from mmaemu.backends import registry

    for spec in registry.discover():
        if spec.module == module and not _arch_ok(sm, spec.sass_archs, spec.ptx_arch):
            raise BackendUnavailableError(
                f"kernel module {module!r} has no binary for sm_{sm} "
                f"(supported SASS {spec.sass_archs}, PTX {spec.ptx_arch}); "
                "NVRTC cannot extend the kernel's supported architectures")
    r = _try_fatbin(module, sm)
    fatbin_error = None
    if isinstance(r, str):
        fatbin_error = r
        r2 = _try_nvrtc(module, sm)
        if isinstance(r2, str):
            raise BackendUnavailableError(f"kernel module {module!r} unavailable on sm_{sm}: fatbin: {fatbin_error}; "
                                          f"nvrtc: {r2}")
        handle, image = r2
        source = "nvrtc"
    else:
        handle, image = r
        source = "fatbin"
        _STATS["fatbin_loads"] += 1
    _STATS["module_loads"] += 1
    return LoadedModule(module=module, ordinal=ordinal, context=ctx, handle=handle, source=source,
                        sha256=hashlib.sha256(image).hexdigest(), arch=sm, fatbin_error=fatbin_error)


def _loaded(module: str, device) -> LoadedModule:
    import torch

    dev = _cuda_device(device)
    with torch.cuda.device(dev):
        ctx = driver.ensure_context(dev.index)
        key = (dev.index, driver.context_id(ctx), module)
        with _LOCK:
            lm = _MODULES.get(key)
            if lm is not None and lm.source in forced_failures():
                lm = None  # a forced failure applies to modules loaded earlier in this process too
            if lm is None:
                lm = _load(module, dev.index, ctx)
                _MODULES[key] = lm
            return lm


def module_info(module: str, device) -> LoadedModule:
    """The LoadedModule of `module` in the current context of `device` (loading it if needed)."""
    return _loaded(module, device)


def get_function(module: str, entry: str, device) -> tuple[int, str]:
    """(CUfunction, "fatbin" | "nvrtc") of `entry` in `module`, loaded in the current context of `device`."""
    lm = _loaded(module, device)
    with _LOCK:
        fn = lm.functions.get(entry)
        if fn is None:
            fn = driver.get_function(lm.handle, entry)
            lm.functions[entry] = fn
    return fn, lm.source


def launch(module: str, entry: str, grid, block, smem: int, args: list, device) -> None:
    """Launch `entry` of `module` under torch.cuda.device(device) on torch.cuda.current_stream(device).
    Dynamic shared memory above 48 KiB is enabled per function, up to the device's opt-in limit."""
    import torch

    dev = _cuda_device(device)
    with torch.cuda.device(dev):
        lm = _loaded(module, dev)
        fn, _ = get_function(module, entry, dev)
        if smem > DEFAULT_DYNAMIC_SMEM and lm.smem_configured.get(entry, 0) < smem:
            limit = driver.device_attribute(dev.index, driver.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
            if smem > limit:
                raise BackendUnavailableError(f"{module}.{entry} needs {smem} bytes of shared memory; the device "
                                              f"allows {limit}")
            driver.set_max_dynamic_smem(fn, smem)
            lm.smem_configured[entry] = smem
        stream = torch.cuda.current_stream(dev).cuda_stream
        driver.launch(fn, grid, block, smem, stream, args)
