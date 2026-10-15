"""Minimal ctypes binding of the CUDA driver API (a port of the tcemu audit's p4-system-backends/d_pack/rtc.py).

Nothing happens at import: libcuda is loaded (and cuInit(0) called) on the first call of ``libcuda()``, i.e. on the
first call of any function below. Search order: $MMAEMU_LIBCUDA_PATH (if set, only that path), libcuda.so.1,
/usr/lib/wsl/lib/libcuda.so.1. MMAEMU_DISABLE_CUDA=1 (read at call time) makes ``libcuda()`` refuse. Every failure
raises mmaemu.errors.BackendUnavailableError naming the driver call and its cuGetErrorName.

These are raw bindings: functions that need a context (module_load, get_function, launch) use the calling thread's
current context, which the caller makes current (the runtime runs them under torch.cuda.device(dev), where torch's
primary context is current). Handles (CUmodule, CUfunction, CUcontext, CUstream) are plain ints.

Runtime helpers: loaded(), device_count(), context_id() (cuCtxGetId where available: context
handles can be reused after cuCtxDestroy, ids cannot), ensure_context() (make the device's primary context current on
a thread that has none) and function_attribute(). They are used by modules.py (the per-(device, context) cache).
"""
from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from typing import Sequence

from mmaemu.errors import BackendUnavailableError

__all__ = ["libcuda", "check", "driver_version", "device_uuid", "current_context", "module_load", "module_unload",
           "get_function", "set_max_dynamic_smem", "device_attribute", "launch", "loaded", "device_count",
           "context_id", "ensure_context", "function_attribute", "CU_FUNC_ATTRIBUTE_NUM_REGS",
           "CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT", "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR",
           "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR", "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN",
           "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES"]

LIBCUDA_SEARCH = ("libcuda.so.1", "/usr/lib/wsl/lib/libcuda.so.1")

# CUdevice_attribute / CUfunction_attribute values used by mmaemu (cuda.h, CUDA 12.x)
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
CU_FUNC_ATTRIBUTE_NUM_REGS = 4

_c_int_p = ctypes.POINTER(ctypes.c_int)
_c_void_pp = ctypes.POINTER(ctypes.c_void_p)
_SIGNATURES = {  # name -> argtypes (restype is CUresult = int for all); optional symbols marked by a leading "?"
    "cuInit": (ctypes.c_uint,),
    "cuDriverGetVersion": (_c_int_p,),
    "cuGetErrorName": (ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)),
    "cuGetErrorString": (ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)),
    "cuDeviceGet": (_c_int_p, ctypes.c_int),
    "cuDeviceGetAttribute": (_c_int_p, ctypes.c_int, ctypes.c_int),
    "?cuDeviceGetUuid_v2": (ctypes.c_char_p, ctypes.c_int),
    "?cuDeviceGetUuid": (ctypes.c_char_p, ctypes.c_int),
    "cuCtxGetCurrent": (_c_void_pp,),
    "cuCtxSetCurrent": (ctypes.c_void_p,),
    "cuDevicePrimaryCtxRetain": (_c_void_pp, ctypes.c_int),
    "?cuCtxGetId": (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong)),
    "cuDeviceGetCount": (_c_int_p,),
    "cuFuncGetAttribute": (_c_int_p, ctypes.c_int, ctypes.c_void_p),
    "cuModuleLoadData": (_c_void_pp, ctypes.c_void_p),
    "cuModuleUnload": (ctypes.c_void_p,),
    "cuModuleGetFunction": (_c_void_pp, ctypes.c_void_p, ctypes.c_char_p),
    "cuFuncSetAttribute": (ctypes.c_void_p, ctypes.c_int, ctypes.c_int),
    "cuLaunchKernel": (ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                       ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, _c_void_pp, _c_void_pp),
}


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
            raise BackendUnavailableError(f"{path}: symbol {name} not found (driver too old?)") from None
        fn.argtypes = argtypes
        fn.restype = ctypes.c_int
    return lib


@lru_cache(maxsize=None)
def _load() -> ctypes.CDLL:
    env = os.environ.get("MMAEMU_LIBCUDA_PATH")
    candidates = (env,) if env else LIBCUDA_SEARCH
    errors = []
    lib = None
    for path in candidates:
        try:
            lib = _open(path)
            break
        except OSError as e:
            errors.append(f"{path}: {e}")
    if lib is None:
        where = "MMAEMU_LIBCUDA_PATH" if env else "search order " + ", ".join(LIBCUDA_SEARCH)
        raise BackendUnavailableError(f"libcuda not loadable ({where}): " + "; ".join(errors))
    res = lib.cuInit(0)
    if res != 0:
        raise BackendUnavailableError(f"cuInit failed: {_error_name(lib, res)} ({res}): {_error_string(lib, res)}")
    return lib


def libcuda() -> ctypes.CDLL:
    """The loaded and initialised libcuda (cached after the first success; failures are not cached)."""
    if os.environ.get("MMAEMU_DISABLE_CUDA") == "1":
        raise BackendUnavailableError("MMAEMU_DISABLE_CUDA=1: the CUDA driver is not used")
    return _load()


libcuda.cache_clear = _load.cache_clear  # for tests that change MMAEMU_LIBCUDA_PATH


def loaded() -> bool:
    """True once libcuda has been opened by this module (diagnostics; it never opens it)."""
    return _load.cache_info().currsize > 0


def _error_name(lib, res: int) -> str:
    s = ctypes.c_char_p()
    if lib is None or lib.cuGetErrorName(res, ctypes.byref(s)) != 0 or not s.value:
        return f"CUDA_ERROR_{res}"
    return s.value.decode()


def _error_string(lib, res: int) -> str:
    s = ctypes.c_char_p()
    if lib is None or lib.cuGetErrorString(res, ctypes.byref(s)) != 0 or not s.value:
        return "unknown error"
    return s.value.decode()


def check(res: int, call: str) -> None:
    """Raise BackendUnavailableError("<call> failed: <cuGetErrorName> (<code>): <cuGetErrorString>") if res != 0."""
    if res == 0:
        return
    lib = _load() if _load.cache_info().currsize else None  # never load libcuda just to name an error
    raise BackendUnavailableError(f"{call} failed: {_error_name(lib, res)} ({res}): {_error_string(lib, res)}")


def driver_version() -> int:
    """cuDriverGetVersion, e.g. 12080 for a CUDA 12.8 driver."""
    v = ctypes.c_int()
    check(libcuda().cuDriverGetVersion(ctypes.byref(v)), "cuDriverGetVersion")
    return v.value


def _device(ordinal: int) -> int:
    d = ctypes.c_int()
    check(libcuda().cuDeviceGet(ctypes.byref(d), int(ordinal)), f"cuDeviceGet({ordinal})")
    return d.value


def device_uuid(ordinal: int) -> str:
    """The device UUID in nvidia-smi form, "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" (ordinal as seen by this
    process, i.e. after CUDA_VISIBLE_DEVICES)."""
    lib = libcuda()
    dev = _device(ordinal)
    buf = ctypes.create_string_buffer(16)
    fn, call = (lib.cuDeviceGetUuid_v2, "cuDeviceGetUuid_v2") if hasattr(lib, "cuDeviceGetUuid_v2") \
        else (lib.cuDeviceGetUuid, "cuDeviceGetUuid")
    check(fn(buf, dev), call)
    h = buf.raw.hex()
    return f"GPU-{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def device_attribute(ordinal: int, attr: int) -> int:
    """cuDeviceGetAttribute(attr) of device `ordinal` (attribute ids: the CU_DEVICE_ATTRIBUTE_* constants)."""
    v = ctypes.c_int()
    check(libcuda().cuDeviceGetAttribute(ctypes.byref(v), int(attr), _device(ordinal)),
          f"cuDeviceGetAttribute({attr})")
    return v.value


def current_context() -> int:
    """The calling thread's current CUcontext handle (0 if none)."""
    ctx = ctypes.c_void_p()
    check(libcuda().cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
    return ctx.value or 0


def device_count() -> int:
    """cuDeviceGetCount (devices visible to this process)."""
    n = ctypes.c_int()
    check(libcuda().cuDeviceGetCount(ctypes.byref(n)), "cuDeviceGetCount")
    return n.value


def context_id(ctx: int) -> int:
    """A process-unique id of context `ctx`: cuCtxGetId where the driver has it (handles can be reused after
    cuCtxDestroy), else the handle itself."""
    lib = libcuda()
    if ctx and hasattr(lib, "cuCtxGetId"):
        v = ctypes.c_ulonglong()
        if lib.cuCtxGetId(ctypes.c_void_p(ctx), ctypes.byref(v)) == 0:
            return int(v.value)
    return int(ctx)


def ensure_context(ordinal: int) -> int:
    """The calling thread's current context; if it has none, device `ordinal`'s primary context (the one torch
    uses) is retained, made current and returned."""
    ctx = current_context()
    if ctx:
        return ctx
    c = ctypes.c_void_p()
    check(libcuda().cuDevicePrimaryCtxRetain(ctypes.byref(c), _device(ordinal)), "cuDevicePrimaryCtxRetain")
    check(libcuda().cuCtxSetCurrent(c), "cuCtxSetCurrent")
    return c.value or 0


def module_load(image: bytes) -> int:
    """Load a cubin / fatbin / NUL-terminated PTX image into the current context; returns the CUmodule handle."""
    if not isinstance(image, (bytes, bytearray)) or not image:
        raise TypeError("image must be non-empty bytes")
    mod = ctypes.c_void_p()
    data = ctypes.c_char_p(bytes(image))
    check(libcuda().cuModuleLoadData(ctypes.byref(mod), data), "cuModuleLoadData")
    return mod.value


def module_unload(module: int) -> None:
    """cuModuleUnload (the module's context must be current)."""
    check(libcuda().cuModuleUnload(ctypes.c_void_p(module)), "cuModuleUnload")


def get_function(module: int, name: str) -> int:
    """The CUfunction handle of the extern "C" kernel `name` in `module`."""
    fn = ctypes.c_void_p()
    check(libcuda().cuModuleGetFunction(ctypes.byref(fn), ctypes.c_void_p(module), name.encode()),
          f"cuModuleGetFunction({name})")
    return fn.value


def set_max_dynamic_smem(fn: int, nbytes: int) -> None:
    """Allow `fn` to be launched with up to `nbytes` of dynamic shared memory (needed above 48 KiB)."""
    check(libcuda().cuFuncSetAttribute(ctypes.c_void_p(fn), CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                       int(nbytes)), "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)")


def function_attribute(fn: int, attr: int) -> int:
    """cuFuncGetAttribute(attr) of `fn` (CU_FUNC_ATTRIBUTE_* ids)."""
    v = ctypes.c_int()
    check(libcuda().cuFuncGetAttribute(ctypes.byref(v), int(attr), ctypes.c_void_p(fn)), f"cuFuncGetAttribute({attr})")
    return v.value


_CTYPES_ARG = (ctypes._SimpleCData, ctypes.Structure, ctypes.Union, ctypes.Array)


def _kernel_arg(i: int, x):
    if isinstance(x, _CTYPES_ARG):
        return x
    if hasattr(x, "data_ptr"):  # a torch.Tensor (or anything exposing a device pointer): pass the pointer
        return ctypes.c_void_p(x.data_ptr())
    raise TypeError(f"kernel argument {i}: pass a ctypes value (c_int, c_uint64, c_float, c_void_p, ...) or a tensor, "
                    f"not {type(x).__name__}: the parameter width cannot be inferred")


def _dims(what: str, d: Sequence[int]) -> tuple[int, int, int]:
    d = tuple(int(x) for x in d)
    if len(d) != 3 or min(d) < 1:
        raise ValueError(f"{what} must be three positive ints, got {d}")
    return d


def launch(fn: int, grid: Sequence[int], block: Sequence[int], smem: int, stream, args: list) -> None:
    """cuLaunchKernel(fn) on `stream` (a CUstream handle int, 0 for the legacy default stream, or an object with
    .cuda_stream such as torch.cuda.Stream). `args`: one ctypes value per kernel parameter, in order; tensors are
    passed as their data_ptr(). Launch-configuration errors raise; asynchronous faults surface at the next sync."""
    gx, gy, gz = _dims("grid", grid)
    bx, by, bz = _dims("block", block)
    if hasattr(stream, "cuda_stream"):
        stream = stream.cuda_stream
    cargs = [_kernel_arg(i, x) for i, x in enumerate(args)]  # kept alive until cuLaunchKernel returns
    params = (ctypes.c_void_p * max(len(cargs), 1))(*[ctypes.addressof(a) for a in cargs])
    check(libcuda().cuLaunchKernel(ctypes.c_void_p(fn), gx, gy, gz, bx, by, bz, int(smem),
                                   ctypes.c_void_p(int(stream or 0)), params, None), "cuLaunchKernel")
