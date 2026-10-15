"""cuBLASLt probe through ctypes: the library version, the GPU identity and cuBLASLt's heuristic choice for the
descriptor torch 2.7.1 builds in ``torch._scaled_mm``.

Port of the tcemu audit's tool verify-c5-splitk-4/lt_ctypes.py (no torch, no C++ extension).

Problem mapping (from torch 2.7.1's own CUBLASLT_LOG_LEVEL=2 trace of ``torch._scaled_mm(a[M,K], b[N,K].t())``):
  m = N, n = M, k = K; A = mat2 (fp8, K x N, ld K, transa = T); B = mat1 (fp8, K x M, ld K, transb = N);
  D = out (N x M, ld N). C has the out dtype, except for fp8 outputs, where torch passes C as fp16 (R_16F; measured
  on the RTX 4060 with torch 2.7.1+cu128, see docs/splitk_tables.md).
Descriptor attributes mirror torch 2.7.1 scaled_gemm: A/B scale pointers, FAST_ACCUM, the D scale pointer only when
``scale_result`` is passed (``d_scale=True``), SM_COUNT_TARGET only if requested, EPILOGUE_BIAS only if requested;
preference = MAX_WORKSPACE_BYTES (+ MIN_ALIGNMENT_A/B/C/D when ``align`` is given, as Transformer Engine sets them);
requestedAlgoCount = 1 (torch and TE use 1).

Nothing here runs at import time. The first call loads libcuda and libcublasLt and calls cuInit (no CUDA context is
created by torch; the probe retains the device's primary context, which is the one torch uses).

Library search order for libcublasLt:
  1. a libcublasLt already mapped into this process (the one torch loaded; /proc/self/maps),
  2. the ``nvidia.cublas`` pip wheel (located without importing it),
  3. ``ctypes.util.find_library("cublasLt")`` / ldconfig.
libcuda: $MMAEMU_LIBCUDA_PATH, libcuda.so.1, /usr/lib/wsl/lib/libcuda.so.1.
"""
from __future__ import annotations

import ctypes as C
import ctypes.util
import importlib.metadata
import importlib.util
import os
import threading
from pathlib import Path

__all__ = ["CublasLtError", "library_path", "version", "version_tuple", "device_name", "device_sm_count",
           "device_capability", "heuristic", "OUT_DTYPES", "FMTS"]

try:  # mmaemu's error hierarchy; a plain RuntimeError when this module is used on its own
    from mmaemu.errors import BackendUnavailableError as _Base  # type: ignore
except Exception:  # noqa: BLE001
    _Base = RuntimeError


class CublasLtError(_Base):
    """libcuda / libcublasLt could not be loaded or a call failed."""


# ---- enums (cublasLt.h / library_types.h of CUDA 12.8) ----------------------------------------------------------
_R32F, _R16F, _R16BF, _E4M3, _E5M2 = 0, 2, 14, 28, 29
OUT_DTYPES = {"float32": _R32F, "bfloat16": _R16BF, "float16": _R16F, "float8_e4m3fn": _E4M3,
              "float8_e5m2": _E5M2}
FMTS = {"e4m3": _E4M3, "e5m2": _E5M2}
_COMPUTE_32F = 68
(_D_TRANSA, _D_TRANSB, _D_EPI, _D_BIAS_PTR, _D_SMT, _D_ASCALE, _D_BSCALE, _D_DSCALE, _D_FAST,
 _D_BIAS_DT) = 3, 4, 7, 8, 15, 17, 18, 20, 25, 26
_EPI_BIAS = 4
_P_MAXWS, _P_ALA, _P_ALB, _P_ALC, _P_ALD = 1, 5, 6, 7, 8
_A_ID, _A_TILE, _A_SPLITK, _A_RED, _A_SWZ, _A_CUSTOM, _A_STAGES = 0, 1, 2, 3, 4, 5, 6
_PROP_MAJOR, _PROP_MINOR, _PROP_PATCH = 0, 1, 2
_CU_ATTR_SM_COUNT, _CU_ATTR_CC_MAJOR, _CU_ATTR_CC_MINOR = 16, 75, 76


class _Algo(C.Structure):
    _fields_ = [("data", C.c_uint64 * 8)]


class _HeurResult(C.Structure):
    _fields_ = [("algo", _Algo), ("workspaceSize", C.c_size_t), ("state", C.c_int), ("wavesCount", C.c_float),
                ("reserved", C.c_int * 4)]


assert C.sizeof(_HeurResult) == 96

_lock = threading.RLock()
_state: dict = {}


# ---- library discovery --------------------------------------------------------------------------------------------
def _mapped(name: str) -> list[str]:
    try:
        with open("/proc/self/maps") as f:
            paths = {ln.split()[-1] for ln in f if name in ln and "/" in ln}
    except OSError:
        return []
    return sorted(p for p in paths if Path(p).name.startswith(name))


def _wheel_dir() -> Path | None:
    try:
        spec = importlib.util.find_spec("nvidia.cublas")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0]) / "lib"


def library_path() -> str:
    """Path of the libcublasLt this module uses (search order in the module docstring)."""
    m = _mapped("libcublasLt.so")
    if m:
        return m[0]
    wd = _wheel_dir()
    if wd is not None and (wd / "libcublasLt.so.12").exists():
        return str(wd / "libcublasLt.so.12")
    found = ctypes.util.find_library("cublasLt")
    if found:
        return found
    raise CublasLtError("libcublasLt not found (no mapped copy, no nvidia.cublas wheel, nothing from ldconfig)")


def _libcuda() -> C.CDLL:
    if "cuda" not in _state:
        cands = [os.environ.get("MMAEMU_LIBCUDA_PATH"), "libcuda.so.1", "/usr/lib/wsl/lib/libcuda.so.1"]
        err = None
        for c in cands:
            if not c:
                continue
            try:
                _state["cuda"] = C.CDLL(c)
                break
            except OSError as e:
                err = e
        else:
            raise CublasLtError(f"libcuda not loadable: {err}")
        r = _state["cuda"].cuInit(0)
        if r != 0:
            raise CublasLtError(f"cuInit failed: {r}")
    return _state["cuda"]


def _lt() -> C.CDLL:
    if "lt" not in _state:
        p = library_path()
        try:
            lt = C.CDLL(p)
        except OSError as e:
            raise CublasLtError(f"cannot load {p}: {e}") from e
        lt.cublasLtGetVersion.restype = C.c_size_t
        _state["lt"], _state["lt_path"] = lt, p
    return _state["lt"]


def _ck(r: int, what: str) -> None:
    if r != 0:
        raise CublasLtError(f"{what} failed with status {r}")


# ---- version -------------------------------------------------------------------------------------------------------
def version_tuple() -> tuple[int, int, int]:
    """(major, minor, patch) reported by the loaded libcublasLt (cublasLtGetProperty)."""
    with _lock:
        lt = _lt()
        out = []
        for prop in (_PROP_MAJOR, _PROP_MINOR, _PROP_PATCH):
            v = C.c_int()
            _ck(lt.cublasLtGetProperty(prop, C.byref(v)), "cublasLtGetProperty")
            out.append(v.value)
        return tuple(out)  # type: ignore[return-value]


def version() -> str:
    """"major.minor.patch[.build]". The build number is not exposed by the library; it is taken from the
    nvidia-cublas-cu12 wheel metadata when the loaded library is that wheel's file and the first three components
    agree (e.g. "12.8.3.14" for torch 2.7.1+cu128's wheel)."""
    vt = version_tuple()
    s = ".".join(map(str, vt))
    try:
        wd = _wheel_dir()
        lp = Path(_state.get("lt_path", library_path())).resolve()
        if wd is not None and lp.parent == wd.resolve():
            for dist in ("nvidia-cublas-cu12", "nvidia-cublas"):
                try:
                    wv = importlib.metadata.version(dist)
                except importlib.metadata.PackageNotFoundError:
                    continue
                parts = wv.split(".")
                if tuple(int(x) for x in parts[:3] if x.isdigit()) == vt and len(parts) >= 4:
                    return wv
    except Exception:  # noqa: BLE001
        pass
    return s


# ---- device identity (driver API; respects CUDA_VISIBLE_DEVICES; no torch) -----------------------------------------
def _ordinal(device) -> int:
    if device is None:
        return 0
    if isinstance(device, int):
        return device
    idx = getattr(device, "index", None)
    if getattr(device, "type", None) is not None:
        if device.type != "cuda":
            raise ValueError(f"not a CUDA device: {device}")
        return 0 if idx is None else int(idx)
    s = str(device)
    if s == "cuda":
        return 0
    if s.startswith("cuda:"):
        return int(s.split(":", 1)[1])
    raise ValueError(f"cannot interpret device {device!r}")


def _cudev(device) -> C.c_int:
    cu = _libcuda()
    d = C.c_int()
    _ck(cu.cuDeviceGet(C.byref(d), _ordinal(device)), "cuDeviceGet")
    return d


def device_name(device=None) -> str:
    with _lock:
        cu = _libcuda()
        buf = C.create_string_buffer(256)
        _ck(cu.cuDeviceGetName(buf, 256, _cudev(device)), "cuDeviceGetName")
        return buf.value.decode()


def _attr(device, attr: int) -> int:
    cu = _libcuda()
    v = C.c_int()
    _ck(cu.cuDeviceGetAttribute(C.byref(v), attr, _cudev(device)), "cuDeviceGetAttribute")
    return v.value


def device_sm_count(device=None) -> int:
    with _lock:
        return _attr(device, _CU_ATTR_SM_COUNT)


def device_capability(device=None) -> tuple[int, int]:
    with _lock:
        return _attr(device, _CU_ATTR_CC_MAJOR), _attr(device, _CU_ATTR_CC_MINOR)


# ---- heuristic query ----------------------------------------------------------------------------------------------
def _ctx(ordinal: int):
    """Primary context of the device + a 64 KiB device buffer used as scale/bias pointer target (never read by the
    heuristic)."""
    key = ("ctx", ordinal)
    if key not in _state:
        cu = _libcuda()
        dev = C.c_int()
        _ck(cu.cuDeviceGet(C.byref(dev), ordinal), "cuDeviceGet")
        ctx = C.c_void_p()
        _ck(cu.cuDevicePrimaryCtxRetain(C.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
        _ck(cu.cuCtxPushCurrent_v2(ctx), "cuCtxPushCurrent")
        try:
            dptr = C.c_uint64()
            _ck(cu.cuMemAlloc_v2(C.byref(dptr), C.c_size_t(1 << 16)), "cuMemAlloc")
        finally:
            _ck(cu.cuCtxPopCurrent_v2(C.byref(C.c_void_p())), "cuCtxPopCurrent")
        _state[key] = (ctx, dptr.value)
    return _state[key]


def _handle():
    if "h" not in _state:
        h = C.c_void_p()
        _ck(_lt().cublasLtCreate(C.byref(h)), "cublasLtCreate")
        _state["h"] = h
    return _state["h"]


def _norm_out(out_dtype) -> str:
    s = str(out_dtype)
    s = s.split(".")[-1] if s.startswith("torch.") else s
    if s not in OUT_DTYPES:
        raise ValueError(f"unsupported out_dtype {out_dtype!r}; expected one of {sorted(OUT_DTYPES)}")
    return s


def heuristic(M: int, N: int, K: int, *, fmt_a: str, fmt_b: str, out_dtype, fast: bool, workspace_bytes: int,
              align: int | None = None, device=None, d_scale: bool = False, bias: str | None = None,
              sm_target: int = 0) -> dict:
    """cuBLASLt's first heuristic result for torch 2.7.1's _scaled_mm descriptor on ``device``.

    Returns dict(status, algo, tile, stages, splitK, reduction, swizzle, custom, workspace, waves_x1000). ``status``
    is the cublasStatus_t of cublasLtMatmulAlgoGetHeuristic (0 = ok); when it is nonzero or no algorithm is returned,
    every other field is -1. ``tile``/``stages``/``reduction`` are cublasLt.h enum values, ``splitK`` = numSplitsK,
    ``custom`` = customOption. fmt_a/fmt_b are the formats of mat1/mat2 ("e4m3"/"e5m2"); out_dtype a torch dtype or
    its name; ``d_scale`` sets the D scale pointer (torch does so iff scale_result is passed)."""
    od = _norm_out(out_dtype)
    if fmt_a not in FMTS or fmt_b not in FMTS:
        raise ValueError(f"fmt_a/fmt_b must be 'e4m3' or 'e5m2', got {fmt_a!r}, {fmt_b!r}")
    with _lock:
        lt = _lt()
        cu = _libcuda()
        ctx, dptr = _ctx(_ordinal(device))
        _ck(cu.cuCtxPushCurrent_v2(ctx), "cuCtxPushCurrent")
        try:
            return _query(lt, int(M), int(N), int(K), fmt_a, fmt_b, od, bool(fast), int(workspace_bytes), align,
                          dptr, d_scale, bias, sm_target)
        finally:
            cu.cuCtxPopCurrent_v2(C.byref(C.c_void_p()))


def _query(lt, M, N, K, fa, fb, od, fast, ws, align, dptr, d_scale, bias, sm_target):
    def setd(desc, attr, val):
        _ck(lt.cublasLtMatmulDescSetAttribute(desc, attr, C.byref(val), C.sizeof(val)), f"descattr {attr}")

    def setp(pref, attr, val):
        _ck(lt.cublasLtMatmulPreferenceSetAttribute(pref, attr, C.byref(val), C.sizeof(val)), f"prefattr {attr}")

    def layout(dt, rows, cols, ld):
        lay = C.c_void_p()
        _ck(lt.cublasLtMatrixLayoutCreate(C.byref(lay), dt, C.c_uint64(rows), C.c_uint64(cols), C.c_int64(ld)),
            "layout")
        return lay

    odt = OUT_DTYPES[od]
    op = C.c_void_p()
    _ck(lt.cublasLtMatmulDescCreate(C.byref(op), _COMPUTE_32F, _R32F), "desc")
    lays, pref = [], C.c_void_p()
    try:
        setd(op, _D_TRANSA, C.c_int32(1))
        setd(op, _D_TRANSB, C.c_int32(0))
        sp = C.c_void_p(dptr)
        setd(op, _D_ASCALE, sp)
        setd(op, _D_BSCALE, sp)
        if d_scale:
            setd(op, _D_DSCALE, C.c_void_p(dptr + 256))
        setd(op, _D_FAST, C.c_int8(1 if fast else 0))
        if sm_target:
            setd(op, _D_SMT, C.c_int32(sm_target))
        cdt = _R16F if od.startswith("float8") else odt
        if bias:
            bdt = OUT_DTYPES[_norm_out(bias)]
            cdt = bdt
            setd(op, _D_BIAS_PTR, C.c_void_p(dptr + 4096))
            setd(op, _D_EPI, C.c_uint32(_EPI_BIAS))
            setd(op, _D_BIAS_DT, C.c_int32(bdt))
        lays = [layout(FMTS[fb], K, N, K), layout(FMTS[fa], K, M, K), layout(cdt, N, M, N), layout(odt, N, M, N)]
        _ck(lt.cublasLtMatmulPreferenceCreate(C.byref(pref)), "pref")
        setp(pref, _P_MAXWS, C.c_uint64(ws))
        if align:
            for a in (_P_ALA, _P_ALB, _P_ALC, _P_ALD):
                setp(pref, a, C.c_uint32(int(align)))
        res = (_HeurResult * 1)()
        got = C.c_int()
        st = lt.cublasLtMatmulAlgoGetHeuristic(_handle(), op, lays[0], lays[1], lays[2], lays[3], pref, 1, res,
                                               C.byref(got))
        keys = ("algo", "tile", "stages", "splitK", "reduction", "swizzle", "custom")
        if st != 0 or got.value < 1:
            return dict(status=int(st) if st != 0 else -2, **{k: -1 for k in keys}, workspace=-1, waves_x1000=-1)
        vals = []
        for a in (_A_ID, _A_TILE, _A_STAGES, _A_SPLITK, _A_RED, _A_SWZ, _A_CUSTOM):
            v, w = C.c_int32(), C.c_size_t()
            _ck(lt.cublasLtMatmulAlgoConfigGetAttribute(C.byref(res[0].algo), a, C.byref(v), 4, C.byref(w)),
                f"cfgattr {a}")
            vals.append(int(v.value))
        out = dict(status=0, **dict(zip(keys, vals)))
        out["workspace"] = int(res[0].workspaceSize)
        out["waves_x1000"] = int(res[0].wavesCount * 1000)
        return out
    finally:
        if pref.value:
            lt.cublasLtMatmulPreferenceDestroy(pref)
        for lay in lays:
            lt.cublasLtMatrixLayoutDestroy(lay)
        lt.cublasLtMatmulDescDestroy(op)
