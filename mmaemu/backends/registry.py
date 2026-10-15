"""Backend selection: CUDA kernels by priority, then the float64 reference.

select(req, backend="auto") -> Backend
  backend (or $MMAEMU_BACKEND when the argument is "auto"): "auto" | "reference" | "cuda" | "cuda:<kernel>".
  - "reference", CPU tensors, MMAEMU_DISABLE_CUDA=1, or no CUDA device: the reference backend (an explicit "cuda..."
    request then raises BackendUnavailableError instead).
  - Otherwise the discovered kernels (kernels.SPECS + mmaemu.backends.cuda.fast.specs.SPECS when importable) are tried
    in priority order (tc3 40 > v9 30 > v4 20 > v1 10). A kernel is skipped, by design and silently, when it does not
    support the request (format pair, preset), the device capability is below min_cc, the device lacks the shared
    memory it needs, or it is gated and the capability is not in validated_on (unless MMAEMU_ALLOW_GATED_KERNELS=1).
    It is skipped as a failure when its module cannot be loaded (fatbin and NVRTC both failed) or a gate fails.
  - The first kernel that loads and passes its gates is returned; if none does, "auto" returns the reference backend
    and "cuda" raises.
  - Exactly one BackendFallbackWarning is emitted per (device, reason) and process: when a kernel was skipped as a
    failure, or the selected kernel was compiled with NVRTC because its fatbin was unusable. By-design skips do not
    warn; status() and explain_selection() report them.
status(device=None) -> {name: BackendStatus} for every discovered kernel and "reference", with reasons.
selftest(device=None, *, kernels="all", force=False) -> {kernel name: [GateResult]} runs the gates of the discovered
(or named) kernels on the device, whether or not they are validated there (diagnostics; selection is unchanged).
Never touches the CUDA driver for CPU devices or when CUDA is disabled or absent.
"""
from __future__ import annotations

import os
import threading
import warnings

from mmaemu.errors import BackendFallbackWarning, BackendUnavailableError

__all__ = ["select", "status", "selftest", "explain_selection", "discover", "probe_modules", "discovery_errors",
           "reset_warnings", "clear"]

_LOCK = threading.RLock()
_WARNED: set[tuple[str, str]] = set()
_SELECTIONS: dict[tuple, tuple] = {}
_DISCOVERY_ERRORS: dict[str, str] = {}
_NAMES = ("auto", "reference", "cuda")


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def discover() -> list:
    """KernelSpecs of kernels.SPECS plus mmaemu.backends.cuda.fast.specs.SPECS (if importable), highest priority
    first."""
    from mmaemu.backends.cuda import kernels

    specs = list(kernels.SPECS)
    try:
        from mmaemu.backends.cuda.fast import specs as fast_specs  # optional

        specs += list(getattr(fast_specs, "SPECS", ()))
        _DISCOVERY_ERRORS.pop("fast", None)
    except ModuleNotFoundError as e:
        if e.name and not e.name.startswith("mmaemu.backends.cuda.fast"):
            _DISCOVERY_ERRORS["fast"] = f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001 - a broken optional package must not break the IEEE kernels
        _DISCOVERY_ERRORS["fast"] = f"{type(e).__name__}: {e}"
    seen, out = set(), []
    for s in sorted(specs, key=lambda s: -s.priority):
        if s.name not in seen:
            seen.add(s.name)
            out.append(s)
    return out


def probe_modules() -> dict[str, str]:
    """Probe modules (stem -> source) shipped in the fatbins: spec.PROBE_MODULES plus fast.specs.PROBE_MODULES."""
    from mmaemu.backends.cuda import spec

    out = dict(getattr(spec, "PROBE_MODULES", {}) or {})
    try:
        from mmaemu.backends.cuda.fast import specs as fast_specs

        out.update(getattr(fast_specs, "PROBE_MODULES", {}) or {})
    except Exception:  # noqa: BLE001
        pass
    return out


def discovery_errors() -> dict[str, str]:
    return dict(_DISCOVERY_ERRORS)


def reset_warnings() -> None:
    """Forget which fallback warnings were emitted (tests)."""
    with _LOCK:
        _WARNED.clear()


def clear() -> None:
    """Forget cached selections and emitted warnings (tests)."""
    with _LOCK:
        _WARNED.clear()
        _SELECTIONS.clear()


def _warn_once(device: str, reason: str) -> None:
    with _LOCK:
        if (device, reason) in _WARNED:
            return
        _WARNED.add((device, reason))
    warnings.warn(reason, BackendFallbackWarning, stacklevel=4)


def _normalise(backend: str) -> str:
    name = backend
    if name == "auto":
        name = os.environ.get("MMAEMU_BACKEND", "").strip() or "auto"
    if name not in _NAMES and not name.startswith("cuda:"):
        raise ValueError(f"unknown backend {name!r}; expected 'auto', 'reference', 'cuda' or 'cuda:<kernel>'")
    return name


def _device(dev):
    import torch

    d = torch.device(dev)
    if d.type == "cuda" and d.index is None:
        d = torch.device("cuda", torch.cuda.current_device())
    return d


def _cuda_unavailable_reason(dev) -> str | None:
    """Why CUDA kernels cannot run for `dev` (without touching the driver), or None."""
    if dev.type != "cuda":
        return f"tensors on {dev.type}"
    if _flag("MMAEMU_DISABLE_CUDA"):
        return "MMAEMU_DISABLE_CUDA=1"
    import torch

    if not torch.cuda.is_available():
        return "no CUDA device available"
    return None


def _design_skip(spec, req, dev, explicit: bool) -> str | None:
    """By-design reason not to use `spec` (no warning), or None."""
    import torch
    from mmaemu.backends.cuda import modules

    if (req.fmt_a, req.fmt_b) not in tuple(tuple(f) for f in spec.formats):
        return f"format pair {req.fmt_a} x {req.fmt_b} not supported (kernels: {spec.formats})"
    if req.preset != "ada":
        return f"preset {req.preset!r} has no CUDA kernel"
    cc = tuple(torch.cuda.get_device_capability(dev))
    if cc < tuple(spec.min_cc):
        return f"needs compute capability >= {spec.min_cc[0]}.{spec.min_cc[1]}, device is {cc[0]}.{cc[1]}"
    if not modules._arch_ok(cc[0] * 10 + cc[1], spec.sass_archs, spec.ptx_arch):
        return f"no binary for sm_{cc[0]}{cc[1]} (SASS {spec.sass_archs}, PTX {spec.ptx_arch})"
    if spec.gated and cc not in spec.validated_on and not explicit and not _flag("MMAEMU_ALLOW_GATED_KERNELS"):
        return (f"gated kernel not validated on sm_{cc[0]}{cc[1]} (validated on "
                f"{sorted(f'sm_{a}{b}' for a, b in spec.validated_on)}; MMAEMU_ALLOW_GATED_KERNELS=1 to try)")
    if spec.smem_bytes > 48 * 1024:
        from mmaemu.backends.cuda import driver

        try:
            limit = driver.device_attribute(dev.index, driver.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
        except BackendUnavailableError as e:
            return f"driver unavailable: {e}"
        if spec.smem_bytes > limit:
            return f"needs {spec.smem_bytes} bytes of shared memory per block, the device allows {limit}"
    return None


def _try_kernel(spec, dev) -> tuple[bool, str, str | None, tuple, str | None]:
    """(ok, failure reason, source, gate results, fatbin error) after loading `spec` and running its gates."""
    from mmaemu.backends.cuda import modules, selftest

    try:
        _, source = modules.get_function(spec.module, spec.entry, dev)
        lm = modules.module_info(spec.module, dev)
    except BackendUnavailableError as e:
        return False, f"cannot load: {e}", None, (), None
    except Exception as e:  # noqa: BLE001
        return False, f"cannot load: {type(e).__name__}: {e}", None, (), None
    try:
        gates = tuple(selftest.run_gates(spec, dev))
    except Exception as e:  # noqa: BLE001
        return False, f"gates could not run: {type(e).__name__}: {e}", source, (), lm.fatbin_error
    bad = [g for g in gates if not g.passed]
    if bad:
        return False, f"gate {bad[0].name!r} failed: {bad[0].reason}", source, gates, lm.fatbin_error
    return True, "", source, gates, lm.fatbin_error


def _env_key() -> tuple:
    return tuple(os.environ.get(k, "") for k in ("MMAEMU_BACKEND", "MMAEMU_DISABLE_CUDA", "MMAEMU_ALLOW_GATED_KERNELS",
                                                 "MMAEMU_TEST_FORCE_FAIL", "MMAEMU_FATBIN_DIR", "MMAEMU_CACHE_DIR"))


def _select(req, backend: str):
    """(Backend, skipped reasons [str], warning message or None)."""
    from mmaemu.backends.reference import REFERENCE

    name = _normalise(backend)
    if name == "reference":
        return REFERENCE, [], None
    dev = _device(req.device)
    why = _cuda_unavailable_reason(dev)
    if why is not None:
        if name != "auto":
            raise BackendUnavailableError(f"backend {name!r} unavailable: {why}")
        return REFERENCE, [f"cuda: {why}"], None
    specs = discover()
    explicit = name.startswith("cuda:")
    if explicit:
        specs = [s for s in specs if f"cuda:{s.name}" == name]
        if not specs:
            raise BackendUnavailableError(f"unknown CUDA kernel {name!r}; available: "
                                          f"{[f'cuda:{s.name}' for s in discover()]}")
    skipped, failures = [], []
    from mmaemu.backends.cuda import CudaKernelBackend

    for spec in specs:
        r = _design_skip(spec, req, dev, explicit)
        if r is not None:
            skipped.append(f"cuda:{spec.name}: {r}")
            continue
        ok, reason, source, _, fatbin_error = _try_kernel(spec, dev)
        if not ok:
            skipped.append(f"cuda:{spec.name}: {reason}")
            failures.append(f"cuda:{spec.name} ({reason})")
            continue
        notes = list(failures)
        if source == "nvrtc" and fatbin_error:
            notes.append(f"cuda:{spec.name} compiled with NVRTC (fatbin unusable: {fatbin_error})")
        msg = None
        if notes:
            msg = f"mmaemu on {dev}: " + "; ".join(notes) + f"; using cuda:{spec.name} ({source})"
        return CudaKernelBackend(spec, source), skipped, msg
    if name != "auto":
        raise BackendUnavailableError(f"no usable CUDA kernel for {name!r} on {dev}: " + "; ".join(skipped))
    msg = None
    if failures:
        msg = (f"mmaemu on {dev}: no CUDA kernel usable (" + "; ".join(failures)
               + "); using the float64 reference backend (much slower)")
    return REFERENCE, skipped, msg


def explain_selection(req, backend: str = "auto"):
    """(Backend, [reason per skipped backend]) without emitting warnings (for mmaemu.explain)."""
    with _LOCK:
        be, skipped, _ = _select_cached(req, backend)
        return be, list(skipped)


def _select_cached(req, backend: str):
    dev = _device(req.device)
    key = (str(dev), req.fmt_a, req.fmt_b, req.preset, backend, _env_key())
    with _LOCK:
        hit = _SELECTIONS.get(key)
        if hit is None:
            hit = _select(req, backend)
            _SELECTIONS[key] = hit
        return hit


def select(req, backend: str = "auto"):
    """The Backend for `req` (see the module docstring). Raises ValueError for an unknown backend name and
    BackendUnavailableError when an explicitly requested CUDA backend cannot be used."""
    be, _, msg = _select_cached(req, backend)
    if msg:
        _warn_once(str(_device(req.device)), msg)
    return be


def status(device=None) -> dict:
    """{backend name: BackendStatus} for every discovered kernel (priority order) and "reference" on `device`
    (default: the current CUDA device if CUDA is available, else CPU). A kernel is available only after it was
    loaded and passed its gates on the device."""
    import torch

    from mmaemu.backends import BackendStatus, ChainRequest

    if device is None:
        device = "cuda" if (not _flag("MMAEMU_DISABLE_CUDA") and torch.cuda.is_available()) else "cpu"
    dev = _device(device)
    out = {}
    why = _cuda_unavailable_reason(dev)
    for spec in discover():
        name = f"cuda:{spec.name}"
        if why is not None:
            out[name] = BackendStatus(name=name, available=False, source=None, validated_here=False, reason=why)
            continue
        cc = tuple(torch.cuda.get_device_capability(dev))
        validated = cc in spec.validated_on
        fa, fb = tuple(spec.formats[0])
        r = _design_skip(spec, ChainRequest(device=dev, fmt_a=fa, fmt_b=fb, preset="ada"), dev, False)
        if r is not None:
            out[name] = BackendStatus(name=name, available=False, source=None, validated_here=validated, reason=r)
            continue
        ok, reason, source, gates, fatbin_error = _try_kernel(spec, dev)
        if ok and source == "nvrtc" and fatbin_error:
            reason = f"compiled with NVRTC (fatbin unusable: {fatbin_error})"
        out[name] = BackendStatus(name=name, available=ok, source=source, validated_here=validated, gates=gates,
                                  reason=reason)
    out["reference"] = BackendStatus(name="reference", available=True, source="torch", validated_here=True,
                                     reason="float64 model on the tensors' device")
    return out


def selftest(device=None, *, kernels="all", force: bool = False) -> dict:
    """{kernel name: [GateResult]} after running (or fetching cached verdicts of) the gates of every discovered kernel,
    or of `kernels` (a comma-separated string or a sequence of names), on `device` (default: the current CUDA device). Gated kernels are tested
    even where they are not validated (their results inform the runbook); kernels that cannot run there get one
    failed "availability" or "load" result with the reason. force=True ignores cached verdicts."""
    import torch

    from mmaemu.backends.cuda import selftest as st
    from mmaemu.backends.cuda.spec import GateResult

    specs = discover()
    if kernels != "all":
        names = kernels.split(",") if isinstance(kernels, str) else kernels  # the CLI passes a tuple
        want = [str(k).strip() for k in names if str(k).strip()]
        unknown = sorted(set(want) - {s.name for s in specs})
        if unknown:
            raise ValueError(f"unknown kernel(s) {unknown}; discovered: {[s.name for s in specs]}")
        specs = [s for s in specs if s.name in want]
    if device is None:
        device = "cuda" if (not _flag("MMAEMU_DISABLE_CUDA") and torch.cuda.is_available()) else "cpu"
    dev = _device(device)
    why = _cuda_unavailable_reason(dev)
    out = {}
    for spec in specs:
        if why is None:
            cc = tuple(torch.cuda.get_device_capability(dev))
            if cc < tuple(spec.min_cc):
                out[spec.name] = [GateResult(name="availability", passed=False,
                                             reason=f"needs compute capability >= {spec.min_cc}, device is {cc}")]
                continue
            try:
                out[spec.name] = list(st.run_gates(spec, dev, force=force))
            except BackendUnavailableError as e:
                out[spec.name] = [GateResult(name="load", passed=False, reason=str(e))]
        else:
            out[spec.name] = [GateResult(name="availability", passed=False, reason=why)]
    return out
