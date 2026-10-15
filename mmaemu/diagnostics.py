"""Diagnostics behind ``python -m mmaemu info`` and ``python -m mmaemu selftest``.

``collect_info()`` gathers versions, the device, which platform profiles match it, the status of every backend
(with the reason when one is unavailable), the shipped split-K tables and fatbins, and the maturity of every
preset. Each probe is isolated: a failure is reported with its reason and never raised, so ``info`` works on a
CPU-only machine, without a driver, and while parts of the package are unavailable.

``run_selftest()`` runs ``mmaemu.selftest`` on one CUDA device and classifies each kernel as *required* on that
device (non-gated kernels the device supports, and gated kernels whose capability is in ``validated_on``) or
*optional* (gated kernels on hardware they were not validated on: a rejection there only means the kernel is not
auto-selected). The verdict is PASS, FAIL (a required kernel failed a gate) or UNAVAILABLE (nothing could be tested).

Importing this module has no side effects: torch is imported lazily and CUDA is touched only by the functions
that need a device, never when ``MMAEMU_DISABLE_CUDA=1``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

INFO_FORMAT = "mmaemu-info/1"
SELFTEST_FORMAT = "mmaemu-selftest/1"
PASS, FAIL, UNAVAILABLE = "PASS", "FAIL", "UNAVAILABLE"
EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_UNAVAILABLE = 0, 1, 2, 4

_PKG_DIR = Path(__file__).resolve().parent
_BIN_DIR = _PKG_DIR / "backends" / "cuda" / "bin"
_DATA_DIR = _PKG_DIR / "platform" / "data"
_SPEC_MODULES = ("mmaemu.backends.cuda.kernels", "mmaemu.backends.cuda.fast.specs")


# ----------------------------------------------------------------------------------------------- helpers
def _api():
    """The public API module (indirection so that tests can substitute a fake)."""
    return importlib.import_module("mmaemu")


def reason(exc: BaseException) -> str:
    """One-line reason for an exception; NotImplementedError("<c>") reads as 'not implemented yet (<c>)'."""
    msg = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    if type(exc) is NotImplementedError:
        return f"not implemented yet ({msg})" if msg else "not implemented yet"
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


def _try(fn, *args, **kwargs) -> tuple[bool, Any]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return True, fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001 - a diagnostic reports every failure instead of raising
        return False, reason(e)


def _call(obj: Any, attr: str, *args, **kwargs) -> tuple[bool, Any]:
    """_try(getattr(obj, attr), ...) with the attribute lookup inside the guard."""
    return _try(lambda: getattr(obj, attr)(*args, **kwargs))


def to_jsonable(obj: Any) -> Any:
    """Dataclasses, tuples, sets, torch dtypes/devices and other objects -> JSON-compatible values."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((to_jsonable(v) for v in obj), key=str)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if hasattr(obj, "_asdict"):
        return to_jsonable(obj._asdict())
    return str(obj)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cuda_disabled() -> bool:
    return os.environ.get("MMAEMU_DISABLE_CUDA", "") == "1"


def kernel_specs() -> tuple[dict[str, Any], list[str]]:
    """KernelSpecs by name from the discovery names, plus the reasons for missing modules."""
    specs: dict[str, Any] = {}
    notes: list[str] = []
    for mod in _SPEC_MODULES:
        ok, m = _try(importlib.import_module, mod)
        if not ok:
            notes.append(f"{mod}: {m}")
            continue
        for s in getattr(m, "SPECS", ()) or ():
            specs[getattr(s, "name", str(s))] = s
    return specs, notes


# ----------------------------------------------------------------------------------------------- device
def device_info(device_index: int = 0) -> dict[str, Any]:
    """torch build and device facts. Never initialises CUDA when MMAEMU_DISABLE_CUDA=1."""
    out: dict[str, Any] = {}
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return {"torch": None, "error": reason(e), "cuda_available": False}
    out["torch"] = torch.__version__
    out["torch_cuda_build"] = getattr(torch.version, "cuda", None)
    if cuda_disabled():
        out.update(cuda_available=False, note="MMAEMU_DISABLE_CUDA=1: CUDA not queried; the reference backend is used")
        return out
    ok, avail = _try(torch.cuda.is_available)
    out["cuda_available"] = bool(ok and avail)
    if not out["cuda_available"]:
        out["note"] = "no CUDA device visible to torch: the reference backend (CPU) is used"
        return out
    ok, n = _try(torch.cuda.device_count)
    out["device_count"] = n if ok else None
    out["device_index"] = device_index
    ok, props = _try(torch.cuda.get_device_properties, device_index)
    if ok:
        out["name"] = props.name
        out["capability"] = [props.major, props.minor]
        out["sm_count"] = props.multi_processor_count
        out["total_memory_mib"] = round(props.total_memory / 1048576)
        uuid = getattr(props, "uuid", None)
        if uuid is not None:
            out["uuid"] = str(uuid)
    else:
        out["error"] = props
    ok, drv = _try(lambda: importlib.import_module("mmaemu.backends.cuda.driver").driver_version())
    out["cuda_driver_api_version"] = drv if ok else None
    if not ok:
        out["cuda_driver_api_version_reason"] = drv
    out["display_driver"] = _nvidia_smi_driver()
    ok, v = _try(lambda: importlib.import_module("mmaemu.platform.cublaslt").version())
    out["cublaslt_version"] = to_jsonable(v) if ok else None
    if not ok:
        out["cublaslt_version_reason"] = v
    ok, v = _try(lambda: importlib.import_module("mmaemu.backends.cuda.nvrtc").version())
    out["nvrtc_version"] = to_jsonable(v) if ok else None
    if not ok:
        out["nvrtc_version_reason"] = v
    return out


def _nvidia_smi_driver() -> str | None:
    exe = shutil.which("nvidia-smi") or ("/usr/lib/wsl/lib/nvidia-smi" if os.path.exists("/usr/lib/wsl/lib/nvidia-smi")
                                         else None)
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None


def _torch_device(info: dict[str, Any]):
    if not info.get("cuda_available"):
        return None
    import torch
    return torch.device("cuda", info.get("device_index", 0))


# ----------------------------------------------------------------------------------------------- package data
def fatbin_info() -> dict[str, Any]:
    man = _BIN_DIR / "manifest.json"
    if not man.is_file():
        return {"present": False, "note": "no release fatbins in this installation: kernels load through NVRTC "
                                          "(if libnvrtc is found) or the reference backend is used"}
    try:
        m = json.loads(man.read_text())
    except ValueError as e:
        return {"present": True, "error": f"manifest.json does not parse: {e}"}
    mods = {}
    for stem, d in sorted((m.get("modules") or {}).items()):
        f = _BIN_DIR / d.get("file", f"{stem}.fatbin")
        mods[stem] = {"file": f.name, "sass_archs": d.get("sass_archs"), "ptx_arch": d.get("ptx_arch"),
                      "sha256_ok": f.is_file() and _sha256(f) == d.get("sha256")}
    return {"present": True, "format": m.get("format"), "nvcc": m.get("nvcc"), "flags": m.get("flags"),
            "modules": mods}


def table_info(table_names: list[str]) -> list[dict[str, Any]]:
    rows = []
    files = sorted(p for p in _DATA_DIR.glob("*") if p.is_file()) if _DATA_DIR.is_dir() else []
    for name in sorted(set(table_names)):
        mine = [p for p in files if p.name.startswith(name + ".") or p.stem == name]
        row: dict[str, Any] = {"name": name, "files": [{"file": p.name, "bytes": p.stat().st_size,
                                                         "sha256": _sha256(p)} for p in mine]}
        if not mine:
            row["note"] = "no data file in mmaemu/platform/data"
        js = [p for p in mine if p.suffix == ".json"]
        if js:
            try:
                meta = json.loads(js[0].read_text())
                for k in ("rows", "n_rows", "cublaslt_version", "gpu", "sm_count", "source_sha256"):
                    if isinstance(meta, dict) and k in meta and not isinstance(meta[k], (dict, list)):
                        row[k] = meta[k]
            except ValueError as e:
                row["error"] = f"{js[0].name} does not parse: {e}"
        rows.append(row)
    other = [p.name for p in files if not any(p.name.startswith(n + ".") or p.stem == n for n in table_names)]
    if other:
        rows.append({"name": "(other package data)", "files": [{"file": f} for f in other]})
    return rows


# ----------------------------------------------------------------------------------------------- info
def collect_info(device_index: int = 0) -> dict[str, Any]:
    rep: dict[str, Any] = {"format": INFO_FORMAT, "problems": []}
    probs = rep["problems"]
    ok, api = _try(_api)
    if not ok:
        rep["mmaemu"] = {"import": api}
        probs.append(f"import mmaemu: {api}")
        return rep
    rep["mmaemu"] = {"version": getattr(api, "__version__", None), "location": str(_PKG_DIR)}
    if rep["mmaemu"]["version"] is None:
        probs.append("mmaemu.__version__ missing")
    rep["python"] = {"version": platform.python_version(), "executable": sys.executable,
                     "platform": platform.platform()}
    rep["env"] = {k: v for k, v in sorted(os.environ.items()) if k.startswith("MMAEMU_")}
    dev = device_info(device_index)
    rep["device"] = dev
    tdev = _torch_device(dev)

    # presets
    presets = []
    ok, names = _try(lambda: list(api.list_presets()))
    if ok:
        for n in names:
            okp, p = _call(api, "get_preset", n)
            presets.append({"name": n, "maturity": getattr(p, "maturity", None),
                            "validated_scope": getattr(p, "validated_scope", None)} if okp
                           else {"name": n, "error": p})
        oka, alias = _call(api, "get_preset", "blackwell")
        if oka and getattr(alias, "name", "blackwell") != "blackwell":
            for row in presets:
                if row["name"] == alias.name:
                    row["aliases"] = ["blackwell"]
    else:
        probs.append(f"presets: {names}")
    rep["presets"] = presets

    # profiles
    profiles = []
    tables = []
    ok, names = _try(lambda: list(api.list_profiles()))
    okd, default = _call(api, "get_profile")
    rep["default_profile"] = getattr(default, "name", None) if okd else None
    if not okd:
        probs.append(f"default profile: {default}")
    if ok:
        for n in names:
            okp, p = _call(api, "get_profile", n)
            if not okp:
                profiles.append({"name": n, "error": p})
                continue
            row = {f: to_jsonable(getattr(p, f, None)) for f in
                   ("name", "preset", "gpu_name", "capability", "sm_count", "cublaslt_version", "caller",
                    "torch_version", "workspace_bytes", "align", "splitk_table", "maturity", "splitk_maturity")}
            if getattr(p, "splitk_table", None):
                tables.append(p.splitk_table)
            if tdev is None:
                row["matches_device"] = None
                row["match_reason"] = "no CUDA device"
            else:
                okm, m = _call(p, "matches_device", tdev)
                row["matches_device"] = bool(m) if okm else None
                if not okm:
                    row["match_reason"] = m
            profiles.append(row)
    else:
        probs.append(f"profiles: {names}")
    rep["profiles"] = profiles
    rep["tables"] = table_info(tables)

    # backends
    ok, st = _call(api, "backends_status", tdev)
    if ok:
        rep["backends"] = {str(k): to_jsonable(v) for k, v in dict(st).items()}
    else:
        rep["backends"] = {}
        probs.append(f"backends_status: {st}")
    specs, notes = kernel_specs()
    rep["kernels"] = {n: {"gated": getattr(s, "gated", None),
                          "validated_on": to_jsonable(getattr(s, "validated_on", None)),
                          "sass_archs": to_jsonable(getattr(s, "sass_archs", None)),
                          "ptx_arch": getattr(s, "ptx_arch", None), "min_cc": to_jsonable(getattr(s, "min_cc", None))}
                      for n, s in sorted(specs.items())}
    rep["kernel_discovery_notes"] = notes
    rep["fatbins"] = fatbin_info()
    return rep


def format_info(rep: dict[str, Any]) -> str:
    L: list[str] = []
    mm = rep.get("mmaemu", {})
    L.append(f"mmaemu {mm.get('version')}  ({mm.get('location', '?')})")
    py = rep.get("python", {})
    if py:
        L.append(f"python {py.get('version')}  {py.get('platform')}")
    dev = rep.get("device", {})
    L.append(f"torch {dev.get('torch')}  (CUDA build {dev.get('torch_cuda_build')})")
    if rep.get("env"):
        L.append("environment: " + ", ".join(f"{k}={v}" for k, v in rep["env"].items()))
    L.append("")
    L.append("Device")
    if dev.get("cuda_available"):
        cc = dev.get("capability") or ["?", "?"]
        L.append(f"  {dev.get('name')}  sm_{cc[0]}{cc[1]}  {dev.get('sm_count')} SMs  "
                 f"{dev.get('total_memory_mib')} MiB")
        L.append(f"  display driver {dev.get('display_driver')}  CUDA driver API {dev.get('cuda_driver_api_version')}"
                 f"  cuBLASLt {dev.get('cublaslt_version')}  NVRTC {dev.get('nvrtc_version')}")
        for k in ("cuda_driver_api_version_reason", "cublaslt_version_reason", "nvrtc_version_reason"):
            if dev.get(k):
                L.append(f"    ({k.replace('_reason', '')}: {dev[k]})")
    else:
        L.append(f"  none: {dev.get('note', dev.get('error', ''))}")
    L.append("")
    L.append(f"Profiles (default: {rep.get('default_profile')})")
    for p in rep.get("profiles", []):
        if "error" in p:
            L.append(f"  {p['name']}: {p['error']}")
            continue
        m = p.get("matches_device")
        ms = "MATCHES this device" if m else ("does not match this device" if m is False else
                                              f"match unknown ({p.get('match_reason', '')})")
        L.append(f"  {p['name']}: preset {p.get('preset')}, {p.get('gpu_name')}, {p.get('sm_count')} SMs, "
                 f"cuBLASLt {p.get('cublaslt_version')}, caller {p.get('caller')}, workspace {p.get('workspace_bytes')}"
                 f", table {p.get('splitk_table')}; maturity {p.get('maturity')}, split-K {p.get('splitk_maturity')}"
                 f"; {ms}")
    L.append("")
    L.append("Presets")
    for p in rep.get("presets", []):
        if "error" in p:
            L.append(f"  {p['name']}: {p['error']}")
            continue
        al = f" (alias: {', '.join(p['aliases'])})" if p.get("aliases") else ""
        L.append(f"  {p['name']}{al}: {p.get('maturity')} - {p.get('validated_scope')}")
    L.append("")
    L.append("Backends")
    for n, b in rep.get("backends", {}).items():
        if isinstance(b, dict):
            state = "available" if b.get("available") else "unavailable"
            val = ", validated here" if b.get("validated_here") else ""
            src = f" via {b.get('source')}" if b.get("source") else ""
            why = f" - {b.get('reason')}" if b.get("reason") else ""
            L.append(f"  {n}: {state}{src}{val}{why}")
        else:
            L.append(f"  {n}: {b}")
    if not rep.get("backends"):
        L.append("  (no backend status available)")
    for note in rep.get("kernel_discovery_notes", []):
        L.append(f"  note: {note}")
    L.append("")
    fb = rep.get("fatbins", {})
    if fb.get("present"):
        L.append(f"Fatbins (nvcc {fb.get('nvcc')}, flags {' '.join(fb.get('flags') or [])})")
        for stem, d in (fb.get("modules") or {}).items():
            L.append(f"  {stem}: SASS {d.get('sass_archs')} PTX {d.get('ptx_arch')} "
                     f"sha256 {'OK' if d.get('sha256_ok') else 'MISMATCH'}")
        if fb.get("error"):
            L.append(f"  {fb['error']}")
    else:
        L.append(f"Fatbins: {fb.get('note', 'none')}")
    L.append("")
    L.append("Split-K tables")
    for t in rep.get("tables", []):
        files = ", ".join(f"{f['file']}" + (f" ({f['bytes']} B, sha256 {f['sha256'][:12]})" if "sha256" in f else "")
                          for f in t.get("files", []))
        extra = ", ".join(f"{k} {t[k]}" for k in ("rows", "n_rows", "cublaslt_version") if k in t)
        L.append(f"  {t['name']}: {files or t.get('note', '')}" + (f"; {extra}" if extra else ""))
    if not rep.get("tables"):
        L.append("  (none)")
    if rep.get("problems"):
        L.append("")
        L.append("Unavailable components")
        for p in rep["problems"]:
            L.append(f"  {p}")
    return "\n".join(L)


# ----------------------------------------------------------------------------------------------- selftest
def classify_kernel(name: str, spec: Any, cc: tuple[int, int] | None) -> tuple[bool, str]:
    """(required, why) for one kernel on a device of capability `cc`."""
    if spec is None:
        return True, "no KernelSpec found: treated as required"
    min_cc = tuple(getattr(spec, "min_cc", (0, 0)) or (0, 0))
    if cc is not None and cc < min_cc:
        return False, f"device sm_{cc[0]}{cc[1]} is below min_cc {min_cc}"
    if not getattr(spec, "gated", False):
        return True, "IEEE-exact kernel: must pass on every supported device"
    vo = {tuple(x) for x in (getattr(spec, "validated_on", ()) or ())}
    if cc in vo:
        return True, f"gated kernel validated on sm_{cc[0]}{cc[1]}: its gates must pass here"
    return False, ("gated kernel not validated on this capability: a rejection only means it is not auto-selected")


def run_selftest(device_index: int = 0, *, kernels: str = "all", force: bool = False) -> dict[str, Any]:
    rep: dict[str, Any] = {"format": SELFTEST_FORMAT, "force": force, "kernels_requested": kernels}
    ok, api = _try(_api)
    if not ok:
        rep.update(verdict=UNAVAILABLE, reason=f"import mmaemu: {api}")
        return rep
    rep["mmaemu"] = getattr(api, "__version__", None)
    dev = device_info(device_index)
    rep["device"] = dev
    tdev = _torch_device(dev)
    if tdev is None:
        rep.update(verdict=UNAVAILABLE, reason=dev.get("note", "no CUDA device: nothing to self-test"), kernels={})
        return rep
    cc = tuple(dev["capability"]) if dev.get("capability") else None
    sel = kernels if kernels == "all" else tuple(k.strip() for k in str(kernels).split(",") if k.strip())
    ok, res = _call(api, "selftest", tdev, kernels=sel, force=force)
    if not ok:
        rep.update(verdict=UNAVAILABLE, reason=f"mmaemu.selftest: {res}", kernels={})
        return rep
    specs, notes = kernel_specs()
    rep["kernel_discovery_notes"] = notes
    out: dict[str, Any] = {}
    failed_required = []
    for name, gates in dict(res).items():
        gates = list(gates or [])
        passed = bool(gates) and all(bool(getattr(g, "passed", False)) for g in gates)
        required, why = classify_kernel(name, specs.get(str(name).split(":", 1)[-1]), cc)
        out[name] = {"required": required, "why": why, "passed": passed,
                     "gates": [to_jsonable(g) for g in gates]}
        if required and not passed:
            failed_required.append(name)
    rep["kernels"] = out
    if not out:
        rep.update(verdict=UNAVAILABLE, reason="mmaemu.selftest returned no kernel results")
    elif failed_required:
        rep.update(verdict=FAIL, reason="required kernel(s) failed: " + ", ".join(failed_required))
    else:
        rep.update(verdict=PASS, reason="every required kernel passed its gates")
    return rep


def format_selftest(rep: dict[str, Any]) -> str:
    L = [f"mmaemu {rep.get('mmaemu')} selftest" + (" (--force)" if rep.get("force") else "")]
    dev = rep.get("device") or {}
    if dev.get("cuda_available"):
        cc = dev.get("capability") or ["?", "?"]
        L.append(f"device: {dev.get('name')} sm_{cc[0]}{cc[1]} {dev.get('sm_count')} SMs")
    for name, k in (rep.get("kernels") or {}).items():
        tag = "PASS" if k["passed"] else ("FAIL" if k["required"] else "REJECTED")
        L.append(f"  {name}: {tag} ({'required' if k['required'] else 'optional'}: {k['why']})")
        for g in k["gates"]:
            if isinstance(g, dict):
                cached = ", cached" if g.get("cached") else ""
                secs = g.get("seconds")
                t = f", {secs:.3f} s" if isinstance(secs, (int, float)) else ""
                L.append(f"      {g.get('name')}: {'pass' if g.get('passed') else 'FAIL'}{t}{cached}"
                         + (f" - {g.get('reason')}" if g.get("reason") else ""))
            else:
                L.append(f"      {g}")
    L.append(f"verdict: {rep.get('verdict')} - {rep.get('reason')}")
    return "\n".join(L)


def selftest_exit_code(rep: dict[str, Any]) -> int:
    return {PASS: EXIT_OK, FAIL: EXIT_FAIL}.get(rep.get("verdict"), EXIT_UNAVAILABLE)
