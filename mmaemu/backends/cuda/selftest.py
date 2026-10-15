"""Per-device kernel gates and their verdict cache.

Every CUDA kernel must pass the "smoke" gate on a device before it is used there; gated kernels (v9, tc3) also need
their own gates (registered by mmaemu.backends.cuda.fast.gates through spec.register_gate).

smoke: the kernel, through its standard-ABI ChainFn, is compared bit for bit (int32 views, +0 != -0) with the CPU
float64 reference model (mmaemu.model.chain.chain_gemm) on
  - 16 x 8 x 64 random codes (randn * 40, as E4M3);
  - 64 x 64 x 256 adversarial codes: subnormals and +-0, +-448 (largest products and sums), exact and near
    cancellation inside blocks, one exponent per (row, 16-block), every finite code, -0 sprinkled everywhere.

Verdicts are cached in memory and in $MMAEMU_CACHE_DIR/verdicts/<sha256>.json under the key
(GPU UUID, compute capability, driver version, mmaemu version, module image sha256, gate name, gate version).
A gate function may carry an integer attribute `version` (default 1). A corrupt or unreadable cache file is ignored
(the gate re-runs); an unwritable cache directory keeps verdicts in memory. Gates that raise are reported as failed
and not cached. MMAEMU_TEST_FORCE_FAIL=<gate> (test-only) fails that gate without running or caching it.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
import time
import uuid

from mmaemu.backends.cuda import driver, modules
from mmaemu.backends.cuda.spec import GATES, GateResult, KernelSpec, register_gate

__all__ = ["run_gates", "smoke_gate", "verdict_key", "clear_memory_cache", "SMOKE_VERSION"]

SMOKE_VERSION = 1
_LOCK = threading.RLock()
_MEMORY: dict[str, GateResult] = {}
_SMOKE_CASES: list | None = None


# ----------------------------------------------------------------------------------------------------------------
# smoke gate data (deterministic, CPU)

def _gauss40(shape, g):
    import torch

    return (torch.randn(shape, generator=g) * 40).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)


def _adversarial(rows: int, K: int, g, side: int):
    """[rows, K] finite E4M3 codes; row groups of 16 exercise different features (see the module docstring)."""
    import torch

    u8 = torch.uint8
    sign = (torch.rand((rows, K), generator=g) < 0.5).to(u8) << 7
    man = torch.randint(0, 8, (rows, K), generator=g, dtype=u8)
    out = torch.empty((rows, K), dtype=u8)
    for r0 in range(0, rows, 16):
        kind = (r0 // 16 + side) % 4
        sl = slice(r0, r0 + 16)
        if kind == 0:   # subnormals and +-0
            c = sign[sl] | man[sl]
        elif kind == 1:  # +-448 mixed with large codes: largest products and sums
            big = torch.rand((16, K), generator=g) < 0.6
            e = torch.randint(12, 16, (16, K), generator=g, dtype=u8)
            c = torch.where(big, sign[sl] | 0x7E, sign[sl] | (e << 3) | man[sl])
        elif kind == 2:  # one exponent per 16-block (accumulator and block products of very different size)
            eb = torch.randint(0, 16, (16, (K + 15) // 16), generator=g).repeat_interleave(16, 1)[:, :K].to(u8)
            c = sign[sl] | (eb << 3) | man[sl]
        else:           # every finite code
            c = torch.randint(0, 256, (16, K), generator=g, dtype=u8)
        c = torch.where((c & 0x7F) == 0x7F, c - 1, c)
        if kind == 3 and side == 0:  # exact and near cancellation: a[2j+1] = -a[2j]
            c[:, 1::2] = c[:, 0::2] ^ 0x80
        out[sl] = c
    negz = torch.rand((rows, K), generator=g) < 0.05
    out = torch.where(negz, torch.full_like(out, 0x80), out)
    if side == 1:  # near cancellation on the B side: b[2j+1] within one code of b[2j]
        step = torch.randint(-1, 2, (rows, K // 2), generator=g, dtype=torch.int16)
        nb = ((out[:, 0::2] & 0x7F).to(torch.int16) + step).clamp(0, 0x7E).to(u8)
        out[:, 1::2] = torch.where(torch.rand((rows, K // 2), generator=g) < 0.5, (out[:, 0::2] & 0x80) | nb,
                                   out[:, 1::2])
    return out


def _smoke_cases():
    """[(a_u8, b_u8, expected fp32 accumulator)] on CPU; the reference runs once per process."""
    global _SMOKE_CASES
    with _LOCK:
        if _SMOKE_CASES is None:
            import torch
            from mmaemu.model.chain import chain_gemm

            g = torch.Generator().manual_seed(20261012)
            cases = [(_gauss40((16, 64), g), _gauss40((8, 64), g)),
                     (_adversarial(64, 256, g, 0), _adversarial(64, 256, g, 1))]
            _SMOKE_CASES = [(a, b, chain_gemm(a, b, fmt_a="e4m3", fmt_b="e4m3", preset="ada").float())
                            for a, b in cases]
        return _SMOKE_CASES


def smoke_gate(spec: KernelSpec, device) -> GateResult:
    """Bitwise comparison of the kernel with the CPU reference on the smoke data (see the module docstring)."""
    import torch
    from mmaemu.backends import ChainRequest
    from mmaemu.backends.cuda.kernels import chain_fn_for

    t0 = time.monotonic()
    fmt_a, fmt_b = tuple(spec.formats[0])
    fn = chain_fn_for(spec, ChainRequest(device=torch.device(device), fmt_a=fmt_a, fmt_b=fmt_b, preset="ada"))
    outputs = mismatches = 0
    first = None
    for a, b, want in _smoke_cases():
        got = fn(a.to(device), b.to(device)).cpu()
        bad = got.view(torch.int32) != want.view(torch.int32)
        n = int(bad.sum())
        outputs += want.numel()
        mismatches += n
        if n and first is None:
            i, j = (int(x) for x in torch.nonzero(bad)[0])
            first = dict(shape=list(want.shape), index=[i, j], got=float(got[i, j]), want=float(want[i, j]))
    ok = mismatches == 0
    reason = "" if ok else f"{mismatches} of {outputs} outputs differ from the CPU reference"
    return GateResult(name="smoke", passed=ok, reason=reason,
                      details=dict(outputs=outputs, mismatches=mismatches, first_mismatch=first),
                      seconds=time.monotonic() - t0, cached=False)


smoke_gate.version = SMOKE_VERSION  # type: ignore[attr-defined]
register_gate("smoke", smoke_gate)


# ----------------------------------------------------------------------------------------------------------------
# verdict cache

def verdict_key(spec: KernelSpec, device, gate: str, gate_version: int) -> dict:
    import torch

    from mmaemu._version import __version__

    dev = torch.device(device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    lm = modules.module_info(spec.module, dev)
    return dict(format="mmaemu-verdict/1", gpu_uuid=driver.device_uuid(idx),
                cc=list(torch.cuda.get_device_capability(idx)), driver=driver.driver_version(), mmaemu=__version__,
                module=spec.module, module_sha256=lm.sha256, gate=gate, gate_version=int(gate_version))


def _key_hash(key: dict) -> str:
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def _disk_path(h: str):
    return modules.cache_dir() / "verdicts" / f"{h}.json"


def _jsonable(x):
    try:
        json.dumps(x)
        return x
    except (TypeError, ValueError):
        return repr(x)


def _lookup(key: dict) -> GateResult | None:
    h = _key_hash(key)
    r = _MEMORY.get(h)
    if r is not None:
        return r
    try:
        d = json.loads(_disk_path(h).read_text())
        if d.get("key") != key:
            return None
        res = d["result"]
        r = GateResult(name=str(res["name"]), passed=bool(res["passed"]), reason=str(res["reason"]),
                       details=dict(res["details"]), seconds=float(res["seconds"]), cached=True)
    except Exception:  # noqa: BLE001 - missing, unreadable or corrupt: run the gate again
        return None
    _MEMORY[h] = r
    return r


def _store(key: dict, r: GateResult) -> None:
    h = _key_hash(key)
    _MEMORY[h] = r
    path = _disk_path(h)
    data = json.dumps(dict(key=key, result=dict(name=r.name, passed=r.passed, reason=r.reason,
                                                 details=_jsonable(r.details), seconds=r.seconds)), indent=1)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(data)
        os.replace(tmp, path)
    except OSError:
        pass  # read-only cache directory: the verdict stays in memory


def clear_memory_cache() -> None:
    """Forget in-memory verdicts and the smoke reference (tests)."""
    global _SMOKE_CASES
    with _LOCK:
        _MEMORY.clear()
        _SMOKE_CASES = None


def run_gates(spec: KernelSpec, device, *, force: bool = False) -> list[GateResult]:
    """Run (or fetch cached verdicts of) "smoke" and then spec.gates for `spec` on `device`, stopping at the first
    failure. force=True ignores cached verdicts. Raises BackendUnavailableError if the kernel module cannot be
    loaded on the device."""
    import torch

    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError(f"gates run on CUDA devices, got {dev}")
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    modules.get_function(spec.module, spec.entry, dev)  # load (fatbin or NVRTC) before anything else
    forced = modules.forced_failures()
    names = ["smoke"] + [g for g in spec.gates if g != "smoke"]
    results: list[GateResult] = []
    with _LOCK:
        for name in names:
            gate = GATES.get(name)
            if name in forced:
                results.append(GateResult(name=name, passed=False, reason="forced by MMAEMU_TEST_FORCE_FAIL",
                                          details={}, seconds=0.0, cached=False))
                break
            if gate is None:
                results.append(GateResult(name=name, passed=False, reason=f"gate {name!r} is not registered",
                                          details={}, seconds=0.0, cached=False))
                break
            key = verdict_key(spec, dev, name, getattr(gate, "version", 1))
            r = None if force else _lookup(key)
            if r is None:
                t0 = time.monotonic()
                try:
                    r = gate(spec, dev)
                    if r.name != name:
                        r = dataclasses.replace(r, name=name)
                    r = dataclasses.replace(r, cached=False, seconds=r.seconds or time.monotonic() - t0)
                    _store(key, r)
                except Exception as e:  # noqa: BLE001 - a crashing gate is a failed gate (not cached)
                    r = GateResult(name=name, passed=False, reason=f"{type(e).__name__}: {str(e)[:300]}",
                                   details={}, seconds=time.monotonic() - t0, cached=False)
            else:
                r = dataclasses.replace(r, cached=True)
            results.append(r)
            if not r.passed:
                break
    return results
