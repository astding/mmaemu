"""CUDA runtime: loader ladder, contexts, streams, guards, fallbacks, files. GPU; run under
the lock:   $GPU timeout 600 python -m pytest -q tests/test_cuda_runtime.py

Fresh-process behaviour (fallback warnings, load time, NVRTC cache reuse, environment failures) runs in child
processes, which import mmaemu from this checkout (PYTHONPATH) and report one JSON line.
Fatbins come from build/fatbin ($MMAEMU_TEST_FATBIN_DIR overrides; tools/build_fatbin.py --out build/fatbin).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
E4M3 = torch.float8_e4m3fn
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")]


def _fatbin_dir() -> Path:
    return Path(os.environ.get("MMAEMU_TEST_FATBIN_DIR", ROOT / "build" / "fatbin")).resolve()


@pytest.fixture(scope="module")
def fatbins():
    if not (_fatbin_dir() / "manifest.json").is_file():
        pytest.skip(f"no fatbin manifest in {_fatbin_dir()}: run tools/build_fatbin.py --out build/fatbin")
    return _fatbin_dir()


@pytest.fixture
def env(fatbins, monkeypatch, tmp_path):
    from mmaemu.backends import registry
    from mmaemu.backends.cuda import modules

    monkeypatch.setenv("MMAEMU_FATBIN_DIR", str(fatbins))
    monkeypatch.setenv("MMAEMU_CACHE_DIR", str(tmp_path / "cache"))
    for k in ("MMAEMU_TEST_FORCE_FAIL", "MMAEMU_BACKEND", "MMAEMU_DISABLE_CUDA"):
        monkeypatch.delenv(k, raising=False)
    modules.clear()
    registry.clear()
    yield torch.device("cuda", torch.cuda.current_device())
    modules.clear()
    registry.clear()


def _data(M, N, K, seed=3):
    g = torch.Generator().manual_seed(seed)
    a = (torch.randn(M, K, generator=g) * 40).clamp(-448, 448).to(E4M3)
    b = (torch.randn(N, K, generator=g) * 40).clamp(-448, 448).to(E4M3)
    return a, b


KERNELS = ["v1", "v4", "h1", "h4"]
PROFILE = {"ada": "ada", "hopper": "hopper"}


def _preset(name):
    from mmaemu.backends.cuda.kernels import SPECS

    return {s.name: s for s in SPECS}[name].preset


def _chain(name, dev):
    from mmaemu.backends import ChainRequest
    from mmaemu.backends.cuda.kernels import SPECS, chain_fn_for

    spec = {s.name: s for s in SPECS}[name]
    return chain_fn_for(spec, ChainRequest(device=dev, fmt_a="e4m3", fmt_b="e4m3", preset=spec.preset))


def _ref(a, b, preset="ada"):
    from mmaemu.model.chain import chain_gemm

    return chain_gemm(a.view(torch.uint8).cpu(), b.view(torch.uint8).cpu(), fmt_a="e4m3", fmt_b="e4m3",
                      preset=preset).float()


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.int32).numpy().tobytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ in-process checks

@pytest.mark.parametrize("kernel", KERNELS)
def test_storage_offsets(env, kernel):
    """Storage offsets 1..15 on either and both operands: no CUDA error (B12), results equal to the aligned ones,
    through the ChainFn and through accumulate()."""
    import mmaemu.api as api
    from mmaemu.testing import assert_bitwise_equal

    M, N, K = 70, 33, 96
    a, b = _data(M, N, K)
    fn = _chain(kernel, env)
    want = fn(a.to(env), b.to(env)).cpu()
    assert_bitwise_equal(want, _ref(a, b, _preset(kernel)))
    base_a = torch.zeros(M * K + 32, dtype=torch.uint8, device=env)
    base_b = torch.zeros(N * K + 32, dtype=torch.uint8, device=env)
    for oa, ob in [(o, 0) for o in range(1, 16)] + [(0, o) for o in range(1, 16)] + [(o, 16 - o) for o in range(1, 16)]:
        va = base_a[oa:oa + M * K].view(M, K)
        vb = base_b[ob:ob + N * K].view(N, K)
        va.copy_(a.view(torch.uint8))
        vb.copy_(b.view(torch.uint8))
        if oa % 4:
            assert va.data_ptr() % 4
        got = fn(va, vb)
        acc = api.accumulate(va.view(E4M3), vb.view(E4M3), use_fast_accum=True, splitk=1,
                             profile=PROFILE[_preset(kernel)], backend=f"cuda:{kernel}")
        torch.cuda.synchronize(env)
        assert_bitwise_equal(got, want, msg=f"offsets {oa}, {ob}")
        assert_bitwise_equal(acc, want, msg=f"accumulate offsets {oa}, {ob}")


@pytest.mark.parametrize("kernel", KERNELS)
def test_non_default_stream(env, kernel):
    """Launches go to torch's current stream: while a sleep kernel holds stream s, our kernel queued on s has not
    run and the default stream is idle."""
    from mmaemu.testing import assert_bitwise_equal

    a, b = _data(130, 70, 256)
    A, B = a.to(env), b.to(env)
    fn = _chain(kernel, env)
    fn(A, B)
    torch.cuda.synchronize(env)
    s = torch.cuda.Stream(env)
    with torch.cuda.stream(s):
        torch.cuda._sleep(int(2e8))
        out = fn(A, B)
    busy = not s.query()
    default_idle = torch.cuda.default_stream(env).query()
    s.synchronize()
    assert busy and default_idle
    assert_bitwise_equal(out.cpu(), _ref(a, b, _preset(kernel)))


@pytest.mark.parametrize("kernel", KERNELS)
def test_wrong_dtype_and_mixed_devices(env, kernel):
    a, b = _data(16, 16, 64)
    fn = _chain(kernel, env)
    A, B = a.to(env), b.to(env)
    for bad in (torch.float8_e5m2, torch.float8_e4m3fnuz, torch.float8_e5m2fnuz, torch.bfloat16, torch.float32,
                torch.int8):
        with pytest.raises(TypeError):
            fn(A.view(torch.uint8).to(bad) if bad.is_floating_point else A.view(torch.uint8).view(torch.int8), B)
        with pytest.raises(TypeError):
            fn(A, B.view(torch.uint8).to(bad) if bad.is_floating_point else B.view(torch.uint8).view(torch.int8))
    with pytest.raises(ValueError, match="mixed devices"):
        fn(A, b)
    with pytest.raises(ValueError, match="mixed devices"):
        fn(a, B)
    with pytest.raises(ValueError):
        fn(A[:, :63], B[:, :63])  # L % 16 != 0 never reaches the kernel
    from mmaemu.backends import ChainRequest
    from mmaemu.backends.cuda.kernels import SPECS, chain_fn_for

    spec = {s.name: s for s in SPECS}[kernel]
    with pytest.raises(ValueError):
        chain_fn_for(spec, ChainRequest(device=env, fmt_a="e5m2", fmt_b="e4m3", preset=spec.preset))
    with pytest.raises(ValueError):
        chain_fn_for(spec, ChainRequest(device=torch.device("cpu"), fmt_a="e4m3", fmt_b="e4m3", preset=spec.preset))
    other = "hopper" if spec.preset == "ada" else "ada"
    with pytest.raises(ValueError, match=f"implements preset {spec.preset!r}"):
        chain_fn_for(spec, ChainRequest(device=env, fmt_a="e4m3", fmt_b="e4m3", preset=other))
    # zero sizes never launch
    assert fn(A[:0], B).shape == (0, 16) and fn(A, B[:0]).shape == (16, 0)
    z = fn(A[:, :0].contiguous(), B[:, :0].contiguous())
    assert z.shape == (16, 16) and int(z.view(torch.int32).abs().sum()) == 0


# ------------------------------------------------------------------------------------------------ child processes

CHILD_PRELUDE = textwrap.dedent("""
    import json, os, re, sys, time, warnings
    warnings.simplefilter("always")
    import torch
    from mmaemu.errors import BackendFallbackWarning
    def maps():
        return sorted({l.split()[-1] for l in open('/proc/self/maps') if re.search(r'/libcuda\\.so', l)})
    def data(M, N, K, seed=3):
        g = torch.Generator().manual_seed(seed)
        a = (torch.randn(M, K, generator=g) * 40).clamp(-448, 448).to(torch.float8_e4m3fn)
        b = (torch.randn(N, K, generator=g) * 40).clamp(-448, 448).to(torch.float8_e4m3fn)
        return a, b
    import hashlib
    def sha(t):
        return hashlib.sha256(t.detach().cpu().contiguous().view(torch.int32).numpy().tobytes()).hexdigest()
    def result(**kw):
        print("RESULT " + json.dumps(kw), flush=True)
""")


def run_child(body: str, env_update: dict | None = None, cwd=None, timeout=300) -> dict:
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    for k, v in (env_update or {}).items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    code = CHILD_PRELUDE + "\n" + textwrap.dedent(body)
    compile(code, "<child>", "exec")  # fail early on indentation mistakes in the test itself
    r = subprocess.run([sys.executable, "-c", code], env=env, cwd=cwd,
                       capture_output=True, text=True, timeout=timeout)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert r.returncode == 0 and lines, f"child failed (rc {r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    return json.loads(lines[-1][7:])


ACCUMULATE_TWICE = """
    from mmaemu.backends import ChainRequest, registry
    from mmaemu.backends.cuda import modules
    import mmaemu.api as api
    dev = torch.device("cuda", 0) if os.environ.get("CUDA_VISIBLE_DEVICES", "x") != "" and torch.cuda.is_available() else torch.device("cpu")
    a, b = data(96, 80, 320)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        req = ChainRequest(device=dev, fmt_a="e4m3", fmt_b="e4m3", preset="ada")
        be = registry.select(req)
        o1 = api.accumulate(a.to(dev), b.to(dev), use_fast_accum=True, splitk=1, profile="ada")
        o2 = api.accumulate(a.to(dev), b.to(dev), use_fast_accum=False, splitk=1, profile="ada")
    w = [str(x.message) for x in rec if issubclass(x.category, BackendFallbackWarning)]
    src = getattr(be, "source", None)
    from mmaemu.backends.cuda import driver
    result(backend=be.name, source=src, warnings=w, sha_fast=sha(o1), sha_slow=sha(o2), stats=modules.stats(),
           driver_loaded=driver.loaded(), maps=maps(), device=str(dev))
"""


@pytest.fixture(scope="module")
def expected_shas():
    import mmaemu.api as api

    a, b = _data(96, 80, 320)
    kw = dict(splitk=1, backend="reference", profile="ada")   # the preset the children select
    return dict(fast=_sha(api.accumulate(a, b, use_fast_accum=True, **kw)),
                slow=_sha(api.accumulate(a, b, use_fast_accum=False, **kw)))


def _child_env(fatbins, cache, **kw):
    e = dict(MMAEMU_FATBIN_DIR=str(fatbins), MMAEMU_CACHE_DIR=str(cache), MMAEMU_TEST_FORCE_FAIL=None,
             MMAEMU_BACKEND=None, MMAEMU_DISABLE_CUDA=None, MMAEMU_ALLOW_GATED_KERNELS=None)
    e.update(kw)
    return e


def _auto_kernel() -> str:
    """The kernel AUTO selects on this device when nothing is forced (docs/backends.md, priority tc3 > v9 > v4 > v1):
    tc3 where the gated kernels are validated (sm_89), v4 elsewhere (tc3 and v9 are skipped by design there)."""
    try:
        from mmaemu.backends.cuda.fast import specs
    except ImportError:
        return "cuda:v4"
    return "cuda:tc3" if tuple(torch.cuda.get_device_capability(0)) in specs.TC3.validated_on else "cuda:v4"


def test_fallback_ladder(fatbins, tmp_path, expected_shas):
    """MMAEMU_TEST_FORCE_FAIL=fatbin -> the same kernel from NVRTC; =fatbin,nvrtc -> reference; exactly one
    BackendFallbackWarning each (two calls); bits unchanged (B5)."""
    top = _auto_kernel()
    base = run_child(ACCUMULATE_TWICE, _child_env(fatbins, tmp_path / "c0"))
    assert base["backend"] == top and base["source"] == "fatbin" and base["warnings"] == [], base
    nv = run_child(ACCUMULATE_TWICE, _child_env(fatbins, tmp_path / "c1", MMAEMU_TEST_FORCE_FAIL="fatbin"))
    assert nv["backend"] == top and nv["source"] == "nvrtc" and len(nv["warnings"]) == 1, nv
    ref = run_child(ACCUMULATE_TWICE, _child_env(fatbins, tmp_path / "c2", MMAEMU_TEST_FORCE_FAIL="fatbin,nvrtc"))
    assert ref["backend"] == "reference" and len(ref["warnings"]) == 1, ref
    for r in (base, nv, ref):
        assert r["sha_fast"] == expected_shas["fast"] and r["sha_slow"] == expected_shas["slow"]
    print(json.dumps(dict(nvrtc_warning=nv["warnings"][0], reference_warning=ref["warnings"][0])))


def test_no_visible_devices(fatbins, tmp_path, expected_shas):
    """CUDA_VISIBLE_DEVICES='': the reference runs, mmaemu never opens libcuda (driver.loaded() is False) and adds
    no libcuda mapping. Note: on this host `import torch` itself maps libcuda (libcublasLt's initialiser dlopens
    it), so the mapping check compares the maps before and after mmaemu runs."""
    body = "before = maps()\n" + textwrap.dedent(ACCUMULATE_TWICE).replace("result(", "result(maps_before=before, ")
    r = run_child(body, _child_env(fatbins, tmp_path / "c", CUDA_VISIBLE_DEVICES=""))
    assert r["backend"] == "reference" and r["device"] == "cpu" and r["warnings"] == []
    assert r["driver_loaded"] is False
    assert r["maps"] == r["maps_before"]
    assert r["sha_fast"] == expected_shas["fast"] and r["sha_slow"] == expected_shas["slow"]
    print(json.dumps(dict(libcuda_mapped_by_torch_import=r["maps_before"])))


def _snapshot(root: Path) -> dict:
    out = {}
    for p in root.rglob("*"):
        try:
            st = p.lstat()
        except OSError:
            continue
        out[str(p)] = st.st_mtime_ns
    return out


def test_no_files_outside_cache_dir(fatbins, tmp_path, expected_shas):
    """Temporary HOME, read-only cwd, private TMPDIR: after NVRTC compiles, gate verdicts and selections, files
    exist only under MMAEMU_CACHE_DIR; the checkout is unchanged. A read-only cache dir also works (memory only)."""
    home, cwd, tmp, cache = tmp_path / "home", tmp_path / "ro", tmp_path / "tmp", tmp_path / "home" / "cachedir"
    for d in (home, cwd, tmp):
        d.mkdir()
    cwd.chmod(0o555)
    repo_before = _snapshot(ROOT / "mmaemu")
    try:
        r = run_child(ACCUMULATE_TWICE, _child_env(fatbins, cache, HOME=str(home), TMPDIR=str(tmp),
                                                   XDG_CACHE_HOME=None, MMAEMU_TEST_FORCE_FAIL="fatbin"),
                      cwd=str(cwd))
    finally:
        cwd.chmod(0o755)
    assert r["source"] == "nvrtc" and r["stats"]["nvrtc_compiles"] >= 1
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    outside = [str(p) for p in files if cache not in p.parents]
    assert not outside, outside
    assert any(p.suffix == ".cubin" for p in files) and any(p.parent.name == "verdicts" for p in files)
    assert _snapshot(ROOT / "mmaemu") == repo_before
    # read-only cache directory: still works, nothing written
    ro_cache = tmp_path / "rocache"
    ro_cache.mkdir()
    ro_cache.chmod(0o555)
    try:
        r2 = run_child(ACCUMULATE_TWICE, _child_env(fatbins, ro_cache / "mmaemu", HOME=str(home), TMPDIR=str(tmp),
                                                    MMAEMU_TEST_FORCE_FAIL="fatbin"), cwd=str(tmp))
    finally:
        ro_cache.chmod(0o755)
    assert r2["source"] == "nvrtc" and r2["stats"]["cache_write_errors"] >= 1
    assert not any(ro_cache.rglob("*"))
    assert r2["sha_fast"] == expected_shas["fast"]


def test_fatbin_load_time_fresh_process(fatbins, tmp_path):
    """A fresh process loads a kernel module from the fatbin in <= 20 ms (CLOCK_MONOTONIC_RAW; CUDA context creation
    excluded; audit: ~2 ms)."""
    body = """
        torch.zeros(1, device="cuda")  # context creation is not part of the module load
        torch.cuda.synchronize()
        from mmaemu.backends.cuda import modules, kernels
        t = {}
        for s in kernels.SPECS:
            t0 = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
            fn, src = modules.get_function(s.module, s.entry, "cuda:0")
            t[s.name] = dict(ms=(time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - t0) * 1e3, source=src)
        result(load=t)
    """
    r = run_child(body, _child_env(fatbins, tmp_path / "c"))
    print(json.dumps(r))
    for name, x in r["load"].items():
        assert x["source"] == "fatbin" and x["ms"] <= 20.0, (name, x)


def test_nvrtc_cache_reused_by_second_process(fatbins, tmp_path, expected_shas):
    """With the fatbin forced off, process 1 compiles with NVRTC into MMAEMU_CACHE_DIR; process 2 compiles nothing
    (nvrtc.compile is made to raise there) and gives the same bits."""
    env = _child_env(fatbins, tmp_path / "cache", MMAEMU_TEST_FORCE_FAIL="fatbin")
    r1 = run_child(ACCUMULATE_TWICE, env)
    assert r1["stats"]["nvrtc_compiles"] >= 1
    body = textwrap.dedent("""
        from mmaemu.backends.cuda import nvrtc
        def _no(*a, **k):
            raise AssertionError("NVRTC compile in the second process")
        nvrtc.compile = _no
        """) + textwrap.dedent(ACCUMULATE_TWICE)
    r2 = run_child(body, env)
    print(json.dumps(dict(first=r1["stats"], second=r2["stats"])))
    assert r2["stats"]["nvrtc_compiles"] == 0 and r2["stats"]["nvrtc_disk_hits"] >= 1
    assert r2["source"] == "nvrtc" and r2["sha_fast"] == expected_shas["fast"] == r1["sha_fast"]


def test_second_context(fatbins, tmp_path):
    """B3: a second CUDA context (cuCtxCreate) on the same device loads its own module (compile-time __constant__
    tables) and gives nonzero, bitwise-correct output; the primary context keeps working afterwards."""
    body = """
        import ctypes
        from mmaemu.backends.cuda import driver, modules, kernels
        from mmaemu.backends import ChainRequest
        from mmaemu.model.chain import chain_gemm
        a, b = data(64, 48, 128, seed=11)
        refs = {p: chain_gemm(a.view(torch.uint8), b.view(torch.uint8), fmt_a="e4m3", fmt_b="e4m3", preset=p).float()
                for p in {s.preset for s in kernels.SPECS}}
        dev = torch.device("cuda", 0)
        req = lambda s: ChainRequest(device=dev, fmt_a="e4m3", fmt_b="e4m3", preset=s.preset)
        out = {}
        def bits_ok(t, s):
            return bool((t.view(torch.int32) == refs[s.preset].view(torch.int32)).all())
        for s in kernels.SPECS:
            out[s.name + "_primary_first"] = bits_ok(kernels.chain_fn_for(s, req(s))(a.to(dev), b.to(dev)).cpu(), s)
        L = driver.libcuda()
        vp, sz = ctypes.c_void_p, ctypes.c_size_t
        L.cuCtxCreate_v2.argtypes = [ctypes.POINTER(vp), ctypes.c_uint, ctypes.c_int]
        L.cuCtxPopCurrent_v2.argtypes = [ctypes.POINTER(vp)]
        L.cuCtxPushCurrent_v2.argtypes = [vp]
        L.cuCtxDestroy_v2.argtypes = [vp]
        L.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), sz]
        L.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        L.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, vp, sz]
        L.cuMemcpyDtoH_v2.argtypes = [vp, ctypes.c_uint64, sz]
        primary = driver.current_context()
        ctx2 = vp()
        driver.check(L.cuCtxCreate_v2(ctypes.byref(ctx2), 0, 0), "cuCtxCreate")   # becomes current
        assert driver.current_context() == ctx2.value != primary
        A = a.view(torch.uint8).contiguous().numpy(); B = b.view(torch.uint8).contiguous().numpy()
        M, K = A.shape; N = B.shape[0]
        for s in kernels.SPECS:
            pa, pb, pc = ctypes.c_uint64(), ctypes.c_uint64(), ctypes.c_uint64()
            for p, n in ((pa, A.nbytes), (pb, B.nbytes), (pc, M * N * 4)):
                driver.check(L.cuMemAlloc_v2(ctypes.byref(p), n), "cuMemAlloc")
            driver.check(L.cuMemcpyHtoD_v2(pa.value, A.ctypes.data, A.nbytes), "HtoD")
            driver.check(L.cuMemcpyHtoD_v2(pb.value, B.ctypes.data, B.nbytes), "HtoD")
            loads0 = modules.stats()["module_loads"]
            args = [ctypes.c_void_p(pa.value), ctypes.c_void_p(pb.value), ctypes.c_void_p(pc.value),
                    ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)]
            modules.launch(s.module, s.entry, (-(-N // s.tile_n), -(-M // s.tile_m), 1), (s.threads, 1, 1),
                           s.smem_bytes, args, dev)
            C = torch.empty(M, N, dtype=torch.float32)
            driver.check(L.cuMemcpyDtoH_v2(C.data_ptr(), pc.value, M * N * 4), "DtoH")   # synchronous
            out[s.name + "_ctx2_new_module"] = modules.stats()["module_loads"] == loads0 + 1
            out[s.name + "_ctx2_nonzero"] = int((C != 0).sum())
            out[s.name + "_ctx2_bitwise"] = bits_ok(C, s)
            for p in (pa, pb, pc):
                L.cuMemFree_v2(p.value)
        popped = vp()
        driver.check(L.cuCtxPopCurrent_v2(ctypes.byref(popped)), "cuCtxPopCurrent")
        driver.check(L.cuCtxDestroy_v2(ctx2), "cuCtxDestroy")
        for s in kernels.SPECS:
            out[s.name + "_primary_after"] = bits_ok(kernels.chain_fn_for(s, req(s))(a.to(dev), b.to(dev)).cpu(), s)
            out[s.name + "_ref_nonzero"] = int((refs[s.preset] != 0).sum())
        result(**out)
    """
    r = run_child(body, _child_env(fatbins, tmp_path / "c"))
    print(json.dumps(r))
    for k in KERNELS:
        assert r[f"{k}_primary_first"] and r[f"{k}_ctx2_new_module"] and r[f"{k}_ctx2_bitwise"]
        assert r[f"{k}_ctx2_nonzero"] == r[f"{k}_ref_nonzero"] > 0 and r[f"{k}_primary_after"]


def test_environment_failure_scenarios(fatbins, tmp_path, expected_shas):
    """The evo2-anywhere failure scenarios of verify-p4-system-backends-0/repro.py and -1/repro_nofallback.py, applied
    to mmaemu: no CUDA_HOME, no nvcc / ninja on PATH, unwritable cache / HOME, no GPU. None raises; bits unchanged."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "python").symlink_to(sys.executable)
    ro = tmp_path / "ro"
    ro.mkdir()
    no_tools = dict(CUDA_HOME=None, CUDA_PATH=None, PATH=str(bindir))
    scenarios = {
        "no_toolkit": no_tools,
        "no_toolkit_no_fatbin": dict(no_tools, MMAEMU_FATBIN_DIR=str(tmp_path / "nofatbin")),
        "unwritable_cache_and_home": dict(MMAEMU_CACHE_DIR=str(ro / "cache"), HOME=str(ro), XDG_CACHE_HOME=None),
        "no_gpu": dict(no_tools, CUDA_VISIBLE_DEVICES=""),
        "everything": dict(no_tools, MMAEMU_FATBIN_DIR=str(tmp_path / "nofatbin"), MMAEMU_CACHE_DIR=str(ro / "c"),
                           HOME=str(ro), XDG_CACHE_HOME=None),
    }
    ro.chmod(0o555)
    res = {}
    try:
        for name, upd in scenarios.items():
            env = _child_env(fatbins, tmp_path / f"cache_{name}")
            env.update(upd)
            r = run_child(ACCUMULATE_TWICE, env)
            res[name] = dict(backend=r["backend"], source=r["source"], warnings=len(r["warnings"]))
            assert r["sha_fast"] == expected_shas["fast"] and r["sha_slow"] == expected_shas["slow"], name
    finally:
        ro.chmod(0o755)
    print(json.dumps(res))
    assert res["no_toolkit"]["source"] == "fatbin" and res["no_gpu"]["backend"] == "reference"
    assert res["no_toolkit_no_fatbin"]["source"] == "nvrtc"  # NVRTC comes from the pip wheel, not a toolkit
