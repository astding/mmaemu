#!/usr/bin/env python3
"""T7: validate a built mmaemu wheel on a GPU in a fresh environment (docs/release.md, docs/testing.md).

    $MMAEMU_GPU_LOCK timeout 600 python tools/validate_release.py dist/mmaemu-<ver>-py3-none-any.whl

1. Creates a fresh venv (uv if available, else python -m venv + pip) in a temporary directory and installs torch
   (default: the version and CUDA flavour of the torch this script runs with, from the PyTorch index) and the wheel.
2. Runs everything in a scrubbed environment: PATH holds only the venv's bin/ (and the WSL driver directory for
   nvidia-smi), so no nvcc is reachable; CUDA_HOME, CUDA_PATH, PYTHONPATH, LD_LIBRARY_PATH, LD_PRELOAD, CONDA_PREFIX,
   CUDA_FORCE_PTX_JIT, CUDA_DISABLE_PTX_JIT and every MMAEMU_* variable are unset; HOME, XDG_CACHE_HOME and
   MMAEMU_CACHE_DIR point into the temporary directory; the working directory is outside the source tree, and the
   import location of mmaemu is checked.
3. Checks, each PASS / FAIL / SKIP with its evidence:
     environment    nvcc not on PATH, CUDA_HOME unset, a CUDA device visible to the venv's torch
     install        mmaemu is imported from the venv; version == wheel; the wheel's fatbins (if any) listed
     info           `python -m mmaemu info --json` exits 0
     selftest       `python -m mmaemu selftest --force --json` exits 0 (verdict PASS; v1 and v4 passed)
     loader         with release fatbins in the wheel, the IEEE kernels load from the fatbin (not NVRTC) in the
                    venv, and every kernel that ran loaded the image whose sha256 the wheel's manifest records
     conformance    T3 subset: every available CUDA kernel vs the CPU reference, bitwise (8 data classes x
                    tile-straddling shapes, slow accumulation, split-K serial/pairwise, NaN rows/columns, a
                    misaligned operand, scaled_mm bf16 through backend="auto")
     evo2           one Evo 2 1B projection (truth evo2.evo2_1b_w2k, call 0, 2048x5760x1920): scaled_mm bf16 output
                    == the recorded native RTX 4060 output bit for bit, and the fp32 accumulator's sha256 == the
                    recorded full-accumulator sha256
4. Writes docs/release_logs/<version>_<gpu>.json (or --out). Exit 0: every check PASS (verdict PASS); 3: no
   failure but a check was skipped (PASS-WITH-SKIPS, e.g. --allow-missing-fatbins); 1: a check failed; 2: usage.

Run `--prepare-only` once outside the GPU lock to warm the package cache, so that the locked run stays well under
10 minutes. Internal: `--inner conformance|evo2` runs a check inside the fresh venv.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "mmaemu-release-validation/1"
EVO2_RECORDING = "data/evo2/evo2_1b_w2k.json"
WSL_DRIVER_DIR = "/usr/lib/wsl/lib"
SCRUB = ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT", "CUDA_TOOLKIT_ROOT_DIR", "CUDACXX", "PYTHONPATH", "PYTHONHOME",
         "LD_LIBRARY_PATH", "LD_PRELOAD", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "VIRTUAL_ENV", "PYTHONSTARTUP",
         "PIP_TARGET", "CUDA_FORCE_PTX_JIT", "CUDA_DISABLE_PTX_JIT")  # and every MMAEMU_* variable (scrubbed_env)


def clock() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW) if hasattr(time, "CLOCK_MONOTONIC_RAW") else time.monotonic()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slug(name: str, cc) -> str:
    s = re.sub(r"\b(nvidia|geforce|tesla|quadro)\b", "", name.lower())
    s = re.sub(r"[^a-z0-9]+", "", s) or "gpu"
    return f"{s}-sm{cc[0]}{cc[1]}" if cc else s


# =============================================================================================== inner checks
# These run inside the fresh venv (`--inner ...`); they import only torch, numpy and the installed mmaemu.
GENS = ["random", "allcodes", "tiny", "sat", "spread", "blockscale", "cancel", "zeros"]
KERNELS = ("cuda:v1", "cuda:v4", "cuda:v9", "cuda:tc3")
REQUIRED_KERNELS = ("cuda:v1", "cuda:v4")  # IEEE-exact: must load and pass everywhere they are supported
SHAPES = [(M, N, K) for K in (16, 48, 1920) for M in (1, 65, 200) for N in (1, 127, 257)]


def _codes(torch, kind, shape, g):
    rows, K = shape
    finite = torch.tensor([c for c in range(256) if c & 0x7F != 0x7F], dtype=torch.uint8)
    subn = torch.tensor([c | s for c in range(1, 8) for s in (0, 0x80)], dtype=torch.uint8)
    e4m3 = torch.float8_e4m3fn
    if kind == "random":
        return (torch.randn(shape, generator=g) * 40).clamp(-448, 448).to(e4m3).view(torch.uint8)
    if kind == "allcodes":
        return finite[torch.randint(0, len(finite), shape, generator=g)]
    if kind == "tiny":
        c = subn[torch.randint(0, len(subn), shape, generator=g)]
        z = torch.rand(shape, generator=g) < 0.1
        return torch.where(z, torch.where(torch.rand(shape, generator=g) < 0.5, 0x00, 0x80).to(torch.uint8), c)
    if kind == "sat":
        return torch.where(torch.rand(shape, generator=g) < 0.5, 0x7E, 0xFE).to(torch.uint8)
    sign = (torch.rand(shape, generator=g) < 0.5).to(torch.uint8) << 7
    man = torch.randint(0, 8, shape, generator=g, dtype=torch.uint8)
    if kind == "spread":
        big = torch.rand(shape, generator=g) < 0.5
        e = torch.where(big, torch.randint(13, 16, shape, generator=g), torch.zeros(shape, dtype=torch.long))
        man = torch.where(big | (man != 0), man, torch.ones_like(man))
        c = sign | (e.to(torch.uint8) << 3) | man
        return torch.where((c & 0x7F) == 0x7F, c - 1, c)
    if kind == "blockscale":
        eb = torch.randint(0, 16, (rows, (K + 15) // 16), generator=g).repeat_interleave(16, 1)[:, :K]
        c = sign | (eb.to(torch.uint8) << 3) | man
        return torch.where((c & 0x7F) == 0x7F, c - 1, c)
    raise ValueError(kind)


def make_case(torch, kind, M, N, K, seed):
    """(a [M,K], b [N,K]) E4M3 on the CPU; the data classes of the audit's kernel test (test_kernel.py)."""
    g = torch.Generator().manual_seed(seed)
    e4m3 = torch.float8_e4m3fn
    if kind == "cancel":
        a = _codes(torch, "random", (M, K), g).view(e4m3).float()
        b = _codes(torch, "random", (N, K), g).view(e4m3).float()
        a[:, 1::2] = -a[:, 0::2]
        bb = b[:, 0::2].to(e4m3).view(torch.uint8).clone()
        step = torch.randint(-1, 2, bb.shape, generator=g, dtype=torch.int16)
        step = torch.where(torch.rand(bb.shape, generator=g) < 0.25, step, torch.zeros_like(step))
        mag = ((bb & 0x7F).to(torch.int16) + step).clamp(0, 0x7E).to(torch.uint8)
        b[:, 1::2] = ((bb & 0x80) | mag).view(e4m3).float()
        return a.to(e4m3), b.to(e4m3)
    if kind == "zeros":
        a = _codes(torch, "random", (M, K), g)
        b = _codes(torch, "random", (N, K), g)
        a[torch.rand(M, generator=g) < 0.3] = 0
        b[torch.rand(N, generator=g) < 0.3] = 0
        a[torch.rand((M, K), generator=g) < 0.2] = 0x80
        b[torch.rand((N, K), generator=g) < 0.2] = 0x80
        return a.view(e4m3), b.view(e4m3)
    return _codes(torch, kind, (M, K), g).view(e4m3), _codes(torch, kind, (N, K), g).view(e4m3)


def inner_conformance(args) -> dict:
    import zlib

    import torch

    import mmaemu
    from mmaemu.testing import assert_bitwise_equal

    # torch's default (one thread per physical core), at most 8: one thread per logical CPU oversubscribes SMT hosts,
    # and with other load the reference's many small ops then ran 300x slower (17.8 s instead of 0.06 s per case).
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    dev = torch.device("cuda", 0)

    def snapshot():
        st = mmaemu.backends_status(dev)
        return {str(getattr(v, "name", k)): {"source": getattr(v, "source", None),
                                             "available": getattr(v, "available", None),
                                             "validated_here": getattr(v, "validated_here", None),
                                             "reason": getattr(v, "reason", "")} for k, v in dict(st).items()}

    out = {"kernels": [], "skipped": {}, "cases": 0, "outputs": 0, "mismatches": [], "status_before": snapshot()}
    # Try every kernel explicitly (a kernel may be reported available only after it was loaded and gated).
    probe_a, probe_b = make_case(torch, "random", 16, 8, 64, 1)
    for k in KERNELS:
        try:
            mmaemu.accumulate(probe_a.to(dev), probe_b.to(dev), use_fast_accum=True, splitk=1, backend=k)
            torch.cuda.synchronize(dev)
            out["kernels"].append(k)
        except Exception as e:  # noqa: BLE001 - BackendUnavailableError and friends: record the reason
            out["skipped"][k] = f"{type(e).__name__}: {str(e).strip().splitlines()[0][:300] if str(e).strip() else ''}"
    kernels = out["kernels"]
    missing = [k for k in REQUIRED_KERNELS if k not in kernels]
    if missing:
        out["error"] = f"required IEEE kernels unavailable on this device: {missing} ({out['skipped']})"
    if not kernels:
        out["status_after"] = snapshot()
        return out

    def compare(label, got, ref):
        out["cases"] += 1
        out["outputs"] += ref.numel()
        try:
            assert_bitwise_equal(got.cpu(), ref.cpu(), msg=label)
        except AssertionError as e:
            out["mismatches"].append(f"{label}: {str(e).splitlines()[0][:300]}")

    def acc(a, b, backend, **kw):
        if backend == "reference":
            return mmaemu.accumulate(a, b, backend="reference", **kw)
        r = mmaemu.accumulate(a.to(dev), b.to(dev), backend=backend, **kw)
        torch.cuda.synchronize(dev)
        return r

    t0 = clock()
    # 1. data classes x tile-straddling shapes, fast accumulation, unsplit
    for kind in GENS:
        for M, N, K in SHAPES:
            a, b = make_case(torch, kind, M, N, K, zlib.crc32(f"{kind}-{M}-{N}-{K}".encode()))
            ref = acc(a, b, "reference", use_fast_accum=True, splitk=1)
            for k in kernels:
                compare(f"{k} {kind} {M}x{N}x{K} fast", acc(a, b, k, use_fast_accum=True, splitk=1), ref)
    # 2. plan compositions: slow accumulation, split-K serial and pairwise, K % 16 != 0 (accumulate pads)
    a, b = make_case(torch, "random", 64, 96, 1920, 11)
    for kw in ({"use_fast_accum": False, "splitk": 1}, {"use_fast_accum": True, "splitk": (3, "pairwise")},
               {"use_fast_accum": True, "splitk": (8, "serial")}, {"use_fast_accum": False, "splitk": (2, "serial")},
               {"use_fast_accum": False, "splitk": (12, "pairwise")}):
        ref = acc(a, b, "reference", **kw)
        for k in kernels:
            compare(f"{k} plan {kw}", acc(a, b, k, **kw), ref)
    a2, b2 = make_case(torch, "random", 33, 40, 200, 12)
    ref = acc(a2, b2, "reference", use_fast_accum=True, splitk=1)
    for k in kernels:
        compare(f"{k} K=200", acc(a2, b2, k, use_fast_accum=True, splitk=1), ref)
    # 3. NaN codes in a row of A and a column of B -> canonical NaN exactly there
    a3, b3 = make_case(torch, "random", 48, 40, 256, 13)
    a3 = a3.view(torch.uint8).clone()
    b3 = b3.view(torch.uint8).clone()
    a3[5, 17] = 0x7F
    b3[9, 200] = 0xFF
    a3, b3 = a3.view(torch.float8_e4m3fn), b3.view(torch.float8_e4m3fn)
    ref = acc(a3, b3, "reference", use_fast_accum=True, splitk=1)
    for k in kernels:
        compare(f"{k} NaN rows/cols", acc(a3, b3, k, use_fast_accum=True, splitk=1), ref)
    # 4. misaligned operand (storage offset 1 byte): accumulate copies, the result is unchanged
    a4, b4 = make_case(torch, "allcodes", 17, 31, 64, 14)
    for k in kernels:
        buf = torch.zeros(17 * 64 + 1, dtype=torch.uint8, device=dev)
        am = buf[1:].view(17, 64)
        am.copy_(a4.view(torch.uint8).to(dev))
        got = mmaemu.accumulate(am.view(torch.float8_e4m3fn), b4.to(dev), use_fast_accum=True, splitk=1,
                                backend=k)
        torch.cuda.synchronize(dev)
        compare(f"{k} misaligned", got, acc(a4, b4, "reference", use_fast_accum=True, splitk=1))
    # 5. scaled_mm through backend="auto" (bf16 output, Evo 2 settings) vs the CPU
    a5, b5 = make_case(torch, "random", 64, 256, 1920, 15)
    sa, sb = torch.tensor(0.0123), torch.tensor(-0.75)
    ref = mmaemu.scaled_mm(a5, b5.t(), sa, sb, out_dtype=torch.bfloat16, use_fast_accum=True, splitk=1,
                           backend="reference")
    got = mmaemu.scaled_mm(a5.to(dev), b5.to(dev).t(), sa.to(dev), sb.to(dev), out_dtype=torch.bfloat16,
                           use_fast_accum=True, splitk=1)
    compare("scaled_mm auto bf16", got, ref)
    out["seconds"] = round(clock() - t0, 3)
    out["status_after"] = snapshot()  # same process, after every kernel ran: the loader each one used
    from mmaemu.backends.cuda import modules
    out["fatbin_dir"] = str(modules.fatbin_dir())
    out["modules"] = {}
    for k in kernels:                 # the image each kernel ran from, to compare with the wheel's manifest
        lm = modules.module_info(k.split(":", 1)[1], dev)
        out["modules"][k] = {"source": lm.source, "sha256": lm.sha256}
    return out


def _load_truth_case(truth: Path):
    import numpy as np

    rec = json.loads((truth / EVO2_RECORDING).read_text())
    case = rec["cases"][0]

    def array(spec):
        with np.load(truth / "data" / "evo2" / spec["file"], allow_pickle=False) as z:
            arr = z[spec["key"]]
        h = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
        if h != spec["sha256"]:
            raise ValueError(f"{spec['key']}: sha256 {h} != recorded {spec['sha256']}")
        return arr

    outs = {o["name"]: o for o in case["outputs"]}
    return rec, case, array(case["inputs"]["a"]), array(case["inputs"]["b"]), outs, array


def inner_evo2(args) -> dict:
    import numpy as np
    import torch

    import mmaemu
    from mmaemu.testing import assert_bitwise_equal

    truth = Path(args.truth_dir)
    rec, case, a_np, b_np, outs, array = _load_truth_case(truth)
    yspec = outs["y_native"]
    if yspec.get("rows") and yspec.get("rows") != [0, yspec["shape"][0]] and "file" in yspec:
        raise ValueError("call 0 must hold the full native output")
    y_bits = array(yspec)
    p = dict(case["params"])
    p.update(yspec.get("params", {}))
    sa = torch.tensor(np.array([p["scale_a"]], dtype=np.uint32).view(np.float32)[0])
    sb = torch.tensor(np.array([p["scale_b"]], dtype=np.uint32).view(np.float32)[0])
    dev = torch.device("cuda", 0)
    a = torch.from_numpy(a_np).view(torch.float8_e4m3fn).to(dev)
    b = torch.from_numpy(b_np).view(torch.float8_e4m3fn).to(dev)
    out = {"recording": rec["id"], "case": case["id"], "shape": [int(a.shape[0]), int(b.shape[0]), int(a.shape[1])]}
    plan = mmaemu.explain(a, b.t(), sa.to(dev), sb.to(dev), out_dtype=torch.bfloat16, use_fast_accum=True)
    out["backend"] = str(getattr(plan, "backend", None))
    out["split"] = str(getattr(plan, "split", None))
    t0 = clock()
    y = mmaemu.scaled_mm(a, b.t(), sa.to(dev), sb.to(dev), out_dtype=torch.bfloat16, use_fast_accum=True)
    torch.cuda.synchronize(dev)
    out["scaled_mm_seconds"] = round(clock() - t0, 3)
    expect = torch.from_numpy(np.ascontiguousarray(y_bits).view(np.int16)).view(torch.bfloat16)
    try:
        assert_bitwise_equal(y.cpu(), expect, msg="evo2 y_native")
        out["y_native_equal"] = True
    except AssertionError as e:
        out["y_native_equal"] = False
        out["y_native_mismatch"] = str(e).splitlines()[0][:300]
    acc = mmaemu.accumulate(a, b, use_fast_accum=True)
    torch.cuda.synchronize(dev)
    h = hashlib.sha256(acc.cpu().contiguous().view(torch.int32).numpy().tobytes()).hexdigest()
    out["acc_sha256"] = h
    out["acc_full_sha256_recorded"] = outs["acc_full"]["sha256"]
    out["acc_equal"] = h == outs["acc_full"]["sha256"]
    return out


# =============================================================================================== outer driver
class Runner:
    def __init__(self, args) -> None:
        self.args = args
        self.checks: list[dict] = []
        self.t_start = clock()

    def add(self, name: str, status: str, **details) -> dict:
        row = {"name": name, "status": status, **details}
        self.checks.append(row)
        extra = details.get("reason") or details.get("summary") or ""
        print(f"[{clock() - self.t_start:6.1f} s] {status:4s} {name}" + (f": {extra}" if extra else ""), flush=True)
        return row

    def remaining(self) -> float:
        return max(30.0, self.args.budget_s - (clock() - self.t_start))


def scrubbed_env(tmp: Path, venv: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in SCRUB and not k.startswith("MMAEMU_")}
    path = [str(venv / "bin")]
    if os.path.isdir(WSL_DRIVER_DIR):
        path.append(WSL_DRIVER_DIR)
    env["PATH"] = os.pathsep.join(path)
    env["HOME"] = str(tmp / "home")
    env["XDG_CACHE_HOME"] = str(tmp / "xdg-cache")
    env["MMAEMU_CACHE_DIR"] = str(tmp / "mmaemu-cache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    for d in ("home", "xdg-cache", "mmaemu-cache", "work"):
        (tmp / d).mkdir(parents=True, exist_ok=True)
    return env


def torch_default(args) -> tuple[str, str]:
    """(requirement, index URL) for torch in the fresh venv."""
    if args.torch:
        return args.torch, args.torch_index
    from importlib.metadata import PackageNotFoundError, version
    try:
        v = version("torch")
    except PackageNotFoundError:
        return "torch", args.torch_index
    base, _, local = v.partition("+")
    index = args.torch_index or (f"https://download.pytorch.org/whl/{local}" if local else None)
    return f"torch=={base}", index


def create_venv(args, venv: Path, wheel: Path, log) -> dict:
    uv = shutil.which("uv") if not args.no_uv else None
    req, index = torch_default(args)
    base_py = args.python or sys.executable
    info = {"tool": "uv" if uv else "venv+pip", "torch_requirement": req, "torch_index": index}
    vpy = venv / "bin" / "python"
    t0 = clock()
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX")}
    if uv:
        cmds = [[uv, "venv", "-q", "--python", base_py, str(venv)],
                [uv, "pip", "install", "-q", "--python", str(vpy), req] + (["--index-url", index] if index else []),
                [uv, "pip", "install", "-q", "--python", str(vpy), str(wheel), "numpy"]]
    else:
        cmds = [[base_py, "-m", "venv", str(venv)],
                [str(vpy), "-m", "pip", "install", "-q", req] + (["--index-url", index] if index else []),
                [str(vpy), "-m", "pip", "install", "-q", str(wheel), "numpy"]]
    for c in cmds:
        r = subprocess.run(c, capture_output=True, text=True, env=env, timeout=args.install_timeout_s)
        log.append({"cmd": c, "returncode": r.returncode, "stderr_tail": r.stderr[-2000:]})
        if r.returncode != 0:
            raise RuntimeError(f"{' '.join(c[:4])} ... failed: {r.stderr.strip()[-500:]}")
    info["seconds"] = round(clock() - t0, 1)
    return info


def run_py(vpy: Path, argv: list[str], env: dict, cwd: Path, timeout: float):
    t0 = clock()
    r = subprocess.run([str(vpy), *argv], capture_output=True, text=True, env=env, cwd=str(cwd), timeout=timeout)
    return r, round(clock() - t0, 2)


def inner_json(vpy: Path, which: str, env: dict, cwd: Path, timeout: float, truth: str):
    out = cwd / f"inner_{which}.json"
    r, secs = run_py(vpy, [str(Path(__file__).resolve()), "--inner", which, "--inner-out", str(out),
                           "--truth-dir", truth], env, cwd, timeout)
    data = json.loads(out.read_text()) if out.is_file() else None
    return r, secs, data


def wheel_facts(wheel: Path) -> dict:
    m = re.fullmatch(r"mmaemu-([^-]+)-py3-none-any\.whl", wheel.name)
    facts = {"file": wheel.name, "sha256": sha256_file(wheel), "bytes": wheel.stat().st_size,
             "version": m.group(1) if m else None}
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        man = "mmaemu/backends/cuda/bin/manifest.json"
        facts["fatbins"] = None
        if man in names:
            mods = json.loads(z.read(man)).get("modules", {})
            facts["fatbins"] = {k: {"sass_archs": v.get("sass_archs"), "ptx_arch": v.get("ptx_arch"),
                                    "sha256": v.get("sha256")} for k, v in mods.items()}
    return facts


def outer(args) -> int:
    wheel = Path(args.wheel).resolve()
    R = Runner(args)
    report: dict = {"format": FORMAT, "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                    "host": {"platform": platform.platform(), "python": platform.python_version(),
                             "machine": platform.machine()},
                    "source_tree_commit": _git_head(), "checks": R.checks}
    if not wheel.is_file():
        print(f"no such wheel: {wheel}", file=sys.stderr)
        return 2
    wf = wheel_facts(wheel)
    report["wheel"] = wf
    if wf["version"] is None:
        print(f"not an mmaemu wheel name: {wheel.name}", file=sys.stderr)
        return 2
    tmp = Path(tempfile.mkdtemp(prefix="mmaemu-t7-", dir=args.tmp_dir))
    venv = Path(args.venv).resolve() if args.venv else tmp / "venv"
    log: list = []
    report["install_log"] = log
    try:
        try:
            report["venv"] = create_venv(args, venv, wheel, log)
            R.add("venv", "PASS", summary=f"{report['venv']['tool']}, {report['venv']['torch_requirement']} "
                  f"({report['venv']['seconds']} s)")
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            R.add("venv", "FAIL", reason=str(e)[:500])
            return finish(args, report, R, wf, None)
        if args.prepare_only:
            kept = "kept" if args.venv or args.keep else "removed again; the package cache is warm"
            print(f"prepared the venv {venv} ({kept}); not validating (--prepare-only)")
            return 0
        vpy = venv / "bin" / "python"
        env = scrubbed_env(tmp, venv)
        work = tmp / "work"

        # environment
        nvcc = shutil.which("nvcc", path=env["PATH"])
        probe = ("import json, torch, sys; ok = torch.cuda.is_available(); "
                 "p = torch.cuda.get_device_properties(0) if ok else None; "
                 "print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, 'available': ok, "
                 "'name': p.name if p else None, 'cc': [p.major, p.minor] if p else None, "
                 "'sm_count': p.multi_processor_count if p else None, 'executable': sys.executable}))")
        r, secs = run_py(vpy, ["-c", probe], env, work, 120)
        dev = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else {}
        report["device"] = dev
        problems = []
        if nvcc:
            problems.append(f"nvcc reachable on PATH: {nvcc}")
        if any(k in env for k in ("CUDA_HOME", "CUDA_PATH")):
            problems.append("CUDA_HOME/CUDA_PATH set")
        if not dev.get("available"):
            problems.append(f"no CUDA device visible to the venv's torch ({r.stderr.strip()[-300:]})")
        R.add("environment", "FAIL" if problems else "PASS", reason="; ".join(problems),
              summary=f"PATH={env['PATH']}; nvcc not found; CUDA_HOME unset; {dev.get('name')} "
                      f"cc {dev.get('cc')} torch {dev.get('torch')}", nvcc_on_path=nvcc, path=env["PATH"],
              seconds=secs)

        # install location and version
        code = "import mmaemu, json; print(json.dumps({'file': mmaemu.__file__, 'version': mmaemu.__version__}))"
        r, secs = run_py(vpy, ["-c", code], env, work, 120)
        inst = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else {}
        ok = (r.returncode == 0 and str(inst.get("file", "")).startswith(str(venv))
              and inst.get("version") == wf["version"])
        R.add("install", "PASS" if ok else "FAIL", summary=f"mmaemu {inst.get('version')} from {inst.get('file')}",
              reason="" if ok else (r.stderr.strip()[-400:] or f"import location/version mismatch: {inst}"),
              seconds=secs, fatbins_in_wheel=wf["fatbins"])

        # info
        info_path = work / "info.json"
        r, secs = run_py(vpy, ["-m", "mmaemu", "info", "--json", str(info_path)], env, work, 300)
        info = json.loads(info_path.read_text()) if info_path.is_file() else None
        report["info_before_selftest"] = info
        R.add("info", "PASS" if r.returncode == 0 and info else "FAIL", seconds=secs,
              reason="" if r.returncode == 0 else r.stderr.strip()[-500:],
              summary=f"profiles matching: {[p['name'] for p in (info or {}).get('profiles', []) if p.get('matches_device')]}")

        # selftest
        st_path = work / "selftest.json"
        r, secs = run_py(vpy, ["-m", "mmaemu", "selftest", "--force", "--json", str(st_path)], env, work, 300)
        st = json.loads(st_path.read_text()) if st_path.is_file() else None
        report["selftest"] = st
        kern = {k.split(":", 1)[-1]: v for k, v in (st or {}).get("kernels", {}).items()}
        need = [k for k in ("v1", "v4") if k not in kern or not kern[k].get("passed")]
        ok = r.returncode == 0 and st and st.get("verdict") == "PASS" and not need
        R.add("selftest", "PASS" if ok else "FAIL", seconds=secs,
              summary=f"verdict {(st or {}).get('verdict')}; " + ", ".join(
                  f"{k}={'pass' if v.get('passed') else 'fail'}" for k, v in kern.items()),
              reason="" if ok else (f"v1/v4 not passed: {need}; " if need else "") + ((st or {}).get("reason") or
                                                                                      r.stderr.strip()[-300:]))

        # conformance (T3 subset)
        try:
            r, secs, data = inner_json(vpy, "conformance", env, work, R.remaining(), args.truth_dir)
        except subprocess.TimeoutExpired:
            r, secs, data = None, None, None
        report["conformance"] = data
        if data is None:
            R.add("conformance", "FAIL", reason="no result" + (f": {r.stderr.strip()[-600:]}" if r else " (timeout)"))
        else:
            ok = r.returncode == 0 and not data.get("error") and not data.get("mismatches") and data.get("kernels")
            R.add("conformance", "PASS" if ok else "FAIL", seconds=secs,
                  summary=f"kernels {data.get('kernels')} (skipped {data.get('skipped')}): {data.get('cases')} "
                          f"comparisons, {data.get('outputs')} outputs, {len(data.get('mismatches', []))} mismatching",
                  reason="" if ok else (data.get("error") or "; ".join(data.get("mismatches", [])[:3])
                                        or r.stderr.strip()[-500:]))

        # loader: the IEEE kernels ran from the shipped fatbin (status snapshot of the conformance process), from the
        # venv's package, and every kernel that ran loaded the image the wheel's manifest records
        conf = report.get("conformance") or {}
        after = conf.get("status_after") or {}
        src = {n: b.get("source") for n, b in after.items() if isinstance(b, dict)}
        if wf["fatbins"]:
            bad = {n: src.get(n) for n in REQUIRED_KERNELS if src.get(n) != "fatbin"}
            for k, m in (conf.get("modules") or {}).items():
                want = (wf["fatbins"].get(k.split(":", 1)[1]) or {}).get("sha256")
                if m.get("source") == "fatbin" and m.get("sha256") != want:
                    bad[k] = f"image sha256 {m.get('sha256')} != wheel manifest {want}"
            fdir = conf.get("fatbin_dir") or ""
            if not Path(fdir).resolve().is_relative_to(venv.resolve()):
                bad["fatbin_dir"] = f"{fdir!r} is not in the venv"
            R.add("loader", "FAIL" if bad else "PASS", summary=f"sources after use: {src}; fatbin dir {fdir}",
                  reason=f"not loaded from the wheel's fatbins: {bad}" if bad else "")
        elif args.allow_missing_fatbins:
            R.add("loader", "SKIP", reason="the wheel has no fatbins (--allow-missing-fatbins): kernels via NVRTC",
                  summary=f"sources after use: {src}")
        else:
            R.add("loader", "FAIL", reason="the wheel carries no release fatbins (bin/manifest.json); pass "
                                           "--allow-missing-fatbins only for a pre-release wheel")

        # Evo 2 projection
        if not args.truth_dir:
            R.add("evo2", "FAIL", reason="no hardware ground truth given (--truth-dir or MMAEMU_TRUTH_DIR); the Evo 2 "
                                         "projection check needs its recording " + EVO2_RECORDING)
        elif not (Path(args.truth_dir) / EVO2_RECORDING).is_file():
            R.add("evo2", "FAIL", reason=f"hardware recording not found: {Path(args.truth_dir) / EVO2_RECORDING} "
                                         "(--truth-dir / MMAEMU_TRUTH_DIR)")
        else:
            try:
                r, secs, data = inner_json(vpy, "evo2", env, work, R.remaining(), args.truth_dir)
            except subprocess.TimeoutExpired:
                r, secs, data = None, None, None
            report["evo2"] = data
            if data is None:
                R.add("evo2", "FAIL", reason="no result" + (f": {r.stderr.strip()[-600:]}" if r else " (timeout)"))
            else:
                ok = r.returncode == 0 and data.get("y_native_equal") and data.get("acc_equal")
                R.add("evo2", "PASS" if ok else "FAIL", seconds=secs,
                      summary=f"{data['recording']} {data['case']} {data['shape']}: bf16 output == native "
                              f"{data.get('y_native_equal')}, accumulator sha256 == recorded {data.get('acc_equal')} "
                              f"(backend {data.get('backend')}, split {data.get('split')})",
                      reason="" if ok else (data.get("y_native_mismatch") or r.stderr.strip()[-500:]))
        return finish(args, report, R, wf, dev)
    finally:
        if not args.keep and not args.venv:
            shutil.rmtree(tmp, ignore_errors=True)
        elif args.keep:
            print(f"kept {tmp}")


def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def finish(args, report: dict, R: Runner, wf: dict, dev: dict | None) -> int:
    statuses = [c["status"] for c in R.checks]
    verdict = "PASS" if statuses and all(s == "PASS" for s in statuses) else "FAIL"
    if verdict == "FAIL" and statuses and all(s in ("PASS", "SKIP") for s in statuses):
        verdict = "PASS-WITH-SKIPS"
    report["verdict"] = verdict
    report["seconds_total"] = round(clock() - R.t_start, 1)
    gpu = slug(dev["name"], dev.get("cc")) if dev and dev.get("name") else "nogpu"
    out = Path(args.out) if args.out else ROOT / "docs" / "release_logs" / f"{wf['version']}_{gpu}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"\nT7 {verdict}: {sum(s == 'PASS' for s in statuses)}/{len(statuses)} checks PASS -> {out}")
    return {"PASS": 0, "PASS-WITH-SKIPS": 3}.get(verdict, 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wheel", nargs="?", help="dist/mmaemu-<ver>-py3-none-any.whl")
    ap.add_argument("--out", help="report path (default docs/release_logs/<version>_<gpu>.json)")
    ap.add_argument("--python", help="base interpreter for the venv (default: this one)")
    ap.add_argument("--torch", help="torch requirement for the venv (default: torch==<this torch's version>)")
    ap.add_argument("--torch-index", help="index URL for torch (default: the PyTorch index of this torch's flavour)")
    ap.add_argument("--no-uv", action="store_true", help="use python -m venv + pip even if uv is available")
    ap.add_argument("--venv", help="create the venv here instead of a temporary directory (kept)")
    ap.add_argument("--prepare-only", action="store_true",
                    help="create the venv and install, then stop (run outside the GPU lock to warm the cache)")
    ap.add_argument("--keep", action="store_true", help="keep the temporary directory")
    ap.add_argument("--tmp-dir", help="parent of the temporary directory")
    ap.add_argument("--allow-missing-fatbins", action="store_true",
                    help="accept a wheel without fatbins (pre-release / sdist-built wheel): the loader check is SKIP")
    ap.add_argument("--truth-dir", default=os.environ.get("MMAEMU_TRUTH_DIR", ""),
                    help="hardware ground truth for the Evo 2 projection (default $MMAEMU_TRUTH_DIR; maintainer data, "
                         "not published)")
    ap.add_argument("--budget-s", type=float, default=560.0, help="overall time budget in seconds")
    ap.add_argument("--install-timeout-s", type=float, default=900.0)
    ap.add_argument("--inner", choices=("conformance", "evo2"), help=argparse.SUPPRESS)
    ap.add_argument("--inner-out", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.inner:
        fn = {"conformance": inner_conformance, "evo2": inner_evo2}[args.inner]
        data = fn(args)
        Path(args.inner_out).write_text(json.dumps(data, indent=1, default=str))
        return 0
    if not args.wheel:
        ap.error("the wheel is required")
    return outer(args)


if __name__ == "__main__":
    sys.exit(main())
