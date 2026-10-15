#!/usr/bin/env python
"""Sustained-throughput benchmark of mmaemu's CUDA kernels (T6; docs/backends.md).

  $GPU timeout 600 python tools/bench_kernels.py --kernels v1,v4 --shape 1b --M 2048 --data gauss40 \
      --fatbin-dir build/fatbin_rel [--out results.json]

Method
  - Timing with CUDA events only. Host timers are not used for GPU work; tools/hostclock.py runs first and its
    ratio (Python's monotonic clock vs CLOCK_MONOTONIC_RAW) is recorded. --timer host-raw (CLOCK_MONOTONIC_RAW around
    synchronised launches) exists for cross-checks and is refused when the host-clock gate fails.
  - Kernels are loaded as for users (fatbin from --fatbin-dir / the package, else NVRTC); performance acceptance
    requires a fatbin (--allow-nvrtc records NVRTC numbers instead, marked as such).
  - Exactness gate before timing: v1 is compared bit for bit with the CPU reference on a row/column subset, and every
    other kernel with v1 on the full output (v1 is checked even when it is not timed); a mismatch stops the run,
    which writes its report and exits 1.
  - Warm-up >= 1.5 s per kernel, then rounds of windows >= 1.0 s each (kernels interleaved, order rotated per round);
    per window: CUDA events around n back-to-back launches. Reported: median over windows of TMAC/s, min/max.
  - nvidia-smi samples SM clock, power and throttle reasons every 200 ms during the windows; MAC/clk/SM uses the mean
    sampled SM clock and the SM count.
Shapes: 1b (K 1920, N 5760), 7b (4096, 12288), 40b (8192, 24576), or MxNxK given explicitly with --shape NxK.
Data: gauss40 (randn * 40 clamped to +-448, as E4M3), adversarial (selftest's adversarial generator), d<k>
(every 16-block spans k exponents), zeros.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

SHAPES = {"1b": (1920, 5760), "7b": (4096, 12288), "40b": (8192, 24576)}
NVSMI_CANDIDATES = ("/usr/lib/wsl/lib/nvidia-smi", "nvidia-smi")


def raw_now() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


class SmiSampler:
    """Background nvidia-smi sampler (SM clock MHz, power W, throttle reasons, temperature)."""

    FIELDS = "clocks.sm,clocks.mem,power.draw,clocks_throttle_reasons.active,temperature.gpu"

    def __init__(self, index: int, period_ms: int = 200):
        self.exe = next((p for p in NVSMI_CANDIDATES if shutil.which(p) or os.path.isfile(p)), None)
        self.index, self.period_ms = index, period_ms
        self.rows: list[dict] = []
        self.proc = None
        self.thread = None

    def start(self):
        if self.exe is None:
            return self
        self.proc = subprocess.Popen([self.exe, f"--id={self.index}", f"--query-gpu={self.FIELDS}",
                                      "--format=csv,noheader,nounits", f"-lms={self.period_ms}"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        return self

    def _read(self):
        def num(x):
            try:
                return float(x)
            except ValueError:
                return None  # e.g. "[N/A]": WSL reports no instantaneous power draw
        for line in self.proc.stdout:
            p = [x.strip() for x in line.split(",")]
            if len(p) < 5 or num(p[0]) is None:
                continue
            self.rows.append(dict(t=raw_now(), sm_mhz=num(p[0]), mem_mhz=num(p[1]), power_w=num(p[2]),
                                  throttle=p[3], temp_c=num(p[4])))

    def power_samples(self) -> dict:
        """nvidia-smi -q -d POWER "Power Samples" (the driver's own recent window; available on WSL, where the
        instantaneous power.draw query is N/A)."""
        if self.exe is None:
            return {}
        try:
            out = subprocess.run([self.exe, f"--id={self.index}", "-q", "-d", "POWER"], capture_output=True,
                                 text=True, timeout=20).stdout
        except (OSError, subprocess.SubprocessError):
            return {}
        res, inside = {}, False
        for line in out.splitlines():
            if "Power Samples" in line:
                inside = True
                continue
            if inside and ":" in line:
                k, v = (x.strip() for x in line.split(":", 1))
                if k in ("Duration", "Number of Samples", "Max", "Min", "Avg"):
                    res[k] = v
                else:
                    break
            if "Current Power Limit" in line and "power_limit" not in res:
                res["power_limit"] = line.split(":", 1)[1].strip()
        return res

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        return self

    def between(self, spans) -> dict:
        """Aggregate the samples taken inside any of the (t0, t1) spans (one kernel's own windows)."""
        r = [x for x in self.rows if any(t0 <= x["t"] <= t1 for t0, t1 in spans)]
        if not r:
            return {}
        sm = [x["sm_mhz"] for x in r]
        pw = [x["power_w"] for x in r if x["power_w"] is not None]
        return dict(n=len(r), sm_mhz_mean=statistics.fmean(sm), sm_mhz_min=min(sm), sm_mhz_max=max(sm),
                    power_w_mean=statistics.fmean(pw) if pw else None, throttle=sorted({x["throttle"] for x in r}),
                    temp_c_max=max((x["temp_c"] for x in r if x["temp_c"] is not None), default=None))


def make_data(kind: str, rows: int, K: int, seed: int, dev):
    import torch

    if kind == "gauss40":
        g = torch.Generator(device=dev).manual_seed(seed)
        return (torch.randn((rows, K), generator=g, device=dev) * 40).clamp(-448, 448).to(torch.float8_e4m3fn)
    if kind == "adversarial":
        from mmaemu.backends.cuda.selftest import _adversarial

        g = torch.Generator().manual_seed(seed)
        return _adversarial(rows, K, g, seed % 2).view(torch.float8_e4m3fn).to(dev)
    if kind.startswith("d") and kind[1:].isdigit():  # every 16-block spans `span` exponent fields
        span = int(kind[1:])
        g = torch.Generator().manual_seed(seed)
        base = torch.randint(1, 16 - min(span, 14), (rows, (K + 15) // 16), generator=g).repeat_interleave(16, 1)[:, :K]
        e = (base + torch.randint(0, span, (rows, K), generator=g)).clamp(1, 15)
        c = ((torch.rand((rows, K), generator=g) < 0.5).to(torch.uint8) << 7) | (e.to(torch.uint8) << 3) | \
            torch.randint(0, 8, (rows, K), generator=g, dtype=torch.uint8)
        c = torch.where((c & 0x7F) == 0x7F, c - 1, c)
        return c.view(torch.float8_e4m3fn).to(dev)
    if kind == "zeros":
        return torch.zeros((rows, K), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
    raise SystemExit(f"unknown --data {kind!r}")


def v1_variant_fn(defines: str, dev):
    """ChainFn of a v1 compile-time variant (defines "MMAEMU_V1_DEC=0,MMAEMU_V1_EXP=1"), built with nvcc -cubin and the
    release flags for this device, loaded and launched directly (bench-only; used for the DEC1 / exp1b decision)."""
    import ctypes
    import tempfile

    import torch

    from mmaemu.backends.cuda import driver
    from mmaemu.backends.cuda.kernels import V1
    from mmaemu.backends.cuda.modules import CSRC

    cc = torch.cuda.get_device_capability(dev)
    cuda_home = os.environ.get("CUDA_HOME")
    nvcc = os.environ.get("NVCC") or (cuda_home and os.path.join(cuda_home, "bin", "nvcc")) or shutil.which("nvcc")
    if not nvcc:
        raise SystemExit("bench_kernels: nvcc not found for the v1 variant build (set NVCC or CUDA_HOME)")
    out = Path(tempfile.mkdtemp()) / "v1var.cubin"
    cmd = [nvcc, "-cubin", "-O3", "--fmad=false", f"-arch=sm_{cc[0]}{cc[1]}", "-o", str(out), "v1.cu"]
    cmd += [f"-D{d}" for d in defines.split(",") if d]
    subprocess.run(cmd, cwd=CSRC, check=True, capture_output=True)
    with torch.cuda.device(dev):
        driver.ensure_context(dev.index)
        fn = driver.get_function(driver.module_load(out.read_bytes()), V1.entry)

    def chain(a, b):
        A, B = a.view(torch.uint8).contiguous(), b.view(torch.uint8).contiguous()
        M, L = A.shape
        N = B.shape[0]
        C = torch.empty((M, N), dtype=torch.float32, device=dev)
        args = [ctypes.c_void_p(A.data_ptr()), ctypes.c_void_p(B.data_ptr()), ctypes.c_void_p(C.data_ptr()),
                ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(L)]
        driver.launch(fn, (-(-N // V1.tile_n), -(-M // V1.tile_m), 1), (V1.threads, 1, 1), 0,
                      torch.cuda.current_stream(dev).cuda_stream, args)
        return C

    return chain, f"nvcc-cubin {' '.join(cmd[1:])}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernels", default="v1,v4", help="comma list of kernel names (v1, v4, v9, tc3) and of "
                    "--v1-variant names")
    ap.add_argument("--v1-variant", action="append", default=[], metavar="NAME=DEFINES",
                    help="bench-only v1 compile-time variant, e.g. dec0=MMAEMU_V1_DEC=0 or "
                    "dec1exp1=MMAEMU_V1_DEC=1:MMAEMU_V1_EXP=1 (defines separated by ':')")
    ap.add_argument("--shape", default="1b", help="1b | 7b | 40b | NxK")
    ap.add_argument("--M", default="2048", help="comma list of M")
    ap.add_argument("--data", default="gauss40")
    ap.add_argument("--fatbin-dir", help="sets MMAEMU_FATBIN_DIR (a local release-flag build)")
    ap.add_argument("--allow-nvrtc", action="store_true", help="time NVRTC-built kernels too (recorded as such)")
    ap.add_argument("--warmup", type=float, default=1.5, help="seconds of warm-up per kernel (>= 1.5)")
    ap.add_argument("--window", type=float, default=1.0, help="seconds per timed window (>= 1.0)")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--timer", choices=("events", "host-raw"), default="events")
    ap.add_argument("--hostclock-seconds", type=float, default=2.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", help="write the JSON report here")
    a = ap.parse_args(argv)
    if a.warmup < 1.5 or a.window < 1.0:
        raise SystemExit("bench_kernels: warm-up must be >= 1.5 s and windows >= 1.0 s")
    if a.fatbin_dir:
        os.environ["MMAEMU_FATBIN_DIR"] = str(Path(a.fatbin_dir).resolve())

    import hostclock

    hc = hostclock.gate(a.hostclock_seconds)
    print(json.dumps(dict(hostclock=hc)), flush=True)
    if a.timer == "host-raw" and not hc["ok"]:
        print("bench_kernels: host clock is off by more than the tolerance; refusing host timing (use CUDA events)",
              file=sys.stderr)
        return 3

    import torch

    from mmaemu.backends import ChainRequest, registry
    from mmaemu.backends.cuda import modules
    from mmaemu.backends.cuda.kernels import chain_fn_for
    from mmaemu.model.chain import chain_gemm

    dev = torch.device(a.device)
    props = torch.cuda.get_device_properties(dev)
    sms = props.multi_processor_count
    K, N = SHAPES[a.shape] if a.shape in SHAPES else tuple(int(x) for x in a.shape.split("x"))
    specs = {s.name: s for s in registry.discover()}
    req = ChainRequest(device=dev, fmt_a="e4m3", fmt_b="e4m3", preset="ada")
    variants = dict(v.split("=", 1) for v in a.v1_variant)
    names = [x.strip() for x in a.kernels.split(",") if x.strip()]
    fns, info = {}, {}
    for nm in names:
        if nm in variants:
            fns[nm], how = v1_variant_fn(variants[nm].replace(":", ","), dev)
            info[nm] = dict(source="nvcc-cubin (bench variant)", build=how)
            continue
        if nm not in specs:
            raise SystemExit(f"unknown kernel {nm!r}; discovered: {sorted(specs)}")
        fn = chain_fn_for(specs[nm], req)
        lm = modules.module_info(specs[nm].module, dev)
        if lm.source != "fatbin" and not a.allow_nvrtc:
            raise SystemExit(f"bench_kernels: {nm} was loaded via {lm.source} ({lm.fatbin_error}); performance "
                             "acceptance needs a fatbin (the package's release fatbins, or --fatbin-dir <dir>) or "
                             "pass --allow-nvrtc")
        fns[nm] = fn
        info[nm] = dict(source=lm.source, image_sha256=lm.sha256, fatbin_dir=str(modules.fatbin_dir()))
    report = dict(device=props.name, cc=list(torch.cuda.get_device_capability(dev)), sms=sms,
                  torch=torch.__version__, shape=a.shape, K=K, N=N, data=a.data, timer=a.timer,
                  warmup_s=a.warmup, window_s=a.window, rounds=a.rounds, hostclock=hc, kernels=info, results=[])
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def timed(f, n):
        if a.timer == "events":
            e0.record()
            for _ in range(n):
                f()
            e1.record()
            e1.synchronize()
            return e0.elapsed_time(e1) / 1e3
        torch.cuda.synchronize(dev)
        t0 = raw_now()
        for _ in range(n):
            f()
        torch.cuda.synchronize(dev)
        return raw_now() - t0

    for M in [int(x) for x in a.M.split(",")]:
        A = make_data(a.data, M, K, 5, dev)
        B = make_data(a.data, N, K, 6, dev)
        macs = M * N * K
        # exactness gate before timing
        exact = {}
        base = fns["v1"](A, B) if "v1" in fns else chain_fn_for(specs["v1"], req)(A, B)  # checked even if not timed
        rs, cs = slice(0, min(M, 64)), slice(0, min(N, 256))
        want = chain_gemm(A[rs].view(torch.uint8).cpu(), B[cs].view(torch.uint8).cpu(), fmt_a="e4m3",
                          fmt_b="e4m3", preset="ada").float()
        bad = int((base[rs, cs].cpu().view(torch.int32) != want.view(torch.int32)).sum())
        exact["v1"] = dict(vs="cpu reference", outputs=want.numel(), mismatches=bad)
        for nm, f in fns.items():
            if nm == "v1":
                continue
            out = f(A, B)
            bad = int((out.view(torch.int32) != base.view(torch.int32)).sum())
            exact[nm] = dict(vs="v1 (full output)", outputs=out.numel(), mismatches=bad)
        torch.cuda.synchronize(dev)
        print(json.dumps(dict(M=M, exactness=exact)), flush=True)
        if any(v["mismatches"] for v in exact.values()):
            print("bench_kernels: exactness gate FAILED; not timing", file=sys.stderr)
            report["results"].append(dict(M=M, exactness=exact, error="exactness gate failed"))
            break
        per = {}
        for nm, f in fns.items():  # warm-up >= a.warmup seconds of GPU time, and an estimate of one launch
            g = (lambda f=f: f(A, B))
            spent, n, t1 = 0.0, 1, 0.0
            while spent < a.warmup:
                t1 = timed(g, n)
                spent += t1
                n = min(n * 2, 1 << 16)
            per[nm] = max(t1 / max(1, n // 2), 1e-6)
        smi = SmiSampler(dev.index if dev.index is not None else 0).start()
        wins = {nm: [] for nm in fns}
        order = list(fns)
        for r in range(a.rounds):
            for nm in order[r % len(order):] + order[:r % len(order)]:
                g = (lambda f=fns[nm]: f(A, B))
                n = max(1, int(a.window / per[nm]) + 1)
                t0 = raw_now()
                dt = timed(g, n)
                while dt < a.window:  # the window must be >= a.window seconds of GPU time
                    n = int(n * a.window / max(dt, 1e-9) * 1.05) + 1
                    t0 = raw_now()
                    dt = timed(g, n)
                wins[nm].append(dict(n=n, s=dt, per_gemm_s=dt / n, tmac_s=macs * n / dt / 1e12, t0=t0, t1=raw_now()))
        smi.stop()
        report.setdefault("power_samples", {})[str(M)] = smi.power_samples()
        for nm, w in wins.items():
            tm = [x["tmac_s"] for x in w]
            clk = smi.between([(x["t0"], x["t1"]) for x in w])
            med = statistics.median(tm)
            row = dict(kernel=nm, M=M, N=N, K=K, data=a.data, source=info[nm]["source"], tmac_s_median=med,
                       tmac_s_min=min(tm), tmac_s_max=max(tm), windows=len(w),
                       window_s=[round(x["s"], 4) for x in w], launches=[x["n"] for x in w],
                       per_gemm_ms_median=statistics.median(x["per_gemm_s"] for x in w) * 1e3, smi=clk,
                       mac_per_clk_per_sm=(med * 1e12 / (clk["sm_mhz_mean"] * 1e6) / sms) if clk else None,
                       exactness=exact.get(nm))
            report["results"].append(row)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=1))
    return 1 if any("error" in r for r in report["results"]) else 0


if __name__ == "__main__":
    sys.exit(main())
