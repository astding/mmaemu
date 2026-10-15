"""Native FP8 GEMM throughput (torch._scaled_mm, E4M3 x E4M3 -> bf16, fast accumulation) with the method of
tools/bench_kernels.py: CUDA events, warm-up >= 1.5 s, windows >= 1.0 s, median over windows; nvidia-smi samples the
SM clock, power and clock-event reasons during the windows. The baseline that mmaemu's kernels are compared with.

usage: python native_fp8_bench.py --out native-fp8.json
"""
import argparse
import json
import statistics
import subprocess
import threading
import time

import torch

SHAPES = {"1b": (1920, 5760), "7b": (4096, 12288), "40b": (8192, 24576)}


def smi_sampler(stop, out):
    q = "clocks.sm,power.draw,clocks_event_reasons.active"
    while not stop.is_set():
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True,
                           text=True)
        if r.returncode == 0:
            out.append(r.stdout.strip())
        time.sleep(0.2)


def bench(M, K, N, warmup=1.5, window=1.0, rounds=5):
    g = torch.Generator(device="cuda").manual_seed(0)
    a = (torch.randn(M, K, device="cuda", generator=g) * 40).clamp(-448, 448).to(torch.float8_e4m3fn)
    b = (torch.randn(N, K, device="cuda", generator=g) * 40).clamp(-448, 448).to(torch.float8_e4m3fn)
    one = torch.ones((), device="cuda")

    def run(n):
        for _ in range(n):
            torch._scaled_mm(a, b.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16, use_fast_accum=True)

    run(3)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); run(10); e.record(); e.synchronize()
    per = s.elapsed_time(e) / 10 / 1e3
    n = max(1, int(window / per))
    t_end = time.monotonic() + warmup
    while time.monotonic() < t_end:
        run(n)
    torch.cuda.synchronize()
    samples, stop = [], threading.Event()
    th = threading.Thread(target=smi_sampler, args=(stop, samples), daemon=True)
    th.start()
    tmacs = []
    for _ in range(rounds):
        s.record(); run(n); e.record(); e.synchronize()
        tmacs.append(M * N * K * n / (s.elapsed_time(e) / 1e3) / 1e12)
    stop.set(); th.join()
    clocks = [float(x.split(",")[0]) for x in samples if x]
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    med = statistics.median(tmacs)
    return dict(M=M, K=K, N=N, launches_per_window=n, tmac_s_median=med, tmac_s_min=min(tmacs), tmac_s_max=max(tmacs),
                sm_clock_mhz_mean=statistics.mean(clocks) if clocks else None,
                mac_per_clk_per_sm=med * 1e12 / (statistics.mean(clocks) * 1e6) / sms if clocks else None,
                smi_samples=samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--M", default="16,64,256,1024,2048,8192")
    a = ap.parse_args()
    p = torch.cuda.get_device_properties(0)
    res = dict(device=p.name, sms=p.multi_processor_count, torch=torch.__version__, cuda=torch.version.cuda,
               op="torch._scaled_mm e4m3 x e4m3 -> bf16, use_fast_accum=True, tensor-wise scales 1",
               timer="CUDA events", results=[])
    for name, (K, N) in SHAPES.items():
        for M in (int(x) for x in a.M.split(",")):
            r = bench(M, K, N)
            r["shape"] = name
            res["results"].append(r)
            print(f"{name} M={M}: {r['tmac_s_median']:.2f} TMAC/s, {r['mac_per_clk_per_sm'] or 0:.0f} MAC/clk/SM")
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
