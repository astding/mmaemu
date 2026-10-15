# T6 on the RTX 4060 (sm_89), 2026-10-14

Verdict: PASS. At M = 2048 every kernel is above its v0.1 acceptance threshold on the Evo 2 1B, 7B and 40B shapes,
with Gaussian and with adversarial data. Before timing, every kernel's output was checked bitwise at every M (v1
against the CPU reference on a 64x256 block, the other kernels against v1 on the full output): 0 mismatches.

- Host and software: `../rtx4060-host/README.md` (preflight 18 of 18 PASS).
- Code: the tree of da9edd4 with the frozen release fatbins, loaded as users get them: every report's `kernels`
  entry has source `fatbin` and the image sha256 of `mmaemu/backends/cuda/bin/manifest.json`.
- Kernel command, under the GPU lock, for `d` in gauss40 and adversarial and `s` in 1b, 7b and 40b:

      python tools/bench_kernels.py --kernels v1,v4,v9,tc3 --shape $s --M 16,64,256,1024,2048,8192 --data $d \
          --out performance-$s-$d.json

- Native baseline: `native_fp8_bench.py` (this directory) times `torch._scaled_mm` (E4M3 x E4M3 to bf16, fast
  accumulation, unit scales) on the same shapes and M with the same method: CUDA events, at least 1.5 s of warm-up,
  five windows of at least 1.0 s, the median. It writes `native-fp8.json`.
- Host clock: perf_counter / CLOCK_MONOTONIC_RAW was 0.99995-1.00001 in every report (tolerance 0.2%; this native
  Linux host does not show the WSL2 drift). All timings are CUDA events.
- Clock and power: on this host `nvidia-smi --query-gpu=power.draw` reads `[N/A]`, so the per-window
  `power_w_mean` is null. Each report's `power_samples` holds the driver's power summary (`nvidia-smi -q -d POWER`)
  read after each M: averages of 109-115 W at the 115 W limit. The clock-event reason 0x4 (SW power cap) appears in
  every measurement, and the mean SM clock at M >= 256 was 2.36-2.58 GHz. MAC/clk/SM divides the clock out.
  `native-fp8.json` records clocks but no power.

## Results

At M = 2048 (TMAC/s, MAC per clock per SM in parentheses):

| Kernel | 1B, gauss40 | 7B, gauss40 | 40B, gauss40 | 1B, adversarial | 7B, adversarial | 40B, adversarial | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| native `torch._scaled_mm` | 29.8 (496) | 30.4 (505) | 30.5 (508) | | | | |
| tc3 | 2.94 (49.0) | 2.93 (49.7) | 2.94 (50.1) | 2.93 | 2.94 | 2.94 | 2.80 |
| v9 | 1.53 (26.7) | 1.54 (27.2) | 1.54 (27.1) | 1.53 | 1.54 | 1.54 | 1.45 |
| v4 | 1.34 (22.3) | 1.36 (22.8) | 1.37 (23.2) | 1.34 | 1.36 | 1.37 | 1.25 |
| v1 | 0.63 (10.4) | 0.63 (10.4) | 0.63 (10.5) | 0.63 | 0.62 | 0.62 | 0.60 |

Adversarial over Gaussian, per kernel, shape and M: 0.990-1.006.

Against M, Evo 2 1B shape, gauss40 (TMAC/s):

| M | tc3 | v9 | v4 | v1 | native |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.378 | 0.384 | 0.172 | 0.149 | 6.17 |
| 64 | 1.475 | 1.429 | 0.666 | 0.586 | 24.35 |
| 256 | 2.831 | 1.524 | 1.273 | 0.626 | 28.20 |
| 1024 | 2.936 | 1.532 | 1.341 | 0.630 | 29.82 |
| 2048 | 2.941 | 1.535 | 1.340 | 0.630 | 29.84 |
| 8192 | 2.928 | 1.535 | 1.343 | 0.630 | 29.92 |

The 7B and 40B rows against M are in the reports; from M = 256 on every kernel is within 0.5% of its M = 2048 value
there. Native FP8 at M = 8192 on the 40B shape reads 32.1 TMAC/s at an unchanged 509 MAC/clk/SM: the clock
was higher in those windows.

## Files

- `performance-<shape>-<data>.json` and `.log`: the reports of `tools/bench_kernels.py`, one row per (kernel, M),
  with the windows, launches, `nvidia-smi` samples, MAC/clk/SM and the exactness check.
- `native_fp8_bench.py`, `native-fp8.json`, `native-fp8.log`: the native baseline.
- `first-pass-m2048/`: an earlier run of the same command at M = 2048 only (gauss40), on the same code. It ran on a
  cooler GPU and is 0.3-0.8% faster than the sweep at M = 2048.
