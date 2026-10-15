# Performance

Emulating FP8 tensor-core arithmetic exactly costs far more than running it natively. This page says how mmaemu
measures speed, what the kernels are expected to reach on the reference GPU, and where the measured release
numbers are recorded.

## Method

All GPU timings use **CUDA events**. Host timers are not used for kernel throughput: on the WSL2 reference host
`time.perf_counter` ran 6-10% fast against an external clock during the audit, which biased every throughput
computed from it. The kernel benchmark (`tools/bench_kernels.py` in the repository):

- checks the full output of the timed kernel bitwise against v1 or the reference before timing;
- warms up for at least 1.5 s, then times windows of at least 1.0 s (sustained load: the RTX 4060 reaches its
  power limit and lowers its clock, so short bursts overstate throughput by 5-10%);
- samples the SM clock and power with `nvidia-smi` and reports TMAC/s (10^12 multiply-accumulates per second) and
  MAC per clock per SM;
- refuses to run when the host clock check (`tools/hostclock.py`) reports a perf_counter drift above 0.2% and no
  CUDA events are in use;
- measures performance on locally built release-flag fatbins only. NVRTC-compiled kernels are checked for equal
  bits; their speed is recorded but not used for thresholds.

## Expected throughput on the RTX 4060 (Evo 2 1B projection, M = 2048, K = 1920, N = 5760)

The audit measured the prototype kernels with CUDA events under sustained load (TMAC/s). The v0.1 kernels are
ports of these prototypes, and v0.1 accepts them only above the thresholds below.

| Implementation | Audit measurement (prototype) | v0.1 acceptance threshold | Notes |
|---|---|---|---|
| native `torch._scaled_mm` (fast accumulation) | about 30-32 | (reference) | the hardware being emulated |
| v1 (CUDA cores, IEEE-exact) | 0.51-0.53; about 0.66 with the shared-memory decode table | >= 0.60 with the shared-memory table, else >= 0.48 | portable floor, sm_75 and newer |
| v4 (CUDA cores, IEEE-defined instructions only) | 1.30-1.38 | >= 1.25 | sm_80 and newer |
| v9 (bf16 tensor cores for exponents, gated) | 1.55-1.58 | >= 1.45 | relies on the "window" property; validated on sm_89 |
| tc3 (fp16 tensor-core "dummy accumulator", gated) | 2.94-3.02 | >= 2.8 | Ada/Ampere fp16 HMMA only (sm_80/86/89 binaries); validated on sm_89 |
| float64 reference, CPU (8 threads) | 116-181 MMAC/s | none | for validation and small problems |

The release measurements of the v0.1 kernels, at M in {16, 64, 256, 1024, 2048, 8192} for the Evo 2 1B, 7B and 40B
shapes with Gaussian, adversarial and real Evo 2 data, are committed under `docs/validation_logs/` (T6) before
the release; until then the table above shows audit measurements of the prototypes, not of the shipped binaries.

Small M runs well below these rates: at M = 16 the prototype v1 reached about a quarter of its M = 2048 rate,
because there is too little parallel work for 24 SMs.

## Selection

With `backend="auto"`, mmaemu uses the highest-priority kernel that loads and passes its gates on the device:
tc3 > v9 > v4 > v1 > reference. `python -m mmaemu info` shows the status of each, with the reason when one is
unavailable; `python -m mmaemu selftest --force` re-runs the gates.

## What this means end to end

For Evo 2 1B on the RTX 4060 the audit measured the emulated FP8 projections as the dominant cost of an emulated
forward pass with the prototype v1 kernel (about 95% of the extra time over native, paired measurement). With the
faster kernels the emulated forward pass is expected to take roughly 1.5-2x the native time instead of about 3.6x
(an estimate from the kernel rates, not a measurement). On the CPU the float64 reference is practical for tests
and single projections, not for whole models.
