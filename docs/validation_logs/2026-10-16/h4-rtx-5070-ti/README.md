# h4 on the RTX 5070 Ti, 2026-10-16

The GPU tests and the sustained throughput of the kernel h4 (the `hopper` preset's fast kernel: v4's design with
32-term blocks; [its build](../h4-fatbin/README.md)), on an RTX 5070 Ti (sm_120, 70 SMs, torch 2.7.1+cu128, WSL2).
Both ran at ebd18e1 with no changes under `mmaemu/`, `tests/` or `tools/`, and every kernel came from the package
fatbins (h4: sha256 `7e1f96c3…`, `mmaemu/backends/cuda/bin/h4.fatbin`).

## GPU tests

[gpu-tests/run.sh](gpu-tests/run.sh) ran four test files ([summary.txt](gpu-tests/summary.txt)):

| File | Result | What it covers for h4 |
|---|---|---|
| [tests/test_kernels_ieee.py](gpu-tests/test_kernels_ieee.log) | 167 passed (43 of them h4) | the eight conformance generators at K = 16, 48 and 1,920 (K = 16 and 48 end in a half block), fatbin and NVRTC builds, NaN codes through the plan, any byte value, slow accumulation, contiguous split-K (S = 2 to 8), K not a multiple of 16; each kernel bit for bit against the CPU reference of its own preset |
| [tests/test_cuda_runtime.py](gpu-tests/test_cuda_runtime.log) | 19 passed | storage offsets 1 to 15 (misaligned operands), a non-default stream, mixed devices and a request of another preset (`ValueError`) |
| [tests/test_h100_mini.py](gpu-tests/test_h100_mini.log) | 435 passed | the 124 CI jobs of the H100 probe on the CPU (124) and on the GPU: on the reference backend (124), on `backend="auto"` (124; auto runs every E4M3 x E4M3 job on `cuda:h4`), and on `backend="cuda:h1"` (the 62 E4M3 x E4M3 jobs); the fixture's coverage check (1) |
| [tests/test_backend_selection.py](gpu-tests/test_backend_selection.log) | 26 passed | within the `hopper` preset h4 before h1, h1 below sm_80 or when h4 fails to load |

The raw h4 kernel, like v4, requires finite codes; the plan replaces NaN codes before the kernel and patches the
result, and the NaN tests check that path.

## Replay of the H100 probe

The probe's 1,450 live calls and 21 Evo 2 1B projections were replayed through `tools/replay_h100_log.py
--backend auto` at the same commit, on the same GPU
([mmaemu_replay_cuda_h4.json](../../2026-10-15/h100-probe/results/analysis/mmaemu_replay_cuda_h4.json)): PASS,
with the same verdict as the reference-backend and h1 replays on every live job and Evo 2 output. h4 ran all 1,043
E4M3 x E4M3 live jobs that ran and all 168 Evo 2 outputs; the 380 mixed-format jobs ran on the reference backend.

## Throughput

[bench-run.sh](bench-run.sh) ran, for each shape ([bench-summary.txt](bench-summary.txt)):

    python tools/bench_kernels.py --kernels v1,v4,h1,h4 --shape <1b|7b|40b> --M 2048 --data gauss40 --out <json>

The method is the one in [docs/performance.md](../../../performance.md): CUDA events, a 1.5 s warm-up, then five
1.0 s windows. The host clock check passed this time (perf_counter / CLOCK_MONOTONIC_RAW = 0.9996 to 1.00003); the
timing uses CUDA events either way. Before timing, each kernel passed an exactness gate:

- v1 and h1 were each bit for bit equal to the CPU reference of their own preset on a 16,384-output slice;
- v4 was bit for bit equal to v1, and h4 to h1, over the full output (11,796,480, 25,165,824 and 50,331,648
    outputs).

The SM clock stayed at 2,707 to 2,722 MHz with no throttle reasons.

| Shape (K x N) | h4 | h1 | v4 | v1 (TMAC/s, median) |
|---|---:|---:|---:|---:|
| Evo 2 1B (1920 x 5760) | 4.133 | 2.026 | 4.006 | 1.947 |
| Evo 2 7B (4096 x 12288) | 4.635 | 2.047 | 4.333 | 1.979 |
| Evo 2 40B (8192 x 24576) | 4.619 | 2.045 | 4.392 | 1.981 |

h4 is 2.0 to 2.3 times as fast as h1 and 3 to 7% faster than v4. h1, v1 and v4 are within 0.003 TMAC/s of their
[earlier run](../h1-rtx-5070-ti/README.md).

- [bench-1b-gauss40.json](bench-1b-gauss40.json), [bench-7b-gauss40.json](bench-7b-gauss40.json),
    [bench-40b-gauss40.json](bench-40b-gauss40.json): the reports (per-window times, launches, nvidia-smi samples,
    the exactness gate).
- [bench-1b-gauss40.log](bench-1b-gauss40.log), [bench-7b-gauss40.log](bench-7b-gauss40.log),
    [bench-40b-gauss40.log](bench-40b-gauss40.log): the console output.
