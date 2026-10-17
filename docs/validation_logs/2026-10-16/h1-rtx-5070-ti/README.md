# h1 on the RTX 5070 Ti, 2026-10-16

The GPU tests and the sustained throughput of the kernel h1 (the `hopper` preset;
[its build](../../2026-10-15/h1-fatbin/README.md)), on an RTX 5070 Ti (sm_120, 70 SMs, torch 2.7.1+cu128, WSL2).
Both ran on committed trees with no changes under `mmaemu/`, and every kernel came from the package fatbins (h1:
sha256 `261a7605…`, `mmaemu/backends/cuda/bin/h1.fatbin`).

## GPU tests

[gpu-tests/run.sh](gpu-tests/run.sh) ran four test files at 3d7dc66 (no changes under `mmaemu/` or `tests/`;
[summary.txt](gpu-tests/summary.txt)):

| File | Result | What it covers for h1 |
|---|---|---|
| [tests/test_kernels_ieee.py](gpu-tests/test_kernels_ieee.log) | 124 passed | the eight conformance generators at K = 16, 48 and 1,920 (K = 16 and 48 end in a half block), fatbin and NVRTC builds, raw NaN codes, any byte value, slow accumulation, contiguous split-K (S = 2 to 8), K not a multiple of 16; each kernel bit for bit against the CPU reference of its own preset |
| [tests/test_cuda_runtime.py](gpu-tests/test_cuda_runtime.log) | 16 passed | storage offsets 1 to 15 (misaligned operands), a non-default stream, mixed devices and a request of another preset (`ValueError`), a second CUDA context |
| [tests/test_h100_mini.py](gpu-tests/test_h100_mini.log) | 373 passed | the 124 CI jobs of the H100 probe on the CPU (124) and on the GPU, on the reference backend and on `backend="auto"` (248; auto runs every E4M3 x E4M3 job on `cuda:h1`), and the fixture's coverage check (1) |
| [tests/test_backend_selection.py](gpu-tests/test_backend_selection.log) | 25 passed | each kernel serves only its own preset |

## Throughput

[bench-run.sh](bench-run.sh) ran, at 9a5d836 (no changes under `mmaemu/` or `tools/`;
[bench-summary.txt](bench-summary.txt)), for each shape:

    python tools/bench_kernels.py --kernels v1,v4,h1 --shape <1b|7b|40b> --M 2048 --data gauss40 --out <json>

The method is the one in [docs/performance.md](../../../performance.md): CUDA events, a 1.5 s warm-up, then five
1.0 s windows. The host clock check fails on WSL2 (perf_counter / CLOCK_MONOTONIC_RAW = 1.058), so no host timer is
used. Before timing, each kernel passed an exactness gate:

- v1 and h1 were each bit for bit equal to the CPU reference of their own preset on a 16,384-output slice;
- v4 was bit for bit equal to v1 over the full output.

The SM clock stayed at 2,707 to 2,722 MHz with no throttle reasons.

| Shape (K x N) | h1 | v1 | v4 (TMAC/s, median) |
|---|---:|---:|---:|
| Evo 2 1B (1920 x 5760) | 2.026 | 1.948 | 4.006 |
| Evo 2 7B (4096 x 12288) | 2.050 | 1.979 | 4.332 |
| Evo 2 40B (8192 x 24576) | 2.045 | 1.981 | 4.392 |

There was no v4-style kernel for the hopper preset yet (h4 followed the same day:
[h4-rtx-5070-ti](../h4-rtx-5070-ti/README.md)). A first run on the uncommitted tree just before 50ff4a3 (the
same fatbin images) gave the same numbers to within 0.003 TMAC/s; it was replaced by this run.

- [bench-1b-gauss40.json](bench-1b-gauss40.json), [bench-7b-gauss40.json](bench-7b-gauss40.json),
    [bench-40b-gauss40.json](bench-40b-gauss40.json): the reports (per-window times, launches, nvidia-smi samples,
    the exactness gate).
- [bench-1b-gauss40.log](bench-1b-gauss40.log), [bench-7b-gauss40.log](bench-7b-gauss40.log),
    [bench-40b-gauss40.log](bench-40b-gauss40.log): the console output.
