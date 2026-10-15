# RTX 5070 Ti validation, 12–13 October 2026

The resumed development branch integrates WS0–WS6, restores complete hardware-mini
coverage and validates the Ada emulator on RTX 5070 Ti. The numerical reference
remains the recorded RTX 4060 platform. The new GPU's native FP8 behavior differs.

Hardware: RTX 5070 Ti, sm_120, 70 SMs, 16 GB; WSL; Windows host driver 610.88;
CUDA driver API 13.3; torch 2.7.1+cu128; cuBLASLt 12.8.3.14; NVRTC/nvcc 12.8.
The existing driver ran every CUDA test; no driver was installed or changed.

## Emulator checks

| Check | Result | Evidence |
|---|---|---|
| Final CPU suite | 3302 passed, 4 skipped, 448 deselected | [cpu.txt](cpu.txt) |
| T3 GPU suite | All per-file runs exit 0; v1/v4 IEEE 79 passed; runtime 13 passed; fast kernels 91 passed, 91 skipped | [t3.txt](t3.txt) |
| Full device independence | 71 passed, 67 skipped; all 21 full accumulator hashes match for each of v1/v4/v9 | [device-independence.txt](device-independence.txt) |
| Kernel selftest | v1/v4 smoke and v9 window pass; tc3 rejected for missing supported sm_120 binary | [selftest.json](selftest.json) |
| CPU mutations | 43 killed, 5 identity controls survive, 2 equivalent, no errors | [mutants-cpu.json](mutants-cpu.json) |
| GPU mutations | 7 killed, 1 identity control survives; NaN rerun is explicitly recorded | [mutants-gpu.json](mutants-gpu.json) |
| Wheel without nvcc | 673 comparisons, 7,493,365 outputs, zero numerical mismatches; Evo2 projection bf16 bits and accumulator hash match | [wheel.json](wheel.json) |
| Distribution | 60 checks, zero failures; strict twine, ruff, actionlint and docs checks pass | [distribution.txt](distribution.txt) |

The wheel is a development build with shipped CUDA sources and NVRTC fallback.
Its release-fatbin loader check is explicitly skipped: release binaries have not
been committed into the wheel. Development fatbins were built with the pinned
release flags and checked for all five modules, with no spills. Profile-device
tests skip because their native platform is the RTX 4060. Unsupported tc3 tests
skip; this report does not claim tc3 support on sm_120. v9 remains an opt-in
backend on this architecture; AUTO selects the validated v4 kernel.

The mutation harness preserves API validation and identity controls. GPU mutation
checks now run raw NVRTC kernels before registry smoke gates can disable them.
This prevents a broken kernel from appearing to pass through skipped tests.

## Native measurements

- FP8 instruction kits run with both sm_120 and sm_120a kind::f8f6f4 and record
  per-hypothesis scores: [c6.json](c6.json), [c6-kind.json](c6-kind.json).
- Full bf16 native/Ada comparisons record 9 projection shapes, nonunit scales,
  cuBLAS kernel names and actual split-K: 2,645,702 of 8,183,808 outputs differ.
  The probe exits 1 for this measured mismatch: [native-gemm.json](native-gemm.json).
- The cuBLAS instruction kit is recorded separately: [native-cublas.json](native-cublas.json).
- 850,320 heuristic queries are recorded at 0/1/4/32 MiB. All 135,360 Evo2 rows
  at each workspace have no split: [heuristics.json](heuristics.json). Candidate
  sm_120 data stays outside shipped profile tables.
- All 150 special-value CPU/GPU twins agree. Native differs in two mixed E5M2
  fp32 observations by one fp32 ulp; the full input/output bits are retained:
  [specials.json](specials.json). All 12 FP8 saturation/scale-result observations
  agree with the emulator: [fp8-specials.json](fp8-specials.json).
- Production-shape epilogue tie recording and replay: 64 passed:
  [ties-replay.txt](ties-replay.txt).
- Raw bf16 window and model analysis: [window-analysis.json](window-analysis.json).
  Design A rejects: 2392/16384 property failures, [design-a.json](design-a.json).
  Informational design-B software/native full-tile results: [design-b.json](design-b.json).

These are measured observations, not a new shipped native preset. Candidate-model
full-model loss simulation and additional architectures remain research work.

## Evo2 and performance

Evo2 1B runs on all six complete windows in native, emulate, compat and AUTO modes.
Before comparing with the Ada TE golden, every array's window, shape and dtype
and every token ID are checked. AUTO and explicit v9 emulation agree bitwise on
all 24 output arrays. Relative to the Ada golden, emulation mean absolute ACGT
dlogp is 0.0352–0.0466 and absolute sequence-score difference is 0.000004–0.001648.
Native mean dlogp is 1.9133–3.4576. Full-model logits differ across architectures
because the non-FP8 path is also device dependent: [evo2.json](evo2.json).

Sustained CUDA-event throughput, M=2048, Gaussian FP8 inputs, three windows:

| Projection (K, N) | v1 TMAC/s | v4 TMAC/s | v9 TMAC/s |
|---|---:|---:|---:|
| 1B (1920, 5760) | 1.938 | 3.990 | 4.819 |
| 7B (4096, 12288) | 1.971 | 4.314 | 5.155 |
| 40B (8192, 24576) | 1.975 | 4.382 | 5.110 |

Before timing, v1 matches a CPU reference slice and v4/v9 match v1 over the full
output. Host-clock checks fail on WSL, so all GPU timing uses CUDA events. Details:
[1B](performance-1b.json), [7B](performance-7b.json), [40B](performance-40b.json).

The new [runbook](../../runbooks/sm120_bringup.md) and runner support G0–G10,
single stages, resuming from a stage and forced failures. G0, G2, G8 and G9, forced
G4 failure and output-preservation controls are exercised separately. Artifacts
here were generated from the working tree after integration commit 1f9ced16;
the historical metadata retains that base commit and explicitly records dirty
source status where available. The complete local logs and compiled artifacts
are kept under build/resume-logs/ and build/fatbin/ in the development checkout.
The [source SHA-256 inventory](source-sha256.json) identifies the final Python
and CUDA sources reviewed in this development iteration. The final wheel has
the same SHA-256 as the wheel tested on the GPU.

This is GPU development validation, not a published release. The full release
matrix, complete T2 truth replay and release-fatbin packaging remain separate
release checks.
