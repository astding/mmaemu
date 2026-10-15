# RTX 4060 (sm_89) on the reviewed tree, 2026-10-15

Verdict: every step exits 0 (`run.txt`): the preflight (18 of 18), T3 (every per-file run, `t3/summary.txt`), T3p
(6 passed), the three split-K envelope sweeps (their rows equal those of 2026-10-14) and a bench smoke.

- Hardware and software: the RTX 4060 host of
  [../../2026-10-14/rtx4060-host/README.md](../../2026-10-14/rtx4060-host/README.md). Preflight 18 of 18 PASS
  (`preflight.out`, `preflight.json`): driver 570.211.01, Python 3.12.2, torch 2.7.1+cu128, cuBLASLt 12.8.3.14,
  the 32 packages of the reference lock, fatbins equal to the manifest's sha256. It ran that directory's
  `preflight_4060.py` with the expected commit set to 49f1ecf.
- Code under test: the tree of 49f1ecf (`git archive`), the fixes of the review of the release candidate
  (2026-10-14). Kernel sources and fatbins are those of the freeze (`git diff e173ecc 49f1ecf` touches no `.cu`,
  `.cuh` or `mmaemu/backends/cuda/bin/` file); the Python under `mmaemu/` changed in 8 files since da9edd4.
- Script: `rerun_4060.sh`. Every GPU step ran under the GPU lock, with the ground truth and the audit data present.

## T3 (`t3/`)

The same counts as on da9edd4 ([../../2026-10-14/t3-rtx-4060-sm89/README.md](../../2026-10-14/t3-rtx-4060-sm89/README.md)),
file by file, except `test_live_cublas`: 2 passed, against 1, because 49f1ecf adds
`test_live_out_and_operand_layouts` (it is also a T3p test, below). Its 8 warnings are the warning mmaemu gives for an
`out=` without unit stride, which the test passes on purpose.

## T3p (`t3p_profile_device.log`, 6 passed)

The four tests of 2026-10-14 and two that 49f1ecf adds:

- `test_live_parity`: 278 live `torch._scaled_mm` calls equal to `mmaemu.scaled_mm` with its reference backend (on
  the same GPU), bit for bit.
- `test_live_out_and_operand_layouts` (new): 112 live calls equal to mmaemu's reference backend, bit for bit, with
  column-major and padded `out=`, an `out=` without unit stride (torch and mmaemu leave it unchanged), a mat1 that
  torch copies (`expand`, step 2), and zero sizes with any out dtype or any float8 input dtype.
- `test_matches_the_reference_gpu_live`: the three RTX 4060 profiles match this device.
- `test_live_requery_of_500_random_rows`: 500 rows of the shipped split-K table re-queried from cuBLASLt, 0 differ.
- `test_live_query_step_resolves_unknown_shapes`: a shape not in the table is resolved by a live query.
- `test_live_heuristic_never_splits_where_the_envelope_infers_s1` (new): 11,760 live heuristic queries above the
  bound where mmaemu infers S = 1 (M*N 2 to 4 times the largest split M*N of each of the 56 settings, K from 1,024 to
  4,194,304); every one is S = 1.

## Split-K envelope sweeps (`sweeps/`)

The three scripts of [../../2026-10-14/splitk-envelope/](../../2026-10-14/splitk-envelope/README.md), unchanged,
importing the 49f1ecf tree. Their per-query rows (not kept) equal those of 2026-10-14 row for row: 306,859 rows from
`sweep_env.py` (181,440 above the envelope, 125,419 below the smallest split K), 663,936 and 666,624. Recomputed from
them, the counts of that directory's `summary.json` show 0 differences. Run times: 19.6 s, 40.7 s and 40.1 s.

## Bench smoke (`bench-smoke-1b-*.json`, `.log`)

`tools/bench_kernels.py --kernels v1,v4,v9,tc3 --shape 1b --M 2048`, with Gaussian (`gauss40`) and adversarial data.
Every kernel ran from the release fatbins, with the image sha256 of the T6 run.

| Kernel | gauss40 TMAC/s | adversarial TMAC/s | T6 at M = 2048 (docs/performance.md) | Exactness |
|---|---:|---:|---:|---|
| tc3 | 2.94 | 2.96 | 2.941 | 0 of 11,796,480 outputs differ from v1 |
| v9 | 1.54 | 1.54 | 1.535 | 0 of 11,796,480 differ from v1 |
| v4 | 1.35 | 1.34 | 1.340 | 0 of 11,796,480 differ from v1 |
| v1 | 0.63 | 0.63 | 0.630 | 0 of 16,384 (its first 64 rows x 256 columns) differ from the CPU reference |

## Not repeated

T6 (performance), the full accumulators of every kernel (`MMAEMU_DI_FULL`) and the GPU tier of T4 ran on da9edd4
(2026-10-14). Their kernels and fatbins are the ones tested here. The Python changes of 49f1ecf are in the API, the
split-K resolution, the device lookup of the backend registry, the CLI, the preset checks and messages; T3 and T3p
above ran on them.
