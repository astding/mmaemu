# RTX 4060 (sm_89) on fa0fd90, 2026-10-16

Repeated after the profile rename (63b7395), on 2648bba, with every result equal once the old profile names are read
as the new ones: [../rtx-4060-sm89-renamed/README.md](../rtx-4060-sm89-renamed/README.md).

Verdict: every step of `final_4060.sh` exits 0 (`run.txt`): the preflight (18 of 18), T3 (every per-file run,
`t3/summary.txt`), the four test files of the hopper preset's kernels, the full accumulators of every GPU kernel, T3p
(6 passed), the three split-K envelope sweeps (rows equal to those of 2026-10-15), the GPU tier of T4, the bench, and
the replay of the H100 PCIe probe through the hopper preset's kernels h4 and h1. The forced h1 replay reports
`"PASS": false` by construction: h1 has no kernel for the 380 mixed-format calls (below).

- Hardware and software: the RTX 4060 host of
  [../../2026-10-14/rtx4060-host/README.md](../../2026-10-14/rtx4060-host/README.md). Preflight 18 of 18 PASS
  (`preflight.out`, `preflight.json`): driver 570.211.01, Python 3.12.2, torch 2.7.1+cu128, cuBLASLt 12.8.3.14, the
  32 packages of the reference lock, fatbins current and equal to the manifest's sha256. It ran that directory's
  `preflight_4060.py` with the expected commit set to fa0fd90.
- Code under test: fa0fd90 (a git checkout; `mmaemu/`, `tools/` and `tests/` unmodified before and after the run,
  `run.txt`). 63b7395, later, renamed the profiles in `mmaemu/`, `tests/` and `tools/`. Its fatbins are those of
  [../rc-fatbin/README.md](../rc-fatbin/README.md).
- Scripts: `final_4060.sh`, one step after another, every GPU step under the GPU lock, with the ground truth and the
  audit data present; `final_4060_hk.sh`, its step 2b alone (below). Paths: `~/` is the host's home directory.

The same script ran on 7612fd8 (the release-candidate freeze) earlier the same day. The commits between change
`validated_on` of h1 and h4 (sm_89 added), make `mmaemu.explain` run the planning checks of `scaled_mm`, and change
docstrings, help text and one test (a CPU test, which `-m gpu` deselects). Every count below equals that run, the
sweep outputs are byte-identical, every record of both probe replays is equal apart from its timings, the bench
medians differ by at most 0.01 TMAC/s, and `python -m mmaemu info` now lists h1 and h4 as "validated here".

## T3 (`t3/`)

The counts of 2026-10-15 ([../../2026-10-15/rtx-4060-sm89/README.md](../../2026-10-15/rtx-4060-sm89/README.md)) file
by file, plus the tests of the hopper preset's kernels added since:

- `test_kernels_ieee` 167 passed (79 then) and `test_cuda_runtime` 19 passed (13 then): the h1 and h4 tests;
- `test_h100_mini` (new in `tools/run_gpu_tests.sh`): 310 passed, its GPU cases (the 124 live H100 calls on the GPU
  through the reference backend and through `auto`, which runs h4, and the E4M3 x E4M3 ones through h1); `-m gpu`
  deselects its 125 CPU cases;
- `test_backend_selection` exits 5, nothing collected: it is a CPU-only test file (fake kernels, monkeypatched
  device) with no GPU-marked test; it runs in T0, and without the marker below.

Full accumulators (`t3/di_full_gpu_kernels.log`, `MMAEMU_DI_FULL=cuda:v1,cuda:v4,cuda:v9,cuda:tc3`): 84 passed, the
21 Evo 2 calls of each kernel bit for bit equal to the RTX 4060's recorded accumulators.

## The hopper preset's kernels h1 and h4 on this GPU (`hopper-kernels/`, `replay-h4/`, `replay-h1/`)

`validated_on` of h1 and h4 is {sm_89, sm_120}. sm_120 rests on the runs of
[../h1-rtx-5070-ti/README.md](../h1-rtx-5070-ti/README.md) and [../h4-rtx-5070-ti/README.md](../h4-rtx-5070-ti/README.md):
their four test files and the replay of the H100 probe. sm_89 rests on the same evidence on this GPU:

| Check | RTX 5070 Ti (sm_120) | RTX 4060 (sm_89), fa0fd90 |
|---|---|---|
| `tests/test_kernels_ieee.py` (no marker filter) | 167 passed | 167 passed |
| `tests/test_cuda_runtime.py` | 19 passed | 19 passed |
| `tests/test_h100_mini.py` | 435 passed | 435 passed |
| `tests/test_backend_selection.py` | 26 passed | 26 passed |
| probe replay, `--backend auto` (h4) | PASS | PASS (`replay-h4/mmaemu_replay.json`) |

- The four test files ran on the package fatbins (`MMAEMU_TEST_FATBIN_DIR`, as `tools/run_gpu_tests.sh` sets it),
  each in its own locked run (`hopper-kernels/summary.txt`). The first run of this step, inside `final_4060.sh`
  before that variable was added to the script, did not set it: the fatbin tests of `test_kernels_ieee` and
  `test_cuda_runtime` skipped (1 passed, 166 skipped; 19 skipped). `final_4060_hk.sh` ran the step again with it
  (`run.txt`, 18:47 to 18:50), and `final_4060.sh` as committed sets it.
- `--backend auto` (h4 for E4M3 x E4M3, the reference backend for the other pairs): the 1,043 E4M3 x E4M3 live calls
  ran on h4 and the 380 mixed-format ones on the reference backend; all 1,423 equal the H100 (1,049 split, 374
  unsplit), and the 27 calls the H100 rejected raise `ReferenceRejectsError`. The 168 Evo 2 outputs ran on h4: 168 of
  168 equal. Every live record equals that of the RTX 5070 Ti's replay of the same tree
  ([../rtx-5070-ti-replays/README.md](../rtx-5070-ti-replays/README.md)) apart from its timings, and every verdict
  that of the RTX 5070 Ti's replay of ebd18e1
  ([../../2026-10-15/h100-probe/results/analysis/mmaemu_replay_cuda_h4.json](../../2026-10-15/h100-probe/results/analysis/mmaemu_replay_cuda_h4.json)).
  The `splitk="auto"` decisions: 1,353 agree with the H100 and 70 raise `SplitKUnknownError`, against 1,341 and 82
  at ebd18e1; the 12 that changed are now inferred S = 1 by the H100 envelope (`source="envelope"`,
  `verified=True`), which ebd18e1 did not have, and agree with the logged decision.
- `--backend cuda:h1` (h1 forced for every call): the 1,043 E4M3 x E4M3 live calls ran on h1 and are equal (746 split,
  297 unsplit); the 168 Evo 2 outputs ran on h1 and are equal. On the 1,070 E4M3 x E4M3 jobs (27 of them rejected as on
  the H100) and the 168 Evo 2 outputs every verdict equals that of the RTX 5070 Ti's h1 replay
  ([../../2026-10-15/h100-probe/results/analysis/mmaemu_replay_cuda.json](../../2026-10-15/h100-probe/results/analysis/mmaemu_replay_cuda.json),
  3d7dc66). The 380 mixed-format calls raise `BackendUnavailableError` (h1 implements E4M3 x E4M3 only), so the
  tool's overall verdict is false; the same 380 calls are equal on the reference backend in the h4 replay. (On the
  RTX 5070 Ti h1's replay ran through `--backend auto` before h4 existed, when h1 was the preset's only kernel.)
- Bench exactness (below): h1 equals the CPU reference of the hopper preset on its first 64 rows x 256 columns and
  h4 equals h1 on every output, at all three shapes.
- `python -m mmaemu info` (`mmaemu_info.txt`, from the preflight): `cuda:h4 (preset hopper): available via fatbin,
  validated here`, and the same for h1.

## T3p (`t3p_profile_device.log`, 6 passed)

The six tests of 2026-10-15: live `torch._scaled_mm` parity with mmaemu's reference backend, `out=` and operand
layouts, the profile match, 500 table rows re-queried from cuBLASLt, the live query step, and 11,760 live queries
above the bound where mmaemu infers S = 1.

## Split-K envelope sweeps (`sweeps/`)

The three scripts of [../../2026-10-14/splitk-envelope/](../../2026-10-14/splitk-envelope/README.md), unchanged,
importing fa0fd90. Their per-query rows (not kept) equal those of 2026-10-15 row for row: 306,859, 663,936 and
666,624 rows; the row files are byte-identical to those of the run on 7612fd8.

## T4, GPU tier

[../t4/mutants-gpu-rtx4060.json](../t4/mutants-gpu-rtx4060.json): PASS, 7 kernel mutants killed, the identity control
survives.

## Bench (`bench-*-gauss40.json`, `.log`)

`tools/bench_kernels.py --kernels v1,v4,v9,tc3,h1,h4 --shape 1b|7b|40b --M 2048 --data gauss40`, every kernel from the
release fatbins (`"source": "fatbin"`). TMAC/s, the median of 5 windows:

| Kernel | 1B (K 1920, N 5760) | 7B (K 4096, N 12288) | 40B (K 8192, N 24576) | T6 (docs/performance.md) | Exactness (0 mismatches) |
|---|---:|---:|---:|---|---|
| tc3 | 2.95 | 2.95 | 2.94 | 2.94 / 2.93 / 2.94 | every output vs v1 |
| v9 | 1.54 | 1.55 | 1.55 | 1.53 / 1.54 / 1.54 | every output vs v1 |
| h4 | 1.35 | 1.41 | 1.41 | (not in T6) | every output vs h1 |
| v4 | 1.35 | 1.37 | 1.37 | 1.34 / 1.36 / 1.37 | every output vs v1 |
| h1 | 0.69 | 0.69 | 0.69 | (not in T6) | 16,384 outputs vs the CPU reference (hopper) |
| v1 | 0.63 | 0.63 | 0.63 | 0.63 / 0.63 / 0.63 | 16,384 outputs vs the CPU reference (ada) |

v1, v4, v9 and tc3 are within 1% of T6.
