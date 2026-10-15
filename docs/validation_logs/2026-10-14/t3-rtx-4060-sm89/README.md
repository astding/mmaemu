# T3 and T3p on the RTX 4060 (sm_89), 2026-10-14

Verdict: every per-file run exits 0 (`summary.txt`), and the four T3p tests pass (`t3p_profile_device.log`). This
is the profile device, so every kernel runs here: tc3, v9, v4, v1 and the reference.

- Hardware and software: the RTX 4060 host described in `../rtx4060-host/README.md` (preflight 18 of 18 PASS:
  torch 2.7.1+cu128, cuBLASLt 12.8.3.14, the 32 packages of the recording venv, driver 570.211.01).
- Code under test: the tree of da9edd4 (`git archive`), with the frozen release fatbins. Under `mmaemu/` it is
  identical to 76fd4ea (`git diff 76fd4ea da9edd4 -- mmaemu/` is empty); da9edd4 changes two tests and four docs
  (see the first run below).
- T3 command: `MMAEMU_TRUTH_DIR=<truth> MMAEMU_AUDIT_DIR=<audit> PYTHON=<venv>/bin/python tools/run_gpu_tests.sh
  --out <dir>`, under the GPU lock. The ground truth and the audit data (`vperf-p3-tensorcore-exact-1/data/ref2`,
  `p2-int-mixed/data/cases.pt`) were present.
- T3p command: `python -m pytest -q -m profile_device tests/`.

| File | Result | Skips |
|---|---|---|
| test_kernels_ieee | 79 passed: v1 and v4 from the fatbin and from NVRTC against the reference, and NVRTC equal to the fatbin | none |
| test_cuda_runtime | 13 passed, including the fallback ladder fatbin -> NVRTC -> reference | none |
| test_fast_kernels | 182 passed: v9 and tc3 against the reference, including the audit's 77 verify2 cases and 22 p2 cases | none |
| test_gates_hmma | 4 passed: tc3's gates pass on this device; forced gate failures fall back with one warning; an unvalidated capability needs the opt-in; the verdict cache survives a second process and corruption | none |
| test_live_cublas | 1 passed: live `torch._scaled_mm` against mmaemu, bit for bit | none |
| test_hw_mini | 5 passed: `reference`, `cuda:v1`, `cuda:v4`, `cuda:v9` and `cuda:tc3` on the hardware-mini recordings | `cuda` is not a status entry |
| test_device_independence | 31 passed: rows 0..255 of all 21 Evo 2 1B accumulators on every backend, the full-accumulator sha256 of all 21 calls for the AUTO kernel (tc3), and the emulator group | 23: `cuda` is not a status entry; 84: the full accumulators of the other backends run only with `MMAEMU_DI_FULL` |
| test_scaffold, test_backend_reference, test_epilogue | 2, 3 and 5 passed | none |

T3p, on the profile device only (`t3p_profile_device.log`, 4 passed):

- `test_live_parity`: live `torch._scaled_mm` calls on this GPU and `mmaemu.scaled_mm` (its reference backend, on
  the same GPU) give the same bits, over
  fast and slow accumulation, every output dtype, the three FP8 format pairs, three scale sets, unsplit shapes,
  shapes whose split-K decision was recorded on this GPU, NaN/Inf inputs and empty dimensions.
- `test_matches_the_reference_gpu_live`: the three RTX 4060 profiles match this device.
- `test_live_requery_of_500_random_rows`: 500 rows of the shipped split-K table (100 of them split) re-queried from
  cuBLASLt on this host give the stored decisions.
- `test_live_query_step_resolves_unknown_shapes`: a shape inside the envelope but not in the table is resolved by a
  live query.

## Full accumulators of every kernel (`di_full_gpu_kernels.log`)

After T6, `../rtx4060-host/di_full.sh` ran `test_evo2_acc_sha256` with `MMAEMU_DI_FULL=cuda:v1,cuda:v4,cuda:v9,cuda:tc3`
(as `tools/run_gpu_tests.sh --di-full` does, without the slow reference backend): 84 passed. The sha256 of the full
[2048, 5760] fp32 accumulator equals the hardware's for all 21 Evo 2 1B projections and each of the four kernels.
The 42 skips are the reference backend (not requested) and the `cuda` alias.

## T3p and the driver

The live tests are the check that this host's driver (570.211.01, Linux) gives the arithmetic and the split-K
decisions recorded under 572.16 on WSL2.

## First run (`run1-76fd4ea/`)

The first T3 run used the tree of 76fd4ea and failed two tests:

- `test_cuda_runtime.py::test_fallback_ladder` expected `cuda:v4` as the kernel with nothing forced. On sm_89 AUTO
  selects tc3, so the test's expectation held only on devices where the gated kernels are skipped (sm_120, sm_90).
- `test_gates_hmma.py::test_forced_gate_failure_falls_back_once` expected two warnings when both tc3's and v9's
  gates are forced to fail. The registry reports the failures of one selection in one warning, which names both
  gates and the kernel used instead (`cuda:v4`), as the log shows.

Both expectations came from the prototype and had not run on an sm_89 device since the gated kernels were
integrated. `ladder_probe.py` (in `../rtx4060-host/`) measured the actual ladder on this device; mmaemu's behaviour
was the documented one, so da9edd4 changes the tests and states the merged warning in docs/backends.md (rule 5),
docs/kernels_fast.md and docs/limitations.md. No file under `mmaemu/` changed. The complete T3 was then run again on
da9edd4 (the files in this directory).

Two other differences between the runs:

- `MMAEMU_AUDIT_DIR` was not set in the first run, so test_fast_kernels gave 26 passed and 4 skipped there; with the
  audit data it gives 182 passed. The count changed with the environment, not with the fix.
- The first run was moved to `run1-76fd4ea/` after it finished, so the last line of its `summary.txt` still names
  the directory of the second run.
