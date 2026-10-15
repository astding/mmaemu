# T3 on an H100 PCIe (sm_90), 2026-10-14

Verdict: every per-file run exits 0 (`summary.txt`). Tests that need the ground truth were skipped, so this run does
not include the device-independence check.

- Hardware: NVIDIA H100 PCIe (sm_90, 114 SMs), rented; torch 2.7.1+cu128, CUDA 12.8 (`fixed-fp8.json`, `fixed-fp8.log`).
- Software: this run did not record the driver, Python or cuBLASLt; the Evo 2 baseline on the rented H100 the same
  day recorded driver 570.148.08, Python 3.12.15 and cuBLASLt 12.8.3.14 (`../evo2-h100-sm90/windows.json`).
- Code under test: the tree of 69cd231 (`git archive`), with the frozen release fatbins (rc1/v0.1.0, which include
  sm_90 SASS).
- Command: `PYTHON=<venv>/bin/python tools/run_gpu_tests.sh --out <dir>`. `MMAEMU_TRUTH_DIR` and `MMAEMU_AUDIT_DIR`
  were not set, because the ground truth and the audit data were not copied to the rented host.

| File | Result | Skips |
|---|---|---|
| test_kernels_ieee | 79 passed: v1 and v4 from the fatbin and from NVRTC against the reference, and NVRTC equal to the fatbin | none |
| test_cuda_runtime | 13 passed | none |
| test_hw_mini | 3 passed: `reference`, `cuda:v1` and `cuda:v4` on the hardware-mini cases recorded on the RTX 4060 | `cuda` is not a status entry; `cuda:v9` is gated and sm_90 is not in its `validated_on`; `cuda:tc3` has no sm_90 binary |
| test_fast_kernels | 12 passed (v9) | 12: no tc3 binary for sm_90; 6: no ground truth or audit data |
| test_gates_hmma | 1 passed | 3: they need a device where the tc3 gates pass |
| test_device_independence | 0 passed | all 138: `MMAEMU_TRUTH_DIR` is not set |
| test_live_cublas | skipped | T3p runs only on the profile device (RTX 4060) |
| test_scaffold, test_backend_reference, test_epilogue | 2, 3 and 5 passed | none |

## Evo 2 fixed inputs (`fixed-fp8.json`)

The same host ran the test bundle's `fixed_fp8.py` before T3.
- Code: mmaemu 5c15e8a, with v1 and v4 compiled by NVRTC (the bundle had no fatbins). Their sources differ from
  69cd231 only in comments. T3's `test_nvrtc_equals_fatbin` passed on this GPU (see test_kernels_ieee above).
- Inputs: the recorded FP8 operands of the 21 input-projection GEMMs of one Evo 2 1B 2 kb window
  (`evo2.evo2_1b_w2k`).
- v1 and v4: all 21 unit-scale fp32 accumulators and all 21 real-scale bf16 outputs are bitwise equal to the
  RTX 4060 recording (42 of 42 rows PASS).
- Native `torch._scaled_mm` on this GPU differs from the RTX 4060 output on all 21 calls (`native_observations`).

## Why sm_90 is not in `validated_on`

v1 and v4 list sm_89 and sm_120. sm_120 was added after the full device-independence suite and the wheel check
without nvcc (validation_logs/2026-10-13-sm120). Neither ran on sm_90. v1 and v4 are not gated, so AUTO still
selects v4 on sm_90; `validated_here` is False there.
