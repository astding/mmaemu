# RTX 4060 test host, 2026-10-14

The release GPU tiers on the profile device (T3, T3p, the GPU tier of T4, and T6) ran on an RTX 4060 in a second
PC: the WSL2 host of the hardware recordings now holds the RTX 5070 Ti. Before any test ran, `preflight_4060.py`
asserted that the device and the software stack match the recording environment (`preflight.json`, 18 of 18
checks PASS).

## What matches the recording environment

| Item | Value | How it was checked |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4060, sm_89, 24 SMs, 8 GB | torch device properties; profile `ada-rtx4060-torch` matches (name, capability, SM count, cuBLASLt) |
| cuBLASLt | 12.8.3.14, loaded from the venv's `nvidia-cublas-cu12` wheel | `mmaemu.platform.cublaslt.version()` and the library path |
| Python and packages | CPython 3.12.2; all 32 installed packages equal to the lock of the recording venv (torch 2.7.1+cu128, nvidia-cublas-cu12 12.8.3.14, numpy, triton, ...) | `uv pip freeze` against the lock (`info.packages` in `preflight.json`) |
| Code | `git archive` of da9edd4; the frozen release fatbins | `build_fatbin --check` (5 modules current) and `check_dist --verify-manifest` (14 checks, 0 failed) |
| Ground truth | the truth manifest of the recordings | manifest sha256 `be1270ed...` |

`mmaemu_info.txt` is `python -m mmaemu info` on this host: every backend (tc3, v9, v4, v1, reference) is available
and validated here, and all three RTX 4060 profiles match the device.

## What differs, and why it does not enter the results

| Item | Recordings | This host |
|---|---|---|
| OS | Windows host, WSL2 | Debian 13, kernel 6.12.111, native Linux |
| Driver | 572.16 (Windows) | 570.211.01, open kernel module, with the one-line build fix below |
| CPU | AMD Ryzen 9 9950X (not in the recordings' metadata) | Intel Core i5-6600K, 4 cores, no AVX-512 |
| PCIe link | not recorded | Gen3 x8 |

- The driver loads but does not compile or choose the GEMM kernels: cuBLASLt (kernels and heuristic) comes from
  the same pip wheel version as during the recordings, and mmaemu's kernels are the frozen fatbins (SASS for sm_89,
  no JIT). cuBLASLt's heuristic reads device attributes through the driver; T3p checks its choices on this host on
  samples (the 12 recorded split-K cases of `test_live_parity`, 500 re-queried rows of the split-K table) and
  compares live `torch._scaled_mm` bit for bit with mmaemu's reference backend; it passed (see
  `../t3-rtx-4060-sm89/README.md`).
- Throughput (T6) is measured with CUDA events on the GPU and the operands stay in device memory, so the PCIe link
  and the CPU do not enter the TMAC/s figures.
- The 570.211.01 runfile does not build against this kernel: Linux 6.12.97 and later (here Debian's 6.12.111)
  carry the stable backport of upstream commit 337b1b566db0, which adds a fourth argument (`exclude_bars`) to
  `pci_resize_resource`. `nv-pci-exclude-bars.patch` passes 0 there, as NVIDIA's own modules do on such kernels
  behind a conftest check from 580.126.09 and 590.48.01 on (no 570-branch release has it). It touches only the BAR1
  resize at probe time.

## Files

- `preflight_4060.py`, `preflight.json`: the checks and their results; `info` records the host, `nvidia-smi`, the
  kernel module version and the installed packages.
- `mmaemu_info.txt`: `python -m mmaemu info`.
- `nv-pci-exclude-bars.patch`: the driver build fix.
- `ladder_probe.py`: the probe that measured the backend fallback ladder on this device after the first T3 run
  (see `../t3-rtx-4060-sm89/README.md`); its output is not kept.
- `di_full.sh`: the full-accumulator device-independence run of every kernel after T6
  (`../t3-rtx-4060-sm89/di_full_gpu_kernels.log`).
