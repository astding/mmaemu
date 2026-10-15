# Changelog

All notable changes to mmaemu. Versions follow [PEP 440](https://peps.python.org/pep-0440/).

## 0.1.0 (unreleased)

First release as the standalone package `mmaemu` (continuing the prototype `tcemu`, whose audit drove this
release).

### Scope

- Bit-accurate emulation of NVIDIA tensor-core FP8 GEMM numerics for one reference platform: RTX 4060 (Ada, sm_89,
  24 SMs), cuBLAS/cuBLASLt 12.8.3.14, torch 2.7.1 `torch._scaled_mm` (1 MiB workspace) or Transformer Engine 2.3
  (4 MiB workspace).
- FP8 inputs (E4M3, E5M2, mixed) with tensor-wise scales; fp32, bf16, fp16, E4M3 and E5M2 outputs.
- `mmaemu.scaled_mm` with the arguments of torch 2.7.1 `torch._scaled_mm` (same names, order and defaults),
  `mmaemu.accumulate` (the fp32 accumulator; `use_fast_accum` required), `mmaemu.explain` (the call plan without
  running it), `mmaemu.model.dot` (the instruction-level chain), `mmaemu.testing.assert_bitwise_equal`.
- Accumulation plans: fast and slow accumulation; split-K with interleaved 64-wide k-tiles and serial or pairwise
  fp32 combination; split decisions from a recorded cuBLASLt 12.8.3.14 heuristic table (`splitk="auto"`), an
  envelope inference outside the split region, or an explicit `splitk=(S, scheme)`.
- Epilogue as the hardware computes it: alpha = fp32(scale_a * scale_b), fma(acc, alpha, +0), double rounding to
  bf16/fp16, fp8 outputs with `scale_result` and saturation, canonical NaN.
- Backends: the float64 reference model (CPU or GPU); CUDA kernels v1 and v4 (IEEE-exact) and, gated by per-device
  self-tests and validated on sm_89 only, v9 and tc3 (each only if it passed its acceptance at the release-candidate
  freeze; otherwise it follows in 0.1.x). Kernels ship as prebuilt fatbins loaded through the CUDA driver API, with
  NVRTC and the reference model as fallbacks: no nvcc, CUDA toolkit or compiler needed.
- `python -m mmaemu info | selftest | explain`.

### Validated hardware

- RTX 4060 (sm_89): the reference platform; T2 (the replay of all 173 hardware recordings through mmaemu) passes.
- RTX 5070 Ti (sm_120): kernels v1 and v4 validated (T3 on the release fatbins, device independence, GPU mutation
  matrix, wheel without nvcc); `validated_on` of v1 and v4 is {sm_89, sm_120}. v9 is opt-in there; tc3 ships no
  sm_120 binary.
- H100 PCIe (sm_90): T3 on the release fatbins and the 21 Evo 2 projections (fixed inputs) pass; sm_90 is not in
  `validated_on` (the full device-independence suite did not run there).

### Documentation and examples

- `examples/evo2_blackwell.ipynb`: upstream Evo 2 1B on an RTX 5070 Ti with its FP8 projections computed by
  mmaemu, checked against the official RTX 4060 outputs shipped in `examples/data/` (one GEMM bit for bit, six
  windows, zero-shot BRCA1); built by `examples/build_evo2_blackwell.py`.
- Evo 2 validation logs: the official Transformer Engine path on an H100 PCIe (the cross-GPU baseline) and the
  notebook's run on the RTX 5070 Ti, under `docs/validation_logs/2026-10-14/`.
- README with the results, uses and citations; new pages `docs/index.md`, `docs/user_guide.md`, `docs/api.md`,
  `docs/evo2.md`, `docs/faq.md`; figures drawn from the committed logs by `tools/make_figures.py`.

### Maturity

- Preset `ada` and profiles `ada-rtx4060-torch` (default), `ada-rtx4060-te` and `ada-rtx4060-ws0`: validated.
  Split decisions with S > 1 on `ada-rtx4060-te` are experimental (`UnverifiedSplitWarning`).
- Presets `hopper` and `b200_mma_sync` (alias `blackwell`), profiles `hopper-experimental` and
  `b200-mma-sync-experimental`: experimental (`ExperimentalPresetWarning`). `b200_mma_sync` is not a model of
  sm_120 and not of B200 cuBLAS.

### Limitations

- Not supported (raise): bias, row-wise scales, autograd (inference only), slow accumulation on the experimental
  presets, CUDA graph capture.
- E5M2 and mixed-format inputs run on the reference backend only.
- No sm_120 model. Split-K is profile-scoped: shapes the profile cannot resolve raise `SplitKUnknownError`.
- CUDA kernels on Linux x86_64 (WSL2 included) only; multi-GPU untested.
- Requires torch >= 2.6: torch 2.5's CPU backend has no float8 `cat` / `fill` kernels, which the reference model
  uses (T0 + T1 fail on torch 2.5.1).

### Changes from the prototype (tcemu), following its audit

- Split-K modelled (silently wrong before); NaN inputs in the CUDA kernel; per-context kernel tables (a second GPU
  or context gave zeros); dtype checks (E5M2 was decoded as E4M3); kernel loading without a compiler, with
  fallbacks; `torch._scaled_mm`-compatible signature and defaults; fp8 outputs and `scale_result`; zero sign and
  NaN bits of the epilogue; misaligned inputs; explicit formats in the instruction model; grad-mode guard.
- The B200 preset is renamed `b200_mma_sync`; documentation scoped to the validated platform (see docs/limitations.md).
- Tests compare bit patterns (never `==`), replay committed hardware recordings, run every CUDA kernel, and kill a
  mutation matrix.
- Licence: Apache-2.0, with the BSD-2 notice of the MATLAB Tensor Core toolbox (NOTICE, LICENSES/).
