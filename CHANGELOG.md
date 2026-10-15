# Changelog

All notable changes to mmaemu. Versions follow [PEP 440](https://peps.python.org/pep-0440/).

## 0.1.0

First release as the standalone package `mmaemu`, continuing the prototype `tcemu`. An internal (unpublished)
review of the prototype drove this release.

### Scope

- Bit-accurate emulation of NVIDIA tensor-core FP8 GEMM numerics for one reference platform: RTX 4060 (Ada, sm_89,
  24 SMs), cuBLAS/cuBLASLt 12.8.3.14, torch 2.7.1 `torch._scaled_mm` (1 MiB workspace) or Transformer Engine 2.3
  (4 MiB workspace).
- FP8 inputs E4M3 x E4M3 or mixed E4M3/E5M2 (E5M2 x E5M2 is rejected, as by torch) with tensor-wise scales; fp32,
  bf16, fp16 and E4M3 outputs, and E5M2 output from mixed inputs.
- `mmaemu.scaled_mm` with the arguments of torch 2.7.1 `torch._scaled_mm` (same names, order and defaults),
  `mmaemu.accumulate` (the fp32 accumulator; `use_fast_accum` required), `mmaemu.explain` (the call plan without
  running it), `mmaemu.model.dot` (the instruction-level chain), `mmaemu.testing.assert_bitwise_equal`.
- Accumulation plans: fast and slow accumulation; split-K with interleaved 64-wide k-tiles and serial or pairwise
  fp32 combination; split decisions from a recorded cuBLASLt 12.8.3.14 heuristic table (`splitk="auto"`), a live
  cuBLASLt query on the reference GPU, S = 1 inferred for M*N above twice the largest split M*N of the setting, or
  an explicit `splitk=(S, scheme)`.
- Epilogue as the hardware computes it: alpha = fp32(scale_a * scale_b), fma(acc, alpha, +0), double rounding to
  bf16/fp16, fp8 outputs with `scale_result` and saturation, canonical NaN.
- Backends: the float64 reference model (CPU or GPU); CUDA kernels v1 and v4 (IEEE-exact) and, gated by per-device
  self-tests and validated on sm_89 only, v9 and tc3 (both passed their acceptance thresholds on the RTX 4060,
  T6). Kernels ship as prebuilt fatbins loaded through the CUDA driver API, with
  NVRTC and the reference model as fallbacks: no nvcc, CUDA toolkit or compiler needed.
- `python -m mmaemu info | selftest | explain`.

### Validated hardware

- RTX 4060 (sm_89): the reference platform; T2 (the CPU replay of the hardware recordings through mmaemu: 147 of
  the 173 have outputs it models) passes.
  On the release fatbins T3 (every kernel available), T3p (live `torch._scaled_mm` against mmaemu's reference)
  and T6 pass; the GPU mutation matrix (NVRTC builds of the mutated sources) passes.
- RTX 5070 Ti (sm_120): kernels v1 and v4 validated: T3 on the release fatbins; the GPU mutation matrix; full
  device independence (v1, v4, v9) and a development wheel installed without nvcc (NVRTC fallback), 13 Oct.
  `validated_on` of v1 and v4 is {sm_89, sm_120}. v9 is opt-in there; tc3 ships no sm_120 binary.
- H100 PCIe (sm_90): T3 passes on the release fatbins; the 21 Evo 2 projections (fixed inputs) pass with v1 and v4
  compiled by NVRTC from 5c15e8a. sm_90 is not in `validated_on` (the full device-independence suite did not run
  there).

### Documentation and examples

- `examples/evo2_blackwell.ipynb`: upstream Evo 2 1B on an RTX 5070 Ti with its FP8 projections computed by
  mmaemu, checked against the official RTX 4060 outputs shipped in `examples/data/` (one GEMM bit for bit, six
  windows, zero-shot BRCA1); built by `examples/build_evo2_blackwell.py`.
- Evo 2 validation logs: the official Transformer Engine path on an H100 PCIe (the cross-GPU baseline) and the
  notebook's run on the RTX 5070 Ti, under `docs/validation_logs/2026-10-14/`.
- `examples/evo2_offload.py`: upstream Evo 2 (the 40B included) on one GPU with its weights in host RAM and its FP8
  projections computed by mmaemu: the upstream test (PASS or FAIL) and the scoring of sequences.
- The scripts that captured the RTX 4060 reference outputs, their logs, and a check that the shipped reference files
  equal the captures (`docs/validation_logs/2026-10-12/evo2-rtx-4060-official/`).
- Evo 2 40B on the RTX 5070 Ti with its weights in host RAM (`docs/validation_logs/2026-10-15/evo2-40b-rtx-5070-ti/`,
  with the scripts): the upstream test, two windows against the official RTX 4060 outputs, and BRCA1 on 100
  variants, with mmaemu, native FP8 and bf16 GEMMs; `examples/evo2_offload.py` checked against its scripts.
- The official Evo 2 40B path on an H100 PCIe (`docs/validation_logs/2026-10-15/evo2-40b-h100-sm90/`): the upstream
  test (the expected loss, 0.2159424), the windows against the RTX 4060 and the RTX 5070 Ti runs, and BRCA1 on the
  notebook's 500 variants.
- README with the results, uses and citations; new pages `docs/index.md`, `docs/user_guide.md`, `docs/api.md`,
  `docs/evo2.md`, `docs/faq.md`; figures drawn from the committed logs by `tools/make_figures.py`.
- README section on related work (MMA-Sim, Hawkeye, FPRev, hardware_rounding_error_predictor); FAQ entries for the
  common questions (different FP8 results on another GPU, Evo 2 1B on RTX 50-series GPUs, golden outputs in CPU
  CI); an `llms.txt` index and a `context7.json` for documentation tools used by coding agents.
- A documentation site of the pages of `docs/` (`mkdocs.yml`, `.github/workflows/docs.yml`): built with `--strict`
  on every push and pull request, deployed to GitHub Pages once the repository is public; links to code and logs
  point to the repository (`tools/mkdocs_hooks.py`).

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
- The CUDA kernels are tested on Linux x86_64 (WSL2 included) only. They load wherever `libcuda.so.1` is found
  (Linux aarch64 is untested); on native Windows and macOS the reference backend is used. Multi-GPU is untested.
- Requires torch >= 2.6: torch 2.5's CPU backend has no float8 `cat` / `fill` kernels, which the reference model
  uses (T0 + T1 fail on torch 2.5.1).

### Changes from the prototype (tcemu), following its review

- Split-K modelled (silently wrong before); NaN inputs in the CUDA kernel; per-context kernel tables (a second GPU
  or context gave zeros); dtype checks (E5M2 was decoded as E4M3); kernel loading without a compiler, with
  fallbacks; `torch._scaled_mm`-compatible signature and defaults; fp8 outputs and `scale_result`; zero sign and
  NaN bits of the epilogue; misaligned inputs; explicit formats in the instruction model; grad-mode guard.
- The B200 preset is renamed `b200_mma_sync`; documentation scoped to the validated platform (see docs/limitations.md).
- Tests compare bit patterns (never `==`), replay committed hardware recordings, run every CUDA kernel, and kill a
  mutation matrix.
- Licence: Apache-2.0, with the BSD-2 notice of the MATLAB Tensor Core toolbox (NOTICE, LICENSES/).

### Changes after the review of the release candidate (2026-10-14)

An adversarial review of the release candidate (8429385) checked the code and the documentation against the
hardware. Changes to behaviour (the first four were measured on the RTX 4060:
[audit-probes](docs/validation_logs/2026-10-14/audit-probes/README.md),
[splitk-envelope](docs/validation_logs/2026-10-14/splitk-envelope/README.md)):

- Split-K inference: S = 1 is inferred only for M*N above twice the largest split M*N of the setting; K no longer
  plays a part. The old rule (above the largest split M*N, or K below the smallest split K) inferred S = 1 for
  shapes that cuBLASLt splits. In 1,512,000 live queries above the largest split M*N, the RTX 4060 split shapes up
  to 1.27 times it; in a separate sweep of 125,419 shapes with K below the table's smallest split K, 18,504 split.
  Off the reference GPU, more shapes now raise `SplitKUnknownError`; the Evo 2 shapes are unaffected.
- A column-major `out=` now uses the key cuBLASLt sees, (N, M, K) with the formats swapped: mmaemu matched the
  hardware on 28 of 406 such cases before and on 406 of 406 now.
- A mat1 that is neither row- nor column-major is copied to a contiguous tensor, as torch does, instead of
  raising.
- Empty calls (M, N or K = 0) return before the out-dtype and format checks, as torch does; an `out=` without unit
  stride is returned unchanged with a `UserWarning`; float8 formats other than E4M3 and E5M2 raise
  `ReferenceRejectsError`; `use_fast_accum` must be a Python bool.
- `explain()` reports the fallbacks for the call's own formats and backend; a `"cuda"` device without an index no
  longer fails on hosts without CUDA; `python -m mmaemu explain` reports bad arguments as usage errors (exit 2) and an
  unusable device with exit 4.
- Tools: `tools/validate_release.py` scrubs every `MMAEMU_*` variable and the PTX JIT variables from the release venv,
  and checks that v1 and v4 ran from the wheel's fatbins and that every kernel loaded from a fatbin used the image
  whose sha256 the wheel's manifest records; `tools/bench_kernels.py` exits 1 on an exactness mismatch.
- Bring-up kits (`tools/bringup/`): they require mmaemu's v0.1 API and raise `ImportError` without it, instead of
  falling back to a module that no longer exists; G1 runs T3 without an overall time limit (each test file keeps its
  own 600 s), so a full T3 run is no longer cut short.

The fixed tree (49f1ecf) ran again on 2026-10-15. On the RTX 4060: the preflight, T3 and T3p pass (T3p now has 6
tests; one makes 11,760 live queries above the new bound, all S = 1), the three split-K sweeps give the same rows,
and a bench smoke is within 1% of the T6 throughput
([log](docs/validation_logs/2026-10-15/rtx-4060-sm89/README.md)). T2 on the CPU passes with every count unchanged
([log](docs/validation_logs/2026-10-15/t2/README.md)).
