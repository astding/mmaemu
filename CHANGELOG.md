# Changelog

All notable changes to mmaemu. Versions follow [PEP 440](https://peps.python.org/pep-0440/).

## 0.1.0

First release as the standalone package `mmaemu`, continuing the prototype `tcemu`. An internal (unpublished)
review of the prototype drove this release.

### Scope

- Bit-accurate emulation of NVIDIA tensor-core FP8 GEMM numerics for two reference platforms, both with
  cuBLAS/cuBLASLt 12.8.3.14 and torch 2.7.1:
  - the H100 PCIe (Hopper, sm_90, 114 SMs), the default: preset `hopper`, profiles `hopper` (`torch._scaled_mm`,
    1 MiB workspace; the default profile), `hopper-te` (Transformer Engine 2.3, 32 MiB workspace) and `hopper-ws0`
    (workspace 0);
  - the RTX 4060 (Ada, sm_89, 24 SMs): preset `ada`, profiles `ada` (`torch._scaled_mm`, 1 MiB workspace), `ada-te`
    (Transformer Engine 2.3, 4 MiB workspace) and `ada-ws0` (workspace 0).

  Without `profile=` or `MMAEMU_PROFILE`, mmaemu returns the H100 PCIe's bits.
- FP8 inputs E4M3 x E4M3 or mixed E4M3/E5M2 (E5M2 x E5M2 is rejected, as by torch) with tensor-wise scales; fp32,
  bf16, fp16 and E4M3 outputs, and E5M2 output from mixed inputs.
- `mmaemu.scaled_mm` with the arguments of torch 2.7.1 `torch._scaled_mm` (same names, order and defaults),
  `mmaemu.accumulate` (the fp32 accumulator; `use_fast_accum` required), `mmaemu.explain` (the call plan without
  running it), `mmaemu.model.dot` (the instruction-level chain), `mmaemu.testing.assert_bitwise_equal`.
- Accumulation plans: fast and slow accumulation (slow: 128-element chunks, on both presets); split-K in the scheme
  of the preset's cuBLASLt kernels, and a preset accepts only the schemes of its own kernels:
  - `ada`: `"serial"` and `"pairwise"`, interleaved 64-wide k-tiles with serial or pairwise fp32 combination;
  - `hopper`: `"contiguous"`, 128-wide k-tiles, each split a contiguous run of ceil(T / S) of them; fast
    accumulation one chain per split, slow accumulation every tile its own chain; serial fp32 combination.
- Split decisions: from the profile's recorded cuBLASLt 12.8.3.14 heuristic table (`splitk="auto"`), a live
  cuBLASLt query on the profile's GPU, an envelope above which S = 1 is inferred, or an explicit
  `splitk=(S, scheme)`. The tables:
  - `heur_h100pcie_lt12.8.3.14` (`tools/export_splitk_table_h100.py`), 1,767,416 rows: the H100 probe's grid
    (850,320 heuristic queries) and the model rows of 9 models (M 1 to 16,384) from the H100 split-K sweep, so
    decode shapes of those models resolve. Its envelope depends on K: S = 1 for M*N above a bound per setting and K,
    1,024 <= K <= 4,194,304. It is validated at 1 MiB and unvalidated at 32 MiB, where its inferences are
    `verified=False` and emit `UnvalidatedEnvelopeWarning` (a subclass of `UnverifiedSplitWarning`). No Evo 2 FP8
    projection shape ((K, N) = (1920, 5760), (4096, 12288), (8192, 24576)) splits in this table;
  - `heur_rtx4060_lt12.8.3.14`: S = 1 is inferred for M*N above twice the largest split M*N of the setting.
- Epilogue as the hardware computes it: alpha = fp32(scale_a * scale_b), fma(acc, alpha, +0), double rounding to
  bf16/fp16, fp8 outputs with `scale_result` and saturation, canonical NaN.
- Backends: the float64 reference model (CPU or GPU) and CUDA kernels for E4M3 x E4M3. Every kernel implements one
  preset (`KernelSpec.preset`), and selection skips a kernel of another preset by design:
  - `ada`: v1 and v4 (IEEE-exact) and, gated by per-device self-tests and validated on sm_89 only, v9 and tc3 (both
    passed their acceptance thresholds on the RTX 4060, T6);
  - `hopper`: h4 and h1 (IEEE-exact: the designs of v4 and v1 with 32-term blocks), validated on sm_89 (RTX 4060)
    and sm_120 (RTX 5070 Ti). h4 (SASS sm_80 to sm_120, PTX compute_80) is preferred to h1 (priority 20 against
    10), which still serves GPUs before sm_80 (SASS sm_75 to sm_120, PTX compute_75).

  Kernels ship as prebuilt fatbins (seven modules: the six kernels and the probes of the gated kernels' self-tests)
  loaded through the CUDA driver API, with NVRTC and the reference model as fallbacks: no nvcc, CUDA toolkit or
  compiler needed.
- `python -m mmaemu info | selftest | explain`. `info` names each CUDA kernel's preset
  (`cuda:h1 (preset hopper): available ...`), and its JSON report gives `kernels.<name>.preset`.

### Validated hardware

- H100 PCIe (sm_90, 114 SMs, driver 570.148.08, rented): a reference platform, from two runs on it. mmaemu replayed
  both through `tools/replay_h100_log.py` on an RTX 5070 Ti (the probe also on the RTX 4060), not on the H100:
  - the bit-exactness probe of 2026-10-15 ([h100-probe](docs/validation_logs/2026-10-15/h100-probe/README.md)):
    how c enters (`MM_cublas64` 445 of 445), the 21 Evo 2 1B projections (168 of 168 outputs, torch and
    Transformer Engine, fast and slow), 374 unsplit and 1,049 split live `torch._scaled_mm` calls, the epilogue
    (64 tie cases) and the zero / special values (150 rows), and the cuBLASLt heuristic (850,320 queries). The
    split-K rule was fitted to the probe's split calls after the data came in. mmaemu reproduces all 1,450 live
    jobs (the 27 e4m3 x e4m3 -> e5m2 jobs raise `ReferenceRejectsError`, as cuBLASLt rejected them) and all 168
    Evo 2 outputs bit for bit on the reference backend, and again with the E4M3 x E4M3 calls on h1 and on h4. On
    fa0fd90 (16 Oct, before the profile rename) the probe replays equal through h4 (`--backend auto`) on the RTX 5070 Ti
    ([rtx-5070-ti-replays](docs/validation_logs/2026-10-16/rtx-5070-ti-replays/README.md)) and on the RTX 4060,
    and its E4M3 x E4M3 calls through h1 on the RTX 4060
    ([rtx-4060-sm89](docs/validation_logs/2026-10-16/rtx-4060-sm89/README.md));
  - the split-K sweep of 2026-10-16
    ([h100-splitk-sweep](docs/validation_logs/2026-10-16/h100-splitk-sweep/README.md)): 11,261,760 heuristic
    queries after a drift gate (the probe's 850,320 re-recorded row for row) and 3,188 live calls. The envelope
    was fitted on the training parts and then tested once on a sealed held-out part: at 1 MiB 154,191 shapes above
    the bound, 0 split; at 32 MiB 77,518, 1 split; at workspace 0 46,654 shapes, 0 split. Its 2,744 live jobs
    (2,416 split, 1,040 of them with K not a multiple of 128 and 1,236 with K > 16,384; the sweep's README gives
    1,228, the count over its 2,400 selected split jobs, which leaves out the 8 split model jobs at K = 18,944),
    run after the split rule was fixed, replay bit for bit
    (`docs/validation_logs/2026-10-16/h100-splitk-sweep/results/analysis/mmaemu_replay_p5.json`), and again on
    fa0fd90 (before the profile rename) through h4 on the RTX 5070 Ti
    ([rtx-5070-ti-replays](docs/validation_logs/2026-10-16/rtx-5070-ti-replays/README.md)). Every split
    combination of the table (140 of 140; 85 before the sweep) and ragged last k-tiles (K % 128 != 0) now have a
    replayed live call.
    `tools/check_splitk_resolution_h100.py` runs `resolve_splitk` against every recorded H100 decision: only the
    one held-out failure differs;
  - CI replays 124 of the probe's live calls from their specs and full-output sha256 (`tests/test_h100_mini.py`,
    `tests/data/h100_mini`).

  As a device: T3 passes on the release fatbins of 2026-10-14 (69cd231; v1, v4 and v9), and the 21 Evo 2
  projections (fixed inputs) pass with v1 and v4 compiled by NVRTC from 5c15e8a
  ([t3-h100-sm90](docs/validation_logs/2026-10-14/t3-h100-sm90/README.md)). sm_90 is not in any kernel's
  `validated_on` (the full device-independence suite did not run there). h1 and h4 have never run on an H100.
- RTX 4060 (sm_89): a reference platform, the one of the hardware ground truth. On fa0fd90 (16 Oct, before the profile
  rename), on the CPU: T2 (the CPU replay of the hardware recordings through mmaemu: 147 of the 173 have outputs it
  models; every count as on 15 Oct) and the CPU tier of the mutation matrix T4 (43 killed, 2 equivalent, the 5
  identity controls survive) pass ([t2](docs/validation_logs/2026-10-16/t2/README.md),
  [t4](docs/validation_logs/2026-10-16/t4/README.md)); on the RTX 4060: T3 on the release fatbins (every kernel they
  contain: v1, v4, v9, tc3, h1 and h4), the full accumulators of the 21 Evo 2 1B projections (v1, v4, v9 and tc3: 84
  passed), T3p (live `torch._scaled_mm` against mmaemu's reference: 6 passed) and the GPU tier of T4 (NVRTC builds of
  the mutated sources: 7 killed, the identity control survives) pass
  ([rtx-4060-sm89](docs/validation_logs/2026-10-16/rtx-4060-sm89/README.md)). T6 passes on da9edd4, on the release
  fatbins of 2026-10-14. Earlier runs: T3, T3p and the GPU tier of T4 on da9edd4; T2, T3 and T3p on 49f1ecf.
  Kernels h1 and h4 validated, on fa0fd90 (16 Oct): their four test files with no marker filter (IEEE
  conformance 167, CUDA runtime 19, H100 CI replay 435 and backend selection 26 passed) and the replay of the H100
  probe through h4 (`--backend auto`: 1,423 live calls equal, the 1,043 E4M3 x E4M3 ones on h4 and the 380
  mixed-format ones on the reference backend, and the 27 calls the H100 rejected raise `ReferenceRejectsError`;
  168 of 168 Evo 2 outputs equal) and through h1 (`--backend cuda:h1`: the 1,043 E4M3 x E4M3 live calls and the 168
  Evo 2 outputs equal; the 380 mixed-format calls raise `BackendUnavailableError`, as h1 is E4M3 x E4M3 only)
  ([rtx-4060-sm89](docs/validation_logs/2026-10-16/rtx-4060-sm89/README.md)). The same runs passed on 7612fd8
  earlier the same day, with the same counts.
- RTX 5070 Ti (sm_120): kernels v1 and v4 validated. On fa0fd90 (16 Oct, before the profile rename): T3 with the full
  accumulators on the release fatbins (`--di-full`: the reference backend, v1 and v4 reproduce the RTX 4060's
  full-accumulator sha256 on all 21 Evo 2 1B projections;
  [t3-rtx-5070-ti-sm120](docs/validation_logs/2026-10-16/t3-rtx-5070-ti-sm120/README.md)) and the GPU tier of T4
  (NVRTC builds of the mutated sources: 7 killed, the identity control survives;
  [t4](docs/validation_logs/2026-10-16/t4/README.md)) pass. Full device independence (v1, v4, v9) and a development
  wheel installed without nvcc (NVRTC fallback) passed on 13 Oct.
  `validated_on` of v1 and v4 is {sm_89, sm_120}. v9 is opt-in there; tc3 ships no sm_120 binary.
  Kernels h1 and h4 validated: their GPU tests from the package fatbins (IEEE conformance against the `hopper`
  reference, the CUDA runtime, the H100 CI replay on the GPU, backend selection) and the replay of the H100 probe
  through each, on 3d7dc66 (h1) and ebd18e1 (h4)
  ([h1-rtx-5070-ti](docs/validation_logs/2026-10-16/h1-rtx-5070-ti/README.md),
  [h4-rtx-5070-ti](docs/validation_logs/2026-10-16/h4-rtx-5070-ti/README.md)). On fa0fd90 T3 runs their GPU
  tests again (IEEE conformance 167, CUDA runtime 19 and the 310 GPU cases of the H100 CI replay passed), and
  the probe's replay through h4 (`--backend auto`: the 1,043 E4M3 x E4M3 live calls and the 168 Evo 2 outputs on
  h4) passes: 1,423 live calls and 168 of 168 Evo 2 outputs equal
  ([rtx-5070-ti-replays](docs/validation_logs/2026-10-16/rtx-5070-ti-replays/README.md)). At M = 2048 on the
  three Evo 2 shapes h1 runs at about 2.0 TMAC/s and h4 at 4.1 to 4.6 TMAC/s. `validated_on` of h1 and h4 is
  {sm_89, sm_120} (sm_89 from the RTX 4060 runs above).

### Documentation and examples

- `examples/evo2_blackwell.ipynb`: upstream Evo 2 1B on an RTX 5070 Ti with its FP8 projections computed by
  mmaemu, checked against the official RTX 4060 outputs shipped in `examples/data/` (one GEMM bit for bit, six
  windows, zero-shot BRCA1); built by `examples/build_evo2_blackwell.py`. Its stored outputs come from its run on
  the code of 0.1.0, with the numbers of its run of 14 October apart from the timings
  ([rtx-5070-ti-renamed](docs/validation_logs/2026-10-16/rtx-5070-ti-renamed/README.md#the-notebook)).
- Evo 2 validation logs: the official Transformer Engine path on an H100 PCIe (the cross-GPU baseline) and the
  notebook's run on the RTX 5070 Ti, under `docs/validation_logs/2026-10-14/`.
- `examples/evo2_offload.py`: upstream Evo 2 (the 40B included) on one GPU with its weights in host RAM and its FP8
  projections computed by mmaemu: the upstream test (PASS or FAIL) and the scoring of sequences.
- The scripts that captured the RTX 4060 reference outputs, their logs, and a check that the shipped reference files
  equal the captures (`docs/validation_logs/2026-10-12/evo2-rtx-4060-official/`).
- Evo 2 40B on the RTX 5070 Ti with its weights in host RAM (`docs/validation_logs/2026-10-15/evo2-40b-rtx-5070-ti/`,
  with the scripts): the upstream test, two windows against the official RTX 4060 outputs, and BRCA1 on 100
  variants, with mmaemu (`ada-te`), native FP8 and bf16 GEMMs; `examples/evo2_offload.py` checked against its
  scripts.
- The official Evo 2 40B path on an H100 PCIe (`docs/validation_logs/2026-10-15/evo2-40b-h100-sm90/`): the upstream
  test (the expected loss, 0.2159424), the windows against the RTX 4060 and the RTX 5070 Ti runs, and BRCA1 on the
  notebook's 500 variants.
- Evo 2 1B and 40B on the RTX 5070 Ti with mmaemu's H100 PCIe profile `hopper-te`
  (`docs/validation_logs/2026-10-16/evo2-h100-profile-rtx-5070-ti/`, with the scripts): the upstream tests, the
  windows and BRCA1 (500 variants for the 1B, 100 for the 40B), against the official H100 and RTX 4060 runs and the
  `ada-te` runs.
- The validation logs of the H100 PCIe work, with their scripts: the probe and the split-K sweep, the builds of the
  h1 and h4 fatbins ([h1-fatbin](docs/validation_logs/2026-10-15/h1-fatbin/README.md),
  [h4-fatbin](docs/validation_logs/2026-10-16/h4-fatbin/README.md)) and their runs on the RTX 5070 Ti and the
  RTX 4060 ([rtx-4060-sm89](docs/validation_logs/2026-10-16/rtx-4060-sm89/README.md)), and the
  fatbin rebuild at the release-candidate freeze ([rc-fatbin](docs/validation_logs/2026-10-16/rc-fatbin/README.md)).
- README with the results, uses and citations; new pages `docs/index.md`, `docs/user_guide.md`, `docs/api.md`,
  `docs/evo2.md`, `docs/faq.md`; figures drawn from the committed logs by `tools/make_figures.py`.
- README section on related work (MMA-Sim, Hawkeye, FPRev, hardware_rounding_error_predictor); FAQ entries for the
  common questions (different FP8 results on another GPU, Evo 2 1B on RTX 50-series GPUs, golden outputs in CPU
  CI); an `llms.txt` index and a `context7.json` for documentation tools used by coding agents.
- A documentation site of the pages of `docs/` (`mkdocs.yml`, `.github/workflows/docs.yml`): built with `--strict`
  on every push and pull request, deployed to GitHub Pages once the repository is public; links to code and logs
  point to the repository (`tools/mkdocs_hooks.py`).

### Maturity

- Presets `ada` and `hopper`: validated. `hopper` is validated for cuBLAS FP8 GEMMs on the H100 PCIe; it is not a
  model of the sm_90 `mma.sync` instruction, which the probe measured as a different unit.
- Profiles `hopper` (default), `hopper-te`, `hopper-ws0`, `ada`, `ada-te` and `ada-ws0`: validated. Split decisions
  are validated on the four `torch._scaled_mm` profiles (workspace 0 resolves every shape to S = 1) and experimental
  with S > 1 on `hopper-te` and `ada-te` (`UnverifiedSplitWarning`). On `hopper-te` the envelope's S = 1 inferences
  are unvalidated as well (`UnvalidatedEnvelopeWarning`).
- Preset `b200_mma_sync` (alias `blackwell`) and profiles `hopper-experimental` (the hopper numerics without a
  reference platform: H100 SXM, H200, other cuBLASLt versions) and `b200-mma-sync-experimental`: experimental
  (`ExperimentalPresetWarning`); neither profile has a split-K table. `b200_mma_sync` is not a model of sm_120 and
  not of B200 cuBLAS.

### Limitations

- Not supported (raise): bias, row-wise scales, autograd (inference only), slow accumulation and split-K on
  `b200_mma_sync`. CUDA graph capture is not supported either: the non-finite-input check synchronises with the
  host.
- E5M2 and mixed-format inputs run on the reference backend only.
- No sm_120 model. The `hopper` preset is validated on one H100 PCIe (114 SMs): the H100 SXM, the H200 and other
  cuBLASLt versions need their own heuristic table (`hopper-experimental` gives the numerics without one).
- Split-K is profile-scoped: shapes the profile cannot resolve raise `SplitKUnknownError`, and on the profiles
  without a table (`hopper-experimental`, `b200-mma-sync-experimental`) `splitk="auto"` always raises. Off the H100,
  `hopper` and `hopper-te` infer S = 1 only for 1,024 <= K <= 4,194,304 and M*N above the envelope's bound, and
  shapes with M above 16,384 or N above 2^20 are extrapolated; `hopper-ws0` resolves every shape to S = 1
  (workspace 0).
- On the H100 profiles a K above 262,144 adds no warning to an explicit `splitk=(S, "contiguous")`, which is
  computed by the contiguous rule although no live split call had such a K; a split that owns no k-tile, or one
  with a customOption that no replayed split call had (4 to 7), warns (`UnverifiedSplitWarning`).
- The CUDA kernels are tested on Linux x86_64 (WSL2 included) only. They load wherever `libcuda.so.1` is found
  (Linux aarch64 is untested); on native Windows and macOS the reference backend is used. Multi-GPU is untested.
  h1 and h4 are validated on sm_89 and sm_120 only; on the H100 they are untested.
- Requires torch >= 2.6: torch 2.5's CPU backend has no float8 `cat` / `fill` kernels, which the reference model
  uses (T0 + T1 fail on torch 2.5.1).

### Changes from the prototype (tcemu), following its review

- Split-K modelled (silently wrong before); NaN inputs in the CUDA kernel; per-context kernel tables (a second GPU
  or context gave zeros); dtype checks (E5M2 was decoded as E4M3); kernel loading without a compiler, with
  fallbacks; `torch._scaled_mm`-compatible signature and defaults; fp8 outputs and `scale_result`; zero sign and
  NaN bits of the epilogue; misaligned inputs; explicit formats in the instruction model; grad-mode guard.
- The B200 preset is renamed `b200_mma_sync`; documentation scoped to the validated platforms (see
  docs/limitations.md).
- Tests compare bit patterns (never `==`), replay committed hardware recordings, run every CUDA kernel, and kill a
  mutation matrix.
- Licence: Apache-2.0, with the BSD-2 notice of the MATLAB Tensor Core toolbox (NOTICE, LICENSES/).

### Changes after the review of the release candidate (2026-10-14)

An adversarial review of the release candidate (8429385) checked the code and the documentation against the
hardware. Changes to behaviour (the first four were measured on the RTX 4060:
[audit-probes](docs/validation_logs/2026-10-14/audit-probes/README.md),
[splitk-envelope](docs/validation_logs/2026-10-14/splitk-envelope/README.md)):

- Split-K inference on the RTX 4060 table: S = 1 is inferred only for M*N above twice the largest split M*N of the
  setting; K no longer plays a part there. The old rule (above the largest split M*N, or K below the smallest
  split K) inferred S = 1 for shapes that cuBLASLt splits. In 1,512,000 live queries above the largest split M*N,
  the RTX 4060 split shapes up to 1.27 times it; in a separate sweep of 125,419 shapes with K below the table's
  smallest split K, 18,504 split. Off the RTX 4060, more shapes now raise `SplitKUnknownError`; the Evo 2 shapes
  are unaffected.
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

### The H100 PCIe as the default platform (2026-10-15 to 2026-10-16)

The candidate that T7 validated on 2026-10-15 (the wheel built from 85c780d; its T7 reports were committed in
a79668a) had one reference platform, the RTX 4060, and no H100 profile. The H100 PCIe probe, the kernels h1 and h4
and the split-K sweep followed, and the H100 PCIe became a reference platform and the default. Changes to
behaviour against that candidate:

- The default profile is `hopper` (it was `ada-rtx4060-torch`, now named `ada`: see the profile names below): without
  `profile=` or `MMAEMU_PROFILE`, mmaemu returns the H100 PCIe's bits. With `splitk="auto"`, a shape that is neither
  in the H100 table nor above its envelope raises `SplitKUnknownError`; pass `profile="ada"` for the previous
  behaviour.
- Preset `hopper` was experimental (instruction-level vectors with c = 0; slow accumulation and split-K raised).
  It is now validated for cuBLAS on the H100 PCIe and models slow accumulation (128-element chunks) and the new
  split-K scheme `"contiguous"`. It is not a model of the sm_90 `mma.sync` instruction, which the probe measured as
  a different unit.
- Profile `hopper-experimental` now means the hopper numerics without a reference platform (H100 SXM, H200, other
  cuBLASLt versions); it emits `ExperimentalPresetWarning` as an experimental profile.
- New: the profiles `hopper`, `hopper-te` and `hopper-ws0`, the table `heur_h100pcie_lt12.8.3.14` with its
  envelope, `UnvalidatedEnvelopeWarning`, the kernels h1 and h4 (ungated; `validated_on` {sm_89, sm_120}, from the
  RTX 4060 and RTX 5070 Ti runs under Validated hardware, decides only "validated here" in `backends_status()` and
  `python -m mmaemu info`, not selection or `selftest`), and `KernelSpec.preset` (see Scope).
- On a GPU preset `hopper` (the H100 profiles and `hopper-experimental`) runs E4M3 x E4M3 on h4 (h1 before sm_80)
  instead of the reference backend.
- `python -m mmaemu info` names each CUDA kernel's preset, and its JSON report gives `kernels.<name>.preset`.
- `mmaemu.explain` (and `python -m mmaemu explain`) now runs the checks `scaled_mm` makes when it plans the GEMM,
  whose errors only `scaled_mm` raised before: a split scheme of another preset (`splitk=(2, "serial")` on a hopper
  profile) and slow accumulation or S > 1 on `b200_mma_sync` are reported in `CallPlan.validation` (`ValueError` /
  `UnsupportedFeatureError`; the CLI exits 1) instead of `"ok"`.
- Split decisions with S > 1 are validated on `hopper` and `hopper-ws0` and experimental on `hopper-te`
  (`UnverifiedSplitWarning`). The split rule was fitted after the probe's data came in; the sweep's 2,416 live
  split calls, run after it was fixed, replay equal (1,630 at 1 MiB, covering every S and customOption of the
  table's 1 MiB split rows, and every K of those rows except 5,760, which only the probe's split calls had); no
  split call ran through TE's own GEMM. The maturity was decided after the sweep, not before it. `"contiguous"`
  splits outside the replayed layouts (a split without k-tiles, a customOption never seen on a replayed split
  call) also warn; a K that is not a multiple of 128 does not (the sweep's 1,040 such live split calls replay
  equal).

h1 and h4 were committed as `--release` builds of their own modules. At the freeze of the release candidate
(2026-10-16) every fatbin module was rebuilt from the committed sources (nvcc 12.8.61, `--release`) and compared
byte for byte with the committed one: h1, h4, v9, tc3 and probes are identical. The rebuilt v1 and v4 have the
same SASS on every architecture and differ only in the hashed names nvcc gives anonymous-namespace symbols in the
PTX, so their committed bytes, on which T3 ran on the RTX 4060, the RTX 5070 Ti and the H100 PCIe, were kept
([rc-fatbin](docs/validation_logs/2026-10-16/rc-fatbin/README.md)).

Evo 2 with the H100 PCIe profile, on 2026-10-16
([evo2-h100-profile-rtx-5070-ti](docs/validation_logs/2026-10-16/evo2-h100-profile-rtx-5070-ti/README.md)): upstream
Evo 2 1B and 40B on the RTX 5070 Ti with their FP8 projections computed by mmaemu with `hopper-te` (on fa0fd90,
where it was named `hopper-h100pcie-te`; kernel h4, S = 1 from the table), as the runs of 14 and 15 October did
with `ada-te`. Both pass the upstream test: the 1B gives 0.5022, the H100's value in bf16 (in fp32 0.5024, the
official H100 run 0.5026, `ada-te` 0.5025), the 40B 0.2161 (expected 0.2159424, the official H100 run's value;
`ada-te` 0.2161). BRCA1 AUROC: 1B 0.7354 on the 500 variants (official H100 0.7332, `ada-te` 0.7323), 40B 0.8569
on the first 100 (official H100 and `ada-te` 0.855). The FP8 GEMMs are the H100's (the 168 outputs of the 21 Evo 2
1B projections captured on the H100 equal mmaemu's on h4 bit for bit; no 40B projection was captured there); the
other operations run natively, so the outputs are not bit-identical to the official H100 run's. On the 40B windows
w2k and w4k the mean |Δ log p(A,C,G,T)| from the official H100 run is 0.0289 and 0.0283 (`ada-te` 0.0295 and
0.0288), the size of the distance between the two official GPUs (0.030 and 0.029).

### The profile names, before the release (2026-10-16)

Before the release the profiles of the two reference platforms were renamed after their preset (63b7395, late on
16 October PDT, after the Evo 2 run above): `hopper-h100pcie-torch` -> `hopper` (the default),
`hopper-h100pcie-te` -> `hopper-te`, `hopper-h100pcie-ws0` -> `hopper-ws0`, `ada-rtx4060-torch` -> `ada`,
`ada-rtx4060-te` -> `ada-te` and `ada-rtx4060-ws0` -> `ada-ws0`. `hopper-experimental` and
`b200-mma-sync-experimental` are unchanged.

- No aliases: an old name raises `KeyError` (through `get_profile`, `profile=` or `MMAEMU_PROFILE`).
- Each profile's description now says where it was measured ("as measured on an H100 PCIe ..." / "... on an
  RTX 4060 ...").
- Logs and data made before the rename keep the old names: every log under `docs/validation_logs/` up to and
  including the runs of fa0fd90 on 2026-10-16 and the split-K sweep's `eval_sealed.json`. The profiles that the
  replays derive for other workspaces, `hopper-h100pcie-torch-ws<N>` in those logs, are now named `hopper-ws<N>`.
- The rename changes names and descriptions only. The validation runs of fa0fd90 on the RTX 4060 and the
  RTX 5070 Ti (T2, T3, T3p, T4, the replays of the H100 probe and sweep, the RTX 4060's split-K sweeps and bench)
  ran again on the code of 63b7395 (2648bba) and gave the same results once the old names are read as the new ones,
  apart from timings (the bench's rates within 0.01 TMAC/s)
  ([rtx-4060-sm89-renamed](docs/validation_logs/2026-10-16/rtx-4060-sm89-renamed/README.md),
  [rtx-5070-ti-renamed](docs/validation_logs/2026-10-16/rtx-5070-ti-renamed/README.md)); so did the upstream tests
  of the Evo 2 run with `hopper-te`, through `examples/evo2_offload.py`, with the same logits
  ([example check](docs/validation_logs/2026-10-16/evo2-h100-profile-rtx-5070-ti/README.md)).
- The shipped table `heur_h100pcie_lt12.8.3.14` keeps `"eval_profile": "hopper-h100pcie-torch"` and
  `"hopper-h100pcie-te"`, the keys of `eval_sealed.json` that it names, so the table files and their sha256 are
  unchanged.
