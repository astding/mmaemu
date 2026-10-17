# Validation evidence

mmaemu's oracle is hardware. Every numerics claim in the documentation rests on outputs recorded on a reference
GPU: for the RTX 4060, one versioned, read-only dataset (the "hardware ground truth", `truth:*`); for the H100 PCIe,
the committed logs of a probe and a split-K sweep (below). A behaviour ships as *validated* only if those
recordings reproduce it bit for bit; otherwise it ships as *experimental* (with a warning) or raises.

**Evidence references.** Every line of this page that states a number carries at least one reference, and
`tools/check_docs.py` checks that each reference resolves:

- `truth:<id>`: a recording of the hardware ground truth (a glob such as `truth:c2.*` names a group);
- `repo:<path>`: a file of this repository;
- `log:<path>`: a committed report under docs/validation_logs/ or docs/release_logs/.

The ground truth is pinned by the sha256 of its manifest, be1270ed64432a1311ab5e2c98ccb69028f72e3424ad4c6148d4cd468fba6695 (`truth:*`).
Each recording stores its inputs, the hardware's output bits (or their sha256), the call parameters, the cuBLASLt
configuration where known, and the hardware and software versions.

**Profile names.** The profiles were renamed in 63b7395, without aliases (an old name raises `KeyError`,
`repo:mmaemu/platform/profiles.py`): `hopper-h100pcie-torch` -> `hopper` (the default), `hopper-h100pcie-te` ->
`hopper-te`, `hopper-h100pcie-ws0` -> `hopper-ws0`, `ada-rtx4060-torch` -> `ada`, `ada-rtx4060-te` -> `ada-te`,
`ada-rtx4060-ws0` -> `ada-ws0`; the replay tool's derived names `hopper-h100pcie-torch-ws<N>` are now `hopper-ws<N>`
(`repo:tools/replay_h100_log.py`). This page uses the new names. What was recorded before the rename keeps the old
names: the logs of every run up to and including the runs on fa0fd90 (for example
`log:docs/validation_logs/2026-10-16/evo2-h100-profile-rtx-5070-ti/logs/1b_test.log`) and the `eval_profile` field
of the shipped H100 split-K table (`repo:mmaemu/platform/data/heur_h100pcie_lt12.8.3.14.json`).

## Reference platform: RTX 4060

| Item | Value | Evidence |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4060, Ada, sm_89, 24 SMs, driver 572.16, WSL2 | `truth:*` |
| CUDA, cuBLAS, cuBLASLt | CUDA 12.8 (nvcc 12.8.61), cuBLAS and cuBLASLt 12.8.3.14 | `truth:*` |
| torch | 2.7.1+cu128 | `truth:*` |
| Transformer Engine | 2.3.0, where a recording says so | `truth:c4.v3_te_linear`, `truth:c5.v4_te_demo`, `truth:evo2.evo2_1b_w2k` |

## Reference platform: H100 PCIe (the default platform)

The H100 evidence is not part of the ground truth: it is a probe on a rented H100 PCIe
(`log:docs/validation_logs/2026-10-15/h100-probe/README.md`) and a split-K sweep on a second rented H100 PCIe
instance with the same driver and cuBLASLt library
(`log:docs/validation_logs/2026-10-16/h100-splitk-sweep/README.md`), each committed with its scripts, metadata,
sha256 of every output and the live calls' records. The probe's inputs include the ground truth's
Evo 2 1B window (`truth:evo2.evo2_1b_w2k`).

| Item | Value | Evidence |
|---|---|---|
| GPU | NVIDIA H100 PCIe, Hopper, sm_90, 114 SMs, 80 GB, driver 570.148.08 | `log:docs/validation_logs/2026-10-15/h100-probe/results/env/env.json` |
| cuBLAS, cuBLASLt | 12.8.3.14 (the torch wheel's), every GEMM an `nvjet_*` kernel (algo 66, k-stage 128) | `log:docs/validation_logs/2026-10-15/h100-probe/README.md` |
| torch | 2.7.1+cu128 | `log:docs/validation_logs/2026-10-15/h100-probe/results/env/env.json` |
| Transformer Engine | 2.3.0 (`general_gemm`, layout TN, 32 MiB workspace) | `log:docs/validation_logs/2026-10-15/h100-probe/results/te_setup/te-setup.log` |

## Scope (the tcemu audit's scope table, updated for v0.1)

One row per row of the audit's scope table.

| Item | v0.1 status | Hardware evidence | Not covered |
|---|---|---|---|
| Ada instruction model | validated, bitwise (preset `ada`), all four FP8 format pairs; a block result that is exactly zero or truncates to zero is +0 | `truth:c2.*`, `truth:c6.kit_MM_ada_sm89`, `truth:c2.S16_zeros` | other Ada GPUs (expected identical, not measured) |
| Hopper preset | validated for cuBLAS on the H100 PCIe, bitwise: c enters in the block's alignment (`MM_cublas64` 445 of 445 through cuBLAS, every stored n x n block), slow accumulation restarts every 128 and adds in K order, the 21 Evo 2 1B projections (168 of 168 outputs), 374 unsplit live calls | `log:docs/validation_logs/2026-10-15/h100-probe/README.md`, the toolbox's H100/H200 vectors (repository test data), `truth:c6.kit_TBX_H100c0_sm89` | the sm_90 `mma.sync` instruction (a different unit); the H100 SXM, the H200, other cuBLASLt versions |
| b200_mma_sync preset (alias blackwell) | experimental: fits the toolbox's B200 `mma.sync` (HMMA-rewrite) vectors; the model is not identified | the toolbox's B200 vectors (repository test data); the same inputs and discriminating probes on the RTX 4060: `truth:c6.kit_TBX_B200_sm89`, `truth:c6.interleave_probe_kit_sm89` | everything about sm_120; B200 cuBLAS (native FP8 path) |
| cuBLAS GEMM, fast accumulation, unsplit | validated, bitwise, tensor-wise scales | `truth:c4.V1`, `truth:c4.V6`, `truth:c4.V5WS0`, `truth:c2.v7_gemm_subnormal`, `truth:misc.random_run*` | other GPUs, SM counts and cuBLAS versions |
| Slow accumulation | validated: 128-element chunks when unsplit, per split when split (RTX 4060 and H100 PCIe) | `truth:misc.slow_accum_sweep_a1_s05`, `truth:c4.V7`, `truth:c5.v3_slow_heuristic`, `truth:c3.v3_slow_splitk`, `log:docs/validation_logs/2026-10-15/h100-probe/README.md` | `b200_mma_sync` (raises) |
| Split-K | validated where the profile's table, workspace 0 or an explicit (S, scheme) decides; interleaved 64-wide k-tiles, serial and pairwise combination. Not validated decisions, and shown as such by `explain` (source `live-query` or `envelope`): on the reference GPU a live cuBLASLt query decides shapes missing from the table; elsewhere S = 1 is an inference, made only above twice the setting's largest split M*N (in 1,512,000 live queries the RTX 4060 split no shape above 1.27 times it) | `truth:c5.*`, `truth:c4.verify2`, `truth:misc.tcemu_tests_recorded_hw`, `truth:c3.v3_order_probe`, `truth:heur.lt_heuristic_12_8_3_14`, `log:docs/validation_logs/2026-10-14/splitk-envelope/README.md` (run again on fa0fd90, every row equal: `log:docs/validation_logs/2026-10-16/rtx-4060-sm89/README.md`) | other GPUs; shapes missing from the table at or below that bound, off the reference GPU (raise); S > 1 on the TE profile (experimental). The H100 PCIe (below) |
| Split-K on the H100 PCIe | validated at 1 MiB and workspace 0, experimental with TE (no split call ran through TE's own GEMM): contiguous runs of 128-wide k-tiles, serial combination, every customOption; fitted after the data came in to the probe's 1,049 live split calls (S = 2 to 6, K from 5,760 to 16,384); the split-K sweep's 2,416 live split calls, run after the rule was fixed (S = 2 to 24, K up to 262,144, 1,040 with K % 128 != 0; 1,630 at 1 MiB), replay equal too, so all 140 split combinations of the table have a reproduced call. The table (probe grid and the sweep's model rows of 9 models) decides its rows; off it, a K-dependent envelope infers S = 1: validated at 1 MiB on a sealed held-out set (154,191 shapes above the bound, 0 split), unvalidated at 32 MiB (77,518, 1 split; `UnvalidatedEnvelopeWarning`); resolve_splitk agrees with every other H100 decision recorded (10,510,083 status-0 heuristic rows of the sweep, 4,167 live calls) | `log:docs/validation_logs/2026-10-15/h100-probe/README.md`, `log:docs/validation_logs/2026-10-16/h100-splitk-sweep/README.md` (results/analysis: mmaemu_replay_p5.json, eval_sealed.json, resolver_check.json) | a split without k-tiles, S above 24, customOption 4 to 7 on split rows (warn); an explicit split with K above 262,144 (computed by the rule without a warning); the TE envelope (warns); shapes below the envelope's bound and off the table, or K outside 1,024..4,194,304, off the H100 (raise) |
| Epilogue | validated: alpha = fp32(scale_a * scale_b), fma(acc, alpha, +0), double rounding to bf16/fp16, fp8 output with scale_result and saturation, canonical NaN, independence from the out= buffer | `truth:epilogue.*`, `truth:c3.gemm_boundary`, `truth:c3.gemm_fp8out`, `truth:c4.verify4_fp8out`, `truth:c3.v0_epi_prefill` | bias (raises) |
| Row-wise scales | not supported in v0.1 (UnsupportedFeatureError); recorded for later work | `truth:c4.V2R`, `truth:c4.verify6_rowwise`, `truth:misc.rowwise_c1_8` | the CUTLASS stream-K path, whose result depends on the SM count |
| CUDA kernels | v1 and v4 (IEEE-exact) and the gated v9 and tc3 (validated on sm_89) must equal the reference bitwise in the GPU tier, and h1 and h4 (IEEE-exact, the `hopper` preset) the hopper reference; NaN inputs, misaligned pointers and second contexts are handled | the prototype kernels' outputs on fixed inputs, kept for device-independence checks: `truth:emulator.tcemu_kernels_fixed_inputs` | v1 and v4: GPUs other than sm_89 and sm_120 (self-test gated, unvalidated; sm_90 passed T3, see below); v9 and tc3: GPUs other than sm_89; h1 and h4: GPUs other than sm_89 and sm_120 (they have never run on an H100); multi-GPU |
| Device independence (CPU = GPU) | every backend equals the CPU reference bitwise, NaN canonical; CPU flush-to-zero raises | `truth:evo2.evo2_1b_w2k`, `truth:emulator.tcemu_kernels_fixed_inputs` | other GPUs until their GPU-tier run |
| Evo 2 end to end | per projection: the 21 FP8 projections of one Evo 2 1B window (accumulator, native bf16 output, TE output) | `truth:evo2.evo2_1b_w2k` | other GPUs; grad mode; full-model runs: Evo 2 1B and 40B end to end, below |
| Native vs Transformer Engine | equal on shapes that do not split on the RTX 4060 (every Evo 2 projection); different on split-prone shapes. On the H100 PCIe TE's logged decision equals the table's (torch descriptor, 32 MiB) on all 168 Evo 2 calls, all S = 1 | `truth:c4.v3_te_linear`, `truth:c5.v4_te_demo`, `truth:c5.v1_workspace_env`, `log:docs/validation_logs/2026-10-15/h100-probe/README.md` | TE split decisions on the H100 (experimental) |

## Claims and their recordings

| Claim | Recordings |
|---|---|
| Zero products are excluded from the block maximum; subnormal inputs keep exponent emin; an all-zero block still truncates c; exact zeros are +0 | `truth:c2.T_phantom`, `truth:c2.T_subnormal`, `truth:c2.T_zeros`, `truth:c2.v4_*` |
| Terms and block sums are truncated to 13 fraction bits (alignment, carry, sign-magnitude truncation) | `truth:c2.T_shift`, `truth:c2.T_maxcarry`, `truth:c2.T_signmag`, `truth:c2.T_ties` |
| A single 16-term block (K = 16) and long chains (K = 2048) follow the same model | `truth:c2.S16_*`, `truth:c2.L_*`, `truth:c2.R_*` |
| The accumulator-only path, over all 2^32 fp32 c patterns | `truth:c2.conly_exhaustive_rerec` |
| The RTX 4060 reproduces the toolbox's recorded Ada and L40S vectors bitwise | `truth:c2.s0_toolbox_vectors_rerec` |
| IEEE specials: every pair of FP8 codes (65,536) through the GEMM epilogue, NaN and Inf rows and columns, NaN under split-K | `truth:c3.gemm_table`, `truth:c3.v1_nan_cublas`, `truth:c3.gemm_struct`, `truth:c5.j7_cublaslt` |
| Mixed E4M3 x E5M2 inputs follow the same model; e5m2 x e5m2 is rejected by torch | `truth:c3.v0_mixed_probe_rerec`, `truth:misc.splitk_probe_c1_6`, `truth:misc.e5m2_ranges_va1_4` |
| Split-K assigns 64-wide k-tiles round-robin to the splits | `truth:c5.v0_ownership_probes`, `truth:c5.tree_probes` |
| Split partials combine serially (customOption 0, 2) or pairwise (customOption 1); for S = 3 pairwise is (p2 + p0) + p1 | `truth:c5.v0_reduce_probes`, `truth:misc.reduction_order_c1_0`, `truth:misc.reduction_order3_c1_0`, `truth:c3.v3_order_probe` |
| Forced split configurations with S = 2..32 and all three customOptions | `truth:c5.v2_forced_probes`, `truth:c5.j2_cublaslt`, `truth:c5.j5_cublaslt` |
| Split-K results are deterministic under scheduling perturbation | `truth:c5.det1_determinism`, `truth:c5.det2_determinism` |
| The output dtype and the workspace size change the split decision for the same inputs; workspace 0 never splits | `truth:misc.outdtype_splitk_va1_0`, `truth:c5.v1_workspace_env`, `truth:c4.verify3`, `truth:c4.V5WS0` |
| Two of the prototype's four recorded test cases split: at 16 x 128 x 8192 (e4m3 and e5m2 x e4m3 inputs) S = 8 with fast accumulation or fp32 output and S = 12 for bf16 output without fast accumulation, all pairwise; the two 64 x 256 x 1920 cases do not split | `truth:misc.tcemu_tests_recorded_hw` |
| The split-K table: cuBLASLt 12.8.3.14 heuristic choices, 906,600 queries over 17,476 shapes | `truth:heur.lt_heuristic_12_8_3_14` |
| Epilogue sign of zero: fma(acc, alpha, +0) with negative, zero, subnormal and underflowing alpha | `truth:epilogue.zero_sign_v01b_a1_1`, `truth:epilogue.c1_4_hw_zero_scale`, `truth:epilogue.c1_4_hw_underflow`, `truth:c3.v0_kern_zero_sign_rerec` |
| bf16/fp16 outputs are rounded from the fp32 value (double rounding), shown on 240 constructed ties | `truth:epilogue.tie_cases_a3_7`, `truth:epilogue.ties_double_rounding_c1_5` |
| fp8 outputs: scale_result is a separate fp32 multiply before saturation to the format's maximum | `truth:epilogue.fp8out_v01_a1_2`, `truth:epilogue.fp8out_v03_a1_2`, `truth:c4.verify4_fp8out`, `truth:c3.gemm_fp8out` |
| NaN outputs are canonical; results do not depend on the out= buffer | `truth:epilogue.zero_nan_v2_a1_5`, `truth:epilogue.zero_specials_a3_7`, `truth:epilogue.c1_4_hw_out_buffer`, `truth:c3.v0_epi_prefill` |
| torch's defaults: out_dtype = the input dtype, use_fast_accum = False | `truth:misc.api_defaults_va1_3`, `truth:c4.v5_rerec` |
| Rejections: the bias dtype checks of torch 2.7.1 (14 recorded errors) | `truth:epilogue.zero_sign_v01b_a1_1` |
| Rejections: cuBLAS refuses e4m3 x e4m3 -> e5m2 on the RTX 4060 (CUBLAS_STATUS_NOT_SUPPORTED) | `truth:c3.gemm_fp8out`, `truth:epilogue.fp8out_v01_a1_2` |
| The 21 FP8 projections of an Evo 2 1B window never split; native bf16 and TE outputs are equal on all of them | `truth:evo2.evo2_1b_w2k` |

The H100 PCIe probe's claims rest on its committed records (`log:docs/validation_logs/2026-10-15/h100-probe/README.md`), not on the ground truth:

| Claim (H100 PCIe) | Records |
|---|---|
| cuBLAS FP8 GEMMs follow the `hopper` model: 32-product blocks, 13 fraction bits, c entering the first block's alignment; `MM_cublas64` 445 of 445, every stored n x n block. The sm_90 `mma.sync` instruction is a different unit (25 fraction bits) | `log:docs/validation_logs/2026-10-15/h100-probe/results/analysis/m1.json`, `log:docs/validation_logs/2026-10-15/h100-probe/results/analysis/edge_variants.log`, `log:docs/validation_logs/2026-10-15/h100-probe/results/analysis/tie_check.log` |
| Slow accumulation restarts every 128 elements and adds the chunk results in K order | `log:docs/validation_logs/2026-10-15/h100-probe/results/analysis/m3.json`, `log:docs/validation_logs/2026-10-15/h100-probe/results/analysis/m4_hopper_R128.json` |
| The 21 Evo 2 1B projections, fp32 accumulator and bf16 output, fast and slow, torch and TE: 168 of 168 full outputs; none splits K | `log:docs/validation_logs/2026-10-15/h100-probe/results/analysis/m2_hopper_R128.json` |
| Split-K: contiguous runs of 128-wide k-tiles, partials added serially, every customOption; of 64 hypotheses for fast and 48 for slow accumulation the only one that fits every live split call of its mode (517 fast, 532 slow; fitted after the data came in) | `log:docs/validation_logs/2026-10-15/h100-probe/results/analysis/splitk_search.log`, `log:docs/validation_logs/2026-10-15/h100-probe/results/analysis/splitk_search.json.gz` |
| Epilogue: RN(RN32(acc * RN32(sa * sb))) in all 64 tie cases; the 150 zeros / specials rows | `log:docs/validation_logs/2026-10-15/h100-probe/results/m5/epilogue_ties.log`, `log:docs/validation_logs/2026-10-15/h100-probe/results/m5/zero_specials.json` |
| Rejections: cuBLASLt refuses e4m3 x e4m3 -> e5m2 (status 15): the 27 live jobs, and the heuristic in all 56,688 of its queries of that combination and in no other | `log:docs/validation_logs/2026-10-15/h100-probe/README.md` |
| Every call gave the same bits twice (1,450 of 1,450 live jobs across processes) | `log:docs/validation_logs/2026-10-15/h100-probe/README.md` |

## mmaemu's own results

The recordings above are what mmaemu must reproduce. The results of replaying them through mmaemu v0.1 are
committed as logs by the test tiers (docs/testing.md in the repository describes the tiers):

- T1 runs on every CI push: the committed subset of the recordings (tests/data/hw_mini), through the reference
  backend.
- T2 is the full replay of the ground truth through mmaemu, compared recording by recording with the audit's
  reference replay; its report goes to docs/validation_logs/.
- T3 and T3p are the GPU tiers: every available kernel against the reference, and live `torch._scaled_mm` calls
  against mmaemu on the GPU of the profiles `ada`, `ada-te` and `ada-ws0` (the H100 profiles have no live tier).
- T4 is the mutation matrix: deliberately broken copies of the model must fail the tests.
- T7 validates the built wheel in a fresh environment on the GPU; its report goes to docs/release_logs/.

The code of v0.1.0 is `mmaemu/` as in 63b7395, the profile rename (above), with the fatbins of the release-candidate
freeze (7612fd8, `log:docs/validation_logs/2026-10-16/rc-fatbin/README.md`), and `tests/` as in 63b7395 apart from
the skip that 59b37e1 added to `tests/test_h100_mini.py`: the module skips itself in a tree without `tools/`, the
sdist (`log:docs/validation_logs/2026-10-17/sdist-tests/README.md`), and in a git checkout runs as before.
`tools/` is that of 63b7395 apart from `tools/make_figures.py`, which relabels the figures and adds the `hopper-te`
runs and the 40B to the upstream-test figure. Later commits change only that script, that skip, the default profile of
`examples/evo2_offload.py` (`hopper-te`) and its docstring, the outputs stored in `examples/evo2_blackwell.ipynb` and
the install line of its setup cell (and of `examples/build_evo2_blackwell.py`), the package description and keywords
in `pyproject.toml`, documentation (CITATION.cff, context7.json, llms.txt and mkdocs.yml included) and logs. The
rows of this table marked fa0fd90 ran on the tree before the rename. Each of them ran again on the code of 63b7395
(2648bba) and gave the same result once the old profile names are read as the new ones: on the
RTX 4060, the bench's rates within 0.01 TMAC/s (`log:docs/validation_logs/2026-10-16/rtx-4060-sm89-renamed/README.md`),
and on the RTX 5070 Ti (`log:docs/validation_logs/2026-10-16/rtx-5070-ti-renamed/README.md`). The sources and
fatbins of v1, v4, v9, tc3 and the probes have not changed since the first freeze (e173ecc). h1 and h4, the `hopper`
preset's kernels, were added after it, each with its own
release build (`log:docs/validation_logs/2026-10-15/h1-fatbin/README.md`, `log:docs/validation_logs/2026-10-16/h4-fatbin/README.md`),
and the release-candidate freeze (7612fd8) rebuilt all seven modules. Commits before fa0fd90 changed Python code,
tests and docs, so each row names the tree it ran on. The table lists T7 on the wheel built from 59b37e1
and the tests of its sdist first, then the CI runs, then the runs on fa0fd90, then the runs on earlier trees that the release still
rests on. T6 (da9edd4,
`log:docs/validation_logs/2026-10-14/t6-rtx-4060-sm89/README.md`), T3 on the H100 PCIe (69cd231) and the Evo 2 fixed
inputs on the H100 (5c15e8a, both `log:docs/validation_logs/2026-10-14/t3-h100-sm90/README.md`) ran before the H100
PCIe work changed `mmaemu/` (from 2bec68a on, the default profile included). Other runs on earlier trees, which the
runs on fa0fd90 repeat, are kept under "Earlier trees" below.

| Tier | Result | Log |
|---|---|---|
| T7 on the RTX 4060: the wheel `mmaemu-0.1.0-py3-none-any.whl` (sha256 f9725a73...) built from 59b37e1, the code of 0.1.0 (fatbins h1, h4, v1, v4, v9, tc3 and the probes) | PASS, 8 of 8: a fresh venv without nvcc; `info` (`ada`, `ada-te` and `ada-ws0` match the device) and `selftest`; all six kernels loaded from the wheel's fatbins (the images its manifest records); 1,346 conformance comparisons, each kernel against the CPU reference of its own preset (14,986,730 outputs), 0 differing; the Evo 2 1B projection (2048 x 5760 x 1920) through `backend="auto"`: under `ada` (on tc3) equal to the RTX 4060 recording bit for bit (bf16 output and fp32 accumulator), and under `hopper` (on h4) its bf16 and fp32 outputs, with fast and with slow accumulation, have the sha256 of the H100 PCIe's | `log:docs/release_logs/0.1.0_rtx4060-sm89.json` |
| T7 on an RTX 5070 Ti (sm_120), the same wheel | PASS, 8 of 8: h1, h4, v1 and v4 from the wheel's fatbins, and v9 when requested explicitly (it is not auto-selected on sm_120; tc3 has no sm_120 binary); 1,122 conformance comparisons (12,494,403 outputs), 0 differing; the Evo 2 projection under `ada` (on v4) equal to the RTX 4060 recording and under `hopper` (on h4) to the H100 PCIe's, bit for bit | `log:docs/release_logs/0.1.0_rtx5070ti-sm120.json` |
| T0 + T1 from the unpacked sdist `mmaemu-0.1.0.tar.gz` built from 59b37e1 (no `tools/` or `docs/`), Python 3.12, torch 2.7.1 | PASS: 3,363 passed, 15 skipped, each skip for a file the sdist does not ship, a git checkout or an installed distribution (and one left to the registry) | `log:docs/validation_logs/2026-10-17/sdist-tests/README.md` |
| CI (T0 + T1, lint, NVRTC, dist) in GitHub Actions, Python 3.10-3.13 x torch 2.6.0, 2.7.1 and 2.14.1, e386606 (snapshot 201441a), and before it on be2af24 and 414b7e3 | success in all 15 jobs of each run, with the same counts: T0 + T1 3,449 to 3,452 passed per job, 44 NVRTC compiles, and T0 + T1 against the built wheel 3,454 passed | `log:docs/validation_logs/2026-10-17/ci/README.md` |
| The same CI steps run locally (`tools/ci_local.sh`), Python 3.12, 59b37e1 | PASS, every step: T0 + T1 3,497 to 3,499 passed, NVRTC 44 passed, `twine check --strict` and `check_dist --require-fatbins` pass, T0 + T1 against the wheel 3,501 passed; the same counts as on 6980b11, before the skip | `log:docs/validation_logs/2026-10-17/ci-local/README.md` |
| T2, full truth replay (CPU), fa0fd90 | PASS: `compare.json` equal in every field to that of the run on 49f1ecf (below): 173 of 173 recordings in the report; 942,445,814 of 942,451,934 values match; 0 zero-sign and 0 NaN-payload differences; the 6,120 value differences all lie at the old emulator recording's NaN-decode outputs, where mmaemu gives the canonical NaN; hash-only 2,251 of 2,251; 113 of 193 hardware errors reproduced, as in the audit's replay; 2,466 outputs not applicable; fastacc cross-check 0 mismatches in 512,512 values. The same replay on 7612fd8 gave the same `compare.json` | `log:docs/validation_logs/2026-10-16/t2/compare.txt`, `log:docs/validation_logs/2026-10-16/t2/README.md` |
| T3 on the RTX 4060 (sm_89, the profile device), release fatbins, fa0fd90 | every per-file run exits 0 except `test_backend_selection`, which exits 5 (nothing collected: a CPU-only file, run in T0): scaffold 2 passed, backend reference 3 passed, epilogue 5 passed, IEEE kernels 167 passed (v1, v4, h1 and h4, each against the CPU reference of its own preset), CUDA runtime 19 passed, fast kernels 182 passed (v9 and tc3, with the audit's verify2 and p2 cases), gates 4 passed, live `torch._scaled_mm` 2 passed, hardware-mini 5 passed and 1 skipped, device independence 31 passed and 107 skipped, H100 mini 310 passed (its GPU cases: the 124 live H100 calls through the reference backend on the GPU and through `auto`, which runs h4, and the E4M3 x E4M3 ones through h1; `-m gpu` deselects its 125 CPU cases). Preflight 18 of 18: driver 570.211.01, torch 2.7.1+cu128, cuBLASLt 12.8.3.14, the packages of the reference lock, fatbins equal to the manifest's sha256 | `log:docs/validation_logs/2026-10-16/rtx-4060-sm89/t3/summary.txt`, `log:docs/validation_logs/2026-10-16/rtx-4060-sm89/README.md`, `log:docs/validation_logs/2026-10-16/rtx-4060-sm89/preflight.out` |
| h1 and h4 on the RTX 4060: their four test files on the package fatbins, fa0fd90 | IEEE kernels 167 passed (no marker filter), CUDA runtime 19 passed, H100 mini 435 passed, backend selection 26 passed, the counts of h4's RTX 5070 Ti run (below). With the probe replays below, this is the evidence for sm_89 in `validated_on` of h1 and h4 ({sm_89, sm_120}); `python -m mmaemu info` lists both as "available via fatbin, validated here". A first run of this step without `MMAEMU_TEST_FATBIN_DIR` skipped the fatbin tests; these counts are from its rerun with the variable set | `log:docs/validation_logs/2026-10-16/rtx-4060-sm89/hopper-kernels/summary.txt`, `log:docs/validation_logs/2026-10-16/rtx-4060-sm89/mmaemu_info.txt`, `log:docs/validation_logs/2026-10-16/rtx-4060-sm89/README.md` |
| Full accumulators on the RTX 4060 (`MMAEMU_DI_FULL=cuda:v1,cuda:v4,cuda:v9,cuda:tc3`), fa0fd90 | 84 passed: the full-accumulator sha256 of all 21 Evo 2 1B projections equals the hardware's for each of v1, v4, v9 and tc3 | `log:docs/validation_logs/2026-10-16/rtx-4060-sm89/t3/di_full_gpu_kernels.log` |
| T3p on the RTX 4060, fa0fd90 | 6 passed, the six tests of the run on 49f1ecf (below): live `torch._scaled_mm` parity with mmaemu's reference backend (278 calls, 0 failures), the `out=` and operand layouts (112 calls, 0 failures), the profile match, 500 split-K table rows re-queried from cuBLASLt (0 differ), the live query step, and 11,760 live queries above the bound where mmaemu infers S = 1 (all S = 1) | `log:docs/validation_logs/2026-10-16/rtx-4060-sm89/t3p_profile_device.log`, `log:docs/validation_logs/2026-10-16/rtx-4060-sm89/README.md` |
| T3 on an RTX 5070 Ti (sm_120), release fatbins, fa0fd90 (`--di-full`) | every per-file run exits 0 except `test_backend_selection`, which exits 5 (nothing collected): scaffold 2 passed, backend reference 3 passed, epilogue 5 passed, IEEE kernels 167 passed, CUDA runtime 19 passed, fast kernels 91 passed and 91 skipped (no tc3 binary for sm_120), gates 1 passed and 3 skipped, live `torch._scaled_mm` 2 skipped (both need the profile device), hardware-mini 3 passed and 3 skipped, device independence 27 passed and 111 skipped, H100 mini 310 passed (its GPU cases; `auto` runs h4 here). Full accumulators: the reference backend, v1 and v4 reproduce the RTX 4060's full-accumulator sha256 on all 21 calls each; v9 (gated, sm_120 not in its `validated_on`) and tc3 (no sm_120 binary) skip | `log:docs/validation_logs/2026-10-16/t3-rtx-5070-ti-sm120/summary.txt`, `log:docs/validation_logs/2026-10-16/t3-rtx-5070-ti-sm120/README.md` |
| T4, CPU tier, fa0fd90 | PASS: 43 killed, 2 equivalent, the 5 identity controls survive; every mutant has its outcome of 2026-10-14. On 7612fd8 the tier failed: the identity control `identity_execute` was killed by `tests/test_h100_mini.py` (a contiguous split), because the mutation tool's own re-implementation of `execute_plan` combined contiguous splits pairwise, where mmaemu combines them serially, and had no +0 partial for a contiguous split without k-tiles. 3fe4e9c fixes the tool and changes nothing under `mmaemu/` or `tests/` | `log:docs/validation_logs/2026-10-16/t4/mutants-cpu.json`, `log:docs/validation_logs/2026-10-16/t4/README.md`, `log:docs/validation_logs/2026-10-16/t4/mutants-cpu-7612fd8.json` |
| T4, GPU tier, RTX 5070 Ti and RTX 4060, fa0fd90 | PASS on both: 7 killed, the identity control survives | `log:docs/validation_logs/2026-10-16/t4/mutants-gpu.json`, `log:docs/validation_logs/2026-10-16/t4/mutants-gpu-rtx4060.json` |
| The H100 PCIe probe replayed through h4 (`tools/replay_h100_log.py --backend auto`), on an RTX 5070 Ti and on the RTX 4060, fa0fd90 | PASS on both GPUs: the 1,043 E4M3 x E4M3 live calls ran on h4 and the 380 mixed-format ones on the reference backend; all 1,423 equal the H100 (1,049 split, 374 unsplit), and the 27 calls the H100 rejected raise `ReferenceRejectsError`; the 168 Evo 2 outputs ran on h4, 168 of 168 equal. `splitk="auto"` agrees with the H100 on 1,353 calls and raises `SplitKUnknownError` on 70; 12 that raised at ebd18e1 (below) are now inferred S = 1 by the H100 envelope and agree with the logged decision. The live records of the two GPUs are equal apart from their timings | `log:docs/validation_logs/2026-10-16/rtx-5070-ti-replays/probe/mmaemu_replay.json`, `log:docs/validation_logs/2026-10-16/rtx-5070-ti-replays/README.md`, `log:docs/validation_logs/2026-10-16/rtx-4060-sm89/replay-h4/mmaemu_replay.json` |
| The H100 PCIe split-K sweep's 2,744 live jobs replayed through h4 (`--backend auto`), on an RTX 5070 Ti, fa0fd90 | PASS: all 2,744 equal the H100 (2,416 split, 328 unsplit), 994 on h4 and 1,750 on the reference backend; every verdict, backend, split count and scheme equals that of the replay of d1ac890 (the out-of-sample one, in the Scope table). `splitk="auto"` agrees with the H100 on 480 calls (248 at d1ac890) and raises `SplitKUnknownError` on 2,264; none disagrees. The added agreements come from the sweep's own rows and envelope, shipped in bdc4bdf after d1ac890 | `log:docs/validation_logs/2026-10-16/rtx-5070-ti-replays/sweep/mmaemu_replay_p5.json`, `log:docs/validation_logs/2026-10-16/rtx-5070-ti-replays/README.md` |
| The same probe replayed through h1 (`--backend cuda:h1`, h1 forced for every call), on the RTX 4060, fa0fd90 | the 1,043 E4M3 x E4M3 live calls ran on h1 and are equal (746 split, 297 unsplit), the 27 calls the H100 rejected raise as there, and the 168 Evo 2 outputs ran on h1, 168 of 168 equal: on these every verdict equals that of the RTX 5070 Ti's h1 replay (3d7dc66, below). The 380 mixed-format calls raise `BackendUnavailableError` (h1 implements E4M3 x E4M3 only), so the tool's overall verdict is false by construction; the same 380 calls are equal on the reference backend in the h4 replay | `log:docs/validation_logs/2026-10-16/rtx-4060-sm89/replay-h1/mmaemu_replay.json`, `log:docs/validation_logs/2026-10-16/rtx-4060-sm89/README.md` |
| Split-K envelope sweeps on the RTX 4060 (the three scripts of 2026-10-14, unchanged), fa0fd90 | per-query rows equal to those of the run on 49f1ecf row for row, which equal those of 2026-10-14: 306,859, 663,936 and 666,624 rows; above the setting's largest split M*N, 8,292 of 663,936 and 274 of 666,624 queries split, none above 1.271 times it | `log:docs/validation_logs/2026-10-16/rtx-4060-sm89/sweeps/sweep_env2.out`, `log:docs/validation_logs/2026-10-16/rtx-4060-sm89/sweeps/sweep_env3.out`, `log:docs/validation_logs/2026-10-16/rtx-4060-sm89/README.md`, `log:docs/validation_logs/2026-10-14/splitk-envelope/README.md` |
| Bench on the RTX 4060, release fatbins, fa0fd90 (outside T6: M = 2048, Gaussian data) | TMAC/s on the Evo 2 1B, 7B and 40B shapes: h4 1.35, 1.41 and 1.41, h1 0.69 on all three; tc3 2.94-2.95, v9 1.54-1.55, v4 1.35-1.37 and v1 0.63, within 1% of T6. Exactness before timing, at all three shapes: h1 equal to the CPU reference of the `hopper` preset on its first 64 rows x 256 columns and h4 equal to h1 on every output (likewise v1 against the CPU reference of `ada`, and v4, v9 and tc3 against v1) | `log:docs/validation_logs/2026-10-16/rtx-4060-sm89/README.md`, `log:docs/validation_logs/2026-10-16/rtx-4060-sm89/bench-1b-gauss40.json` |
| T3 on an H100 PCIe (sm_90), release fatbins, 69cd231 | IEEE kernels 79 passed, CUDA runtime 13 passed, hardware-mini 3 passed (reference, v1, v4), fast kernels 12 passed (no tc3 binary for sm_90); device independence not run (no ground truth on the rented host); T3p: profile device unavailable | `log:docs/validation_logs/2026-10-14/t3-h100-sm90/summary.txt`, `log:docs/validation_logs/2026-10-14/t3-h100-sm90/README.md` |
| Evo 2 fixed inputs on the H100 (sm_90), NVRTC build of 5c15e8a (v1/v4 sources equal to the RC apart from comments) | v1 and v4: all 21 fp32 accumulators and bf16 outputs bitwise equal to the RTX 4060 recording; native `torch._scaled_mm` differs on all 21. sm_90 stays out of `validated_on`: the full device-independence suite did not run there | `log:docs/validation_logs/2026-10-14/t3-h100-sm90/fixed-fp8.json` |
| The H100 PCIe probe replayed through mmaemu (`tools/replay_h100_log.py`, reference backend on an RTX 5070 Ti), b3c4c8a | PASS: all 1,450 live jobs reproduce the H100 with the logged split (1,423 equal by full-output sha256, 1,049 of them split; the 27 e4m3 x e4m3 -> e5m2 jobs raise as on the H100); `splitk="auto"` gives the logged decision on all 1,000 M6 jobs and on 341 M4 jobs, the other 82 M4 jobs are off the table's grid and raise, none disagrees; the 21 Evo 2 1B projections 168 of 168 outputs, S = 1 from the table. CI replays a subset (`tests/test_h100_mini.py`) | `log:docs/validation_logs/2026-10-15/h100-probe/results/analysis/mmaemu_replay.json`, `log:docs/validation_logs/2026-10-15/h100-probe/README.md` |
| The same probe replayed through the CUDA kernel h1 (`--backend auto` on an RTX 5070 Ti), 3d7dc66 | PASS, with the same verdict as the reference-backend replay on every live job and Evo 2 output. h1 ran all 1,043 E4M3 x E4M3 live jobs that ran (746 of them split) and all 168 Evo 2 outputs; the 380 mixed-format jobs ran on the reference backend | `log:docs/validation_logs/2026-10-15/h100-probe/results/analysis/mmaemu_replay_cuda.json` |
| h1 on an RTX 5070 Ti (sm_120), 3d7dc66 | IEEE kernels 124 passed (v1, v4 and h1, each against the CPU reference of its own preset), CUDA runtime 16 passed, H100 mini 373 passed (124 jobs on the CPU, and on the GPU on the reference backend and on `backend="auto"`, which runs every E4M3 x E4M3 job on h1), backend selection 25 passed | `log:docs/validation_logs/2026-10-16/h1-rtx-5070-ti/README.md` |
| h4 on an RTX 5070 Ti (sm_120), ebd18e1 | IEEE kernels 167 passed (43 of them h4; each kernel against the CPU reference of its own preset), CUDA runtime 19 passed, H100 mini 435 passed (on the GPU on the reference backend, on `backend="auto"`, which runs every E4M3 x E4M3 job on h4, and on `cuda:h1`), backend selection 26 passed; h4 equal to h1 over the full output of the three Evo 2 shapes at M = 2048 | `log:docs/validation_logs/2026-10-16/h4-rtx-5070-ti/README.md` |
| T6, performance on the RTX 4060, release fatbins, da9edd4 | PASS: at M = 2048 on the Evo 2 1B, 7B and 40B shapes tc3 2.93-2.94, v9 1.53-1.54, v4 1.34-1.37 and v1 0.62-0.63 TMAC/s, every kernel above its threshold, with Gaussian and adversarial data; native FP8 29.8-30.5 TMAC/s (docs/performance.md). On fa0fd90 the bench above gave v1, v4, v9 and tc3 within 1% of these rates at M = 2048 with Gaussian data | `log:docs/validation_logs/2026-10-14/t6-rtx-4060-sm89/README.md` |
| Release fatbin build, e173ecc | nvcc 12.8.61, 5 modules (v1, v4, v9, tc3, probes), 0 spill bytes | `log:docs/validation_logs/2026-10-14/fatbin-release/build_report.json` |
| Release builds of h1 and h4, one module each | nvcc 12.8.61, 0 spill bytes; a second build of each gave a byte-identical fatbin | `log:docs/validation_logs/2026-10-15/h1-fatbin/README.md`, `log:docs/validation_logs/2026-10-16/h4-fatbin/README.md` |
| Release fatbin rebuild at the 0.1.0 freeze (7612fd8): the fatbins of fa0fd90 | all 7 modules, nvcc 12.8.61, 0 spill bytes: h1, h4, v9, tc3 and the probes byte-identical to the committed fatbins; v1 and v4 differ only in the hash in the names of anonymous-namespace symbols in the PTX, with identical SASS, so their committed bytes were kept | `log:docs/validation_logs/2026-10-16/rc-fatbin/README.md`, `log:docs/validation_logs/2026-10-16/rc-fatbin/v1_v4_sass_compare.txt` |

The GPU development validation is recorded in
[the sm_120 report](validation_logs/2026-10-13-sm120/README.md), including CPU/GPU
tests, mutations, Evo2 and a wheel without nvcc. This is separate from the full
release checklist in [release.md](release.md).

### Earlier trees

The runs on fa0fd90 above repeat the runs below, which are kept as history. All of them but the last ran before the
H100 PCIe work changed `mmaemu/` (from 2bec68a on, the default profile included); the last is the h4 replay on ebd18e1
that the replay on fa0fd90 repeats. The review of the release candidate
(8429385, `log:docs/validation_logs/2026-10-14/audit-probes/README.md`) changed Python code under `mmaemu/` (49f1ecf);
T2, T3 and T3p ran again on that tree (`log:docs/validation_logs/2026-10-15/t2/README.md`,
`log:docs/validation_logs/2026-10-15/rtx-4060-sm89/README.md`).

| Tier | Result | Log |
|---|---|---|
| T2, full truth replay (CPU), 49f1ecf | PASS: all 173 recordings in the report, 147 of them with outputs that mmaemu models (the 15 tables are listed without comparison; 11 bias, row-wise, fp16-accumulator or HMMA recordings have none); 0 zero-sign and 0 NaN-payload differences; the 6,120 value differences all lie at the old emulator recording's NaN-decode outputs, where mmaemu gives the canonical NaN (per-output counts equal to the audit's replay); 113 of 193 hardware errors reproduced, as in the audit's replay; the extra not-applicable outputs are the bias and row-wise features that v0.1 rejects. Every count equals the run of 2026-10-14 on 6492713 | `log:docs/validation_logs/2026-10-15/t2/compare.txt`, `log:docs/validation_logs/2026-10-15/t2/README.md`, `log:docs/validation_logs/2026-10-14/t2/README.md` |
| T3 on the RTX 4060 (sm_89, the profile device), release fatbins, 49f1ecf | IEEE kernels 79 passed, CUDA runtime 13 passed, fast kernels 182 passed (v9 and tc3, with the audit's verify2 and p2 cases), gates 4 passed, live `torch._scaled_mm` 2 passed, hardware-mini 5 passed (reference, v1, v4, v9, tc3), device independence 31 passed (rows 0..255 of the 21 Evo 2 1B accumulators on every backend, the full-accumulator sha256 of all 21 for tc3). On da9edd4 (2026-10-14) the same counts, with 1 live test; a first run on 76fd4ea failed two test expectations written for devices without the gated kernels, which da9edd4 corrects (nothing under `mmaemu/` changed) | `log:docs/validation_logs/2026-10-15/rtx-4060-sm89/t3/summary.txt`, `log:docs/validation_logs/2026-10-15/rtx-4060-sm89/README.md`, `log:docs/validation_logs/2026-10-14/t3-rtx-4060-sm89/README.md`, `log:docs/validation_logs/2026-10-14/rtx4060-host/README.md` |
| Full accumulators on the RTX 4060 (`MMAEMU_DI_FULL`), da9edd4 | 84 passed: the full-accumulator sha256 of all 21 Evo 2 1B projections equals the hardware's for each of v1, v4, v9 and tc3 | `log:docs/validation_logs/2026-10-14/t3-rtx-4060-sm89/di_full_gpu_kernels.log` |
| T3p on the RTX 4060, 49f1ecf | 6 passed: live `torch._scaled_mm` parity (278 calls), the `out=` and mat1 layouts and empty calls (112 calls), the profile match, 500 split-K table rows re-queried from cuBLASLt, the live query step, and 11,760 live queries above the bound where mmaemu infers S = 1, all S = 1. On da9edd4 the four tests of that tree passed | `log:docs/validation_logs/2026-10-15/rtx-4060-sm89/t3p_profile_device.log`, `log:docs/validation_logs/2026-10-14/t3-rtx-4060-sm89/t3p_profile_device.log` |
| T3 on an RTX 5070 Ti (sm_120), release fatbins, e173ecc | IEEE kernels 79 passed, CUDA runtime 13 passed, fast kernels 91 passed (no tc3 binary for sm_120), device independence 27 passed; T3p: profile device unavailable | `log:docs/validation_logs/2026-10-14/t3-rtx-5070-ti-sm120/summary.txt` |
| T4, CPU tier, 5f1a55c | PASS: 43 killed, 2 equivalent, the 5 identity controls survive | `log:docs/validation_logs/2026-10-14/t4/mutants-cpu.json` |
| T4, GPU tier (RTX 5070 Ti), 5f1a55c | PASS: 7 killed, the identity control survives | `log:docs/validation_logs/2026-10-14/t4/mutants-gpu.json` |
| T4, GPU tier (RTX 4060), da9edd4 | PASS: 7 killed, the identity control survives | `log:docs/validation_logs/2026-10-14/t4/mutants-gpu-rtx4060.json` |
| The H100 PCIe probe replayed through the CUDA kernel h4 (`--backend auto` on an RTX 5070 Ti), ebd18e1 | PASS, with the same verdict as the reference-backend and h1 replays above on every live job and Evo 2 output. h4 ran all 1,043 E4M3 x E4M3 live jobs that ran and all 168 Evo 2 outputs | `log:docs/validation_logs/2026-10-15/h100-probe/results/analysis/mmaemu_replay_cuda_h4.json` |

## Evo 2 1B end to end

Model-level measurements of the 1B model with mmaemu's H100 PCIe profile `hopper-te` (first row, the scripts of the 40B
section: `log:docs/validation_logs/2026-10-16/evo2-h100-profile-rtx-5070-ti/compare.json`) and with its RTX 4060
profile `ada-te` (the Evo 2 example, `repo:examples/evo2_blackwell.ipynb`), both on the RTX 5070 Ti, and of the
official path on two GPUs. They are not part of the bitwise validation above: only the FP8 GEMMs are emulated (with
`hopper-te` they give the H100's bits, as verified above for the projections of one 1B window recorded on the H100),
and the rest of the model runs natively, so the outputs are close to the official ones, not equal to them
(docs/evo2.md explains the setup).

| Measurement | Result | Evidence |
|---|---|---|
| Evo 2 1B with `hopper-te` on the RTX 5070 Ti, fa0fd90 (`hopper-h100pcie-te` in the log): the 40B section's scripts with this mode added; on seven windows their logits equal those of the model held on the GPU bit for bit | passes the upstream test: loss 0.5022, the H100's value in bf16 (fp32 0.5024; H100 official 0.5026, `ada-te` 0.5025), accuracy 0.7961; the logits are not the H100's bit for bit on any of the seven windows. Six windows, mean \|Δ log p(A,C,G,T)\| against the RTX 4060: 0.0407 (`ada-te` 0.0403). BRCA1, 500 variants: AUROC 0.7354 (H100 official 0.7332, `ada-te` 0.7323); score changes against the H100's: Pearson 0.988 (`ada-te` 0.986) | `log:docs/validation_logs/2026-10-16/evo2-h100-profile-rtx-5070-ti/compare.json` |
| The RTX 4060 reference shipped with the example (`repo:examples/data/evo2_1b_ada_rtx4060_te.npz`) | the official path (`evo2.Evo2` with TE FP8) captured on the RTX 4060 on 11 and 12 October; every shipped array equals its capture, and each capture's full logits have the SHA-256 its metadata records | `log:docs/validation_logs/2026-10-12/evo2-rtx-4060-official/check_reference.txt` |
| First FP8 projection (block 0, 2,048-bp window) on the RTX 5070 Ti | mmaemu's bf16 output (`ada-te`) has the SHA-256 of the RTX 4060's TE output; natively 8,293,802 of 11,796,480 outputs differ | `log:docs/validation_logs/2026-10-14/evo2-rtx-5070-ti/results.json`, `truth:evo2.evo2_1b_w2k` |
| Upstream test loss (prompts p0-p3, fp32 cross entropy) of the official path | RTX 4060 0.5024, H100 PCIe 0.5026 (the test expects 0.502) | `log:docs/validation_logs/2026-10-14/evo2-h100-sm90/upstream_test.json` |
| Upstream test loss on the RTX 5070 Ti (bf16 cross entropy on the GPU, as the test computes it) | mmaemu (`ada-te`) 0.5027, native FP8 1.2480, bf16 GEMMs 1.3652 | `log:docs/validation_logs/2026-10-14/evo2-rtx-5070-ti/results.json` |
| Six windows, mean \|Δ log p(A,C,G,T)\| against the RTX 4060 | H100 official 0.0411; RTX 5070 Ti: mmaemu (`ada-te`) 0.0403, native FP8 2.58, bf16 GEMMs 2.85 | `log:docs/validation_logs/2026-10-14/evo2-h100-sm90/comparison_vs_ada.json`, `log:docs/validation_logs/2026-10-14/evo2-rtx-5070-ti/results.json` |
| BRCA1 zero-shot AUROC, 500 variants | RTX 4060 official 0.7378, H100 official 0.7332; RTX 5070 Ti: mmaemu (`ada-te`) 0.7323, native FP8 0.5586, bf16 GEMMs 0.4842 | `log:docs/validation_logs/2026-10-14/evo2-h100-sm90/comparison_vs_ada.json`, `log:docs/validation_logs/2026-10-14/evo2-rtx-5070-ti/results.json` |
| BRCA1 score changes, Pearson against the RTX 4060 | H100 official 0.987; RTX 5070 Ti: mmaemu (`ada-te`) 0.986, native FP8 0.059, bf16 GEMMs 0.069 | `log:docs/validation_logs/2026-10-14/evo2-h100-sm90/comparison_vs_ada.json`, `log:docs/validation_logs/2026-10-14/evo2-rtx-5070-ti/results.json` |

## Evo 2 40B end to end

The original 40B checkpoint with mmaemu's H100 PCIe profile `hopper-te`
(`log:docs/validation_logs/2026-10-16/evo2-h100-profile-rtx-5070-ti/compare.json`) and with its RTX 4060 profile
`ada-te` (`log:docs/validation_logs/2026-10-15/evo2-40b-rtx-5070-ti/README.md`), on the RTX 5070 Ti with its weights
in host RAM and one block at a time on the GPU; only the FP8 GEMMs are emulated (docs/evo2.md).

| Measurement | Result | Evidence |
|---|---|---|
| The scripts, on Evo 2 1B, against the model held on the GPU | in all three modes (mmaemu with `ada-te`, native FP8, bf16 GEMMs): the logits of five windows (vortex's forward pass) and the BRCA1 scores of 40 windows (`evo2.scoring.score_sequences`) equal bit for bit; the upstream-test losses equal the 1B log's (0.5027, 1.2480, 1.3652) | `log:docs/validation_logs/2026-10-15/evo2-40b-rtx-5070-ti/check_1b.json`, `log:docs/validation_logs/2026-10-15/evo2-40b-rtx-5070-ti/brca1_check_1b.json` |
| Evo 2 40B with `hopper-te` on the RTX 5070 Ti, the scripts above with this mode added | passes the upstream test: loss 0.2161 in bf16, as with `ada-te` (fp32 0.2158; H100 official 0.2160), accuracy 0.9164; the logits are not the H100's bit for bit on any prompt. Two windows (2,048 and 4,096 bp), mean \|Δ log p(A,C,G,T)\| against the official H100: 0.0289 and 0.0283 (`ada-te` 0.0295 and 0.0288; official H100 vs RTX 4060: 0.030 and 0.029). BRCA1, first 100 variants: AUROC 0.8569 (H100 official 0.855, `ada-te` 0.855); score changes against the H100's: Pearson 0.9986, Spearman 0.991, same sign 99% (`ada-te` 0.9981, 0.986, 99%) | `log:docs/validation_logs/2026-10-16/evo2-h100-profile-rtx-5070-ti/compare.json` |
| Upstream test loss of the 40B on the RTX 5070 Ti (bf16 cross entropy on the GPU, as the test computes it) | mmaemu (`ada-te`) 0.2161 (accuracy 0.9169), native FP8 1.4023 (0.2568), bf16 GEMMs 1.4043 (0.2559); the test expects 0.2159424 (0.91673) | `log:docs/validation_logs/2026-10-15/evo2-40b-rtx-5070-ti/test_40b.json` |
| The 40B reference on the RTX 4060 | the upstream blocks with TE FP8 run one at a time (the official 40B path does not fit in 8 GB); run this way, the 1B gives the logits of the real upstream model bit for bit on all six windows; the 40B log's file equals the captures | `log:docs/validation_logs/2026-10-12/evo2-rtx-4060-official/check_reference.txt` |
| Two windows (2,048 and 4,096 bp), mean \|Δ log p(A,C,G,T)\| against the official path on the RTX 4060 | mmaemu (`ada-te`) 0.031 and 0.029 (most likely base agrees at 99.90% and 99.93%); native FP8 3.97 and 4.07; bf16 GEMMs 3.98 and 4.07 | `log:docs/validation_logs/2026-10-15/evo2-40b-rtx-5070-ti/test_40b.json`, `log:docs/validation_logs/2026-10-15/evo2-40b-rtx-5070-ti/rtx4060_official_te_40b.npz` |
| BRCA1 AUROC of the 40B on the RTX 5070 Ti, the first 100 of the notebook's 500 variants (20 loss-of-function) | mmaemu (`ada-te`) 0.855, native FP8 0.503, bf16 GEMMs 0.590; the 1B with the official path on the RTX 4060, same variants: 0.719 | `log:docs/validation_logs/2026-10-15/evo2-40b-rtx-5070-ti/brca1_40b.json` |
| The official 40B on an H100 PCIe (`log:docs/validation_logs/2026-10-15/evo2-40b-h100-sm90/README.md`), the same block-by-block method; on the 1B there, the logits of 11 windows and 20 BRCA1 scores equal the real upstream model's bit for bit | upstream test 0.2159424 (the expected value; accuracy 0.9167) | `log:docs/validation_logs/2026-10-15/evo2-40b-h100-sm90/results/1b_check.json`, `log:docs/validation_logs/2026-10-15/evo2-40b-h100-sm90/results/40b/test.json` |
| Two windows, mean \|Δ log p(A,C,G,T)\|: H100 against RTX 4060 (both official), and the RTX 5070 Ti against the H100 | official H100 vs RTX 4060: 0.030 and 0.029; mmaemu (`ada-te`) vs H100: 0.0295 and 0.0288; native FP8 vs H100: 3.97 and 4.07 | `log:docs/validation_logs/2026-10-15/evo2-40b-h100-sm90/compare.json` |
| BRCA1 against the official 40B on the H100 | first 100 variants: H100 AUROC 0.855, the same as mmaemu's with `ada-te`; mmaemu's (`ada-te`) per-variant score changes vs the H100's: Pearson 0.998, Spearman 0.986, same sign 99% (native FP8: 0.17, 0.12, 55%); all 500 variants: H100 AUROC 0.868 | `log:docs/validation_logs/2026-10-15/evo2-40b-h100-sm90/compare.json` |

## Hardware recordings by group

Generated from the truth manifest by `python tools/check_docs.py --fix`; the check fails if it is out of date.
"Outputs" counts recorded output arrays (including hash-only ones and hardware errors), "output values" their
elements.

<!-- truth-inventory:begin (generated by tools/check_docs.py --fix) -->
| Group | Kinds | Levels | Recordings | Recorded outputs | Output values | Evidence |
|---|---|---|---:|---:|---:|---|
| c2 | gemm, mma, table | mma.sync, torch._scaled_mm | 45 | 576 | 192,911,137 | `truth:c2.*` |
| c3 | gemm, mma | mma.sync, torch._scaled_mm | 17 | 2,247 | 254,550,651 | `truth:c3.*` |
| c4 | gemm, table | torch._scaled_mm, transformer_engine | 29 | 16,433 | 1,204,289,600 | `truth:c4.*` |
| c5 | gemm, table | cublasLt, cublasLt.heuristic, torch._scaled_mm, transformer_engine | 25 | 2,390 | 108,642,122 | `truth:c5.*` |
| c6 | gemm, mma | mma.sync, torch._scaled_mm | 9 | 21 | 29,860 | `truth:c6.*` |
| emulator | gemm | tcemu CUDA emulation kernel | 1 | 24 | 10,031,088 | `truth:emulator.*` |
| epilogue | gemm, mma | mma.sync, torch._scaled_mm | 17 | 2,441 | 13,146,670 | `truth:epilogue.*` |
| evo2 | gemm | torch._scaled_mm | 1 | 102 | 614,891,520 | `truth:evo2.*` |
| heur | table | cublasLt.heuristic | 2 | 0 | 0 | `truth:heur.*` |
| hmma | mma | mma.sync | 7 | 69 | 79,927,040 | `truth:hmma.*` |
| misc | gemm | torch._scaled_mm | 20 | 5,830 | 22,397,184 | `truth:misc.*` |
| all | | | 173 | 30,133 | 2,500,816,872 | `truth:*` |
<!-- truth-inventory:end -->
