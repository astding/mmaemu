# Validation evidence

mmaemu's oracle is hardware. Every numerics claim in the documentation rests on outputs recorded on the reference
GPU, collected in one versioned, read-only dataset (the "hardware ground truth", `truth:*`). A behaviour ships as
*validated* only if those recordings reproduce it bit for bit; otherwise it ships as *experimental* (with a
warning) or raises.

**Evidence references.** Every line of this page that states a number carries at least one reference, and
`tools/check_docs.py` checks that each reference resolves:

- `truth:<id>`: a recording of the hardware ground truth (a glob such as `truth:c2.*` names a group);
- `repo:<path>`: a file of this repository;
- `log:<path>`: a committed report under docs/validation_logs/ or docs/release_logs/.

The ground truth is pinned by the sha256 of its manifest, be1270ed64432a1311ab5e2c98ccb69028f72e3424ad4c6148d4cd468fba6695 (`truth:*`).
Each recording stores its inputs, the hardware's output bits (or their sha256), the call parameters, the cuBLASLt
configuration where known, and the hardware and software versions.

## Reference platform

| Item | Value | Evidence |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4060, Ada, sm_89, 24 SMs, driver 572.16, WSL2 | `truth:*` |
| CUDA, cuBLAS, cuBLASLt | CUDA 12.8 (nvcc 12.8.61), cuBLAS and cuBLASLt 12.8.3.14 | `truth:*` |
| torch | 2.7.1+cu128 | `truth:*` |
| Transformer Engine | 2.3.0, where a recording says so | `truth:c4.v3_te_linear`, `truth:c5.v4_te_demo`, `truth:evo2.evo2_1b_w2k` |

## Scope (the tcemu audit's scope table, updated for v0.1)

One row per row of the audit's scope table.

| Item | v0.1 status | Hardware evidence | Not covered |
|---|---|---|---|
| Ada instruction model | validated, bitwise (preset `ada`), all four FP8 format pairs; a block result that is exactly zero or truncates to zero is +0 | `truth:c2.*`, `truth:c6.kit_MM_ada_sm89`, `truth:c2.S16_zeros` | other Ada GPUs (expected identical, not measured) |
| Hopper preset | experimental: instruction level with c = 0 only; slow accumulation raises | the toolbox's author-recorded H100/H200 vectors (repository test data); the same inputs measured on the RTX 4060: `truth:c2.s0_toolbox_vectors_rerec`, `truth:c6.kit_TBX_H100c0_sm89` | how c enters the sum; GEMM level; slow accumulation |
| b200_mma_sync preset (alias blackwell) | experimental: fits the toolbox's B200 `mma.sync` (HMMA-rewrite) vectors; the model is not identified | the toolbox's B200 vectors (repository test data); the same inputs and discriminating probes on the RTX 4060: `truth:c6.kit_TBX_B200_sm89`, `truth:c6.interleave_probe_kit_sm89` | everything about sm_120; B200 cuBLAS (native FP8 path) |
| cuBLAS GEMM, fast accumulation, unsplit | validated, bitwise, tensor-wise scales | `truth:c4.V1`, `truth:c4.V6`, `truth:c4.V5WS0`, `truth:c2.v7_gemm_subnormal`, `truth:misc.random_run*` | other GPUs, SM counts and cuBLAS versions |
| Slow accumulation | validated: 128-element chunks when unsplit, per split when split | `truth:misc.slow_accum_sweep_a1_s05`, `truth:c4.V7`, `truth:c5.v3_slow_heuristic`, `truth:c3.v3_slow_splitk` | non-`ada` presets (raise) |
| Split-K | validated where the profile's table, workspace 0 or an explicit (S, scheme) decides; interleaved 64-wide k-tiles, serial and pairwise combination. Not validated decisions, and shown as such by `explain` (source `live-query` or `envelope`): on the reference GPU a live cuBLASLt query decides shapes missing from the table; elsewhere S = 1 is an inference, made only above twice the setting's largest split M*N (in 1,512,000 live queries the RTX 4060 split no shape above 1.27 times it) | `truth:c5.*`, `truth:c4.verify2`, `truth:misc.tcemu_tests_recorded_hw`, `truth:c3.v3_order_probe`, `truth:heur.lt_heuristic_12_8_3_14`, `log:docs/validation_logs/2026-10-14/splitk-envelope/README.md` | other GPUs; shapes missing from the table at or below that bound, off the reference GPU (raise); S > 1 on the TE profile (experimental) |
| Epilogue | validated: alpha = fp32(scale_a * scale_b), fma(acc, alpha, +0), double rounding to bf16/fp16, fp8 output with scale_result and saturation, canonical NaN, independence from the out= buffer | `truth:epilogue.*`, `truth:c3.gemm_boundary`, `truth:c3.gemm_fp8out`, `truth:c4.verify4_fp8out`, `truth:c3.v0_epi_prefill` | bias (raises) |
| Row-wise scales | not supported in v0.1 (UnsupportedFeatureError); recorded for later work | `truth:c4.V2R`, `truth:c4.verify6_rowwise`, `truth:misc.rowwise_c1_8` | the CUTLASS stream-K path, whose result depends on the SM count |
| CUDA kernels | v1 and v4 (IEEE-exact) and the gated v9 and tc3 (validated on sm_89) must equal the reference bitwise in the GPU tier; NaN inputs, misaligned pointers and second contexts are handled | the prototype kernels' outputs on fixed inputs, kept for device-independence checks: `truth:emulator.tcemu_kernels_fixed_inputs` | v1 and v4: GPUs other than sm_89 and sm_120 (self-test gated, unvalidated; sm_90 passed T3, see below); v9 and tc3: GPUs other than sm_89; multi-GPU |
| Device independence (CPU = GPU) | every backend equals the CPU reference bitwise, NaN canonical; CPU flush-to-zero raises | `truth:evo2.evo2_1b_w2k`, `truth:emulator.tcemu_kernels_fixed_inputs` | other GPUs until their GPU-tier run |
| Evo 2 end to end | per projection: the 21 FP8 projections of one Evo 2 1B window (accumulator, native bf16 output, TE output) | `truth:evo2.evo2_1b_w2k` | other GPUs; grad mode; full-model runs: Evo 2 1B and 40B end to end, below |
| Native vs Transformer Engine | equal on shapes that do not split on the RTX 4060 (every Evo 2 projection); different on split-prone shapes | `truth:c4.v3_te_linear`, `truth:c5.v4_te_demo`, `truth:c5.v1_workspace_env` | Hopper (TE uses a 32 MiB workspace there) |

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

## mmaemu's own results

The recordings above are what mmaemu must reproduce. The results of replaying them through mmaemu v0.1 are
committed as logs by the test tiers (docs/testing.md in the repository describes the tiers):

- T1 runs on every CI push: the committed subset of the recordings (tests/data/hw_mini), through the reference
  backend.
- T2 is the full replay of the ground truth through mmaemu, compared recording by recording with the audit's
  reference replay; its report goes to docs/validation_logs/.
- T3 and T3p are the GPU tiers: every available kernel against the reference, and live `torch._scaled_mm`
  calls against mmaemu on the reference GPU.
- T4 is the mutation matrix: deliberately broken copies of the model must fail the tests.
- T7 validates the built wheel in a fresh environment on the GPU; its report goes to docs/release_logs/.

Results for the v0.1.0 release candidate (its frozen fatbins included). Kernel sources and fatbins have not changed
since the freeze (e173ecc); later commits change Python code, tests and docs, so each row names the tree it ran on.
The review of the release candidate changed Python code under `mmaemu/` (49f1ecf); T2, T3 and T3p ran again on
that tree (`log:docs/validation_logs/2026-10-15/t2/README.md`,
`log:docs/validation_logs/2026-10-15/rtx-4060-sm89/README.md`).

| Tier | Result | Log |
|---|---|---|
| T2, full truth replay (CPU), 49f1ecf | PASS: all 173 recordings in the report, 147 of them with outputs that mmaemu models (the 15 tables are listed without comparison; 11 bias, row-wise, fp16-accumulator or HMMA recordings have none); 0 zero-sign and 0 NaN-payload differences; the 6,120 value differences all lie at the old emulator recording's NaN-decode outputs, where mmaemu gives the canonical NaN (per-output counts equal to the audit's replay); 113 of 193 hardware errors reproduced, as in the audit's replay; the extra not-applicable outputs are the bias and row-wise features that v0.1 rejects. Every count equals the run of 2026-10-14 on 6492713 | `log:docs/validation_logs/2026-10-15/t2/compare.txt`, `log:docs/validation_logs/2026-10-15/t2/README.md`, `log:docs/validation_logs/2026-10-14/t2/README.md` |
| T3 on the RTX 4060 (sm_89, the profile device), release fatbins, 49f1ecf | IEEE kernels 79 passed, CUDA runtime 13 passed, fast kernels 182 passed (v9 and tc3, with the audit's verify2 and p2 cases), gates 4 passed, live `torch._scaled_mm` 2 passed, hardware-mini 5 passed (reference, v1, v4, v9, tc3), device independence 31 passed (rows 0..255 of the 21 Evo 2 1B accumulators on every backend, the full-accumulator sha256 of all 21 for tc3). On da9edd4 (2026-10-14) the same counts, with 1 live test; a first run on 76fd4ea failed two test expectations written for devices without the gated kernels, which da9edd4 corrects (nothing under `mmaemu/` changed) | `log:docs/validation_logs/2026-10-15/rtx-4060-sm89/t3/summary.txt`, `log:docs/validation_logs/2026-10-15/rtx-4060-sm89/README.md`, `log:docs/validation_logs/2026-10-14/t3-rtx-4060-sm89/README.md`, `log:docs/validation_logs/2026-10-14/rtx4060-host/README.md` |
| Full accumulators on the RTX 4060 (`MMAEMU_DI_FULL`), da9edd4 | 84 passed: the full-accumulator sha256 of all 21 Evo 2 1B projections equals the hardware's for each of v1, v4, v9 and tc3 | `log:docs/validation_logs/2026-10-14/t3-rtx-4060-sm89/di_full_gpu_kernels.log` |
| T3p on the RTX 4060, 49f1ecf | 6 passed: live `torch._scaled_mm` parity (278 calls), the `out=` and mat1 layouts and empty calls (112 calls), the profile match, 500 split-K table rows re-queried from cuBLASLt, the live query step, and 11,760 live queries above the bound where mmaemu infers S = 1, all S = 1. On da9edd4 the four tests of that tree passed | `log:docs/validation_logs/2026-10-15/rtx-4060-sm89/t3p_profile_device.log`, `log:docs/validation_logs/2026-10-14/t3-rtx-4060-sm89/t3p_profile_device.log` |
| T3 on an RTX 5070 Ti (sm_120), release fatbins, e173ecc | IEEE kernels 79 passed, CUDA runtime 13 passed, fast kernels 91 passed (no tc3 binary for sm_120), device independence 27 passed; T3p: profile device unavailable | `log:docs/validation_logs/2026-10-14/t3-rtx-5070-ti-sm120/summary.txt` |
| T3 on an H100 PCIe (sm_90), release fatbins, 69cd231 | IEEE kernels 79 passed, CUDA runtime 13 passed, hardware-mini 3 passed (reference, v1, v4), fast kernels 12 passed (no tc3 binary for sm_90); device independence not run (no ground truth on the rented host); T3p: profile device unavailable | `log:docs/validation_logs/2026-10-14/t3-h100-sm90/summary.txt`, `log:docs/validation_logs/2026-10-14/t3-h100-sm90/README.md` |
| Evo 2 fixed inputs on the H100 (sm_90), NVRTC build of 5c15e8a (v1/v4 sources equal to the RC apart from comments) | v1 and v4: all 21 fp32 accumulators and bf16 outputs bitwise equal to the RTX 4060 recording; native `torch._scaled_mm` differs on all 21. sm_90 stays out of `validated_on`: the full device-independence suite did not run there | `log:docs/validation_logs/2026-10-14/t3-h100-sm90/fixed-fp8.json` |
| T4, CPU tier, 5f1a55c | PASS: 43 killed, 2 equivalent, the 5 identity controls survive | `log:docs/validation_logs/2026-10-14/t4/mutants-cpu.json` |
| T4, GPU tier (RTX 5070 Ti), 5f1a55c | PASS: 7 killed, the identity control survives | `log:docs/validation_logs/2026-10-14/t4/mutants-gpu.json` |
| T4, GPU tier (RTX 4060), da9edd4 | PASS: 7 killed, the identity control survives | `log:docs/validation_logs/2026-10-14/t4/mutants-gpu-rtx4060.json` |
| T6, performance on the RTX 4060, release fatbins, da9edd4 | PASS: at M = 2048 on the Evo 2 1B, 7B and 40B shapes tc3 2.93-2.94, v9 1.53-1.54, v4 1.34-1.37 and v1 0.62-0.63 TMAC/s, every kernel above its threshold, with Gaussian and adversarial data; native FP8 29.8-30.5 TMAC/s (docs/performance.md) | `log:docs/validation_logs/2026-10-14/t6-rtx-4060-sm89/README.md` |
| Release fatbin build | nvcc 12.8.61, 5 modules, 0 spill bytes | `log:docs/validation_logs/2026-10-14/fatbin-release/build_report.json` |
| T7 on the RTX 4060: the wheel `mmaemu-0.1.0-py3-none-any.whl` (sha256 a619cb84...) built from 85c780d | PASS, 8 of 8: a fresh venv without nvcc; `info` (the three RTX 4060 profiles match) and `selftest`; v1, v4, v9 and tc3 loaded from the wheel's fatbins (the images its manifest records); 897 conformance comparisons against the CPU reference (9,985,692 outputs), 0 differing; the Evo 2 1B projection (2048 x 5760 x 1920) equal to the hardware recording bit for bit (bf16 output and fp32 accumulator) | `log:docs/release_logs/0.1.0_rtx4060-sm89.json` |
| T7 on an RTX 5070 Ti (sm_120), the same wheel | PASS, 8 of 8: v1 and v4 from the wheel's fatbins, and v9 when requested explicitly (it is not auto-selected on sm_120; tc3 has no sm_120 binary); 673 conformance comparisons (7,493,365 outputs), 0 differing; the Evo 2 projection equal to the RTX 4060 recording bit for bit | `log:docs/release_logs/0.1.0_rtx5070ti-sm120.json` |

The GPU development validation is recorded in
[the sm_120 report](validation_logs/2026-10-13-sm120/README.md), including CPU/GPU
tests, mutations, Evo2 and a wheel without nvcc. This is separate from the full
release checklist in [release.md](release.md).

## Evo 2 1B end to end

Model-level measurements of the Evo 2 example (`repo:examples/evo2_blackwell.ipynb`) and of the official
path on a second GPU. They are not part of the bitwise validation above: only the FP8 GEMMs are emulated, and the
rest of the model runs natively (docs/evo2.md explains the setup).

| Measurement | Result | Evidence |
|---|---|---|
| The RTX 4060 reference shipped with the example (`repo:examples/data/evo2_1b_ada_rtx4060_te.npz`) | the official path (`evo2.Evo2` with TE FP8) captured on the RTX 4060 on 11 and 12 October; every shipped array equals its capture, and each capture's full logits have the SHA-256 its metadata records | `log:docs/validation_logs/2026-10-12/evo2-rtx-4060-official/check_reference.txt` |
| First FP8 projection (block 0, 2,048-bp window) on the RTX 5070 Ti | mmaemu's bf16 output has the SHA-256 of the RTX 4060's TE output; natively 8,293,802 of 11,796,480 outputs differ | `log:docs/validation_logs/2026-10-14/evo2-rtx-5070-ti/results.json`, `truth:evo2.evo2_1b_w2k` |
| Upstream test loss (prompts p0-p3, fp32 cross entropy) of the official path | RTX 4060 0.5024, H100 PCIe 0.5026 (the test expects 0.502) | `log:docs/validation_logs/2026-10-14/evo2-h100-sm90/upstream_test.json` |
| Upstream test loss on the RTX 5070 Ti (bf16 cross entropy on the GPU, as the test computes it) | mmaemu 0.5027, native FP8 1.2480, bf16 GEMMs 1.3652 | `log:docs/validation_logs/2026-10-14/evo2-rtx-5070-ti/results.json` |
| Six windows, mean \|Δ log p(A,C,G,T)\| against the RTX 4060 | H100 official 0.0411; RTX 5070 Ti: mmaemu 0.0403, native FP8 2.58, bf16 GEMMs 2.85 | `log:docs/validation_logs/2026-10-14/evo2-h100-sm90/comparison_vs_ada.json`, `log:docs/validation_logs/2026-10-14/evo2-rtx-5070-ti/results.json` |
| BRCA1 zero-shot AUROC, 500 variants | RTX 4060 official 0.7378, H100 official 0.7332; RTX 5070 Ti: mmaemu 0.7323, native FP8 0.5586, bf16 GEMMs 0.4842 | `log:docs/validation_logs/2026-10-14/evo2-h100-sm90/comparison_vs_ada.json`, `log:docs/validation_logs/2026-10-14/evo2-rtx-5070-ti/results.json` |
| BRCA1 score changes, Pearson against the RTX 4060 | H100 official 0.987; RTX 5070 Ti: mmaemu 0.986, native FP8 0.059, bf16 GEMMs 0.069 | `log:docs/validation_logs/2026-10-14/evo2-h100-sm90/comparison_vs_ada.json`, `log:docs/validation_logs/2026-10-14/evo2-rtx-5070-ti/results.json` |

## Evo 2 40B end to end

The original 40B checkpoint on the RTX 5070 Ti (`log:docs/validation_logs/2026-10-15/evo2-40b-rtx-5070-ti/README.md`),
with its weights in host RAM and one block at a time on the GPU; only the FP8 GEMMs are emulated (docs/evo2.md).

| Measurement | Result | Evidence |
|---|---|---|
| The scripts, on Evo 2 1B, against the model held on the GPU | in all three modes: the logits of five windows (vortex's forward pass) and the BRCA1 scores of 40 windows (`evo2.scoring.score_sequences`) equal bit for bit; the upstream-test losses equal the 1B log's (0.5027, 1.2480, 1.3652) | `log:docs/validation_logs/2026-10-15/evo2-40b-rtx-5070-ti/check_1b.json`, `log:docs/validation_logs/2026-10-15/evo2-40b-rtx-5070-ti/brca1_check_1b.json` |
| Upstream test loss of the 40B on the RTX 5070 Ti (bf16 cross entropy on the GPU, as the test computes it) | mmaemu 0.2161 (accuracy 0.9169), native FP8 1.4023 (0.2568), bf16 GEMMs 1.4043 (0.2559); the test expects 0.2159424 (0.91673) | `log:docs/validation_logs/2026-10-15/evo2-40b-rtx-5070-ti/test_40b.json` |
| The 40B reference on the RTX 4060 | the upstream blocks with TE FP8 run one at a time (the official 40B path does not fit in 8 GB); run this way, the 1B gives the logits of the real upstream model bit for bit on all six windows; the 40B log's file equals the captures | `log:docs/validation_logs/2026-10-12/evo2-rtx-4060-official/check_reference.txt` |
| Two windows (2,048 and 4,096 bp), mean \|Δ log p(A,C,G,T)\| against the official path on the RTX 4060 | mmaemu 0.031 and 0.029 (most likely base agrees at 99.90% and 99.93%); native FP8 3.97 and 4.07; bf16 GEMMs 3.98 and 4.07 | `log:docs/validation_logs/2026-10-15/evo2-40b-rtx-5070-ti/test_40b.json`, `log:docs/validation_logs/2026-10-15/evo2-40b-rtx-5070-ti/rtx4060_official_te_40b.npz` |
| BRCA1 AUROC of the 40B on the RTX 5070 Ti, the first 100 of the notebook's 500 variants (20 loss-of-function) | mmaemu 0.855, native FP8 0.503, bf16 GEMMs 0.590; the 1B with the official path on the RTX 4060, same variants: 0.719 | `log:docs/validation_logs/2026-10-15/evo2-40b-rtx-5070-ti/brca1_40b.json` |
| The official 40B on an H100 PCIe (`log:docs/validation_logs/2026-10-15/evo2-40b-h100-sm90/README.md`), the same block-by-block method; on the 1B there, the logits of 11 windows and 20 BRCA1 scores equal the real upstream model's bit for bit | upstream test 0.2159424 (the expected value; accuracy 0.9167) | `log:docs/validation_logs/2026-10-15/evo2-40b-h100-sm90/results/1b_check.json`, `log:docs/validation_logs/2026-10-15/evo2-40b-h100-sm90/results/40b/test.json` |
| Two windows, mean \|Δ log p(A,C,G,T)\|: H100 against RTX 4060 (both official), and the RTX 5070 Ti against the H100 | official H100 vs RTX 4060: 0.030 and 0.029; mmaemu vs H100: 0.0295 and 0.0288; native FP8 vs H100: 3.97 and 4.07 | `log:docs/validation_logs/2026-10-15/evo2-40b-h100-sm90/compare.json` |
| BRCA1 against the official 40B on the H100 | first 100 variants: H100 AUROC 0.855, the same as mmaemu's; mmaemu's per-variant score changes vs the H100's: Pearson 0.998, Spearman 0.986, same sign 99% (native FP8: 0.17, 0.12, 55%); all 500 variants: H100 AUROC 0.868 | `log:docs/validation_logs/2026-10-15/evo2-40b-h100-sm90/compare.json` |

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
