# Presets and profiles

mmaemu separates two questions:

- **Preset**: what does one tensor-core instruction compute? (block size, fraction bits, how c enters, slow-chunk
  length.) `mmaemu.get_preset(name)`, `mmaemu.list_presets()`.
- **Profile**: which platform does a GEMM call reproduce? (GPU, SM count, cuBLASLt version, the caller and its
  workspace size, the split-K table, what cuBLAS rejects.) `mmaemu.get_profile(name)`, `mmaemu.list_profiles()`.

A call uses the profile's preset. `python -m mmaemu info` lists both, with their maturity, and tells you whether
your GPU matches a profile.

## Presets

| Preset | Parameters (block / term bits / result bits / min exp, c) | Slow accumulation | Maturity | Validated scope |
|---|---|---|---|---|
| `ada` | 16 / 13 / 13 / -132, c in the alignment | 128-element chunks | **validated** | RTX 4060 `mma.sync` and cuBLAS GEMM recordings (see [validation.md](validation.md)) |
| `hopper` | 32 / 13 / 13 / -132, c in the alignment | raises `UnsupportedFeatureError` | **experimental** | instruction level only: the toolbox's H100/H200 vectors, recorded with c = 0; no GEMM-level hardware evidence |
| `b200_mma_sync` (alias `blackwell`) | 32 / 25 / 23 / -133, c added after the block sum (fp32 RNE) | raises `UnsupportedFeatureError` | **experimental** | fits the toolbox's B200 `mma.sync` vectors (HMMA rewrite on sm_90/sm_100); not a model of sm_120, not B200 cuBLAS |

- Using an experimental preset emits `ExperimentalPresetWarning` (a `UserWarning`) once per process and preset.
  The alias `blackwell` warns on every use: it fits the toolbox B200 `mma.sync` (HMMA-rewrite) vectors, it is
  NOT a model of sm_120 / RTX 5070 Ti, and NOT B200 cuBLAS.
- The names `sm_120` and `rtx_blackwell` are not accepted: there is no hardware evidence for sm_120 FP8 numerics
  yet. The bring-up runbook (docs/runbooks/sm120_bringup.md in the repository) describes how such evidence is
  collected.
- You can build your own `Preset(...)` (maturity `"custom"`), for example to test a hypothesis. A custom preset is
  never a profile default.

## Profiles

| Profile | Preset | Caller, workspace | Alignment | Split-K table | Maturity | Split-K maturity |
|---|---|---|---|---|---|---|
| `ada-rtx4060-torch` (default) | `ada` | `torch._scaled_mm`, 1 MiB | none | `heur_rtx4060_lt12.8.3.14` | validated | validated |
| `ada-rtx4060-te` | `ada` | Transformer Engine 2.3, 4 MiB | 256 | same table | validated | experimental |
| `ada-rtx4060-ws0` | `ada` | workspace 0 (`CUBLASLT_WORKSPACE_SIZE=0`) | none | same table (never splits) | validated | validated |
| `hopper-experimental` | `hopper` | n/a | n/a | none | experimental | none |
| `b200-mma-sync-experimental` | `b200_mma_sync` | n/a | n/a | none | experimental | none |

All three RTX 4060 profiles describe the same GPU (24 SMs), cuBLASLt 12.8.3.14 and torch 2.7.1; they differ in the
caller's cuBLAS workspace, which changes the split-K choices.

- Choose a profile per call (`profile="ada-rtx4060-te"`) or with `MMAEMU_PROFILE`. Evo 2 (through
  evo2-anywhere) uses `ada-rtx4060-te`, because the reference outputs were produced by Transformer Engine.
- `Profile.matches_device(device)` is True only when the GPU name, the SM count *and* the cuBLASLt version all
  match. The compute capability alone never matches a profile: two Ada GPUs with different SM counts can choose
  different split-K kernels for the same shape.
- **Why TE split decisions are experimental.** On the measured shapes TE's cuBLASLt choices equal torch's
  descriptor with a 256-byte minimum alignment, but the reduction scheme of TE split outputs has not been replayed
  independently. Decisions with S > 1 on `ada-rtx4060-te` therefore emit `UnverifiedSplitWarning`. S = 1 decisions
  do not warn, and every Evo 2 projection shape is S = 1 on this GPU.

## Split-K resolution

`splitk="auto"` (the default) resolves (S, scheme) in this order; `mmaemu.explain(...)` shows the decision and its
source:

1. **explicit**: `splitk=1`, `splitk=(S, "serial")`, `splitk=(S, "pairwise")` or a `SplitK` object. An int > 1
   without a scheme raises `ValueError`.
2. **workspace0**: a profile with a 0-byte workspace never splits.
3. **table**: the shape and setting (M, N, K, formats, output dtype, fast/slow, workspace) is in the profile's
   recorded cuBLASLt heuristic table.
4. **live-query**: the current GPU matches the profile, so cuBLASLt's own heuristic is queried.
5. **envelope**: the shape lies outside the region where the table ever split for this setting (M*N above the
   largest split M*N, or K below the smallest split K), so S = 1 is inferred. This is an inference, and the
   decision's note says so. `MMAEMU_SPLITK_STRICT=1` (or `strict=True`) turns this step off.
6. Otherwise `SplitKUnknownError`, whose message gives three ways out: pass `splitk=1`, pass `splitk=(S, scheme)`,
   or record the shape on the reference GPU with `tools/record_heuristics.py`.

Profiles without a table (`hopper-experimental`, `b200-mma-sync-experimental`) have no envelope: `"auto"` always
raises there; pass `splitk` explicitly.

`UnverifiedSplitWarning` is emitted once per combination when a decision's (fast, S, scheme, output dtype, format
pair) was never replayed on hardware, or when S > 1 on a profile whose split-K maturity is experimental.

### The shipped table

`heur_rtx4060_lt12.8.3.14` holds cuBLASLt 12.8.3.14's heuristic choices on the RTX 4060 for 906,600 queries over
17,476 shapes (hardware recording `heur.lt_heuristic_12_8_3_14` of the ground truth). Facts measured on it while
writing the v0.1 specification:

- 108,286 rows split K, with S in {2..10, 12, 14, 16, 24, 29} and customOption 0, 1 or 2;
- a 0-byte workspace never splits (0 of 209,712 queries);
- the alignment setting (none vs 256 bytes) never changes the choice (0 of 419,424 query pairs), so the lookup key
  omits it;
- the Evo 2 projection shapes, (K, N) = (1920, 5760), (4096, 12288) and (8192, 24576), never split at any recorded
  workspace.

Split envelopes of the e4m3-family settings (largest M*N that split, smallest K that split):

| Workspace | Output | Fast accumulation | Slow accumulation |
|---|---|---|---|
| 1 MiB | fp32 | M*N <= 100,352, K >= 1,824 | M*N <= 106,496, K >= 16 |
| 1 MiB | bf16 | M*N <= 100,352, K >= 16 | M*N <= 131,072, K >= 2,048 |
| 1 MiB | fp16 | M*N <= 135,168, K >= 2,048 | M*N <= 106,496, K >= 16 |
| 4 MiB | all | M*N <= 983,040, K >= 16 .. 1,792 | M*N <= 983,040, K >= 16 .. 1,792 |

Inside an envelope, not every shape splits: a shape missing from the table raises instead of guessing.
The table's per-setting envelopes and coverage are documented with the table itself (docs/splitk_tables.md in
the repository).

## Environment variables

| Variable | Effect |
|---|---|
| `MMAEMU_PROFILE` | default profile name |
| `MMAEMU_BACKEND` | default backend (`auto`, `reference`, `cuda`, `cuda:v1`, `cuda:v4`, `cuda:v9`, `cuda:tc3`) |
| `MMAEMU_SPLITK_STRICT=1` | no envelope inference (step 5 above) |
| `MMAEMU_DISABLE_CUDA=1` | never touch CUDA; use the reference backend |
| `MMAEMU_ALLOW_GATED_KERNELS=1` | allow the gated kernels (v9, tc3) on capabilities they were not validated on, if their gates pass |
| `MMAEMU_CACHE_DIR` | NVRTC cubins and self-test verdicts (default `${XDG_CACHE_HOME:-~/.cache}/mmaemu`) |
| `MMAEMU_LIBCUDA_PATH`, `MMAEMU_NVRTC_PATH` | explicit driver / NVRTC library paths |
