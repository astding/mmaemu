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
- The names `sm_120` and `rtx_blackwell` are not accepted: sm_120 is not modelled (no preset, profile or split-K
  table). Khattak and Mikaitis (arXiv:2512.07004v4, section 4.1.8, Tables 3-4) report RTX PRO 6000 (sm_120)
  tensor-core numerics identical to the B200's, and probes on an RTX 5070 Ti are consistent with that model, but it
  is not validated at GEMM level. The bring-up runbook (docs/runbooks/sm120_bringup.md in the repository) describes
  how the remaining evidence is collected.
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
5. **envelope**: the shape's M*N is more than twice the largest M*N that split in the table for this setting, so
   S = 1 is inferred (whatever K is). This is an inference, and the decision's note says so: on the RTX 4060,
   cuBLASLt splits shapes up to 1.27 times that M*N, and at K below the smallest K that split in the table.
   `MMAEMU_SPLITK_STRICT=1` (or `strict=True`) turns this step off.
6. Otherwise `SplitKUnknownError`, whose message gives three ways out: pass `splitk=1`, pass `splitk=(S, scheme)`,
   or record the shape on the reference GPU with `tools/record_heuristics.py`.

Profiles without a table (`hopper-experimental`, `b200-mma-sync-experimental`) have no envelope: `"auto"` always
raises there; pass `splitk` explicitly.

`UnverifiedSplitWarning` is emitted once per combination when a decision's (fast, S, scheme, output dtype, format
pair) was never replayed on hardware, or when S > 1 on a profile whose split-K maturity is experimental.

### The shipped table

`heur_rtx4060_lt12.8.3.14` holds 1,228,448 heuristic choices of cuBLASLt 12.8.3.14 on the RTX 4060 over 18,497
shapes, merged from the ground truth's table (`heur.lt_heuristic_12_8_3_14`, 906,600 queries) and the recordings
R2, R2x and R3:

- 125,980 rows split K, with S in {2..10, 12, 14, 16, 24, 29} and customOption 0, 1 or 2;
- a 0-byte workspace never splits (0 of 307,112 workspace-0 rows);
- the alignment setting (none vs 256 bytes) never changes the choice (0 of 1,422,624 query pairs), so the lookup
  key omits it;
- the Evo 2 projection shapes, (K, N) = (1920, 5760), (4096, 12288) and (8192, 24576), never split at 0, 1 or
  4 MiB; at 32 MiB, (8192, 24576) splits at M = 65..74 with fp16 or fp8 outputs.

Largest split M*N of the e4m3 x e4m3 settings (S = 1 is inferred only above twice this value):

| Workspace | Output | Fast accumulation | Slow accumulation |
|---|---|---|---|
| 1 MiB | fp32 | 100,352 | 106,496 |
| 1 MiB | bf16 | 100,352 | 131,072 |
| 1 MiB | fp16 | 135,168 | 106,496 |
| 1 MiB | E4M3 | 131,072 | 98,304 |
| 4 MiB | all | 983,040 | 983,040 |

Below that bound, not every shape splits: a shape missing from the table raises instead of guessing.
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
