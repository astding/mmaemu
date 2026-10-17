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
| `hopper` | 32 / 13 / 13 / -132, c in the alignment | 128-element chunks | **validated** | cuBLAS FP8 GEMMs on an H100 PCIe (sm_90 `nvjet` kernels, cuBLASLt 12.8.3.14), see [validation.md](validation.md); not a model of the sm_90 `mma.sync` instruction, which is a different unit |
| `b200_mma_sync` (alias `blackwell`) | 32 / 25 / 23 / -133, c added after the block sum (fp32 RNE) | raises `UnsupportedFeatureError` | **experimental** | fits the toolbox's B200 `mma.sync` vectors (HMMA rewrite on sm_90/sm_100); not a model of sm_120, not B200 cuBLAS |

- Using an experimental preset emits `ExperimentalPresetWarning` (a `UserWarning`) once per process and preset.
    An experimental profile whose preset is validated (`hopper-experimental`) emits it once per process and profile.
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
| `hopper` (default) | `hopper` | `torch._scaled_mm`, 1 MiB | none | `heur_h100pcie_lt12.8.3.14` | validated | validated |
| `hopper-te` | `hopper` | Transformer Engine 2.3, 32 MiB | 256 | same table | validated | experimental |
| `hopper-ws0` | `hopper` | workspace 0 (`CUBLASLT_WORKSPACE_SIZE=0`) | none | same table (never splits) | validated | validated |
| `ada` | `ada` | `torch._scaled_mm`, 1 MiB | none | `heur_rtx4060_lt12.8.3.14` | validated | validated |
| `ada-te` | `ada` | Transformer Engine 2.3, 4 MiB | 256 | same table | validated | experimental |
| `ada-ws0` | `ada` | workspace 0 (`CUBLASLT_WORKSPACE_SIZE=0`) | none | same table (never splits) | validated | validated |
| `hopper-experimental` | `hopper` | n/a (no reference platform) | n/a | none | experimental | none |
| `b200-mma-sync-experimental` | `b200_mma_sync` | n/a | n/a | none | experimental | none |

All three RTX 4060 profiles describe the same GPU (24 SMs), cuBLASLt 12.8.3.14 and torch 2.7.1; they differ in the
caller's cuBLAS workspace, which changes the split-K choices. The three H100 PCIe profiles do the same for an H100
PCIe (sm_90, 114 SMs), cuBLASLt 12.8.3.14 and torch 2.7.1, with the workspaces logged there (torch 1 MiB, TE 32 MiB).
Other Hopper platforms (H100 SXM with 132 SMs, H200, other cuBLASLt versions) have no profile of their own:
`hopper-experimental` gives the hopper numerics without a split-K table.

- Choose a profile per call (`profile="hopper-te"`) or with `MMAEMU_PROFILE`. Evo 2 computes its FP8 projections
    with Transformer Engine, so it takes a TE profile: `hopper-te` for TE's GEMM on the H100 PCIe (bit for bit on the
    21 projections of one Evo 2 1B window captured on the H100, all at S = 1) or `ada-te` for the RTX 4060's. Evo 2 1B
    and 40B were run end to end with each on an RTX 5070 Ti ([evo2.md](evo2.md),
    [log](validation_logs/2026-10-16/evo2-h100-profile-rtx-5070-ti/README.md)). The notebook
    (`examples/evo2_blackwell.ipynb`) uses `ada-te`, because its reference outputs in `examples/data/` were produced
    by Transformer Engine on the RTX 4060.
- `Profile.matches_device(device)` is True only when the GPU name, the SM count *and* the cuBLASLt version all
    match. The compute capability alone never matches a profile: two Ada GPUs with different SM counts can choose
    different split-K kernels for the same shape.
- **Why TE split decisions are experimental.** On the measured shapes TE's cuBLASLt choices equal torch's
    descriptor with a 256-byte minimum alignment, but the reduction scheme of TE split outputs has not been replayed
    independently. Decisions with S > 1 on `ada-te` therefore emit `UnverifiedSplitWarning`. S = 1 decisions
    do not warn, and every Evo 2 projection shape is S = 1 on this GPU.
- **Why H100 split decisions are validated at 1 MiB and experimental with TE.** The H100's split-K rule
    (`"contiguous"`, below) was found after the probe's data came in, by fitting it to the 1,049 live split calls; it
    then reproduced all of them through mmaemu. The 2,416 live split calls of the later split-K sweep, run after the
    rule was fixed, replay equal too. Their 1,630 calls at 1 MiB cover every S (2 to 6) and customOption (0 to 3) of
    the table's 1 MiB split rows and every K of them (up to 28,672) except 5,760, which only the probe's split calls
    cover (84 at 1 MiB, the calls the rule was fitted to). Decisions with S > 1 on `hopper` and `hopper-ws0`
    are validated (a decision taken after the sweep). The 32 MiB calls ran `torch._scaled_mm` with TE's workspace; no
    split call ran through TE's own GEMM, so decisions with S > 1 on `hopper-te` emit
    `UnverifiedSplitWarning`. No Evo 2 FP8 projection shape (the input projections, (K, N) = (1920, 5760),
    (4096, 12288), (8192, 24576)) splits on the H100.
- **Why the TE envelope warns on the H100.** The H100 envelope (step 5 below) passed its sealed held-out test at
    torch's 1 MiB workspace (154,191 shapes above the bound, none split) and failed it at TE's 32 MiB (77,518 shapes,
    one split). Its S = 1 inferences at 32 MiB are therefore unverified and emit `UnvalidatedEnvelopeWarning`.
- **Profile names.** The profiles were renamed (old -> new): `hopper-h100pcie-torch` -> `hopper`,
    `hopper-h100pcie-te` -> `hopper-te`, `hopper-h100pcie-ws0` -> `hopper-ws0`, `ada-rtx4060-torch` -> `ada`,
    `ada-rtx4060-te` -> `ada-te`, `ada-rtx4060-ws0` -> `ada-ws0`. The per-workspace profiles that the H100 replay tools
    derive are now named `hopper-ws<N>` (N the workspace in bytes); the replay logs name them
    `hopper-h100pcie-torch-ws<N>`. `hopper-experimental` and `b200-mma-sync-experimental` keep their names. There are
    no aliases: `get_profile` raises `KeyError` for an old name. The logs of the runs made before the rename (every run
    up to and including the runs on fa0fd90 of 16 October) use the old names; the run logs of the Evo 2 run with
    the H100 profile (16 October, mmaemu fa0fd90) name it `hopper-h100pcie-te`. The shipped H100 table
    (`heur_h100pcie_lt12.8.3.14`) keeps `eval_profile` with the old names (`hopper-h100pcie-torch`,
    `hopper-h100pcie-te`), because it names the key of the split-K sweep's `eval_sealed.json`, so the rename left the
    table files and their sha256 unchanged.

## Split-K resolution

`splitk="auto"` (the default) resolves (S, scheme) in this order; `mmaemu.explain(...)` shows the decision and its
source:

1. **explicit**: `splitk=1`, `splitk=(S, scheme)` or a `SplitK` object. The scheme belongs to the preset's
    cuBLASLt kernels: `"serial"` or `"pairwise"` for `ada` (64-wide k-tiles dealt out t mod S), `"contiguous"` for
    `hopper` (each split a contiguous run of 128-wide k-tiles). An int > 1 without a scheme, or a scheme that is
    none of these three, raises `ValueError` here; a scheme of another preset is returned as given (`verified=False`,
    with `UnverifiedSplitWarning`) and raises `ValueError` when the GEMM is planned (`scaled_mm`, `accumulate`;
    `explain` reports it in `CallPlan.validation`).
2. **workspace0**: a profile with a 0-byte workspace never splits.
3. **table**: the shape and setting (M, N, K, formats, output dtype, fast/slow, workspace) is in the profile's
    recorded cuBLASLt heuristic table.
4. **live-query**: the current GPU matches the profile, so cuBLASLt's own heuristic is queried.
5. **envelope**: S = 1 is inferred from the shape's M*N. This is an inference, and the decision's note says so.
    `MMAEMU_SPLITK_STRICT=1` (or `strict=True`) turns this step off.
    - RTX 4060: M*N is more than twice the largest M*N that split in the table for this setting (whatever K is);
        cuBLASLt splits shapes up to 1.27 times that M*N, and at K below the smallest K that split in the table.
    - H100 PCIe: M*N is above the setting's bound at K, from the split-K sweep of 2026-10-16: the bound grows with K
        (from 15,360 at K <= 5,120 to about 268,000 at 1 MiB and 11.8 million at 32 MiB), K is rounded up to the next
        of 33 grid values, and only 1,024 <= K <= 4,194,304 is claimed. Validated at 1 MiB; unvalidated at 32 MiB,
        where it warns (`UnvalidatedEnvelopeWarning`).
6. Otherwise `SplitKUnknownError`, whose message gives three ways out: pass `splitk=1`, pass `splitk=(S, scheme)`,
    or record the shape on the reference GPU with `tools/record_heuristics.py`.

Profiles without a table (`hopper-experimental`, `b200-mma-sync-experimental`) have no envelope: `"auto"` always
raises there; pass `splitk` explicitly.

`UnverifiedSplitWarning` is emitted once per combination when a decision's (fast, S, scheme, output dtype, format
pair) was never replayed on hardware, when a `"contiguous"` split lies outside the replayed calls (a split without
k-tiles, a customOption never seen on a replayed split call), or when S > 1 on a profile whose split-K maturity is
experimental. Its subclass `UnvalidatedEnvelopeWarning` is emitted once per setting for an S = 1 inferred by an
envelope that did not pass its held-out test (the H100 at 32 MiB).

### The shipped tables

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

`heur_h100pcie_lt12.8.3.14` holds 1,767,416 heuristic choices of cuBLASLt 12.8.3.14 on the H100 PCIe: the probe's
grid (396,816 rows over 3,543 shapes: the c5, Evo 2 and Evo 2 small-M grids, workspaces 0, 1, 4 and 32 MiB) and the
split-K sweep's model rows (the (K, N) of the projections of Evo 2 1B/7B/40B, Llama 2 7B, Llama 3 8B, Mistral 7B,
Qwen2.5 7B, Llama 3 70B and Llama 3.2 1B, every M up to a few hundred, then every 16th to 8,192 and every 256th to
16,384, at 1 and 32 MiB):

- 36,457 rows split K, all at K >= 5760 and multiples of 128, with S = 2 to 6, customOption 0 to 5 and the
    reduction scheme `COMPUTE_TYPE`;
- a 0-byte workspace never splits (0 of 915,980 workspace-0 queries);
- the alignment setting never changes the choice (0 of 2,031,000 query pairs);
- e4m3 x e4m3 -> e5m2 is rejected (status 15) on every queried shape;
- the Evo 2 projection shapes never split, at any recorded M or workspace;
- decode (small M) of the 9 models is in the table: M = 1 of Llama 3 70B's down projection (K 28,672) splits, for
    example.

Off the table, the envelope decides where M*N is large enough; elsewhere `"auto"` raises (pass `splitk`, or run on
the H100). The rule, its held-out test and its assumptions are in docs/splitk_tables.md and the sweep's validation
log (docs/validation_logs/2026-10-16/h100-splitk-sweep in the repository).

## Environment variables

| Variable | Effect |
|---|---|
| `MMAEMU_PROFILE` | default profile name |
| `MMAEMU_BACKEND` | default backend (`auto`, `reference`, `cuda`, `cuda:v1`, `cuda:v4`, `cuda:v9`, `cuda:tc3`, `cuda:h1`, `cuda:h4`) |
| `MMAEMU_SPLITK_STRICT=1` | no envelope inference (step 5 above) |
| `MMAEMU_DISABLE_CUDA=1` | never touch CUDA; use the reference backend |
| `MMAEMU_ALLOW_GATED_KERNELS=1` | allow the gated kernels (v9, tc3) on capabilities they were not validated on, if their gates pass |
| `MMAEMU_CACHE_DIR` | NVRTC cubins and self-test verdicts (default `${XDG_CACHE_HOME:-~/.cache}/mmaemu`) |
| `MMAEMU_LIBCUDA_PATH`, `MMAEMU_NVRTC_PATH` | explicit driver / NVRTC library paths |
