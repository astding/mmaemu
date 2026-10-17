# Split-K tables

cuBLAS decides per GEMM whether to split the K dimension (`numSplitsK = S`) and how to combine the partial sums
(`customOption`). This is a heuristic decision of the GPU, the cuBLASLt version, the workspace size, the output
dtype and fast/slow accumulation. It cannot be derived from the shape alone (audit C5-F6 / C4-F4). mmaemu therefore
reproduces cuBLAS's split-K numerics only on a **profile's reference platform**, from a table recorded there.

This page describes the tables shipped with mmaemu (RTX 4060 and H100 PCIe), how `resolve_splitk` uses them, what
they cover, and how to record a table for another GPU.

## The shipped table: `heur_rtx4060_lt12.8.3.14`

| item | value |
|---|---|
| platform | NVIDIA GeForce RTX 4060, sm_89, 24 SMs, driver 572.16, cuBLASLt 12.8.3.14 (torch 2.7.1+cu128 wheel) |
| descriptor | torch 2.7.1 `scaled_gemm`: m=N, n=M, k=K; A = mat2 (transa = T), B = mat1; D = out dtype; C = out dtype, **fp16 for fp8 outputs**; A/B scale pointers; FAST_ACCUM; MAX_WORKSPACE_BYTES; requestedAlgoCount 1 |
| key | (M, N, K, fmt_a, fmt_b, out dtype, fast, workspace) |
| rows | 1,228,448, of which 125,980 split |
| S values | 2-10, 12, 14, 16, 24, 29 |
| (reductionScheme, customOption) of split rows | (2, 1) 106,947; (2, 2) 11,798; (2, 0) 7,235 |
| workspaces | 0, 1, 4, 32 MiB |
| package data | `mmaemu/platform/data/heur_rtx4060_lt12.8.3.14.npz` (236,342 B, sha256 `304203ef...`), `.json` (sources, envelopes), `verified_split_combos.json`; 402,581 B in total |

Sources (each sha256 is in the `.json` sidecar):
- the truth table `heur.lt_heuristic_12_8_3_14` (906,600 queries, sha256 `992aa95c...`, equal to the truth
  manifest entry): c5 grid, the c4 shape sets, the Evo 2 grid; 48 e4m3 x e4m3 settings; mixed formats at
  (1 MiB, no alignment) and (4 MiB, 256 B);
- R2 (`r2.lt_heur_fp8out_12_8_3_14`, 1,711,680 queries): fp8 outputs at every workspace, alignment, D-scale
  pointer and format pair;
- R2x (`r2.lt_heur_mixed_allsettings_12_8_3_14`, `r2.lt_heur_smallk_12_8_3_14`): mixed formats with
  fp32/bf16/fp16 outputs at every workspace and alignment; a small-K shape set;
- R3 (`r3.lt_heur_evo2_small_m_12_8_3_14`): the three Evo 2 (K, N) at every M = 1..256 at all 72 truth settings.

R2, R2x and R3 are pre-swap recordings in `recordings/rtx4060_2026-10` (see its README.txt).

### How the table is built

`python tools/export_splitk_table.py` (CPU; reads the truth table read-only):
1. checks the truth table's sha256 against the truth manifest;
2. asserts **alignment invariance**: every (no alignment preference, 256 B) row pair with the same key has identical
   (status, algo, tile, stages, splitK, reduction, swizzle, custom). Result: 0 of 1,422,624 pairs differ (truth
   table: 0 of 419,424; R2 0 of 427,920; R2x 0 of 427,920 + 0 of 128,928; R3 0 of 18,432). The key therefore omits
   the alignment, and the Transformer Engine profile (256 B) uses the same rows;
3. asserts that rows queried in several sources agree (331,896 duplicate keys, 0 conflicts) and that the D scale
   pointer does not change any fp8-output decision (0 of 855,840 pairs);
4. drops the 71,320 rows with a nonzero status: all of them are e4m3 x e4m3 -> e5m2, which cuBLASLt 12.8.3 rejects
   on sm_89 (CUBLAS_STATUS_NOT_SUPPORTED);
5. writes a byte-deterministic npz (fixed zip member order, timestamps and compression), the JSON sidecar with the
   per-setting envelopes, and `verified_split_combos.json`.

Running it twice gives identical files (`sha256sum` before/after; `--check` compares with the committed files).

## The H100 PCIe table: `heur_h100pcie_lt12.8.3.14`

| item | value |
|---|---|
| platform | NVIDIA H100 PCIe, sm_90, 114 SMs, driver 570.148.08, cuBLASLt 12.8.3.14 (torch 2.7.1+cu128 wheel) |
| descriptor | torch 2.7.1 `scaled_gemm`, as for the RTX 4060 table (recorded by `tools/record_heuristics.py`) |
| sources | the probe's grid (c5, Evo 2 and Evo 2 small-M: 3,543 shapes, 4 workspaces) and the split-K sweep's model rows (p3: the (K, N) of 9 models, M from 1 to 16,384, 1 and 32 MiB) |
| rows | 1,767,416, of which 36,457 split (probe 396,816, p3 1,498,784, 128,184 keys in both) |
| K values of the rows | 1024, 1920, 2048, 3584, 4096, 5120, 5760, 8192, 11008, 11264, 12288, 14336, 16384, 18944, 22528, 28672 |
| S values of the rows | 2-6, at K = 5760 and above, all multiples of 128 |
| (reductionScheme, customOption) of split rows | reduction 2 (`COMPUTE_TYPE`) always; customOption 0 to 5 |
| kernels | algo 66 (`nvjet_*`) on every row, k-stage 128 (`MATMUL_STAGES_128xAUTO`) |
| envelope | K-dependent, from the sweep: validated at 1 MiB, unvalidated at 32 MiB (below) |
| package data | `heur_h100pcie_lt12.8.3.14.npz` (224,792 B), `.json`, `verified_split_combos_h100pcie_lt12.8.3.14.json` |

`python tools/export_splitk_table_h100.py` (CPU) builds it from the probe's recording
(docs/validation_logs/2026-10-15/h100-probe/results/m6_heur/heur.npz, sha256 checked against the log's
SHA256SUMS) and the sweep's p3 rows (docs/validation_logs/2026-10-16/h100-splitk-sweep/results/p3). It asserts the
hardware of the recordings, alignment invariance (0 of 425,160 probe pairs and 0 of 1,605,840 p3 pairs differ), that
the rows with a nonzero status are exactly the e4m3 x e4m3 -> e5m2 ones (status 15), that a key of both recordings
has the same row, and what the split rule rests on: every split row has K % 128 == 0, no split without k-tiles, the
reduction scheme `COMPUTE_TYPE` and a nonzero workspace. One exception to the same-row check: 28 unsplit keys (M 272
and 254, N 12288, K 4096) have customOption 5 in the probe and 7 in p3, with the same kernel otherwise; the probe's
row is kept and the keys are listed in the json (`unsplit_customOption_differs`). cuBLASLt's customOption of an
unsplit row is therefore not a function of the key alone; mmaemu does not use it for S = 1.

The json differs from the RTX 4060 table's in these entries, which `resolve_splitk` reads:

- `codes."custom -> scheme"`: every customOption seen on a replayed live split call (0 to 3) means `"contiguous"`;
  any other customOption (4 and 5 occur on split rows of p3) also gives `"contiguous"`, with a caveat that makes the
  decision unverified (`"custom -> scheme, unlisted"`);
- `split_layouts`: which layouts of the contiguous rule replayed live calls cover. A ragged last k-tile
  (K % 128 != 0) is covered (1,040 live split calls of the sweep), a split without k-tiles is not (0 calls);
- `envelope`: `"kind": "k_grid"`, the sweep's bounds and their held-out result (below);
- `workspace0_note`: 0 split rows among 915,980 workspace-0 queries (probe 212,580, sweep p1 653,400, sealed p4
  50,000);
- `verified_combos`: the file of the replayed combinations.

### The H100 envelope

The split-K sweep of 2026-10-16 (docs/validation_logs/2026-10-16/h100-splitk-sweep, about 12 million heuristic
queries on one H100 PCIe after a drift gate that re-recorded the probe's grid byte for byte) fitted a bound per
setting (workspace, output, fast/slow, format pair) and K on its training parts and only then opened a sealed
held-out part (p4):

- **the rule**: at each of 33 grid K (1,024 to 4,194,304), the next p1 level (M*N = 2^(i/16) x 122,880) above the
  largest split M*N, raised by the split rows of the other training parts, times 2, and never smaller at a larger K;
  at least 15,360. A shape of the setting is claimed unsplit when 1,024 <= K <= 4,194,304 and M*N > B(K), with B
  at the smallest grid K >= K. The rule and its assumptions are in `results/analysis/envelope_rule.json`; the
  table carries its bounds unchanged (the export asserts the rule's sha256 is the one tested);
- **the held-out test** (`results/analysis/eval_sealed.json`, run once):

  | workspace (profile) | held-out shapes above the bound | split | result |
  |---|---:|---:|---|
  | 1 MiB (`hopper`) | 154,191 | 0 | validated; the closest split is at 0.49 x its bound |
  | 32 MiB (`hopper-te`) | 77,518 | 1 | unvalidated: M 69, N 29456, K 19872, e4m3 x e5m2 -> fp32 slow split in 2 at 1.23 x the bound |
  | 0 (`hopper-ws0`) | 46,654 (every status-0 row) | 0 | S = 1 holds |

  At 32 MiB above K = 65,536 the bound is near the top of the tested M*N range, so few held-out shapes test it
  there (about 4,300). Between 15,360 and 30,720 only model rows test the rule (532 per profile).
- **in mmaemu**: an inference of the 1 MiB envelope is `verified=True`. An inference of the 32 MiB envelope (any
  profile at that workspace) is `verified=False` and emits `UnvalidatedEnvelopeWarning` (a subclass of
  `UnverifiedSplitWarning`) once per setting. A workspace without bounds (4 MiB has table rows) claims nothing.
- **the resolver against every H100 decision** (`tools/check_splitk_resolution_h100.py`,
  `results/analysis/resolver_check.json`): every status-0 row of p1-p4 and every live call of the probe and the
  sweep through `resolve_splitk`: no decision differs from the H100's except that one p4 row.

The split rule (scheme `"contiguous"`) and how it was found are in
[semantics.md](semantics.md#2-a-gemm-accumulation-plan-on-the-reference-profile) and in the probe's log; the
sweep's 2,416 live split calls (S 2 to 24, K up to 262,144, 1,040 with K % 128 != 0), run after the rule was fixed,
replay equal (`results/analysis/mmaemu_replay_p5.json`). A combination is verified when a live H100 call with that
logged split exists and mmaemu reproduces its full output (sha256) in one of the two replays
(`tools/replay_h100_log.py`): all 140 split combinations of the table, 444 in all.

## Resolution: `resolve_splitk`

`mmaemu.platform.splitk_table.resolve_splitk(M, N, K, *, a_dtype, b_dtype, out_dtype, use_fast_accum, profile=None,
splitk="auto", layout_std=True, strict=None, dry=False)` returns a `SplitDecision(S, scheme, source, verified,
table_id, note)`. The order is:

1. **explicit**: `splitk=1`, `SplitK(S, scheme)` or `(S, scheme)`; an int > 1 without a scheme raises ValueError;
2. **workspace0**: the profile's workspace is 0 -> S = 1 (0 split rows among 209,712 workspace-0 queries on the
   RTX 4060, 915,980 on the H100);
3. **table**: the key is in the table -> the recorded (numSplitsK, customOption), the scheme by the table's rule:
   on the RTX 4060 customOption 0 and 2 mean "serial", 1 means "pairwise"; on the H100 every customOption means
   "contiguous";
4. **live-query**: the current GPU is the profile's reference platform (`Profile.matches_device`: GPU name, SM
   count and cuBLASLt version) -> `mmaemu.platform.cublaslt.heuristic` is queried with the profile's workspace and
   alignment;
5. **envelope**, not strict -> S = 1. This is an inference; the note says so. On the RTX 4060: M*N above twice
   the largest M*N that split in the setting (`ENVELOPE_MARGIN = 2`), K plays no part; the margin and the absence
   of a K rule come from live sweeps on the RTX 4060 (below). On the H100: M*N above the setting's bound at the
   next grid K, 1,024 <= K <= 4,194,304 ([The H100 envelope](#the-h100-envelope)); at 32 MiB the decision is
   unverified and warns (`UnvalidatedEnvelopeWarning`);
6. otherwise `SplitKUnknownError`, which names the setting and its envelope and gives three ways out: `splitk=1`,
   `splitk=(S, scheme)`, or recording the shape with `tools/record_heuristics.py` on the reference GPU.

Zero-size GEMMs (M, N or K = 0) resolve to S = 1: torch never calls cuBLASLt for them (`r6.zero_dims`). Profiles
without a table (`hopper-experimental`, `b200-mma-sync-experimental`) have no envelope: `splitk="auto"` always
raises there. `strict` defaults to `$MMAEMU_SPLITK_STRICT == "1"`.

`UnverifiedSplitWarning` is emitted once per combination when the (fast, S, scheme, out dtype, format pair)
combination was never replayed on hardware, when a `"contiguous"` split lies outside the replayed calls (a split
without k-tiles, an unlisted customOption; K not a multiple of 128 where the table does not cover it), or when S > 1
on a profile whose split decisions are experimental (the RTX 4060 TE profile: the reduction scheme of 46 TE split
outputs in the truth was filled from the table itself; the H100 TE profile: no split call ran through TE's own
GEMM).
`dry=True` (what `mmaemu.explain` uses) emits the warning on every call and does not count it as the one warning.

### Profiles

| profile | workspace | alignment | split maturity |
|---|---|---|---|
| `ada` | 1 MiB (torch's default; logged `workSpaceSizeInBytes=1048576` on every recorded call) | none | validated |
| `ada-te` (Transformer Engine 2.3, Evo 2) | 4 MiB | 256 B | experimental |
| `ada-ws0` | 0 | none | validated |
| `hopper` (default) | 1 MiB (logged on every recorded call) | none | validated |
| `hopper-te` (Transformer Engine 2.3, Evo 2) | 32 MiB (logged) | 256 B | experimental |
| `hopper-ws0` | 0 | none | validated |
| `hopper-experimental`, `b200-mma-sync-experimental` | n/a | n/a | none (no table) |

## Coverage

### Hardware verification of every combination

A (fast, S, scheme, out dtype, format pair) combination is *verified* when a hardware output with that logged or
captured split config exists and the audit FixedModel replays it bitwise.

- Truth recordings: 211 combinations (S = 2-16, 20, 24, 32). The 3 + 46 outputs whose split was filled from the
  table by the truth's `apply_ltconfig.py` are excluded.
- Of the truth table's 316 combinations, 156 (12,434 rows) had no hardware replay, among them the 3 with S = 29
  (recomputed by `tools/record/coverage.py`; list in `recordings/rtx4060_2026-10/r4_unverified_combos.json`).
- R4 (in `recordings/rtx4060_2026-10`) recorded the smallest torch-reachable row of each of them (and of the 153
  combinations that only the R2/R2x rows add) on randn*40 and split-structured adversarial data. The logged config
  equalled the table row in 465 of 465 calls; FixedModel replayed 3,576 of 3,576 stored values. In 281 of the 309
  combinations every alternative plan (other scheme, S-1, S+1, contiguous tile ownership, unsplit, slow chunks of
  length 64) changes a stored bit.
- Result: all 486 combinations of the shipped table are verified (`verified_split_combos.json`).

### Live checks against the table

| check | result |
|---|---|
| R4: logged (algo, tile, numSplitsK, customOption) vs table row | 465 / 465 equal |
| R5: 300 random in-envelope shapes at 1 MiB | 300 / 300 equal |
| R2-verify: 100 random fp8-output rows + e4m3 out at 224x176x80 and 176x224x80 | 112 / 112 equal |
| `tests/test_splitk_table.py -m profile_device`: 500 random rows re-queried live | 0 differ |
| `tools/record_heuristics.py` on the 4060 (74,304 queries) vs the shipped table | 0 differ |
| `tests/test_splitk_table.py -m truth`: in-table truth gemm outputs with a hardware-captured split | 22,306 / 22,306 agree |
| `tests/test_splitk_table.py -m profile_device`: shapes with M*N 2-4 x the largest split M*N of every setting of both profiles, K = 1,024..4,194,304 | 11,760 / 11,760 live S = 1 (on fa0fd90: [log](validation_logs/2026-10-16/rtx-4060-sm89/t3p_profile_device.log); after the profile rename: [log](validation_logs/2026-10-16/rtx-4060-sm89-renamed/t3p_profile_device.log)) |

### Evo 2

- (K, N) = (1920, 5760), (4096, 12288), (8192, 24576): no split for M = 16..8192 (step 16) at any workspace, and
  no split for M = 1..256 at 0, 1 and 4 MiB.
- R3 finding: (8192, 24576) splits (S = 14, customOption 2) at M = 65..74 with fast accumulation at **32 MiB**,
  for **fp16 and fp8 (e4m3 / e5m2) outputs** and every allowed format pair: 80 rows of the shipped table (R3: fp16
  out, e4m3 x e4m3, both alignments = 20 of its 55,296 rows; R2 / R2x: the fp8 outputs and the mixed formats). It
  does not split with fp32 or bf16 output, nor at 0, 1 or 4 MiB. Evo 2 uses bf16 output at 1 MiB (torch) or 4 MiB
  (TE), where every M resolves to S = 1.
- On `hopper` and `hopper-te`, `resolve_splitk` gives S = 1 for every Evo 2 shape with M = 1..256 and
  M = 16..16384 (step 16), from the table up to M = 8192 and from the envelope above it, and never raises.

### Split envelopes

Per setting: rows in the table, rows that split, the largest M*N and the smallest K among split rows, and the S
values that occur. A shape absent from the table resolves to S = 1 by inference only when its M*N is more than
twice the largest split M*N of its setting; otherwise it raises (unless the live query is available). Inside that
bound only part of the shapes split, so the envelope is not a predictor (audit claim 5). Workspace 0 has no split
rows (28 settings).

The smallest split K is reported but not used. Live sweeps of cuBLASLt's heuristic on the RTX 4060 (both table
profiles, every setting with split rows; [splitk-envelope](validation_logs/2026-10-14/splitk-envelope/README.md); run
again on fa0fd90 and after the profile rename with every row equal,
[log](validation_logs/2026-10-16/rtx-4060-sm89/README.md),
[log](validation_logs/2026-10-16/rtx-4060-sm89-renamed/README.md)) found:

- shapes absent from the table split at up to 1.27 times the largest split M*N of their setting (8,292 of 663,936
  queries with M*N between 1 and 4 times it; 52 of the 56 settings; K from 3,072 to 4,194,304);
- 18,504 of 125,419 queries with K below the smallest split K of the table (and M*N inside the envelope) split, at
  K down to 1,760: the table's K grid has gaps (for example none between 2,064 and 3,072);
- no split above 1.271 times the largest split M*N in 1,512,000 queries up to 17 times it.

Before this was measured, mmaemu inferred S = 1 for every shape above the largest split M*N or below the smallest
split K, which gives wrong bits for such shapes off the reference GPU.

| ws | formats | out | accumulation | rows | split rows | max split M*N | min split K | S values |
|---|---|---|---|---|---|---|---|---|
| 1 MiB | e4m3xe4m3 | float32 | fast | 18497 | 1907 | 100352 | 1824 | 2,3,4,5,6,7,8,9,10,12,16 |
| 1 MiB | e4m3xe4m3 | float32 | slow | 18497 | 3633 | 106496 | 16 | 2,3,4,5,6,7,8,12,14,24 |
| 1 MiB | e4m3xe4m3 | bfloat16 | fast | 18497 | 2264 | 100352 | 16 | 2,3,4,6,8,10,12,14 |
| 1 MiB | e4m3xe4m3 | bfloat16 | slow | 18497 | 1840 | 131072 | 2048 | 2,3,4,5,6,8,12,16,24 |
| 1 MiB | e4m3xe4m3 | float16 | fast | 18497 | 1851 | 135168 | 2048 | 2,3,4,5,6,7,8,10,14 |
| 1 MiB | e4m3xe4m3 | float16 | slow | 18497 | 3734 | 106496 | 16 | 2,3,4,5,6,8,10,12,14,24 |
| 1 MiB | e4m3xe4m3 | float8_e4m3fn | fast | 8915 | 347 | 131072 | 4096 | 2,3,4,5,6,7,8,10 |
| 1 MiB | e4m3xe4m3 | float8_e4m3fn | slow | 8915 | 1709 | 98304 | 16 | 2,3,4,5,6,8,10,12,14 |
| 1 MiB | e4m3xe5m2 | float32 | fast | 8915 | 334 | 98304 | 1920 | 2,3,4,5,6,7,8,9,12 |
| 1 MiB | e4m3xe5m2 | float32 | slow | 8915 | 1759 | 98304 | 16 | 2,3,4,5,6,8,12,14 |
| 1 MiB | e4m3xe5m2 | bfloat16 | fast | 8915 | 400 | 98304 | 16 | 2,3,4,6,8,10,12,14 |
| 1 MiB | e4m3xe5m2 | bfloat16 | slow | 8915 | 335 | 131072 | 2048 | 2,3,4,5,6,8,12,16 |
| 1 MiB | e4m3xe5m2 | float16 | fast | 8915 | 347 | 131072 | 4096 | 2,3,4,5,6,7,8,10 |
| 1 MiB | e4m3xe5m2 | float16 | slow | 8915 | 1709 | 98304 | 16 | 2,3,4,5,6,8,10,12,14 |
| 1 MiB | e4m3xe5m2 | float8_e4m3fn | fast | 8915 | 347 | 131072 | 4096 | 2,3,4,5,6,7,8,10 |
| 1 MiB | e4m3xe5m2 | float8_e4m3fn | slow | 8915 | 1709 | 98304 | 16 | 2,3,4,5,6,8,10,12,14 |
| 1 MiB | e4m3xe5m2 | float8_e5m2 | fast | 8915 | 347 | 131072 | 4096 | 2,3,4,5,6,7,8,10 |
| 1 MiB | e4m3xe5m2 | float8_e5m2 | slow | 8915 | 1709 | 98304 | 16 | 2,3,4,5,6,8,10,12,14 |
| 1 MiB | e5m2xe4m3 | float32 | fast | 8915 | 334 | 98304 | 1920 | 2,3,4,5,6,7,8,9,12 |
| 1 MiB | e5m2xe4m3 | float32 | slow | 8915 | 1759 | 98304 | 16 | 2,3,4,5,6,8,12,14 |
| 1 MiB | e5m2xe4m3 | bfloat16 | fast | 8915 | 400 | 98304 | 16 | 2,3,4,6,8,10,12,14 |
| 1 MiB | e5m2xe4m3 | bfloat16 | slow | 8915 | 335 | 131072 | 2048 | 2,3,4,5,6,8,12,16 |
| 1 MiB | e5m2xe4m3 | float16 | fast | 8915 | 347 | 131072 | 4096 | 2,3,4,5,6,7,8,10 |
| 1 MiB | e5m2xe4m3 | float16 | slow | 8915 | 1709 | 98304 | 16 | 2,3,4,5,6,8,10,12,14 |
| 1 MiB | e5m2xe4m3 | float8_e4m3fn | fast | 8915 | 347 | 131072 | 4096 | 2,3,4,5,6,7,8,10 |
| 1 MiB | e5m2xe4m3 | float8_e4m3fn | slow | 8915 | 1709 | 98304 | 16 | 2,3,4,5,6,8,10,12,14 |
| 1 MiB | e5m2xe4m3 | float8_e5m2 | fast | 8915 | 347 | 131072 | 4096 | 2,3,4,5,6,7,8,10 |
| 1 MiB | e5m2xe4m3 | float8_e5m2 | slow | 8915 | 1709 | 98304 | 16 | 2,3,4,5,6,8,10,12,14 |
| 4 MiB | e4m3xe4m3 | float32 | fast | 18497 | 2157 | 983040 | 16 | 2,3,4,5,6,8,9,10,12,14,16,29 |
| 4 MiB | e4m3xe4m3 | float32 | slow | 18497 | 2183 | 983040 | 1792 | 2,3,4,5,6,7,8,9,10,12,24 |
| 4 MiB | e4m3xe4m3 | bfloat16 | fast | 18497 | 2523 | 983040 | 1664 | 2,3,4,5,6,8,9,10,12,14,16 |
| 4 MiB | e4m3xe4m3 | bfloat16 | slow | 18497 | 2322 | 983040 | 1728 | 2,3,4,5,6,8,9,10,12,24 |
| 4 MiB | e4m3xe4m3 | float16 | fast | 18497 | 3591 | 983040 | 16 | 2,3,4,5,6,8,9,10,12,14,16 |
| 4 MiB | e4m3xe4m3 | float16 | slow | 18497 | 4105 | 983040 | 16 | 2,3,4,5,6,8,9,10,12,14,24 |
| 4 MiB | e4m3xe4m3 | float8_e4m3fn | fast | 8915 | 1684 | 983040 | 16 | 2,3,4,5,6,8,9,10,12,14 |
| 4 MiB | e4m3xe4m3 | float8_e4m3fn | slow | 8915 | 1791 | 983040 | 16 | 2,3,4,5,6,8,9,10,12,14 |
| 4 MiB | e4m3xe5m2 | float32 | fast | 8915 | 417 | 786432 | 16 | 2,3,4,5,6,8,10,12,14,29 |
| 4 MiB | e4m3xe5m2 | float32 | slow | 8915 | 440 | 983040 | 1920 | 2,3,4,5,6,7,8,9,10,12 |
| 4 MiB | e4m3xe5m2 | bfloat16 | fast | 8915 | 476 | 983040 | 1920 | 2,3,4,5,6,8,9,10,12,14,16 |
| 4 MiB | e4m3xe5m2 | bfloat16 | slow | 8915 | 471 | 983040 | 1920 | 2,3,4,5,6,8,10,12 |
| 4 MiB | e4m3xe5m2 | float16 | fast | 8915 | 1684 | 983040 | 16 | 2,3,4,5,6,8,9,10,12,14 |
| 4 MiB | e4m3xe5m2 | float16 | slow | 8915 | 1791 | 983040 | 16 | 2,3,4,5,6,8,9,10,12,14 |
| 4 MiB | e4m3xe5m2 | float8_e4m3fn | fast | 8915 | 1684 | 983040 | 16 | 2,3,4,5,6,8,9,10,12,14 |
| 4 MiB | e4m3xe5m2 | float8_e4m3fn | slow | 8915 | 1791 | 983040 | 16 | 2,3,4,5,6,8,9,10,12,14 |
| 4 MiB | e4m3xe5m2 | float8_e5m2 | fast | 8915 | 1684 | 983040 | 16 | 2,3,4,5,6,8,9,10,12,14 |
| 4 MiB | e4m3xe5m2 | float8_e5m2 | slow | 8915 | 1791 | 983040 | 16 | 2,3,4,5,6,8,9,10,12,14 |
| 4 MiB | e5m2xe4m3 | float32 | fast | 8915 | 417 | 786432 | 16 | 2,3,4,5,6,8,10,12,14,29 |
| 4 MiB | e5m2xe4m3 | float32 | slow | 8915 | 440 | 983040 | 1920 | 2,3,4,5,6,7,8,9,10,12 |
| 4 MiB | e5m2xe4m3 | bfloat16 | fast | 8915 | 476 | 983040 | 1920 | 2,3,4,5,6,8,9,10,12,14,16 |
| 4 MiB | e5m2xe4m3 | bfloat16 | slow | 8915 | 471 | 983040 | 1920 | 2,3,4,5,6,8,10,12 |
| 4 MiB | e5m2xe4m3 | float16 | fast | 8915 | 1684 | 983040 | 16 | 2,3,4,5,6,8,9,10,12,14 |
| 4 MiB | e5m2xe4m3 | float16 | slow | 8915 | 1791 | 983040 | 16 | 2,3,4,5,6,8,9,10,12,14 |
| 4 MiB | e5m2xe4m3 | float8_e4m3fn | fast | 8915 | 1684 | 983040 | 16 | 2,3,4,5,6,8,9,10,12,14 |
| 4 MiB | e5m2xe4m3 | float8_e4m3fn | slow | 8915 | 1791 | 983040 | 16 | 2,3,4,5,6,8,9,10,12,14 |
| 4 MiB | e5m2xe4m3 | float8_e5m2 | fast | 8915 | 1684 | 983040 | 16 | 2,3,4,5,6,8,9,10,12,14 |
| 4 MiB | e5m2xe4m3 | float8_e5m2 | slow | 8915 | 1791 | 983040 | 16 | 2,3,4,5,6,8,9,10,12,14 |
| 32 MiB | e4m3xe4m3 | float32 | fast | 18497 | 2205 | 6389760 | 16 | 2,3,4,5,6,8,9,10,12,14,16,29 |
| 32 MiB | e4m3xe4m3 | float32 | slow | 18497 | 2263 | 7864320 | 1792 | 2,3,4,5,6,7,8,9,10,12,24 |
| 32 MiB | e4m3xe4m3 | bfloat16 | fast | 18497 | 2546 | 1216768 | 1664 | 2,3,4,5,6,8,9,10,12,14,16 |
| 32 MiB | e4m3xe4m3 | bfloat16 | slow | 18497 | 2349 | 4194304 | 1728 | 2,3,4,5,6,8,9,10,12,24 |
| 32 MiB | e4m3xe4m3 | float16 | fast | 18497 | 3696 | 6496512 | 16 | 2,3,4,5,6,8,9,10,12,14,16 |
| 32 MiB | e4m3xe4m3 | float16 | slow | 18497 | 4192 | 6406400 | 16 | 2,3,4,5,6,8,9,10,12,14,24 |
| 32 MiB | e4m3xe4m3 | float8_e4m3fn | fast | 8915 | 1733 | 6291456 | 16 | 2,3,4,5,6,8,9,10,12,14 |
| 32 MiB | e4m3xe4m3 | float8_e4m3fn | slow | 8915 | 1825 | 6291456 | 16 | 2,3,4,5,6,8,9,10,12,14 |
| 32 MiB | e4m3xe5m2 | float32 | fast | 8915 | 429 | 2097152 | 16 | 2,3,4,5,6,8,10,12,14,29 |
| 32 MiB | e4m3xe5m2 | float32 | slow | 8915 | 469 | 5898240 | 1920 | 2,3,4,5,6,7,8,9,10,12 |
| 32 MiB | e4m3xe5m2 | bfloat16 | fast | 8915 | 485 | 1048576 | 1920 | 2,3,4,5,6,8,9,10,12,14,16 |
| 32 MiB | e4m3xe5m2 | bfloat16 | slow | 8915 | 483 | 4194304 | 1920 | 2,3,4,5,6,8,10,12 |
| 32 MiB | e4m3xe5m2 | float16 | fast | 8915 | 1733 | 6291456 | 16 | 2,3,4,5,6,8,9,10,12,14 |
| 32 MiB | e4m3xe5m2 | float16 | slow | 8915 | 1825 | 6291456 | 16 | 2,3,4,5,6,8,9,10,12,14 |
| 32 MiB | e4m3xe5m2 | float8_e4m3fn | fast | 8915 | 1733 | 6291456 | 16 | 2,3,4,5,6,8,9,10,12,14 |
| 32 MiB | e4m3xe5m2 | float8_e4m3fn | slow | 8915 | 1825 | 6291456 | 16 | 2,3,4,5,6,8,9,10,12,14 |
| 32 MiB | e4m3xe5m2 | float8_e5m2 | fast | 8915 | 1733 | 6291456 | 16 | 2,3,4,5,6,8,9,10,12,14 |
| 32 MiB | e4m3xe5m2 | float8_e5m2 | slow | 8915 | 1825 | 6291456 | 16 | 2,3,4,5,6,8,9,10,12,14 |
| 32 MiB | e5m2xe4m3 | float32 | fast | 8915 | 429 | 2097152 | 16 | 2,3,4,5,6,8,10,12,14,29 |
| 32 MiB | e5m2xe4m3 | float32 | slow | 8915 | 469 | 5898240 | 1920 | 2,3,4,5,6,7,8,9,10,12 |
| 32 MiB | e5m2xe4m3 | bfloat16 | fast | 8915 | 485 | 1048576 | 1920 | 2,3,4,5,6,8,9,10,12,14,16 |
| 32 MiB | e5m2xe4m3 | bfloat16 | slow | 8915 | 483 | 4194304 | 1920 | 2,3,4,5,6,8,10,12 |
| 32 MiB | e5m2xe4m3 | float16 | fast | 8915 | 1733 | 6291456 | 16 | 2,3,4,5,6,8,9,10,12,14 |
| 32 MiB | e5m2xe4m3 | float16 | slow | 8915 | 1825 | 6291456 | 16 | 2,3,4,5,6,8,9,10,12,14 |
| 32 MiB | e5m2xe4m3 | float8_e4m3fn | fast | 8915 | 1733 | 6291456 | 16 | 2,3,4,5,6,8,9,10,12,14 |
| 32 MiB | e5m2xe4m3 | float8_e4m3fn | slow | 8915 | 1825 | 6291456 | 16 | 2,3,4,5,6,8,9,10,12,14 |
| 32 MiB | e5m2xe4m3 | float8_e5m2 | fast | 8915 | 1733 | 6291456 | 16 | 2,3,4,5,6,8,9,10,12,14 |
| 32 MiB | e5m2xe4m3 | float8_e5m2 | slow | 8915 | 1825 | 6291456 | 16 | 2,3,4,5,6,8,9,10,12,14 |
## Recording a table for another GPU

```
$MMAEMU_GPU_LOCK timeout 600 python tools/record_heuristics.py \
    --grid c5,evo2,evo2_small_m --ws 0,1,4,32 --out results/bringup/<run>/heur_sm120.npz
```

The tool queries `mmaemu.platform.cublaslt.heuristic` with torch's descriptor (no kernel runs). It writes the same
columns as the tables above, and a JSON sidecar with the GPU, SM count, compute capability, driver, cuBLASLt version
and library path, per-setting envelopes, the Evo 2 split status per workspace, and a candidate profile name such as
`rtx5070ti-70sm-lt<version>`. It never overwrites an existing output. On the 4060 its 74,304 rows equal the shipped
table. A new table is candidate data: it ships only as an experimental profile after review, and its split
combinations need hardware replays like R4 before they count as verified (runbook gate G6,
`docs/runbooks/sm120_bringup.md`).

## Evidence commands

```
export PYTHONPATH=<repo> PYTHONDONTWRITEBYTECODE=1                # environment: docs/testing.md
python tools/export_splitk_table.py --check                       # byte-identical re-export (needs the truth)
python tools/record/coverage.py                                   # 316 / 156 / 12,434 / S=29
python -m pytest -q tests/test_splitk_table.py tests/test_profiles.py
MMAEMU_TRUTH_DIR=<truth checkout> python -m pytest -q -m truth tests/test_splitk_table.py
$MMAEMU_GPU_LOCK timeout 600 python -m pytest -q -m profile_device tests/test_splitk_table.py
python tools/record/replay_fixed.py                               # FixedModel over R1-R7 (maintainer only)
```
