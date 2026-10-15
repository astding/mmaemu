# Split-K tables

cuBLAS decides per GEMM whether to split the K dimension (`numSplitsK = S`) and how to combine the partial sums
(`customOption`). This is a heuristic decision of the GPU, the cuBLASLt version, the workspace size, the output
dtype and fast/slow accumulation. It cannot be derived from the shape alone (audit C5-F6 / C4-F4). mmaemu therefore
reproduces cuBLAS's split-K numerics only on a **profile's reference platform**, from a table recorded there.

This page describes the table shipped with mmaemu v0.1, how `resolve_splitk` uses it, what it covers, and how to
record a table for another GPU.

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
| package data | `mmaemu/platform/data/heur_rtx4060_lt12.8.3.14.npz` (236,342 B, sha256 `304203ef...`), `.json` (sources, envelopes), `verified_split_combos.json`; 402,715 B in total |

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

## Resolution: `resolve_splitk`

`mmaemu.platform.splitk_table.resolve_splitk(M, N, K, *, a_dtype, b_dtype, out_dtype, use_fast_accum, profile=None,
splitk="auto", layout_std=True, strict=None)` returns a `SplitDecision(S, scheme, source, verified, table_id,
note)`. The order is:

1. **explicit**: `splitk=1`, `SplitK(S, scheme)` or `(S, scheme)`; an int > 1 without a scheme raises ValueError;
2. **workspace0**: the profile's workspace is 0 -> S = 1 (0 split rows among 209,712 workspace-0 queries);
3. **table**: the key is in the table -> the recorded (numSplitsK, customOption); customOption 0 and 2 mean
   "serial", 1 means "pairwise";
4. **live-query**: the current GPU is the profile's reference platform (`Profile.matches_device`: GPU name, SM
   count and cuBLASLt version) -> `mmaemu.platform.cublaslt.heuristic` is queried with the profile's workspace and
   alignment;
5. **envelope**: outside the split envelope of the setting (M*N above the largest M*N that split, or K below the
   smallest K that split) and not strict -> S = 1. This is an inference; the note says so;
6. otherwise `SplitKUnknownError`, which names the setting and its envelope and gives three ways out: `splitk=1`,
   `splitk=(S, scheme)`, or recording the shape with `tools/record_heuristics.py` on the reference GPU.

Zero-size GEMMs (M, N or K = 0) resolve to S = 1: torch never calls cuBLASLt for them (`r6.zero_dims`). Profiles
without a table (`hopper-experimental`, `b200-mma-sync-experimental`) have no envelope: `splitk="auto"` always
raises there. `strict` defaults to `$MMAEMU_SPLITK_STRICT == "1"`.

`UnverifiedSplitWarning` is emitted once per combination when the (fast, S, scheme, out dtype, format pair)
combination was never replayed on hardware, or when S > 1 on a profile whose split decisions are experimental (the
TE profile: the reduction scheme of 46 TE split outputs in the truth was filled from the table itself).

### Profiles

| profile | workspace | alignment | split maturity |
|---|---|---|---|
| `ada-rtx4060-torch` (default) | 1 MiB (torch's default; logged `workSpaceSizeInBytes=1048576` on every recorded call) | none | validated |
| `ada-rtx4060-te` (Transformer Engine 2.3, Evo 2) | 4 MiB | 256 B | experimental |
| `ada-rtx4060-ws0` | 0 | none | validated |
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

### Evo 2

- (K, N) = (1920, 5760), (4096, 12288), (8192, 24576): no split for M = 16..8192 (step 16) at any workspace, and
  no split for M = 1..256 at 0, 1 and 4 MiB.
- R3 finding: (8192, 24576) splits (S = 14, customOption 2) at M = 65..74 with fast accumulation at **32 MiB**,
  for **fp16 and fp8 (e4m3 / e5m2) outputs** and every allowed format pair: 80 rows of the shipped table (R3: fp16
  out, e4m3 x e4m3, both alignments = 20 of its 55,296 rows; R2 / R2x: the fp8 outputs and the mixed formats). It
  does not split with fp32 or bf16 output, nor at 0, 1 or 4 MiB. Evo 2 uses bf16 output at 1 MiB (torch) or 4 MiB
  (TE), where every M resolves to S = 1.
- On the `-torch` and `-te` profiles, `resolve_splitk` gives S = 1 for every Evo 2 shape with M = 1..256 and
  M = 16..16384 (step 16), from the table up to M = 8192 and from the envelope above it, and never raises.

### Split envelopes

Per setting: rows in the table, rows that split, the largest M*N and the smallest K among split rows, and the S
values that occur. Shapes outside the envelope resolve to S = 1 by inference; shapes inside it but absent from the
table raise (unless the live query is available). Inside the envelope only part of the shapes split, so the envelope
is not a predictor (audit claim 5). Workspace 0 has no split rows (28 settings).

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
