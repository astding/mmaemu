recordings/rtx4060_2026-10 -- mmaemu WS2 pre-swap recordings on the reference GPU
==================================================================================

Hardware/software of every recording (live metadata is in each recording's "hardware" field):
  NVIDIA GeForce RTX 4060, sm_89, 24 SMs, driver 572.16, WSL2 (Linux 5.15.167.4-microsoft-standard-WSL2),
  torch 2.7.1+cu128, CUDA 12.8, cuBLASLt 12.8.3.14 (the nvidia-cublas-cu12 wheel torch loads).
Recorded 2026-10-12 by mmaemu workstream WS2 (SPEC v0.1, section 11 WS2, items R1-R6), before the GPU swap.

Format and layout
-----------------
Format "tcemu_ada_truth/1", written with the truth package's own writer
(/data/quant/tcemu_ada_truth/tcemu_ada_truth/format.py, RecordingWriter / write_table, root=<this dir>/data, every
npz part <= 15 MB) and imported read-only with PYTHONDONTWRITEBYTECODE=1. Inputs are stored raw (no generators).
  manifest.json                 every recording: id, json path, kind, level, counts, every file with bytes + sha256
  data/<group>/<name>.json      one recording's metadata;  data/<group>/<name>[.partN].npz  its arrays (bit patterns)
  gpu_runs.jsonl                every locked GPU run of the recorders (start, end, seconds inside the lock)
  replay_fixed.json             the audit FixedModel replay of these recordings (acceptance A3; see below)
  r4_unverified_combos.json     the recomputed 156-combination list (+ S=29, + the 153 WS2-only combinations), the
                                job built for each, and whether every alternative plan was distinguished
  r1_r6_summary.json, r4_summary.json, r5_summary.json, r2v_summary.json   per-item call summaries
Load:
  PYTHONPATH=/data/quant/tcemu_ada_truth python -c "from tcemu_ada_truth import loader; \
      r = loader.load('r4.unverified_split_combos', root='recordings/rtx4060_2026-10'); print(r.cases[0].inputs())"
Check (acceptance A1): python tools/record/check_loads.py -> "9 recordings load with sha256 OK: 4815 inputs, 2184
stored outputs, 9395 hw_error outputs, 4 table payloads".

Recordings
----------
id                                         kind   content
r2.lt_heur_fp8out_12_8_3_14                table  R2: 1,711,680 heuristic queries, fp8 outputs (D = e4m3/e5m2, C = fp16
                                                  as torch passes it): ws {0,1,4,32 MiB} x fast/slow x align {none,256}
                                                  x D-scale pointer {off,on} x (out, pair) {e4m3: e4m3xe4m3, e4m3xe5m2,
                                                  e5m2xe4m3; e5m2: e4m3xe5m2, e5m2xe4m3, e4m3xe4m3}; shapes: c5 grid +
                                                  small-K set (5,372) + Evo 2 grid (M 16..8192 step 16) + Evo 2 M 1..256
r2.lt_heur_mixed_allsettings_12_8_3_14     table  R2x: 855,840 queries, mixed formats x fp32/bf16/fp16 out at every ws x
                                                  align (the truth table has mixed formats only at (1 MiB, none) and
                                                  (4 MiB, 256))
r2.lt_heur_smallk_12_8_3_14                table  R2x: 257,856 queries, e4m3xe4m3 small-K set x the 48 main settings
r3.lt_heur_evo2_small_m_12_8_3_14          table  R3: Evo 2 (K,N) in {(1920,5760),(4096,12288),(8192,24576)} x M = 1..256
                                                  (every integer) x the 72 settings of heur.lt_heuristic_12_8_3_14
r2.fp8out_live_verify                      gemm   R2-verify: 100 random R2 rows live (scale_result passed on 50) +
                                                  critic gap 17 (e4m3 out at 224x176x80 / 176x224x80, 12 calls)
r4.unverified_split_combos                 gemm   R4: 465 calls -- the smallest torch-reachable table row of each of the
                                                  156 unverified (fast, S, scheme, out, fmt pair) combinations of the
                                                  truth table (incl. the 3 S=29 ones) on randn*40 and split-structured
                                                  adversarial data, + 153 combinations only the WS2 rows add (adversarial)
r5.autosplit_in_envelope                   gemm   R5: 300 random in-envelope table shapes at 1 MiB (28 out/fast/format
                                                  settings; 217 split rows), randn*40 / adversarial
r1.api_parity_grid                         gemm   R1: 10,352 torch._scaled_mm calls (API parity grid; see
                                                  tools/record/r1_api_grid.py): raise/no-raise, error class + first
                                                  line, returned dtype/shape/strides/requires_grad, bits, cuBLASLt config
r6.zero_dims                               gemm   R6: 280 zero-dimension calls (M/N/K = 0, every out dtype x format pair x
                                                  fast/slow, real scales, scale_result for fp8 out) + K = 0 with scale_a
                                                  in {+1, -1, +0, -0, NaN, +Inf, -Inf} (70)
R7 (optional c4.V2L layouts with numSplitsK): not recorded.

Inputs of the live items are generated by deterministic integer hashing (tools/record/live.py make_codes): "randn40"
= codes distributed as RNE_fp8(clamp(40*N(0,1))); "adv" = per-tile magnitudes 2^[-6,5]; "adv_split" = split-structured
(split 0 and split S-1 own exactly mirrored large tiles, p_{S-1} = -p_0, so the result is decided by rounding at the
large scale: the reduction order, tile ownership and S become visible in bf16/fp16/fp8 outputs too). The GPU call uses
the full operands; R2v/R4/R5 store a row/column subset with full K (case params hw_shape + sub) chosen on CPU so that
the alternatives (other scheme, S-1, S+1, contiguous tiles, unsplit, slow restart every 64) change a stored bit;
params.discrimination records the outcome. CPU == GPU input bytes were checked for every call (877/877).
Every output's params.splitk / params.cublas is the LOGGED config of that very call (CUBLASLT_LOG_LEVEL=2 trace,
delivered through cublasLtLoggerSetCallback; the file log is kept in the stage dir), never the table.

Progress log (locked GPU runs; seconds inside the lock, from gpu_runs.jsonl)
-----------------------------------------------------------------------------
2026-10-12 13:45:54  R2 heuristic queries                          36.4 s
2026-10-12 13:47:18  R2x + R3 heuristic queries                    22.2 s
2026-10-12 14:07:36  R4 live calls (465)                           16.7 s
2026-10-12 14:11:01  R5 live calls (300)                            6.0 s
2026-10-12 14:11:14  R2-verify live calls (112)                     9.6 s
2026-10-12 14:17:26  R1 smoke test (902 non-grid calls; superseded) 41.0 s
2026-10-12 14:19:21  R1/R6 calls 0..3599                           78.2 s
2026-10-12 14:28:07  R1/R6 calls 3600..7199                       186.6 s
2026-10-12 14:32:35  R1/R6 calls 7200..10701                       67.8 s
2026-10-12 14:35:40  R1 layout calls re-recorded (150)              1.8 s
Longest locked run: 186.6 s (< 10 min). Total GPU time of the recorders: 466 s. gpu_runs.jsonl also lists, as
"logged after the fact", the short runs made through gpu.sh directly (three probes before the recorders existed,
the first A8 pytest run, the record_heuristics.py test; each < 1 min) and three invocations that failed at argument
parsing (0.2 s each; their original lines were removed by mistake and restored as one note).
Re-recording: the first R1 run built mat2's padded-ld / storage-offset variants row-major instead of column-major (a
recorder bug, visible in params.mat_meta strides); those 150 layout calls were re-run after the fix and the re-run
replaces them (r1_r6_summary.json "re_recorded").

Acceptance A2: logged config == table row (R2v, R4, R5)
-------------------------------------------------------
Every call's logged (algoId, tile, numSplitsK, customOption) was compared with the heuristic-table row it was built
from (params.config_check):  R4 465/465,  R5 300/300,  R2-verify 112/112 equal; 0 discrepancies, 0 calls without a
log line, logged workSpaceSizeInBytes == requested CUBLASLT_WORKSPACE_SIZE in 877/877 calls. R4's 78 combinations whose
smallest truth row is align 256 (TE descriptor) were recorded at the align-none row of the union table (torch's
descriptor); alignment never changes a decision (0 of 1,422,624 pairs, docs/splitk_tables.md).

Acceptance A3: audit FixedModel replay (tools/record/replay_fixed.py, own process, the only tcemu import)
--------------------------------------------------------------------------------------------------------
tcemu_ada_truth.models.FixedModel, replay loop of tcemu_ada_truth.replay with root=this dir, fastacc from a private
build (the truth's fastacc.lib() is never called), PYTHONDONTWRITEBYTECODE=1. Report: replay_fixed.json. 9.5 s.
recording                       stored values   match         hw_error reproduced   notes
r2.fp8out_live_verify               477,670     477,670 100%       -
r4.unverified_split_combos            3,576       3,576 100%       -               all 309 combinations incl. S=29
r5.autosplit_in_envelope              9,110       9,110 100%       -
r1.api_parity_grid                  134,720     134,720 100%   2,934 / 9,319      (a) (b) (c)
r6.zero_dims                         25,088      17,408         70 / 70            (b) (d)
total                               650,164     642,484        3,004 / 9,389
(a) 6,385 hw_error outputs not reproduced: torch 2.7.1 argument checks the FixedModel does not model (K % 16, N % 16,
    row/column-major layout, storage offset / leading dimension -> CUBLAS_STATUS_NOT_SUPPORTED, fp8 dtype checks incl.
    fnuz (torch INTERNAL ASSERT in CUDADataType.h), scale dtype/shape, bias length/dtype, out= dtype). They are the
    authority for mmaemu's ReferenceRejectsError rules (WS1); the FixedModel is not.
(b) FINDING: e4m3 x e4m3 -> e5m2 output is rejected (CUBLAS_STATUS_NOT_SUPPORTED) only when cuBLASLt is called; for
    zero-size GEMMs (M, N or K = 0) torch returns without calling cuBLASLt and the call succeeds (44 R1 + 14 R6
    outputs: FixedModel "rejects but hardware ran"). The profile reject rule must apply only when M, N, K > 0.
(c) n/a 6: row-wise scales with a non-bf16 output (FixedModel's row-wise path is bf16-only; torch rejects them).
(d) FINDING: K = 0 returns +0 everywhere for every scale_a in {+1, -1, +0, -0, NaN, +Inf, -Inf}, every out dtype and
    fast/slow, and no cuBLASLt call is made: the epilogue is NOT applied (SPEC section 3.2 "K=0 gives acc = +0
    everywhere, and the epilogue then applies" holds for finite scales only). FixedModel applies the epilogue and
    predicts NaN for NaN/+-Inf scales: 30 outputs x 256 values = 7,680 value mismatches, all in the cases
    zeroK_scale_{nan,inf,minf}_*. Zero-size outputs (M or N = 0) are empty tensors of the requested dtype.

Other findings (measured; details in docs/splitk_tables.md)
-----------------------------------------------------------
- R3: the Evo 2 40B projection (K,N) = (8192, 24576) SPLITS (S = 14, customOption 2) at M = 65..74 with fast
  accumulation at 32 MiB for fp16 output (R3: 20 of its 55,296 rows, both alignments) -- and, in the R2 / R2x rows,
  for e4m3 / e5m2 outputs and the mixed format pairs too (80 rows of the exported table). Never with fp32 / bf16
  output, never at 0 / 1 / 4 MiB. Evo 2 itself uses bf16 output at 1 MiB (torch) or 4 MiB (TE): S = 1 there for
  every M = 1..256.
- R2: the fp8-output decision does not depend on the D scale pointer (0 of 855,840 pairs) nor on the alignment
  preference (0 of 427,920 pairs). e4m3 x e4m3 -> e5m2: status 15 (CUBLAS_STATUS_NOT_SUPPORTED) for every one of the
  71,320 queried shapes/settings. Critic gap 17: e4m3 out at 224x176x80 and 176x224x80 -> S = 1 at every workspace,
  format pair and fast/slow; confirmed live (gap17_*).
- R1: storage offsets of 1/2/4/8 B on either operand and a leading dimension of K+1 are rejected by cuBLASLt
  (CUBLAS_STATUS_NOT_SUPPORTED); 16/32 B offsets and ld = K+16 run. mat1 column-major or mat2 row-major: "Only
  multiplication of row-major and column-major matrices is supported by cuBLASLt". M has no divisibility rule; K and N
  must be multiples of 16. The default out_dtype is mat1's dtype. out= with a wrong shape is resized (deprecation
  warning) and the result is written.

Size and integrity
------------------
du -sh: 36 MB (limit 40 MB); largest file data/r4/unverified_split_combos.npz 10.9 MB (limit 20 MB).
The truth dir was not written by WS2 (acceptance A5):
- `GIT_OPTIONAL_LOCKS=0 git -C /data/quant/tcemu_ada_truth status --short` is empty.
- `find /data/quant/tcemu_ada_truth -newer <stamp>` is empty for the stamps taken right before each FixedModel
  replay and before the truth-marked test runs. For the session-wide stamps (taken before the first recorder run) it
  lists exactly one entry: the directory /data/quant/tcemu_ada_truth/.git itself, mtime 14:08:11 -- a lock file
  created and removed by a git process (every WS2 git call on the truth after the stamp used GIT_OPTIONAL_LOCKS=0;
  .git/index is unchanged since 03:12). With .git pruned (`find $TRUTH -path $TRUTH/.git -prune -o -newer <stamp> -print`) it is empty.
