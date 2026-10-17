# H100 PCIe bit-exactness probe (Phase A), 15 October 2026

Question: can mmaemu reproduce cuBLASLt's FP8 GEMMs on an H100 bit for bit, as it does on the RTX 4060? Until this
run the `hopper` preset rested on instruction-level vectors recorded with c = 0, which cannot tell how the
accumulator enters, and on nothing at the GEMM level.

**Verdict: GO.** All six bitwise criteria fixed before the rental are met (below). Two results were found after the
data came in and were not pre-registered: the split-K rule, and the reading of the GO #1 tie as one class. Both are
marked as such.

## Setup

- GPU: H100 PCIe (sm_90, 114 SMs, 80 GB), driver 570.148.08, one instance rented for this run (Lambda), 2026-10-15
    19:08 to 20:04 PDT. The measurements took 14 minutes (`results/status.txt`, times in UTC).
- Software: Python 3.12.15, torch 2.7.1+cu128, cuBLAS / cuBLASLt 12.8.3.14 (`libcublasLt.so.12` sha256 `4ef47ce6…`,
    `results/env/env.json`), Transformer Engine 2.3.0 installed by `scripts/te_setup.sh` (as on 14 and 15 October).
- Code: the probe bundle (sha256 `fee1a8cc…`) built by `scripts/build_bundle.sh`: mmaemu at a79668a
    (`git archive`), the c6 vector sets `MM_cublas64`, `MM_cublas64_sm120` and `MM_sm90mma`, the scripts in
    `scripts/probe/` and `scripts/run_all.sh`. Every GPU step ran under a lock with a 10-minute limit, every call was
    logged with its cuBLASLt configuration (`CUBLASLT_LOG_LEVEL=2`) and its kernel names, and every call ran twice;
    the live recorder jobs ran twice in separate processes.
- Inputs: the 21 FP8 projection inputs of Evo 2 1B on the w2k window (the inputs of the RTX 4060 truth), 1,450 live
    `torch._scaled_mm` jobs (`scripts/m4_jobs/` and the M6 jobs built on the GPU from its own heuristic table), the c6
    vector sets embedded in the production shapes, and pattern probes for the slow-accumulation promotion.

Steps (`scripts/run_all.sh`):

| Step | What it records |
|---|---|
| M1 | how c enters: the `MM_cublas64` (445 vectors, 133 Hopper variants) and `MM_cublas64_sm120` (834 vectors, 12,881 models) sets through cuBLAS, at K = 64 and embedded in the three Evo 2 shapes (the last 32 k-slots hold the vector, k-slot 0 makes c), fast and slow, torch and TE; the stored n × n block (all a_i × b_j pairs); the instruction-level `mma.sync` kit (`MM_sm90mma`, `TBX_H100c0`, `TBX_B200`) compiled for sm_90a on the host |
| M2 | the 21 Evo 2 projections, full outputs: fp32 accumulator and bf16 output, fast and slow, torch and TE |
| M3 | the promotion of slow accumulation: rows with at most three nonzero products at chosen k (pairs, triples, a bf16 tie), at the Evo 2 shapes and 1024 × 1024 × 4096 |
| M4 | 450 live jobs × 2 passes: random data, all output dtypes and format pairs, workspaces 0 / 1 / 4 / 32 MiB, long K (to 16,384), small K (16 to 96) |
| M5 | the epilogue ties kit (64 discriminating cases) and the zeros / specials probes (150 rows) |
| M6 | the cuBLASLt heuristic over the c5 / Evo 2 / Evo 2 small-M grids (850,320 queries), then 1,000 live split-K jobs built from its split rows (adversarial and random data) |
| M7 | environment (`results/env/`) |

## Dry runs (before the rental)

The bundle ran on the RTX 4060 (three times) and on an RTX 5070 Ti (twice). On the RTX 4060 the offline analysis
reproduces the existing model end to end (`results/dry_runs/rtx4060_analysis_ada_R128.json`): only `ada` fits M1,
M3 gives a restart every 128 with sequential combination, M2 is 21 of 21 for all eight outputs, M4 663 of 663 (27
calls rejected by cuBLASLt). `MM_cublas64` gives `ada` 445 of 445.

One RTX 4060 dry run had one job (`m4_0184`, pass 1) whose GPU-generated inputs differed from their CPU
regeneration; four reruns of its chunk equal pass 2. Before any H100 data existed (19:10 PDT) the plan was amended:
a job with an input fault is judged on its clean pass, determinism compares only pairs with clean inputs in both
passes, and a fault in both passes counts as a failure. The H100 run had no input faults.

## Results

**Determinism.** Every recorded call gave the same bits twice: all M1 / M2 / M3 calls, and 1,450 of 1,450 live jobs
across processes (`results/summary.json`). The 54 recorder errors are the 27 e4m3 × e4m3 → e5m2 jobs per pass, which
cuBLASLt rejects (status 15, CUBLAS_STATUS_NOT_SUPPORTED); the heuristic rejects exactly this combination and no
other (56,688 queries, every shape). All GEMMs ran cuBLAS's `nvjet_*` kernels (algo 66, k-stage 128).

**M1, how c enters** (`results/m1_kit/kit.json`, `results/m1/`, `results/analysis/m1.json`).

- `MM_cublas64`, E4M3, fast and slow, K = 64: the `hopper` preset 445 of 445 (`ada` 383, the B200 model 106). Eight
    of the 133 variants fit: flat 32-product blocks, early chain, c entering the first block's alignment, 13 fraction
    bits; they differ only in `zero_exp`, `min_exp` and `c_exp_sub`.
- Every stored n × n block (40 calls; 198,025 or 695,556 values each): only `hopper` with c in the same chain matches
    all values in fast mode and at K = 64; in slow mode at K ≥ 1920 only `hopper` with c in an earlier chain (the
    128-element restart) does. TE: `use_split_accumulator=False` behaves as fast, `True` as slow.
- The instruction-level `mma.sync` on sm_90 is a different unit: `MM_sm90mma` fits 25 fraction bits, c added late
    with RNE, interleaved pairs (16 models, 488 / 461 vectors) and the paper's `TBX_H100c0` vectors fit the 25-bit
    B200 model. cuBLAS does not use `mma.sync`; mmaemu models the cuBLAS path.

**M2, the 21 Evo 2 projections** (`results/analysis/m2_hopper_*.json`): `hopper` with R = 128 reproduces every
output bit for bit: fp32 accumulator and bf16 output, fast and slow, torch and TE, 168 of 168 full outputs (the
largest is 2048 × 24576). None of these calls splits K.

**M3, slow accumulation** (`results/analysis/m3.json`): fast is one chain over K; slow restarts every 128 and adds
the chunk results in K order (the triples rule out a tree). No column dependence (colvar 0 everywhere).

**M4, live jobs** (`results/analysis/m4_hopper_R128.json`): every unsplit job equals `hopper` / R = 128 bit for bit,
186 fast and 188 slow (random data, all output dtypes, format pairs, workspaces, long and small K).

**M5** (`results/m5/`): in all 64 epilogue tie cases (16384 × 768 outputs each, bf16 and fp16, fast and slow) the
hardware equals the epilogue model, RN(RN32(acc · RN32(sa · sb))), and every case was chosen so that the
alternatives (a single rounding, or (acc · sa) · sb) give a different output somewhere; the 150 zero / specials
rows (signed zeros, NaN, Inf, overflow, underflow) equal mmaemu's reference on CPU and on GPU.

**M6, heuristic and split-K** (`results/m6_heur/`): 51,270 of 850,320 queries split K, all at K ≥ 5760 with S = 2
to 6, `REDUCTION_SCHEME_COMPUTE_TYPE`; **no Evo 2 (K, N) shape splits at any recorded M or workspace** (0, 1, 4,
32 MiB).

## Split-K (found after the data came in)

`scripts/probe/splitk_search.py` scored, against the stored outputs of all 1,049 live split jobs (S = 2 to 6,
customOption 0 to 3, K from 5760 to 16,384), 64 hypotheses for fast and 48 for slow: interleaved or contiguous
partitions of 64-, 128- or 256-wide k-tiles, four combination orders, alpha after the sum or on each partial.
Exactly one fits every job (`results/analysis/splitk_search.*`):

- K is cut into 128-wide k-tiles; split s owns the contiguous tiles [s·q, (s+1)·q) with q = ⌈T / S⌉ (T = number of
    tiles);
- fast: one chain per split from +0; slow: every tile is its own chain, added serially in fp32 from +0;
- the partials are added serially in fp32, ((p0 + p1) + p2) + …, and alpha is applied afterwards (the epilogue);
- the customOption does not change the result.

This differs from the RTX 4060 (interleaved 64-wide tiles, restart every two own tiles, a pairwise order for
customOption 1). The rule was fitted to these 1,049 jobs and checked on the same jobs; ragged K (K mod 128 ≠ 0) and
K > 16,384 did not occur among the split rows and are not covered.

## Replay through mmaemu (added after the probe, same day)

The rule above and the H100 table (`mmaemu/platform/data/heur_h100pcie_lt12.8.3.14.*` in the repository, exported
from `results/m6_heur/`) went into mmaemu. `tools/replay_h100_log.py` then replays the probe through the package
itself (`mmaemu.scaled_mm` and `mmaemu.explain`, reference backend, profiles `hopper-h100pcie-torch` / `-te` /
`-ws0`) and compares full outputs by sha256 (`results/analysis/mmaemu_replay.json`, whose `environment` names the
commit and whether the replayed code was modified). It ran on an RTX 5070 Ti, not on the H100.

- Live jobs with the logged split passed explicitly: all 1,450 reproduce the H100. 1,423 are equal (1,049 split,
    374 unsplit). The 27 e4m3 × e4m3 → e5m2 jobs raise `ReferenceRejectsError`, just as cuBLASLt rejected them.
- Live jobs with `splitk="auto"`:
    - on all 1,000 M6 jobs the table gives the logged decision;
    - of the 423 M4 jobs that ran, 341 agree, and 82 are off the table's grid and raise `SplitKUnknownError` (the
        H100 table infers nothing between its rows);
    - none disagrees.
- M2, the 21 Evo 2 projections run from the truth's inputs: 168 of 168 outputs are equal, and `splitk="auto"`
    gives S = 1 from the table on all of them.

A second replay ran on the same GPU through the hopper preset's CUDA kernel h1 (`--backend auto`, added the next
day; `results/analysis/mmaemu_replay_cuda.json`). Every live job and Evo 2 output has the same verdict as above.
h1 ran all 1,043 E4M3 × E4M3 live jobs that ran (746 of them split) and all 168 Evo 2 outputs; the 380
mixed-format jobs have no kernel and ran on the reference backend. A third replay, through the faster kernel h4
(`--backend auto` again, which now prefers h4; `results/analysis/mmaemu_replay_cuda_h4.json`), gave the same verdicts,
with h4 on the same 1,043 live jobs and 168 Evo 2 outputs.

The repository's CI replays a subset of these live jobs on every push (`tests/test_h100_mini.py`, built by
`tools/export_h100_mini.py`). The split rule is still post hoc: the replay checks the package against the same jobs
the rule was fitted to.

## The criteria (fixed before the rental; plan §5)

| # | Criterion | Result |
|---|---|---|
| 1 | `MM_cublas64` E4M3 fast: exactly one distinguishable class fits 100% | met: the 8 fitting variants are one class (below) |
| 2 | `hopper` with the measured c mode and the logged plan reproduces the 21 projections, fp32 and bf16, torch (and TE if measured) | met: 168 of 168 |
| 3 | every unsplit random call (fast) bit for bit | met: 186 of 186 |
| 4 | slow: one promotion structure reproduces every slow call | met: restart every 128, sequential; 188 of 188 unsplit live jobs, 84 of 84 slow M2 outputs, all M3 rows |
| 5 | two passes identical for every configuration | met: 1,450 of 1,450, no input faults |
| 6 | if an Evo 2 shape splits, its reduction is determined | met without a case: no Evo 2 shape splits on this GPU |

GO #1, the tie (found after the data came in): the 8 variants differ in `zero_exp`, `min_exp` and `c_exp_sub`, and
their predictions are identical on every vector of both sets (`results/analysis/edge_variants.log`). In a cuBLAS GEMM
these parameters never act: every chain starts from +0 and c is the previous block's result, so every nonzero
product has exponent ≥ 2·emin and c is a multiple of 2^(2·emin − 13), never an fp32 subnormal and never below the
clamp. `scripts/probe/tie_check.py` confirms it on 640,000 chains (all four format pairs; uniform, zero-heavy,
subnormal-heavy and cancelling inputs; fast and slow): 0 differing outputs (`results/analysis/tie_check.log`). The
single late-RNE model left in slow mode by the 12,881-model set is not evidence for these parameters: that set has
only one late-RNE member with this core.

## What this does not show

- One H100 PCIe (114 SMs) on one instance, cuBLASLt 12.8.3.14, torch 2.7.1. The H100 SXM (132 SMs) needs its own
    heuristic table; other cuBLASLt versions may choose other kernels.
- Split-K: the rule above is post hoc; ragged K and K > 16,384 are not covered.
- TE: `general_gemm` with layout TN and a 32 MiB workspace, as Evo 2 uses it.
- The raw bits (1.2 GB of npz) are kept by the maintainer, listed with their sha256 in `results/SHA256SUMS`;
    `results/live_jobs.tar.gz` has every live job's record (cuBLASLt configuration, full-output sha256, input check).
    Host names, addresses, the GPU serial and task directory names are redacted (`<host>`, `<task-root>`, …).

## Files

- `scripts/`: `build_bundle.sh`, `run_all.sh`, `te_setup.sh`, `probe/` (recorders: `m1_cvec.py`, `m2_evo2.py`,
    `m3_patterns.py`, `m3_promo.py`, `m4_jobs.py`, `m4_run.py`, `m6_split_jobs.py`, `env_record.py`, `summarize.py`,
    `pc.py`; checks and analysis: `cpu_checks.py`, `analyze.py`, `splitk_search.py`, `tie_check.py`,
    `edge_variants.py`), `m4_jobs/` (the M4 job files).
- `results/`: the recordings' metadata and logs by step, `analysis/` (the offline analysis reports), `dry_runs/`.
