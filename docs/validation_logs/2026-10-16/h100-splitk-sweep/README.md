# H100 PCIe split-K sweep, 16 October 2026

Question: which shapes off the H100 table's grid can mmaemu decide without the GPU? The probe of 15 October
(`../../2026-10-15/h100-probe`) recorded cuBLASLt's heuristic on 3,543 shapes (850,320 queries) and fitted the
split-K reduction to its 1,049 live split calls. With that table alone, `splitk="auto"` on the H100 profiles raised
`SplitKUnknownError` for every other shape, among them 4096³, 2048 × 11008 × 4096, M = 1 decode and Evo 2 1B at
M = 1000. This sweep records the heuristic densely, fits an envelope above which the H100 does not split K, tests
it once on a sealed held-out part, and runs live split calls the reduction rule was not fitted to.

**Verdict.**

| Workspace (profile) | Envelope | Sealed held-out test (p4) |
|---|---|---|
| 1 MiB (`hopper-h100pcie-torch`) | validated | 154,191 shapes above the bound, 0 split; the closest split is at 0.49 × the bound |
| 32 MiB (`hopper-h100pcie-te`) | unvalidated (shipped with a warning) | 77,518 shapes above the bound, **1 split** (at 1.23 × the bound) |
| 0 (`hopper-h100pcie-ws0`) | S = 1 for every shape | 46,654 shapes, 0 split |

All 2,744 live jobs (2,416 split, 328 unsplit) replay through mmaemu bit for bit, so the probe's reduction rule now
also holds on calls it was not fitted to, including 1,040 split calls with K not a multiple of 128 and 1,228 with
K > 16,384.

## Setup

- GPU: H100 PCIe (sm_90, 114 SMs, 80 GB), driver 570.148.08, one instance rented for this run (Lambda), 2026-10-16
  11:58 to 13:24 PDT. The measurements took 67 minutes, 12:06:50 to 13:14:05 PDT (`results/status.txt`, times in
  UTC).
- Software: Python 3.12.15, torch 2.7.1+cu128, cuBLAS / cuBLASLt 12.8.3.14 (`libcublasLt.so.12` sha256
  `4ef47ce6…`, the probe's library; `results/env/env.json`).
- Code: a bundle built by `scripts/build_bundle.sh` from mmaemu at 57bcf0d (`git archive`), the scripts in
  `scripts/sweep/` and the shape lists of `scripts/sweep/make_shapes.py`, generated before the bundle was built
  (`scripts/shapes/index.json`: rows and sha256 per part). Every query is the call of `tools/record_heuristics.py`
  (torch 2.7.1's `_scaled_mm` descriptor, `requestedAlgoCount` 1, no kernel runs); every live call is a
  `tools/record/live.py` job logged with its cuBLASLt configuration (`CUBLASLT_LOG_LEVEL=2`).
- Plan, fixed before the rental: the parts and their shape lists; the envelope is fitted on the training data
  only (p0 to p3) and committed before p4 is opened; a single split row above the bound in p4 means the envelope
  cannot ship as validated, only as unvalidated with a warning or not at all, and p4 is never used to refit;
  workspace 0 must have no split row in p1 and p4; p3's two alignments must agree before its rows enter the table;
  every live call must run cleanly with the queried configuration and give the same bits in both passes. The p4
  files' sha256 were recorded when they arrived and `eval_sealed.py` checks them.

## Parts

`scripts/run_all.sh` ran, in this order:

| Part | What it records | Queries |
|---|---|---|
| p0 | drift gate: the probe's grid re-recorded (`tools/record_heuristics.py --grid c5,evo2,evo2_small_m --ws 0,1,4,32`) | 850,320 |
| p1 | envelope grid: 24 M (1 to 16,384) × M*N on 148 levels f × R (R = 122,880, the probe table's largest split M*N at 1 MiB; f = 2^(i/16) for i = -64..80, then 48, 64, 128) × 33 K (1,024 to 4,194,304, 9 of them not a multiple of 128), all 30 settings (out dtype × format pair × fast) of each workspace; workspace 0 every fourth shape | 5,880,600 |
| p4 | sealed held-out part: 200,000 random rows per table workspace and 50,000 at workspace 0 (M, M*N / R and K log-uniform, half of the K multiples of 128), and 24 random M for every (K, N) of 10 models that are not in p3 | 497,880 |
| p2 | K dense: M in {1, 16, 128, 1024} × f in {0.25, 0.5, 1, 1.5, 2.5} × K from 1,024 to 16,384 in steps of 16, to 65,536 in steps of 128, then 8 values per octave to 2^22 | 1,671,600 |
| p3 | model rows: the (K, N) of 36 projections of 9 models (Evo 2 1B / 7B / 40B, Llama 2 7B, Llama 3 8B, Mistral 7B, Qwen2.5 7B, Llama 3 70B, Llama 3.2 1B; of Evo 2's four, only in_proj is an FP8 GEMM in Evo 2) × every M up to max(256, ⌈8 × 245,760 / N⌉), then every 16th to 8,192 and every 256th to 16,384, at both alignments of each workspace | 3,211,680 |
| p5 | 2,744 live `torch._scaled_mm` jobs chosen on the GPU host from its own p1 to p3 rows (`scripts/sweep/live_jobs.py`), 444 of them run twice in separate processes | 3,188 calls |

The 11,261,760 queries after the gate have 10,510,083 rows that cuBLASLt's heuristic accepts (status 0); the
others are the format and output combinations it rejects, as in the probe.

## Results

**Drift gate** (`results/p0/drift.json`): GO. The 850,320 rows equal the probe's recording that the table was
exported from in every column, on the same driver and library; the raw file (`p0/heur.npz`) has the probe's
sha256 `4f349703…` and is not stored twice.

**Heuristic** (`results/summary.json`; per part and profile).

- 1 MiB: the largest split M*N is 129,024 at every K (p1); splits begin at K = 5,744 (p1) and K = 5,136 (p2),
  with S up to 31 and customOption 0 to 3.
- 32 MiB: the split region grows with K: the largest split M*N is 524,288 at K ≤ 16,384, 3,932,160 from 16,385 to
  65,536, and at K ≥ 65,536 splits reach f = 32 (3.93 million), the top of p1's dense levels, while f = 48 (5.90
  million) never splits. S goes up to 91 and customOption to 7. 200,713 p1 split rows lie above twice the probe
  table's largest split M*N of their setting, so a K-free bound (the RTX 4060 tables' rule) cannot work here.
- Workspace 0: no split in 653,400 queries.
- Splits at K not a multiple of 128 are common (110,900 p1 rows at 1 MiB, 185,688 at 32 MiB); the model rows of p3
  split only at K ≥ 8,192, with S = 2 or 3.
- p3 gives the same decision at both alignments (0 and 256 bytes) of each workspace.

**The envelope** (`scripts/analysis/fit_envelope.py`, rule `results/analysis/envelope_rule.json`, committed in
ac159de / f4ea31e before p4 was opened). Per setting s = (workspace, out, fast, fmt_a, fmt_b) of the two table
workspaces:

- at each of p1's 33 grid K, `lo` is the largest split M*N of p1, and the basis is the next p1 level above it
  (every p1 row of that level is unsplit at this K); p0, p2 and p3 add their split M*N as raw lower bounds at the
  smallest grid K at or above their K;
- B_s(K) = 2 × the largest of these over the grid K at or below K's grid K, at least 15,360 (twice the lowest p1
  level);
- a shape with 1,024 ≤ K ≤ 4,194,304 and M*N > B_s(K) is claimed unsplit. Nothing is claimed outside this K range,
  at other workspaces, or at M*N ≤ B.

The bounds at 1 MiB stay at 15,360 to K = 5,120 and reach at most 268,004; at 32 MiB they reach 1,072,014 at
K = 16,384 and 11,796,480 from K = 65,536 on. The rule rests on assumptions that only the held-out test checks
(listed in the script): no split far above the largest split M*N seen at a K (the split depends on M and N
separately, and p1 has only 24 M values: p3's dense M found a split at 1.12 × the p1 basis, which the margin of 2
covers); the split region does not shrink as K grows; alignment does not matter; shapes beyond the tested M and N
are extrapolated in M*N.

**Sealed held-out test** (`scripts/analysis/eval_sealed.py`, `results/analysis/eval_sealed.json`, run once at
13:58 PDT on f4ea31e; it refuses to run unless the rule, the training data and the shape generator are committed
and the p4 files have their sealed sha256).

- 1 MiB: 223,940 rows, 154,191 above the bound, 0 split: **pass**. The closest split of all 20,727 split rows has
  M*N = 0.49 × its bound (M 2047, N 64, K 946,176).
- 32 MiB: 223,940 rows, 77,518 above the bound, 1 split: **fail**. The row is M 69, N 29,456, K 19,872,
  e4m3 × e5m2 → fp32, slow, S = 2, customOption 5; M*N = 2,032,464 > B = 1,653,270 (the bound at grid K = 20,480).
  Above K = 65,536 only 4,255 TE rows are above the bound (most rows there are below it).
- Workspace 0: 46,654 accepted rows, 0 split: **pass**.

Between the floor (15,360) and 30,720 only the 532 model rows per profile test the rule (the random rows start at
R / 4). The result is final: the TE envelope ships unvalidated with a warning (below); nothing was refitted.

**Live calls** (`results/live_jobs.tar.gz`, `results/summary.json`). The jobs: one adversarial split job per
verification key (fast, S, out, format pair, reduction scheme) on its cheapest row, with K not a multiple of 128
where the key has such a row (444 keys, run twice); one per key with K > 16,384; one random-data job per key; then
one per (key, workspace, customOption) not yet covered, cheapest first; 80 unsplit jobs at K not a multiple of 128;
264 model jobs (M = 1, 16, 128 at each projection with N × K ≤ 2^27, both workspaces). Split jobs are limited to
K ≤ 262,144 and (M + N) × K ≤ 2^26, and to 2,400 in all. No live split call has customOption 4 to 7: 3,331 split
rows of p1 to p3 with them were within the size limits, but the cheapest of them ((M + N) × K = 25,559,040) costs
more than the costliest fill job (5,111,808), and the fill reached the cap (444 + 724 + 1,232 = 2,400) first. Every one of the 3,188
calls ran without error, with clean inputs, and with the logged cuBLASLt configuration equal to the queried row;
the 444 twice-run jobs gave the same bits in both processes.

## Replay through mmaemu (after the sweep, same day)

`tools/replay_h100_log.py --live` replayed all 2,744 jobs on an RTX 5070 Ti at d1ac890 (`--backend auto`: 994 on the
hopper kernel h4, 1,750 on the reference backend; `results/analysis/mmaemu_replay_p5.json`, unmodified code): all
2,744 are equal to the H100 (2,416 split, 328 unsplit), and the two passes agree on 444 of 444. The probe's split
rule was fixed before this data existed, so these are out-of-sample calls of it.

Then the sweep went into mmaemu (`tools/export_splitk_table_h100.py`, `mmaemu/platform/data/heur_h100pcie_lt12.8.3.14.*`):

- the table gains the p3 rows (1,498,784 rows; 1,767,416 with the probe's). 28 keys differ between the probe and p3
  only in customOption, all unsplit (M 272 at 1 MiB and M 254 at 32 MiB, N 12,288, K 4,096, customOption 5 in the
  probe and 7 in p3); the probe's row is kept and the keys are listed in the table's `unsplit_customOption_differs`;
- the envelope, as fitted, with its validation per workspace: at 1 MiB an envelope decision is `verified`; at
  32 MiB it is `verified=False` and emits `UnvalidatedEnvelopeWarning`;
- every split combination of the table (140) now has a replayed live call (85 before);
  `verified_split_combos_h100pcie_lt12.8.3.14.json` lists 444 combinations from both replays;
- `"contiguous"` splits with K not a multiple of 128 no longer warn (1,040 replayed calls). A split that owns no
  k-tile still warns (no live call has one), and so does a customOption never seen on a replayed split call
  (4 and 5 appear on split rows of the table).

`tools/check_splitk_resolution_h100.py` resolves every accepted row of p1 to p4 and every equal live call of both
replays through `resolve_splitk` (each workspace's profile, no live query) and compares S with the H100's
(`results/analysis/resolver_check.json`, written from a clean checkout). The only difference is the TE row of p4
above.

## What this does not show

- One H100 PCIe (114 SMs) on one instance, driver 570.148.08, cuBLASLt 12.8.3.14, torch 2.7.1's descriptor; the
  heuristic of another GPU, driver or library may decide otherwise.
- The envelope is fitted, not derived. At 32 MiB it failed its held-out test once, and above K = 65,536 it was tested
  on few rows. Shapes with M above 16,384 or N above 2^20 are extrapolated.
- Below K = 1,024 and above K = 4,194,304 nothing is inferred; between the table's rows and below the bound a shape
  still raises `SplitKUnknownError` unless the call runs on the H100 (live query) or passes `splitk`.
- The reduction rule is fitted, not derived: it was found on the probe's data, and this sweep is a set of calls it
  predicted. On this evidence mmaemu marks S > 1 validated on `hopper-h100pcie-torch` and `-ws0` (decided after the
  sweep); with TE it stays experimental (`UnverifiedSplitWarning`): the 32 MiB calls ran `torch._scaled_mm`, none
  went through TE's own GEMM.
- Live split calls cover customOption 0 to 3 and K up to 262,144; customOption 4 to 7 and splits that own no k-tile
  were not run.
- TE: the 32 MiB decisions come from heuristic queries with torch's descriptor and 256-byte alignment, as in the
  probe's table; the live calls at 32 MiB ran `torch._scaled_mm`, not TE's GEMM.
- The seal of p4 is procedural: one analyst, who did not read p4 before the rule was committed; the shape list of
  p4 was fixed in the bundle beforehand.
- The raw recordings (102 MB, among them `p0/heur.npz`, the live calls' npz files and their cuBLASLt logs) are kept
  by the maintainer and listed with their sha256 in `results/SHA256SUMS`. The npz files of p1 to p4 are stored here
  unchanged; in the text files host names, addresses and task directory names are redacted (`<host>`,
  `<task-root>`, `<sweep-root>`), so their sha256 differ from the list.

## Files

- `scripts/`: `build_bundle.sh`, `run_all.sh`, `shapes/index.json`, `sweep/` (`sc.py` shared definitions,
  `make_shapes.py`, `drift.py`, `q.py` queries, `live_jobs.py`, `live_run.py`, `env_record.py`, `summarize.py`),
  `analysis/` (`fit_envelope.py`, `eval_sealed.py`).
- `results/`: `p0/` (drift gate), `p1/` to `p3/`, `p4_sealed/` (heuristic recordings: one npz and one json per
  chunk), `env/`, `status.txt`, `summary.json`, `live_jobs.tar.gz` (per live call: its job, cuBLASLt
  configuration, full-output sha256, input check), `SHA256SUMS`, `analysis/` (`envelope_rule.json`,
  `eval_sealed.json`, `mmaemu_replay_p5.json`, `resolver_check.json`).

| File | sha256 |
|---|---|
| `results/analysis/envelope_rule.json` | `cf3ffb7ffc2b7ac2…` |
| `results/analysis/eval_sealed.json` | `0da1b44bd233420b…` |
| `results/analysis/mmaemu_replay_p5.json` | `b7583b65755df216…` |
| `results/live_jobs.tar.gz` | `68e3157ff0aa8b10…` |
| `results/p4_sealed/torch_a0_c00.npz` | `d58775569db947f0…` |
| `results/p4_sealed/te_a256_c00.npz` | `b3471ea5fabedba5…` |
| `results/p4_sealed/ws0_a0_c00.npz` | `c95cc465cea87426…` |
