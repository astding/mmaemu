# The H100 PCIe logs replayed on an RTX 5070 Ti on fa0fd90, 2026-10-16

`tools/replay_h100_log.py --device cuda --backend auto` on an RTX 5070 Ti (sm_120) from a clean worktree of fa0fd90
(`environment` in each report: `mmaemu_commit`, `"dirty": false`); 63b7395, later, renamed the profiles (the replays
on its code: [../rtx-5070-ti-renamed/README.md](../rtx-5070-ti-renamed/README.md)). `--backend auto` runs the
E4M3 x E4M3 calls on the hopper preset's kernel h4 and the other format pairs on the reference backend. The calls ran in
chunks under the GPU lock and were merged (`probe/merge.out`, `sweep/merge.out`). `final_local.sh` is the local chain
that ran them: T3 (`../t3-rtx-5070-ti-sm120/`), the GPU tier of T4, the probe replay, T2 (`../t2/`), the CPU tier of
T4 (`../t4/`), the CPU suite, then the sweep replay (`<src>` the worktree, `<venv>` and `<scratch>` the test
environment). `status.txt` has the chain's steps with their exit codes and times (PDT), and the commit and cleanliness
of the worktree at its start and end.

## The probe of 2026-10-15 (`probe/mmaemu_replay.json`)

The live calls of [../../2026-10-15/h100-probe/README.md](../../2026-10-15/h100-probe/README.md) and its 21 Evo 2 1B
projections (`--truth $MMAEMU_TRUTH_DIR`): **PASS**. The 1,043 E4M3 x E4M3 live calls ran on h4 and the 380
mixed-format ones on the reference backend; all 1,423 equal the H100 (1,049 split, 374 unsplit), and the 27 calls the
H100 rejected raise `ReferenceRejectsError`. The 168 Evo 2 outputs ran on h4: 168 of 168 equal. The
`splitk="auto"` decisions: 1,353 agree with the H100 and 70 raise `SplitKUnknownError`.

Every live verdict and backend equals that of the replay of ebd18e1
([../../2026-10-15/h100-probe/results/analysis/mmaemu_replay_cuda_h4.json](../../2026-10-15/h100-probe/results/analysis/mmaemu_replay_cuda_h4.json));
12 `splitk="auto"` decisions that raised `SplitKUnknownError` there are now inferred S = 1 by the H100 envelope
(`source="envelope"`, `verified=True`) and agree with the logged decision. Every live record equals that of the
replay on the RTX 4060 of the same tree ([../rtx-4060-sm89/README.md](../rtx-4060-sm89/README.md),
`replay-h4/mmaemu_replay.json`) apart from its timings.

## The split-K sweep of 2026-10-16 (`sweep/mmaemu_replay_p5.json`)

The 2,744 live jobs of [../h100-splitk-sweep/README.md](../h100-splitk-sweep/README.md)
(`--live ../h100-splitk-sweep/results/live_jobs.tar.gz`, six chunks of 500): **PASS**. All 2,744 equal the H100
(2,416 split, 328 unsplit; 1,208 split and 164 unsplit each with fast and with slow accumulation), 994 of them on h4
and 1,750 on the reference backend.

Against the replay of d1ac890
([../h100-splitk-sweep/results/analysis/mmaemu_replay_p5.json](../h100-splitk-sweep/results/analysis/mmaemu_replay_p5.json)),
the same live jobs (`live_sha256`): every verdict, backend, split count and scheme is equal. Only the
`splitk="auto"` decisions differ, in 326 records, and none disagrees with the H100: 480 decisions agree with it (248
on d1ac890) and 2,264 raise `SplitKUnknownError` (2,496). 232 calls that raised there now have a decision that
agrees (216 from table rows, 16 S = 1 from the envelope at 1 MiB, `verified=True`), and 94 table decisions that
agreed with `verified=False` now have `verified=True`. The reason is bdc4bdf, after d1ac890: it added the sweep's
rows, its replayed split combinations and the envelope to mmaemu. These decisions therefore come from the sweep
itself, so they show that the shipped data reproduce it, not an out-of-sample test; the out-of-sample result is
the d1ac890 replay, whose verdicts this one repeats.

## The CPU suite (`pytest_full.out`)

`pytest -q -rs tests` on the CPU (`CUDA_VISIBLE_DEVICES=""`) with the ground truth and the audit data: 3,511
passed, 846 skipped (839 need a CUDA device and 4 the RTX 4060: T3 and T3p run them; 3 test an installed
package), in 12 minutes.
