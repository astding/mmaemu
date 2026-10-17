# RTX 5070 Ti (sm_120) after the profile rename, 2026-10-16 and 17

Verdict: every step of `f2_local.sh` exits 0 (`status.txt`, 23:13 PDT on 16 October to 02:47 PDT on 17 October). It
first runs the local chain of fa0fd90, unchanged
([../rtx-5070-ti-replays/final_local.sh](../rtx-5070-ti-replays/final_local.sh)): T3 with the full accumulators, the
GPU tier of T4, the replay of the H100 PCIe probe through h4 and the reference backend (`--backend auto`), T2, the CPU
tier of T4, the CPU suite and the same replay of the split-K sweep's 2,744 live jobs. Then it runs the export check
of the H100 split-K table, the resolver check, the Evo 2 example check and the notebook. The outputs of the local
chain equal those of the same steps on fa0fd90 once the old profile names are read as the new ones
(`compare_to_fa0fd90.txt`: 0 differences over the JSON reports, the T3 logs and `pytest_full.out`; `t2/compare.txt`
and the two `merge.out` files are byte for byte equal, and `t4/mutants-cpu.txt` lists the same rows, apart from
times, in another order). The resolver check equals that of bdc4bdf (logged in af9a366), which used the old names.
The export check has no earlier run to compare with. The example check and the notebook are compared below with the
runs they repeat.

- Hardware and software: the host, environments, ground truth and audit data of the runs on fa0fd90
  ([../rtx-5070-ti-replays/README.md](../rtx-5070-ti-replays/README.md),
  [../t3-rtx-5070-ti-sm120/README.md](../t3-rtx-5070-ti-sm120/README.md)); the Evo 2 steps use the environment of
  [../evo2-h100-profile-rtx-5070-ti/README.md](../evo2-h100-profile-rtx-5070-ti/README.md) (`status.txt` gives
  mmaemu, evo2 and torch).
- Code under test: 2648bba, a clean git worktree at the start (`status.txt`: `clean=0`). Its code is that of 63b7395,
  which renamed the profiles ([../rtx-4060-sm89-renamed/README.md](../rtx-4060-sm89-renamed/README.md) lists the
  names); 2648bba adds logs only. Its fatbins are those of fa0fd90 ([../rc-fatbin/README.md](../rc-fatbin/README.md)).
  During the run `mmaemu/`, `tests/` and `tools/` did not change: at the end (`status.txt`, `F2_LOCAL_DONE
  clean=11`) the worktree's only changes are the example check's ten outputs, written into its log directory, and the
  notebook, executed in place.
- Script: `f2_local.sh`, one step after another, every GPU step under the GPU lock. Paths: `<src>` is the worktree,
  `<venv>` an environment, `<scratch>` the scratch directory that holds the chain's `rc2/final_local.sh` and its CPU
  test environment, `<out>` the output directory, `<models>` the checkpoints and
  `<brca1-data>` the BRCA1 inputs.

## Equal to the runs on fa0fd90

`compare_to_fa0fd90.py` compares the JSON outputs of T2, T4, the probe replay and the sweep replay, the T3 logs and
`pytest_full.out` with those of fa0fd90, and `resolver_check.json` with that of bdc4bdf. Before comparing, it replaces
the old profile names by the new ones and drops timings, dates, commits, command lines, file paths and the Python
version. For pytest's output it compares the per-test progress characters, the skip reasons and the counts. Its
output is `compare_to_fa0fd90.txt`.

| Output | Run on fa0fd90 | Result |
|---|---|---|
| `t3/` (`summary.txt` and one log per pytest run) | [../t3-rtx-5070-ti-sm120/](../t3-rtx-5070-ti-sm120/README.md) | the same exit codes and counts in every run: IEEE kernels 167, CUDA runtime 19 and the H100 CI replay's 310 GPU cases passed; the reference backend, v1 and v4 reproduce the RTX 4060's full-accumulator sha256 on all 21 Evo 2 1B projections |
| `t4/mutants-gpu.json`, `.txt` | [../t4/](../t4/README.md) | PASS: 7 killed, the identity control survives; each mutant killed by the same tests |
| `probe/mmaemu_replay.json` | [../rtx-5070-ti-replays/](../rtx-5070-ti-replays/README.md) | every record equal: the 1,423 live calls equal the H100 (1,043 on h4, 380 on the reference backend), the 27 calls the H100 rejected raise `ReferenceRejectsError`, and the 168 Evo 2 outputs on h4 equal; `"PASS": true` |
| `t2/compare.json`, `compare.txt` | [../t2/](../t2/README.md) | PASS; both files byte for byte equal. `replay_mmaemu.json` (not kept, 4.5 MB; `repo_commit` 2648bba) equals that of fa0fd90 apart from two paths |
| `t4/mutants-cpu.json`, `.txt` | [../t4/](../t4/README.md) | PASS: 43 killed, 2 equivalent, the 5 identity controls survive |
| `pytest_full.out` | [../rtx-5070-ti-replays/](../rtx-5070-ti-replays/README.md) | 3,511 passed, 846 skipped, with the same skip reasons |
| `sweep/mmaemu_replay_p5.json` | [../rtx-5070-ti-replays/](../rtx-5070-ti-replays/README.md) | every record equal: the 2,744 live jobs equal the H100 (2,416 split), 994 on h4 and 1,750 on the reference backend; `splitk="auto"` agrees on 480 and raises on 2,264; `"PASS": true` |
| `resolver_check.json`, `.out` | [../h100-splitk-sweep/results/analysis/resolver_check.json](../h100-splitk-sweep/results/analysis/resolver_check.json) (run on bdc4bdf, logged in af9a366) | passed: only the sealed test's one failure differs, as there |
| `export_h100_check.out` | | `check: OK`: `tools/export_splitk_table_h100.py --check` rebuilds the shipped H100 table from the recordings and finds the committed files equal |

## The Evo 2 example check

`examples/evo2_offload.py` with `--profile`, run by
[../evo2-h100-profile-rtx-5070-ti/example_check/run_checks.sh](../evo2-h100-profile-rtx-5070-ti/example_check/run_checks.sh);
its outputs and their reading are in
[../evo2-h100-profile-rtx-5070-ti/README.md](../evo2-h100-profile-rtx-5070-ti/README.md). On the 1B, with `ada-te` and
with `hopper-te`, `test` gives the logits, losses and accuracies of the runs it repeats and `score` their 40 BRCA1
scores, bit for bit; on the 40B, `test` with `hopper-te` gives the logits of the run it repeats and passes (0.2161).

## The notebook

`jupyter nbconvert --to notebook --execute --inplace evo2_blackwell.ipynb` in `examples/` with the 1B checkpoint
(profile `ada-te`), 1,724 s. The executed notebook is the one shipped in `examples/`, and
[notebook_results.json](notebook_results.json) is the `results.json` it wrote. Every number it reports equals that of
its run of 14 October ([../../2026-10-14/evo2-rtx-5070-ti/README.md](../../2026-10-14/evo2-rtx-5070-ti/README.md)):
`results.json` is equal apart from timings, and the BRCA1 scores of the three modes (`brca1_*.npz`, not
kept) are equal byte for byte. Its printed outputs differ from those of 14 October only in the version line
(mmaemu 0.1.0), the backend table (h1 and h4 added), the profile name in the printed plan (`ada-te`), the plan's
fallback list (it now names only the backends tried before the chosen one, so not the reference backend), the order
in which the 26 extra state_dict keys are printed when the checkpoint loads (a Python set, whose order changes from run
to run) and the timings.

`versions.mmaemu` in `notebook_results.json` says 0.1.0.dev0: the notebook reads it with `importlib.metadata`, which
reports an older mmaemu installed in the environment. The code imported was the worktree's (on `PYTHONPATH`): the
notebook's first output says mmaemu 0.1.0, and `status.txt` gives the file imported.

## Files

| File | Content |
|---|---|
| `status.txt` | the chain's record (PDT): the commit and cleanliness of the worktree at the start, the ground truth's manifest sha256, per step its start, end, exit code and seconds, and at the end the worktree's changes |
| `f2_local.sh` | the chain |
| `t3/`, `t4/`, `t2/`, `probe/`, `sweep/`, `pytest_full.out` | the outputs of the local chain of fa0fd90's steps |
| `export_h100_check.out`, `resolver_check.json`, `resolver_check.out` | the H100 split-K table's checks |
| `compare_to_fa0fd90.py`, `compare_to_fa0fd90.txt` | the comparison with the runs on fa0fd90 |
| `notebook_results.json` | the notebook's `results.json` |
