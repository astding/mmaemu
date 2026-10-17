# CI (T0 + T1, lint, NVRTC, dist) on the code of 0.1.0, 2026-10-17

Three GitHub Actions runs of `.github/workflows/ci.yml` (workflow `ci`, push to `master` of the private CI repository),
each with conclusion **success** in all 15 jobs and the same counts in every job (table below).

| Run | Time | Head commit | Its tree, identical to that of |
|---|---|---|---|
| 39024948378 | 10:27 to 10:31 UTC (03:27 to 03:31 PDT) | c1e8b3a (`c1e8b3a626edcaf2680f628c8dc343de5ee94876`) | 414b7e3 (tree `a582f63e5ebab9edc63d9978f9a97adc81a93eef`) |
| 39026557909 | 10:52 to 10:56 UTC (03:52 to 03:56 PDT) | 21d30a5 (`21d30a5799e9a454ce8d5136e20b1ad92c1dcf6f`) | be2af24 (tree `e8adfec035b179d6bd41deb26022943f67548a72`) |
| 39029044525 | 11:31 to 11:35 UTC (04:31 to 04:35 PDT) | 201441a (`201441ad7d74893ef120c7e731b127813db820c3`) | e386606 (tree `b4d265a1f89fe72dc103f9c27a0a48b7a47a58f7`) |

- Head commits: snapshots, each with the tree of a development commit. The third tree, e386606, holds the code of
  0.1.0: `mmaemu/` as in 63b7395 (the profile rename) and `tests/` as in 63b7395 apart from the skip that 59b37e1
  added to `tests/test_h100_mini.py` for a tree without `tools/` (it skips nothing here). The first two trees have
  `tests/` without that skip. be2af24 adds to 414b7e3 only documentation and logs (the CHANGELOG change of 4aa20f7, the
  T7 reports and their rows); e386606 adds to be2af24 the changes of 59b37e1 (that skip, documentation, the install
  line of the notebook's setup cell and of its generator) and logs. The lint job's documentation check and the `docs`
  workflow ran on each tree.
- Runners: `ubuntu-24.04` (Linux x86_64). CPU torch wheels: 2.6.0+cpu, 2.7.1+cpu and the latest, 2.14.1+cpu (the same
  in all three runs).
- The run logs are kept by GitHub Actions for a limited time; the numbers below are copied from them. On Python 3.12
  the runners skip 47 more tests than the local run of the same code with the same torch (and against the wheel)
  ([../ci-local/README.md](../ci-local/README.md): 3 to 5 skips, 1 against the wheel); pytest runs there without
  `-rs`, so the logs do not name them.
- The `docs` workflow of the same pushes (runs 39024948390, 39026557905 and 39029044510) built the site with
  `mkdocs build --strict`: success (its deploy job is skipped in the private repository).

| Job | Result (all three runs) |
|---|---|
| T0+T1 py3.10 torch 2.6.0 | 3449 passed, 53 skipped, 703 deselected |
| T0+T1 py3.10 torch 2.7.1 | 3451 passed, 51 skipped, 703 deselected |
| T0+T1 py3.10 torch latest | 3450 passed, 52 skipped, 703 deselected |
| T0+T1 py3.11 torch 2.6.0 | 3450 passed, 52 skipped, 703 deselected |
| T0+T1 py3.11 torch 2.7.1 | 3452 passed, 50 skipped, 703 deselected |
| T0+T1 py3.11 torch latest | 3451 passed, 51 skipped, 703 deselected |
| T0+T1 py3.12 torch 2.6.0 | 3450 passed, 52 skipped, 703 deselected |
| T0+T1 py3.12 torch 2.7.1 | 3452 passed, 50 skipped, 703 deselected |
| T0+T1 py3.12 torch latest | 3451 passed, 51 skipped, 703 deselected |
| T0+T1 py3.13 torch 2.6.0 | 3450 passed, 52 skipped, 703 deselected |
| T0+T1 py3.13 torch 2.7.1 | 3452 passed, 50 skipped, 703 deselected |
| T0+T1 py3.13 torch latest | 3451 passed, 51 skipped, 703 deselected |
| lint (ruff, actionlint, docs, fatbin staleness, banned patterns, workflow policy) | success |
| NVRTC compile of every kernel source x arch (no GPU) | 44 passed |
| build, check_dist, fresh-venv install + T0/T1 against the wheel | 3454 passed, 48 skipped, 703 deselected |
