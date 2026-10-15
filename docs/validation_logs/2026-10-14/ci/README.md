# CI (T0 + T1, lint, NVRTC, dist), 2026-10-14

GitHub Actions run 38686435969 of `.github/workflows/ci.yml` (workflow `ci`, push to `master`), completed
2026-10-15 05:25 UTC, conclusion **success**.

- Head commit: 89977293 (`899772934293a2ee79c8fff0480042346b02d2ef`), a public snapshot whose tree
  `2e16076dfab6fe1cf69c6e91d29e36e7b1c02ba7` is identical to the tree of commit 8429385. The run therefore tested
  the code of 8429385. The fixes of the review of the release candidate (49f1ecf) have not run on GitHub Actions;
  [../../2026-10-15/ci-local/README.md](../../2026-10-15/ci-local/README.md) records the CI steps run locally.
- Runners: `ubuntu-24.04` (Linux x86_64). CPU torch wheels: 2.6.0+cpu, 2.7.1+cpu and the latest, 2.14.1+cpu.
- The run log is kept by GitHub Actions for a limited time; the numbers below are copied from it.

| Job | Result |
|---|---|
| T0+T1 py3.10 torch 2.6.0 | 3258 passed, 42 skipped, 296 deselected |
| T0+T1 py3.10 torch 2.7.1 | 3259 passed, 41 skipped, 296 deselected |
| T0+T1 py3.10 torch latest | 3258 passed, 42 skipped, 296 deselected |
| T0+T1 py3.11 torch 2.6.0 | 3259 passed, 41 skipped, 296 deselected |
| T0+T1 py3.11 torch 2.7.1 | 3260 passed, 40 skipped, 296 deselected |
| T0+T1 py3.11 torch latest | 3259 passed, 41 skipped, 296 deselected |
| T0+T1 py3.12 torch 2.6.0 | 3259 passed, 41 skipped, 296 deselected |
| T0+T1 py3.12 torch 2.7.1 | 3260 passed, 40 skipped, 296 deselected |
| T0+T1 py3.12 torch latest | 3259 passed, 41 skipped, 296 deselected |
| T0+T1 py3.13 torch 2.6.0 | 3259 passed, 41 skipped, 296 deselected |
| T0+T1 py3.13 torch 2.7.1 | 3260 passed, 40 skipped, 296 deselected |
| T0+T1 py3.13 torch latest | 3259 passed, 41 skipped, 296 deselected |
| lint (ruff, actionlint, docs, fatbin staleness, banned patterns, workflow policy) | success |
| NVRTC compile of every kernel source x arch (no GPU) | 31 passed |
| build, check_dist, fresh-venv install + T0/T1 against the wheel | 3262 passed, 38 skipped, 296 deselected |

The deselected tests are the GPU, slow and truth tiers (docs/testing.md). The skips differ between jobs by
torch and Python version (for example the live torch 2.7.1 schema comparison runs only on torch 2.7.x).
