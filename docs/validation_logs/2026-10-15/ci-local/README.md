# CI steps run locally, 2026-10-15

`tools/ci_local.sh` runs the CPU steps of `.github/workflows/ci.yml` on one machine. It ran on the fixes of the review
of the release candidate (49f1ecf, with the logs of 2026-10-15 staged: tree 7ab0709, which differs from the tree
of 5edd6bc only in this README), because the GitHub Actions run of
[../../2026-10-14/ci/README.md](../../2026-10-14/ci/README.md) tested 8429385. Verdict: **PASS**, every step.

- Command: `MMAEMU_CI_TORCH="2.6.0 2.7.1 latest" tools/ci_local.sh` (all jobs: lint, test, nvrtc, dist).
- Host: Linux x86_64 (WSL2), no GPU visible (`CUDA_VISIBLE_DEVICES=""`), 4 OpenMP threads; uv 0.11.16.
- Python 3.12.2 only. GitHub's matrix also covers Python 3.10, 3.11 and 3.13, which this run does not.
- CPU torch wheels: 2.6.0+cpu, 2.7.1+cpu and the latest, 2.14.1+cpu; numpy 2.5.4.

| Job | Step | Result |
|---|---|---|
| lint | ruff, actionlint, check_docs, `build_fatbin --check`, manifest sha256, no `torch.equal` in tests/ and tools/, no `perf_counter` in `tools/bench_kernels.py`, workflow policy | PASS (8 of 8) |
| test | T0+T1 torch 2.6.0 | 3321 passed, 5 skipped, 298 deselected |
| test | T0+T1 torch 2.7.1 | 3323 passed, 3 skipped, 298 deselected |
| test | T0+T1 torch latest (2.14.1) | 3322 passed, 4 skipped, 298 deselected |
| nvrtc | NVRTC compile of every kernel source x arch (no GPU) | 31 passed |
| dist | build, `twine check --strict`, `check_dist --require-fatbins`, wheel from the sdist, `check_dist --from-sdist`, `python -m mmaemu info` from the wheel | PASS (6 of 6) |
| dist | T0+T1 against the wheel (torch 2.7.1) | 3325 passed, 1 skipped, 298 deselected |

The counts differ from the GitHub run (Python 3.12, torch 2.7.1 there: 3260 passed, 40 skipped, 296 deselected):
49f1ecf adds tests (3,624 collected here against 3,596), and fewer tests skip on this host than on GitHub's runners.
