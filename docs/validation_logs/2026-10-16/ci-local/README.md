# CI steps run locally, 2026-10-16

`tools/ci_local.sh` runs the CPU steps of `.github/workflows/ci.yml` on one machine. It ran on a clean worktree of
56b1cb3: the code of fa0fd90 (`mmaemu/`, `tests/` and `tools/` equal), before the profile rename (63b7395), with
`examples/evo2_offload.py`'s `--profile` option and the logs of 16 October up to that commit. Verdict: **PASS**,
every step ([ci_local.out](ci_local.out); `<ci>` is the scratch directory of its venvs and logs).

- Command: `MMAEMU_CI_TORCH="2.6.0 2.7.1 latest" tools/ci_local.sh` (all jobs: lint, test, nvrtc, dist).
- Host: Linux x86_64 (WSL2), no GPU visible (`CUDA_VISIBLE_DEVICES=""`), 4 OpenMP threads; uv 0.11.16.
- Python 3.12.2 only. GitHub's matrix also covers Python 3.10, 3.11 and 3.13, which this run does not.
- CPU torch wheels: 2.6.0+cpu, 2.7.1+cpu and the latest, 2.14.1+cpu; numpy 2.5.4.

| Job | Step | Result |
|---|---|---|
| lint | ruff, actionlint, check_docs, `build_fatbin --check`, manifest sha256, no `torch.equal` in tests/ and tools/, no `perf_counter` in `tools/bench_kernels.py`, workflow policy | PASS (8 of 8) |
| test | T0+T1 torch 2.6.0 | 3497 passed, 5 skipped, 703 deselected |
| test | T0+T1 torch 2.7.1 | 3499 passed, 3 skipped, 703 deselected |
| test | T0+T1 torch latest (2.14.1) | 3498 passed, 4 skipped, 703 deselected |
| nvrtc | NVRTC compile of every kernel source x arch (no GPU) | 44 passed |
| dist | build, `twine check --strict`, `check_dist --require-fatbins`, wheel from the sdist, `check_dist --from-sdist`, `python -m mmaemu info` from the wheel | PASS (6 of 6) |
| dist | T0+T1 against the wheel (torch 2.7.1) | 3501 passed, 1 skipped, 703 deselected |

Against the run of 2026-10-15 ([../../2026-10-15/ci-local/README.md](../../2026-10-15/ci-local/README.md), on
49f1ecf): the H100 PCIe profiles, the kernels h1 and h4 and their tests followed, so 4,205 tests are collected (3,624
then) and the NVRTC step passes 44 tests (31). The `check_docs` step checked the documentation of
56b1cb3. 63b7395, later, renamed the profiles in `mmaemu/`, `tests/` and `tools/`, so this run predates the code of
0.1.0.
