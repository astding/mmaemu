# T0 + T1 from the unpacked sdist, 2026-10-17

The sdist ships `mmaemu/` (without fatbins), `tests/` with their data, and the licence and metadata files, but not
`tools/` or `docs/`. This run checks that its test suite runs as shipped. Verdict: **PASS**, 3363 passed, 15 skipped,
393 deselected ([pytest.out](pytest.out); `<sdist>` is the unpacked sdist, `<venv>` the Python environment), on
17 October from 04:20 to 04:22 PDT.

- sdist: `mmaemu-0.1.0.tar.gz` built from 59b37e1 (`python -m build --sdist --wheel`, hatchling 1.32.4), sha256
  `740b69ac7e10f0542399fbecbea216e638dfe7249ff70ec4c6474cf9a15a5a13`.
- Command, in the unpacked sdist: `CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  -p no:cacheprovider tests -m "not gpu and not slow and not truth and not profile_device" -rs` (the CPU tiers of
  `.github/workflows/ci.yml`).
- Environment: Linux x86_64 (WSL2), no GPU visible; Python 3.12.2, torch 2.7.1+cu128, numpy 2.5.4, pytest 9.1.1.
- The 15 skips and their reasons are listed at the end of `pytest.out`: each needs a file the sdist does not ship
  (`tools/`, `docs/`, `.github/workflows/`, the release fatbins, the split-K sweep's logs), a git checkout or an
  installed distribution, plus one that leaves the decision to the registry.
- Before 59b37e1, `tests/test_h100_mini.py` imported `tools/record/live.py` when the module loaded, and collecting the
  sdist's tests failed with `ModuleNotFoundError: No module named 'live'`, which stopped the run (sdist built from
  be2af24). Since 59b37e1 the module skips itself when `tools/record/live.py` is absent; in a git checkout it runs as
  before.
