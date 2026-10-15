# T3 on an RTX 5070 Ti (sm_120), 2026-10-14

The GPU test suite on the release fatbins of e173ecc ([../fatbin-release](../fatbin-release/README.md)), run by
`tools/run_gpu_tests.sh --out <this directory>`: one pytest run per test file, its output in
`test_<file>.log`, and [summary.txt](summary.txt), the script's one line per file (exit code, seconds, pytest's
last line). Every file exits 0. The skips (listed in each log): the `cuda` alias, which is not a status entry;
`cuda:v9`, gated, with sm_120 not in its `validated_on`; `cuda:tc3`, which has no sm_120 binary; the full
accumulators of the reference and v1, which run only with `MMAEMU_DI_FULL`; three gate tests that need a device where
tc3's gates pass; and the live cuBLAS test, which needs the profile device (the RTX 4060), as do the T3p tests.
[docs/validation.md](../../../validation.md) cites this run.
