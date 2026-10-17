# T3 on an RTX 5070 Ti (sm_120) on fa0fd90, 2026-10-16

The GPU test suite on the fatbins of the release-candidate freeze ([../rc-fatbin/README.md](../rc-fatbin/README.md)),
run by `tools/run_gpu_tests.sh --di-full --out <this directory>` from a clean worktree of fa0fd90 (63b7395, later,
renamed the profiles; T3 on its code: [../rtx-5070-ti-renamed/README.md](../rtx-5070-ti-renamed/README.md)), with the
ground truth and the audit data present: one pytest run per test file under the GPU lock, its output in
`test_<file>.log` and `di_full_*.log`, and [summary.txt](summary.txt), the script's one line per run. Every run exits
0 (or 5, nothing collected). `<src>` is the worktree, `<out>` the scratch output directory.

Against the run of 2026-10-14 on e173ecc ([../../2026-10-14/t3-rtx-5070-ti-sm120/README.md](../../2026-10-14/t3-rtx-5070-ti-sm120/README.md)),
file by file:

- equal: scaffold, backend reference, epilogue, fast kernels (91 passed; 91 skip, no tc3 binary for sm_120), gates
  (1 passed, 3 skipped), hardware mini (3 passed, 3 skipped), device independence (27 passed);
- `test_kernels_ieee` 167 passed (79 then) and `test_cuda_runtime` 19 passed (13 then): the tests of h1 and h4,
  each kernel against the CPU reference of its own preset;
- `test_live_cublas` 2 skipped (1 then): its second test, added since, also needs the profile device (the RTX 4060);
- `test_h100_mini` (new in the script): 310 passed, its GPU cases (the 124 live H100 calls through the reference
  backend on the GPU and through `auto`, which runs h4 here, and the E4M3 x E4M3 ones through h1); `-m gpu`
  deselects its 125 CPU cases;
- `test_backend_selection` exits 5, nothing collected: a CPU-only test file with no GPU-marked test (it runs in T0).

Full accumulators (`--di-full`, `MMAEMU_DI_FULL=<backend>`, 7 Evo 2 calls per locked run): the reference backend, v1
and v4 reproduce the RTX 4060's full-accumulator sha256 on all 21 calls each; `cuda` (an alias, not a status
entry), v9 (gated, sm_120 not in its `validated_on`) and tc3 (no sm_120 binary) skip.

The same suite ran on 7612fd8 (the release-candidate freeze) earlier the same day with the same counts in every file
but `test_scaffold` (102 deselected then, 103 now: a CPU test added since). A first run on 7612fd8 pointed
`MMAEMU_AUDIT_DIR` one directory too high, so the fast-kernel cases that read the audit data were not collected (13
passed, 17 skipped); the run with the correct directory replaced it.
