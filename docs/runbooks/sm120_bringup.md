# RTX 5070 Ti bring-up

Run from the repository with Python 3.12, torch 2.7.1+cu128, numpy and pytest.
The October 2026 test machine is WSL, RTX 5070 Ti, sm_120, 70 SMs. Its existing
Windows host driver works. WSL uses that driver; do not install a Linux display
driver inside WSL. CUDA 12.8 is needed to build sm_120 probe extensions.

```sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_HOME=<CUDA 12.8 toolkit>                # nvcc >= 12.8 for the sm_120 probe builds
export PATH=<venv>/bin:$CUDA_HOME/bin:$PATH
export MMAEMU_TRUTH_DIR=<truth checkout>             # the unpublished hardware ground truth (G3)
export MMAEMU_GPU_LOCK="flock -w 3600 /tmp/mmaemu-gpu.lock"   # GPU lock command prefix (default: in $TMPDIR)
export MMAEMU_CACHE_DIR="$PWD/build/mmaemu-cache"
python tools/bringup/run_bringup.py
```

Every command runs under `timeout 580`, except G1's T3 step: it runs the whole
tools/run_gpu_tests.sh, which takes the GPU lock and a 600-second timeout for
each test file itself (about 280 s in all on the RTX 5070 Ti, about 560 s on an
RTX 4060). Each other GPU command takes the GPU lock separately (the lock prefix
above waits up to 3600 s); build and CPU analysis commands run outside the
lock. The shipped fatbins in mmaemu/backends/cuda/bin are tested; to test a
local build instead, run `python tools/build_fatbin.py --release --out
build/fatbin` and export `MMAEMU_FATBIN_DIR=$PWD/build/fatbin`.
An inherited lock is detected to avoid taking the same lock twice. Prefer
running the runner outside the lock so CPU work does not hold the GPU lock.

Each run creates a unique directory under results/bringup/. `--out DIR`
requires a new directory. `--stage G8` runs just that gate; `--from G8`
resumes at G8 in a new directory, preserving earlier evidence. `--dry-run`
executes G0-G3, G8 and G10 on sm_89 or sm_120 and skips native research.
`--stage G4 --force-fail G4` must fail and name G4 in summary.json. G0 and G2
write the GPU's UUID (`info.json`, `selftest.json`); redact it before committing
bring-up outputs.

| Gate | Evidence and acceptance |
|---|---|
| G0 | GPU, driver, libraries, compiler, info and host-clock status. Use CUDA events when the host clock fails. |
| G1 | CPU suite and per-file T3 GPU suite; zero unexpected failures. Ada profile-device tests skip on sm_120. |
| G2 | v1/v4 selftests pass; v9 window result recorded; tc3 has no supported sm_120 binary. |
| G3 | Full hashes of all 21 Evo2 accumulators for v1/v4/v9. v9 is explicitly opted in for this measurement. |
| G4 | c6 native FP8 instructions, including kind::f8f6f4, scored against all hypotheses. No preset is changed. |
| G5 | cuBLAS kit and full bf16 GEMMs with nonunit scales, kernel names and logged split-K. Native/Ada differences are recorded; probe errors stop the run. |
| G6 | Candidate cuBLASLt heuristic table at 0/1/4/32 MiB, kept outside package data. |
| G7 | Signed zero, NaN, underflow/overflow, FP8 saturation and scale_result; production-shape epilogue ties. CPU/GPU twins and tie replay must pass. |
| G8 | Raw bf16 window measurements and CPU analysis; design-A rejection and design-B measurements. Native model scores are observations. |
| G9 | Six Evo2 1B windows in native/emulate/compat/auto modes using a local Evo 2 consumer checkout (`--evo2-root` or `EVO2_ROOT`) and the Ada golden outputs (`--golden` or `MMAEMU_EVO2_GOLDEN`). |
| G10 | CUDA-event timing for all three projection sizes, v1/v4/v9, with numerical checks before timing. |

G4-G9 native observations characterize the new GPU. They do not change the
Ada numerical contract. A native mismatch in G5 exits 1 in g1_cublas.py and is
retained as a measured difference by the runner. Full-model cross-device
comparisons require identical windows, shapes, dtypes and token IDs before
computing tolerances; bitwise claims apply to the FP8 GEMMs. Candidate native
model loss simulation remains research work and is not a shipped profile.

All probe inputs and sources are vendored with SHA-256 provenance; runtime
outputs and caches stay in the new run directory. The c6 README is the
audit's, unmodified: its ../family.py is vendored as c6/family.py, while
../pool.py, ../results/ and the MM_sm90mma and MM_cublas64 sets (GH200) are
not vendored. Truth, audit recordings and
Ada golden files are read-only. G3 requires the pinned local truth package
(the runner refuses G3 and G9 without `MMAEMU_TRUTH_DIR`); G9 additionally
requires cached model blocks and the local consumer. The truth, the
golden files and the consumer are maintainer-only data, so G3 and G9 run only on
the maintainer's host.
