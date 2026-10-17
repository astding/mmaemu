# Backends: CUDA kernels, loader ladder, gates, cache

mmaemu computes the same bits on every backend. A backend only supplies the *chain function*. That is the block-FMA
chain of the request's preset (`ada`: 16-term blocks, `hopper`: 32-term blocks) over a list of K indices, starting from
+0. Everything around it (split-K plans, slow-accumulation chunks, non-finite inputs, epilogue) is shared torch code in
`mmaemu.model`.

| Backend | What runs | Where | Formats | Needs |
|---|---|---|---|---|
| `cuda:tc3` | fp16-HMMA "dummy accumulator" kernel (gated, optional) | CUDA, cc 8.0/8.6/8.9 SASS only | E4M3 x E4M3 | gates `smoke` + `design_a` |
| `cuda:v9` | bf16 tensor-core exponent pass + FFMA.RZ chain (gated, optional) | CUDA, SASS sm_80/86/89/90/100/120 only (no PTX) | E4M3 x E4M3 | gates `smoke` + `window` |
| `cuda:v4` | IEEE-only CUDA-core kernel (p2 v4 cfg9) | CUDA sm_80+ | E4M3 x E4M3 | gate `smoke` |
| `cuda:v1` | IEEE-only fp32-pipe kernel (tcemu v1 cfg4, DEC1 shared LUT) | CUDA sm_75+ | E4M3 x E4M3 | gate `smoke` |
| `cuda:h4` | the `hopper` preset's fast kernel: v4's design with 32-term blocks | CUDA sm_80+ | E4M3 x E4M3 | gate `smoke` |
| `cuda:h1` | the `hopper` preset's portable kernel: v1's design with 32-term blocks | CUDA sm_75+ | E4M3 x E4M3 | gate `smoke` |
| `reference` | float64 model (`mmaemu.model.chain`) | the tensors' device (CPU or CUDA) | all | nothing |

Every kernel implements one preset (`KernelSpec.preset`): v1, v4, v9 and tc3 the `ada` preset, h1 and h4 the `hopper`
preset. v1, v4, h1 and h4 use only IEEE-defined instructions: fp32 adds and multiplies with explicit rounding modes,
`fma.rn.bf16x2`, and integer byte SIMD. Their result is therefore the same on every GPU they run on. A failure of their
smoke gate is a bug, not a hardware property. v4's sm_80/86 builds and its compute_80 PTX decode E4M3 in software; they
have been compiled but not run on hardware, and the smoke gate checks them on first use. h4 has run on sm_89 and sm_120 only. v9 and tc3 depend on how a
GPU's tensor cores round, so they are *gated*: they are auto-selected only on validated capabilities (today sm_89) and
after their gates pass.

## Selection (`mmaemu.backends.registry`)

`select(req, backend="auto")` takes `backend` from the argument. When the argument is `"auto"`, `$MMAEMU_BACKEND`
overrides it. Accepted values are `"auto"`, `"reference"`, `"cuda"` and `"cuda:<kernel>"`.

1. The reference backend is used in these cases:
   - `"reference"` was requested;
   - the tensors are on the CPU;
   - `MMAEMU_DISABLE_CUDA=1` is set;
   - no CUDA device is visible.

   In the last three cases an explicit `"cuda..."` request raises `BackendUnavailableError` instead. These paths
   never touch the CUDA driver.
2. Otherwise the discovered kernels are tried in priority order: tc3 (40), v9 (30), v4 (20), v1 (10) for the `ada`
   preset, h4 (20) and h1 (10) for the `hopper` preset. A kernel serves only its own preset, so the two orders never
   mix.
   `kernels.SPECS` is always discovered; `mmaemu.backends.cuda.fast.specs.SPECS` is added when it is importable.
3. Some kernels are skipped **by design**. This is silent, and `status()` / `explain_selection()` give the reason.
   It happens when:
   - the format pair or preset is unsupported: a kernel serves only its own preset, and E5M2, mixed formats and
     the presets without a kernel (`b200_mma_sync`, custom presets) always run on the reference backend;
   - the compute capability is below `min_cc`;
   - the architecture is outside the kernel's supported SASS/PTX targets; NVRTC
     does not extend that support (tc3 therefore stays unavailable on sm_120);
   - the device's opt-in shared memory is too small;
   - the kernel is gated and the capability is not in `validated_on`. `MMAEMU_ALLOW_GATED_KERNELS=1` lifts this
     check, and so does an explicit `"cuda:<kernel>"`; the kernel's gates still run in both cases.
4. Other skips count as **failures**: the module cannot be loaded (fatbin and NVRTC both failed), or a gate failed.
   The next kernel is tried. If none is left, `"auto"` uses the reference backend and `"cuda"` raises.
5. One `BackendFallbackWarning` is emitted per (device, reason) and process. It is emitted when a kernel failed, or
   when the selected kernel was compiled with NVRTC because its fatbin was unusable. By-design skips never warn.
   The failures of one selection form one reason: a single warning names every kernel that failed and the backend
   that is used instead.

`status(device=None)` returns `{name: BackendStatus}` for every discovered kernel and `reference`. A kernel is
`available` only after it was loaded *and* passed its gates on that device. `validated_here` means the capability is
in `validated_on`. `source` is `"fatbin"` or `"nvrtc"`. `reason` explains every unavailable kernel.

v1 and v4 have also passed the [RTX 5070 Ti validation](validation_logs/2026-10-13-sm120/README.md)
on sm_120 and report `validated_here=True` there. For the `ada` preset AUTO selects v4 on that device.
h1 and h4 report `validated_here=True` on sm_89 and sm_120, from their runs on the
[RTX 4060](validation_logs/2026-10-16/rtx-4060-sm89/README.md) and the RTX 5070 Ti
([h1](validation_logs/2026-10-16/h1-rtx-5070-ti/README.md),
[h4](validation_logs/2026-10-16/h4-rtx-5070-ti/README.md)).
On fa0fd90, T3 passed on the RTX 5070 Ti with v1, v4, h1 and h4
([T3](validation_logs/2026-10-16/t3-rtx-5070-ti-sm120/README.md)), and the replay of the H100 probe through h4
passed there and on the RTX 4060 ([replay](validation_logs/2026-10-16/rtx-5070-ti-replays/README.md)); both again
after the profile rename ([RTX 5070 Ti](validation_logs/2026-10-16/rtx-5070-ti-renamed/README.md),
[RTX 4060](validation_logs/2026-10-16/rtx-4060-sm89-renamed/README.md)).
They are ungated, so `validated_on` decides only `validated_here` for them: selection and `selftest` do not read it.
v9's sm_120 window measurements pass, while its default selection remains gated
until that architecture is added to its validation scope.

On an H100 PCIe (sm_90), v1 and v4 passed T3 with the release fatbins; NVRTC builds of the same kernels (5c15e8a,
sources equal up to comments) matched the RTX 4060 on all 21 Evo 2 projection calls
([H100 report](validation_logs/2026-10-14/t3-h100-sm90/README.md)). The full device-independence
suite did not run there, so sm_90 is not in their `validated_on`. For the `ada` preset AUTO still selects v4 on
sm_90, but `validated_here` is False, and a consumer that requires it falls back to the reference backend.

## Loader ladder (`mmaemu.backends.cuda.modules`)

Kernel modules are cached per **(device ordinal, CUDA context)**, keyed by `cuCtxGetId`. A module and its
compile-time `__constant__` decode tables belong to the context that loaded it. A second context or a second device
therefore loads its own copy (the prototype's kernels gave zeros there).

1. **Fatbin.** Release fatbins live in the package's `mmaemu/backends/cuda/bin/` together with `manifest.json`
   (format `mmaemu-fatbin/1`). The loader uses a module's fatbin only if all of these hold:
   - the file's sha256 equals the manifest;
   - the manifest's `source_sha256` equals the hash of the kernel sources shipped next to it (otherwise the fatbin is
     stale);
   - the device can run one of its SASS architectures, or its PTX can be JIT-compiled for the device.

   Loading a module takes about 2 ms (audit); `tests/test_cuda_runtime.py::test_fatbin_load_time_fresh_process`
   checks that it stays at or below 20 ms.
2. **NVRTC.** The shipped `.cu` source is compiled for the device's `sm_XY`. The compile options are fixed:
   `--fmad=false -default-device --std=c++17`; callers can add options but cannot change these.
   - libnvrtc comes from torch's `nvidia-cuda-nvrtc-cu12` wheel, so neither nvcc nor `CUDA_HOME` is needed.
   - The cubin is cached atomically in `$MMAEMU_CACHE_DIR/nvrtc/<sha256>.cubin`. The key covers the source, the
     headers, the options, the architecture and the NVRTC version.
   - If the cache directory cannot be written, the cubin is kept in memory.
   - NVRTC builds are bitwise equal to the fatbin builds (tested). An internal review of the prototype measured them 4-6% slower;
     unpaired runs here put v9 and tc3 within 1.5% of the fatbin builds ([kernels_fast.md](kernels_fast.md)).
3. Otherwise `BackendUnavailableError`, which names both reasons. The registry then moves to the next backend.

Launches run under `torch.cuda.device(device)` on `torch.cuda.current_stream(device)`. CUDA graph capture is not
supported in v0.1: the non-finite-code check of the plan synchronises with the host.

## Kernel ABI v1 and the ChainFn wrapper (`mmaemu.backends.cuda.kernels`)

```
extern "C" __global__ void mmaemu_<name>_e4m3e4m3(const uint8_t* A, const uint8_t* B, float* C, int M, int N, int K);
```

- A is `[M][K]` E4M3 codes and B is `[N][K]`, both row-major. C is `[M][N]` fp32.
- `C[m][n]` is the fast chain of the kernel's preset (`KernelSpec.preset`) over k = 0..K-1 from +0.
- Sources are device-only: no torch headers, no host code, no standard headers (NVRTC has none).
- The kernel never faults, whatever the byte values.

`chain_fn_for(spec, req)` returns the ChainFn that the plan executor calls. It:

- accepts only `torch.uint8` codes or `torch.float8_e4m3fn` and raises `TypeError` otherwise (B4: e5m2, fnuz and
  bf16 never reach a kernel);
- raises `ValueError` for mixed devices;
- makes the operands contiguous, and clones an operand whose address is not aligned to `align_bytes` (B12: a
  misaligned 32-bit load is a sticky error that kills the CUDA context);
- raises `ValueError` for a request of another preset;
- checks `L % 16 == 0` and `L < k_max` (v4 and h4: 2^24, v1 and h1: 2^30); h1 and h4 end a chain with L % 32 == 16
  in a half block of zero codes, which the model excludes;
- returns empty or +0 results for M, N or L = 0 without a launch;
- launches in row chunks when `ceil(M / tile_m)` exceeds 65535.

The chain is given finite codes only: `execute_plan` sanitises NaN codes and patches the IEEE result afterwards.
The raw v1 and h1 kernels also propagate NaN codes to canonical NaN by themselves (the prototype's v1 decoded them
as 0).

| Kernel | Tile | Threads | Dynamic smem | SASS | PTX | k_max |
|---|---|---|---|---|---|---|
| v1 | 64 x 64 | 256 | 0 (12 KiB static) | sm_75, 80, 86, 89, 90, 100, 120 | compute_75 | 2^30 |
| v4 | 128 x 128 | 256 | 36,864 B | sm_80, 86, 89, 90, 100, 120 | compute_80 | 2^24 |
| h1 | 64 x 64 | 256 | 0 (22 KiB static) | sm_75, 80, 86, 89, 90, 100, 120 | compute_75 | 2^30 |
| h4 | 128 x 128 | 256 | 73,728 B | sm_80, 86, 89, 90, 100, 120 | compute_80 | 2^24 |

Shared memory above 48 KiB is enabled per function up to the device's opt-in limit
(`CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`). tc3 (81,920 B) and h4 (73,728 B) need it: `modules.launch`
opts in, and selection skips a kernel if the device's limit is lower.

### v1 decode variants (compile-time switches in `v1.cu`)

| Variant | Switch | Shipped |
|---|---|---|
| decode values from a per-block shared-memory copy of the `__constant__` table (DEC1) | `MMAEMU_V1_DEC=1` | see below |
| carry-free biased exponent bytes (exp1b) | `MMAEMU_V1_EXP=1` | see below |

The rule is to ship a variant only if it is bitwise equal and at least 1.2x faster on sm_89.
The measured decision and its numbers are recorded in [the v1 decision section](#v1-variant-decision).

## Gates and the verdict cache (`mmaemu.backends.cuda.selftest`)

- **smoke** (every kernel) compares the kernel's ChainFn bit for bit with the CPU float64 reference on two cases:
  - 16x8x64 random codes;
  - 64x64x256 adversarial codes: subnormals and ±0, ±448, cancellation inside blocks, one exponent per 16-block,
    every finite code, and -0 everywhere.
- **window** (v9) and **design_a** (tc3) are registered by `mmaemu.backends.cuda.fast.gates` through
  `spec.register_gate`. A gate function may carry an integer `version` attribute.
- `run_gates(spec, device, force=False)` runs `smoke` and then `spec.gates`, and stops at the first failure.
- Verdicts are cached in memory and in `$MMAEMU_CACHE_DIR/verdicts/<sha256>.json`. The cache key is:
  - GPU UUID, compute capability and driver version;
  - mmaemu version;
  - sha256 of the loaded module image;
  - gate name and gate version.
- A corrupt cache file is ignored and the gate runs again. A gate that raises is reported as failed and is not
  cached. The v9 and tc3 gates catch errors inside their stages and return a failed `<gate>.error: ...` verdict,
  which is cached like any other: after a transient error (for example out of memory), run
  `python -m mmaemu selftest --force`.

## Environment variables (read at call time)

| Variable | Effect |
|---|---|
| `MMAEMU_BACKEND` | default backend when the call says `"auto"` |
| `MMAEMU_DISABLE_CUDA=1` | never use CUDA kernels; the driver is not touched |
| `MMAEMU_ALLOW_GATED_KERNELS=1` | auto-select gated kernels on capabilities outside `validated_on` (their gates still run) |
| `MMAEMU_CACHE_DIR` | NVRTC cubins and gate verdicts (default `${XDG_CACHE_HOME:-~/.cache}/mmaemu`). mmaemu writes no file anywhere else. It must not be writable by other users: cubins and verdicts are loaded from it without verification |
| `MMAEMU_LIBCUDA_PATH` | driver library; when set, the kernel loader tries only this path (otherwise `libcuda.so.1`, then `/usr/lib/wsl/lib/libcuda.so.1`); the live split-K query tries it first, then those two |
| `MMAEMU_NVRTC_PATH` | NVRTC library; when set, only this path is tried (otherwise the `nvidia.cuda_nvrtc` wheel, `$CONDA_PREFIX/lib`, `$CUDA_HOME/lib64`, `find_library("nvrtc")`) |
| `MMAEMU_FATBIN_DIR` | development/tests only: load fatbins from a local build instead of the package's `bin/` |
| `MMAEMU_TEST_FORCE_FAIL` | tests only: comma list of `fatbin`, `nvrtc` and gate names that fail on purpose (forced gate failures are not cached) |

Nothing is initialised at import time. `import mmaemu` loads neither libcuda nor libnvrtc.

On the reference host, `import torch` itself maps libcuda: libcublasLt's initialiser dlopens it.
`tests/test_cuda_runtime.py::test_no_visible_devices` therefore checks two things: that mmaemu never opens the driver
(`driver.loaded()` is False), and that it adds no mapping.

## Building fatbins (`tools/build_fatbin.py`)

```
python tools/build_fatbin.py --out build/fatbin            # development (-lineinfo)
python tools/build_fatbin.py --release --out <dir>         # release flags: -O3 --fmad=false, relative paths
python tools/build_fatbin.py --check [--out <dir>]         # hash-only staleness check (CI; no nvcc)
```

- The tool refuses nvcc older than 12.8, which is needed for sm_100 and sm_120. It takes nvcc from `--nvcc`,
  `$NVCC`, `$CUDA_HOME/bin` or `PATH`; release fatbins are built with nvcc 12.8.61. A distribution's nvcc 12.0 is
  refused.
- Architectures come from the KernelSpecs. Probe modules (`PROBE_MODULES`) get SASS sm_80..sm_120 and no PTX.
- Besides the fatbins and `manifest.json`, the tool writes:
  - `cuobjdump.txt`: the `--list-elf --list-ptx` listing, which must equal the requested architectures. It uses a
    cuobjdump of at least 12.8 (a distribution's cuobjdump may be 12.0), so the tool also looks next to nvcc and
    in triton's wheel.
  - `build_report.json`: ptxas registers and spills, compiler warnings, and the forbidden-PTX grep.
- It fails when any of these hold:
  - a module's PTX for any of its architectures contains `fma.rz.bf16` or `fma.rz.bf16x2`, which ptxas 12.0-13.3
    miscompiles;
  - ptxas reports spill bytes, unless `--allow-spills` is given;
  - the listing differs from the KernelSpecs.
- Release fatbins are `--release` builds committed to `mmaemu/backends/cuda/bin/`. A branch that adds or changes a
  kernel commits that module's build (`--release --modules <module>`; its entry is added to `bin/manifest.json`,
  the other entries unchanged), with its build log under `docs/validation_logs/<date>/`, as h1
  ([build](validation_logs/2026-10-15/h1-fatbin/README.md)) and h4
  ([build](validation_logs/2026-10-16/h4-fatbin/README.md)) were. At the release-candidate freeze the maintainer
  rebuilds every module with `--release`, compares it byte for byte with the committed fatbins and runs T3 on
  them; the log goes under `docs/validation_logs/<date>/` ([release.md](release.md), section 2). What the 0.1.0
  freeze found, and why the committed v1 and v4 bytes were kept, is in
  [rc-fatbin](validation_logs/2026-10-16/rc-fatbin/README.md).

## Benchmarks (`tools/bench_kernels.py`, `tools/hostclock.py`)

```
$GPU timeout 600 python tools/bench_kernels.py --kernels v1,v4 --shape 1b --M 2048 --data gauss40 \
    --fatbin-dir build/fatbin_rel --out results.json
```

Method:

- **Timing:** CUDA events only. The host-clock gate runs first and the
  `perf_counter / CLOCK_MONOTONIC_RAW` ratio is recorded; on this WSL2 host perf_counter runs 6-10% fast.
- **Load source:** kernels are loaded as users get them. Performance acceptance needs a fatbin (the package's release
  fatbins, or `--fatbin-dir`); `--allow-nvrtc` records NVRTC numbers instead, marked as such. The report records
  each kernel's image sha256.
- **Exactness gate before timing:**
  - v1 and h1 are compared with the CPU reference of their own preset on the first min(M, 64) rows x 256 columns;
  - every other kernel is compared with the IEEE kernel of its preset (v1 for `ada`, h1 for `hopper`) on the full
    output;
  - any mismatch stops the run, which writes its report and exits 1.
- **Warm-up and windows:** at least 1.5 s of warm-up per kernel, then interleaved windows of at least 1.0 s each.
  The reported value is the median TMAC/s over the windows.
- **Clocks:** `nvidia-smi` (`/usr/lib/wsl/lib/nvidia-smi` on WSL) samples the SM clock, power and throttle reasons
  every 200 ms. MAC/clk/SM = TMAC/s / (mean SM clock × SM count).
- `--timer host-raw` (CLOCK_MONOTONIC_RAW around synchronised launches) exists for cross-checks. It is refused when
  the host-clock gate fails.

## Platforms

- The CUDA kernels are tested on Linux x86_64 only (WSL2 included). mmaemu does not check the platform: they load
  wherever `libcuda.so.1` is found (Linux aarch64 is untested and self-test gated like any untested device). On
  native Windows that library is not found, so mmaemu warns once and uses the reference backend.
- Multi-GPU is untested. The per-(device, context) cache is exercised with a second context on one GPU.
- Validation status:
  - v1 and v4 are validated on sm_89 (RTX 4060) and sm_120 (RTX 5070 Ti). On sm_90 (H100) they passed T3 but are
    not in `validated_on` (see above).
  - v9 and tc3 are validated on sm_89 only.
  - h1 and h4 are validated on sm_89 (RTX 4060) and sm_120 (RTX 5070 Ti); they have never run on an H100 (sm_90).
  - On other capabilities the kernels are unvalidated: the smoke gate runs on first use, and v1/v4/h1/h4 are
    IEEE-exact by construction.

<a id="v1-variant-decision"></a>
## v1 variant decision

Measured on the RTX 4060 (sm_89, 24 SMs, driver 572.16, WSL2) on 2026-10-12 under the GPU lock with
`tools/bench_kernels.py`:

- workload: Evo 2 1B shape (M 2048, N 5760, K 1920), gauss40 data;
- builds: `nvcc 12.8.61 -O3 --fmad=false`;
- timing: CUDA events, 1.5 s warm-up, 5 interleaved windows of at least 1.0 s each.

Every variant was bitwise equal to the shipped v1 on all 11,796,480 outputs, and v1 itself was equal to the CPU
reference on a 64x256 block. The raw report is `build/bench/v1_variants_1b_2048_gauss40.json` (not committed).

| v1 variant | TMAC/s (median) | vs base | vs DEC1 |
|---|---|---|---|
| base: `__constant__` LUT, `__vadd4/__vmaxs4` (DEC0, EXP0; = tcemu v1 cfg4 + fixes) | 0.515 | 1.00x | |
| DEC1 shared-memory LUT (DEC1, EXP0) | 0.636 | **1.235x** | 1.00x |
| exp1b only (DEC0, EXP1) | 0.652 | 1.265x | |
| DEC1 + exp1b (DEC1, EXP1) | 0.725 | 1.406x | **1.138x** |

The decision applies the rule to each change relative to what is already adopted, in the spec's order:

1. DEC1 is shipped (`MMAEMU_V1_DEC=1`): bitwise equal and 1.235x faster than the base.
2. exp1b on top of DEC1 adds only 1.138x, below the 1.2x bar, so it is **not** shipped (`MMAEMU_V1_EXP=0`).
   It stays a compile-time switch, which keeps the measurement reproducible:
   `--v1-variant dec1exp1=MMAEMU_V1_DEC=1:MMAEMU_V1_EXP=1`.
   Adopting it is a maintainer decision: 0.725 vs 0.636 TMAC/s.

The shipped v1 SASS is instruction-for-instruction identical to the audit's DEC1 build of
`verify-p0-profile-0/src/variants.cu` (cfg4, DEC 1): 2784 instructions, 0 differing lines after normalising.
The shipped v4 SASS is identical to the audit's p2 v4 cfg9 build: 4320 instructions, 0 differing lines.
So the audit's exactness and speed evidence carries over.

Acceptance run (a development tree, before the v0.1 components were integrated):

- fatbin: release-flag fatbin built locally by `tools/build_fatbin.py --release`;
- workload: 1b shape, M 2048, gauss40;
- tree: the v0.1 scaffold and numerics with the CUDA runtime and kernels v1 / v4;
- report: `build/bench/A7_v1_v4_1b_2048_gauss40.json` (not committed).

| Kernel | TMAC/s median (min-max) | Mean SM clock in its windows | MAC/clk/SM | Threshold |
|---|---|---|---|---|
| v1 (DEC1) | 0.637 (0.634-0.638) | 2555 MHz | 10.4 | >= 0.60 |
| v4 cfg9 | 1.354 (1.352-1.356) | 2512 MHz | 22.5 | >= 1.25 |

Conditions during the run:

- Throttle reason 0x4: the software power cap.
- `nvidia-smi -q -d POWER` samples averaged 110 W against the 115 W limit. WSL reports no instantaneous
  `power.draw`.
- The host-clock ratio `perf_counter / CLOCK_MONOTONIC_RAW` was 1.031. This is recorded only; all timing used CUDA
  events.
