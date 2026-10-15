# Testing mmaemu

mmaemu's oracle is hardware, not the paper: every bit claim is checked against outputs recorded on an RTX 4060
(sm_89, 24 SMs, cuBLASLt 12.8.3.14, torch 2.7.1). The full recording set is a read-only dataset of 173 recordings
("the truth"), the hardware ground truth of the tcemu audit. It is not published. The repository commits a 19 MB
subset of it, `hw_mini`, so that CI and every user can replay real hardware bits without the dataset.

All commands below assume the repository root as the working directory, and this environment:

```
# a venv with torch 2.7.1+cu128 (pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128) and pytest
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1        # test the working tree; never write bytecode
export MMAEMU_GPU_LOCK="flock -w 3600 /tmp/mmaemu-gpu.lock"   # GPU lock command prefix (default: this, in $TMPDIR)
GPU=$MMAEMU_GPU_LOCK                                    # exclusive GPU lock; each locked run < 10 min
```

nvcc is needed only to build fatbins (`tools/build_fatbin.py`: nvcc >= 12.8 from `--nvcc`, `$NVCC`, `$CUDA_HOME/bin`
or `PATH`; release fatbins are built with 12.8.61). The kernels otherwise compile through NVRTC from torch's wheels.

Some tiers need data that only the maintainer has; without it they skip or refuse with a message naming the
variable:

| Variable | Data | Used by |
|---|---|---|
| `MMAEMU_TRUTH_DIR` | the hardware ground truth (unpublished) | truth-marked tests, T2 (`tools/replay_truth.py`), the Evo 2 check of T7, the exporters and recorders under `tools/` |
| `MMAEMU_AUDIT_DIR` | the tcemu audit's work directory (unpublished) | extra GPU data sets of `tests/test_fast_kernels.py` (not collected without it), `tools/bringup/vendor_kits.py` |
| `MMAEMU_TCEMU_DIR` | a checkout of the tcemu prototype | `tools/record/replay_fixed.py` only |

T0, T1, T3 and T4 run without them (the truth-marked tests in T3 skip).

## Tiers

| Tier | When | Command | Budget |
|---|---|---|---|
| T0 unit | CI, every push | `CUDA_VISIBLE_DEVICES= python -m pytest -q tests -m "not gpu and not slow and not truth" --ignore tests/test_hw_mini.py` | < 3 min |
| T1 hw_mini | CI, every push | `CUDA_VISIBLE_DEVICES= python -m pytest -q tests/test_hw_mini.py` (4 torch threads) | <= 20 MiB data, < 5 min |
| T1s slow | nightly / manual | `python -m pytest -q -m slow tests` | < 1 h |
| T2 full truth replay | before a release, maintainer only (needs the truth) | `python tools/replay_truth.py --out docs/validation_logs/<date> --compare $MMAEMU_TRUTH_DIR/replay/replay_fixed.json` | about 80 min on CPU |
| T3 GPU | any CUDA device | `tools/run_gpu_tests.sh` (one locked run per test file) | < 10 min per run |
| T3p profile device | RTX 4060 only | `$GPU timeout 600 python -m pytest -q tests/test_live_cublas.py` | < 10 min |
| T4 mutation | weekly (CPU), before a release (GPU) | `python tools/mutants/run.py --tier cpu`; `$GPU timeout 600 python tools/mutants/run.py --tier gpu` | < 20 min CPU |

T0 + T1 together are what CI runs: `python -m pytest -q tests -m "not gpu and not slow and not truth"`.

## Markers

Registered by `tests/conftest.py`:

- `gpu`: needs a CUDA device. Skipped without one.
- `slow`: T1s.
- `truth`: needs `$MMAEMU_TRUTH_DIR`. Skipped without it.
- `profile_device("<profile>")`: runs only on the reference device of that profile. Elsewhere it is skipped with
  "profile device unavailable".

`tests/conftest.py` also blocks `import tcemu` (the audited prototype), and fails the session if `tcemu` was imported.

## Bitwise policy

- Every bit claim goes through `mmaemu.testing.assert_bitwise_equal(actual, expected, nan="bits")`. It compares
  integer views, so +0 differs from -0 and NaN payloads must match.
- `torch.equal` and `==` are banned for bit claims. CI greps `tests/` and `tools/` for `torch.equal(`.
- `nan="mask"` (NaN positions must match, payload ignored) is allowed only against a reference that does not
  canonicalise NaN. Each use carries an inline `# nan=mask: <reason>` comment.
  - mmaemu canonicalises NaN on every backend (fp32 0x7FFFFFFF, bf16/fp16 0x7FFF, fp8 0x7F), and so does the RTX 4060.
  - The hardware tests in this file therefore never use `nan="mask"`.
- Uses of `nan="mask"` in the repository (find them with `grep -rn 'nan="mask"' tests tools`), as of this commit:

  | File | Reason |
  |---|---|
  | tests/test_scaffold.py (`test_comparator_nan_payload`) | tests the mask mode of the comparator itself |

- The old-kernel emulator recordings (truth `emulator.*`) are not compared with `nan="mask"`. Their 6,120 NaN-decode
  outputs are compared with the canonical NaN instead (see T2 and the device-independence test).
- A strict xfail always carries `raises=`.

## T1: hw_mini (`tests/data/hw_mini/`, format `mmaemu-hwmini/1`)

### Contents

`tools/export_hw_mini.py` builds the subset from the truth, which it only reads (`sys.dont_write_bytecode`). It
writes `COVERAGE.txt` and `coverage.json` next to the data. These list every truth recording id with the outputs
included, the n/a reasons, and the 239 split/accumulation combinations. The exporter selects:

- at least one output of every non-table truth recording that mmaemu v0.1 can evaluate;
- every hw_error output that the audit's FixedModel reproduces (113 outputs, including the 14 torch bias-check
  rejections);
- for every hardware-replayed `(fast, S, scheme, out_dtype, fmt_a, fmt_b)` combination (239, of which 211 are split
  combinations), the smallest recorded case;
  - It keeps 1 output row and 4 spread columns. The split-K partition depends on K only, so any row/column subset of
    a recorded output is reproduced by the GEMM of the subset operands.
- the c2 `T_*` and `S16_*` instruction families, at least 4,096 vectors each, and all of `S16_zeros`;
- every `epilogue.*` case whose stored output has M*N <= 65,536;
  - When the operands exceed 48 KiB, it keeps the 8 rows and 32 columns richest in special outputs (+-0, NaN, Inf,
    subnormal, fp8 saturation).
- `misc.tcemu_tests_recorded_hw` in full (it replaces the old `tests/data/hw/*.pt`);
- rows 0..31 of the Evo 2 1B projections `evo2.evo2_1b_w2k` calls 0, 10 and 20 (unit-scale fp32 accumulator, real-scale
  bf16, TE bf16), on 256 columns (a full 1B weight is 11 MB);
- the special-value table `c3.gemm_table` (scale set s00);
- one scope-boundary call per recording that has bias or row-wise outputs;
- from the pre-swap recordings (`recordings/rtx4060_2026-10`, when present): R1 in full, at least one case per R4
  combination, and at least one output of every other recording.

The exporter fails if the directory would exceed the budget (`--budget-mb`, default 19.5 MiB).

```
python tools/export_hw_mini.py --truth $MMAEMU_TRUTH_DIR [--ws2 recordings/rtx4060_2026-10] [--dry-run]
```

### Format

`index.json` holds the following keys:

- `format` ("mmaemu-hwmini/1");
- `truth_manifest_sha256` (pinned; T1 checks it when `MMAEMU_TRUTH_DIR` is set);
- `sources`, `selection` (the exporter options);
- `shards` (name -> sha256, bytes);
- `cases`.

Shards are `<group>.npz` files (`allow_pickle=False`) of raw bit patterns, deduplicated by content. Each case is a
dict with these keys:

| key | meaning |
|---|---|
| `id` | `<truth id>/<case>/<output>[@<selector>]`, unique |
| `truth_id` | the truth recording id (`ws2:<id>` for the pre-swap recordings) |
| `kind` | `"mma"` (instruction level) or `"gemm"` |
| `why` | selector: `hw_error`, `project`, `c2fam`, `epilogue`, `evo2`, `combo`, `extra`, `scope`, `one`, `ws2_*` |
| `src` | truth case/output names and the row/column/tile/vector selection (output coordinates) |
| `params` | see below |
| `inputs` | name -> `[shard, key]` (gemm: `a` uint8 [M,K], `b` uint8 [N,K] = mat2 transposed; mma: `a`, `b`, `c` uint32) |
| `expect` | `{"bits": [shard, key], "dtype", "shape", "sha256"}` or `{"error": "<exact mmaemu error class>"}` |
| `scope` | absent = in scope; `"out"` = the hardware ran it, mmaemu v0.1 does not model it (expected error) |

**mma params:** `fmt_a`, `fmt_b`, `layout` (`vector` or `tile`), `preset`, `K`. For the tile layout:
`D[t,i,j] = chain_k(a[t,i,k] * b[t,j,k])`, starting from `c[t,i,j]`.

**gemm params:**

- the normalised call: `fmt_a`, `fmt_b`, `out_dtype`, `fast`, `scale_a_bits`, `scale_b_bits`, `scale_result_bits`
  (fp32 bit patterns or null), and `splitk` `{S, scheme}` (the recorded configuration; always passed explicitly);
- `api`:
  - `scaled_mm`: the stored operands form a valid torch call;
  - `accumulate`: `mmaemu.accumulate` followed by `mmaemu.model.epilogue.apply`. Used for subsets whose N or K is
    not a multiple of 16, for cuBLASLt-, TE- and emulator-level outputs, and for `out_select: "diagonal"`;
- `call`: how the hardware call passed its arguments. Omitted keys take the defaults:
  - `out_dtype` and `use_fast_accum` passed as keywords;
  - `scale_result` passed iff `scale_result_bits` is not null;
  - no bias; no `out=`.

  Non-defaults are:
  - `out_dtype: false` / `use_fast_accum: false` (the torch default was used);
  - `bias: "positional"`;
  - `out_fill: <float | "nan" | "inf">` (`out=` buffer prefilled);
  - `requires_grad: [operand names]`;
- `zero_operands` + `call_shape` [M, N, K]: validation cases (expected errors). The operands are zeros of the
  hardware call's shape and dtypes. Bias, when present, is zeros [N] of `bias_dtype`.

**Expected error classes (matched exactly):**

- `ReferenceRejectsError`: torch or cuBLAS rejected the hardware call, and the FixedModel reproduces it (e5m2 x e5m2;
  e4m3 x e4m3 -> e5m2; the torch bias checks).
- `UnsupportedFeatureError`: the hardware ran the call, but mmaemu v0.1 does not model it (bias, row-wise scales,
  requires_grad under grad mode).

### Tests

`tests/hwmini.py` reads the format with numpy only. `tests/test_hw_mini.py` runs:

- `test_index`: format, shard sha256, size <= 20 MiB, no in-scope recording missing, pinned truth manifest;
- `test_case[<id>]`: every case on the CPU reference backend;
- `test_gpu_backend[<backend>]` (marker gpu): every case on cuda:0, for each backend that
  `mmaemu.backends_status()` reports available.
  - A case whose formats the backend does not support (BackendUnavailableError) is counted as not supported.
  - Every available backend must run at least one case.

## T2: full truth replay

T2 needs the unpublished hardware ground truth, so only the maintainer runs it; its reports are committed under
`docs/validation_logs/`.

```
MMAEMU_TRUTH_DIR=<truth checkout> PYTHONPATH=$PWD:$MMAEMU_TRUTH_DIR \
python tools/replay_truth.py --out docs/validation_logs/<date> --compare $MMAEMU_TRUTH_DIR/replay/replay_fixed.json
```

The run proceeds in five steps:

1. `tools/replay_truth.py` installs a `tcemu` import blocker.
2. It snapshots the truth tree (size and mtime of every file, `.git/` excluded) and pins its `manifest.json` sha256.
3. It calls `tcemu_ada_truth.replay.replay([MmaemuTruthModel()])` programmatically.
4. It writes `replay_mmaemu.json`, `compare.json` and `compare.txt` under `--out`, never into the truth.
5. It prints a per-recording delta table against the audit's FixedModel report, then `PASS` or `FAIL`.

The adapter (`tools/truth_adapter.py`, `MmaemuTruthModel`) works as follows:

- **mma:** `mmaemu.model.dot`.
- **gemm validation:** `mmaemu.explain()` on the hardware call's shape and dtypes. It maps outcomes by exact class:
  - `ReferenceRejectsError` -> `ModelRejects`, for torch-level outputs;
  - `UnsupportedFeatureError` -> `NotApplicable`;
  - everything else propagates. It never uses `except RuntimeError`.
- **gemm compute:** the recorded split-K is always passed explicitly. GEMMs of at most 3e7 MAC run on the CPU
  reference.
- **Large GEMMs:** they run mmaemu's own plan, epilogue and specials with a ChainFn on the truth's AVX-512 oracle
  `fastacc.chain_acc`, or mmaemu's CUDA backend with `--device cuda` under the GPU lock. Each such case is
  cross-checked against the mmaemu reference on 2 rows x 32 columns.
- **fastacc library:** the adapter never calls `GemmCtx.acc`, `fastacc.lib()` or `fastacc.available()`. It ctypes-loads
  the shipped `libfastref3.so` only if it is at least as new as its source; otherwise it builds a private copy in
  `$MMAEMU_CACHE_DIR/truth_fastref3/`.
- **n/a set:** the same as the FixedModel's: HMMA, fp16-accumulator mma, workspace dumps, the torch row-wise stride bug,
  and split-K "unknown".

The verdict is PASS iff all of the following hold:

- every in-scope output the FixedModel matched matches;
- zero_sign = nan_payload = 0;
- every hash-only output in scope matches (>= 2,539 minus out of scope);
- all 113 FixedModel-reproduced hw_error outputs are reproduced;
- new n/a outputs are only bias, row-wise or fused-bias, each listed by id;
- the emulator group differs from its recording only at the 6,120 NaN-decode outputs, and there it gives the canonical
  NaN;
- the fastacc cross-check has 0 mismatches;
- `tcemu` is not in `sys.modules`;
- the truth listing and mtimes are unchanged.

A run restricted with `--only`/`--skip`/`--max-cases` gives a partial verdict. `tests/test_truth_full.py` (marker
truth) runs such a partial replay on 21 representative recordings.

## T3: GPU

`tools/run_gpu_tests.sh` runs every GPU test file in its own locked run of less than 10 minutes, and writes logs to
`docs/validation_logs/<date>/t3-<gpu>-sm<cc>/`. Expected outcomes depend on the device capability.

The fatbin tests (`tests/test_kernels_ieee.py`, `tests/test_cuda_runtime.py`) load the fatbins of
`$MMAEMU_TEST_FATBIN_DIR`; the runner sets it to the package's `mmaemu/backends/cuda/bin/` when that has a
manifest (the release fatbins), else to `build/fatbin`. A run in which they skip for want of a manifest is a
failure, so that untested kernels never pass as tested.

`tests/test_device_independence.py` (markers gpu + truth; runbook gate G3) has three parts:

1. `test_evo2_acc_rows[<backend>]`: rows 0..255 of all 21 Evo 2 1B accumulators equal the stored hardware rows, on
   every available backend. The `reference` backend on the GPU checks 4 rows.
2. `test_evo2_acc_sha256[<backend>-callNN]`: the sha256 of the full [2048, 5760] accumulator equals the hardware's.
   - By default this runs only for the backend that `backend="auto"` selects.
   - `MMAEMU_DI_FULL=all` (or a backend name) extends it. `tools/run_gpu_tests.sh --di-full` runs every backend in
     chunks of 7 calls per locked run.
3. `test_emulator_group[<backend>]`: the fixed inputs of truth `emulator.tcemu_kernels_fixed_inputs` give the recorded
   old-kernel bits on every backend, except at the NaN-decode outputs, where the result is the canonical NaN. The CPU
   reference gives the same bits.

## T4: mutation matrix (`tools/mutants/`)

T4 runs on the repository alone: the CPU tier kills mutants with T0 + T1, the GPU tier with T3. The catalogue's
`source` fields name where a mutant comes from (for some, a script of the tcemu audit); they are labels in the report
and are not read at run time.

`tools/mutants/catalogue.py` lists every mutant, grouped as follows:

- instruction model: the c2 features, the c4 cuBLAS-level mutants and the a1 zero-sign mutants;
- plan: split layout, reduction scheme, slow chunking and restarts, chunk-sum order, alpha before the reduction;
- epilogue: rounding, scale order, saturation, scale_result, NaN payload, zero sign;
- API: defaults, positional bias, the e4m3 x e4m3 -> e5m2 rejection;
- kernel (GPU tier).

`tools/mutants/run.py` prepares and runs each mutant:

1. It copies the repository to a temporary directory; large data is symlinked.
2. It appends an override to the module that defines the frozen function. The override comes from
   `tools/mutants/impl.py`, a self-contained re-implementation with one modelling decision changed. Kernel mutants
   are exact text edits of the kernel sources; the committed fatbins are removed, so NVRTC compiles the mutated
   source.
3. It runs the tier's tests. A mutant counts as killed when pytest exits with 1, and the first failing test is
   recorded.

`identity_*` mutants inject the overrides with no change. They must survive: this shows that the injection is sound.
`control_sign_flip` must be killed. Mutants that the audit showed to be unobservable are reported as `equivalent`,
with the evidence cited (the min_exp variants). Any other survivor fails the run.

```
python tools/mutants/run.py --tier cpu [--jobs 4] [--only REGEX]       # -> docs/validation_logs/<date>/mutants.json
$GPU timeout 600 python tools/mutants/run.py --tier gpu                # -> docs/validation_logs/<date>/mutants_gpu.json
```

## Pre-swap recordings in hw_mini

R1 (the API parity grid) is exported in full. The call spec above models these parameters:

- `fmt_a`, `fmt_b`, `out_dtype`, `fast_accum`;
- scales `scale_a`, `scale_b` (fp32 bits) and `scale_result`;
- `splitk`;
- `bias_input` / `bias_dtype`;
- `scaling`;
- `explicit_kwargs` / `call_variant`;
- `requires_grad`;
- `hw_shape`.

The exporter stops on any other R1 parameter rather than drop it silently. The next exporter change adds the R1
layouts, storage offsets, scale shapes/dtypes, `out=` variants and zero-dimension calls to the call spec, together
with tests/test_hw_mini.py.
