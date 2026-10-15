# API reference (v0.1)

Everything listed here is importable from the top-level package (`import mmaemu`); importing it has no side
effects (it does not import torch or initialise CUDA until a name is used). The arithmetic behind each function is
specified in [semantics.md](semantics.md).

- [GEMM entry points](#gemm-entry-points): `scaled_mm`, `accumulate`, `explain`, `CallPlan`
- [Instruction model](#instruction-model): `model.dot`, `Preset`, `get_preset`, `list_presets`
- [Platforms and split-K](#platforms-and-split-k): `Profile`, `get_profile`, `list_profiles`, `resolve_splitk`,
  `SplitK`, `SplitDecision`
- [Backends](#backends): `backends_status`, `selftest`
- [Testing helpers](#testing-helpers): `testing.assert_bitwise_equal`, `testing.bits`
- [Errors and warnings](#errors-and-warnings)
- [Command line](#command-line)

## GEMM entry points

### `mmaemu.scaled_mm`

```python
mmaemu.scaled_mm(input, mat2, scale_a, scale_b, bias=None, scale_result=None, out_dtype=None,
                 use_fast_accum=False, *, out=None, profile=None, splitk="auto", backend="auto") -> torch.Tensor
```

`torch._scaled_mm` of torch 2.7.1 (same positional names, order and defaults; torch's `self` is `input`),
computed with the numerics of the profile's platform, on the tensors' device (CPU or CUDA).

| Argument | Meaning |
|---|---|
| `input` | [M, K] `float8_e4m3fn` or `float8_e5m2`, row-major |
| `mat2` | [K, N] FP8, column-major (`w.t()` of a row-major [N, K] weight) |
| `scale_a`, `scale_b` | one-element fp32 tensors (tensor-wise scales); the result is multiplied by both |
| `bias` | must be None (`UnsupportedFeatureError` otherwise) |
| `scale_result` | one-element fp32 tensor or None; used only for FP8 outputs |
| `out_dtype` | `float32`, `bfloat16`, `float16`, `float8_e4m3fn`, `float8_e5m2`; default: `input.dtype` |
| `use_fast_accum` | fast accumulation (one chain over K) or slow (128-element chunks); default False, as torch |
| `out` | optional output tensor; its previous contents never matter |
| `profile` | a `Profile` or its name; default `$MMAEMU_PROFILE` or `"ada-rtx4060-torch"` |
| `splitk` | `"auto"` (from the profile), `1`, `(S, "serial")`, `(S, "pairwise")` or a `SplitK` |
| `backend` | `"auto"`, `"reference"`, `"cuda"`, `"cuda:v1"`, `"cuda:v4"`, `"cuda:v9"`, `"cuda:tc3"` |

Returns the [M, N] result of `out_dtype` on `input.device`. Raises `ReferenceRejectsError` wherever torch 2.7.1
or the profile's cuBLAS would reject the call, `UnsupportedFeatureError` for bias, row-wise scales or grad mode,
`SplitKUnknownError` when `splitk="auto"` cannot decide. Not traced by `torch.compile` (the entry points are
wrapped with `torch.compiler.disable`).

### `mmaemu.accumulate`

```python
mmaemu.accumulate(a, b, *, use_fast_accum, profile=None, splitk="auto", backend="auto",
                  out_dtype_for_split=torch.float32) -> torch.Tensor
```

The fp32 [M, N] accumulator of `a @ b.T` before the scales, as the profile's cuBLAS computes it. `a` is [M, K] and
`b` is [N, K] (both as rows of K; any layout, alignment and K are accepted). `use_fast_accum` has no default.
`out_dtype_for_split` is the output dtype under which the split-K decision is looked up (cuBLAS chooses splits per
output dtype).

### `mmaemu.explain`

```python
mmaemu.explain(input, mat2, scale_a, scale_b, bias=None, scale_result=None, out_dtype=None,
               use_fast_accum=False, *, out=None, profile=None, splitk="auto", backend="auto") -> CallPlan
```

Validates a `scaled_mm` call and reports what it would do, without computing it. Never raises for an invalid call:
the error is reported in `CallPlan.validation`.

### `mmaemu.CallPlan`

| Field | Content |
|---|---|
| `validation` | `"ok"`, or `(error class name, message)` if the call would raise |
| `profile`, `preset`, `preset_maturity` | the platform and instruction model used, and its maturity |
| `split` | the `SplitDecision` |
| `backend` | the backend that would run (`"reference"`, `"cuda:v4"`, ...) |
| `fallbacks` | why each backend tried before the chosen one was skipped for this call (its formats, preset and device); `cuda: <reason>` when no CUDA kernel can run |
| `warnings` | the warnings the call would emit |

## Instruction model

### `mmaemu.model.dot`

```python
mmaemu.model.dot(a, b, c=None, *, preset="ada", fmt=None) -> torch.Tensor
```

One tensor-core chain per row: d = c + a_0 b_0 + ... in blocks of the preset's size, with the preset's alignment
and truncation. `a` and `b` have the same shape [..., K] (no broadcasting): FP8 tensors, or float64 tensors
holding FP8 values together with `fmt` (`"e4m3"`, `"e5m2"` or a pair). `c` is None (+0) or fp32 of shape
`a.shape[:-1]`. Returns fp32 of shape `a.shape[:-1]`.

### `mmaemu.Preset`, `get_preset(name)`, `list_presets()`

A `Preset` is the tensor-core arithmetic: `nfma` (block size), `frac` / `frac_out` (fraction bits kept for terms
and for the block result), `min_exp`, `late_c` (c added after the block sum instead of inside the alignment),
`slow_chunk` (None: slow accumulation raises), `maturity` (`"validated"`, `"experimental"`, `"custom"`),
`validated_scope` and `evidence`. Shipped presets: `ada` (validated), `hopper` and `b200_mma_sync` (alias
`blackwell`, experimental). `get_preset` warns once per process for an experimental preset (every time for the
alias); the names `sm_120` and `rtx_blackwell` are refused. A `Preset(...)` you build yourself has maturity
`"custom"`. See [presets_profiles.md](presets_profiles.md).

## Platforms and split-K

### `mmaemu.Profile`, `get_profile(name_or_profile=None)`, `list_profiles()`

A `Profile` is a reference platform: `name`, `preset`, `gpu_name`, `capability`, `sm_count`, `cublaslt_version`,
`caller` (`"torch._scaled_mm"` or `"transformer_engine"`), `torch_version`, `workspace_bytes`, `align`,
`splitk_table`, `rejects` (what its cuBLAS refuses), `maturity` and `splitk_maturity`.
`Profile.matches_device(device=None)` is True only if the CUDA device has the profile's GPU name, SM count *and*
cuBLASLt version. Shipped profiles: `ada-rtx4060-torch` (default), `ada-rtx4060-te`, `ada-rtx4060-ws0` (validated),
`hopper-experimental`, `b200-mma-sync-experimental` (experimental).

### `mmaemu.resolve_splitk`

```python
mmaemu.resolve_splitk(M, N, K, *, a_dtype, b_dtype, out_dtype, use_fast_accum, profile=None, splitk="auto",
                      layout_std=True, strict=None, dry=False) -> SplitDecision
```

The split-K decision for an M x N x K call on the profile's platform, in the order explicit, workspace0, table,
live-query, envelope ([presets_profiles.md](presets_profiles.md#split-k-resolution)). `strict=True` (or
`MMAEMU_SPLITK_STRICT=1`) disables the envelope inference. `dry=True` (used by `explain`) warns without using up
the one-time `UnverifiedSplitWarning` of a later call.

### `mmaemu.SplitK`, `mmaemu.SplitDecision`

`SplitK(S=1, scheme="none")` is an explicit split: S > 1 needs `scheme="serial"` or `"pairwise"`.
`SplitDecision` has `S`, `scheme`, `source` (`"explicit"`, `"workspace0"`, `"table"`, `"live-query"`,
`"envelope"`), `verified` (this kind of decision was replayed on hardware), `table_id` and `note`.

## Backends

### `mmaemu.backends_status(device=None)`

Backend name -> `BackendStatus(name, available, source, validated_here, gates, reason)` for the device:
`available` means the kernel loaded and passed its gates; `source` is `"fatbin"`, `"nvrtc"` or `"torch"`;
`validated_here` means the device's capability is in the kernel's validated set; `reason` says why a backend is
unavailable.

### `mmaemu.selftest(device=None, *, kernels="all", force=False)`

Runs the gates of every kernel (or of `kernels`, a comma-separated string or a sequence such as `["v1", "v4"]`) on
the device and returns kernel name -> list of gate results. Verdicts are cached per device and kernel binary;
`force=True` re-runs them.

## Testing helpers

### `mmaemu.testing.assert_bitwise_equal(actual, expected, *, nan="bits", msg="")`

Asserts equal dtype, shape and bits (on the CPU; the tensors may live anywhere). `nan="mask"` accepts any NaN
payload where both are NaN. On failure, the `AssertionError` counts the mismatches by class (zero sign, NaN payload,
value) and lists the first ten.

### `mmaemu.testing.bits(t)`

The integer view of `t` with the same element size (uint8, int16, int32 or int64), without a copy.

## Errors and warnings

| Class | Base classes | Raised / emitted when |
|---|---|---|
| `MmaemuError` | `Exception` | base of every mmaemu error; also: CPU flush-to-zero is active |
| `ReferenceRejectsError` | `MmaemuError`, `RuntimeError` | torch 2.7.1 or the profile's cuBLAS rejects the call |
| `UnsupportedFeatureError` | `MmaemuError`, `NotImplementedError` | bias, row-wise scales, grad mode, uncharacterised preset features |
| `SplitKUnknownError` | `MmaemuError`, `LookupError` | `splitk="auto"` cannot decide |
| `BackendUnavailableError` | `MmaemuError`, `RuntimeError` | an explicitly requested backend cannot serve the call |
| `ExperimentalPresetWarning` | `UserWarning` | an experimental preset or profile is used |
| `UnverifiedSplitWarning` | `UserWarning` | a split decision that was not replayed on hardware |
| `BackendFallbackWarning` | `RuntimeWarning` | a faster CUDA kernel failed to load or failed a gate, or a kernel had to be compiled with NVRTC; a slower kernel, the same kernel built by NVRTC, or the reference model runs instead (the message says which) |

## Command line

```text
python -m mmaemu info     [--device N] [--json PATH]
python -m mmaemu selftest [--device N] [--kernels all|v1,v4,...] [--force] [--json PATH]
python -m mmaemu explain  M N K [--a e4m3|e5m2] [--b e4m3|e5m2] [--out DTYPE] [--fast]
                          [--scale-a X] [--scale-b X] [--scale-result X] [--profile NAME]
                          [--splitk auto|1|S,serial|S,pairwise] [--backend NAME] [--device auto|cpu|cuda|cuda:N]
                          [--json PATH]
```

Exit status: 0 success; for `selftest` 1 = a required kernel failed and 4 = nothing could be tested; for `explain`
1 = the call would raise and 4 = explain is unavailable; 2 = usage error.
