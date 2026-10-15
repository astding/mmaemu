# User guide

This guide walks through mmaemu by task: installing and checking it, computing a first GEMM, choosing a platform
profile, dealing with split-K, picking a backend, putting mmaemu inside a model, and testing with it. The exact
arithmetic is specified in [semantics.md](semantics.md); every function is listed in [api.md](api.md).

## 1. What mmaemu does, in one paragraph

An FP8 GEMM on NVIDIA tensor cores is not an IEEE computation. The tensor core adds blocks of products after
aligning them to the block's largest exponent and truncating them, cuBLAS may split the K dimension and add the
partial sums in an order of its own choosing, and the epilogue of `torch._scaled_mm` rounds twice. Two GPUs with
different tensor cores, or the same GPU with a different cuBLAS, therefore give different bits for the same FP8
GEMM. mmaemu recomputes the GEMM exactly as one *reference platform* computes it (in v0.1: the RTX 4060, Ada,
with cuBLASLt 12.8.3.14), on the CPU or on another CUDA GPU, with kernels that return the reference bits wherever
they run: IEEE-exact kernels everywhere, tensor-core kernels only where a per-device self-test confirms the
properties they rely on.

## 2. Install and check

mmaemu needs Python 3.10 or newer, PyTorch 2.6 or newer and NumPy. Until 0.1.0 is on PyPI, install it from the
repository:

```bash
pip install "git+https://github.com/astding/mmaemu"
```

No CUDA toolkit, nvcc or C++ compiler is needed: the package contains prebuilt CUDA kernels (fatbins) that are
loaded through the NVIDIA driver. If they cannot be used, mmaemu compiles the shipped kernel sources with NVRTC
(which comes with PyTorch's CUDA wheels), and otherwise falls back to the exact reference model, with one
`BackendFallbackWarning`.

Check the installation:

```bash
python -m mmaemu info       # versions, device, matching profiles, backend status with reasons, presets
python -m mmaemu selftest   # runs the kernel self-tests on the GPU; exit 0 = every required kernel passed
```

`info` tells you which backend `scaled_mm` will use on your GPU and why the others are unavailable, and whether
your GPU *is* a reference platform (`Profile.matches_device`). On a GPU that is not one, mmaemu still computes the
reference platform's result. What it computes does not depend on where it runs, with one exception: the split-K
decision of a shape missing from the profile's table is queried live on the reference GPU, and elsewhere inferred
(S = 1 when M*N is more than twice every split M*N of the table's setting) or refused with `SplitKUnknownError`.

## 3. A first GEMM

`mmaemu.scaled_mm` takes the arguments of torch 2.7.1 `torch._scaled_mm` (same names, order and defaults):

```python
import torch
import mmaemu

torch.manual_seed(0)
x = (torch.randn(64, 1920) * 40).clamp(-448, 448).to(torch.float8_e4m3fn)    # input  [M, K], row-major
w = (torch.randn(5760, 1920) * 40).clamp(-448, 448).to(torch.float8_e4m3fn)  # weight [N, K], row-major
scale_x, scale_w = torch.tensor(1 / 64), torch.tensor(1 / 32)                # tensor-wise fp32 scales

y = mmaemu.scaled_mm(x, w.t(), scale_x, scale_w, out_dtype=torch.bfloat16, use_fast_accum=True)
```

- `mat2` is [K, N] column-major, which is what `w.t()` gives for a row-major [N, K] weight, as in torch.
- The defaults follow torch: `use_fast_accum=False`, and `out_dtype` is the input dtype. Most FP8 inference code
  (Transformer Engine, Evo 2) uses fast accumulation with a bf16 output; pass both explicitly.
- The scales multiply the result: y = (x @ w.T) * scale_x * scale_w, with the rounding of the hardware epilogue.
  If your code stores *quantisation* scales (x_fp8 = x * s), pass their reciprocals, as Transformer Engine does.

The same call on CUDA tensors runs the CUDA kernels and returns the same bits:

```python
yg = mmaemu.scaled_mm(x.cuda(), w.cuda().t(), scale_x.cuda(), scale_w.cuda(),
                      out_dtype=torch.bfloat16, use_fast_accum=True)
mmaemu.testing.assert_bitwise_equal(yg.cpu(), y)
```

To see what a call will do without running it:

```python
print(mmaemu.explain(x, w.t(), scale_x, scale_w, out_dtype=torch.bfloat16, use_fast_accum=True))
```

```text
CallPlan(validation='ok', profile='ada-rtx4060-torch', preset='ada', preset_maturity='validated',
  split=SplitDecision(S=1, scheme='none', source='table', verified=True, table_id='heur_rtx4060_lt12.8.3.14',
                      note='table row: algo 60, tile 18, numSplitsK 1, customOption 0'),
  backend='reference', fallbacks=['cuda: tensors on cpu'], warnings=[])
```

The same is available from the shell: `python -m mmaemu explain 64 5760 1920 --out bf16 --fast` (M N K).

`mmaemu.accumulate(x, w, use_fast_accum=True)` returns the fp32 accumulator before the scales are applied (note:
`w`, not `w.t()`; it takes both operands as [rows, K]). It has no default for `use_fast_accum` on purpose.

## 4. Choose a profile

A *profile* names the platform whose results you want: the GPU, its SM count, the cuBLASLt version, and the caller
with its cuBLAS workspace size. The caller matters because the workspace size changes which split-K kernel cuBLAS
picks.

| Profile | Use it for results produced by |
|---|---|
| `ada-rtx4060-torch` (default) | `torch._scaled_mm` on an RTX 4060 (1 MiB workspace) |
| `ada-rtx4060-te` | Transformer Engine 2.3 on an RTX 4060 (4 MiB workspace), e.g. Evo 2 |
| `ada-rtx4060-ws0` | cuBLAS with `CUBLASLT_WORKSPACE_SIZE=0` on an RTX 4060 (never splits) |

Pass `profile="ada-rtx4060-te"` per call, or set `MMAEMU_PROFILE`. The experimental profiles
(`hopper-experimental`, `b200-mma-sync-experimental`) evaluate experimental instruction models without a cuBLAS
platform; they warn, and have no split-K table. Details: [presets_profiles.md](presets_profiles.md).

## 5. Split-K

cuBLAS splits the K dimension of some GEMMs (mostly small M x N with large K) and adds the partial results in an
order that depends on the shape, the formats, the output dtype, the accumulation mode, the workspace and the GPU.
mmaemu reads that decision from a table recorded on the reference GPU (`splitk="auto"`, the default).

- If the table decides, the call runs. `explain` shows the decision and where it came from.
- On the reference GPU itself, a shape missing from the table is decided by a live query of cuBLASLt's heuristic.
- Elsewhere, if the shape's M*N is more than twice the largest M*N that ever split in the table for its setting,
  mmaemu infers S = 1 and says so in the decision's note (`MMAEMU_SPLITK_STRICT=1` turns this inference off). The
  margin is measured: on the RTX 4060, cuBLASLt splits shapes up to 1.27 times that M*N.
- Otherwise the call raises `SplitKUnknownError` instead of guessing. Pass `splitk=1` (no split) or
  `splitk=(S, "serial")` / `splitk=(S, "pairwise")` if you know the decision, or record the shape on the reference
  GPU with `tools/record_heuristics.py`.

Typical model projections (large M x N) never split on the reference GPU; all Evo 2 projection shapes are S = 1.

## 6. Backends and devices

| Backend | Where | Speed (RTX 5070 Ti, Evo 2 1B shape) |
|---|---|---|
| `reference` | the tensors' device, CPU or CUDA: the float64 model | slow; for tests and small problems |
| `cuda:v1` | CUDA, sm_75 and newer, IEEE-exact | 1.9 TMAC/s |
| `cuda:v4` | CUDA, sm_80 and newer, IEEE-exact | 4.0 TMAC/s |
| `cuda:v9` | CUDA, gated: bf16 tensor cores for the exponents | 4.8 TMAC/s (opt-in outside sm_89) |
| `cuda:tc3` | CUDA, gated: fp16 tensor cores, sm_80/86/89 binaries only | no sm_120 binary (2.94 TMAC/s on the RTX 4060, T6) |

With `backend="auto"` mmaemu takes the highest-priority kernel that loaded and passed its self-test on the device
(tc3 > v9 > v4 > v1 > reference). v1 and v4 use only IEEE-defined instructions, so their results do not depend on
the GPU; they are validated on sm_89 (RTX 4060) and sm_120 (RTX 5070 Ti). v9 and tc3 depend on how a GPU's
tensor cores round, so they are auto-selected only on sm_89, where they were validated; elsewhere
`MMAEMU_ALLOW_GATED_KERNELS=1` allows them after their gates pass. E5M2 and mixed-format inputs always use the
reference backend. Force a backend with `backend="cuda:v4"` or `MMAEMU_BACKEND`. Details: [backends.md](backends.md).

The CPU is a full platform: every result can be computed with CPU tensors, no GPU needed. On the CPU, mmaemu
refuses to run while flush-to-zero is enabled (`torch.set_flush_denormal(True)`), because that would change
subnormal arithmetic.

## 7. Inside a model

mmaemu replaces one call: wherever a model computes an FP8 GEMM with `torch._scaled_mm` (or Transformer Engine's
FP8 `Linear`), call `mmaemu.scaled_mm` with the same FP8 operands and scales. A minimal module:

```python
class EmulatedFP8Linear(torch.nn.Module):
    """y = x @ W.T with E4M3 operands, as Transformer Engine computes it on an RTX 4060 (no bias)."""

    def __init__(self, weight, scale_x, scale_w):
        super().__init__()
        self.scale_x, self.scale_w = scale_x, scale_w                       # quantisation scales (fp32 tensors)
        self.wq = (weight.detach().float() * scale_w).clamp(-448, 448).to(torch.float8_e4m3fn)

    def forward(self, x):
        xq = (x.float() * self.scale_x).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(-1, x.shape[-1])
        y = mmaemu.scaled_mm(xq, self.wq.t(), 1 / self.scale_x, 1 / self.scale_w, out_dtype=torch.bfloat16,
                             use_fast_accum=True, profile="ada-rtx4060-te")
        return y.reshape(*x.shape[:-1], -1)
```

Things to get right, all of which change bits:

- **The quantisation.** The FP8 operands must be the ones the reference run used: same scales, same saturation
  (`clamp` to +-448 for E4M3 before the cast), same rounding of the cast. Hash them against the reference if you
  can; [the Evo 2 notebook](https://github.com/astding/mmaemu/blob/master/examples/evo2_blackwell.ipynb) does.
- **Padding.** Some callers pad M (vortex pads the sequence length to a multiple of 16 on its TE path). Padding
  rows do not change the other rows' results, but a different M can change the split-K decision.
- **Inference only.** mmaemu refuses inputs that require grad while grad mode is on. Detach the weight, and run
  the model under `torch.inference_mode()`.
- **bias.** Not supported in v0.1 (raises). Add the bias yourself after the GEMM only if the reference did the same.

What this gives you: each emulated GEMM returns the reference GPU's bits. The rest of the model (bf16 matrix
products, convolutions, normalisations, attention) still runs natively on your GPU and is not emulated, so a full
model's outputs are close to the reference GPU's, not identical; [evo2.md](evo2.md) measures how close for Evo 2.

## 8. Testing with mmaemu

`mmaemu.testing.assert_bitwise_equal(actual, expected)` compares dtypes, shapes and bit patterns, so +0 and -0
differ and NaN payloads count; on failure it reports the mismatch count by class (zero sign, NaN payload, value)
and the first ten mismatching elements. `mmaemu.testing.bits(t)` gives the integer view of a tensor.

Typical uses:

- **Golden outputs without a GPU.** Compute expected FP8 GEMM results on a CPU-only CI runner and compare your
  kernel's output on the GPU later.
- **Checking a native result.** On an RTX 4060 with cuBLASLt 12.8.3.14 and torch 2.7.1, `torch._scaled_mm` and
  `mmaemu.scaled_mm` (profile `ada-rtx4060-torch`) return the same bits for the validated scope; a difference
  there is a bug report.
- **Attributing a discrepancy.** When a model differs between two GPUs, replace its FP8 GEMMs with mmaemu on both:
  what remains comes from the non-FP8 operations.

## 9. Errors and warnings

| Exception / warning | Meaning | What to do |
|---|---|---|
| `ReferenceRejectsError` | torch 2.7.1 or the profile's cuBLAS would reject this call (layout, divisibility, dtypes) | fix the call as you would for `torch._scaled_mm` |
| `UnsupportedFeatureError` | bias, row-wise scales, grad mode, or a feature of an experimental preset | see [limitations.md](limitations.md) |
| `SplitKUnknownError` | the split-K decision is unknown for this shape | pass `splitk` explicitly (section 5) |
| `BackendUnavailableError` | an explicitly requested backend cannot run here | use `backend="auto"` |
| `MmaemuError` | e.g. CPU flush-to-zero is on | `torch.set_flush_denormal(False)` |
| `ExperimentalPresetWarning` | an experimental preset or profile is in use | read its validated scope |
| `UnverifiedSplitWarning` | a split decision whose reduction was never replayed on hardware | treat the result as experimental |
| `BackendFallbackWarning` | a faster CUDA kernel failed to load or failed a gate, or a kernel had to be compiled with NVRTC; a slower kernel, the same kernel built by NVRTC, or the reference model runs instead (the message says which) | `python -m mmaemu info` shows why |

The exception classes above derive from `mmaemu.MmaemuError` and from the matching built-in class (`RuntimeError`,
`NotImplementedError`, `LookupError`); argument errors raise the built-in `TypeError`, `ValueError` or `KeyError`.

## 10. Environment variables

| Variable | Effect |
|---|---|
| `MMAEMU_PROFILE` | default profile |
| `MMAEMU_BACKEND` | default backend (`auto`, `reference`, `cuda`, `cuda:v1`, `cuda:v4`, `cuda:v9`, `cuda:tc3`) |
| `MMAEMU_SPLITK_STRICT=1` | no envelope inference of S = 1 |
| `MMAEMU_DISABLE_CUDA=1` | never touch CUDA |
| `MMAEMU_ALLOW_GATED_KERNELS=1` | allow v9 / tc3 on capabilities they were not validated on, if their gates pass |
| `MMAEMU_CACHE_DIR` | NVRTC cubins and self-test verdicts (default `${XDG_CACHE_HOME:-~/.cache}/mmaemu`) |
| `MMAEMU_LIBCUDA_PATH`, `MMAEMU_NVRTC_PATH` | explicit driver / NVRTC library paths |
