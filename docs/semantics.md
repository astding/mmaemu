# What mmaemu computes (numerics contract, v0.1)

This page describes, operation by operation, the arithmetic that mmaemu reproduces. It is the user-facing rendering
of section 3 of the v0.1 specification. The ground truth is hardware, not a paper: a behaviour is called
*validated* only when recordings of the reference GPU reproduce it bit for bit
([validation.md](validation.md)). Everything else is *experimental* (it warns) or raises.

**Reference platform.** NVIDIA GeForce RTX 4060 (Ada, sm_89, 24 SMs), cuBLAS/cuBLASLt 12.8.3.14, called through
torch 2.7.1 `torch._scaled_mm` (1 MiB cuBLAS workspace) or through Transformer Engine 2.3 (4 MiB workspace). A
*profile* names this platform; a *preset* names the tensor-core arithmetic
([presets_profiles.md](presets_profiles.md)).

"Bitwise" and "bit for bit" mean equal bit patterns, including the sign of zero and the NaN payload. Numeric
equality (`==`, `torch.equal`) is weaker: it treats +0 and -0 as equal.

## 1. The instruction: one FP8 tensor-core step (preset `ada`)

The Ada FP8 `mma.sync` instruction (m16n8k32 in SASS: QMMA.16832) computes, for every output element, a chain of
16-term blocks:

    d = c + sum over i < 16 of a_i * b_i          (one block step; c is the running fp32 accumulator)

The hardware does not round each product or each addition. Instead, per block:

1. Input exponents are clamped at the format's minimum normal exponent (E4M3: -6, E5M2: -14), so subnormal
   inputs keep their true exponent.
2. The block maximum exponent E is taken over the nonzero products and c. Products that are exactly zero do
   not take part.
3. Every product and c are aligned to 2^E and truncated toward zero to 13 fraction bits below 2^E.
4. The aligned terms are summed exactly. The sum is truncated toward zero to 13 fraction bits relative to its
   own leading bit, and becomes the new accumulator (fp32).
5. A block result that is exactly zero, or that truncates to zero, is +0 (hardware rule c2-M1 of the RTX 4060).
6. NaN results are the canonical fp32 NaN 0x7FFFFFFF.

`mmaemu.model.dot(a, b, c=None, *, preset="ada", fmt=None)` evaluates exactly this chain. The format is inferred
from float8 dtypes; float64 inputs need an explicit `fmt` (a wrong default format changes results for E5M2 values
below 2^-6). Mismatched K raises `ValueError`; there is no broadcasting.

## 2. A GEMM: accumulation plan on the reference profile

`mmaemu.accumulate(a, b, *, use_fast_accum, ...)` returns the fp32 accumulator of `a @ b.T` *before* the scale is
applied. A GEMM runs as a plan: a list of chains, each starting from +0, and an fp32 order in which their results
are combined.

| Case | Chains | Combination |
|---|---|---|
| fast accumulation (`use_fast_accum=True`), unsplit | one chain over k = 0 .. K-1, in order | none |
| slow accumulation (`use_fast_accum=False`), unsplit | one chain per consecutive 128-element chunk of K | `((+0 + ch0) + ch1) + ...`, fp32 round-to-nearest-even |
| split-K, S > 1 | see below | serial or pairwise (below) |

**Split-K.** cuBLAS may split K across S thread blocks. The split count and the reduction scheme are chosen by the
cuBLASLt heuristic. They depend on M, N, K, the formats, the output dtype, fast/slow accumulation, the workspace size
and the GPU, and cannot be derived from the shape alone. When S > 1:

- K is cut into 64-wide k-tiles t (a ragged last tile is zero-padded). Split s owns the tiles with
  t mod S = s, in increasing t. Adjacent tiles therefore always belong to different splits.
- Fast accumulation: one chain per split over its tiles. Slow accumulation: within a split, a new chain starts
  at every 2nd tile *of that split*, and the chunk results are added serially from +0 in fp32.
- The S partial results p_0 .. p_(S-1) are combined in fp32 round-to-nearest-even:
  - **serial** (cuBLASLt customOption 0 or 2): `((p0 + p1) + p2) + ...`;
  - **pairwise** (customOption 1): h = ceil(S/2); slot_i = p_i + p_(i+h) if i+h < S-1, else p_i; the result is
    `(((p_(S-1) + slot_0) + slot_1) + ...) + slot_(h-1)`. For S = 3 this is `(p2 + p0) + p1`.
- The scale (alpha, below) is applied after the combination.

How mmaemu decides S and the scheme (`splitk="auto"`) is described in
[presets_profiles.md](presets_profiles.md#split-k-resolution). You can always pass it explicitly:
`splitk=1`, or `splitk=(S, "serial")` / `splitk=(S, "pairwise")`.

**Non-finite inputs.** Wherever the IEEE value of sum(a*b) is non-finite, the result is that value: NaN
codes, Inf * 0 and +Inf + -Inf give NaN, so any NaN code in row m of A or column n of B makes output (m, n) NaN.
The patch is applied after the combination; NaN is canonical.

**Empty dimensions.** M = 0 or N = 0 gives an empty result. K = 0 gives an accumulator of +0 everywhere, and the
epilogue then applies. (These rules are preliminary until the dedicated zero-dimension recordings are replayed.)

## 3. The epilogue of `torch._scaled_mm` (tensor-wise scales)

`mmaemu.scaled_mm(input, mat2, scale_a, scale_b, bias=None, scale_result=None, out_dtype=None,
use_fast_accum=False, *, out=None, profile=None, splitk="auto", backend="auto")` takes the arguments of torch 2.7.1
`torch._scaled_mm` (same names, order and defaults; `self` is exposed as `input`). After the accumulator:

1. alpha = RN32(scale_a * scale_b).
2. y = RN32(acc * alpha + (+0)): a fused multiply-add with a +0 addend, computed exactly in float64 and rounded
   once to fp32. Consequences: a zero product with a negative sign gives +0, never -0 (acc = 0 with alpha < 0,
   alpha = +-0, or alpha underflowing to 0), and Inf * 0 gives NaN.
3. Output dtype:
   - `torch.float32`: y.
   - `torch.bfloat16` / `torch.float16`: y rounded to nearest even. This is a *double* rounding (fp32 first),
     and it differs from rounding the exact product once on constructed ties.
   - `torch.float8_e4m3fn` / `torch.float8_e5m2`: if `scale_result` is given, y = RN32(y * scale_result) (a second
     fp32 multiply); then saturation to +-448 (E4M3) or +-57344 (E5M2), Inf included, NaN passing through; then
     round-to-nearest-even. `scale_result` is ignored for the other output dtypes.
4. NaN is canonical: fp32 0x7FFFFFFF, bf16/fp16 0x7FFF, fp8 0x7F.
5. The result never depends on the previous contents of an `out=` buffer.

The default `out_dtype` is the input dtype and the default `use_fast_accum` is False, as in torch. Evo 2 calls
`use_fast_accum=True` with bf16 output.

## 4. What raises instead of approximating

| Error | When |
|---|---|
| `TypeError` | wrong Python argument types; non-FP8 or `fnuz` dtypes where FP8 is required |
| `ReferenceRejectsError` | wherever torch 2.7.1 or the profile's cuBLAS rejects the call: shape/divisibility/layout and storage-offset rules, e5m2 x e5m2, scale dtype/numel, out= dtype/shape, e4m3 x e4m3 -> e5m2 output on the RTX 4060 profiles |
| `UnsupportedFeatureError` (a `NotImplementedError`) | `bias`, row-wise scales, inputs that require grad under grad mode (mmaemu is inference-only), preset features that are not characterised (slow accumulation on `hopper` / `b200_mma_sync`) |
| `SplitKUnknownError` (a `LookupError`) | `splitk="auto"` cannot determine S for this shape on this profile; the message names three ways out |
| `MmaemuError` | CPU flush-to-zero is active (`torch.set_flush_denormal(True)`), which would change subnormal arithmetic |

`mmaemu.accumulate` is more permissive than `scaled_mm`: any K (zero-padded; zero products never set the block
maximum), any alignment and layout (a misaligned operand is copied), and `use_fast_accum` must always be given
explicitly.

## 5. Device independence

Every backend gives the same bits as `backend="reference"` on the CPU: the float64 reference model, the CUDA
kernels, and the reference model on a CUDA device. The CUDA kernels are integer/IEEE-exact implementations of the
same arithmetic, and the gated fast kernels pass a per-device self-test before they are used. Public entry points
run outside `torch.compile` graphs, because Inductor's fp8 conversion double-rounds on some versions.

## 6. Other presets

| Preset | Block size | Fraction bits (terms / result) | c enters | Slow accumulation | Maturity |
|---|---|---|---|---|---|
| `ada` | 16 | 13 / 13 | in the alignment (early) | 128-element chunks | validated |
| `hopper` | 32 | 13 / 13 | in the alignment (early) | raises | experimental |
| `b200_mma_sync` (alias `blackwell`) | 32 | 25 / 23 | after the block sum, fp32 RNE (late) | raises | experimental |

The `hopper` preset fits the toolbox's H100/H200 vectors, which were recorded with c = 0 and therefore cannot show
how c enters the sum. The `b200_mma_sync` preset fits the toolbox's B200 vectors, which were produced by `mma.sync`
compiled for sm_90/sm_100 into two fp16 HMMA instructions plus an fp32 add; it is not a model of the B200's
native FP8 path, nor of sm_120 GPUs (RTX 50 series such as the RTX 5070 Ti). See
[limitations.md](limitations.md).
