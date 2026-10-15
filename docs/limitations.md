# Limitations of mmaemu v0.1

mmaemu v0.1 reproduces one reference platform: the RTX 4060 (Ada, sm_89, 24 SMs) with cuBLAS/cuBLASLt 12.8.3.14,
called through torch 2.7.1 `torch._scaled_mm` (1 MiB workspace) or Transformer Engine 2.3 (4 MiB workspace). This
page lists what that scope excludes. [validation.md](validation.md) shows the evidence for what it includes, and
the scope table there covers every row of the scope table of an internal (unpublished) review of the prototype.

## What mmaemu reproduces, and what it does not

- mmaemu computes **what the reference GPU would output** for a GEMM, on whatever device it runs on (CPU, or a
  CUDA GPU through its kernels). It does **not** predict what another GPU's own FP8 tensor cores would output:
  a different GPU, SM count, cuBLAS version or workspace size can choose different kernels and split-K settings and
  give different bits.
- Validated: the `ada` preset and the three `ada-rtx4060-*` profiles, with the exceptions below. The `hopper` and
  `b200_mma_sync` (alias `blackwell`) presets and their profiles are **experimental**: they warn on use.
- **sm_120 is not modelled.** mmaemu ships no preset, profile or split-K table for RTX 50-series GPUs (sm_120, e.g.
  RTX 5070 Ti), and `b200_mma_sync` is not such a model: it fits vectors produced by an `mma.sync` rewrite into
  fp16 HMMA instructions on sm_90/sm_100, and the paper's own sm_120 model disagrees with it. Khattak and Mikaitis
  (arXiv:2512.07004v4, section 4.1.8, Tables 3-4) report RTX PRO 6000 (sm_120) tensor-core numerics identical to
  the B200's; probes on an RTX 5070 Ti are consistent with that model
  ([2026-10-13-sm120](validation_logs/2026-10-13-sm120/README.md)), but it is not validated at GEMM level. The
  names `sm_120` and `rtx_blackwell` are rejected.
- Hopper (`hopper`): validated by the toolbox only at instruction level and only with c = 0, so how the accumulator
  enters the sum is not constrained; there is no GEMM-level evidence and slow accumulation raises.

## Features outside v0.1 (they raise, never approximate)

| Feature | Behaviour | Why |
|---|---|---|
| `bias` | `UnsupportedFeatureError` (also when passed positionally) | the fused-bias epilogue was measured only in an internal (unpublished) review of the prototype |
| row-wise scales (`scale_a` [M,1], `scale_b` [1,N]) | `UnsupportedFeatureError` | cuBLAS routes them to a CUTLASS stream-K kernel whose results depend on the SM count |
| inputs requiring grad under grad mode | `UnsupportedFeatureError` | mmaemu is inference-only; TE also recomputes scales from its amax history when grad is on |
| slow accumulation on `hopper` / `b200_mma_sync` | `UnsupportedFeatureError` | not characterised |
| `splitk="auto"` on a shape the profile cannot resolve | `SplitKUnknownError` | split decisions come from recorded tables, not from the shape |
| CPU flush-to-zero active | `MmaemuError` | it would change subnormal arithmetic |
| calls torch 2.7.1 or the profile's cuBLAS rejects | `ReferenceRejectsError` | e.g. e5m2 x e5m2, e4m3 x e4m3 -> e5m2 on the RTX 4060, misaligned storage offsets, K or N not a multiple of 16 |

Also not in v0.1: CUDA graph capture (the non-finite-input check synchronises with the host), a `torch.library`
custom op, a compiled CPU backend.

## Split-K is profile-scoped

- cuBLAS splits K on small M*N shapes. Which shapes split, and how the partial sums are combined, depends on the
  output dtype, fast/slow accumulation, the workspace size and the GPU's SM count; it is read from a table recorded
  on the RTX 4060, not derived from the shape.
- On the profile's own GPU, a shape that is not in the table is decided by a live cuBLASLt query. Elsewhere it
  raises `SplitKUnknownError`, unless its M*N is more than twice the largest M*N that splits in the table for its
  setting: then S = 1 is *inferred*. On the RTX 4060, cuBLASLt splits shapes up to 1.27 times that M*N, and at K
  below the smallest K that splits in the table, so neither the table's region nor a K bound alone is safe
  ([splitk-envelope](validation_logs/2026-10-14/splitk-envelope/README.md)). `MMAEMU_SPLITK_STRICT=1` disables the
  inference.
- Split decisions with S > 1 on the TE profile (`ada-rtx4060-te`) are experimental (`UnverifiedSplitWarning`).
- Split combinations that were never replayed on hardware warn (`UnverifiedSplitWarning`).
- Other Ada GPUs (e.g. the L40S with 142 SMs) may split shapes that never split on the 24-SM RTX 4060. They need
  their own profile.

## Backends and platforms

- E5M2 and mixed E4M3/E5M2 inputs, and non-`ada` presets, run on the reference backend (float64 model) only:
  correct but slow. The CUDA kernels cover E4M3 x E4M3 with the `ada` preset.
- The CUDA kernels are tested on Linux x86_64 (WSL2 included). They load wherever `libcuda.so.1` is found (Linux
  aarch64 is untested); on native Windows and macOS the reference backend is used.
- An unsplit fast-accumulation chain with K >= 2^24 raises `ValueError` on the kernels v4, v9 and tc3 (v1 takes
  chains up to 2^30); pass `backend="cuda:v1"` or `"reference"` for such K.
- The kernels v1 and v4 are IEEE-exact by construction and must pass a smoke self-test on each device. The fast
  kernels v9 and tc3 (both passed acceptance on the RTX 4060, T6) rely on properties of the
  fp16/bf16 tensor cores; they are validated on sm_89 only, are
  auto-selected only there (or with `MMAEMU_ALLOW_GATED_KERNELS=1` after their gates pass), and tc3 ships no
  binary for capabilities other than sm_80/86/89.
- Multi-GPU use is untested (the per-context module cache is tested with a second context on one GPU).
- The release fatbins are built with CUDA 12.8; sm_100/sm_120 binaries need a driver that supports them (R570 or
  newer, by NVIDIA's documentation, not measured here). Without a usable binary mmaemu falls back to NVRTC (if
  `libnvrtc` is found) and then to the reference backend, with one `BackendFallbackWarning` that names the failed
  kernels.
- Performance: emulation is far slower than native FP8 tensor cores. See [performance.md](performance.md).

## Version drift

The profiles are pinned to cuBLASLt 12.8.3.14 and torch 2.7.1. Another cuBLAS version may choose different
kernels or splits; the API is tested against torch 2.6.0, 2.7.1 and the latest torch on the CPU, but the
numerics are only claimed for the profile's versions. The zero-dimension rules (M, N or K = 0) and the API
rejection rules follow the R6 and R1 recordings of the RTX 4060 (`recordings/rtx4060_2026-10`) and the probes of
[audit-probes](validation_logs/2026-10-14/audit-probes/README.md).

## Evo 2

The Evo 2 result that motivated mmaemu (FP8-trained models giving the reference GPU's outputs elsewhere) holds in
this scope: RTX 4060, inference with the checkpoint's frozen FP8 scales, finite inputs, positive scales. On this
GPU the Evo 2 projection shapes never use split-K at 0, 1 or 4 MiB with the bf16 output Evo 2 uses (they split only
at 32 MiB: (K, N) = (8192, 24576), M = 65..74, fp16 or fp8 output), which is why torch (1 MiB) and Transformer
Engine (4 MiB) agree on them. They disagree on split-prone shapes on the same GPU, so "native equals
TE" is a property of these shapes, not of the two callers.

## Corrections of earlier statements

The prototype (tcemu) and its notes made statements that an internal (unpublished) review of the prototype found
unsupported. v0.1 replaces them:

| Earlier statement | v0.1 |
|---|---|
| The B200 preset was presented as a model of B200 and RTX Blackwell GPUs, and the toolbox as modelling one by the other. | Renamed `b200_mma_sync` (alias `blackwell`, warning). It fits the toolbox's B200 `mma.sync` vectors, which come from an HMMA rewrite; the model is not identified, sm_120 is untested and not modelled. |
| The small-M*N, large-K gap was described as periodic chain restarts with an unknown combination order. | Interleaved split-K: 64-wide k-tiles assigned round-robin to S splits, one chain per split, serial or pairwise fp32 combination selected by cuBLASLt's customOption; the slow-accumulation chunking applies per split. |
| The epilogue was written as y = fp32(acc * alpha), and `scaled_mm` was called identical to `torch._scaled_mm` without a scope. | y = fma(acc, alpha, +0) computed exactly; fp8 outputs, `scale_result` and canonical NaN are modelled; bias and row-wise scales raise; every bitwise claim names the reference platform. |
| A kernel comment stated that cuBLAS's sm_89 kernels do not split K. | Removed: cuBLAS does split K on small M*N shapes (see above). |
| The post-swap checklist compared the toolbox's B200 vectors on the new GPU. | Those vectors cannot discriminate between the candidate models; the sm_120 bring-up runbook uses discriminating probes instead. |
| NaN inputs were described as treated as 0. | The prototype kernel v1 decoded NaN codes as 0 and v0 as +-480. v0.1 implements the hardware rule: a NaN code anywhere in row m of A or column n of B makes output (m, n) the canonical NaN. |
| A full-matrix comparison count was quoted that covered only part of the shapes and was not saved. | Evidence is the recorded hardware outputs listed in [validation.md](validation.md), replayed by mmaemu's test tiers. |
| Native and TE were described as bitwise equal in general. | Only on shapes that do not split on the RTX 4060 (all Evo 2 projections); see "Evo 2" above. |
| Claims covered all GPUs, called results assured, or listed sm_90 as supported. | Every claim names the GPU, SM count, cuBLAS, TE and torch versions and the shape range. sm_90 native FP8 is not validated. |
