# FAQ

### Why does my FP8 model give different results on another GPU?

FP8 GEMMs on tensor cores do not round like IEEE arithmetic: the tensor cores align and truncate blocks of
products, cuBLAS may split K and add the partial sums in an order of its own choosing, and the `torch._scaled_mm`
epilogue rounds twice ([semantics.md](semantics.md)). Two GPUs can therefore return different bits for the same
FP8 GEMM, and a model trained with FP8 can depend on those bits. Routing its FP8 GEMMs through `mmaemu.scaled_mm`
computes them as the profile's reference GPU (the H100 PCIe by default, or the RTX 4060) does, on the CPU or on the
GPU the model runs on.

### Why does Evo 2 1B give a wrong loss on an RTX 5070 Ti or RTX 5090?

In evo2 issue #157 the upstream test gives a loss of 1.248 on an RTX 5070 Ti, where 0.502 is expected, and 1.365
with FP8 disabled; an RTX 5090 shows the same failure. With its FP8 projections computed by mmaemu, the loss on the
RTX 5070 Ti, computed as the upstream test does (in bf16), is 0.5022 with the H100 PCIe profile `hopper-te`, the
H100's value in bf16, and 0.5027 with the RTX 4060 profile `ada-te` (in fp32: 0.5024 with `hopper-te`, 0.5025
with `ada-te`, 0.5026 on the H100). The FP8 GEMMs return the bits of the profile's reference GPU; the rest of the
model runs natively, so the logits are not bit-identical to the H100's or the RTX 4060's ([evo2.md](evo2.md),
[log](validation_logs/2026-10-16/evo2-h100-profile-rtx-5070-ti/README.md)). mmaemu was not run on an RTX 5090, nor
on the 20B model; the 7B needs no emulation, since it passes natively
([measured after 0.1.0](https://github.com/astding/mmaemu/blob/master/README.md#measured-after-010)).
NVIDIA BioNeMo also offers a 1B checkpoint fine-tuned for bf16 (`evo2/1b-8k-bf16:1.0`), with other weights than the
original checkpoint.

### Does it work for Evo 2 40B?

On the RTX 5070 Ti, yes, for the upstream test and the windows that were run: with the weights in host RAM (82 GB)
and one block at a time on the 16 GB GPU, the original 40B checkpoint scores a loss of 0.2161 on the upstream test
with either TE profile, `hopper-te` (H100 PCIe) or `ada-te` (RTX 4060), where 0.2159 is expected (the official path
on an H100 gives 0.2159). On 100 BRCA1 variants the AUROC is 0.8569 with `hopper-te` and 0.855 with `ada-te`; the
official 40B on the H100 gives 0.855 ([log](validation_logs/2026-10-16/evo2-h100-profile-rtx-5070-ti/README.md)).
With `hopper-te` the FP8 GEMMs are computed as the H100 PCIe computes them, but no 40B projection was captured on an
H100: of Evo 2's projections, only the 21 of one 1B window were captured on the H100 and replayed bit for bit.
Natively the RTX 5070 Ti gives 1.4023 (1.4043 with bf16 GEMMs) and predicts the next base correctly about one time
in four ([evo2.md](evo2.md#evo-2-40b)). To run it: [examples/evo2_offload.py](../examples/evo2_offload.py)
(`--profile hopper-te`, the default, or `ada-te`; [how](evo2.md#running-evo-2-40b)); the notebook covers the 1B
only.

### Does mmaemu make my GPU compute like the reference GPU?

Only for the FP8 GEMMs you route through `mmaemu.scaled_mm` (or `accumulate`). Each of those returns the bits of the
profile's reference GPU: the H100 PCIe by default, the RTX 4060 with `ada`, `ada-te` or `ada-ws0`. Everything else in
your program (bf16/fp16 matrix products, convolutions, reductions, attention kernels) runs natively, with your GPU's
own rounding. For Evo 2 1B on an RTX 5070 Ti this brings the loss of the upstream test from 1.248 (the value in
evo2 issue #157) back to 0.5022 with the profile `hopper-te` and 0.5027 with `ada-te` (expected: 0.502; all computed
as the upstream test does), but the logits are not bit-identical to the H100's or the RTX 4060's
([evo2.md](evo2.md)).

### Why not run the FP8 layers in bf16 instead?

Because the model's weights can depend on how its FP8 GEMMs round. Evo 2 1B computes its projections in FP8, and
replacing them with bf16 GEMMs (what evo2 does for its 7B models without Transformer Engine; the 1B refuses to load)
gives a loss of 1.365 on the upstream test prompts, worse than the native FP8 result of the RTX 5070 Ti.

### Is mmaemu a model of Blackwell's (or Hopper's) FP8 arithmetic?

Of Hopper's, for cuBLAS on one GPU: the three H100 PCIe profiles, `hopper` (the default), `hopper-te` and
`hopper-ws0`, reproduce `torch._scaled_mm` and Transformer Engine on an H100 PCIe with cuBLASLt 12.8.3.14
([validation.md](validation.md)); with Transformer Engine, split decisions with S > 1 and the envelope's S = 1
inferences are experimental and warn. That is not a model of the sm_90 `mma.sync` instruction, the H100 SXM or the
H200. The other reference platform of 0.1.0 is the RTX 4060 (Ada, sm_89), also with cuBLASLt 12.8.3.14. On the GPU
it runs on, mmaemu computes the reference platform's result, not that GPU's own.

Of Blackwell's, no. sm_120 is not modelled: mmaemu ships no preset, profile or split-K table for it. Khattak and
Mikaitis report RTX PRO 6000 (sm_120) tensor-core numerics identical to the B200's, and probes on an RTX 5070 Ti are
consistent with that, but it is not validated at GEMM level. The `b200_mma_sync` preset is an experimental
instruction-level fit to published test vectors, not a platform model ([limitations.md](limitations.md)).

### Which GPUs has it been run on?


| GPU | Role | Result |
|---|---|---|
| RTX 4060 (sm_89) | reference platform of the profiles `ada`, `ada-te` and `ada-ws0`: the hardware ground truth comes from it; on fa0fd90 (16 Oct; repeated after the profile rename, [log](validation_logs/2026-10-16/rtx-4060-sm89-renamed/README.md) and [log](validation_logs/2026-10-16/rtx-5070-ti-renamed/README.md)): T2 on the CPU ([log](validation_logs/2026-10-16/t2/README.md)); T3, T3p and the runs that validated h1 and h4, the `hopper` preset's kernels ([log](validation_logs/2026-10-16/rtx-4060-sm89/README.md)); T4 ([log](validation_logs/2026-10-16/t4/README.md)); T6 on da9edd4 | T2 passes: replayed on the CPU, mmaemu reproduces every output it models in 147 of the 173 recordings (the 15 tables and 11 bias, row-wise, fp16-accumulator or HMMA recordings have none), apart from the old emulator recording's 6,120 NaN-decode outputs; v1, v4, v9 and tc3 pass T3 and T6 there, h1 and h4 pass T3 and replay the H100 probe's E4M3 x E4M3 calls and Evo 2 outputs equal, live `torch._scaled_mm` equals mmaemu's reference bit for bit (T3p), and T4 kills every kernel mutant |
| RTX 5070 Ti (sm_120) | kernels v1 and v4 validated: on fa0fd90 (16 Oct; repeated after the profile rename, [log](validation_logs/2026-10-16/rtx-5070-ti-renamed/README.md)) T3 with the full Evo 2 accumulators ([log](validation_logs/2026-10-16/t3-rtx-5070-ti-sm120/README.md)) and T4 ([log](validation_logs/2026-10-16/t4/README.md)); full device independence (v1, v4, v9) and a development wheel installed without nvcc, 13 Oct; h1 and h4 (the `hopper` preset) validated, 16 Oct ([h1](validation_logs/2026-10-16/h1-rtx-5070-ti/README.md), [h4](validation_logs/2026-10-16/h4-rtx-5070-ti/README.md)), and the H100 probe replayed through h4 on fa0fd90 ([log](validation_logs/2026-10-16/rtx-5070-ti-replays/README.md)) | bitwise equal to the CPU reference (the full Evo 2 accumulators of v1 and v4 to the RTX 4060's) |
| H100 PCIe (sm_90) | reference platform of the profiles `hopper` (the default), `hopper-te` and `hopper-ws0`: a probe of 1,450 live `torch._scaled_mm` calls and the 21 Evo 2 1B projections, 15 Oct; a split-K sweep with 2,744 live jobs (2,416 of them split), 16 Oct; before that, T3 on the release kernels and the 21 Evo 2 projections (fixed inputs) with v1 and v4 compiled by NVRTC | mmaemu reproduces all 1,450 probe calls (1,423 bit for bit; the 27 e4m3 x e4m3 -> e5m2 calls that cuBLASLt rejected raise `ReferenceRejectsError`), and every sweep job and every Evo 2 output bit for bit (replayed on an RTX 5070 Ti: the probe on the reference backend, h1 and h4, the sweep on h4 and the reference backend; on fa0fd90, the probe through h4 on the RTX 5070 Ti and the RTX 4060, and its E4M3 x E4M3 calls through h1 on the RTX 4060); v1 and v4 passed T3, but sm_90 is not yet in their validated set (h1 and h4 are validated on sm_89 and sm_120 and have never run on an H100) |
| CPU (x86_64) | the float64 reference model | the oracle the kernels are compared with |

See [validation.md](validation.md) for the logs.

### How fast is it?

Much slower than native FP8. On the RTX 4060 the emulated Evo 2 1B projection runs at 2.94 TMAC/s with the
`ada` preset's default kernel there (tc3), against 29.8 TMAC/s for native FP8. On an RTX 5070 Ti it runs at about
4 TMAC/s with that preset's default kernel there (v4), and an Evo 2 1B forward pass over 8,192 bp takes about 1.0 s
with mmaemu against 0.45 s natively. The H100 profiles run on their own kernel, h4, at about 4.1 to 4.6 TMAC/s on
the RTX 5070 Ti (h4 needs sm_80; on sm_75 GPUs they run on h1, which reaches about 2 TMAC/s on the RTX 5070 Ti).
The CPU reference model is far slower: the prototype's reached 116-181 million MAC/s (about 0.15 GMAC/s)
with 8 threads in an internal review: fine for tests and single layers, too slow for whole models. See
[performance.md](performance.md).

### Can I use it without a GPU, for example for golden FP8 outputs in CPU-only CI?

Yes. Every function runs on CPU tensors with the exact float64 reference model, and gives the same bits as the
CUDA kernels.

### Does it support training, bias, row-wise scales?

Not in v0.1. mmaemu is inference-only: it raises for inputs that require grad under grad mode, for `bias` and for
row-wise scales, rather than approximate them.

### Does it work with `torch.compile`?

The entry points are excluded from tracing (`torch.compiler.disable`): a compiled model calls them as ordinary
Python functions, with a graph break. This is deliberate: an internal review of the prototype found Inductor (with
Triton 3.8 or older) double-rounding FP8 conversions.

### How do I use it with Transformer Engine?

Use the TE profile of the platform whose results you want: `hopper-te` (TE's 32 MiB cuBLAS workspace on the H100
PCIe; its S > 1 split decisions and envelope inferences warn) or `ada-te` (TE's 4 MiB workspace on the RTX 4060).
Quantise the input and the weight with TE's FP8 scales, and pass the *reciprocal* scales to `scaled_mm`, as TE does.
[examples/evo2_offload.py](../examples/evo2_offload.py) shows a complete replacement of TE's FP8 projections, with
`hopper-te` (its default) or `ada-te`, and the
[Evo 2 notebook](https://github.com/astding/mmaemu/blob/master/examples/evo2_blackwell.ipynb) with `ada-te`.

### How do I know a call is within the validated scope?

`mmaemu.explain(...)` shows the preset's maturity, the split-K decision with its source and whether that kind of
decision was replayed on hardware (`verified`). Calls outside the validated scope warn
(`ExperimentalPresetWarning`, `UnverifiedSplitWarning`) or raise (`SplitKUnknownError`, `UnsupportedFeatureError`).

### Why does a shape raise `SplitKUnknownError`?

cuBLAS may split K for that shape on the reference GPU, the recorded table does not contain it, and its M*N is not
large enough for the profile's envelope to infer S = 1. mmaemu does not guess: pass `splitk=1` if you know cuBLAS
does not split it, `splitk=(S, "contiguous")` on the H100 profiles or `(S, "serial" | "pairwise")` on the RTX 4060
ones if you know how it does, or record it on the profile's GPU with `tools/record_heuristics.py`.

### Can mmaemu be extended to another reference GPU?

Yes, with hardware evidence: instruction-level probes, cuBLAS GEMM recordings and a split-K table recorded on that
GPU. The [sm_120 bring-up runbook](runbooks/sm120_bringup.md) describes the procedure. A new preset or profile
starts as experimental.

### How does mmaemu differ from MMA-Sim, Hawkeye or FPRev?

Those projects model tensor-core instructions on many GPUs or measure accumulation orders. mmaemu reproduces whole
`torch._scaled_mm` FP8 calls of two reference platforms (the H100 PCIe and the RTX 4060), including the split-K plan
cuBLASLt chooses and the epilogue, and validates them against recordings of those GPUs. The README's
[related-work section](https://github.com/astding/mmaemu/blob/master/README.md#related-work) describes each.

### Where does the tensor-core model come from?

From Khattak and Mikaitis, "Accurate Models of NVIDIA Tensor Cores" (ACM TACO, 2026), and their MATLAB Tensor Core
toolbox (BSD-2-Clause). mmaemu re-validates the Ada model against its own RTX 4060 recordings and the Hopper
model against a probe of an H100 PCIe, which also measured how c enters (the toolbox's H100 vectors have c = 0), and
adds the cuBLAS layer (split-K, epilogue) on top. See the citation section of the README.
