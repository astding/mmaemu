# FAQ

### Why does my FP8 model give different results on another GPU?

FP8 GEMMs on tensor cores do not round like IEEE arithmetic: the tensor cores align and truncate blocks of
products, cuBLAS may split K and add the partial sums in an order of its own choosing, and the `torch._scaled_mm`
epilogue rounds twice ([semantics.md](semantics.md)). Two GPUs can therefore return different bits for the same
FP8 GEMM, and a model trained with FP8 can depend on those bits. Routing its FP8 GEMMs through `mmaemu.scaled_mm`
computes them as the reference GPU (RTX 4060) does, on the CPU or on the GPU the model runs on.

### Why does Evo 2 1B give a wrong loss on an RTX 5070 Ti or RTX 5090?

In evo2 issue #157 the upstream test gives a loss of 1.248 on an RTX 5070 Ti, where 0.502 is expected, and 1.365
with FP8 disabled; an RTX 5090 shows the same failure. With its FP8 projections computed by mmaemu, the loss on the
RTX 5070 Ti is 0.5027 (computed as the upstream test does). mmaemu was not run on an RTX 5090, nor on the 7B or
20B models, and it reproduces the RTX 4060 (Ada), not Hopper ([evo2.md](evo2.md)). NVIDIA BioNeMo also offers
a 1B checkpoint fine-tuned for bf16 (`evo2/1b-8k-bf16:1.0`), with other weights than the original checkpoint.

### Does it work for Evo 2 40B?

On the RTX 5070 Ti, yes, for the upstream test and the windows that were run: with the weights in host RAM (82 GB)
and one block at a time on the 16 GB GPU, the original 40B checkpoint scores a loss of 0.2161 on the upstream test,
where 0.2159 is expected (the official path on an H100 gives 0.2159), and on 100 BRCA1 variants the same AUROC as
the official 40B on the H100 (0.855). Natively the RTX 5070 Ti gives 1.4023 (1.4043 with bf16 GEMMs) and predicts the next base
correctly about one time in four ([evo2.md](evo2.md#evo-2-40b)). To run it:
[examples/evo2_offload.py](../examples/evo2_offload.py) ([how](evo2.md#running-evo-2-40b)); the notebook covers the
1B only.

### Does mmaemu make my GPU compute like an RTX 4060?

Only for the FP8 GEMMs you route through `mmaemu.scaled_mm` (or `accumulate`). Each of those returns the RTX 4060's
bits. Everything else in your program (bf16/fp16 matrix products, convolutions, reductions, attention kernels) runs
natively, with your GPU's own rounding. For Evo 2 1B on an RTX 5070 Ti this brings the loss of the upstream test
from 1.248 (the value in evo2 issue #157) back to 0.5027 (expected: 0.502; both computed as the upstream test
does), but the logits are not bit-identical to the RTX 4060's ([evo2.md](evo2.md)).

### Why not run the FP8 layers in bf16 instead?

Because the model's weights can depend on how its FP8 GEMMs round. Evo 2 1B computes its projections in FP8, and
replacing them with bf16 GEMMs (what evo2 does for its 7B models without Transformer Engine; the 1B refuses to load)
gives a loss of 1.365 on the upstream test prompts, worse than the native FP8 result of the RTX 5070 Ti.

### Is mmaemu a model of Blackwell's (or Hopper's) FP8 arithmetic?

No. v0.1 reproduces one reference platform, the RTX 4060 (Ada, sm_89) with cuBLASLt 12.8.3.14. It runs on newer
GPUs, but what it computes there is still the RTX 4060's result. sm_120 is not modelled: mmaemu ships no preset,
profile or split-K table for it. Khattak and Mikaitis report RTX PRO 6000 (sm_120) tensor-core numerics identical
to the B200's, and probes on an RTX 5070 Ti are consistent with that, but it is not validated at GEMM level. The
`hopper` and `b200_mma_sync` presets are experimental instruction-level fits to published test vectors, not platform
models ([limitations.md](limitations.md)).

### Which GPUs has it been run on?


| GPU | Role | Result |
|---|---|---|
| RTX 4060 (sm_89) | reference platform: the hardware recordings come from it | T2 passes: replayed on the CPU, mmaemu reproduces every output it models in 147 of the 173 recordings (the 15 tables and 11 bias, row-wise, fp16-accumulator or HMMA recordings have none), apart from the old emulator recording's 6,120 NaN-decode outputs; every kernel passes T3 and T6 there, and live `torch._scaled_mm` equals mmaemu's reference bit for bit (T3p) |
| RTX 5070 Ti (sm_120) | kernels v1 and v4 validated: T3 on the release kernels; full device independence (v1, v4, v9) and a development wheel installed without nvcc, 13 Oct | bitwise equal to the CPU reference |
| H100 PCIe (sm_90) | T3 on the release kernels; the 21 Evo 2 projections (fixed inputs) with v1 and v4 compiled by NVRTC | passed; not yet in the kernels' validated set |
| CPU (x86_64) | the float64 reference model | the oracle the kernels are compared with |

See [validation.md](validation.md) for the logs.

### How fast is it?

Much slower than native FP8. On the RTX 4060 the emulated Evo 2 1B projection runs at 2.94 TMAC/s with the
default kernel there (tc3), against 29.8 TMAC/s for native FP8. On an RTX 5070 Ti it runs at about 4 TMAC/s with
the default kernel (v4), and an Evo 2 1B forward pass over 8,192 bp takes about 1.0 s with mmaemu against 0.45 s
natively. The CPU reference model is far slower: the prototype's reached 116-181 million MAC/s (about 0.15 GMAC/s)
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

Use the profile `ada-rtx4060-te` (TE's 4 MiB cuBLAS workspace), quantise the input and the weight with TE's FP8
scales, and pass the *reciprocal* scales to `scaled_mm`, as TE does. The [Evo 2 notebook](https://github.com/astding/mmaemu/blob/master/examples/evo2_blackwell.ipynb)
shows a complete replacement of TE's FP8 projections.

### How do I know a call is within the validated scope?

`mmaemu.explain(...)` shows the preset's maturity, the split-K decision with its source and whether that kind of
decision was replayed on hardware (`verified`). Calls outside the validated scope warn
(`ExperimentalPresetWarning`, `UnverifiedSplitWarning`) or raise (`SplitKUnknownError`, `UnsupportedFeatureError`).

### Why does a shape raise `SplitKUnknownError`?

cuBLAS may split K for that shape on the reference GPU, and the recorded table does not contain it. mmaemu does not
guess: pass `splitk=1` if you know cuBLAS does not split it, `splitk=(S, "serial" | "pairwise")` if you know how it
does, or record it on an RTX 4060 with `tools/record_heuristics.py`.

### Can mmaemu be extended to another reference GPU?

Yes, with hardware evidence: instruction-level probes, cuBLAS GEMM recordings and a split-K table recorded on that
GPU. The [sm_120 bring-up runbook](runbooks/sm120_bringup.md) describes the procedure. A new preset or profile
starts as experimental.

### How does mmaemu differ from MMA-Sim, Hawkeye or FPRev?

Those projects model tensor-core instructions on many GPUs or measure accumulation orders. mmaemu reproduces whole
`torch._scaled_mm` FP8 calls of one reference platform, including the split-K plan cuBLASLt chooses and the
epilogue, and validates them against recordings of that GPU. The README's
[related-work section](https://github.com/astding/mmaemu/blob/master/README.md#related-work) describes each.

### Where does the tensor-core model come from?

From Khattak and Mikaitis, "Accurate Models of NVIDIA Tensor Cores" (ACM TACO, 2026), and their MATLAB Tensor Core
toolbox (BSD-2-Clause). mmaemu re-validates the Ada model against its own RTX 4060 recordings and adds the cuBLAS
layer (split-K, epilogue) on top. See the citation section of the README.
