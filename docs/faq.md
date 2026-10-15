# FAQ

**Does mmaemu make my GPU compute like an RTX 4060?**
Only for the FP8 GEMMs you route through `mmaemu.scaled_mm` (or `accumulate`). Each of those returns the RTX 4060's
bits. Everything else in your program (bf16/fp16 matrix products, convolutions, reductions, attention kernels) runs
natively, with your GPU's own rounding. For Evo 2 1B on an RTX 5070 Ti this brings the loss of the upstream test
from 1.248 (the value in evo2 issue #157) back to 0.503 (expected: 0.502), but the logits are not bit-identical
to the RTX 4060's ([evo2.md](evo2.md)).

**Why not run the FP8 layers in bf16 instead?**
Because the model was trained with FP8 GEMMs, and its weights depend on how they round. For Evo 2 1B, replacing the
FP8 projections with bf16 GEMMs (what evo2 does when Transformer Engine is missing) gives a loss of 1.365 on the
upstream test prompts, worse than the native FP8 result of the RTX 5070 Ti.

**Is mmaemu a model of Blackwell's (or Hopper's) FP8 arithmetic?**
No. v0.1 reproduces one reference platform, the RTX 4060 (Ada, sm_89) with cuBLASLt 12.8.3.14. It runs on newer
GPUs, but what it computes there is still the RTX 4060's result. sm_120 is not modelled: there is no hardware
evidence for its FP8 arithmetic yet. The `hopper` and `b200_mma_sync` presets are experimental instruction-level
fits to published test vectors, not platform models ([limitations.md](limitations.md)).

**Which GPUs has it been run on?**

| GPU | Role | Result |
|---|---|---|
| RTX 4060 (sm_89) | reference platform: the hardware recordings come from it | all 173 recordings reproduced (T2) |
| RTX 5070 Ti (sm_120) | kernels v1 and v4 validated (T3, device independence, wheel without nvcc) | bitwise equal to the CPU reference |
| H100 PCIe (sm_90) | T3 on the release kernels; Evo 2 fixed inputs | passed; not yet in the kernels' validated set |
| CPU (x86_64) | the float64 reference model | the oracle the kernels are compared with |

See [validation.md](validation.md) for the logs.

**How fast is it?**
Much slower than native FP8. On an RTX 5070 Ti the emulated Evo 2 1B projection runs at about 4 TMAC/s with the
default kernel (v4), and an Evo 2 1B forward pass over 8,192 bp takes about 1.0 s with mmaemu against 0.45 s
natively. The CPU reference model reaches about 0.1-0.2 TMAC/s with 8 threads: fine for tests and single layers,
too slow for whole models. See [performance.md](performance.md).

**Can I use it without a GPU?**
Yes. Every function runs on CPU tensors with the exact float64 reference model, and gives the same bits as the
CUDA kernels.

**Does it support training, bias, row-wise scales?**
Not in v0.1. mmaemu is inference-only: it raises for inputs that require grad under grad mode, for `bias` and for
row-wise scales, rather than approximate them.

**Does it work with `torch.compile`?**
The entry points are excluded from tracing (`torch.compiler.disable`): a compiled model calls them as ordinary
Python functions, with a graph break. This is deliberate: some Inductor versions double-round FP8 conversions.

**How do I use it with Transformer Engine?**
Use the profile `ada-rtx4060-te` (TE's 4 MiB cuBLAS workspace), quantise the input and the weight with TE's FP8
scales, and pass the *reciprocal* scales to `scaled_mm`, as TE does. The [Evo 2 notebook](https://github.com/astding/mmaemu/blob/master/examples/evo2_blackwell.ipynb)
shows a complete replacement of TE's FP8 projections.

**How do I know a call is within the validated scope?**
`mmaemu.explain(...)` shows the preset's maturity, the split-K decision with its source and whether that kind of
decision was replayed on hardware (`verified`). Calls outside the validated scope warn
(`ExperimentalPresetWarning`, `UnverifiedSplitWarning`) or raise (`SplitKUnknownError`, `UnsupportedFeatureError`).

**Why does a shape raise `SplitKUnknownError`?**
cuBLAS may split K for that shape on the reference GPU, and the recorded table does not contain it. mmaemu does not
guess: pass `splitk=1` if you know cuBLAS does not split it, `splitk=(S, "serial" | "pairwise")` if you know how it
does, or record it on an RTX 4060 with `tools/record_heuristics.py`.

**Can mmaemu be extended to another reference GPU?**
Yes, with hardware evidence: instruction-level probes, cuBLAS GEMM recordings and a split-K table recorded on that
GPU. The [sm_120 bring-up runbook](runbooks/sm120_bringup.md) describes the procedure. A new preset or profile
starts as experimental.

**Where does the tensor-core model come from?**
From Khattak and Mikaitis, "Accurate Models of NVIDIA Tensor Cores" (ACM TACO, 2026), and their MATLAB Tensor Core
toolbox (BSD-2-Clause). mmaemu re-validates the Ada model against its own RTX 4060 recordings and adds the cuBLAS
layer (split-K, epilogue) on top. See the citation section of the README.
