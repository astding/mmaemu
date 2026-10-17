# mmaemu

[![PyPI](https://img.shields.io/pypi/v/mmaemu.svg)](https://pypi.org/project/mmaemu/)
[![Licence: Apache-2.0](https://img.shields.io/badge/licence-Apache--2.0-blue.svg)](https://github.com/astding/mmaemu/blob/master/LICENSE)

**Bit-exact NVIDIA FP8 tensor-core GEMMs in software.** `mmaemu.scaled_mm` takes the arguments of
`torch._scaled_mm` and returns the bits that `torch._scaled_mm` (torch 2.7.1) or Transformer Engine 2.3 return on a
reference GPU, an NVIDIA H100 PCIe (the default) or a GeForce RTX 4060, each with cuBLASLt 12.8.3.14. It computes
them on a CPU or on another CUDA GPU, and where a case lacks hardware evidence it warns or raises instead of
guessing, apart from three documented split-K exceptions ([how it works](#how-it-works)).

FP8 GEMMs on tensor cores do not round like IEEE arithmetic: the hardware aligns and truncates blocks of products
(Fasi et al. 2021; Khattak and Mikaitis 2026), cuBLAS may split K and add the partial sums in an order of its own,
and the `torch._scaled_mm` epilogue rounds twice. Two GPUs can therefore return different bits for the same FP8
GEMM. Often that does no harm: on the 21 FP8 projections of an Evo 2 1B window, the H100 PCIe's FP8 GEMMs return
different bits from the RTX 4060's on every projection, and Evo 2 gives its expected results on both GPUs.

**Example: Evo 2 on an RTX 5070 Ti.** [Evo 2](https://github.com/ArcInstitute/evo2), the Arc Institute's DNA
language model, computes its input projections as FP8 GEMMs through Transformer Engine. On an RTX 5070 Ti
(Blackwell, sm_120) the difference is of another size: 70% of the first projection's outputs differ from the
RTX 4060's, by up to 0.31, and Evo 2 1B scores a loss of 1.248 on its upstream test, where 0.502 is expected
([evo2 issue #157](https://github.com/ArcInstitute/evo2/issues/157)). With only its FP8 GEMMs computed by mmaemu as
Transformer Engine computes them on an H100 PCIe (`profile="hopper-te"`), the same GPU passes the test: 0.5022 for
the 1B and 0.2161 for the 40B, where 0.502 and 0.2159 are expected (losses in bf16, as the test computes them). The
RTX 4060 profile, `ada-te`, gives 0.5027 and 0.2161.

What is exact is the FP8 GEMMs. The rest of the model (bf16 matrix products, convolutions, attention) runs natively
on the RTX 5070 Ti, so Evo 2's outputs are close to the official ones, not identical: by the log-probabilities of
the four bases and the most likely base, about as far from an official run as the official runs on the H100 PCIe
and the RTX 4060 are from each other (on the 40B, from the H100 PCIe's run; on the 1B, from the RTX 4060's, since
the repository holds no per-position H100 output of the 1B). mmaemu does not model RTX 50-series GPUs; it runs on
them and computes the H100 PCIe's or the RTX 4060's FP8 GEMMs there.

> **Status: 0.1.0, alpha (first release).**
>
> - **H100 PCIe** (Hopper, sm_90, 114 SMs; profiles `hopper`, `hopper-te`, `hopper-ws0`). Calls recorded on rented
>   H100 PCIe GPUs and replayed through mmaemu elsewhere: on an RTX 4060 and an RTX 5070 Ti, the 1,423 live
>   `torch._scaled_mm` calls of a probe that cuBLASLt accepted return the H100's bits, given the split-K the H100
>   used, and the 27 it rejected raise, as on the H100. The 21 FP8 projections of an Evo 2 1B window, through torch
>   and Transformer Engine, also return the H100's bits on both GPUs (168 of 168 outputs). The rule for adding the
>   split partial sums was fitted to the probe's 1,049 split calls; the 2,416 split calls of a later sweep, run after
>   the rule was fixed, also replay bit for bit on the RTX 5070 Ti, at the split the H100 logged. mmaemu's own
>   kernels for these profiles (h4, h1) are checked on the RTX 4060 and the RTX 5070 Ti; they have not run on an H100.
> - **RTX 4060** (Ada, sm_89, 24 SMs; profiles `ada`, `ada-te`, `ada-ws0`). 173 hardware recordings (2.5 billion
>   values); the 147 with outputs that mmaemu models replay bit for bit, apart from 6,120 NaN-decode outputs of an
>   old emulator's recording, where mmaemu gives the canonical NaN. Live `torch._scaled_mm` calls on the card equal
>   mmaemu's.
> - **Not modelled:** the H100 SXM, the H200, RTX 50-series GPUs, other cuBLASLt versions. Inference only: bias,
>   row-wise scales and autograd raise.
> - Every number, with the log or recording behind it:
>   [validation](https://github.com/astding/mmaemu/blob/master/docs/validation.md). Before relying on a result:
>   [scope and limitations](#scope-and-limitations).

    pip install mmaemu
    python -m mmaemu info      # versions, device, matching profiles, and each backend's status with the reason

[Quickstart](#quickstart) · [Evo 2](#evo-2-on-rtx-50-series-gpus) ·
[Notebook](https://github.com/astding/mmaemu/blob/master/examples/evo2_blackwell.ipynb) ·
[Documentation](https://astding.github.io/mmaemu/) · [Citation](#citation)

![Upstream-test loss of Evo 2 1B and 40B, in fp32. 1B: official H100 PCIe 0.5026, official RTX 4060 0.5024, RTX 5070 Ti with mmaemu 0.5024 (hopper-te) and 0.5025 (ada-te), RTX 5070 Ti with bf16 projections 1.3642, with native FP8 1.2488; expected 0.502 (bf16). 40B: official H100 PCIe 0.2160, RTX 5070 Ti with mmaemu 0.2158 with either profile, bf16 1.4057, native FP8 1.4048; expected 0.2159 (bf16).](https://github.com/astding/mmaemu/blob/master/docs/images/evo2_upstream_test.png?raw=true)

*Loss in fp32 from the bf16 logits; the upstream test computes it in bf16, which gives the values quoted above. The
lines for the expected losses (0.502, 0.2159) and for issue #157 (1.248, 1.365) are such bf16 values. `hopper-te`
and `ada-te`: the FP8 GEMMs computed by mmaemu as Transformer Engine computes them on the H100 PCIe and on the
RTX 4060.*

- [Quickstart](#quickstart)
- [Evo 2 on RTX 50-series GPUs](#evo-2-on-rtx-50-series-gpus)
- [Is mmaemu for you?](#is-mmaemu-for-you)
- [Profiles and presets](#profiles-and-presets), [inside a model](#inside-a-model)
- [How it works](#how-it-works), [validation](#validation), [speed](#speed)
- [Scope and limitations](#scope-and-limitations), [related work](#related-work)
- [Documentation](#documentation), [contributing](#contributing-and-reporting-results), [citation](#citation),
  [licence](#licence)

## Quickstart

    pip install mmaemu

Python >= 3.10, PyTorch >= 2.6, NumPy. No CUDA toolkit, nvcc or C++ compiler is needed. On the CPU:

```python
import torch
import mmaemu

torch.manual_seed(0)
x = (torch.randn(64, 1920) * 40).clamp(-448, 448).to(torch.float8_e4m3fn)    # input  [M, K], row-major
w = (torch.randn(5760, 1920) * 40).clamp(-448, 448).to(torch.float8_e4m3fn)  # weight [N, K]
scale_x = torch.tensor(1 / 64)
scale_w = torch.tensor(1 / 32)

# The arguments of torch._scaled_mm: mat2 is [K, N] column-major, i.e. w.t(). On the CPU this runs the exact
# float64 reference model: seconds to tens of seconds for this shape.
y = mmaemu.scaled_mm(x, w.t(), scale_x, scale_w, out_dtype=torch.bfloat16, use_fast_accum=True,
                     profile="hopper")   # the default: torch._scaled_mm on an H100 PCIe
y_ada = mmaemu.scaled_mm(x, w.t(), scale_x, scale_w, out_dtype=torch.bfloat16, use_fast_accum=True,
                         profile="ada")  # the RTX 4060's bits: about 8% of these outputs differ from y

acc = mmaemu.accumulate(x, w, use_fast_accum=True)   # the fp32 accumulator before scaling, [M, N]
print(mmaemu.explain(x, w.t(), scale_x, scale_w, out_dtype=torch.bfloat16, use_fast_accum=True))
# CallPlan(validation='ok', profile='hopper', preset='hopper', preset_maturity='validated',
#          split=SplitDecision(S=1, scheme='none', source='table', verified=True, ...), backend='reference', ...)
```

On a GPU the same call runs the CUDA kernels and gives the same bits:

```python
yg = mmaemu.scaled_mm(x.cuda(), w.cuda().t(), scale_x.cuda(), scale_w.cuda(),
                      out_dtype=torch.bfloat16, use_fast_accum=True)
mmaemu.testing.assert_bitwise_equal(yg.cpu(), y)     # compares bit patterns: +0 != -0, NaN payloads
```

On an H100 PCIe, the reference GPU of the default profile `hopper`, with cuBLASLt 12.8.3.14 and torch 2.7.1, `y`
should equal the native `torch._scaled_mm(x.cuda(), w.cuda().t(), scale_x.cuda(), scale_w.cuda(),
out_dtype=torch.bfloat16, use_fast_accum=True)`: this shape is unsplit in the split-K table recorded on the H100
(`explain` shows `S=1`, `source='table'`), and all 374 unsplit live calls of the H100 probe replay bit for bit.
This snippet itself has not run on an H100. With `profile="ada"` on an RTX 4060, native `torch._scaled_mm` and
mmaemu are compared live on 278 calls of other shapes, all equal.

Good to know:

- Defaults follow torch: `use_fast_accum=False`, and `out_dtype` is the input dtype. Evo 2 uses
  `use_fast_accum=True` with bf16 output. `mmaemu.accumulate` has no default for `use_fast_accum`.
- `profile=` (or `MMAEMU_PROFILE` for the process) picks the reference platform: `hopper` (the default) and
  `hopper-te` for the H100 PCIe, `ada` and `ada-te` for the RTX 4060; the plain name for results produced by
  `torch._scaled_mm`, `-te` for results produced by Transformer Engine ([all profiles](#profiles-and-presets)).
- `splitk="auto"` takes cuBLAS's split-K decision (whether cuBLASLt splits K, and into how many parts S) from the
  table recorded on the profile's GPU. For a shape missing from the table, the profile's envelope, a bound on M*N
  set from the recorded data (on the H100 one bound per K, for K from 1,024 to 4,194,304 only), infers S = 1 for
  large enough outputs (`source='envelope'`). A shape that neither decides raises `SplitKUnknownError` rather than
  guess: on a CPU, under the default profile, M, N, K = 16, 256, 128 does (K is below that range). Run
  `mmaemu.explain` first, and pass `splitk=1` only when you know the reference GPU did not split that call, or the
  known split: `splitk=(S, "contiguous")` on the H100 profiles, `(S, "serial")` or `(S, "pairwise")` on the RTX 4060
  ones ([split-K tables](https://github.com/astding/mmaemu/blob/master/docs/splitk_tables.md)).
- `mmaemu.explain(...)` shows the profile, preset maturity, split-K decision and its source, the backend and any
  fallback, without running the GEMM; `python -m mmaemu explain M N K --out bf16 --fast` does the same from the
  shell. `python -m mmaemu selftest` runs the kernel self-tests on the GPU (exit 0: all required kernels passed).

More in the [user guide](https://github.com/astding/mmaemu/blob/master/docs/user_guide.md).

## Evo 2 on RTX 50-series GPUs

mmaemu makes an RTX 50-series GPU compute Evo 2's FP8 projections as Transformer Engine 2.3 computes them on an
H100 PCIe (`hopper-te`) or on an RTX 4060 (`ada-te`); the rest of the model runs natively. As of October 2026, the
[Evo 2 README](https://github.com/ArcInstitute/evo2) lists FP8 through Transformer Engine on a Hopper GPU as a
requirement for the 1B, 20B and 40B checkpoints; mmaemu's default reference platform is one Hopper GPU, the H100
PCIe. Measured on an RTX 5070 Ti (16 GB) with evo2 0.6.0 and torch 2.7.1+cu128, without Transformer Engine. Not
run: the RTX 5090 (issue #157 reports the same failure on it) and the other RTX 50-series GPUs, and Evo 2 7B and
20B.

The examples are not in the wheel; clone the repository:

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install evo2==0.6.0 vtx==1.1.0
pip install flash-attn==2.8.0.post2 --no-build-isolation      # no Transformer Engine is needed
git clone https://github.com/astding/mmaemu && cd mmaemu && pip install .
hf download arcinstitute/evo2_1b_base evo2_1b_base.pt --local-dir .
python examples/evo2_offload.py test --model evo2_1b_base --checkpoint evo2_1b_base.pt --profile hopper-te
hf download arcinstitute/evo2_40b --local-dir evo2_40b       # 82 GB in two parts; joining them needs 82 GB more
cat evo2_40b/evo2_40b.pt.part0 evo2_40b/evo2_40b.pt.part1 > evo2_40b.pt
python examples/evo2_offload.py test --model evo2_40b --checkpoint evo2_40b.pt --profile hopper-te
python examples/evo2_offload.py score --model evo2_40b --checkpoint evo2_40b.pt --input seqs.fasta --out scores.tsv
```

[examples/evo2_offload.py](https://github.com/astding/mmaemu/blob/master/examples/evo2_offload.py) runs the upstream
model (evo2 0.6.0 with vtx and flash-attn) with each FP8 projection computed by mmaemu: `--profile hopper-te` (the
default) or `--profile ada-te`. `test` runs the upstream test and prints PASS or FAIL; `score` scores sequences as
evo2's `score_sequences` does. Every run reported here fed one sequence at a time, of up to 8,192 bp, so no FP8
projection had more than 8,192 rows (M); up to that size every FP8 projection of the 1B and the 40B has a shape
recorded in both profiles' split-K tables (S = 1, `source='table'`). Larger M, from longer sequences or from several
sequences batched into one call (the projection flattens the batch into rows), was not run: there both profiles infer
S = 1 from their envelope (under `hopper-te` a few sizes up to 16,384 are table rows, also S = 1), and `hopper-te`'s
envelope inferences warn (`UnvalidatedEnvelopeWarning`), because the H100 envelope failed its held-out test at
Transformer Engine's 32 MiB workspace. The weights stay in host RAM and the model runs one block at a time on the
GPU. The 40B's weights take 82 GB of host RAM. On the RTX 5070 Ti, with a 180 GiB host, the 40B upstream test,
loading included, took 7.6 minutes and peaked at 123 GiB resident with `hopper-te` (17 October, after the profile
rename) and 5.6 minutes and 157 GiB with `ada-te` (15 October, before it). These are two separate runs, not a
comparison of the profiles; the resident sets count the pages of the memory-mapped checkpoint, and hosts with less RAM
were not tried. The script was checked
against the validation runs: with either profile it gives the 1B's logits and losses and the scores of 40 BRCA1
windows bit for bit, and with `hopper-te` the 40B's logits
([log](https://github.com/astding/mmaemu/blob/master/docs/validation_logs/2026-10-16/evo2-h100-profile-rtx-5070-ti/README.md#the-example-check-example_check)).
[examples/evo2_blackwell.ipynb](https://github.com/astding/mmaemu/blob/master/examples/evo2_blackwell.ipynb) runs the
1B step by step with `ada-te`, against the official RTX 4060 outputs shipped in `examples/data/`, so no second GPU is
needed. Guide and discussion: [docs/evo2.md](https://github.com/astding/mmaemu/blob/master/docs/evo2.md).

### Upstream test

The four prompts of `evo2/test/test_evo2.py`: next-token cross entropy, in bf16 as the test computes it and in fp32
from the bf16 logits.

| Platform | FP8 GEMMs computed by | 1B loss, bf16 | 1B loss, fp32 | 40B loss, bf16 | 40B loss, fp32 |
|---|---|---:|---:|---:|---:|
| expected by the upstream test | | 0.502 | | 0.2159 | |
| H100 PCIe (Hopper, sm_90) | official: Transformer Engine FP8 | 0.5022 | 0.5026 | 0.2159 | 0.2160 |
| RTX 4060 (Ada, sm_89) | official: Transformer Engine FP8 | – | 0.5024 | – | – |
| **RTX 5070 Ti (Blackwell, sm_120)** | **mmaemu, `hopper-te` (H100 PCIe)** | **0.5022** | **0.5024** | **0.2161** | **0.2158** |
| RTX 5070 Ti | mmaemu, `ada-te` (RTX 4060) | 0.5027 | 0.5025 | 0.2161 | 0.2158 |
| RTX 5070 Ti | native FP8 (`torch._scaled_mm`) | 1.2480 | 1.2488 | 1.4023 | 1.4048 |
| RTX 5070 Ti | bf16 GEMMs, no FP8 | 1.3652 | 1.3642 | 1.4043 | 1.4057 |

– : not run or not recorded. The bf16 column is the loss the test checks and the one issue #157 reports (1.248 and
1.365 there). With `hopper-te` the 1B's bf16 loss equals the H100's on every prompt, an effect of rounding the loss
to bf16: the fp32 losses differ (0.5024 against 0.5026), and no prompt's logits are bit-identical to the H100's. The
official 40B on the H100 ran the upstream blocks with Transformer Engine one at a time, as mmaemu's runs do; on the
1B that method gives `evo2.Evo2`'s logits bit for bit. Logs:
[1B and 40B, `hopper-te`](https://github.com/astding/mmaemu/tree/master/docs/validation_logs/2026-10-16/evo2-h100-profile-rtx-5070-ti)
(its `compare.json` also has the official and `ada-te` values these runs are compared with);
[1B, `ada-te`, native FP8 and bf16 GEMMs, bf16 losses](https://github.com/astding/mmaemu/tree/master/docs/validation_logs/2026-10-14/evo2-rtx-5070-ti);
[1B, official H100 and RTX 4060, fp32 losses](https://github.com/astding/mmaemu/tree/master/docs/validation_logs/2026-10-14/evo2-h100-sm90);
[40B, `ada-te`, native FP8 and bf16 GEMMs](https://github.com/astding/mmaemu/tree/master/docs/validation_logs/2026-10-15/evo2-40b-rtx-5070-ti)
(its `check_1b.json` has the 1B's fp32 losses in these modes);
[40B, official H100](https://github.com/astding/mmaemu/tree/master/docs/validation_logs/2026-10-15/evo2-40b-h100-sm90)
(it also has the 1B's bf16 losses on the H100).

### How close, and what is exact

- **One FP8 GEMM.** The model's first FP8 projection (block 0 on a 2,048-bp window: M 2048, N 5760, K 1920),
  against the SHA-256 of the RTX 4060's Transformer Engine output: mmaemu with `ada-te` returns it bit for bit on the
  RTX 5070 Ti; natively, 8,293,802 of the 11,796,480 bf16 outputs (70.3%) differ, by up to 0.31.
- **The H100's FP8 GEMMs.** All 21 FP8 projections of that window, captured on an H100 PCIe, replay bit for bit
  through mmaemu's `hopper` model (168 of 168 outputs, torch and Transformer Engine). No 40B projection was captured
  on an H100: for the 40B, whose projections are unsplit in the H100 table, the H100's bits are mmaemu's prediction
  from the validated model.
- **The whole model.** On six 1B windows (the four test prompts, and the first 8,192 and 2,048 bp of their
  concatenation), compared with the official RTX 4060 output (the repository holds no per-position H100 output of
  the 1B windows), the mean \|Δ log p(A,C,G,T)\| is 0.0407 with
  `hopper-te`, 0.0403 with `ada-te` and 0.0411 for the official H100 (native FP8: 2.58; bf16 GEMMs: 2.85), and the
  most likely base agrees at 99.40%, 99.42% and 99.41% of the positions. On the largest difference in a sequence
  score, mmaemu is about twice as far as the H100 (0.0019 with `hopper-te`, 0.0016 with `ada-te`, against 0.0009;
  native FP8: 1.12). On two 40B windows (2,048 and 4,096 bp), against the official H100: 0.0289 and 0.0283 with
  `hopper-te`, 0.0295 and 0.0288 with `ada-te`; the official RTX 4060 is 0.0300 and 0.0292 from the H100, native FP8
  3.97 and 4.07. The most likely base agrees with the H100's at 100.0% and 99.98% of the positions with `hopper-te`,
  100.0% and 99.93% with `ada-te`, and 99.90% on both windows for the official RTX 4060.
- **Not measured:** whether an FP8 GEMM of the same E4M3 operands accumulated in IEEE fp32, without the tensor
  cores' block alignment and truncation, would also restore Evo 2, and how far an FP8 GEMM may deviate from the
  reference GPUs' before Evo 2 fails. What is known: a particular GPU's exact bits are not required, since the H100
  PCIe's and the RTX 4060's FP8 GEMMs differ on every projection and both give the expected results.

### Zero-shot BRCA1 variant effect prediction

The task of the upstream [BRCA1 notebook](https://github.com/ArcInstitute/evo2/tree/main/notebooks/brca1): score
8,192-bp windows around the single-nucleotide variants measured by Findlay et al. (2018), and separate the
loss-of-function variants by the score change Δ. The same random 500 variants (111 loss of function) on every
platform for the 1B; their first 100 (20 loss of function) for the 40B.

| Platform | 1B AUROC | 1B Δ Pearson with the H100's | 40B AUROC | 40B Δ Pearson with the H100's |
|---|---:|---:|---:|---:|
| H100 PCIe, official TE FP8 | 0.733 | 1 | 0.855 | 1 |
| RTX 4060, official TE FP8 | 0.738 | 0.987 | – | – |
| **RTX 5070 Ti + mmaemu, `hopper-te`** | **0.735** | **0.988** | **0.857** | **0.9986** |
| RTX 5070 Ti + mmaemu, `ada-te` | 0.732 | 0.986 | 0.855 | 0.9981 |
| RTX 5070 Ti, native FP8 | 0.559 | 0.06 † | 0.503 | 0.17 |
| RTX 5070 Ti, bf16 GEMMs | 0.484 | 0.07 † | 0.590 | 0.19 |

† against the official RTX 4060; the 1B's native runs were not compared with the H100's.

The 95% bootstrap intervals of the AUROCs are about ±0.06 for the 1B and ±0.1 for the 40B (0.675-0.793 and
0.749-0.940 for `hopper-te`), so the official and mmaemu AUROCs cannot be told apart, and no ranking among them
should be read. The per-variant score changes carry the comparison: with `hopper-te` they follow the official H100's
with Pearson 0.988 on the 1B, as closely as the two official GPUs follow each other (0.987), and 0.9986 on the 40B.
Natively, with FP8 or bf16 GEMMs, the AUROC falls to or near chance (0.48 to 0.59) and the score changes are
essentially unrelated to the official ones. On all 500 variants the official 40B on the H100 gives 0.868; mmaemu's
40B ran on the first 100.

An alternative that needs no emulation: NVIDIA BioNeMo publishes Evo 2 checkpoints fine-tuned for GPUs other than
Hopper, among them a 1B for bf16, `evo2/1b-8k-bf16:1.0`
([BioNeMo Evo 2 documentation](https://docs.nvidia.com/bionemo-framework/latest/main/developer-guide/bionemo-evo2/bionemo-evo2-Overview/)).
They run at native speed, with other weights than the original checkpoints, which are the ones mmaemu runs.

## Is mmaemu for you?

Use it when you need the FP8 GEMM outputs of an H100 PCIe or an RTX 4060 (cuBLASLt 12.8.3.14) somewhere else:

- **An FP8 model on a GPU whose FP8 GEMMs differ.** Evo 2 is the model this has been shown on (above). Whether
  another FP8 model needs it depends on that model; mmaemu makes its FP8 GEMMs the reference GPU's, and the rest of
  the model still runs natively.
- **Golden FP8 outputs without the GPU:** the reference GPU's exact `torch._scaled_mm` outputs on a CPU, in CI or on
  a laptop, for regression tests or to check numbers computed on the reference platform.
- **Testing kernels and libraries:** compare a custom FP8 kernel, a new cuBLAS version or a new GPU with the
  reference bits; `mmaemu.testing.assert_bitwise_equal` locates each difference by class (zero sign, NaN payload,
  value). A difference on another platform is not by itself a bug.
- **Debugging cross-GPU discrepancies:** route a model's FP8 GEMMs through mmaemu on two machines; whatever still
  differs comes from the other operations. `mmaemu.explain` shows each call's split-K decision and backend.
- **Numerics research:** the instruction model (`mmaemu.model.dot`) and the presets expose block alignment and
  truncation; `scaled_mm` and `accumulate` expose fast and slow accumulation (`use_fast_accum`: one tensor-core
  chain, or a new chain every 128 elements of K with the chain results added in fp32) and split-K with the order of
  its partial sums (`splitk=`), and `scaled_mm` the epilogue's double rounding (`out_dtype`). `mmaemu.model.dot`
  with a custom `Preset` evaluates a hypothetical accumulator design (`scaled_mm` takes only the shipped presets,
  through its profiles).

Possible, but outside what 0.1.0 has been validated for: the same FP8 GEMM outputs on every cloud GPU type, by
pinning them to one reference platform (the rest of the model still runs natively and differs by GPU, as the Evo 2
results show); auditing FP8 experiments by recomputing their GEMMs; characterising a new
GPU's FP8 arithmetic with the probes of a bring-up runbook and adding it as a reference platform; teaching what
tensor cores compute.

It is not the tool when you need:

- your own GPU's native FP8 results predicted: mmaemu reproduces the reference GPU, not the GPU it runs on;
- another reference: the H100 SXM, the H200, B200 cuBLAS, an RTX 50-series GPU or another cuBLASLt version;
- training or other FP8 features: autograd, bias and row-wise scales raise; scales are tensor-wise only; CUDA graph
  capture is not supported;
- a whole model bit-identical: only the GEMMs routed through mmaemu are exact;
- native speed: on an RTX 4060 an emulated GEMM is about 10 (`ada`) to 22 (`hopper`) times slower than native FP8.

## Profiles and presets

A **profile** is a reference platform: the GPU and its SM count, the cuBLASLt version, the caller with its cuBLAS
workspace, the split-K table recorded on that GPU, and the calls its cuBLAS rejects. A **preset** is the arithmetic
of one tensor-core instruction. `hopper` and `ada` name both: profile `hopper` uses preset `hopper`, and profile
`ada` preset `ada`. `scaled_mm`, `accumulate` and `explain` take `profile=`; only `mmaemu.model.dot` takes
`preset=`.

| Profile | Reproduces | Reference GPU, with cuBLASLt 12.8.3.14 | cuBLAS workspace | Split-K |
|---|---|---|---|---|
| `hopper` (default) | `torch._scaled_mm`, torch 2.7.1 | H100 PCIe (Hopper, sm_90, 114 SMs) | 1 MiB | validated |
| `hopper-te` | Transformer Engine 2.3's FP8 GEMM | H100 PCIe | 32 MiB | splits (S > 1) experimental; the envelope's S = 1 unvalidated; both warn |
| `hopper-ws0` | cuBLAS with workspace 0 | H100 PCIe | 0 | never splits |
| `ada` | `torch._scaled_mm`, torch 2.7.1 | GeForce RTX 4060 (Ada, sm_89, 24 SMs) | 1 MiB | validated |
| `ada-te` | Transformer Engine 2.3's FP8 GEMM | RTX 4060 | 4 MiB | splits (S > 1) experimental; warn |
| `ada-ws0` | cuBLAS with workspace 0 | RTX 4060 | 0 | never splits |

Presets:

- `hopper`: blocks of 32 products, truncated to 13 fraction bits. Validated for cuBLAS FP8 GEMMs on the H100 PCIe
  (cuBLASLt 12.8.3.14), not for the sm_90 `mma.sync` instruction, the H100 SXM or the H200.
- `ada`: blocks of 16 products, the model of Khattak and Mikaitis ([citation](#citation)), re-validated on RTX 4060
  recordings.
- `b200_mma_sync` (alias `blackwell`) is **experimental**: it fits the toolbox's B200 `mma.sync` test vectors, not
  B200 cuBLAS, and not RTX 50-series GPUs. It warns when used, as do the profiles `hopper-experimental` (the
  `hopper` arithmetic without a split-K table) and `b200-mma-sync-experimental`.

Validation logs made before the profiles were renamed, on 16 October 2026, use the former names:
`hopper-h100pcie-torch` (now `hopper`), `hopper-h100pcie-te` (`hopper-te`), `hopper-h100pcie-ws0` (`hopper-ws0`),
`ada-rtx4060-torch` (`ada`), `ada-rtx4060-te` (`ada-te`) and `ada-rtx4060-ws0` (`ada-ws0`). 0.1.0 accepts only the
new names. More: [presets and profiles](https://github.com/astding/mmaemu/blob/master/docs/presets_profiles.md).

## Inside a model

mmaemu does not patch `torch._scaled_mm` or Transformer Engine; you replace the call. An FP8 linear layer without
bias:

```python
import torch
import mmaemu

class EmulatedFP8Linear(torch.nn.Module):
    """y = x @ W.T with E4M3 operands, as Transformer Engine 2.3 computes it on an H100 PCIe."""

    def __init__(self, weight, scale_x, scale_w, profile="hopper-te"):
        super().__init__()
        self.scale_x, self.scale_w, self.profile = scale_x, scale_w, profile   # quantisation scales (fp32 tensors)
        self.wq = (weight.detach().float() * scale_w).clamp(-448, 448).to(torch.float8_e4m3fn)

    def forward(self, x):
        xq = (x.float() * self.scale_x).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(-1, x.shape[-1])
        y = mmaemu.scaled_mm(xq, self.wq.t(), 1 / self.scale_x, 1 / self.scale_w,   # torch takes the inverse scales
                             out_dtype=torch.bfloat16, use_fast_accum=True, profile=self.profile)
        return y.reshape(*x.shape[:-1], -1)
```

Each of these changes bits: the FP8 operands must be the reference run's (the same scales, saturation to ±448 and
cast); padding can change M and with it the split-K decision; mmaemu is inference only (run the model under
`torch.inference_mode()`). The Evo 2 notebook patches the 1B's 21 FP8 projections this way, with the scales stored
in the checkpoint. More: [user guide](https://github.com/astding/mmaemu/blob/master/docs/user_guide.md).

## How it works

For each call, mmaemu reproduces the steps that decide the reference GPU's bits:

1. **Profile:** the reference platform. Calls its cuBLAS rejects raise `ReferenceRejectsError`.
2. **Split-K:** whether cuBLASLt splits K for this shape, and into how many parts (S), read from the table recorded
   on that GPU for the caller and workspace. `explain` shows the source (`table`, `workspace0`, `explicit`,
   `live-query` or `envelope`); if none decides, `SplitKUnknownError` is raised.
3. **Tensor-core chains** (the preset): products added in blocks of 32 (`hopper`) or 16 (`ada`), aligned to the
   block's largest exponent and truncated to 13 fraction bits.
4. **Combination of the splits** in fp32, in the order cuBLASLt uses: contiguous runs of 128-wide k-tiles added
   serially on the H100 PCIe; interleaved 64-wide k-tiles added serially or pairwise on the RTX 4060.
5. **Epilogue:** alpha = scale_a x scale_b in fp32, a fused multiply-add, rounding to the output type (via fp32 for
   bf16 and fp16, so twice), saturation for FP8 outputs, canonical NaN.

The same arithmetic runs as an exact float64 reference model (on the CPU or the GPU) and as CUDA kernels that use
only IEEE-defined instructions (h4 and h1 for the `hopper` preset, v4 and v1 for `ada`) or the tensor cores
themselves (v9 and tc3, `ada` only, auto-selected only on sm_89). The kernels must return the reference model's
bits: each passes a self-test on the device before it is used; the IEEE-exact kernels are checked bit for bit on
the RTX 4060 (sm_89) and the RTX 5070 Ti (sm_120), and v9 and tc3 are validated on sm_89 only. The package ships
prebuilt kernels (fatbins), loaded through the NVIDIA driver; if they cannot be used, mmaemu compiles the shipped
sources with NVRTC (`libnvrtc` comes with PyTorch's CUDA wheels), and otherwise falls back to the reference model on
the tensor's device; either fallback emits one `BackendFallbackWarning`.

**Evidence first.** A behaviour ships as *validated* only when outputs recorded on the reference GPU reproduce it bit
for bit; otherwise it is *experimental* (it warns) or it raises. Three documented exceptions, all in split-K:

- on the reference GPU itself, a live cuBLASLt query decides a shape missing from the table
  (`source='live-query'`);
- elsewhere, above the envelope of the setting (output type, accumulation, formats, workspace), S = 1 is inferred
  (`source='envelope'`): on the H100 a bound per K, fitted to a dense sweep and tested once on a sealed held-out
  set; on the RTX 4060 twice the largest M*N that splits in that setting's table. `MMAEMU_SPLITK_STRICT=1` makes
  such shapes raise;
- on the H100 profiles, a K above 262,144 adds no warning to an explicit `splitk=(S, "contiguous")`, which is
  computed by the contiguous rule although no live split call had such a K (on `hopper-te` every S > 1 warns
  anyway, as experimental).

The numerics contract: [docs/semantics.md](https://github.com/astding/mmaemu/blob/master/docs/semantics.md).

## Validation

| GPU | Role | What was checked | Result |
|---|---|---|---|
| H100 PCIe (sm_90, 114 SMs) | reference GPU of `hopper`, `hopper-te`, `hopper-ws0` | a probe of its cuBLAS FP8 GEMMs (1,450 live `torch._scaled_mm` calls; the 21 Evo 2 1B projections through torch and TE; the epilogue; the cuBLASLt heuristic) and a split-K sweep (2,744 live jobs; the heuristic on a dense grid), replayed through mmaemu (the probe on the RTX 4060 and the RTX 5070 Ti, the sweep on the RTX 5070 Ti) | the 1,423 calls cuBLASLt accepted and all 2,744 sweep jobs bit for bit, at the split the H100 logged; 168 of 168 Evo 2 outputs; the 27 rejected calls raise. Split-K decisions: from the recorded table; above it an envelope, validated for `hopper`, unvalidated for `hopper-te` |
| RTX 4060 (sm_89, 24 SMs) | reference GPU of `ada`, `ada-te`, `ada-ws0`; device | 173 hardware recordings (2.5 billion output values); live `torch._scaled_mm` calls; every CUDA kernel | the 147 recordings with outputs mmaemu models replay bit for bit, apart from 6,120 NaN-decode outputs of an old emulator's recording; live calls equal; every kernel equals the reference model |
| RTX 5070 Ti (sm_120, 70 SMs) | device, not modelled | the IEEE-exact kernels (h4, h1, v4, v1) against the reference model; the H100 replays; Evo 2 1B and 40B end to end | kernels bit for bit; replays equal; Evo 2 passes the upstream test with `hopper-te` and with `ada-te` |
| H100 PCIe | device | v4 and v1 (the `ada` preset's kernels) on the Evo 2 projections | the RTX 4060's bits on all 21; h4 and h1 have not run on an H100 |
| CPU (Linux x86_64) | device | the CPU test suite (T0 + T1) with torch 2.6.0, 2.7.1 and the latest, on the code of 0.1.0: on Python 3.10 to 3.13 in GitHub CI ([log](https://github.com/astding/mmaemu/blob/master/docs/validation_logs/2026-10-17/ci/README.md)) and on Python 3.12 locally ([log](https://github.com/astding/mmaemu/blob/master/docs/validation_logs/2026-10-17/ci-local/README.md)); mutation testing ([log](https://github.com/astding/mmaemu/tree/master/docs/validation_logs/2026-10-16/t4)) | pass |

How the evidence was produced:

- **Hardware is the oracle.** Outputs recorded on the reference GPU decide; mmaemu's own reference model is checked
  against them, and its CUDA kernels against the reference model.
- **Criteria fixed in advance.** Both H100 rentals had their pass criteria written down before the rental. The
  probe's rule for adding the split partial sums was found after its data came in and is labelled as fitted; the
  sweep's 2,416 split calls, run after it was fixed (S = 2 to 24, K up to 262,144) and replayed at the split the
  H100 logged, test it out of sample.
- **A sealed held-out test, reported when it failed.** The H100 envelope was fitted on part of the sweep and tested
  once on a sealed part: at torch's 1 MiB workspace, 0 of 154,191 held-out shapes above the bound split
  (validated); at Transformer Engine's 32 MiB, 1 of 77,518 did, so `hopper-te` warns
  (`UnvalidatedEnvelopeWarning`).
- **Edge cases, not only random data.** The RTX 4060 recordings include every one of the 2^32 fp32 values as the
  accumulator input, every pair of FP8 codes (65,536) through the epilogue, 240 constructed rounding ties, and NaN
  and Inf rows and columns.
- **Tests that can fail.** Of 45 deliberately broken copies of the model, the tests catch 43; the other 2 do not
  change the arithmetic. They also catch all 7 broken kernels, on both GPUs.
- **Determinism.** Every live call of the H100 probe gave the same bits twice, in separate processes (1,450 of
  1,450).

Every number, with the log or recording behind it, the runs per GPU and the release checks:
[docs/validation.md](https://github.com/astding/mmaemu/blob/master/docs/validation.md). `tools/check_docs.py` checks
that each reference resolves.

## Speed

Exact emulation is slower than native FP8, but a model slows down less, because only its FP8 GEMMs are emulated. On
the RTX 5070 Ti an Evo 2 1B forward pass over 8,192 bp takes about 1.0 s with mmaemu (`ada-te`) and 0.45 s
natively; scoring the 1B's 935 BRCA1 windows (500 with a variant, 435 distinct references) takes about 16 minutes
with `ada-te` and 7 natively, with the model held on the GPU as in the notebook; a `hopper-te` run of the validation
scripts, which keep the weights in host RAM and run the model block by block, took 15.5 minutes.

Per GEMM, at M = 2048 on the Evo 2 shapes (1 TMAC/s = 10^12 multiply-accumulates per second): on the RTX 4060, native
`torch._scaled_mm` runs at 29.8 to 30.5 TMAC/s, the `hopper` preset's kernel h4 at 1.35 to 1.41 (about 22 times
slower), and the `ada` preset's fastest kernel, tc3 (tensor cores, auto-selected only on sm_89), at 2.94 (about 10
times slower). On the RTX 5070 Ti, h4 runs at 4.1 to 4.6 TMAC/s and v4 at 4.0 to 4.4; native FP8 was not measured
there. Kernels, method and logs: [performance](https://github.com/astding/mmaemu/blob/master/docs/performance.md).

## Scope and limitations

In 0.1.0:

- **Reference platforms:** NVIDIA H100 PCIe (Hopper, sm_90, 114 SMs; the default) with cuBLAS/cuBLASLt 12.8.3.14,
  called through torch 2.7.1 `torch._scaled_mm` (1 MiB cuBLAS workspace) or Transformer Engine 2.3 (32 MiB); and
  NVIDIA GeForce RTX 4060 (Ada, sm_89, 24 SMs) with the same cuBLASLt and torch (1 MiB) or Transformer Engine 2.3
  (4 MiB). In this scope mmaemu's outputs equal the reference GPU's bit for bit on the recorded evidence; with
  Transformer Engine, split-K decisions with S > 1 are experimental on both GPUs, and the H100 envelope's S = 1
  inferences are unvalidated (both warn). These versions describe the measured reference platform, not your
  environment: a native `torch._scaled_mm` on your own GPU equals mmaemu only on the reference platform itself.
- **The H100 PCIe evidence** comes from two rented sessions, a probe and a split-K sweep; there is no standing live
  check against an H100. mmaemu's kernels for the `hopper` preset, h4 and h1, are checked on the RTX 4060 and the
  RTX 5070 Ti and have not run on an H100.
- **Split-K** is decided by the recorded tables. Off the reference GPU, shapes that neither the table nor the
  envelope decides raise `SplitKUnknownError`: with `splitk="auto"`, 70 of the H100 probe's 1,423 calls do and the
  other 1,353 agree with the H100's decision; of the split-K sweep's 2,744 live jobs, 2,264 raise (2,200 of its
  2,416 split jobs) and 480 agree (248 before the sweep's own table rows and envelope were shipped). None disagrees.
- **Formats:** FP8 inputs E4M3 x E4M3 or mixed E4M3/E5M2 (E5M2 x E5M2 is rejected, as by torch), tensor-wise
  scales; fp32, bf16, fp16 and E4M3 outputs, and E5M2 output from mixed inputs. E5M2 and mixed-format inputs run on
  the (slow) reference model.
- **Devices:** the CPU (exact float64 reference model in plain torch; CI on Linux x86_64) and NVIDIA GPUs through
  prebuilt CUDA kernels, tested on Linux x86_64 (WSL2 included). They load wherever `libcuda.so.1` is found (Linux
  aarch64 is untested); on native Windows and macOS the reference model is used. The IEEE-exact kernels (h4, h1,
  v4, v1) are validated on sm_89 and sm_120, the tensor-core kernels v9 and tc3 on sm_89 only. On other GPUs the
  IEEE-exact kernels run once they pass their self-test, without being validated there (v4 and v1 also passed the
  GPU tests on an H100); v9 and tc3 are auto-selected only on sm_89 (elsewhere they run when named, as in
  `backend="cuda:v9"`, or with `MMAEMU_ALLOW_GATED_KERNELS=1`, and only after their self-tests pass), and tc3 ships
  binaries for sm_80, sm_86 and sm_89 only.
- The `b200_mma_sync` preset (alias `blackwell`) is **experimental**: it fits the toolbox's B200 `mma.sync`
  vectors (an fp16 HMMA rewrite of the instruction), not B200 cuBLAS. It warns when used. Do not use it, or
  `b200-mma-sync-experimental`, to reproduce an RTX 50-series GPU or B200 cuBLAS.
- **sm_120 is not modelled** (RTX 50-series GPUs such as the RTX 5070 Ti and the RTX 5090): mmaemu ships no preset,
  profile or split-K table for it, and `b200_mma_sync` is not such a model. Khattak and Mikaitis (arXiv:2512.07004v4,
  section 4.1.8, Tables 3-4) report RTX PRO 6000 (sm_120) tensor-core numerics identical to the B200's; probes on
  an RTX 5070 Ti are consistent with that model
  ([log](https://github.com/astding/mmaemu/tree/master/docs/validation_logs/2026-10-13-sm120)), but it is not
  validated at GEMM level. mmaemu runs on these GPUs, emulating the H100 PCIe or the RTX 4060.
- **Inference only:** no bias, no row-wise scales, no autograd, no CUDA graph capture. Unsupported features raise
  instead of approximating.
- **Whole models:** mmaemu emulates the FP8 GEMMs routed through it. The rest of a model runs natively, so
  full-model outputs are close to the reference GPU's, not bit-identical (see the Evo 2 results above).

The full list: [docs/limitations.md](https://github.com/astding/mmaemu/blob/master/docs/limitations.md).

## Related work

mmaemu builds on the tensor-core models of Khattak and Mikaitis (see [Citation](#citation)). Other projects model
GPU floating-point arithmetic at other levels:

- [MMA-Sim](https://github.com/microsoft/MMA-Sim) (Xie et al., [arXiv:2511.10909](https://arxiv.org/abs/2511.10909)):
  bit-accurate Python models of the matrix instructions (`mma.sync`, `wgmma`, `tcgen05`, `v_mfma`) of ten GPU
  architectures, from NVIDIA Volta to RTX Blackwell and AMD CDNA1 to CDNA3, in formats from FP64 to FP8 and FP4.
- Hawkeye (Badash, Boneh, Komargodski and Srivastava, MLSys 2026,
  [arXiv:2603.20421](https://arxiv.org/abs/2603.20421), [code](https://github.com/badasherez/gpu-simulator)):
  re-executes tensor-core matrix multiplications on a CPU; the paper reports FP16, BF16 and FP8 on Ampere, Hopper
  and Lovelace GPUs.
- [FPRev](https://github.com/peichenxie/FPRev) (Xie et al., USENIX ATC 2025): reveals the accumulation order of
  floating-point implementations in software and hardware.
- [hardware_rounding_error_predictor](https://github.com/NaciCankaya/hardware_rounding_error_predictor) (Cankaya,
  [arXiv:2606.00279](https://arxiv.org/abs/2606.00279)): bit-exact re-computation of bf16 transformer inference on
  A100, L40S and H100 GPUs, including the cuBLAS kernel choice and its split-K variants.

mmaemu works at the level of the library call: the FP8 GEMM as `torch._scaled_mm` and Transformer Engine call
cuBLASLt, on two reference platforms. It reproduces, behind the `torch._scaled_mm` API and with CUDA kernels, the
split-K plan cuBLASLt picks, the order in which it adds the partial sums, the epilogue's scaling and double rounding,
and the calls it rejects, each checked against recordings made on the reference GPU, apart from the three split-K
exceptions of [how it works](#how-it-works). The split-K plan depends on the shape, output type, workspace and SM
count; mmaemu reads it from tables recorded on the reference GPU, and off that GPU it infers S = 1 above an envelope
and raises for shapes that neither decides. With Transformer Engine, splits (S > 1) are experimental and the H100
envelope's S = 1 inferences unvalidated; both warn. For instruction-level numerics of other architectures, including
RTX Blackwell, see MMA-Sim. Libraries that emulate FP8 by quantizing and dequantizing compute the product in higher
precision, so they do not return a tensor core's bits.

## Documentation

- [Documentation site](https://astding.github.io/mmaemu/) (the pages of `docs/`), or the
  [documentation index](https://github.com/astding/mmaemu/blob/master/docs/index.md) on GitHub
- [User guide](https://github.com/astding/mmaemu/blob/master/docs/user_guide.md),
  [API reference](https://github.com/astding/mmaemu/blob/master/docs/api.md),
  [Evo 2 guide](https://github.com/astding/mmaemu/blob/master/docs/evo2.md),
  [FAQ](https://github.com/astding/mmaemu/blob/master/docs/faq.md)
- [Semantics](https://github.com/astding/mmaemu/blob/master/docs/semantics.md): exactly what is computed.
- [Presets and profiles](https://github.com/astding/mmaemu/blob/master/docs/presets_profiles.md): maturity,
  split-K resolution, environment variables.
- [Validation](https://github.com/astding/mmaemu/blob/master/docs/validation.md),
  [limitations](https://github.com/astding/mmaemu/blob/master/docs/limitations.md),
  [performance](https://github.com/astding/mmaemu/blob/master/docs/performance.md),
  [testing](https://github.com/astding/mmaemu/blob/master/docs/testing.md),
  [releasing](https://github.com/astding/mmaemu/blob/master/docs/release.md),
  [changelog](https://github.com/astding/mmaemu/blob/master/CHANGELOG.md).

## Contributing and reporting results

Bug reports and questions: [GitHub issues](https://github.com/astding/mmaemu/issues). Results from GPUs that have
not been tested are welcome, whether they pass or fail: an RTX 5090 or another RTX 50-series GPU, an H100 (where
mmaemu's `hopper` kernels have never run), an H100 SXM. Please include the output of `python -m mmaemu info` and of
`python -m mmaemu selftest` or `python examples/evo2_offload.py test ...`. Adding a reference platform needs hardware
recordings; the [sm_120 bring-up runbook](https://github.com/astding/mmaemu/blob/master/docs/runbooks/sm120_bringup.md)
describes the probes.

## Citation

If you use mmaemu, please cite it ([CITATION.cff](https://github.com/astding/mmaemu/blob/master/CITATION.cff)):

```bibtex
@software{mmaemu,
  author  = {Ding, Xun},
  title   = {mmaemu: bit-accurate emulation of {NVIDIA} tensor-core {FP8} {GEMM} numerics},
  version = {0.1.0},
  year    = {2026},
  url     = {https://github.com/astding/mmaemu}
}
```

and the tensor-core model and toolbox it builds on:

```bibtex
@article{khattak2026accurate,
  title   = {Accurate Models of {NVIDIA} Tensor Cores},
  author  = {Khattak, Faizan A. and Mikaitis, Mantas},
  journal = {ACM Transactions on Architecture and Code Optimization},
  volume  = {23},
  number  = {3},
  pages   = {1--25},
  year    = {2026},
  doi     = {10.1145/3830409},
  note    = {arXiv:2512.07004}
}

@software{matlab_tensor_core,
  title   = {{MATLAB} Tensor Core toolbox},
  author  = {Khattak, Faizan A. and Mikaitis, Mantas},
  year    = {2025},
  url     = {https://github.com/north-numerical-computing/MATLAB-tensor-core}
}
```

If you use the Evo 2 results or the notebook, please also cite Evo 2 and, for the BRCA1 experiment, the saturation
genome editing data:

```bibtex
@article{Brixi2026,
  title   = {Genome modelling and design across all domains of life with {Evo} 2},
  author  = {Brixi, Garyk and Durrant, Matthew G. and Ku, Jerome and Naghipourfar, Mohsen and Poli, Michael and
             Sun, Gwanggyu and Brockman, Greg and Chang, Daniel and Fanton, Alison and Gonzalez, Gabriel A. and
             King, Samuel H. and Li, David B. and Merchant, Aditi T. and Nguyen, Eric and Ricci-Tam, Chiara and
             Romero, David W. and Schmok, Jonathan C. and Taghibakhshi, Ali and Vorontsov, Anton and Yang, Brandon
             and Deng, Myra and Gorton, Liv and Nguyen, Nam and Wang, Nicholas K. and Pearce, Michael T. and
             Simon, Elana and Adams, Etowah and Amador, Zachary J. and Ashley, Euan A. and Baccus, Stephen A. and
             Dai, Haoyu and Dillmann, Steven and Ermon, Stefano and Guo, Daniel and Herschl, Michael H. and
             Ilango, Rajesh and Janik, Ken and Lu, Amy X. and Mehta, Reshma and Mofrad, Mohammad R. K. and
             Ng, Madelena Y. and Pannu, Jaspreet and R{\'e}, Christopher and St. John, John and Sullivan, Jeremy
             and Tey, Joseph and Viggiano, Ben and Zhu, Kevin and Zynda, Greg and Balsam, Daniel and Collison,
             Patrick and Costa, Anthony B. and Hernandez-Boussard, Tina and Ho, Eric and Liu, Ming-Yu and
             McGrath, Thomas and Powell, Kimberly and Pinglay, Sudarshan and Burke, Dave P. and Goodarzi, Hani
             and Hsu, Patrick D. and Hie, Brian L.},
  journal = {Nature},
  volume  = {652},
  number  = {8112},
  pages   = {1349--1361},
  year    = {2026},
  doi     = {10.1038/s41586-026-10176-5}
}

@article{findlay2018accurate,
  title   = {Accurate classification of {BRCA1} variants with saturation genome editing},
  author  = {Findlay, Gregory M. and Daza, Riza M. and Martin, Beth and Zhang, Melissa D. and Leith, Anh P. and
             Gasperini, Molly and Janizek, Joseph D. and Huang, Xingfan and Starita, Lea M. and Shendure, Jay},
  journal = {Nature},
  volume  = {562},
  number  = {7726},
  pages   = {217--222},
  year    = {2018},
  doi     = {10.1038/s41586-018-0461-z}
}
```

Related reading: Fasi, Higham, Mikaitis and Pranesh, "Numerical behavior of NVIDIA tensor cores", PeerJ Computer
Science 7:e330 (2021), [doi:10.7717/peerj-cs.330](https://doi.org/10.7717/peerj-cs.330); Micikevicius et al.,
"FP8 Formats for Deep Learning", [arXiv:2209.05433](https://arxiv.org/abs/2209.05433) (2022).

## Acknowledgements

The Ada instruction model, the kernels' arithmetic and several test vectors derive from the MATLAB Tensor Core
toolbox of Faizan A. Khattak and Mantas Mikaitis. The Evo 2 experiments use the model and data of the Arc
Institute's [evo2](https://github.com/ArcInstitute/evo2) repository, the [vortex](https://github.com/Zymrael/vortex)
inference library and NVIDIA [Transformer Engine](https://github.com/NVIDIA/TransformerEngine).

## Licence

mmaemu is licensed under the Apache License 2.0 ([LICENSE](https://github.com/astding/mmaemu/blob/master/LICENSE)).
Its instruction model, kernels and test vectors are derived from the MATLAB Tensor Core toolbox
(https://github.com/north-numerical-computing/MATLAB-tensor-core) by Faizan A. Khattak and Mantas Mikaitis, under
the BSD 2-Clause License; see [NOTICE](https://github.com/astding/mmaemu/blob/master/NOTICE) and
[LICENSES/](https://github.com/astding/mmaemu/tree/master/LICENSES).

## Affiliation

mmaemu is an independent project by Xun Ding, developed on personal time with personal equipment and resources. It is
not affiliated with, funded by or endorsed by the author's employer. It is also not affiliated with NVIDIA or the Arc
Institute; NVIDIA, GeForce, RTX, CUDA and other NVIDIA product names are trademarks of NVIDIA Corporation, and
Evo 2 is developed by the Arc Institute and its collaborators.

Contact: Xun Ding, dxun1@alumni.jh.edu.
