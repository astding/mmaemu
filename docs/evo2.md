# Evo 2 on GPUs other than the reference

This page explains why Evo 2 gives wrong results on some GPUs, how mmaemu restores its FP8 GEMMs, and what was
measured. The runnable version is
[examples/evo2_blackwell.ipynb](https://github.com/astding/mmaemu/blob/master/examples/evo2_blackwell.ipynb).

## The problem

[Evo 2](https://github.com/ArcInstitute/evo2) 1B (`evo2_1b_base`) has 25 blocks. In 21 of them (the Hyena blocks
0-2, 4-9, 11-16 and 18-23) the input projection is an FP8 GEMM computed by Transformer Engine (TE): the
activations and the weight are cast to E4M3 with per-tensor scales, multiplied on the tensor cores with fast
accumulation, and returned in bf16. The four attention blocks (3, 10, 17, 24) have no FP8 GEMM. The model's
outputs depend on how the GPU computes those FP8 GEMMs:

- On an RTX 5070 Ti (Blackwell, sm_120) with TE 2.5.0, upstream issue
  [#157](https://github.com/ArcInstitute/evo2/issues/157) reports a loss of 1.248 on the upstream test where 0.502
  is expected, and 1.365 with FP8 disabled.
- evo2 0.6.0 refuses to load the 1B model when TE is missing (only the 7B model has a bf16 fallback), and the
  upstream README lists FP8 through TE on a Hopper GPU as a requirement for the 1B, 20B and 40B checkpoints.

Replacing the FP8 GEMM by a bf16 GEMM with the original weights does not help: the loss is then 1.365, worse than
the RTX 5070 Ti's native FP8.

## What mmaemu changes

Each FP8 projection is replaced by a module that reproduces TE's computation on an RTX 4060:

1. The FP8 scales are read from the checkpoint: TE stores them in `blocks.N.projections._extra_state`
   (`scale_fwd[:2]`: input, weight).
2. Input and weight are cast to E4M3 as TE does: multiply by the scale, saturate to +-448, cast.
3. The GEMM is `mmaemu.scaled_mm(xq, wq.t(), 1 / scale_x, 1 / scale_w, out_dtype=torch.bfloat16,
   use_fast_accum=True, profile="ada-rtx4060-te")`. The profile `ada-rtx4060-te` is the RTX 4060 with TE's 4 MiB
   cuBLAS workspace; every Evo 2 projection shape is unsplit on it.
4. vortex pads the sequence length to a multiple of 16 on its TE path; the module does the same and drops the
   padding rows afterwards.

The model itself is the upstream one: `vortex.model.model.StripedHyena` built from evo2's own configuration with
`use_fp8_input_projections=False` (plain bf16 linear layers, which the patch then replaces), and loaded with
vortex's checkpoint loader. The whole patch is one notebook cell, tagged `evo2-fp8-patch`; it needs no TE.

The other operations of the model (bf16 matrix products, the Hyena convolutions, normalisations, attention) still
run natively, so the full model is not bit-identical to the RTX 4060. The results below measure how close it is,
against the distance between two GPUs that both run the official path.

## Running the notebook

Requirements: an NVIDIA GPU with about 12 GB of free memory (the BRCA1 windows are 8,192 bp), Python 3.12,
torch 2.7.1 (CUDA 12.8), evo2 0.6.0, vtx 1.1.0, flash-attn 2.8.0.post2, mmaemu, and matplotlib, pandas,
scikit-learn, openpyxl, jupyter. Transformer Engine is not needed. Run it from `examples/`:

```bash
cd examples
jupyter nbconvert --to notebook --execute --inplace evo2_blackwell.ipynb   # or open it in Jupyter
```

| Variable | Default | Meaning |
|---|---|---|
| `EVO2_1B_CHECKPOINT` | downloaded from Hugging Face (`arcinstitute/evo2_1b_base`, 2.7 GB) | a local `evo2_1b_base.pt` |
| `EVO2_BLACKWELL_OUT` | `evo2_blackwell_outputs/` | downloads, logits, scores and `results.json` |
| `EVO2_BLACKWELL_VARIANTS` | 500 | BRCA1 variants to score (a prefix of the 500-variant subset) |

On the RTX 5070 Ti the whole notebook takes about half an hour, almost all of it BRCA1 scoring (about 15 minutes
with mmaemu, 7 natively and 7 with bf16 projections).

### The reference data

`examples/data/evo2_1b_ada_rtx4060_te.npz` holds the outputs of the official path on the RTX 4060 (evo2 0.6.0,
TE 2.3.0, driver 572.16), compact enough to ship (0.7 MB):

| Key | Content |
|---|---|
| `<w>/ids` | the token ids of window `<w>` (p0-p3, w8k, w2k) |
| `<w>/acgt` | the fp32 log-probabilities of A, C, G, T at every position (softmax over the 512-token vocabulary) |
| `<w>/next` | the fp32 log-probability of the true next token at every position |
| `brca1/rs`, `brca1/vs` | the scores of the 435 reference windows and 500 variant windows |
| `brca1/pos`, `brca1/ref`, `brca1/alt`, `brca1/lof` | the variants of the subset and their class |

`examples/data/evo2_1b_ada_rtx4060_te.json` describes the arrays and the run, and records the SHA-256 of the input,
weight and TE output of the first projection on the 2,048-bp window (from the hardware recording
`evo2.evo2_1b_w2k`) and the H100 baseline numbers. The scripts that captured these outputs, their logs and a check that the shipped
arrays equal the captures are in the [RTX 4060 reference log](validation_logs/2026-10-12/evo2-rtx-4060-official/README.md).

## Results on the RTX 5070 Ti

Measured on an RTX 5070 Ti (70 SMs, 16 GB, WSL2), torch 2.7.1+cu128, cuBLASLt 12.8.3.14, evo2 0.6.0, vtx 1.1.0,
mmaemu 0.1.0.dev0 with kernel v4. Logs:
[evo2-rtx-5070-ti](https://github.com/astding/mmaemu/tree/master/docs/validation_logs/2026-10-14/evo2-rtx-5070-ti)
(this run) and
[evo2-h100-sm90](https://github.com/astding/mmaemu/tree/master/docs/validation_logs/2026-10-14/evo2-h100-sm90)
(the official path on an H100 PCIe).

### One GEMM

Block 0's projection on the 2,048-bp window (M 2048, N 5760, K 1920): with mmaemu its bf16 output has the SHA-256
of the RTX 4060's TE output; natively, 8,293,802 of the 11,796,480 outputs (70.3%) differ, by up to 0.31. The
input and the weight are identical in both modes, so the difference is the GEMM alone. The same rows computed by
mmaemu on the CPU equal the GPU result bit for bit.

### Six windows

**Mean \|Δ log p(A,C,G,T)\| against the RTX 4060**

| Window | bp | H100 official | 5070 Ti + mmaemu | 5070 Ti native FP8 | 5070 Ti bf16 |
|---|---:|---:|---:|---:|---:|
| p0 | 6,538 | 0.0375 | 0.0377 | 2.3062 | 3.1632 |
| p1 | 7,056 | 0.0391 | 0.0387 | 1.9133 | 1.8873 |
| p2 | 6,160 | 0.0399 | 0.0391 | 2.6397 | 2.7073 |
| p3 | 7,616 | 0.0464 | 0.0447 | 3.4576 | 3.4794 |
| w8k | 8,192 | 0.0351 | 0.0352 | 2.0656 | 2.7402 |
| w2k | 2,048 | 0.0485 | 0.0466 | 3.0922 | 3.1524 |

**Most likely base agrees with the RTX 4060 (fraction of positions)**

| Window | bp | H100 official | 5070 Ti + mmaemu | 5070 Ti native FP8 | 5070 Ti bf16 |
|---|---:|---:|---:|---:|---:|
| p0 | 6,538 | 0.9982 | 0.9974 | 0.6418 | 0.4079 |
| p1 | 7,056 | 0.9894 | 0.9899 | 0.2954 | 0.3624 |
| p2 | 6,160 | 0.9904 | 0.9912 | 0.3680 | 0.3519 |
| p3 | 7,616 | 0.9947 | 0.9955 | 0.3136 | 0.3363 |
| w8k | 8,192 | 0.9957 | 0.9957 | 0.5743 | 0.4087 |
| w2k | 2,048 | 0.9961 | 0.9956 | 0.3877 | 0.3516 |

**Mean next-token log-likelihood (the sequence score)**

| Window | bp | RTX 4060 official | H100 official | 5070 Ti + mmaemu | 5070 Ti native FP8 | 5070 Ti bf16 |
|---|---:|---:|---:|---:|---:|---:|
| p0 | 6,538 | -0.2253 | -0.2253 | -0.2253 | -0.9051 | -1.3307 |
| p1 | 7,056 | -0.7801 | -0.7806 | -0.7803 | -1.3935 | -1.3719 |
| p2 | 6,160 | -0.5600 | -0.5603 | -0.5601 | -1.3312 | -1.3721 |
| p3 | 7,616 | -0.4443 | -0.4441 | -0.4443 | -1.3656 | -1.3820 |
| w8k | 8,192 | -0.3840 | -0.3842 | -0.3838 | -1.0072 | -1.3387 |
| w2k | 2,048 | -0.2010 | -0.2019 | -0.2027 | -1.3167 | -1.3463 |

Log-probabilities are taken from the softmax over the full 512-token vocabulary. The sequence score is what
`evo2.score_sequences` returns. The logits of all three RTX 5070 Ti modes were bit-identical between two runs of
the notebook.

**The upstream test** (`evo2/test/test_evo2.py`, prompts p0-p3), computed as the test computes it: cross entropy
in bf16 on the GPU, argmax accuracy. The RTX 4060 row is the fp32 value from the shipped log-probabilities.

| Run | Loss | Accuracy |
|---|---:|---:|
| expected by the upstream test | 0.502 | 0.7956 |
| RTX 4060, official TE FP8 | 0.5024 | |
| RTX 5070 Ti + mmaemu | 0.5027 | 0.7960 |
| RTX 5070 Ti, native FP8 | 1.2480 | 0.3848 |
| RTX 5070 Ti, bf16 GEMMs | 1.3652 | 0.3262 |
| issue #157: RTX 5070 Ti, TE 2.5.0 FP8 | 1.248 | |
| issue #157: RTX 5070 Ti, FP8 disabled | 1.365 | |

Computed in fp32 instead (device-independent; the H100 log's `upstream_test.py`), the losses are 0.5024 (RTX 4060),
0.5026 (H100), 0.5025 (mmaemu), 1.2488 (native) and 1.3642 (bf16).

### BRCA1

500 variants (111 loss of function), 435 reference windows; Δ is the variant's score minus its reference window's.

| Platform | AUROC | Δ Pearson vs RTX 4060 | Δ Spearman vs RTX 4060 | Δ same sign as RTX 4060 | Max \|Δ sequence score\| | Minutes |
|---|---:|---:|---:|---:|---:|---:|
| RTX 4060, official TE FP8 | 0.7378 | 1 | 1 | 100% | 0 | |
| H100, official TE FP8 | 0.7332 | 0.987 | 0.922 | 84.2% | 0.0004 | 2.5 |
| RTX 5070 Ti + mmaemu | 0.7323 | 0.986 | 0.912 | 84.0% | 0.0004 | 15.9 |
| RTX 5070 Ti, native FP8 | 0.5586 | 0.059 | 0.165 | 59.0% | 0.18 | 7.0 |
| RTX 5070 Ti, bf16 GEMMs | 0.4842 | 0.069 | 0.078 | 56.4% | 0.62 | 7.2 |

Minutes: one pass over the 935 windows with upstream `score_sequences` (host timer; on the WSL2 host of the RTX 5070 Ti
`time.perf_counter` runs 6-10% fast, see [performance.md](performance.md); the H100 row is from the rented H100
host).

With mmaemu, the RTX 5070 Ti reproduces the task nearly as well as a second official GPU does: AUROC 0.732 against
0.733 on the H100 (0.738 on the RTX 4060), and nearly the same agreement of the per-variant score changes with the RTX
4060 (Pearson 0.986 against 0.987). Natively, its FP8 GEMMs make the model nearly useless for this task (AUROC 0.559),
and bf16 GEMMs make it worse than chance (0.484); the per-variant score changes no longer track the official ones.

## How to read these results

- **The FP8 GEMMs are reproduced exactly.** Each emulated projection returns the RTX 4060's bits; nothing else in
  the model is changed.
- **The model is reproduced about as closely as the official implementation reproduces itself.** The official path on
  an H100 differs from the RTX 4060 on more than half of the bf16 logits; with mmaemu the RTX 5070 Ti is at the
  same distance (mean \|Δ log p\| 0.040 against 0.041, argmax agreement 99.4% for both), while its native FP8 and
  bf16 GEMMs are about 65 and 70 times further away.
- **Task metrics are a weaker test than per-variant scores.** The score changes of single-nucleotide variants are tiny (mean \|Δ\| 6e-4 on the
  RTX 4060) and comparable to the differences between two GPUs (up to 4.5e-4 between the official RTX 4060 and H100
  runs), so even the two official runs agree on them only to Pearson 0.987 and Spearman 0.922, and their AUROCs
  differ by 0.005. mmaemu lands just outside that band on most BRCA1 measures (Pearson 0.986, Spearman 0.912, same
  sign 84.0%, AUROC 0.0055 below the RTX 4060's, largest score-change difference 4.8e-4); native FP8 and bf16 GEMMs
  fall far outside it.
- **Not covered:** the 7B and 20B models were not run (the 7B's projection shape, (K, N) = (4096, 12288), never
  splits on the reference GPU with the bf16 output Evo 2 uses, so `splitk="auto"` resolves it); the 40B ran outside
  the notebook ([below](#evo-2-40b)); GPUs other than the RTX 5070 Ti were not run with the notebook; training and
  fine-tuning are out of scope (mmaemu is inference-only).

## Evo 2 40B

The [BioNeMo Evo 2 documentation](https://docs.nvidia.com/bionemo-framework/latest/main/developer-guide/bionemo-evo2/bionemo-evo2-Overview/)
describes the original 40B checkpoint (`evo2_40b`, 82 GB) as sensitive to Hopper's FP8 arithmetic, reports a much
higher loss for it with FP8 on a Blackwell GPU than on a Hopper GPU, and publishes a fine-tuned variant, with other
weights, for other GPUs. Here the original checkpoint ran on the RTX 5070 Ti (16 GB)
with its 42 FP8 projections computed by mmaemu in the same way as the 1B's (profile `ada-rtx4060-te`; every
projection shape, K 8192 and N 24576, is unsplit in the RTX 4060's split-K table). The weights stay in host RAM
(82 GB resident) and the upstream model runs one block at a time on the GPU (peak 11.2 GiB). The scripts, checked
on the 1B against the model held on the GPU (logits and BRCA1 scores bit for bit), and all results are in the
[validation log](validation_logs/2026-10-15/evo2-40b-rtx-5070-ti/README.md).

**Upstream test** (prompts p0-p3, mean):

| Run | Loss (bf16, as the test) | Loss (fp32) | Accuracy |
|---|---:|---:|---:|
| expected by the upstream test | 0.2159 | | 0.9167 |
| H100, official TE FP8 ([log](validation_logs/2026-10-15/evo2-40b-h100-sm90/README.md)) | 0.2159 | 0.2160 | 0.9167 |
| RTX 5070 Ti + mmaemu | 0.2161 | 0.2158 | 0.9169 |
| RTX 5070 Ti, native FP8 | 1.4023 | 1.4048 | 0.2568 |
| RTX 5070 Ti, bf16 GEMMs | 1.4043 | 1.4057 | 0.2559 |

The upstream test passes within 1e-3 of the expected loss, which mmaemu's run meets (1.2e-4). The official path
on an H100, run block by block in the same way, gives the expected loss to all seven of the test's digits
(0.2159424). Natively, with FP8 or bf16 GEMMs, the model predicts the next base about as often as guessing would.

**Against the official path on an RTX 4060** (the first 2,048 and 4,096 bp of the prompts' concatenation; the
upstream blocks with TE FP8 run one at a time on the RTX 4060, which on the 1B gives the real upstream model's
logits bit for bit: [log](validation_logs/2026-10-12/evo2-rtx-4060-official/README.md)): with mmaemu the mean
\|Δ log p(A,C,G,T)\| is 0.031 and 0.029, and the most likely base agrees at 99.90% and 99.93% of the positions
(native FP8: 3.97 and 4.07, 23%). The official path on an H100 is 0.030 and 0.029 from the RTX 4060 on these two
windows, and mmaemu's run 0.0295 and 0.0288 from the H100: as far from either official GPU as they are from each
other (for the 1B: 0.040 for mmaemu, 0.041 between the RTX 4060 and an H100).

**BRCA1** (the first 100 variants of the notebook's subset of 500, 20 of them loss-of-function;
[log](validation_logs/2026-10-15/evo2-40b-rtx-5070-ti/README.md#brca1-brca1_40bjson-brca1_40blog)): the 40B with
mmaemu gives an AUROC of 0.855, as the official 40B on the H100 does on the same variants; natively with FP8 0.503
and with bf16 GEMMs 0.590 (the 1B with the official path on the RTX 4060: 0.719). With 20 loss-of-function variants
the AUROCs are rough (95% bootstrap interval of mmaemu's: 0.75-0.93), but per variant mmaemu's score changes
follow the official 40B's closely: Pearson 0.998, Spearman 0.986, the same sign for 99 of the 100 (native FP8:
Pearson 0.17). On all 500 variants the official 40B on the H100 gives 0.868 (the 1B on the RTX 4060: 0.738).

### Running Evo 2 40B

[examples/evo2_offload.py](../examples/evo2_offload.py) is the method of the validation log as one command-line
tool. It takes any Evo 2 model (`--model evo2_40b`, `evo2_20b`, `evo2_1b_base`, ...); only the 1B and the 40B were
run. It needs what the notebook needs (evo2 0.6.0 with vtx and flash-attn, torch with CUDA, mmaemu; no Transformer
Engine) and host RAM for the checkpoint (82 GB for the 40B) plus 2 bytes x the hidden size (8,192 for the 40B) for
every token of the sequences run together. `--ram-gb` (default 32) caps the latter; more sequences run in several
passes. The 40B checkpoint is published in two parts:

```bash
hf download arcinstitute/evo2_40b --local-dir evo2_40b
cat evo2_40b/evo2_40b.pt.part0 evo2_40b/evo2_40b.pt.part1 > evo2_40b.pt
python examples/evo2_offload.py test --model evo2_40b --checkpoint evo2_40b.pt --out test.json
python examples/evo2_offload.py score --model evo2_40b --checkpoint evo2_40b.pt --input seqs.fasta --out scores.tsv
```

- `test` runs the four prompts of `evo2/test/test_evo2.py` and prints each loss and accuracy, their mean and PASS or
  FAIL against the value the test expects (tolerance 1e-3; exit code 1 on FAIL). `--out` writes them to a JSON file
  with the SHA-256 of each prompt's logits.
- `score` reads a FASTA file (or one sequence per line) and writes a TSV of name, length and score: the mean
  log-probability of each next token, as `evo2.scoring.score_sequences` computes it (`--reduce sum`: the sum).
- `--fp8 native` and `--fp8 bf16` run the same model with this GPU's own FP8 GEMMs or with bf16 GEMMs.

Checked against the scripts of the validation log
([run_checks.out](validation_logs/2026-10-15/evo2-40b-rtx-5070-ti/evo2_offload_check/run_checks.out)): on the 1B,
in all three modes, `test` gives the same logits, losses and accuracies and `score` the same scores (40 BRCA1
windows, as `score_sequences`), bit for bit; on the 40B, `test` gives the same logits and passes, in 5.6 minutes on
the RTX 5070 Ti, loading included. That run's peak resident set was 157 GiB on the 180 GiB host, counting the pages
of the memory-mapped checkpoint; hosts with less RAM were not tried.

## Troubleshooting

- *"Model 'evo2_1b_base' requires FP8 via Transformer Engine"*: you called `evo2.Evo2("evo2_1b_base")` without TE.
  Build the model with the notebook's `load_evo2` instead.
- *`UnsupportedFeatureError: mmaemu is inference-only`*: run the model under `torch.inference_mode()`; the FP8
  weight must be derived from `weight.detach()`.
- *vortex warns that flash-attn 2.8.0.post2 is outside its tested range*: harmless here; the outputs above were
  produced with it.
- *Out of memory on the BRCA1 cells*: lower `EVO2_BLACKWELL_VARIANTS`, or score the windows in smaller lists.
