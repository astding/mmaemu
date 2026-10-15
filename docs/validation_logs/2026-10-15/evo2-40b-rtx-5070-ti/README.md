# Evo 2 40B on an RTX 5070 Ti with mmaemu, 15 October 2026

Upstream Evo 2 40B (`evo2_40b`, evo2 0.6.0, vtx 1.1.0) without Transformer Engine, on a 16 GB RTX 5070 Ti, with
its 42 FP8 projections computed in three ways, as in the 1B notebook
([../../2026-10-14/evo2-rtx-5070-ti/README.md](../../2026-10-14/evo2-rtx-5070-ti/README.md)): by mmaemu
(`profile="ada-rtx4060-te"`), natively by `torch._scaled_mm`, and as bf16 GEMMs. The weights stay in host RAM and
the model runs one block at a time on the GPU.

## Setup

- GPU: NVIDIA GeForce RTX 5070 Ti (sm_120, 16 GB), Windows host driver 610.88, WSL2 with 180 GB of RAM.
- Python 3.12.2, torch 2.7.1+cu128, evo2 0.6.0, vtx 1.1.0, flash-attn 2.8.0.post2; numpy 2.5.4, scikit-learn
  1.9.1, scipy 1.18.1, pandas 3.0.6, biopython 1.88.
- mmaemu: the source tree of c61717e (`PYTHONPATH`), backend `cuda:v4` from the release fatbins. Every
  projection shape of these runs (K 8192, N 24576, M from 2,048 to 8,192) has S = 1 from the RTX 4060's split-K
  table (`mmaemu.explain`: source `table`, verified).
- Checkpoint: `evo2_40b.pt` of `arcinstitute/evo2_40b`, the concatenation of its two parts (each part's SHA-256
  equals the Hugging Face LFS id; the merged file is `dd299612…`). It loads with every model tensor present; the
  checkpoint's extra keys are the TE `_extra_state` entries and `unembed.weight` (tied to the embedding), as for
  the 1B.
- Scripts: [run_offload.py](run_offload.py) (the model, the three projection modes, the upstream test) and
  [brca1_offload.py](brca1_offload.py) (BRCA1). The model is the upstream `vortex.model.model.StripedHyena`, built
  from evo2's configuration with `use_fp8_input_projections=False` and loaded with vortex's checkpoint loader; the
  projections are replaced as in the notebook's `evo2-fp8-patch` cell. It is built on the CPU, and each block is
  copied to the GPU, run on every window and dropped again. Host RAM: 82 GB resident; GPU peak 11.2 GiB.

## Checks of the scripts (Evo 2 1B)

- [check_1b.json](check_1b.json), [check_1b.log](check_1b.log): the 1B through `run_offload.py`. In all three
  modes and on all five windows the logits equal, bit for bit, those of the same model held on the GPU and run
  through vortex's own forward pass. The upstream-test losses are those of the 1B log of 2026-10-14: mmaemu
  0.5027, native FP8 1.2480, bf16 GEMMs 1.3652.
- [brca1_check_1b.json](brca1_check_1b.json), [brca1_check_1b.log](brca1_check_1b.log): the 1B through
  `brca1_offload.py` on the first 20 variants (40 windows). In all three modes the scores equal, bit for bit, those
  of `evo2.scoring.score_sequences` (batch size 1) on the model held on the GPU. This also checks that computing a
  long Hyena filter once per block and window length gives the same bits as computing it for every window.

## Upstream test and two windows ([test_40b.json](test_40b.json), [test_40b.log](test_40b.log))

The four prompts of `evo2/test/test_evo2.py` (p0-p3) and the first 2,048 and 4,096 bp of their concatenation
(w2k, w4k; the windows of the [RTX 4060 reference log](../../2026-10-12/evo2-rtx-4060-official/README.md)).

**Upstream test** (mean over p0-p3; the loss as the test computes it, in bf16 on the GPU, and in fp32):

| Run | Loss (bf16, as the test) | Loss (fp32) | Accuracy |
|---|---:|---:|---:|
| expected by the upstream test | 0.2159424 | | 0.91673 |
| RTX 5070 Ti + mmaemu | 0.2161 | 0.2158 | 0.91688 |
| RTX 5070 Ti, native FP8 | 1.4023 | 1.4048 | 0.25679 |
| RTX 5070 Ti, bf16 GEMMs | 1.4043 | 1.4057 | 0.25592 |

The test passes when the mean loss is within 1e-3 of the expected value; mmaemu's is 1.2e-4 from it. An accuracy
of 0.25 is what guessing one of the four bases gives.

**Against the official path on an RTX 4060** ([rtx4060_official_te_40b.npz](rtx4060_official_te_40b.npz)): the
log-probabilities of A, C, G and T at every position and of every next token, from the upstream blocks with TE FP8
on the RTX 4060, run one at a time with vortex's own loaders (run this way, the 1B gives the logits of the real
upstream model bit for bit). They were captured on 12 October 2026 by `golden_oracle.py`; the script, the run logs
and a check that this file equals the captures are in the
[RTX 4060 reference log](../../2026-10-12/evo2-rtx-4060-official/README.md).

| Run | Window | Mean \|Δ log p(A,C,G,T)\| | Max | Most likely base agrees | Mean log p(next) |
|---|---|---:|---:|---:|---:|
| RTX 4060, official TE FP8 | w2k | | | | -0.0718 |
| RTX 5070 Ti + mmaemu | w2k | 0.0309 | 0.909 | 0.9990 | -0.0720 |
| RTX 5070 Ti, native FP8 | w2k | 3.9742 | 8.071 | 0.2339 | -1.4155 |
| RTX 5070 Ti, bf16 GEMMs | w2k | 3.9758 | 8.071 | 0.2349 | -1.4171 |
| RTX 4060, official TE FP8 | w4k | | | | -0.0603 |
| RTX 5070 Ti + mmaemu | w4k | 0.0285 | 0.574 | 0.9993 | -0.0603 |
| RTX 5070 Ti, native FP8 | w4k | 4.0657 | 8.013 | 0.2363 | -1.3999 |
| RTX 5070 Ti, bf16 GEMMs | w4k | 4.0668 | 8.031 | 0.2346 | -1.4008 |

As for the 1B, only the FP8 GEMMs are the RTX 4060's; the other operations run natively on the RTX 5070 Ti, so the
outputs are close to the RTX 4060's but not equal to them. The official 40B path on an H100, run the same way the
same day ([H100 log](../evo2-40b-h100-sm90/README.md)), is 0.030 and 0.029 from the RTX 4060 on this measure, and
mmaemu's run is 0.0295 and 0.0288 from the H100; the H100 also gives the upstream test's expected loss, 0.2159424.
The per-position arrays of the three runs are in [test_40b.npz](test_40b.npz).

## BRCA1 ([brca1_40b.json](brca1_40b.json), [brca1_40b.log](brca1_40b.log))

The zero-shot BRCA1 test of the 1B notebook (section 3): each variant's score is the change in the mean
log-probability of its 8,192-bp window against the reference window (`evo2.scoring.score_sequences`, batch size 1),
and the AUROC measures how well a lower score picks out the loss-of-function variants. Here the first 100 variants of
the notebook's random subset of 500 (seed 0): 20 loss-of-function, 98 reference and 100 variant windows, all three
modes in one run of [brca1_offload.py](brca1_offload.py). [brca1_40b.npz](brca1_40b.npz) holds the scores.

| Run (the same 100 variants) | AUROC | Minutes |
|---|---:|---:|
| Evo 2 40B, official TE FP8 on an H100 ([H100 log](../evo2-40b-h100-sm90/README.md)) | 0.855 | |
| Evo 2 1B, official TE FP8 on the RTX 4060 (from `examples/data/evo2_1b_ada_rtx4060_te.npz`) | 0.719 | |
| Evo 2 40B, RTX 5070 Ti + mmaemu | 0.855 | 84.6 |
| Evo 2 40B, RTX 5070 Ti, native FP8 | 0.503 | 28.3 |
| Evo 2 40B, RTX 5070 Ti, bf16 GEMMs | 0.590 | 29.2 |

With only 20 loss-of-function variants these AUROCs are rough estimates (95% bootstrap intervals about ±0.1 for
mmaemu, ±0.15 natively, in the H100 log), but the variant scores themselves can be compared: against the official
40B on the H100, mmaemu's score changes have Pearson 0.998 and Spearman 0.986 and the same sign for 99 of the 100
variants; natively with FP8 Pearson 0.17, with bf16 GEMMs 0.19. On all 500 variants the official 40B on the H100
gives 0.868. Natively, with FP8 or bf16 GEMMs, the 40B scores the variants at or near chance, as its next-base
accuracy in the upstream test (0.26) is that of a guess. The minutes are host-clock times of the three passes, run one after the other, each copying all 50 blocks
from host RAM; mmaemu's FP8 GEMMs make its pass about three times longer.

## examples/evo2_offload.py ([evo2_offload_check/](evo2_offload_check/))

`examples/evo2_offload.py` is this log's method as a command-line tool. [run_checks.sh](evo2_offload_check/run_checks.sh)
runs it on the 1B in all three modes (`test`, and `score` on the 40 BRCA1 windows of the 1B check above, which
[make_inputs.py](evo2_offload_check/make_inputs.py) writes as a FASTA file) and on the 40B with mmaemu (`test`).
[compare.py](evo2_offload_check/compare.py) checks each prompt's logits SHA-256, loss and accuracy against
`check_1b.json` and `test_40b.json`, and the scores against [brca1_check_1b.npz](brca1_check_1b.npz), bit for bit:
0 failures ([run_checks.out](evo2_offload_check/run_checks.out)). On the 1B, `test` exits with 1 for native FP8 and
bf16 GEMMs, as it should: the upstream test fails there. The 40B `test` took 5.6 minutes, loading included, and
its peak resident set was 157 GiB, which counts the pages of the memory-mapped checkpoint read while loading (the
weights take 82 GB). Hosts with less RAM were not tried.

## Not covered

- One GPU (RTX 5070 Ti) and one host. The 7B and 20B models were not run.
- The official 40B references: the RTX 4060 (two windows) and an H100 (the test prompts, the windows and the 500
  BRCA1 variants; [H100 log](../evo2-40b-h100-sm90/README.md)). The 40B with mmaemu was scored on 100 of the 500
  variants.
- The times are host-clock times of runs that compute three modes together and copy every block from host RAM;
  they are not a speed measurement.
