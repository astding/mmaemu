# Evo 2 1B and 40B with mmaemu's H100 PCIe profile on an RTX 5070 Ti, 2026-10-16

Upstream Evo 2 1B (`evo2_1b_base`) and 40B (`evo2_40b`), evo2 0.6.0 and vtx 1.1.0 without Transformer Engine, on a
16 GB RTX 5070 Ti, with their FP8 input projections computed by mmaemu with the H100 PCIe profile `hopper-te`
(Transformer Engine 2.3's GEMM with its 32 MiB workspace, as measured on an H100 PCIe). The H100 PCIe is mmaemu's
default platform; the earlier whole-model runs with mmaemu, the 1B notebook of 14 October
([../../2026-10-14/evo2-rtx-5070-ti/README.md](../../2026-10-14/evo2-rtx-5070-ti/README.md)) and the 40B of
15 October ([../../2026-10-15/evo2-40b-rtx-5070-ti/README.md](../../2026-10-15/evo2-40b-rtx-5070-ti/README.md)),
used the RTX 4060 profile `ada-te`. This run repeats their tests with the H100's GEMM: the upstream test of each
model, its windows, and BRCA1 (the 500 variants for the 1B, the first 100 for the 40B). It compares them with the
official path on the H100 PCIe ([1B, 14 October](../../2026-10-14/evo2-h100-sm90/README.md),
[1B and 40B, 15 October](../../2026-10-15/evo2-40b-h100-sm90/README.md)) and on the RTX 4060, and with the `ada-te`
runs. The weights stay in host RAM and the model runs one block at a time on the GPU, as in the 40B log.

## Setup

- GPU: NVIDIA GeForce RTX 5070 Ti (sm_120, 16 GB), WSL2, the host of the 40B log of 15 October. This run did not
  record the driver version.
- Python 3.12.2, torch 2.7.1+cu128 (`status.txt`). `status.txt` has no package line: `run_chain.sh` lists the
  packages with `pip`, which the venv does not have. Read from the same venv afterwards (`importlib.metadata`):
  evo2 0.6.0, vtx 1.1.0, flash-attn 2.8.0.post2, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.0.6,
  biopython 1.88, the versions of the 40B log.
- Checkpoints (paths in `run_chain.sh` and `results/`): `<models>/evo2_1b_base/evo2_1b_base.pt`, SHA-256
  `8ffba7d0…`, and `<models>/evo2_40b/evo2_40b.pt`, `dd299612…` (hashed after the run), the files of the 14 and
  15 October runs.
- [status.txt](status.txt): the chain ran on 2026-10-16 from 21:16 to 22:58 PDT, the four steps one after the other
  under the GPU lock, each with exit code 0. Per step:

| Step | Seconds (loading included) | MemAvailable at the start | Peak resident set | GPU peak |
|---|---:|---:|---:|---:|
| 1B test ([logs/1b_test.log](logs/1b_test.log)) | 17 | 163 GiB | 4.8 GiB | 2.3 GiB |
| 1B BRCA1 ([logs/1b_brca1.log](logs/1b_brca1.log)) | 937 | 163 GiB | 36 GiB | 2.2 GiB |
| 40B test ([logs/40b_test.log](logs/40b_test.log)) | 329 | 133 GiB | 120 GiB | 10.9 GiB |
| 40B BRCA1 ([logs/40b_brca1.log](logs/40b_brca1.log)) | 4,875 | 135 GiB | 138 GiB | 10.7 GiB |

The peak resident set is GNU time's `maxrss` at the end of each log; the GPU peak is the largest the log reports
per block. The 40B test loaded the model in 121 s and ran its seven windows in 203 s.

## Code under test

- mmaemu: a clean worktree of fa0fd90 on `PYTHONPATH`. `status.txt` gives its commit at the start of the chain and
  `clean=0` (the number of lines of `git status --porcelain`) at the start and at the end.
- The profile was named `hopper-h100pcie-te` there (and `ada-te` was `ada-rtx4060-te`); the logs and the scripts
  keep that name (`results/` names the mode, `mmaemu_h100`). 63b7395, after this run, renamed the profiles. In
  `mmaemu/platform/profiles.py` it changes the names, the default profile's name, the descriptions and the
  docstrings, and no other field of a profile (in `mmaemu/api.py` only docstrings); the split-K table files are
  unchanged. Whether the outputs are unchanged on the renamed tree is what the example check below tests.
- Each distinct call shape was planned on kernel `cuda:h4` with S = 1 from the H100 table: its `mmaemu.explain`
  plan, in the logs, is `validation='ok'`, preset `hopper` (`validated`), `source='table'`, `verified=True`, table
  `heur_h100pcie_lt12.8.3.14`, backend `cuda:h4`, `warnings=[]` (seven shapes in each test, M from 2,048 to 8,192
  rows, and M = 8,192 for BRCA1; (K, N) = (1920, 5760) for the 1B and (8192, 24576) for the 40B). The logs contain
  no split-K warning.

## The harness (scripts/)

- [run_offload.py](scripts/run_offload.py) and [brca1_offload.py](scripts/brca1_offload.py) are copies of the
  scripts of the 40B log of 15 October, whose README describes the method: the upstream model
  (`vortex.model.model.StripedHyena`) built on the CPU, each block copied to the GPU, run on every window and
  dropped, and the projections replaced as in the notebook. `run_offload.py` adds the mode `mmaemu_h100`:
  `mmaemu.scaled_mm` with the profile `hopper-h100pcie-te`, and the `mmaemu.explain` plan of each distinct call
  logged once. `brca1_offload.py` takes its projections from `run_offload.py` (`patch_projections`, `set_mode`), so
  it has the mode too; it differs from the 15 October copy only in the path of the RTX 4060 1B reference
  (`ADA_1B`). [harness.diff](scripts/harness.diff) is the diff of both against the 15 October scripts.
- The 40B log checked both scripts on the 1B against the model held on the GPU (logits and BRCA1 scores bit for
  bit). Here `--check-resident` repeats the check for the 1B test: on all seven windows the logits equal those of
  the model held on the GPU, bit for bit (`resident_equal` in `results/1b_test.json` and `compare.json`).
- `run_offload.py` is byte for byte the file that ran (its sha256 is the one in `status.txt`).
  `brca1_offload.py` and [run_chain.sh](scripts/run_chain.sh) differ from the files that ran only in redacted
  paths: `<src>` (the mmaemu worktree), `<venv>` (the Python environment), `<models>` (the checkpoints) and
  `$MMAEMU_GPU_LOCK` (the GPU lock). With `<src>` replaced by the worktree's path, `brca1_offload.py` has the
  sha256 of `status.txt`. The logs are redacted the same way; `<run>` there is the run directory.

## Results

Every number below is in [compare.json](compare.json) (see [compare.py and compare.json](#comparepy-and-comparejson))
unless another file is named. `hopper-te` is this run. `ada-te` is the RTX 4060 profile on the same GPU: for the
1B the notebook run of 14 October (windows, BRCA1) and the 1B check of the 40B log of 15 October (the upstream test,
`check_1b.json`, whose bf16 loss and accuracy are the notebook's), for the 40B the run of 15 October. "Official" is
the upstream path with Transformer Engine FP8 on the GPU the row names.

### Evo 2 1B: the upstream test ([results/1b_test.json](results/1b_test.json))

The four prompts of `evo2/test/test_evo2.py` (p0-p3); the mean loss as the test computes it (bf16 on the GPU) and
in fp32.

| Run | Loss (bf16, as the test) | Loss (fp32) | Accuracy |
|---|---:|---:|---:|
| expected by the upstream test | 0.501953125 | | 0.79556 |
| H100 PCIe, official | 0.5022 | 0.5026 | 0.79529 |
| RTX 4060, official | | 0.5024 | 0.79515 |
| RTX 5070 Ti + mmaemu, `hopper-te` | 0.5022 | 0.5024 | 0.79611 |
| RTX 5070 Ti + mmaemu, `ada-te` | 0.5027 | 0.5025 | 0.79604 |

The test passes (within 1e-3 of the expected loss). The bf16 loss is 0.5022, the H100's value in bf16; in fp32 it
is 0.5024 against the H100's 0.5026. Per prompt (the H100 from `results/1b_official/test.json` of the
[H100 log of 15 October](../../2026-10-15/evo2-40b-h100-sm90/README.md)):

| Prompt | Loss (bf16), `hopper-te` and H100 | Loss (fp32), `hopper-te` / H100 | Accuracy, `hopper-te` / H100 |
|---|---:|---:|---:|
| p0 | 0.2256 | 0.22549 / 0.22529 | 0.92504 / 0.92443 |
| p1 | 0.7813 | 0.78016 / 0.78059 | 0.66534 / 0.66506 |
| p2 | 0.5586 | 0.55991 / 0.56031 | 0.77870 / 0.77724 |
| p3 | 0.4434 | 0.44421 / 0.44413 | 0.81536 / 0.81445 |

The bf16 losses of the four prompts equal the H100's, but the fp32 losses and the accuracies differ on every
prompt, and the logits differ from the H100's (SHA-256) on every prompt and window. On the 2,048-bp window even the
bf16 losses differ (0.203125 against the H100's 0.2021484375). The equal bf16 mean is a coincidence of rounding
each prompt's loss to bf16.

### Evo 2 1B: six windows against the official RTX 4060 output

The H100's per-position 1B log-probabilities are not in the repository (the H100 log of 15 October keeps per window
the loss, the mean log p(next) and the logits SHA-256), so the windows are compared with the official RTX 4060 output
(`examples/data/evo2_1b_ada_rtx4060_te.npz`), with the measures of the notebook log: over the six windows p0-p3,
w8k and w2k, the mean of each window's mean \|Δ log p(A,C,G,T)\| (range in brackets), the share of positions where
the most likely of A, C, G, T agrees, and the largest \|Δ sequence score\| (a window's mean log p(next)).

| Run, against the RTX 4060 official | Mean \|Δ log p(A,C,G,T)\| | Most likely base agrees | Max \|Δ sequence score\| |
|---|---:|---:|---:|
| H100 PCIe, official | 0.0411 (0.035-0.049) | 0.9941 | 0.0009 |
| RTX 5070 Ti + mmaemu, `hopper-te` | 0.0407 (0.036-0.047) | 0.9940 | 0.0019 |
| RTX 5070 Ti + mmaemu, `ada-te` | 0.0403 (0.035-0.047) | 0.9942 | 0.0016 |

The `hopper-te` maximum is on w2k (mean log p(next) -0.2029 against the RTX 4060's -0.2010). The H100 and `ada-te`
rows are those of the notebook log.

### Evo 2 1B: BRCA1 ([results/1b_brca1.json](results/1b_brca1.json))

The notebook's zero-shot test: 500 variants (111 loss of function), 435 reference and 500 variant windows of
8,192 bp, each scored as the mean log-probability of its next tokens (`brca1_offload.py`, as in the 40B log); the
AUROC measures how well a lower score change picks out the loss-of-function variants (95% bootstrap interval,
2,000 resamples).

| Run | AUROC | 95% interval |
|---|---:|---:|
| H100 PCIe, official (14 October) | 0.7332 | 0.673-0.792 |
| RTX 4060, official | 0.7378 | 0.680-0.797 |
| RTX 5070 Ti + mmaemu, `hopper-te` | 0.7354 | 0.675-0.793 |
| RTX 5070 Ti + mmaemu, `ada-te` | 0.7323 | 0.672-0.791 |

The runs against each other: Pearson and Spearman of the 500 score changes (Δ), the share of variants whose Δ has
the same sign, and over the 935 window scores the largest difference and the number of equal scores.

| Pair | Δ Pearson | Δ Spearman | Same sign | Max \|score difference\| | Equal scores (of 935) |
|---|---:|---:|---:|---:|---:|
| `hopper-te` vs H100 official | 0.988 | 0.929 | 86.0% | 0.00039 | 2 |
| `hopper-te` vs RTX 4060 official | 0.987 | 0.920 | 86.2% | 0.00037 | 1 |
| `ada-te` vs H100 official | 0.986 | 0.921 | 84.2% | 0.00038 | 0 |
| `ada-te` vs RTX 4060 official | 0.986 | 0.912 | 84.0% | 0.00040 | 0 |
| H100 official vs RTX 4060 official | 0.987 | 0.922 | 84.2% | 0.00042 | 1 |
| `hopper-te` vs `ada-te` | 0.987 | 0.923 | 86.6% | 0.00039 | 0 |

The `hopper-te` pass took 15.5 minutes (937 s for the step with loading, `status.txt`).

### Evo 2 40B: the upstream test and seven windows ([results/40b_test.json](results/40b_test.json))

| Run | Loss (bf16, as the test) | Loss (fp32) | Accuracy |
|---|---:|---:|---:|
| expected by the upstream test | 0.2159424 | | 0.91673 |
| H100 PCIe, official | 0.2159424 | 0.2160 | 0.91667 |
| RTX 5070 Ti + mmaemu, `hopper-te` | 0.2161 | 0.2158 | 0.91638 |
| RTX 5070 Ti + mmaemu, `ada-te` | 0.2161 | 0.2158 | 0.91688 |

The test passes; the bf16 mean is the same value as `ada-te`'s (0.216064). Per prompt in bf16, p0 and p2 equal the
H100's (0.1523, 0.4453) and p1 and p3 do not (0.17676 against 0.17578, 0.08984 against 0.09033); the logits differ
from the H100's on every prompt.

Against the official 40B on the H100 (`h100_official_te_40b.npz` of the
[H100 log](../../2026-10-15/evo2-40b-h100-sm90/README.md)), every position of the seven windows (`ada-te` ran
w2k, w4k and p0-p3):

| Window (bp) | `hopper-te` vs H100: mean \|Δ log p(A,C,G,T)\| | Max | Most likely base agrees | `ada-te` vs H100: mean | `hopper-te` vs `ada-te`: mean |
|---|---:|---:|---:|---:|---:|
| w2k (2,048) | 0.0289 | 0.61 | 1.0000 | 0.0295 | 0.0299 |
| w4k (4,096) | 0.0283 | 0.62 | 0.9998 | 0.0288 | 0.0297 |
| w8k (8,192) | 0.0323 | 1.31 | 0.9985 | | |
| p0 (6,538) | 0.0267 | 0.56 | 0.9988 | 0.0269 | 0.0268 |
| p1 (7,056) | 0.0511 | 1.33 | 0.9959 | 0.0514 | 0.0511 |
| p2 (6,160) | 0.0331 | 0.57 | 0.9946 | 0.0354 | 0.0353 |
| p3 (7,616) | 0.0523 | 1.91 | 0.9962 | 0.0525 | 0.0539 |

The two official GPUs are 0.0300 (w2k) and 0.0292 (w4k) apart on this measure. Against the official RTX 4060 output
(the 40B log's `rtx4060_official_te_40b.npz`, w2k and w4k only) `hopper-te` is 0.0303 and 0.0297 away, `ada-te`
0.0309 and 0.0285 (the 40B log).

### Evo 2 40B: BRCA1 ([results/40b_brca1.json](results/40b_brca1.json))

The first 100 of the 500 variants (20 loss of function, 98 reference and 100 variant windows), as in the 40B log;
the H100 row is the official 40B's scores of the same windows (`results/40b/brca1.npz` of the H100 log).

| Run (the same 100 variants) | AUROC | 95% interval |
|---|---:|---:|
| H100 PCIe, official | 0.855 | 0.745-0.940 |
| RTX 5070 Ti + mmaemu, `hopper-te` | 0.8569 | 0.749-0.940 |
| RTX 5070 Ti + mmaemu, `ada-te` | 0.855 | 0.751-0.934 |

| Pair | Δ Pearson | Δ Spearman | Same sign | Max \|score difference\| | Equal scores (of 198) |
|---|---:|---:|---:|---:|---:|
| `hopper-te` vs H100 official | 0.9986 | 0.991 | 99% | 0.00048 | 0 |
| `ada-te` vs H100 official | 0.9981 | 0.986 | 99% | 0.00046 | 1 |
| `hopper-te` vs `ada-te` | 0.9981 | 0.985 | 100% | 0.00068 | 0 |

With 20 loss-of-function variants these AUROCs are rough estimates (intervals of about ±0.1). The `hopper-te` pass
took 79.7 minutes (4,875 s for the step with loading, `status.txt`); the times are host-clock times of runs that
copy every block from host RAM, not a speed measurement.

## Reading

- The FP8 GEMMs are the H100's. The 168 outputs of the 21 Evo 2 1B projections captured on the H100 PCIe (the
  probe of 15 October, Transformer Engine and torch callers) equal mmaemu's `hopper` preset on kernel h4 bit for bit
  on fa0fd90, on this GPU ([../rtx-5070-ti-replays/README.md](../rtx-5070-ti-replays/README.md)) and on the
  RTX 4060 ([../rtx-4060-sm89/README.md](../rtx-4060-sm89/README.md)). No 40B projection was captured on an H100.
- The whole model is not bit-identical to the H100's: the other operations run natively on the RTX 5070 Ti, and no
  logits equal the H100's. Its distance to the official H100 run is the size of the distance between the two
  official GPUs: on the 40B windows w2k and w4k 0.0289 and 0.0283 from the H100, where the H100 and the RTX 4060 are
  0.0300 and 0.0292 apart. On the 1B windows (of the official runs only the RTX 4060's per-position output is
  committed) it is 0.0407 from the RTX 4060, and the H100 is 0.0411 from it.
- Between the two profiles the end-to-end differences are small. `hopper-te`'s score changes correlate slightly
  more with the H100's (1B: Pearson 0.988 and Spearman 0.929 against `ada-te`'s 0.986 and 0.921; 40B: 0.9986 and
  0.991 against 0.9981 and 0.986), and on every 40B window compared its mean \|Δ log p\| from the H100 is slightly
  smaller (the table above). The AUROCs do not follow: on the 40B `ada-te` gives the H100's 0.855 and
  `hopper-te` 0.8569; on the 1B `ada-te`'s 0.7323 is nearer the H100's 0.7332 than `hopper-te`'s 0.7354. All of
  them are well inside each other's 95% intervals, and the two profiles are as far from each other (the last
  column of the 40B windows; 1B Δ Pearson 0.987) as either is from an official GPU.
- The 1B's bf16 test loss equal to the H100's (0.5022) is the rounding of the per-prompt losses to bf16, not equal
  outputs: the fp32 losses are 0.5024 and 0.5026.

## compare.py and compare.json

[compare.py](compare.py) computes the comparisons of `compare.json` from committed files: this log's `results/`,
the official H100 runs of 14 and 15 October, the official RTX 4060 outputs
(`examples/data/evo2_1b_ada_rtx4060_te.npz`, the 14 October `upstream_test.json` and the 40B log's
`rtx4060_official_te_40b.npz`) and the `ada-te` runs of 14 and 15 October; its docstring names each file. It
writes [compare.json](compare.json), which also records the sha256 of each file of `results/`. In `compare.json`,
`rtx5070ti_h100_profile` is `hopper-te`, `rtx5070ti_rtx4060_profile` is `ada-te`, and `h100_official` and
`rtx4060_official` are the official runs. To rerun it, from this directory, with the venv of the runs (it needs
numpy, scipy and scikit-learn, no GPU):

```
python compare.py
```

A rerun with `--out` to another file gave `compare.json` byte for byte.

## The example check (example_check/)

`examples/evo2_offload.py`, the shipped command-line tool, with `--profile` on the renamed tree: 2648bba (the code of
63b7395) in a clean worktree, on this GPU, on 17 October from 02:08 to 02:18 PDT, a step of the chain of
[../rtx-5070-ti-renamed/README.md](../rtx-5070-ti-renamed/README.md) (its `status.txt`).
[run_checks.sh](example_check/run_checks.sh) runs `test` on the 1B and `score` on 40 BRCA1 windows (20 reference and
20 variant, written by [make_inputs.py](example_check/make_inputs.py)), each with `ada-te` and with `hopper-te`, then
`test` on the 40B with `hopper-te`. [compare.py](example_check/compare.py) compares the outputs with the runs they
repeat: for `hopper-te` this log's `results/1b_test.json`, `1b_brca1.npz` and `40b_test.json` (fa0fd90, the profile
then named `hopper-h100pcie-te`), for `ada-te` the 40B log of 15 October (`check_1b.json`, `brca1_check_1b.npz`).

[run_checks.out](example_check/run_checks.out): every command exits 0, and `FAILURES 0`.

- 1B, both profiles: per prompt the logits SHA-256, the loss and the accuracy are equal, and the 40 BRCA1 scores are
  equal bit for bit. Mean loss (bf16) 0.5022 with `hopper-te` and 0.5027 with `ada-te`; both PASS.
- 40B, `hopper-te`: per prompt the logits SHA-256, the loss and the accuracy equal those of `results/40b_test.json`;
  mean loss 0.2161, PASS (0.2159 expected, tolerance 1e-3). 458 s, loading included; peak resident set 123 GiB (GNU
  time's `maxrss`, which counts the pages of the memory-mapped checkpoint). The log contains no warning.

The rename leaves these outputs unchanged, and the shipped script with `--profile hopper-te` gives the outputs of this
log's harness.

## Files

| File | Content |
|---|---|
| [status.txt](status.txt) | the chain's record (PDT): the mmaemu commit and its cleanliness at the start and the end, Python and torch, the sha256 of the two scripts, and per step its start, end, exit code, seconds and `MemAvailable` |
| [scripts/run_chain.sh](scripts/run_chain.sh) | the chain: the four steps, one after the other, under the GPU lock and GNU time |
| [scripts/run_offload.py](scripts/run_offload.py) | the model, the projection modes, the upstream test and the windows (`--ref`, `--check-resident`) |
| [scripts/brca1_offload.py](scripts/brca1_offload.py) | BRCA1 |
| [scripts/harness.diff](scripts/harness.diff) | the two scripts against those of the 40B log of 15 October |
| [logs/](logs/) | each step's output: the `mmaemu.explain` plans, per-block times and GPU peaks, the results, GNU time's wall time and `maxrss` |
| [results/](results/) | per step a .json (the metrics; for the tests also the logits SHA-256 of each window) and a .npz: for the tests the log-probabilities of A, C, G, T and of the next token at every position (`mmaemu_h100/<window>/acgt`, `mmaemu_h100/<window>/next`), for BRCA1 the scores (`mmaemu_h100/rs`, `mmaemu_h100/vs`, `ref_index`, `lof`) |
| [compare.py](compare.py), [compare.json](compare.json) | the comparisons |
| [example_check/](example_check/) | `examples/evo2_offload.py` with `--profile` on the renamed tree (above) |
