# Evo 2 40B on an H100 PCIe through the official path, 15 October 2026

The official Evo 2 40B path (the upstream blocks with Transformer Engine FP8) on a second GPU, for the
[Evo 2 40B log](../evo2-40b-rtx-5070-ti/README.md): until this run its only references were the RTX 4060 outputs
of two windows and the upstream test's expected values. mmaemu is not involved in this run.

## Setup

- GPU: H100 PCIe (sm_90, 114 SMs, 80 GB), driver 570.148.08, rented for this run (Lambda); 221 GB of RAM.
- Software, as on the RTX 4060 and on the H100 of [14 October](../../2026-10-14/evo2-h100-sm90/README.md): Python
  3.12.15, torch 2.7.1+cu128, cuBLAS/cuBLASLt 12.8.3.14, Transformer Engine 2.3.0, evo2 0.6.0, vtx 1.1.0, flash-attn
  2.8.0.post2, `NVTE_DISABLE_NVRTC=1`. [te_setup.sh](scripts/te_setup.sh) installs TE as on 14 October (the PyPI
  wheels and the RTX 4060 environment's own `transformer_engine_torch` build, checked against
  [local_te_so.sha256](scripts/local_te_so.sha256)); [results/te-setup.log](results/te-setup.log) records it.
- Checkpoints: `evo2_40b.pt`, the two parts of `arcinstitute/evo2_40b` downloaded and joined by
  [download_40b.sh](scripts/download_40b.sh), SHA-256 `dd299612…`, the file of the RTX 5070 Ti runs
  ([results/download.log](results/download.log)); `evo2_1b_base.pt` `8ffba7d0…`.
- Inputs: [make_inputs.py](scripts/make_inputs.py) (run on the maintainer's host, with its paths) writes the token
  ids: the test windows, checked against the RTX 4060 captures, and the windows of the notebook's 500 BRCA1 variants,
  from the 40B log's `brca1_offload.py`.
- Method: [official_blockwise.py](scripts/official_blockwise.py), the method of the RTX 4060 40B reference
  (`golden_oracle.py` in the [RTX 4060 reference log](../../2026-10-12/evo2-rtx-4060-official/README.md)): each
  block is built by vortex's `get_block` and loaded by vortex's own loaders, and every sequence passes through it
  before the next block is loaded. In inference mode TE keeps the checkpoint's FP8 scales, so the sequences can be
  run in groups. [run_job.sh](scripts/run_job.sh) ran everything; [results/status.txt](results/status.txt) has the
  times (UTC).

## Check of the method on this GPU (Evo 2 1B)

[check_1b.py](scripts/check_1b.py) compares the 1B through `official_blockwise.py` with the real upstream model,
`evo2.Evo2("evo2_1b_base")`, through [real_1b.py](scripts/real_1b.py): the logits of all 11 test windows are equal
bit for bit, and so are the BRCA1 scores of 20 windows, scored by `official_blockwise.py` in five groups and by
`score_sequences` ([results/1b_check.json](results/1b_check.json)). The 1B logits also equal, bit for bit, those of
the H100 of 14 October, another instance, on all six of its windows (`compare.json`).

## Upstream test ([results/40b/test.json](results/40b/test.json))

| Run | Loss (bf16, as the test) | Loss (fp32) | Accuracy |
|---|---:|---:|---:|
| expected by the upstream test | 0.2159424 | | 0.91673 |
| **H100, official TE FP8** | **0.2159424** | 0.2160 | 0.91667 |
| RTX 5070 Ti + mmaemu | 0.2161 | 0.2158 | 0.91688 |
| RTX 5070 Ti, native FP8 | 1.4023 | 1.4048 | 0.25679 |
| RTX 5070 Ti, bf16 GEMMs | 1.4043 | 1.4057 | 0.25592 |

The H100's mean loss is the expected value to all seven digits the test gives. Per prompt (bf16 loss, accuracy):
p0 0.1523, 0.9426; p1 0.1758, 0.9373; p2 0.4453, 0.8225; p3 0.0903, 0.9641. The RTX 5070 Ti rows are those of the
40B log.

## The two official GPUs, and the RTX 5070 Ti against the H100

[compare.py](compare.py) → [compare.json](compare.json). The measures are those of the 1B logs: mean and max
\|Δ log p(A,C,G,T)\| over every position, the share of positions where the most likely of A, C, G, T agrees, and
for the two official runs the share of bf16 logits that differ.

| Window (bp) | H100 vs RTX 4060, both official: logits differing | mean \|Δ\| | max | most likely base agrees |
|---|---:|---:|---:|---:|
| w2k (2,048) | 60.5% | 0.0300 | 0.94 | 0.9990 |
| w4k (4,096) | 61.0% | 0.0292 | 0.73 | 0.9990 |
| p1, first 4,096 | 73.9% | 0.0495 | 1.30 | 0.9966 |
| p2, first 4,096 | 66.2% | 0.0380 | 0.60 | 0.9949 |
| p3, first 4,096 | 74.8% | 0.0488 | 1.32 | 0.9971 |

(The first 4,096 bp of p0 are w4k.)

| Window | RTX 5070 Ti + mmaemu vs H100: mean \|Δ\| | most likely base agrees | native FP8: mean \|Δ\| | agrees | bf16 GEMMs: mean \|Δ\| | agrees |
|---|---:|---:|---:|---:|---:|---:|
| w2k | 0.0295 | 1.0000 | 3.97 | 0.234 | 3.97 | 0.235 |
| w4k | 0.0288 | 0.9993 | 4.07 | 0.236 | 4.07 | 0.235 |
| p0 | 0.0269 | 0.9991 | 3.75 | 0.259 | 3.75 | 0.259 |
| p1 | 0.0514 | 0.9956 | 4.38 | 0.238 | 4.38 | 0.237 |
| p2 | 0.0354 | 0.9950 | 3.07 | 0.275 | 3.07 | 0.271 |
| p3 | 0.0525 | 0.9965 | 5.47 | 0.248 | 5.47 | 0.247 |

On w2k and w4k the RTX 5070 Ti with mmaemu is as far from the H100 (0.0295, 0.0288) as from the RTX 4060 (0.031,
0.029, in the 40B log) and as the two official GPUs are from each other (0.030, 0.029).

## BRCA1 ([results/40b/brca1.json](results/40b/brca1.json), [results/40b/brca1.npz](results/40b/brca1.npz))

The notebook's 500 variants (111 loss-of-function; 435 reference and 500 variant windows of 8,192 bp), scored as
`evo2.scoring.score_sequences` does (batch size 1). The first 100 variants are those of the 40B log. AUROC with a
95% bootstrap interval (2,000 resamples of the variants):

| Run | 500 variants | first 100 (20 loss-of-function) |
|---|---:|---:|
| **Evo 2 40B, H100, official TE FP8** | **0.868** (0.829-0.904) | 0.855 (0.745-0.940) |
| Evo 2 40B, RTX 5070 Ti + mmaemu | | 0.855 (0.751-0.934) |
| Evo 2 40B, RTX 5070 Ti, native FP8 | | 0.503 (0.338-0.658) |
| Evo 2 40B, RTX 5070 Ti, bf16 GEMMs | | 0.590 (0.444-0.719) |
| Evo 2 1B, official TE FP8, RTX 4060 (`examples/data/evo2_1b_ada_rtx4060_te.npz`) | 0.738 (0.680-0.797) | 0.719 (0.549-0.871) |

Per variant, against the H100 (the first 100 variants; the H100's mean \|Δ\| is 2.9e-3):

| RTX 5070 Ti | Pearson of Δ | Spearman of Δ | same sign | max \|score difference\| over the 198 windows |
|---|---:|---:|---:|---:|
| mmaemu | 0.998 | 0.986 | 99% | 4.6e-4 |
| native FP8 | 0.170 | 0.117 | 55% | 0.69 |
| bf16 GEMMs | 0.193 | 0.219 | 55% | 0.69 |

On these 100 variants the RTX 5070 Ti with mmaemu and the H100 give the same AUROC, and their variant scores
agree more closely than the 1B's do between the RTX 4060 and the H100 of 14 October (Pearson 0.987, Spearman 0.922),
where the score changes are five times smaller (mean \|Δ\| 6.1e-4).

Time on the H100: 2.4 s per 8,192-bp window (GPU memory peak 35.1 GiB with 198 windows, 41.6 GiB with 250), 36
minutes for the 935 windows in four groups; the upstream test took 36 s.

## Files

- [scripts/](scripts/): the job as run on the instance: `make_inputs.py`, `download_40b.sh`, `te_setup.sh`,
  `official_blockwise.py`, `real_1b.py`, `check_1b.py`, `run_job.sh`, `local_te_so.sha256`. The environment was
  built by the maintainer's Lambda launcher, as on 14 October.
- [results/](results/): status, `nvidia-smi`, package lists before and after the TE setup, the logs of every
  step, the JSON outputs, the BRCA1 scores of the 1B check and of the 40B run, and `SHA256SUMS` of the full output
  directory. Checksums that differ: `status.txt` (its final `DONE` line was written after `SHA256SUMS`), and
  `te-setup.log` and `1b_official.log`, where the task directory on the instance is replaced by `$R` and the
  provider's apt mirror by a placeholder; the `test.npz` files are not committed.
- [h100_official_te_40b.npz](h100_official_te_40b.npz): per test window, `ids`, `acgt` and `next`, written by
  [make_reference.py](make_reference.py), which checks each window's logits against the SHA-256 in
  `results/40b/test.json`. The bf16 logits (5.8 MB for the 40B, 5.2 and 4.2 MB for the 1B runs) are not committed.
- [compare.py](compare.py), [compare.json](compare.json): everything above from the files of this repository, except
  the logits-differing column and the 4,096-bp prefixes, which need the RTX 4060 captures and the H100 logits
  (`--rtx4060-captures`, `--h100-logits`).

## Not covered

- One H100, one run; determinism on the 40B was not rerun (the 1B logits equal those of another H100 instance).
- The 40B with mmaemu was run on 100 of the 500 BRCA1 variants, not all of them.
