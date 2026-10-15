# Evo 2 1B on an RTX 5070 Ti with mmaemu, 14 October 2026

The run of [examples/evo2_blackwell.ipynb](../../../../examples/evo2_blackwell.ipynb) whose outputs are in the
committed notebook. Upstream Evo 2 1B (evo2 0.6.0, vtx 1.1.0) without Transformer Engine, with its 21 FP8
projections computed in three ways: by mmaemu (`profile="ada-rtx4060-te"`, kernel v4), natively by
`torch._scaled_mm`, and as bf16 GEMMs. Everything is compared with the official path (evo2 + TE 2.3.0 FP8) on an
RTX 4060, whose outputs ship in `examples/data/`, and with the same official path on an H100 PCIe
([evo2-h100-sm90](../evo2-h100-sm90/README.md)).

## Setup

- GPU: NVIDIA GeForce RTX 5070 Ti (sm_120, 70 SMs, 16 GB), Windows host driver 610.88, WSL2.
- Python 3.12.2, torch 2.7.1+cu128 (cuBLAS/cuBLASLt 12.8.3.14), evo2 0.6.0, vtx 1.1.0, flash-attn 2.8.0.post2,
  mmaemu 0.1.0.dev0 (a wheel of this package source, which differs from it only in a code comment; backend
  `cuda:v4` from the release fatbins).
- Checkpoint: `evo2_1b_base.pt` from `arcinstitute/evo2_1b_base` (the file of the RTX 4060 and H100 runs).
- BRCA1: the 500-variant subset (`df.sample(n=500, random_state=0)` of the 3,893 variants of Findlay et al. 2018),
  8,192-bp windows of GRCh37 chr17, 435 unique reference windows; data files from the upstream evo2 repository.

## Results

All numbers are in [results.json](results.json); the notebook shows them with plots.

**One GEMM** (block 0's projection on the 2,048-bp window, M 2048, N 5760, K 1920): the E4M3 input and weight
have the SHA-256 of the RTX 4060's (hardware recording `evo2.evo2_1b_w2k`) in both modes. mmaemu's bf16 output
has the SHA-256 of the RTX 4060's TE output; the native output differs in 8,293,802 of 11,796,480 values (70.3%),
by up to 0.3125. A 32-row slice computed by mmaemu on the CPU equals the GPU result bit for bit.

**Six windows** against the official RTX 4060 output (mean over the windows, range in brackets):

| Run | Mean \|Δ log p(A,C,G,T)\| | Most likely base agrees | Max \|Δ sequence score\| | 8,192-bp forward pass |
|---|---:|---:|---:|---:|
| H100, official TE FP8 (baseline) | 0.0411 (0.0351-0.0485) | 0.9941 | 0.0009 | |
| RTX 5070 Ti + mmaemu | 0.0403 (0.0352-0.0466) | 0.9942 | 0.0016 | 1.03 s |
| RTX 5070 Ti, native FP8 | 2.5791 (1.9133-3.4576) | 0.4301 | 1.1157 | 0.45 s |
| RTX 5070 Ti, bf16 GEMMs | 2.8550 (1.8873-3.4794) | 0.3698 | 1.1453 | 0.46 s |

The logits of all three modes were bit-identical to those of an earlier run of the notebook (20 BRCA1 variants)
on the same machine.

**The upstream test** (prompts p0-p3; cross entropy in bf16 on the GPU, as `evo2/test/test_evo2.py` computes it):
mmaemu 0.5027 (accuracy 0.7960), native FP8 1.2480 (0.3848), bf16 GEMMs 1.3652 (0.3262); expected 0.502 (0.7956).
Issue #157 reports 1.248 and 1.365 for an RTX 5070 Ti with TE FP8 and with FP8 disabled.

**BRCA1** (500 variants, 111 loss of function):

| Run | AUROC | Δ Pearson vs RTX 4060 | Δ Spearman vs RTX 4060 | Max \|Δ sequence score\| | Minutes |
|---|---:|---:|---:|---:|---:|
| RTX 4060, official TE FP8 | 0.7378 | | | | |
| H100, official TE FP8 | 0.7332 | 0.987 | 0.922 | 0.0004 | |
| RTX 5070 Ti + mmaemu | 0.7323 | 0.986 | 0.912 | 0.0004 | 15.9 |
| RTX 5070 Ti, native FP8 | 0.5586 | 0.059 | 0.165 | 0.18 | 7.0 |
| RTX 5070 Ti, bf16 GEMMs | 0.4842 | 0.069 | 0.078 | 0.62 | 7.2 |

Times are from the host timer (`time.perf_counter` around synchronised calls), which runs 6-10% fast on this WSL2
host (docs/performance.md), so the three BRCA1 passes add up to more than the 28 minutes of wall-clock time that the
whole notebook took.

## Files

| File | Content |
|---|---|
| [results.json](results.json) | every number of the notebook, written by its last code cell: GPU, versions, the one-GEMM check, the per-window and per-mode metrics, the upstream test, BRCA1 |
| [brca1_mmaemu.npz](brca1_mmaemu.npz), [brca1_native.npz](brca1_native.npz), [brca1_bf16.npz](brca1_bf16.npz) | the BRCA1 sequence scores of each mode: `rs` (435 reference windows), `vs` (500 variants), `ref_index` |

The full logits of the six windows (`windows_logits.npz`, 7.9 MB) are not committed.
