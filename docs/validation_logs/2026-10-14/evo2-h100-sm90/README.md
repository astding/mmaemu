# Evo 2 1B on an H100 PCIe through the official path, 14 October 2026

A baseline for the Evo 2 results of mmaemu: how far two GPUs running the *official* implementation (upstream
`evo2.Evo2("evo2_1b_base")` with Transformer Engine FP8) are from each other. The reference is the RTX 4060 run
whose outputs ship with the notebook (`examples/data/evo2_1b_ada_rtx4060_te.npz`). mmaemu is not involved in this
run.

## Setup

- GPU: H100 PCIe (sm_90, 114 SMs), driver 570.148.08, rented for this run (Lambda).
- Software, the same as the RTX 4060 run: torch 2.7.1+cu128, cuBLAS/cuBLASLt 12.8.3.14, Transformer Engine 2.3.0,
  evo2 0.6.0, vtx 1.1.0, flash-attn 2.8.0.post2, `NVTE_DISABLE_NVRTC=1`.
  - `libtransformer_engine.so` (PyPI `transformer_engine_cu12` 2.3.0) is byte-identical to the RTX 4060
    environment's copy; `transformer_engine_torch` is that environment's own build, copied over (it needs
    GLIBCXX_3.4.32, from the Ubuntu toolchain PPA).
  - The checkpoint `evo2_1b_base.pt` is the same file (sha256 8ffba7d0…6f64).
- FP8 check: the script asserts that the model has FP8 TE projections (21 modules). The cuBLASLt log of the
  window stage has 252 E4M3 GEMMs (21 projections x 6 windows x 2 runs), all algoId 66 with a 32 MiB workspace:
  [cublaslt_fp8_calls.json](cublaslt_fp8_calls.json).
- Inputs: the six windows use the token ids of the RTX 4060 run; BRCA1 uses the same 500-variant subset (seed 0
  sample of the Findlay et al. 2018 table, 435 unique reference windows of 8,192 bp) and upstream
  `score_sequences` with its default arguments.
- Determinism: both stages ran twice; logits and scores were bit-identical between the runs.

## H100 vs RTX 4060, both official TE FP8

| Window | bf16 logits differing | max abs | mean \|Δ log p(A,C,G,T)\| | ACGT argmax agree | mean next-token log p, H100 / RTX 4060 |
|---|---:|---:|---:|---:|---|
| p0 | 53.7% | 2.25 | 0.0375 | 0.9982 | -0.22529 / -0.22534 |
| p1 | 57.8% | 1.75 | 0.0391 | 0.9894 | -0.78059 / -0.78014 |
| p2 | 56.1% | 1.00 | 0.0399 | 0.9904 | -0.56031 / -0.56000 |
| p3 | 62.8% | 2.25 | 0.0464 | 0.9947 | -0.44413 / -0.44432 |
| w8k | 55.5% | 2.25 | 0.0351 | 0.9957 | -0.38415 / -0.38401 |
| w2k | 56.6% | 1.25 | 0.0485 | 0.9961 | -0.20190 / -0.20101 |

BRCA1 subset (500 variants, 111 loss of function):

- AUROC 0.7332 on the H100, 0.7378 on the RTX 4060.
- Sequence scores: 1 of 935 identical; Pearson 0.999999; max abs difference 4.2e-4.
- Variant score differences (Δ): Pearson 0.987, Spearman 0.922, same sign for 84.2%; max abs difference 4.5e-4,
  against a mean \|Δ\| of 6.1e-4 on the RTX 4060.

The metric of the upstream test (`evo2/test/test_evo2.py`, prompts p0-p3; cross entropy computed in fp32 from
the bf16 logits, see [upstream_test.py](upstream_test.py)):

| Platform | Loss | Accuracy |
|---|---:|---:|
| RTX 4060, official TE FP8 | 0.5024 | 0.7951 |
| H100 PCIe, official TE FP8 | 0.5026 | 0.7953 |
| expected by the upstream test | 0.502 | 0.7956 |

So two GPUs that both run the official FP8 path already disagree on more than half of the bf16 logits; the model's
predictions agree closely. This is the scale against which [the RTX 5070 Ti results](../evo2-rtx-5070-ti/README.md)
are read.

## Files

| File | Content |
|---|---|
| [windows.json](windows.json) | environment, and per window: length, logits sha256, CUDA-event time, rerun bit-identical |
| [brca1.json](brca1.json) | environment, subset size, wall time of both runs, rerun bit-identical |
| [comparison_vs_ada.json](comparison_vs_ada.json) | the comparison above, as computed by `compare_h100.py` |
| [brca1_scores.npz](brca1_scores.npz) | the H100 sequence scores: `rs` (435 reference windows), `vs` (500 variants), `ref_index` |
| [upstream_test.json](upstream_test.json) | the upstream test metric of both official runs, per prompt, by `upstream_test.py` |
| [upstream_test.py](upstream_test.py) | the upstream test metric from saved bf16 logits |
| [cublaslt_fp8_calls.json](cublaslt_fp8_calls.json) | the FP8 GEMMs of the window stage, from the cuBLASLt log |
| [h100_baseline.py](h100_baseline.py) | the baseline script (the host's bundle path replaced by `$BASELINE_ROOT`) |
| [compare_h100.py](compare_h100.py) | the comparison script |

The full H100 logits (2.7 MB) and the cuBLASLt log (0.9 MB) are not committed; their sha256 and summary are above.
