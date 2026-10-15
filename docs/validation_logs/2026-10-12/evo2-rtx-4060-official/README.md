# Evo 2 on the RTX 4060: the official-path reference outputs, 11-12 October 2026

Before the RTX 4060 was replaced by the RTX 5070 Ti, the official Evo 2 path (evo2 with Transformer Engine FP8) ran
on it. Every Evo 2 comparison in this repository is against these captures:

- `examples/data/evo2_1b_ada_rtx4060_te.npz` and `.json`, used by the notebook and the README's Evo 2 1B results;
- `rtx4060_official_te_40b.npz` in the [Evo 2 40B log](../../2026-10-15/evo2-40b-rtx-5070-ti/README.md).

Hardware and software: NVIDIA GeForce RTX 4060 (sm_89, 24 SMs, 8 GB), driver 572.16, WSL2; Python 3.12.2,
torch 2.7.1+cu128, cuBLAS/cuBLASLt 12.8.3.14, Transformer Engine 2.3.0, evo2 0.6.0, vtx (vortex) 1.1.0,
flash-attn 2.8.0.post2 (each capture's `.json` in `data/` records them).

## Where each shipped array comes from

| Shipped | Produced by | Run |
|---|---|---|
| `evo2_1b_ada_rtx4060_te.npz`: `<w>/ids`, `<w>/acgt`, `<w>/next` (p0-p3, w8k, w2k) | [golden_real.py](scripts/golden_real.py): `evo2.Evo2("evo2_1b_base")`, the real upstream model, through [run_te.sh](scripts/run_te.sh) | 12 Oct, 00:21; [data/evo2_1b_base_real-te-fp8.json](data/evo2_1b_base_real-te-fp8.json) |
| `brca1/rs`, `brca1/vs` (435 reference and 500 variant windows) | [brca1_source_script.py](scripts/brca1_source_script.py), its first pass (`run("TE_FP8")`: upstream `score_sequences` with TE FP8), cached as `brca1_subset500_te_scores.npz` | 11 Oct, 09:34-09:56; [logs/brca1_run2.log](logs/brca1_run2.log), AUROC 0.7378 |
| `brca1/pos`, `brca1/ref`, `brca1/alt`, `brca1/lof` | the upstream BRCA1 notebook's table (Findlay et al. 2018), `sample(n=500, random_state=0)`, as in the script | |
| `.json`: `windows` (bp, mean log p(next), SHA-256 of the full logits) | the same capture as the arrays | [data/evo2_1b_base_real-te-fp8.json](data/evo2_1b_base_real-te-fp8.json) |
| `.json`: `projection0_w2k` | the hardware recording `evo2.evo2_1b_w2k` | [docs/validation.md](../../../validation.md) |
| `.json`: `h100_official_te` | the H100 run of the official path | [2026-10-14/evo2-h100-sm90](../../2026-10-14/evo2-h100-sm90/README.md) |
| 40B log: `rtx4060_official_te_40b.npz` (w2k, w4k: `acgt`, `next`) | [golden_oracle.py](scripts/golden_oracle.py) `--model evo2_40b` (see below) | w2k: 12 Oct, 00:26-00:31, [logs/40b_w2k.log](logs/40b_w2k.log); w4k: 00:51-00:54 by [chain_40b.sh](scripts/chain_40b.sh), [logs/40b_w4k.log](logs/40b_w4k.log), [logs/chain_40b.log](logs/chain_40b.log) |

The two shipped `.npz` files were assembled from these captures; the assembly was not kept as a script.
[check_reference.py](check_reference.py) checks every shipped array against its capture, and the SHA-256 of every
capture's full logits against its `.json`: 0 failures ([check_reference.txt](check_reference.txt)). The captures
with the full bf16 logits (3.5 MB for the 1B) are not in the repository; their `.json` files here record the
SHA-256 of each window's logits.

The windows: p0-p3 are the four prompts of `evo2/test/data/prompts.csv`, w8k the first 8,192 bp of their
concatenation, w2k and w4k its first 2,048 and 4,096 bp. They are defined by `windows()` of
[bench.py](scripts/bench.py) (evo2-anywhere, commit 5d6a1d9, which [common.py](scripts/common.py) imports), and
`acgt` and `next` are computed by its `summarize()`: `log_softmax` of the bf16 logits in fp32. The rest of
`bench.py` runs evo2-anywhere's own models and was not used.

## The 40B reference: the official blocks, one at a time

The official 40B path does not fit in 8 GB. [golden_oracle.py](scripts/golden_oracle.py) runs the upstream model one
block at a time: each block is built by vortex's `get_block` and loaded by vortex's own
`StripedHyena.to_bfloat16_except_pr_lc` and `custom_load_state_dict` (the TE FP8 recipe, the Wqkv permutation),
and no other code touches the weights. Run the same way on the 1B, it gives the logits of the real upstream model
bit for bit on all six windows ([check_reference.txt](check_reference.txt); the two captures,
[data/evo2_1b_base_real-te-fp8.json](data/evo2_1b_base_real-te-fp8.json) and
[data/evo2_1b_base_oracle-te-fp8_p0-p1-p2-p3-w8k-w2k.json](data/evo2_1b_base_oracle-te-fp8_p0-p1-p2-p3-w8k-w2k.json),
record the same logits SHA-256). For the 40B, only w2k and w4k fit in 8 GB; p0-p3 and w8k ran out of memory
([logs/chain_40b.log](logs/chain_40b.log)). The same w2k and w4k arrays had been produced on 11 October by an earlier
block-by-block runner in evo2-anywhere, bit for bit.

## Files

- `scripts/`: the capture scripts as run, with their paths on the capture host: [common.py](scripts/common.py),
  [golden_real.py](scripts/golden_real.py), [golden_oracle.py](scripts/golden_oracle.py),
  [run_te.sh](scripts/run_te.sh) (the TE environment, with a cuBLASLt trace of every call),
  [chain_40b.sh](scripts/chain_40b.sh), [compare_goldens.py](scripts/compare_goldens.py),
  [brca1_source_script.py](scripts/brca1_source_script.py), [bench.py](scripts/bench.py).
  `/data/quant/tcemu_audit/gpu.sh`, called by `chain_40b.sh`, only takes a lock on the GPU (`flock`).
- `data/`: the metadata of the captures used (windows, logits SHA-256, versions, block hashes).
- `logs/`: the 40B runs and the BRCA1 run. The owner column of the `ls` listing in `chain_40b.log` is redacted.
- [check_reference.py](check_reference.py), [check_reference.txt](check_reference.txt): the check above, run on
  15 October 2026 with `--goldens <captures> --xlsx 41586_2018_461_MOESM3_ESM.xlsx`.

Not used by anything in this repository: the second pass of `brca1_source_script.py` (a lookup-table experiment;
its result is also in the log), the 4 kb prefixes of p0-p3 and the cuBLASLt heuristic table that `chain_40b.sh`
also produced. The first attempt at the BRCA1 run (10 October) stopped with a CUDA error in that second pass,
before the scores were saved, so the BRCA1 scores were recorded on 11 October, not on 10 October as the BRCA1
capture's `.json` and, until 15 October, `examples/data/evo2_1b_ada_rtx4060_te.json` said.
