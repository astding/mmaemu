# T2: full truth replay, 2026-10-14

Verdict: **PASS** (`compare.txt`). 173 of 173 recordings were compared with the audit's FixedModel replay.

- Command: `python tools/replay_truth.py --out <dir> --compare $MMAEMU_TRUTH_DIR/replay/replay_fixed.json --threads 12`.
  It ran on the CPU with `CUDA_VISIBLE_DEVICES=""`, torch 2.7.1+cu128 and Python 3.12.2.
- Wall time: 26 min (1568 s).
- Truth: `manifest.json` sha256 `be1270ed64432a1311ab5e2c98ccb69028f72e3424ad4c6148d4cd468fba6695`. The truth tree
  was unchanged after the run.
- Code under test: the run started at commit 6492713 (tree c4b4cb0). The `repo_commit` field in
  `replay_mmaemu.json` says e173ecc, because HEAD was read when the report was written and the branch moved during
  the run. The two commits have identical `mmaemu/` Python sources, `tools/truth_adapter.py` and
  `tools/replay_truth.py`. The only `mmaemu/` change is `mmaemu/backends/cuda/bin/`, which a CPU replay does not load.

## Totals: mmaemu vs FixedModel

| Quantity | mmaemu | FixedModel |
|---|---|---|
| Matching values | 942,445,814 of 942,451,934 | 1,023,913,574 of 1,023,919,694 |
| Zero-sign mismatches | 0 | 0 |
| NaN-payload mismatches | 0 | 0 |
| Value mismatches | 6,120 | 6,120 |
| Hash-only recordings equal | 2,251 / 2,251 | 2,539 / 2,539 |
| Hardware errors reproduced | 113 / 193 | 113 / 193 |
| Not applicable | 2,466 | 255 |

All 6,120 value mismatches are in `emulator.tcemu_kernels_fixed_inputs`, for both models, with equal counts per
output. The adapter's emulator check places every one of them at the old-kernel emulator's NaN-decode outputs
(docs/testing.md), where mmaemu gives the canonical NaN.

FixedModel also produces the outputs that mmaemu reports as not applicable, so its value counts are larger. The 2,211
additional not-applicable outputs are features that mmaemu 0.1 rejects with `UnsupportedFeatureError`
(see docs/limitations.md): fused bias (1,962) and row-wise scales (249). The fastacc cross-check found 0 mismatches
in 512,512 values.
