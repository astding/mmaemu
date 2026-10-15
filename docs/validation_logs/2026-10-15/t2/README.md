# T2: full truth replay on the reviewed tree, 2026-10-15

Verdict: **PASS** (`compare.txt`), with every count equal to the run of 2026-10-14
([../../2026-10-14/t2/README.md](../../2026-10-14/t2/README.md), which explains the totals): 173 of 173 recordings in
the report; 942,445,814 of 942,451,934 values match; 0 zero-sign and 0 NaN-payload mismatches; the same 6,120 value
mismatches at the old emulator recording's NaN-decode outputs; hash-only 2,251 of 2,251; hardware errors 113 of 193
reproduced; 2,466 not applicable; fastacc cross-check 0 mismatches in 512,512 values.

- Code under test: the tree of 49f1ecf (`git archive`, so `repo_commit` in `replay_mmaemu.json` is empty). In the
  reports `<src>` stands for that tree and `<out>` for the output directory.
- Command: `python tools/replay_truth.py --out <out> --compare $MMAEMU_TRUTH_DIR/replay/replay_fixed.json
  --threads 12`, on the CPU with `CUDA_VISIBLE_DEVICES=""`, torch 2.7.1+cpu and Python 3.12.2 (the run of 2026-10-14
  used torch 2.7.1+cu128 with no visible GPU).
- Wall time: 26 min (1547 s).
- Truth: `manifest.json` sha256 `be1270ed64432a1311ab5e2c98ccb69028f72e3424ad4c6148d4cd468fba6695`, unchanged after
  the run.

Compared field by field with the reports of 2026-10-14, `compare.json` and `replay_mmaemu.json` differ only in the
message of the `UnsupportedFeatureError` for a fused bias, which 49f1ecf rewords (it appears in the not-applicable
reasons), and in the run metadata (start time, durations, paths, torch build, commit).
