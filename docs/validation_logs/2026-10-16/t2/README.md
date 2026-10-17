# T2: full truth replay on fa0fd90, 2026-10-16

Verdict: **PASS** (`compare.txt`). `compare.json` equals that of 2026-10-15
([../../2026-10-15/t2/README.md](../../2026-10-15/t2/README.md), which gives the counts) in every field: 173 of 173
recordings in the report; 942,445,814 of 942,451,934 values match; 0 zero-sign and 0 NaN-payload mismatches; the same
6,120 value mismatches at the old emulator recording's NaN-decode outputs; hash-only 2,251 of 2,251; hardware errors
113 of 193 reproduced; 2,466 not applicable; fastacc cross-check 0 mismatches in 512,512 values.

- Code under test: fa0fd90 (a clean git worktree; `repo_commit` in `replay_mmaemu.json`); 63b7395, later, renamed the
  profiles; T2 on its code gave the same `compare.json`
  ([../rtx-5070-ti-renamed/README.md](../rtx-5070-ti-renamed/README.md)). On fa0fd90 the adapter
  (`tools/truth_adapter.py`) replays the RTX 4060 recordings under the profile `ada-rtx4060-torch`, which it names
  explicitly; from 63b7395 on it names `ada`.
- Command: `python tools/replay_truth.py --out <out> --compare $MMAEMU_TRUTH_DIR/replay/replay_fixed.json
  --threads 12`, on the CPU with `CUDA_VISIBLE_DEVICES=""`, torch 2.7.1+cpu, numpy 2.5.4 and Python 3.12.2, as on
  2026-10-15. `<src>` stands for the worktree and `<out>` for the output directory.
- Wall time: 67 min (4011 s), with other CPU work on the host; 30 min (1772 s) on 7612fd8 (below).
- Truth: `manifest.json` sha256 `be1270ed64432a1311ab5e2c98ccb69028f72e3424ad4c6148d4cd468fba6695`, unchanged after
  the run.

`replay_mmaemu.json` differs from that of 2026-10-15 only in the run metadata (start time, durations, paths, version
string `0.1.0` instead of `0.1.0.dev0`, commit) and in the seconds of each recording; every count and verdict is
equal. The same replay ran on 7612fd8 (the release-candidate freeze) earlier the same day, with the same
`compare.json` and every count and verdict of `replay_mmaemu.json` equal.
