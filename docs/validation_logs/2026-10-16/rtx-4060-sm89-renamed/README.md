# RTX 4060 (sm_89) after the profile rename, 2026-10-16

Verdict: every step of `final_f2_4060.sh` exits 0 (`run.txt`, 23:11 to 23:43 PDT): the preflight (18 of 18), T3
(every per-file run, `t3/summary.txt`), the four test files of the hopper preset's kernels, the full accumulators of
every GPU kernel, T3p (6 passed), the three split-K envelope sweeps, the GPU tier of T4, the bench, and the replay of
the H100 PCIe probe through h4 and h1. Every result equals that of the same script on fa0fd90
([../rtx-4060-sm89/README.md](../rtx-4060-sm89/README.md)) once the old profile names are read as the new ones
(below); only timings, paths and the commit differ.

- Hardware and software: the RTX 4060 host of
  [../../2026-10-14/rtx4060-host/README.md](../../2026-10-14/rtx4060-host/README.md). Preflight 18 of 18 PASS
  (`preflight.out`, `preflight/preflight.json`): driver 570.211.01, Python 3.12.2, torch 2.7.1+cu128, cuBLASLt
  12.8.3.14, the 32 packages of the reference lock, fatbins current and equal to the manifest's sha256, profile `ada`
  matches the device. The expected commit of `preflight_4060.py` was set to 2648bba.
- Code under test: 2648bba, a git checkout; `mmaemu/`, `tools/` and `tests/` unmodified before and after the run
  (`run.txt`). Its code is that of 63b7395, which renamed the profiles (`hopper-h100pcie-torch` -> `hopper`,
  `hopper-h100pcie-te` -> `hopper-te`, `hopper-h100pcie-ws0` -> `hopper-ws0`, `ada-rtx4060-torch` -> `ada`,
  `ada-rtx4060-te` -> `ada-te`, `ada-rtx4060-ws0` -> `ada-ws0`) and changed their descriptions; 2648bba adds logs
  only. Its fatbins are those of fa0fd90 ([../rc-fatbin/README.md](../rc-fatbin/README.md)).
- Script: `final_f2_4060.sh`, the `final_4060.sh` of fa0fd90 writing to `logs-f2/` and running the sweep scripts
  with the new profile names (`sweeps/name_mapping.diff`). One step after another, every GPU step under the GPU lock,
  with the ground truth and the audit data present. Paths: `~/` is the host's home directory.

## Equal to the run on fa0fd90

Compared after replacing the old profile names by the new ones and dropping timings, paths and commits:

| Output | Result |
|---|---|
| `t3/summary.txt` | the same passed, skipped and deselected counts in every file (`test_backend_selection` exits 5 in both: a CPU-only file, nothing collected under `-m gpu`) |
| `t3/di_full_gpu_kernels.log` | 84 passed: the 21 Evo 2 calls of v1, v4, v9 and tc3 bit for bit equal to the RTX 4060's recorded accumulators |
| `hopper-kernels/summary.txt` | `test_kernels_ieee` 167, `test_cuda_runtime` 19, `test_h100_mini` 435, `test_backend_selection` 26 passed |
| `replay-h4/mmaemu_replay.json` (`--backend auto`) | every record equal: all 1,423 live calls equal the H100 (1,049 split, 374 unsplit; 1,043 on h4, 380 mixed-format on the reference backend), the 27 rejected calls raise `ReferenceRejectsError`, 168 of 168 Evo 2 outputs on h4 equal; `"PASS": true` |
| `replay-h1/mmaemu_replay.json` (`--backend cuda:h1`) | every record equal: the 1,043 E4M3 x E4M3 live calls and the 168 Evo 2 outputs on h1 equal; the 380 mixed-format calls raise `BackendUnavailableError`, so `"PASS": false` by construction, as on fa0fd90 |
| `t3p_profile_device.log` | 6 passed, the same six tests |
| `sweeps/` | the same query, split and max-f counts (306,859, 663,936 and 666,624 queries); the per-query row files (not kept, 127 MB) are byte-identical to those of fa0fd90 once its profile names are mapped |
| `t4/mutants-gpu-rtx4060.json` | PASS: the 7 kernel mutants killed, the identity control survives, each with its earlier outcome |
| `mmaemu_info.txt` | the same lines apart from the profile names; h1 and h4 "validated here" |

## Bench (`bench-*-gauss40.json`, `.log`)

`tools/bench_kernels.py --kernels v1,v4,v9,tc3,h1,h4 --shape 1b|7b|40b --M 2048 --data gauss40`, every kernel from the
release fatbins, 0 mismatches in every exactness check. TMAC/s, the median of 5 windows, within 0.01 of the run on
fa0fd90:

| Kernel | 1B (K 1920, N 5760) | 7B (K 4096, N 12288) | 40B (K 8192, N 24576) |
|---|---:|---:|---:|
| tc3 | 2.94 | 2.95 | 2.94 |
| v9 | 1.54 | 1.55 | 1.55 |
| h4 | 1.34 | 1.40 | 1.41 |
| v4 | 1.34 | 1.36 | 1.37 |
| h1 | 0.69 | 0.69 | 0.69 |
| v1 | 0.63 | 0.62 | 0.63 |
