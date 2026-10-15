# T4 mutation matrix, 2026-10-14

| Report | Tier | Device | Code under test |
|---|---|---|---|
| `mutants-cpu.json`, `.txt` | CPU (T0 + T1) | none | 5f1a55c (`repo_commit`) |
| `mutants-gpu.json`, `.txt` | GPU | RTX 5070 Ti (sm_120) | 5f1a55c (`repo_commit`) |
| `mutants-gpu-rtx4060.json`, `.txt` | GPU | RTX 4060 (sm_89) | da9edd4, run from a `git archive` (no `.git`, so `repo_commit` is empty; see `../rtx4060-host/README.md`) |

Between 5f1a55c and da9edd4, `mmaemu/` changes in three places only: `selftest()` also accepts a sequence of kernel
names (the CLI passes a tuple), a comment on `validated_on`, and a path in the split-K table's metadata
(`$MMAEMU_TRUTH_DIR` instead of a local path). No kernel source changed.

The GPU tier runs `tools/mutants/test_gpu.py` and the T3 files on NVRTC builds of the mutated sources; the kernel
mutants edit `v1.cu` and code shared by all kernels (docs/testing.md, T4).
