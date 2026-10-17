# T4 mutation matrix on fa0fd90, 2026-10-16

| Report | Tier | Device | Code under test | Verdict |
|---|---|---|---|---|
| `mutants-cpu.json`, `.txt` | CPU (T0 + T1) | none | fa0fd90 (`repo_commit`) | PASS: 43 killed, 2 equivalent, the 5 identity controls survive |
| `mutants-gpu.json`, `.txt` | GPU | RTX 5070 Ti (sm_120) | fa0fd90 | PASS: 7 killed, the identity control survives |
| `mutants-gpu-rtx4060.json`, `.txt` | GPU | RTX 4060 (sm_89) | fa0fd90 (see `../rtx-4060-sm89/README.md`) | PASS: 7 killed, the identity control survives |
| `mutants-cpu-7612fd8.json`, `.txt` | CPU (T0 + T1) | none | 7612fd8 | FAIL: `identity_execute` killed (below) |

63b7395, later, renamed the profiles; on its code every tier gave the same outcomes
([../rtx-5070-ti-renamed/README.md](../rtx-5070-ti-renamed/README.md),
[../rtx-4060-sm89-renamed/README.md](../rtx-4060-sm89-renamed/README.md)). Every mutant has the outcome it had on
2026-10-14 ([../../2026-10-14/t4/README.md](../../2026-10-14/t4/README.md)), and on 7612fd8 (GPU tiers) and 3fe4e9c
(CPU tier) earlier the same day.

The CPU tier on 7612fd8 killed the identity control `identity_execute` with
`tests/test_h100_mini.py::test_case[m6_040_adv_split]` (K 8192, S 3, `"contiguous"`, slow accumulation): the
mutation tool's own re-implementation of `execute_plan` (`tools/mutants/impl.py`) combined every scheme but
`"serial"` in the pairwise order, while mmaemu combines `"contiguous"` splits serially, and it had no +0 partial for a
contiguous split without k-tiles. The identity control did what it is for: it showed that the injection was unsound
for the hopper preset, whose tests this tier had not run before. 3fe4e9c fixes the tool and changes nothing under
`mmaemu/` or `tests/`; the CPU tier passes from 3fe4e9c on. The GPU tier does not use this re-implementation (its
mutants edit the kernel sources).

Paths: `<src>` is the worktree, `<out>` the scratch output directory, `~/` the RTX 4060 host's home directory.
