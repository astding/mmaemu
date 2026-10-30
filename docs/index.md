# mmaemu documentation

## Paper

Xun Ding and Ziyuan Qin. [Accuracy is not compatibility: two Evo 2 checkpoints hinge on one FP8 matrix multiplication](https://arxiv.org/abs/2610.39415).
arXiv:2610.39415 (2026). [PDF](https://arxiv.org/pdf/2610.39415) · [DOI](https://doi.org/10.48550/arXiv.2610.39415).

```bibtex
@misc{ding2026accuracy,
  author        = {Ding, Xun and Qin, Ziyuan},
  title         = {Accuracy is not compatibility: two {Evo} 2 checkpoints hinge on one {FP8} matrix multiplication},
  year          = {2026},
  eprint        = {2610.39415},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2610.39415},
  url           = {https://arxiv.org/abs/2610.39415}
}
```

## Using mmaemu

| Page | What it covers |
|---|---|
| [User guide](user_guide.md) | install and check, a first GEMM, profiles, split-K, backends, using mmaemu inside a model, testing |
| [API reference](api.md) | every public function and class, the exceptions and warnings, the command line |
| [Evo 2 on other GPUs](evo2.md) | why Evo 2 needs FP8 numerics, how to route its FP8 projections through mmaemu, measured results |
| [Evo 2 on Blackwell notebook](https://github.com/astding/mmaemu/blob/master/examples/evo2_blackwell.ipynb) | the runnable version of the above, with its outputs on an RTX 5070 Ti |
| [FAQ](faq.md) | short answers to the questions people ask first |

## Numerics

| Page | What it covers |
|---|---|
| [Semantics](semantics.md) | the numerics contract: the instruction, the accumulation plan, the epilogue, what raises |
| [Presets and profiles](presets_profiles.md) | tensor-core models and reference platforms, their maturity, split-K resolution, environment variables |
| [Split-K tables](splitk_tables.md) | the recorded cuBLASLt heuristic tables (RTX 4060 and H100 PCIe), their coverage and how they are used |

## Implementation

| Page | What it covers |
|---|---|
| [Backends](backends.md) | the CUDA kernels, how they are loaded (fatbin, NVRTC, reference), their gates, platforms |
| [Gated fast kernels](kernels_fast.md) | v9 and tc3: how they use the tensor cores, their gates and their measured evidence |

## Evidence

| Page | What it covers |
|---|---|
| [Validation](validation.md) | the hardware recordings behind every claim, the scope table and the release results, with references |
| [Limitations](limitations.md) | what v0.1 does not do, and corrections of statements made about the prototype |
| [Performance](performance.md) | how speed is measured and the expected kernel throughput |
| [Testing](testing.md) | the test tiers (T0-T7), markers and the bitwise policy |
| [Validation logs](validation_logs/) | the committed logs of every validation run |

## Maintainers

| Page | What it covers |
|---|---|
| [Releasing](release.md) | the release checklist: freeze, distributions, publishing |
| [sm_120 bring-up runbook](runbooks/sm120_bringup.md) | how to collect hardware evidence for a new GPU's FP8 arithmetic |
