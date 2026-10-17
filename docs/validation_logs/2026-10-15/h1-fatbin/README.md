# h1 fatbin build, 2026-10-15

The build log of `mmaemu/backends/cuda/bin/h1.fatbin`, the CUDA kernel of the `hopper` preset (32-term blocks;
`mmaemu/backends/cuda/csrc/h1.cu`). It was written by
`python tools/build_fatbin.py --release --modules h1 --out <dir>` with nvcc 12.8.61 and the release flags
(`-O3 --fmad=false`), as in [docs/release.md](../../../release.md), section 2. Only this module was built. Its
manifest entry was added to `mmaemu/backends/cuda/bin/manifest.json`, and the entries of the five frozen release
modules (v1, v4, v9, tc3, probes; [their build](../../2026-10-14/fatbin-release/README.md)) are unchanged:
`tools/build_fatbin.py --check` passes for all six. A second build gave a byte-identical fatbin (sha256
`261a7605…`).

Commit 3d7dc66 changed only a comment in `h1.cu` (the unroll note now states the register counts below). The
rebuild was byte-identical (sha256 `261a7605…`, same build report and cuobjdump listing), so only the manifest's
`source_sha256` for h1 changed (`1b5e42b9…`).

- [build_report.json](build_report.json): the nvcc command (SASS sm_75 to sm_120, PTX compute_75) and the ptxas
  register, stack and spill counts of every architecture: 63 to 96 registers and 0 spill bytes everywhere. The
  product loop is unrolled 16-way, because a full 32-way unroll spilled 16 bytes on sm_80. There are no compiler
  warnings and no forbidden PTX.
- [cuobjdump.txt](cuobjdump.txt): `cuobjdump --list-elf --list-ptx` of the fatbin. The architectures equal the
  requested ones.
