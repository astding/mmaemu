# Release fatbin build, 2026-10-14

The build log of the fatbins in `mmaemu/backends/cuda/bin/` (commit e173ecc), written by
`python tools/build_fatbin.py --release --out build/fatbin-release` with nvcc 12.8.61, as in
[docs/release.md](../../../release.md), section 2.

- [build_report.json](build_report.json): the nvcc version and, per module (v1, v4, v9, tc3, probes), the nvcc
  command (`-O3 --fmad=false` and the requested architectures), the ptxas register, stack and spill counts of
  every entry point (0 spill bytes in every module) and the compiler warnings (none).
- [cuobjdump.txt](cuobjdump.txt): `cuobjdump --list-elf --list-ptx` of each fatbin; the architectures equal the
  requested ones.
