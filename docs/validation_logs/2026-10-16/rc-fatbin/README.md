# Release-candidate fatbin rebuild, 2026-10-16

At the freeze of the 0.1.0 release candidate with the H100 PCIe as the default platform, all seven modules were
rebuilt from the committed sources with `python tools/build_fatbin.py --release` and nvcc 12.8.61, as in
[docs/release.md](../../../release.md), section 2, and compared byte for byte with the fatbins in
`mmaemu/backends/cuda/bin/`. The only source change since the last build is a comment in `h1.cu` (the bounds of
a block's terms and partial sums).

| Module | Rebuilt bytes | Kept in `bin/` |
|---|---|---|
| h1, h4, v9, tc3, probes | identical to the committed fatbins | unchanged |
| v1, v4 | differ; the SASS of every architecture (sm_75 to sm_120) is identical and the files have the same size ([v1_v4_sass_compare.txt](v1_v4_sass_compare.txt)); the PTX differs only in the hash nvcc puts into the names of anonymous-namespace symbols (`_INTERNAL_<hash>_5_v1_cu_...`) | the committed bytes (the fatbins of rc1/v0.1.0, which T3 ran on the RTX 4060, the RTX 5070 Ti and the H100 PCIe) |

`mmaemu/backends/cuda/bin/manifest.json` changes only in h1's `source_sha256` (the comment). Afterwards
`tools/build_fatbin.py --check --require-manifest` reports the seven modules current and
`tools/check_dist.py --verify-manifest` passes.

- [build_report.json](build_report.json): per module the nvcc command (`-O3 --fmad=false`, the requested
  architectures), the ptxas register, stack and spill counts of every entry point (0 spill bytes in every module),
  the forbidden-PTX check (none) and the compiler warnings (none).
- [cuobjdump.txt](cuobjdump.txt): `cuobjdump --list-elf --list-ptx` of each rebuilt fatbin; the SASS architectures
  equal every KernelSpec's `sass_archs` (h1, v1: sm_75 to sm_120 with PTX 7.5; h4, v4: sm_80 to sm_120 with PTX 8.0;
  v9, probes: sm_80 to sm_120; tc3: sm_80, sm_86, sm_89).
- Paths: `~/` is the build host's home directory, `<venv>` the venv that provided cuobjdump, `<build>` the scratch
  build directory.
