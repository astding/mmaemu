# h4 fatbin build, 2026-10-16

The build log of `mmaemu/backends/cuda/bin/h4.fatbin`, the fast CUDA kernel of the `hopper` preset (kernel v4's
arithmetic with 32-term blocks; `mmaemu/backends/cuda/csrc/h4.cu`). It was written by
`python tools/build_fatbin.py --release --modules h4 --out <dir>` with nvcc 12.8.61 and the release flags
(`-O3 --fmad=false`), as in [docs/release.md](../../../release.md), section 2. Only this module was built. Its
manifest entry was added to `mmaemu/backends/cuda/bin/manifest.json`; the entries of the five frozen release modules
(v1, v4, v9, tc3, probes; [their build](../../2026-10-14/fatbin-release/README.md)) and of h1
([its build](../../2026-10-15/h1-fatbin/README.md)) are unchanged: `tools/build_fatbin.py --check` passes for all
seven. A second build gave a byte-identical fatbin (sha256 `7e1f96c3…`) and the same build report.

- [build_report.json](build_report.json): the nvcc command (SASS sm_80 to sm_120, PTX compute_80; like v4, h4 needs
  the bf16x2 FMA of sm_80) and the ptxas register, stack and spill counts of every architecture: 248 to 255
  registers (`__launch_bounds__(256, 1)`) and 0 spill bytes everywhere. There are no compiler warnings and no
  forbidden PTX.
- [cuobjdump.txt](cuobjdump.txt): `cuobjdump --list-elf --list-ptx` of the fatbin. The architectures equal the
  requested ones.
