# Gated fast kernels: v9 and tc3

mmaemu ships two extra CUDA kernels for the Ada FP8 fast-accumulation chain, besides the IEEE-exact v1 and v4:

| kernel | origin | method | status |
|---|---|---|---|
| `v9` | p2-int-mixed `v9.cu` cfg 1 | block exponent on the **bf16 tensor cores**, products on an FFMA.RZ chain | gated, validated on sm_89 |
| `tc3` | p3-tensorcore-exact `tc_emu3.cu` cfg 6 | whole block on the **fp16 tensor cores** with a dummy accumulator ("design A") | gated, validated on sm_89; SASS for sm_80/86/89 only |

Both kernels are bit-exact only because of how particular tensor cores round. PTX does not specify that behaviour,
and it differs between GPU generations. Their numerics are therefore **hardware-dependent**. The rules are:

- A gated kernel is auto-selected only on a capability listed in `validated_on` (today `{(8, 9)}`, the RTX 4060), or
  when `MMAEMU_ALLOW_GATED_KERNELS=1` is set.
- In both cases the kernel's gate must also pass on the device. Gates probe the exact instruction property the
  kernel relies on, then check the kernel itself at GEMM level.
- When a gate fails, selection falls back to the next kernel with one `BackendFallbackWarning`. The order is
  tc3 > v9 > v4 > v1 > reference.

Neither kernel is release-blocking. If one has not passed acceptance at the release-candidate freeze, its
`KernelSpec` is left out of the release fatbin. For 0.1.0 both passed on the RTX 4060 (T6,
[performance.md](performance.md)).

Files:

- `mmaemu/backends/cuda/csrc/v9.cu` and `v9_common.cuh`
- `mmaemu/backends/cuda/csrc/tc3.cu`
- `mmaemu/backends/cuda/csrc/probes.cu`
- `mmaemu/backends/cuda/fast/specs.py`: the `SPECS` and `PROBE_MODULES` discovery
- `mmaemu/backends/cuda/fast/gates.py`: the `design_a` and `window` gates
- `mmaemu/backends/cuda/fast/data/gate_gemm_v1.npz`
- `tests/test_gates_hmma.py` and `tests/test_fast_kernels.py`

## 1. The kernels

### v9 (KernelSpec "v9", priority 30)

Per 16-element Ada block, v9 works in three steps:

1. **E-pass.** The block exponent comes from a bf16 tensor-core product of powers of two.
   - It runs `mma.sync.m16n8k16 bf16 → f32` on 2^(6 ea − 7) × 2^(6 eb − 7) with c = 2^−86.
   - The sum of at most 17 powers of two whose largest term is 2^(6E−14) lies in [1, 17] × 2^(6E−14). So its fp32
     exponent field is 6E + 113 + d with d ∈ {0..4}, and `pbits64_raw` maps it back to 2^E.
   - This is the **window property**. Every published tensor-core model satisfies it, but it must be checked per
     device.
2. **Products.** They are accumulated by an FFMA.RZ chain on the grid 2^(E−13).
3. **Correction.** A byte-SIMD step turns floor into trunc.

Launch parameters and limits:

- Tile 64 × 64, 128 threads, `__launch_bounds__(128, 2)`.
- Dynamic shared memory is 2 × sizeof(Tile8<64, 64>) = **36,864 B**. That is below the 48 KiB default, so no
  opt-in attribute is needed. The loader checks it against the device limit; the audited code had a hard-coded
  "≤ 99 KB" instead.
- 4-byte alignment.
- `k_max = 1 << 24`, which keeps the block exponent ≤ 55. That is the limit of the byte encodings: `LA_ZERO = 60`
  and the `tt` byte arithmetic.

### tc3 (KernelSpec "tc3", priority 40)

Per 16-element Ada block with block exponent E = max(max product exponent, exponent(acc)), tc3 works as follows:

- **Exponent pass.** It computes X = Σ 2^(5 ea) 2^(5 eb) with a bf16 m16n8k16 (c = +0). A 256-entry table maps
  floor(log2 X) ∈ [5 maxep, 5 maxep + 4] to 2^(maxep + 11), which is again a window property.
- **Product pass.** Each 8-term half goes twice through `mma.sync.m16n8k8 f16 → f32`, with the dummy accumulator
  c = +2^(E+11) and with c = −2^(E+11).
  - On an HMMA with 8-term blocks, 24 fraction bits and c in the alignment (Ampere/Ada), the products are aligned
    to the exponent of c and truncated to the grid 2^(E−13).
  - The result whose c has the opposite sign of the half sum is then exact. tc3 takes
    `s = rp < cp ? rp − cp : rn + cp`.
- **Deferred truncation.** The running value is kept untruncated. Ada's per-block truncation is merged into the
  next block's truncation of c, then applied once at the end.

Launch parameters and limits:

- Tile 128 × 128, 256 threads, `__launch_bounds__(256)`.
- Dynamic shared memory is 2 buffers × 256 rows × 40 halves × 2 arrays × 2 B = **81,920 B**, plus 1 KiB static
  for `s_cpx`.
- That is above 48 KiB, so the loader must set `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` and check the call
  against the device's opt-in limit (99 KiB on sm_86/89, 163 KiB on sm_80).
- **16-byte alignment** of A and B: 16-byte vector loads.
- SASS only for **sm_80, sm_86 and sm_89**, with no PTX.

### Changes against the audited sources

The arithmetic is unchanged. The changes:

- **Device-only.** No torch/c10/CUDA headers, no `TORCH_CHECK`, no host code, one `extern "C"` ABI-v1 entry
  (`mmaemu_v9_e4m3e4m3`, `mmaemu_tc3_e4m3e4m3`).
- **NVRTC needs no CUDA headers.**
  - stdint types are typedef'd under `__CUDACC_RTC__`.
  - `cuda_fp16.h` is replaced by the identical inline PTX of `__half22float2` (v9, two `cvt.f32.f16`) and
    `__hmul2` (tc3, `mul.f16x2`).
  - tc3 sm_89 SASS compared with the audited build: same 3,328 instructions and 175 registers; only the `s_cpx`
    initialisation prologue is scheduled differently.
- **v9 K = 0 guard.** The audited kernel loaded block 0 unconditionally and faulted for K = 0. The guard goes
  through the load bounds (`Mg = nb > 0 ? M : 0`, same for N): no global load happens, the block loop runs 0 times
  and the store writes `final_trunc(+0) = +0`.
  - This keeps the audited code shape: 220 vs 216 registers on sm_89.
  - In an interleaved A/B run (1B shape, CUDA events) it measured 1.547 TMAC/s against 1.549 for the unguarded
    original at M = 2048.
  - Predicating the whole prologue instead cost 2% (1.52 TMAC/s).
- **tc3 and K = 0.** tc3 already guards every load with `gk < K`, so K = 0 writes +0 without reading A or B.
- **Any byte value is safe.** No address depends on the data, with one exception: tc3's `s_cpx[exponent field of X]`.
  X ≥ +0 by construction (non-negative bf16 powers of two, c = +0), so that index stays in [0, 255].
  - NaN codes (0x7F/0xFF) give meaningless numbers but no fault.
  - mmaemu never passes them to a kernel: `execute_plan` sanitises non-finite codes and patches the IEEE result
    (docs/semantics.md).
- **One configuration per kernel.** The other configurations are not shipped.

### k_max of tc3: 1 << 24

- The kernel arithmetic depends on K only through the size of the accumulator.
  - With |a·b| ≤ 448² = 200,704 < 2^17.62, |acc| < K · 2^17.62. So for K < 2^24 its exponent is ≤ 41, and every
    block exponent E that tc3 forms satisfies E ≤ 41.
  - That gives c = 2^(E+11) ≤ 2^52 and grid 2^(E−13) ≥ 2^−25 (the smallest product grid). Both are far inside the
    fp32 range.
  - The two half sums and the truncated accumulator live on the grid 2^(E−13) with magnitude < 2^(E+7), so 20
    significant bits: the fp32 additions are exact for any K.
  - Indices are `size_t`; `nchunk` is an `int`.
- So the bound that matters is the **hardware** one: the dummy-accumulator property must hold for every offset
  E − maxep(half) the kernel can produce.
  - For K < 2^24, E ≤ 41 and maxep ≥ −12, so the offset is ≤ 53.
  - The `design_a` gate's probe domain covers offsets 0..55, capped at E = 43, plus all-zero-product halves with
    E ∈ [−30, 41] ∪ {−127}. −127 is tc3's c for acc = +0.
  - A larger K would need a wider probe domain, so k_max stays at 1 << 24, the same as v4 and v9.
- Measured on hardware so far:
  - K up to 8,240 (verify2: 48×70×8208, 16×777×8240);
  - K = 8,192 with saturating data (p2 cases8k: all-448, |acc| up to 1.43e9 ≈ 2^30.4);
  - K = 1,920 on all 21 Evo 2 1B projections.

## 2. Gates

Both gates use the frozen signature `fn(spec, device) -> GateResult`, carry `.version = "1"`, and are registered
with `register_gate`. Each runs as stages. Every stage fails closed: an unavailable probe or kernel rejects with
`<gate>.error`.

Reasons start with a stable code:

- `design_a.property`, `design_a.xpass`, `design_a.gemm`
- `window.ratio`, `window.offset`, `window.mapping`, `window.gemm`
- `<gate>.error`
- `forced` (from `MMAEMU_TEST_FORCE_FAIL=<gate>`, which the gates honour themselves)

### design_a (tc3)

1. **property** is the full-tile, two-sign dummy-accumulator check.
   - It uses `probes.cu` `mmaemu_probe_hmma_f16_k8`, one `mma.sync.m16n8k8 f16` per 16 × 8 tile. Each of the 128
     outputs carries its own 8-term vector.
   - Every vector runs with +c and with −c, and the gate checks exactly tc3's combine against the exact half sum
     S·u, bitwise.
   - Vectors: 4 data classes × 32 tiles × 128 outputs = **16,384**. Offsets E − maxep ∈ [0, 7] for half of them and [0, 55] for
     the other half (E ≤ 43), plus outputs with no nonzero product.
   - This is the sharpened check of `verify-p3-tensorcore-exact-1` (`scripts/sim_gate_v2.py`).
   - It fixes the false reject at p3 `scripts/selftest.py:58`. That check ran one random sign per vector and
     compared the same-sign half with the Ada truncation model. It therefore rejected an HMMA whose final
     normalisation is RNE, under which tc3 is still exact (`sim_gate_v2.log`: `ada_final_RNE` 3705/4096 under
     the old check, 4096/4096 under the new one, truth_A True).
   - Test: `test_design_a_no_false_reject_rne_final`.
2. **xpass** checks tc3's base-32 exponent pass.
   - It runs bf16 m16n8k16 (`mmaemu_probe_mma_bf16_raw`) on 2^(5 ea) × 2^(5 eb) with c = +0, over the structural
     tiles of the window gate: 974,848 outputs.
   - The kernel's table lookup must return 2^(maxep + 11) for every output.
3. **gemm** runs tc3 on 8 cases, bitwise against stored reference results:
   - 48×40×256 in three classes;
   - 17×9×80 (ragged tiles) in three classes;
   - 130×136×1936 in two classes.

### window (v9)

1. **epass** is the v9 E-pass probe.
   - It uses `mmaemu_probe_epass_raw`, which is v9's own data path: `decode_word` → byte-placed bf16 → ldmatrix →
     m16n8k16 with c = 2^−86. The probe returns the raw fp32 bits.
   - Inputs are the **structural rows**: block exponent E ∈ [−12, 16] × 1..16 terms at E × 4 patterns of the other
     terms (zero, E−1, floor −12, random below) × 32 placements. That is 58,880 rows in 7,360 tiles, with all 128
     row × column pairs evaluated. 256 random tiles are added, for 974,848 outputs.
   - Each output must satisfy d / 2^(6E−14) ∈ [1, 17] (`window.ratio`), offset ∈ {0..4} (`window.offset`) and
     `pbits64_raw(d) == 2^E` (`window.mapping`).
   - This is a port of `verify-p2-int-mixed-3/analyse.py` (hw part) with `vectors.py` `structural()`, rebuilt on
     a self-contained integer generator.
2. **gemm** runs v9 on the same 8 cases.

### Stored GEMM reference results

`data/gate_gemm_v1.npz` is 123,523 B. It holds uint32 bit patterns, a JSON `meta` array and no pickle.

- **Why stored:** the gate compares the kernel with fixed bits and does not run the CPU reference.
- **Inputs** come from a splitmix64 integer generator, so they do not depend on torch's RNG.
  - The file records the sha256 of every input.
  - `gemm_expected()` refuses to run if the generator no longer reproduces them.
- **Provenance check:** `test_gemm_data_equals_reference` (slow marker) recomputes every matrix with
  `mmaemu.model.chain.chain_gemm` and compares bitwise.
- **Regeneration:** `python -m mmaemu.backends.cuda.fast.gates --write-gemm-data [--device cuda]`.

### What the gates are expected to do elsewhere

| capability | v9 / window | tc3 / design_a |
|---|---|---|
| (8, 9) RTX 4060 | **PASS** (measured) | **PASS** (measured) |
| (8, 0), (8, 6) | expected pass [inference: toolbox A100/A2 bf16/fp16 vectors fit the Ada model] | expected pass [same inference]; must pass for tc3 to be used |
| (9, 0), (10, 0) | (9, 0): pass implied by T3 on an H100 (the gates ran in `tests/test_fast_kernels.py`, no xfail; no verdict recorded). (10, 0): expected pass [inference: the published Hopper/B200 bf16 model satisfies the window property; CPU simulation passes] | no tc3 binary; the probe would reject (CPU simulation of the F = 25, 16-term model: `design_a.property`, 2,392/16,384 outputs violate) |
| (12, 0) RTX 5070 Ti | **PASS** (measured: [selftest.json](validation_logs/2026-10-13-sm120/selftest.json), 974,848 E-pass outputs and the 8 GEMM cases exact); v9 stays unselected unless `MMAEMU_ALLOW_GATED_KERNELS=1` until it is added to `validated_on` with evidence | no tc3 binary; the probes module, run there, rejects design A (`design_a.property`, 2,392/16,384 outputs, as the CPU simulation predicts: [design-a.json](validation_logs/2026-10-13-sm120/design-a.json)). Design A cannot work on a 25-bit HMMA |

Simulated models used by `tests/test_gates_hmma.py`. They are the Khattak–Mikaitis block-FMA family, from the port
of `verify-p3-tensorcore-exact-1/scripts/sim_gate.py` `hmma_gen`.

- **design_a**
  - Ada F = 24 / 8-term passes, and so does its RNE-final variant.
  - The paper's Hopper/Blackwell F = 25 / 16-term model is rejected with `design_a.property`, before the GEMM
    check runs.
  - late-c, RD alignment, F25/8, F23/4 and F26/16 are rejected the same way.
- **window**
  - The published Ada and Hopper bf16 models pass.
  - The range-limited mutant (alignment clamped at −60) is rejected with `window.ratio`.

## 3. Measured evidence (RTX 4060, sm_89, driver 572.16, torch 2.7.1+cu128, nvcc 12.8.61)

### How the evidence was produced

- **Binaries:** locally built release-flag fatbins:
  `nvcc -O3 --fmad=false -fatbin -gencode=arch=compute_XX,code=sm_XX`, with no `-lineinfo` and the KernelSpecs'
  `sass_archs` lists. sha256: v9 `bca69d0a…`, tc3 `e8da5876…`, probes `b96703f2…`.
- **Runtime:** every GPU run used the GPU lock (`$GPU timeout 600`).
- **Development stand-in:** the runs were made while the fast kernels were developed apart from the rest of v0.1.
  The interfaces they use (kernel contract, driver bindings, runtime, reference model) came from a development
  stand-in outside the repository.
  - It implements the v0.1 signatures of `spec.py`, `driver.py`, `modules.py` and the registry.
  - Its `chain_gemm` wraps the pre-v0.1 reference model.
  - The registry, verdict-cache and benchmark results below are therefore results against that stand-in. They
    were re-run on the integrated runtime on the RTX 4060 on 2026-10-14: T3
    ([log](validation_logs/2026-10-14/t3-rtx-4060-sm89/README.md)) and T6
    ([log](validation_logs/2026-10-14/t6-rtx-4060-sm89/README.md)). T3 ran again on fa0fd90
    (fast kernels 182 passed, gates 4 passed, the full accumulators of v9 and tc3 equal to the hardware's;
    [log](validation_logs/2026-10-16/rtx-4060-sm89/README.md)) and, with the same counts, after the profile rename
    ([log](validation_logs/2026-10-16/rtx-4060-sm89-renamed/README.md)).
- **ptxas:** 0 spill bytes for every kernel and arch.
  - Registers: v9 218 (sm_80/86), 220 (sm_89/90), 229 (sm_100/120); tc3 175; probes 20–32.
  - NVRTC (libnvrtc 12.8 from the torch wheel) compiles all three sources for every listed arch with no CUDA
    headers.

### Binaries (cuobjdump 12.8 `--list-elf` / `--list-ptx`)

| fatbin | SASS | PTX |
|---|---|---|
| probes | sm_80, sm_86, sm_89, sm_90, sm_100, sm_120 (each contains `HMMA.1688.F32` and `HMMA.16816.F32.BF16`) | none |
| tc3 | sm_80, sm_86, sm_89 | none |
| v9 | sm_80, sm_86, sm_89, sm_90, sm_100, sm_120 | none |

### Gates (`tests/test_gates_hmma.py`, 21 passed)

- **design_a: PASS in 0.27 s.**
  - property: 16,384 outputs × 2 signs, 0 violations;
  - xpass: 974,848 outputs, 0 violations;
  - gemm: 8 cases, 0 mismatches.
- **window: PASS in 0.21 s.**
  - 974,848 outputs, ratio ∈ [1.0, 17.0];
  - offset histogram {0: 413,937, 1: 207,080, 2: 121,945, 3: 195,013, 4: 36,873} (identical to the CPU
    simulation of the Ada bf16 model);
  - gemm: 8 cases, 0 mismatches.
- **Fresh processes** (run_gates: smoke + gate, including data generation): tc3 0.63 s and v9 0.88 s. Under the
  host's load, the third-process re-run took up to 1.46 s.
- **Forced failure:** `MMAEMU_TEST_FORCE_FAIL=design_a` selects `cuda:v9` with exactly one
  `BackendFallbackWarning` over two `select()` calls. With `design_a,window` there are two warnings and neither
  fast kernel is selected. (This was the prototype registry. The integrated registry reports both failed gates in
  one warning and selects v4; see [backends.md](backends.md).)
- **Unvalidated capability:** with `torch.cuda.get_device_capability` monkeypatched to (8, 6), neither fast kernel
  is selected. With `MMAEMU_ALLOW_GATED_KERNELS=1`, tc3 is selected.
- **Verdict cache:** a second process hits it in 0.9 ms (tc3) and 0.4 ms (v9). No gate function ran, and
  `modules.launch` and `DeviceRunner` were patched to raise. After every cached verdict file was corrupted, the
  gates re-ran and passed.

### Conformance (`tests/test_fast_kernels.py`, 92 passed + 2 truth tests skipped without `MMAEMU_TRUTH_DIR`)

0 bit mismatches for v9 and for tc3 against the mmaemu reference (float64 on the GPU):

- **Conformance matrix** (port of the tcemu audit's `verify-a3-integration-review-5/test_kernel.py`): 8 classes ×
  48 shapes, 4,247,040 outputs per kernel.
- **The 77 verify2 cases:** 37,753,738 outputs. The cached references are spot-checked against mmaemu's reference.
- **p2 sets:**
  - check: 22 cases, 231,254 outputs;
  - fuzz: seed 202, 60 cases, 1,425,596 outputs, the same count as the audit's `fuzz_202.json`;
  - cases8k: 6 cases at K = 8192.
- **K = 0:** through the wrapper and through a raw kernel launch with empty buffers, the output is all +0.
- **All-0xFF and all-0x7F inputs:** raw launches at 130×136×1936, 1×1×16 and 64×64×8192 raise no CUDA error, and a
  following GEMM is exact.

### Evo 2 truth (`MMAEMU_TRUTH_DIR=<truth checkout> pytest -m truth`)

Truth `evo2.evo2_1b_w2k`, 21 projections 2048×5760×1920:

- **Accumulator sha256:** 21/21 for v9 and 21/21 for tc3.
- **bf16 rows:** stored `y_native` rows equal for 21/21.
- **Full-output bf16 sha256:** equal for 18/18. The other 3 calls store their full rows, which were compared
  directly.
- **Caveat:** in these runs, the bf16 path was the shared epilogue (`model.epilogue.apply`) on the kernel's
  accumulator, not `scaled_mm(profile="ada-te")`. On the integrated tree, set `MMAEMU_REQUIRE_SCALED_MM=1`
  to require the scaled_mm path.
- No file under the truth directory changed (`find -newer` stamp: 0).

### NVRTC builds

The same gates, conformance (90 tests), 77 cases and truth runs were repeated without the fatbins (NVRTC cubins from
the shipped sources): all bitwise equal. Speed is recorded only: v9 1.557 and tc3 2.995 TMAC/s at 1B M = 2048.

### Performance

Method:

- Shape 1B (K = 1920, N = 5760), gauss40 data.
- CUDA events, warm-up ≥ 1.5 s, three windows ≥ 1.0 s each; the median window is reported.
- Each kernel's full output was checked bitwise against the other kernel and against a strided 64 × 512 reference
  slice before timing.
- The host clock ratio perf_counter / CLOCK_MONOTONIC_RAW was 1.031. It is not used for timing.
- The SM clock was sampled during the windows by nvidia-smi. The clock event reason was 0x4 (SW power cap).

| M | v9 TMAC/s | v9 MAC/clk/SM | tc3 TMAC/s | tc3 MAC/clk/SM |
|---|---|---|---|---|
| 16 | 0.384 | 6.45 | 0.380 | 5.88 |
| 64 | 1.454 | 24.69 | 1.489 | 23.32 |
| 256 | 1.540 | 26.79 | 2.858 | 45.29 |
| 1024 | 1.543 | 26.93 | 2.969 | 48.17 |
| **2048** | **1.536** | 26.56 | **2.957** | 47.99 |
| 8192 | 1.541 | 26.90 | 2.958 | 48.17 |

- **Acceptance (M = 2048):** tc3 ≥ 2.8 and v9 ≥ 1.45. Both met; the audit reported 2.94–3.02 and 1.55–1.58.
- **tc3 below M = 2048:** these are the first such measurements.
  - Small M lacks parallelism: at M = 16 the 128 × 128 tile leaves most of the GPU idle.
  - At M = 16, v9 (64 × 64 tile) is 1% faster than tc3. At M ≥ 64, tc3 is faster.
- **Per-M dispatch:** no lower-priority kernel measured here is ≥ 10% faster than tc3 at any Evo 2 M,
  so no per-M table is needed. v1 and v4 were not in this run; the audit measured them at
  0.13–0.53 and 1.30–1.38 TMAC/s.

## 4. Commands

In the environment of [testing.md](testing.md) (`$GPU` is the GPU lock), from the repository root:

```sh
$GPU timeout 600 python -m pytest -q tests/test_gates_hmma.py                       # gates
$GPU timeout 600 python -m pytest -q tests/test_fast_kernels.py                     # conformance (+ $MMAEMU_AUDIT_DIR sets)
MMAEMU_TRUTH_DIR=<truth checkout> $GPU timeout 600 python -m pytest -q -m truth tests/test_fast_kernels.py   # Evo 2 truth
$GPU timeout 600 python tools/bench_kernels.py --kernels v1,v4,v9,tc3 --shape 1b --M 16,64,256,1024,2048,8192       # performance
cuobjdump --list-elf --list-ptx mmaemu/backends/cuda/bin/{probes,tc3,v9}.fatbin                                   # binaries
python -m mmaemu.backends.cuda.fast.gates --write-gemm-data --device cuda           # regenerate the gate data
```

## 5. Interface assumptions (the integrated runtime passed T3 and T6 with them, 2026-10-14)

These were checked against the drafts of `spec.py` and `driver.py` that existed on 2026-10-12. The others were taken
from the v0.1 interface contract.

1. **Launch arguments.** `modules.launch(module, entry, grid, block, smem, args, device)` passes `args` to
   `driver.launch`, which accepts ctypes values or tensors (the draft: yes). The gates pass `ctypes.c_void_p` and
   `ctypes.c_int`.
2. **Probe module and header.**
   - The runtime and `build_fatbin` find the `probes` module through `PROBE_MODULES`. `fast.specs` also adds it to
     `spec.PROBE_MODULES`.
   - `v9.cu` and `probes.cu` include `"v9_common.cuh"` from `csrc/`. NVRTC needs it in `headers=`, and the
     manifest's `source_sha256` must include it.
3. **tc3 shared memory.** The loader opts in to more than 48 KiB of dynamic shared memory for tc3 (81,920 B),
   checking both the call and the device limit.
4. **Gate version.** `run_gates` reads the gate version with `getattr(fn, "version", ...)`.
5. **Forced failures.** A gate forced to fail with `MMAEMU_TEST_FORCE_FAIL=<gate>` must fail even when a PASS
   verdict for it is cached, and the forced failure must not be written to the cache.
   - `test_forced_gate_failure_falls_back_once` relies on this: it reuses the cache directory of an unforced run.
   - The fast-kernel gates honour the variable themselves, but only when they are called. So `run_gates` must check the
     variable before the cache lookup.
6. **Capability source.** The registry reads the capability with `torch.cuda.get_device_capability`. The test
   monkeypatches that function.
7. **Cache files.** Verdict cache files are plain files under `MMAEMU_CACHE_DIR` other than `*.cubin`, `*.fatbin`
   and `*.ptx`. The corruption test overwrites all such files.
8. **Truth test.** It calls `mmaemu.scaled_mm(..., profile="ada-te", backend="cuda:<kernel>")`. If
   `scaled_mm` is a stub, it uses `model.epilogue.apply`. Set `MMAEMU_REQUIRE_SCALED_MM=1` to require the
   scaled_mm path.
