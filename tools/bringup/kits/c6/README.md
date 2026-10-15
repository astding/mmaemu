# Must-measure kit for FP8 tensor-core models (c6 audit)

The vectors were selected so that the candidate models of each target give different bitwise predictions. Measuring
them on the target GPU therefore identifies the model up to "pool-equivalence". The models are described in
`family.py` (a copy of `../family.py`).

Pool-equivalences that are provable, and therefore not measurement gaps:
- `min_exp` (-132 vs -133) and the exponent used for a subnormal c. Any nonzero FP8 product has exponent >= -12 (E4M3)
  or >= -28 (E5M2), so these act only when all products are 0 and |c| < 2^-126; the output truncation at
  2^(-126-frac_out) then gives the same result.
- `in_sub` for E5M2: E5M2 subnormals are also fp16 subnormals.
- `in_sub` for E4M3 when c is added late and frac >= 21: no product bit is ever cut.
- Group order when the groups are combined by one fp32 addition, which is commutative.

## Sets

| set | target | instruction path | candidates | vectors (E4M3 / E5M2) | classes (E4M3 / E5M2) |
|---|---|---|---|---|---|
| `MM_sm120` | RTX 5070 Ti / RTX 50xx / RTX PRO 6000 (sm_120) | `mma.sync.m16n8k32`, which lowers to QMMA.16832 | full core grid (9,720) + presets × secondary axes + all B200 survivors = 12,881 | 951 / 928 | 6,897 / 7,670 |
| `MM_cublas64_sm120` | the same, at GEMM level | cuBLAS `torch._scaled_mm`, diagonal embedding (see below) | same 12,881 | 834 / 897 | 6,758 / 7,661 |
| `MM_sm90mma` | GH200 / H100 (sm_90) | `mma.sync`, which lowers to F2FP + 2×HMMA.16816 + FADD (the same SASS as B200) | the 3,136 models that fit the toolbox B200 vectors, plus Fig. 5c variants | 488 / 461 | 137 / 137 |
| `MM_cublas64` | GH200 / H100, at GEMM level | cuBLAS (wgmma / QGMMA) | Hopper accumulator-input variants + presets = 133 | 445 / 425 | 21 / 13 |
| `MM_ada` | RTX 4060 (sm_89), already measured | QMMA.16832 | the 16 Ada survivors | 434 / 413 | 4 / 2 |
| `TBX_B200` | any GPU | `mma.sync` | the 6 named models × secondary axes + Ada/Hopper/B200 survivors = 3,296 | 5000 / 5000 (the toolbox's B200 inputs and c) | — |
| `TBX_H100c0` | any GPU | `mma.sync` | the same 3,296 | 5000 / 5000 (the toolbox's H100 inputs, c = 0) | — |

The `TBX_*` sets contain the toolbox recordings as `d_<SET>_recorded_fp32.txt`, and `run_kit.py` reports
`equals_recorded_bitwise`:
- On a GH200, `TBX_B200` = 5000 is direct evidence that the B200 vectors are the sm_90/sm_100 HMMA path.
  `TBX_H100c0` should *not* equal the H100 recording, which was made with wgmma.
- On sm_120, `TBX_B200` = 5000 means "behaves like the B200 `mma.sync` path"; about 3207 means "like `B200TC.m`
  (Fig. 5e, c early)". These sets are weakly discriminating: 3137 of 3139 B200-consistent candidates predict
  exactly what `blackwell` predicts on all ten toolbox sets. Use `MM_sm120` for identification.

## File format

The files use the same layout as the toolbox and as `/data/quant/tcemu/tests/data`:

- `sets/<SET>/<FMT>/a_<SET>_<FMT>.txt`: one vector per line, with 32 fp32 words in hex. The values are exactly representable in FP8 `<FMT>`.
- `b_<SET>_<FMT>.txt`: the same format.
- `c_<SET>_fp32.txt`: one 32-bit binary string per line, holding the fp32 accumulator input c.
- `tags_<SET>_<FMT>.txt`: the probe name for each line.
  - `P1`…`P9` are the canonical probes (see `../pool.py`, `canonical()`):
    P1/P1b group membership (2^12 + 2^-12 + 2^-12 at positions 0, 1, j); P2 cancellation / block order;
    P3 c early vs late; P4 c only (incl. subnormal, huge, ±0); P5 signed zeros; P6 subnormal input sets the
    alignment exponent; P7 maximum carry; P8 direction of negative truncation; P9 only tiny products.
  - `R_<kind>` are structured random vectors.
- `pred_<SET>_<FMT>.npz` has two entries:
  - `bits`: uint32 `[n_models, n_vectors]`, the predicted fp32 bit patterns, with NaN canonicalised to `0x7FC00000`.
  - `models`: a JSON list of model specs (see `family.py`).
- `classes_<SET>_<FMT>.json`: the pool-equivalence classes, given as indices into `models`.
- `cgen_<SET>_<FMT>.txt`: only in the `MM_cublas64*` sets. Each line holds two hex fp32 words `cx cy` (FP8 values) with `c = cx*cy` exactly.
- `d_<SET>_<probe>.txt`: written by the runners. It holds the measured fp32 bits, one binary string per line.

Each vector computes `d = sum_k a[k]*b[k] + c` (k = 0..31). For the `mma.sync` probe this is element (0,0) of
one `m16n8k32` MMA, with the toolbox kernel's operand layout. `E5M2` sets use e5m2 for both operands. The set names
are not `H100`/`H200`, so `/data/quant/tcemu/bench/mma_vectors.py` would not zero c for them. However, that script
hardcodes its data root and arch list, so use the runners below instead.

## How to run

Build outside any GPU lock first. Each command writes a JSON file that lists every candidate model with 100%
bitwise agreement and the score of each tcemu preset.

```
# RTX 5070 Ti (sm_120)
python run_kit.py --sets MM_sm120 MM_ada --arch 12.0 --compile-only
python run_kit.py --sets MM_sm120 MM_ada TBX_B200 TBX_H100c0 --arch 12.0 --out result_5070ti_mma.json
python run_kit.py --sets MM_sm120 --arch 12.0a --kind --out result_5070ti_mma_kind.json   # kind::f8f6f4 form
python run_cublas_kit.py --set MM_cublas64_sm120 --out result_5070ti_cublas.json

# GH200 (sm_90)
python run_kit.py --sets MM_sm90mma TBX_B200 TBX_H100c0 --arch 9.0a --out result_gh200_mma.json  # HMMA path: is it Fig. 5c?
python run_cublas_kit.py --set MM_cublas64 --out result_gh200_cublas.json       # QGMMA path: how c enters
```

`run_cublas_kit.py` embeds each test vector in a GEMM with K = 64:

- Row i of A is `[cx_i, 0 ×31, a_i]` and row i of B is `[cy_i, 0 ×31, b_i]`.
- `D[i,i] = MMA(a_i, b_i, c_i)`, with `c_i` produced by the first K-slice.
- It runs M = N = 1024 and 4096, each with `use_fast_accum` True and False.
- Only E4M3 can be run this way, because torch rejects e5m2 × e5m2.

On the RTX 4060 this embedding reproduces the `ada` preset on 445/445 vectors in all four runs
(`../results/kit_MM_cublas64_rtx4060.json`).

## Self-test on the RTX 4060 (sm_89, driver 572.16, torch 2.7.1+cu128)

| run | `ada` preset | models at 100% | result file |
|---|---|---|---|
| `MM_ada` | 434/434 and 413/413 | the `ada` class only | `../results/kit_MM_ada_rtx4060.json` |
| `MM_sm120` | 951/951 and 928/928 | 4 of 12,881 (E4M3) and 8 (E5M2), all in the `ada` class; the other classes differ only on axes that provably cannot be observed | `../results/kit_MM_sm120_sm90mma_on_rtx4060.json` |
| `MM_sm90mma` | not a candidate | 0, as expected: Ada has no HMMA path | same file |
