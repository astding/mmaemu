# Probes of torch._scaled_mm on the RTX 4060, 2026-10-14

These probes, run for the review of the release candidate (2026-10-14), settle how torch 2.7.1 calls cuBLASLt in
cases that mmaemu at 8429385 got wrong or left open. The labels `a01`, `a09` (JSON keys) and A-01 to A-09 (script
comments) are that review's internal finding numbers. They ran on the RTX 4060 host of
[../rtx4060-host/README.md](../rtx4060-host/README.md): driver 570.211.01, torch 2.7.1+cu128, cuBLASLt 12.8.3.14.
mmaemu was compared offline on the CPU with the reference backend.

| File | What |
|---|---|
| `cases.json` | the 406 column-major cases: `[key, decision, decision of the swapped key]` |
| `probe_hw.py` | runs every case with a row-major and a column-major `out=`, plus the strided-mat1 and empty-dimension calls; writes `hw_out.npz` (not kept: 648 MB) and `hw_out.json` |
| `hw_out.json` | the metadata `probe_hw.py` wrote |
| `compare.py` | compares `hw_out.npz` with mmaemu as found on `PYTHONPATH` |
| `compare-8429385.txt` | its output with mmaemu at 8429385 |
| `compare-fixed.txt` | its output with the review's fixes (commit 49f1ecf; rerun on the committed tree, identical) |
| `probe_ld.py`, `probe_ld.txt` | does a padded leading dimension of `out=` change the result? |
| `probe3.py`, `probe3_summary.json` | the first envelope check, an `out=` without unit stride, and the fnuz/e8m0 formats |

## A column-major `out=` swaps the cuBLASLt key

With a column-major `out=`, torch hands cuBLASLt the transposed problem: the heuristic sees the key
(N, M, K, fb, fa) instead of (M, N, K, fa, fb). Where the swapped key records another split-K decision, the bits
change.

- Cases: keys of the shipped `ada-rtx4060-torch` table (1 MiB) whose swapped key is in the table with another
  decision, with M*N*K at or below 2^27; up to two per (decision, swapped decision, out dtype, accumulation), 406 in
  all.
- Row-major `out=` equals the default call in 406 of 406 cases, and column-major `out=` equals it in 28.
- The column-major hardware output equals mmaemu run on the swapped key in 406 of 406 cases.
- mmaemu at 8429385 used the unswapped key: it matched the column-major hardware output in 28 of 406 cases. With the
  fix (`mmaemu/api.py`), 406 of 406.
- `probe_ld.py`: a leading dimension padded by 16 does not change the result, for row-major (406 of 406) or
  column-major (406 of 406) `out=`.

## A mat1 that is neither row- nor column-major

`x.expand(64, K)` and `big[:, ::2]` as mat1: torch copies them to a contiguous tensor and runs. mmaemu at 8429385
raised `ReferenceRejectsError`. With the fix it copies too, and both outputs equal the hardware.

## Empty dimensions

- With M = 0, N = 0 or K = 0, torch returns before it checks the out dtype: float64 and int32 out work. With M, N
  and K all non-zero, both fail with `CUBLAS_STATUS_NOT_SUPPORTED`.
- mmaemu at 8429385 raised for every empty call with these out dtypes. With the fix it returns, with the same shape
  and dtype as torch. `compare.py` compares only shape and dtype for these calls; for K = 0 with float64 out the
  probe kept no bits anyway (its bit view has no 64-bit case: the `KeyError: 8` in `hw_out.json` was raised after
  torch returned).
- K = 0 gives +0 everywhere, without the epilogue: with scale 2, with a bias of 3.0, and with `scale_result` 2 and
  E4M3 out. mmaemu at 8429385 already did this.

## probe3.py

- **Envelope:** M*N at about 1.05, 2 and 8 times the largest split M*N of each setting, for M in 16, 64, 256
  and 1024 and K from 16,384 to 1,048,576. Of 3,360 queries, 251 split, all at the 1.05 point (largest factor 1.083).
  The dense sweeps of [../splitk-envelope](../splitk-envelope/README.md) supersede this check.
- **`out=` without unit stride** (`[:, ::2]` of a NaN-filled tensor): torch returns the `out` tensor, and it is
  still all NaN. torch computes into a copy and does not copy back. mmaemu now does the same and warns.
- **fnuz and e8m0 formats** (`float8_e4m3fnuz`, `float8_e5m2fnuz`, `float8_e8m0fnu` as mat1): empty calls return
  (K = 0 gives zeros). Non-empty calls fail with an internal assert of torch ("Cannot convert ScalarType ...").
  mmaemu now returns on the empty calls and raises `ReferenceRejectsError`, naming the format, on the others.
