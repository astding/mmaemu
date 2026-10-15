# Split-K envelope sweeps, RTX 4060, 2026-10-14

Step 5 of the split-K resolution ([splitk_tables.md](../../../splitk_tables.md)) infers S = 1 for a shape that is
not in the shipped table, when no live query is possible. At 8429385 it did so, without a warning, when M*N was above
the largest M*N that split in the setting, or K was below the smallest K that split. The review of the release
candidate (2026-10-14) asked whether cuBLASLt agrees. These sweeps ask it directly on the RTX 4060.

## Method

- Query: `mmaemu.platform.cublaslt.heuristic`, the cuBLASLt heuristic with torch 2.7.1's `_scaled_mm` descriptor and
  the profile's workspace and alignment. No GEMM runs.
- Settings: both table profiles, `ada-rtx4060-torch` (1 MiB) and `ada-rtx4060-te` (4 MiB, alignment 256). For each,
  every (out dtype, fast accumulation, format pair) that has split rows in the shipped table: 28 per profile, 56 in
  all.
- Factor f = M*N / (largest split M*N of the setting in the shipped table). For each M and f, N is rounded up to a
  multiple of 16, so the real factor is slightly above f; every factor quoted below is the real one.
- Device: NVIDIA GeForce RTX 4060 (24 SMs), driver 570.211.01, torch 2.7.1+cu128, cuBLASLt 12.8.3.14 (the host of
  [../rtx4060-host/README.md](../rtx4060-host/README.md)).
- Code: on 2026-10-14 the scripts imported the review's work-in-progress tree. The parts they use
  (`cublaslt.heuristic`, `SplitKTable.envelope` and the shipped table) return the same values as at 8429385. On
  2026-10-15 they ran again, unchanged, on the tree of 49f1ecf: every per-query row was equal
  ([../../2026-10-15/rtx-4060-sm89/README.md](../../2026-10-15/rtx-4060-sm89/README.md)).

## Results

Shapes above the largest split M*N:

| Script | M | f | K | Queries | Split (S > 1) | Largest factor that split |
|---|---|---|---|---:|---:|---:|
| `sweep_env.py` | 15 values, 16-8192 | 1.0-4.0 (step 0.1), 5, 6, 8, 12, 16 | 2,048-4,194,304 (6 values) | 181,440 | 1,419 | 1.208 |
| `sweep_env2.py` | 16 values, 1-8192 | 1.00-2.00 (step 0.02), 2.25-4.0 (step 0.25) | 1,024-4,194,304 (13 values) | 663,936 | 8,292 | 1.271 |
| `sweep_env3.py` | 16-1024 (step 16) | 1.25-2.0 (step 0.025) | 3,072-32,768 (6 values) | 666,624 | 274 | 1.271 |
| all | | | | 1,512,000 | 9,985 | 1.271 |

- Shapes above the old envelope split in 52 of the 56 settings, at K from 3,072 to 4,194,304 (none at K = 1,024 or
  2,048).
- No shape split above a factor of 1.271. The sweeps reach a factor of 17.0 in `sweep_env.py` and 5.2 in
  `sweep_env2.py`.

Shapes with K below the smallest split K (`sweep_env.py`):

- 23 of the 56 settings have a smallest split K above 16 in the table: from 1,664 to 4,096.
- For those, the sweep ran K from 16 up to that value (step 16), with M in 16-256 and N in 16-1024, and M*N at or below
  the largest split M*N.
- Of 125,419 queries, 18,504 split, in 13 of the 23 settings. The smallest K that split was 1,760.
- The K rule therefore did not hold: the smallest split K of the table is a property of its K grid, not of cuBLASLt.

`summary.json` gives the counts and the largest factor that split for each setting and sweep.

## Code change

`mmaemu/platform/splitk_table.py` now infers S = 1 only when M*N is above `ENVELOPE_MARGIN` (2) times the setting's
largest split M*N. K plays no part. The largest factor that split in these sweeps is 1.271.

- A shape that is not in the table and is not above that bound raises `SplitKUnknownError`. Without the reference GPU,
  more shapes raise than before; on the RTX 4060 the live query (step 4) resolves them first.
- The Evo 2 shapes resolve as before, to S = 1: from the table up to M = 8192, by the envelope above
  (`tests/test_splitk_table.py::test_evo2_shapes_are_unsplit_and_never_raise`).
- The bound is a measurement, not a proof: it holds for this device, driver and cuBLASLt version on the shapes
  queried. T3p (`tests/test_splitk_table.py::test_live_heuristic_never_splits_where_the_envelope_infers_s1`) queries
  shapes at 2-4 times the largest split M*N of every setting live (11,760 queries) and requires S = 1.

## Not kept

The per-query row files (`sweep_env.json` 30 MB, `sweep_env2.json` 61 MB, `sweep_env3.json` 61 MB) are not in the
repository. The scripts rebuild them on the reference GPU in under a minute each (19 s, 40 s and 42 s here).
