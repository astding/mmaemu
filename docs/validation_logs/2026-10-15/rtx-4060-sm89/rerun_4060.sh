#!/usr/bin/env bash
# Rerun on the RTX 4060 host after the release-candidate review: preflight, T3, T3p, the three split-K envelope
# sweeps and a bench smoke, all on the git archive in ~/mmaemu-4060/src (src/COMMIT), under the GPU lock.
# usage: bash rerun_4060.sh   (from anywhere; writes ~/mmaemu-4060/logs-review/)
set -u
H=$HOME/mmaemu-4060
L=$H/logs-review
PY=$H/venv/bin/python
LOCK="flock -w 3600 /tmp/mmaemu-gpu.lock"
mkdir -p "$L" "$L/sweeps"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$H" || exit 2
echo "commit $(cat src/COMMIT)" | tee "$L/run.txt"

# 1. preflight (EXPECT commit = src/COMMIT of this archive)
PYTHONPATH=$H/src $PY preflight_4060.py --out "$L/preflight" > "$L/preflight.out" 2>&1
echo "preflight rc=$?" | tee -a "$L/run.txt"

# 2. T3: every GPU test file in its own locked run
(cd src && env PYTHON=$PY MMAEMU_TRUTH_DIR=$H/truth MMAEMU_AUDIT_DIR=$H/audit \
    tools/run_gpu_tests.sh --out "$L/t3-rtx-4060-sm89" > "$L/t3.out" 2>&1)
echo "T3 rc=$?" | tee -a "$L/run.txt"

# 3. T3p: the profile-device tests
(cd src && PYTHONPATH=$H/src MMAEMU_TRUTH_DIR=$H/truth $LOCK timeout 1200 $PY -m pytest -q -p no:cacheprovider -s \
    -m profile_device tests/ > "$L/t3p_profile_device.log" 2>&1)
echo "T3p rc=$?" | tee -a "$L/run.txt"
(cd src && PYTHONPATH=$H/src $PY -m pytest -q -p no:cacheprovider --collect-only -m profile_device tests/ 2>/dev/null \
    | grep "::" >> "$L/t3p_profile_device.log")

# 4. the split-K envelope sweeps, as committed, importing the archive
cp src/docs/validation_logs/2026-10-14/splitk-envelope/sweep_env*.py "$L/sweeps/"
for s in sweep_env sweep_env2 sweep_env3; do
  (cd "$L/sweeps" && PYTHONPATH=$H/src $LOCK timeout 1800 $PY $s.py > $s.out 2>&1)
  echo "$s rc=$?" | tee -a "$L/run.txt"
done

# 5. bench smoke: the release fatbins, every kernel, one shape and M, both data sets
for d in gauss40 adversarial; do
  (cd src && PYTHONPATH=$H/src $LOCK timeout 900 $PY tools/bench_kernels.py --kernels v1,v4,v9,tc3 --shape 1b \
      --M 2048 --data $d --out "$L/bench-smoke-1b-$d.json" > "$L/bench-smoke-1b-$d.log" 2>&1)
  echo "bench $d rc=$?" | tee -a "$L/run.txt"
done
echo REVIEW_RERUN_DONE | tee -a "$L/run.txt"
