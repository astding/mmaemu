#!/usr/bin/env bash
# RTX 4060 host, final release tree: step 2b of final_4060.sh alone (the hopper preset's kernels: their four test
# files without a marker filter, on the package fatbins, each in its own locked run; then mmaemu info), on the
# git checkout in ~/mmaemu-4060/src (src/COMMIT), every GPU step under the GPU lock.
# usage: bash final_4060_hk.sh   (step 2b of final_4060.sh alone: rewrites ~/mmaemu-4060/logs-final/hopper-kernels/)
set -u
H=$HOME/mmaemu-4060
L=$H/logs-final
PY=$H/venv/bin/python
LOCK="flock -w 3600 /tmp/mmaemu-gpu.lock"
mkdir -p "$L/hopper-kernels"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MMAEMU_TRUTH_DIR=$H/truth MMAEMU_AUDIT_DIR=$H/audit
cd "$H" || exit 2
log() { echo "$(date '+%F %T') $*" | tee -a "$L/run.txt"; }
log "commit $(cat src/COMMIT) head $(git -C src rev-parse HEAD) dirty $(git -C src status --porcelain -- mmaemu tools tests | wc -l)"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader >> "$L/run.txt"

# 2b. the hopper preset's kernels: their four test files, no marker filter, each in its own locked run; mmaemu info
HK=$L/hopper-kernels
export MMAEMU_TEST_FATBIN_DIR=$H/src/mmaemu/backends/cuda/bin   # the package fatbins, as tools/run_gpu_tests.sh tests them
{ echo "commit $(git -C src rev-parse HEAD)"; echo "dirty mmaemu/ tests/: $(git -C src status --porcelain -- mmaemu tests | wc -l) files"
  PYTHONPATH=$H/src $PY -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"; } > "$HK/summary.txt"
for f in test_kernels_ieee test_cuda_runtime test_h100_mini test_backend_selection; do
  (cd src && PYTHONPATH=$H/src $LOCK timeout 1800 $PY -m pytest -q -p no:cacheprovider -rs tests/$f.py > "$HK/$f.log" 2>&1)
  rc=$?; echo "$f rc=$rc $(tail -1 "$HK/$f.log")" >> "$HK/summary.txt"
done
echo DONE >> "$HK/summary.txt"
log "hopper-kernels done"
(cd src && PYTHONPATH=$H/src $LOCK timeout 600 $PY -m mmaemu info > "$L/mmaemu_info.txt" 2>&1)
log "info rc=$?"

log "hopper-kernels rerun done"
