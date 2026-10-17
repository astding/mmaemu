#!/usr/bin/env bash
# T3 GPU tier: every GPU test file in its OWN locked run of < 10 minutes on the shared GPU.
#
# usage: tools/run_gpu_tests.sh [--di-full] [--out DIR] [extra pytest args...]
#   --di-full   also run tests/test_device_independence.py::test_evo2_acc_sha256 with MMAEMU_DI_FULL=<backend>
#               for every backend, 7 Evo 2 calls per locked run (runbook gate G3 on a new GPU)
#   --out DIR   logs (default docs/validation_logs/<date>/t3-<gpu-slug>)
# environment: MMAEMU_GPU_LOCK (command prefix that serialises GPU work; default: flock on $TMPDIR/mmaemu-gpu.lock;
#              GPU is accepted as an older name), MMAEMU_TRUTH_DIR (device independence),
#              PYTHON (default python). Exit status: 0 iff no run failed (pytest exit 5 = nothing collected is OK).
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
read -r -a GPU <<< "${MMAEMU_GPU_LOCK:-${GPU:-flock -w 3600 ${TMPDIR:-/tmp}/mmaemu-gpu.lock}}"
PYTHON=${PYTHON:-python}
DI_FULL=0
OUT=""
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --di-full) DI_FULL=1; shift ;;
    --out) OUT=$2; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
slug=$("${GPU[@]}" timeout 120 "$PYTHON" -c "import torch; n=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'nogpu'; c=torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0,0); print(n.lower().replace('nvidia','').replace('geforce','').strip().replace(' ','-')+'-sm%d%d'%c)" 2>/dev/null || echo unknown)
OUT=${OUT:-$here/docs/validation_logs/$(date +%F)/t3-$slug}
mkdir -p "$OUT"
cd "$here" || exit 2

# Fatbins under test (tests/test_kernels_ieee.py, tests/test_cuda_runtime.py): $MMAEMU_TEST_FATBIN_DIR, else the
# package's committed release fatbins if present, else the development build in build/fatbin. A file whose fatbin
# tests skip for want of a manifest counts as a FAILED run: a skipped kernel test must never look like a pass.
if [ -z "${MMAEMU_TEST_FATBIN_DIR:-}" ]; then
  if [ -f "$here/mmaemu/backends/cuda/bin/manifest.json" ]; then
    export MMAEMU_TEST_FATBIN_DIR="$here/mmaemu/backends/cuda/bin"
  else
    export MMAEMU_TEST_FATBIN_DIR="$here/build/fatbin"
  fi
fi

# one file per locked run; files that do not exist are skipped and reported
FILES=(tests/test_scaffold.py tests/test_backend_reference.py tests/test_epilogue.py tests/test_kernels_ieee.py
       tests/test_cuda_runtime.py tests/test_fast_kernels.py tests/test_gates_hmma.py tests/test_live_cublas.py
       tests/test_hw_mini.py tests/test_device_independence.py tests/test_h100_mini.py tests/test_backend_selection.py)
fail=0
summary="$OUT/summary.txt"
: > "$summary"
echo "fatbins under test: $MMAEMU_TEST_FATBIN_DIR" | tee -a "$summary"
run() {  # name, then the command
  local name=$1; shift
  local t0=$(date +%s)
  "${GPU[@]}" timeout 600 "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  local last=$(grep -E "passed|failed|error|skipped|no tests ran" "$OUT/$name.log" | tail -1)
  printf '%-48s rc=%-3s %4ss  %s\n' "$name" "$rc" "$dt" "$last" | tee -a "$summary"
  if [ $rc -ne 0 ] && [ $rc -ne 5 ]; then fail=1; fi
  if grep -q "no fatbin manifest" "$OUT/$name.log"; then
    printf '%-48s FAIL: fatbin tests skipped (no manifest in %s)\n' "$name" "$MMAEMU_TEST_FATBIN_DIR" | tee -a "$summary"
    fail=1
  fi
}
for f in "${FILES[@]}"; do
  if [ ! -f "$f" ]; then
    printf '%-48s SKIPPED (not in this checkout)\n' "$f" | tee -a "$summary"
    continue
  fi
  # test_live_cublas.py is T3p (profile device only; it skips itself elsewhere) and is not marker-filtered
  marker=(-m gpu)
  case "$f" in tests/test_live_cublas.py) marker=() ;; esac
  run "$(basename "$f" .py)" "$PYTHON" -m pytest -q -p no:cacheprovider -rs "${marker[@]}" "$f" "${EXTRA[@]}"
done
if [ "$DI_FULL" = 1 ] && [ -f tests/test_device_independence.py ]; then
  for be in reference cuda cuda_v1 cuda_v4 cuda_v9 cuda_tc3; do       # pytest ids use "_" for ":"
    for chunk in "00 01 02 03 04 05 06" "07 08 09 10 11 12 13" "14 15 16 17 18 19 20"; do
      calls=$(for c in $chunk; do printf ' or %s-call%s]' "$be" "$c"; done | sed 's/^ or //')
      sel="test_evo2_acc_sha256 and ($calls)"
      MMAEMU_DI_FULL=${be/_/:} run "di_full_${be}_${chunk// /-}" "$PYTHON" -m pytest -q -p no:cacheprovider -rs \
        -m gpu tests/test_device_independence.py -k "$sel" "${EXTRA[@]}"
    done
  done
fi
echo "logs: $OUT" | tee -a "$summary"
exit $fail
