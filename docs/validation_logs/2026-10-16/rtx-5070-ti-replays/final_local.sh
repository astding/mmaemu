#!/usr/bin/env bash
# Final-tree local validation chain (detached clean worktree), strictly sequential, every GPU step under the lock.
# usage: bash final_local.sh SRC OUT
set -u
SRC=$1
L=$2
PY=<venv>/bin/python
PYCPU=<scratch>/ci/venv-test-torch2.7.1/bin/python
export MMAEMU_TRUTH_DIR=$MMAEMU_TRUTH_DIR MMAEMU_AUDIT_DIR=$MMAEMU_AUDIT_DIR
export MMAEMU_GPU_LOCK=$MMAEMU_GPU_LOCK PYTHONDONTWRITEBYTECODE=1
LOCK=$MMAEMU_GPU_LOCK
mkdir -p "$L/t4" "$L/replay-probe" "$L/replay-sweep"
ST=$L/status.txt
cd "$SRC" || exit 2
step() {  # name, then the command
  local name=$1; shift
  local t0=$(date +%s)
  echo "$(date '+%F %T') START $name" >> "$ST"
  "$@" > "$L/$name.out" 2>&1
  local rc=$?
  echo "$(date '+%F %T') END   $name rc=$rc $(( $(date +%s) - t0 ))s" >> "$ST"
}
echo "commit $(git rev-parse HEAD) clean=$(git status --porcelain | wc -l) audit=$MMAEMU_AUDIT_DIR" > "$ST"
sha256sum "$MMAEMU_TRUTH_DIR/manifest.json" >> "$ST"

# T3 on the RTX 5070 Ti: every GPU test file, the release fatbins, DI_FULL for every backend
step t3 env PYTHON="$PY" bash tools/run_gpu_tests.sh --di-full --out "$L/t3-rtx-5070-ti-sm120"

# T4 GPU tier
step t4gpu env PYTHONPATH="$SRC" $LOCK timeout 600 "$PY" tools/mutants/run.py --tier gpu \
  --out "$L/t4/mutants-gpu.json"

# H100 PCIe probe replay through h4 (--backend auto), in chunks, then merged
for off in 0 250 500 750 1000 1250; do
  step "replay_probe_live_$off" env PYTHONPATH="$SRC" $LOCK timeout 1500 "$PY" tools/replay_h100_log.py \
    --device cuda --backend auto --parts live --offset $off --limit 250 --part "$L/replay-probe/live_$off.json"
done
for c in 0,1,2,3,4,5,6,7,8,9,10 11,12,13,14,15,16,17,18,19,20; do
  step "replay_probe_m2_${c%%,*}" env PYTHONPATH="$SRC" $LOCK timeout 1500 "$PY" tools/replay_h100_log.py \
    --device cuda --backend auto --parts m2 --truth "$MMAEMU_TRUTH_DIR" --m2-calls $c \
    --part "$L/replay-probe/m2_${c%%,*}.json"
done
step replay_probe_merge env PYTHONPATH="$SRC" "$PY" tools/replay_h100_log.py --merge "$L"/replay-probe/live_*.json \
  "$L"/replay-probe/m2_*.json --out "$L/replay-probe/mmaemu_replay.json"

# T2: CPU full truth replay (torch 2.7.1+cpu, no GPU visible)
step t2 env CUDA_VISIBLE_DEVICES= PYTHONPATH="$SRC:$MMAEMU_TRUTH_DIR" "$PYCPU" tools/replay_truth.py \
  --out "$L/t2" --compare "$MMAEMU_TRUTH_DIR/replay/replay_fixed.json" --threads 12
sha256sum "$MMAEMU_TRUTH_DIR/manifest.json" >> "$ST"

# T4 CPU tier
step t4cpu env CUDA_VISIBLE_DEVICES= PYTHONPATH="$SRC" "$PY" tools/mutants/run.py --tier cpu --jobs 4 \
  --out "$L/t4/mutants-cpu.json"

# Full CPU suite with the truth and the audit data (T0 + T1 + slow + truth tests; the GPU tests are T3)
step pytest_full env CUDA_VISIBLE_DEVICES= PYTHONPATH="$SRC" "$PY" -m pytest -q -p no:cacheprovider -rs tests
# split-K sweep p5 replay through h4 (--backend auto), in chunks, then merged
SW=docs/validation_logs/2026-10-16/h100-splitk-sweep/results/live_jobs.tar.gz
for off in 0 500 1000 1500 2000 2500; do
  step "replay_sweep_$off" env PYTHONPATH="$SRC" $LOCK timeout 2400 "$PY" tools/replay_h100_log.py \
    --device cuda --backend auto --parts live --live "$SW" --offset $off --limit 500 \
    --part "$L/replay-sweep/live_$off.json"
done
step replay_sweep_merge env PYTHONPATH="$SRC" "$PY" tools/replay_h100_log.py --merge "$L"/replay-sweep/live_*.json \
  --live "$SW" --out "$L/replay-sweep/mmaemu_replay_p5.json"

echo "$(date '+%F %T') CHAIN DONE clean=$(git status --porcelain | wc -l)" >> "$ST"
