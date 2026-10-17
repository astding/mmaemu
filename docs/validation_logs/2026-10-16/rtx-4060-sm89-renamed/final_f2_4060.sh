#!/usr/bin/env bash
# RTX 4060 host, the tree after the profile rename (F2; final_4060.sh of fa0fd90 with logs-f2 and the sweeps' profile
# names mapped): preflight, T3 (every GPU test file, incl.
# the hopper preset's h100_mini and backend_selection), the full accumulators of every GPU kernel, T3p, the three
# split-K envelope sweeps, T4 GPU tier, bench (v1, v4, v9, tc3, h1, h4; 1b/7b/40b, M = 2048, gauss40), and the
# H100 PCIe probe replay through h4 (--backend auto) and h1 (--backend cuda:h1). All on the git checkout in
# ~/mmaemu-4060/src (src/COMMIT), strictly one step after another, every GPU step under the GPU lock.
# usage: bash final_f2_4060.sh   (writes ~/mmaemu-4060/logs-f2/)
set -u
H=$HOME/mmaemu-4060
L=$H/logs-f2
PY=$H/venv/bin/python
LOCK="flock -w 3600 /tmp/mmaemu-gpu.lock"
mkdir -p "$L" "$L/sweeps" "$L/t4" "$L/replay-h4" "$L/replay-h1" "$L/hopper-kernels"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MMAEMU_TRUTH_DIR=$H/truth MMAEMU_AUDIT_DIR=$H/audit
cd "$H" || exit 2
log() { echo "$(date '+%F %T') $*" | tee -a "$L/run.txt"; }
: > "$L/run.txt"
log "commit $(cat src/COMMIT) head $(git -C src rev-parse HEAD) dirty $(git -C src status --porcelain -- mmaemu tools tests | wc -l)"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader >> "$L/run.txt"

# 1. preflight (EXPECT commit = src/COMMIT)
PYTHONPATH=$H/src $PY preflight_4060.py --out "$L/preflight" > "$L/preflight.out" 2>&1
log "preflight rc=$?"

# 2. T3: every GPU test file in its own locked run
(cd src && env PYTHON=$PY MMAEMU_GPU_LOCK="$LOCK" tools/run_gpu_tests.sh --out "$L/t3-rtx-4060-sm89" > "$L/t3.out" 2>&1)
log "T3 rc=$?"

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

# 3. full accumulators of every GPU kernel (MMAEMU_DI_FULL), as in di_full.sh of 2026-10-14
(cd src && PYTHONPATH=$H/src MMAEMU_DI_FULL=cuda:v1,cuda:v4,cuda:v9,cuda:tc3 $LOCK timeout 3000 $PY -m pytest -q \
    -p no:cacheprovider -rs -m gpu tests/test_device_independence.py -k test_evo2_acc_sha256 \
    > "$L/t3-rtx-4060-sm89/di_full_gpu_kernels.log" 2>&1)
log "DI_FULL rc=$?"

# 4. T3p: the profile-device tests
(cd src && PYTHONPATH=$H/src $LOCK timeout 1200 $PY -m pytest -q -p no:cacheprovider -s \
    -m profile_device tests/ > "$L/t3p_profile_device.log" 2>&1)
log "T3p rc=$?"
(cd src && PYTHONPATH=$H/src $PY -m pytest -q -p no:cacheprovider --collect-only -m profile_device tests/ 2>/dev/null \
    | grep "::" >> "$L/t3p_profile_device.log")

# 5. the split-K envelope sweeps, as committed, importing the checkout
cp src/docs/validation_logs/2026-10-14/splitk-envelope/sweep_env*.py "$L/sweeps/"
# the sweeps as committed name the profiles of then; map them to the new names (the only change)
sed -i -e 's/ada-rtx4060-torch/ada/g' -e 's/ada-rtx4060-te/ada-te/g' "$L"/sweeps/sweep_env*.py
for s in sweep_env sweep_env2 sweep_env3; do diff src/docs/validation_logs/2026-10-14/splitk-envelope/$s.py "$L/sweeps/$s.py"; done \
  > "$L/sweeps/name_mapping.diff"
for s in sweep_env sweep_env2 sweep_env3; do
  (cd "$L/sweeps" && PYTHONPATH=$H/src $LOCK timeout 1800 $PY $s.py > $s.out 2>&1)
  log "$s rc=$?"
done

# 6. T4 GPU tier
(cd src && PYTHONPATH=$H/src $LOCK timeout 1800 $PY tools/mutants/run.py --tier gpu --out "$L/t4/mutants-gpu-rtx4060.json" \
    > "$L/t4/mutants-gpu-rtx4060.txt" 2>&1)
log "T4gpu rc=$?"

# 7. bench: the release fatbins, every kernel, the three Evo 2 in_proj shapes, M = 2048, gauss40
for sh in 1b 7b 40b; do
  (cd src && PYTHONPATH=$H/src $LOCK timeout 1500 $PY tools/bench_kernels.py --kernels v1,v4,v9,tc3,h1,h4 --shape $sh \
      --M 2048 --data gauss40 --out "$L/bench-$sh-gauss40.json" > "$L/bench-$sh-gauss40.log" 2>&1)
  log "bench $sh rc=$?"
done

# 8. H100 PCIe probe replay through h4 (auto) and h1, in chunks, then merged
for be in auto cuda:h1; do
  R=$L/replay-h4; [ "$be" = cuda:h1 ] && R=$L/replay-h1
  for off in 0 250 500 750 1000 1250; do
    (cd src && PYTHONPATH=$H/src $LOCK timeout 1500 $PY tools/replay_h100_log.py --backend $be --parts live \
        --offset $off --limit 250 --part "$R/live_$off.json" > "$R/live_$off.log" 2>&1)
    log "replay $be live $off rc=$?"
  done
  for c in 0,1,2,3,4,5,6,7,8,9,10 11,12,13,14,15,16,17,18,19,20; do
    (cd src && PYTHONPATH=$H/src $LOCK timeout 1500 $PY tools/replay_h100_log.py --backend $be --parts m2 \
        --truth "$MMAEMU_TRUTH_DIR" --m2-calls $c --part "$R/m2_${c%%,*}.json" > "$R/m2_${c%%,*}.log" 2>&1)
    log "replay $be m2 ${c%%,*} rc=$?"
  done
  (cd src && PYTHONPATH=$H/src $PY tools/replay_h100_log.py --merge "$R"/live_*.json "$R"/m2_*.json \
      --out "$R/mmaemu_replay.json" > "$R/merge.out" 2>&1)
  log "replay $be merge rc=$?"
done
log "src dirty after the run: $(git -C src status --porcelain -- mmaemu tools tests | wc -l)"
log FINAL_F2_4060_DONE
