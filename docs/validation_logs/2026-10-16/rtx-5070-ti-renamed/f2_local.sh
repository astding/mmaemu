#!/usr/bin/env bash
# F2 (the tree after the profile rename) on the RTX 5070 Ti: final_local.sh (T3 --di-full, T4 gpu, probe replay, T2,
# T4 cpu, pytest_full, sweep replay), then the Evo 2 example check (examples/evo2_offload.py, both TE profiles), then
# the notebook executed in place. Strictly sequential, GPU steps under the lock.
# usage: bash f2_local.sh SRC OUT
set -u
SRC=$1; L=$2
S=<scratch>
LOCK=$MMAEMU_GPU_LOCK
NB=<venv>/bin
CK1=<models>/evo2_1b_base/evo2_1b_base.pt
CK40=<models>/evo2_40b/evo2_40b.pt
BRCA=<brca1-data>
mkdir -p "$L"
bash "$S/rc2/final_local.sh" "$SRC" "$L"
ST=$L/status.txt
step() {  # name, then the command
  local name=$1; shift
  local t0=$(date +%s)
  echo "$(date '+%F %T') START $name clean=$(git -C "$SRC" status --porcelain | wc -l)" >> "$ST"
  "$@" > "$L/$name.out" 2>&1
  local rc=$?
  echo "$(date '+%F %T') END   $name rc=$rc $(( $(date +%s) - t0 ))s" >> "$ST"
}
export PYTHONDONTWRITEBYTECODE=1
(cd /tmp && PYTHONPATH=$SRC $NB/python -c "import mmaemu, evo2, torch; print('mmaemu', mmaemu.__file__, mmaemu.__version__, 'evo2', evo2.__file__, 'torch', torch.__version__)") >> "$ST" 2>&1
PYP=<venv>/bin/python
step export_h100_check bash -c "cd '$SRC' && CUDA_VISIBLE_DEVICES= $PYP tools/export_splitk_table_h100.py --check"
step resolver_check bash -c "cd '$SRC' && CUDA_VISIBLE_DEVICES= $PYP tools/check_splitk_resolution_h100.py --jobs 16 --out '$L/resolver_check.json'"
EC=$SRC/docs/validation_logs/2026-10-16/evo2-h100-profile-rtx-5070-ti/example_check
step example_check $LOCK timeout 5400 bash "$EC/run_checks.sh" $NB/python "$CK1" "$CK40" "$BRCA"
mkdir -p "$L/notebook_out"
step notebook bash -c "cd '$SRC/examples' && PYTHONPATH='$SRC' EVO2_1B_CHECKPOINT='$CK1' EVO2_BLACKWELL_OUT='$L/notebook_out' \
  $LOCK timeout 5400 $NB/jupyter nbconvert --to notebook --execute --inplace evo2_blackwell.ipynb"
echo "$(date '+%F %T') F2_LOCAL_DONE clean=$(git -C "$SRC" status --porcelain | wc -l)" >> "$ST"
git -C "$SRC" status --porcelain >> "$ST"
