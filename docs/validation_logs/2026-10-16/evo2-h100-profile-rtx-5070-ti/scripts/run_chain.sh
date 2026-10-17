#!/usr/bin/env bash
# Evo 2 with mmaemu's H100 PCIe profile (hopper-h100pcie-te) on the RTX 5070 Ti, strictly sequential, every GPU step
# under the GPU lock. mmaemu from the clean worktree of fa0fd90 (PYTHONPATH).
# usage: bash run_chain.sh
set -u
H=$(cd "$(dirname "$0")" && pwd)
PY=<venv>/bin/python
SRC=<src>
LOCK=$MMAEMU_GPU_LOCK
CK1=<models>/evo2_1b_base/evo2_1b_base.pt
CK40=<models>/evo2_40b/evo2_40b.pt
REF1=$SRC/examples/data/evo2_1b_ada_rtx4060_te.npz
REF40=$SRC/docs/validation_logs/2026-10-15/evo2-40b-h100-sm90/h100_official_te_40b.npz
WIN=w2k,w4k,w8k,p0,p1,p2,p3
O=$H/out
ST=$O/status.txt
export PYTHONPATH=$SRC PYTHONDONTWRITEBYTECODE=1
cd "$H" || exit 2
T="/usr/bin/time -f wall_%e_s_maxrss_%M_kB"
step() {  # name, timeout (s), then the arguments of python
  local name=$1 to=$2; shift 2
  local t0=$(date +%s)
  echo "$(date '+%F %T') START $name (MemAvailable $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) GiB)" >> "$ST"
  $LOCK timeout "$to" $T "$PY" "$@" > "$O/$name.log" 2>&1
  local rc=$?
  echo "$(date '+%F %T') END   $name rc=$rc $(( $(date +%s) - t0 ))s" >> "$ST"
}
{
  echo "mmaemu $(git -C "$SRC" rev-parse HEAD) clean=$(git -C "$SRC" status --porcelain | wc -l)"
  "$PY" -c "import torch, sys; print('python', sys.version.split()[0], 'torch', torch.__version__)"
  "$PY" -m pip list 2>/dev/null | grep -i -E '^(evo2|vtx|flash.attn|numpy|scikit.learn|scipy|pandas|biopython|torch) '
  sha256sum run_offload.py brca1_offload.py
} > "$ST"

step 1b_test 1800 run_offload.py --model evo2_1b_base --checkpoint "$CK1" --modes mmaemu_h100 --windows $WIN \
  --ref "$REF1" --check-resident --out "$O/1b_test.json"
step 1b_brca1 5400 brca1_offload.py --model evo2_1b_base --checkpoint "$CK1" --variants 500 --modes mmaemu_h100 \
  --out "$O/1b_brca1.json"
step 40b_test 3000 run_offload.py --model evo2_40b --checkpoint "$CK40" --modes mmaemu_h100 --windows $WIN \
  --ref "$REF40" --out "$O/40b_test.json"
step 40b_brca1 12000 brca1_offload.py --model evo2_40b --checkpoint "$CK40" --variants 100 --modes mmaemu_h100 \
  --out "$O/40b_brca1.json"

echo "$(date '+%F %T') CHAIN DONE clean=$(git -C "$SRC" status --porcelain | wc -l)" >> "$ST"
