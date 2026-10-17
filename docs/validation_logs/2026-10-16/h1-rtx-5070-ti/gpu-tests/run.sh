#!/bin/bash
# GPU tests of h1 (and v1/v4 under both presets) at the committed tree, RTX 5070 Ti
cd /data/quant/mmaemu_work/wt/hopper
O=${OUT:?set OUT to the output directory}
PY=/data/quant/stsc/research/round4/probe/.venv/bin/python
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 MMAEMU_TEST_FATBIN_DIR=$PWD/mmaemu/backends/cuda/bin
{ echo "commit $(git rev-parse HEAD)"; echo "dirty mmaemu/ tests/: $(git status --porcelain -- mmaemu tests | wc -l) files"
  $PY -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"; } > $O/summary.txt 2>&1
for t in test_kernels_ieee test_cuda_runtime test_h100_mini test_backend_selection; do
  /data/quant/tcemu_audit/gpu.sh timeout 600 $PY -m pytest -p no:cacheprovider -q -rs tests/$t.py > $O/$t.log 2>&1
  echo "$t rc=$? $(tail -1 $O/$t.log)" >> $O/summary.txt
done
echo DONE >> $O/summary.txt
