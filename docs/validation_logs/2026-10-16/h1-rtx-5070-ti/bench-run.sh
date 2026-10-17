#!/bin/bash
# h1 / v1 / v4 throughput at the committed tree, RTX 5070 Ti, M = 2048, gauss40
cd /data/quant/mmaemu_work/wt/hopper
O=${OUT:?set OUT to the output directory}
PY=/data/quant/stsc/research/round4/probe/.venv/bin/python
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1
{ echo "commit $(git rev-parse HEAD)"; echo "dirty mmaemu/ tools/: $(git status --porcelain -- mmaemu tools | wc -l) files"
  $PY -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"; } > $O/summary.txt 2>&1
for sh in 1b 7b 40b; do
  /data/quant/tcemu_audit/gpu.sh timeout 900 $PY tools/bench_kernels.py --kernels v1,v4,h1 --shape $sh --M 2048 --data gauss40 \
     --out $O/bench-$sh-gauss40.json > $O/bench-$sh-gauss40.log 2>&1
  echo "$sh rc=$?" >> $O/summary.txt
done
echo DONE >> $O/summary.txt
