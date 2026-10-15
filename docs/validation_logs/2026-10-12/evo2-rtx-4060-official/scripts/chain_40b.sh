#!/bin/bash
# 40B TE goldens one window at a time (default CUDA allocator, as the earlier successful block oracle), falling back
# to the first 4 kb of a window when the official path does not fit 8 GB; then the cuBLASLt heuristic table.
cd /data/quant/evo2_ada_goldens/scripts
unset PYTORCH_CUDA_ALLOC_CONF
run() {
  timeout 3000 /data/quant/tcemu_audit/gpu.sh ./run_te.sh 40b_oracle_te_$1 golden_oracle.py --model evo2_40b --windows $1 > ../logs/40b_$1.log 2>&1
  grep -qE '^\{"tag"' ../logs/40b_$1.log
}
for w in w4k p0 p1 p2 p3 w8k; do
  echo "== 40B $w $(date +%T)"
  if run $w; then echo "OK $w"; else
    echo "FAILED $w: $(grep -E 'Error' ../logs/40b_$w.log | tail -1 | cut -c1-120)"
    if [ $w != w4k ] && [ $w != w8k ]; then run ${w}_4k && echo "OK ${w}_4k" || echo "FAILED ${w}_4k"; fi
  fi
done
echo "== heuristics $(date +%T)"
L=/data/quant/stsc/research/round4/probe/.venv/lib/python3.12/site-packages/nvidia/cublas/lib/libcublasLt.so.12
/data/quant/tcemu_audit/gpu.sh /data/quant/stsc/research/round4/probe/.venv/bin/python /data/quant/tcemu_audit/work/verify-c5-splitk-4/lt_ctypes.py $L ../heuristics/jobs.json ../heuristics/heur_rtx4060_cublasLt12.8.3.json > ../logs/heur.log 2>&1
tail -2 ../logs/heur.log; ls -la ../heuristics
echo "CHAIN_DONE $(date +%T)"
