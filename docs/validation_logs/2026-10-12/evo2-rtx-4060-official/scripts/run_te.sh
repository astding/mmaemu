#!/bin/bash
# run a golden script in the Transformer Engine env, logging every cuBLASLt call (algo, splitK, workspace)
P=/data/quant/stsc/research/round4/probe
export NVTE_DISABLE_NVRTC=1 HF_HOME=/data/quant/cache/hf TMPDIR=/data/quant/stsc/research/round4/scratch/tmp
export LD_LIBRARY_PATH=$(ls -d $P/.venv-te/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')
export CUBLASLT_LOG_LEVEL=2 CUBLASLT_LOG_FILE=/data/quant/evo2_ada_goldens/logs/cublaslt_$1_%i.log
shift
cd /data/quant/evo2_ada_goldens/scripts && exec $P/.venv-te/bin/python "$@"
