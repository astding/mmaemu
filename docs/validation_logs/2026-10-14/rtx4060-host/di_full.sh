#!/bin/sh
# Full-accumulator device independence for every GPU kernel on the profile device (after T6).
cd ~/mmaemu-4060/src || exit 1
until grep -q NATIVE_DONE ../logs/t6/native-fp8.log 2>/dev/null; do sleep 10; done
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 MMAEMU_TRUTH_DIR=$HOME/mmaemu-4060/truth MMAEMU_AUDIT_DIR=$HOME/mmaemu-4060/audit
export MMAEMU_DI_FULL=cuda:v1,cuda:v4,cuda:v9,cuda:tc3
flock -w 3600 /tmp/mmaemu-gpu.lock timeout 3000 ../venv/bin/python -m pytest -q -p no:cacheprovider -rs -m gpu \
  tests/test_device_independence.py -k test_evo2_acc_sha256 > ../logs/t3-rtx-4060-sm89/di_full_gpu_kernels.log 2>&1
echo "DI_FULL_DONE rc=$?" >> ../logs/t3-rtx-4060-sm89/di_full_gpu_kernels.log
