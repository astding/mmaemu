#!/bin/bash
# The H100 run: TE setup; the method checked on the 1B against the real upstream model on this GPU; the 40B test
# sequences; the 40B BRCA1 windows (first 100 variants first, then the rest of the 500). usage: run_job.sh <task root>
set -u
R=$1; J=/home/ubuntu/h100-40b; O=$J/out; mkdir -p $O; S=$O/status.txt
PY=$R/venv/bin/python; SP=$R/venv/lib/python3.12/site-packages
note() { echo "$1 $(date -u +%FT%TZ)" >> $S; }
note "start"
[ -f /home/ubuntu/models/evo2_40b/download.log ] || { setsid nohup bash $J/download_40b.sh > /dev/null 2>&1 & note "download started"; }
bash $J/te_setup.sh $R $J; rc=$?; note "te_setup rc=$rc"; [ $rc = 0 ] || exit 1
export NVTE_DISABLE_NVRTC=1 HF_HOME=$R/cache/hf LD_LIBRARY_PATH=$(ls -d $SP/nvidia/*/lib | tr '\n' ':')
nvidia-smi > $O/nvidia-smi.txt 2>&1
CK1=$(ls $R/cache/hf/hub/models--arcinstitute--evo2_1b_base/snapshots/*/evo2_1b_base.pt)
cd $J
$PY official_blockwise.py --model evo2_1b_base --checkpoint $CK1 --inputs inputs.npz --out $O/1b_official \
  --groups test,brca1 --brca1-variants 10 --first-chunk-variants 4 --chunk-tokens $((3 * 8192)) \
  > $O/1b_official.log 2>&1; note "1b_official rc=$?"
$PY real_1b.py --inputs inputs.npz --out $O/1b_real --brca1-variants 10 > $O/1b_real.log 2>&1; note "1b_real rc=$?"
$PY check_1b.py $O/1b_official $O/1b_real $O/1b_check.json > $O/1b_check.log 2>&1; rc=$?; note "1b_check rc=$rc"
if [ $rc != 0 ]; then                    # decided by hand: touch $J/continue, or kill this job
  note "1b_check FAILED: waiting for $J/continue"
  while [ ! -f $J/continue ]; do sleep 10; done; note "continuing after the failed 1b_check"
fi
while ! grep -q "DOWNLOAD_DONE\|DOWNLOAD_FAILED" /home/ubuntu/models/evo2_40b/download.log 2>/dev/null; do sleep 20; done
grep -q DOWNLOAD_DONE /home/ubuntu/models/evo2_40b/download.log || { note "STOP: 40B download failed"; exit 1; }
CK40=/home/ubuntu/models/evo2_40b/evo2_40b.pt
$PY official_blockwise.py --model evo2_40b --checkpoint $CK40 --inputs inputs.npz --out $O/40b --groups test \
  > $O/40b_test.log 2>&1; note "40b_test rc=$?"
$PY official_blockwise.py --model evo2_40b --checkpoint $CK40 --inputs inputs.npz --out $O/40b --groups brca1 \
  --brca1-variants 500 --first-chunk-variants 100 > $O/40b_brca1.log 2>&1; note "40b_brca1 rc=$?"
(cd $O && find . -type f ! -name SHA256SUMS -exec sha256sum {} + | sort -k2 > SHA256SUMS)
note "DONE"
