#!/bin/bash
# examples/evo2_offload.py against the scripts of this log:
#   bash run_checks.sh <python> <evo2_1b_base.pt> <evo2_40b.pt> <BRCA1 data dir of make_inputs.py>
# the 1B: test and score (40 BRCA1 windows) in the three modes; the 40B: test with mmaemu. Writes here.
set -u
H=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$H/../../../../.." && pwd)
PY=$1; CK1=$2; CK40=$3; DATA=$4
cd "$H" || exit 1
export PYTHONPATH=$ROOT
OFF=$ROOT/examples/evo2_offload.py
T="/usr/bin/time -f wall_%e_s_maxrss_%M_kB"
$PY make_inputs.py --data "$DATA" || exit 1
for m in mmaemu native bf16; do
  $T $PY $OFF test --model evo2_1b_base --checkpoint "$CK1" --fp8 $m --out test_1b_$m.json > test_1b_$m.log 2>&1
  echo "test 1b $m rc=$?"
  $T $PY $OFF score --model evo2_1b_base --checkpoint "$CK1" --fp8 $m --input brca1_40.fasta --out score_1b_$m.tsv \
    > score_1b_$m.log 2>&1
  echo "score 1b $m rc=$?"
done
$PY compare.py 1b
$T $PY $OFF test --model evo2_40b --checkpoint "$CK40" --fp8 mmaemu --out test_40b_mmaemu.json > test_40b_mmaemu.log 2>&1
echo "test 40b mmaemu rc=$?"
$PY compare.py 40b
rm -f brca1_40.fasta  # 0.3 MB; make_inputs.py writes it again
