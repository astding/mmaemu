#!/bin/bash
# examples/evo2_offload.py with --profile, against the runs of this log and of the 40B log of 15 October:
#   bash run_checks.sh <python> <evo2_1b_base.pt> <evo2_40b.pt> <BRCA1 data dir of make_inputs.py>
# the 1B: test and score (40 BRCA1 windows) with each profile; the 40B: test with hopper-te. Writes here.
set -u
H=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$H/../../../../.." && pwd)
PY=$1; CK1=$2; CK40=$3; DATA=$4
cd "$H" || exit 1
export PYTHONPATH=$ROOT
OFF=$ROOT/examples/evo2_offload.py
T="/usr/bin/time -f wall_%e_s_maxrss_%M_kB"
echo "examples/evo2_offload.py of $(git -C "$ROOT" rev-parse HEAD) clean=$(git -C "$ROOT" status --porcelain | wc -l)"
$PY make_inputs.py --data "$DATA" || exit 1
for p in ada-te hopper-te; do
  $T $PY $OFF test --model evo2_1b_base --checkpoint "$CK1" --profile $p --out test_1b_$p.json > test_1b_$p.log 2>&1
  echo "test 1b $p rc=$?"
  $T $PY $OFF score --model evo2_1b_base --checkpoint "$CK1" --profile $p --input brca1_40.fasta \
    --out score_1b_$p.tsv > score_1b_$p.log 2>&1
  echo "score 1b $p rc=$?"
done
$PY compare.py 1b
p=hopper-te
$T $PY $OFF test --model evo2_40b --checkpoint "$CK40" --profile $p --out test_40b_$p.json > test_40b_$p.log 2>&1
echo "test 40b $p rc=$?"
$PY compare.py 40b
rm -f brca1_40.fasta  # 0.3 MB; make_inputs.py writes it again
