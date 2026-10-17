#!/bin/bash
# H100 split-K sweep (/data/quant/mmaemu_work/h100_splitk_sweep_plan.md). Heuristic queries (no kernels) in parts
# p0..p4, then the live calls of p5. Every GPU step runs under the GPU lock with a time limit; status.txt gets one
# line per step (rc, wall seconds); SHA256SUMS last.
#   p0  drift gate: re-record the probe's grid, compare with the shipped table's source row for row. NO-GO on the
#       H100 (mode h100) stops the run here (fetch, terminate, report). Mode dry continues (other GPUs differ).
#   p1  envelope (torch, te; ws0 on every 4th shape), p4 sealed held-out (counted only), p2 K dense, p3 model rows
#       (both alignments), p5 live jobs selected from p1-p3 on this host, run twice (p1, p2 passes: determinism
#       across processes). Query parts are saved in chunks of 500,000 (a timeout loses at most one chunk).
# usage: run_all.sh <python> <h100|dry>
# env: SWEEP_GPU_WRAP  GPU lock prefix (default: flock on out/.gpu.lock)
set -u
J=$(cd "$(dirname "$0")" && pwd); PY=$1; MODE=${2:?h100 or dry}
O=$J/out; mkdir -p "$O"; ST=$O/status.txt
export SWEEP_SRC=$J/src PYTHONPATH=$J/src PYTHONDONTWRITEBYTECODE=1 MMAEMU_CUBLASLT_HEADER=$J/include/cublasLt.h
WRAP=${SWEEP_GPU_WRAP:-flock $O/.gpu.lock}
note() { echo "$(date -u +%FT%TZ) $*" | tee -a "$ST"; }

step() {  # step <name> <limit s> <script> [args...]: GPU step under the lock, output in out/<name>/
  local n=$1 lim=$2; shift 2; local d=$O/$n; mkdir -p "$d"; local t0=$(date +%s)
  (cd "$J/sweep" && SWEEP_OUT=$d $WRAP timeout "$lim" "$PY" "$@" >> "$d/stdout.log" 2>&1)
  local rc=$?; note "$n $(basename "$1") ${*:2} rc=$rc $(( $(date +%s) - t0 ))s"; return $rc
}

finish() {
  (cd "$J/sweep" && "$PY" summarize.py "$O" > "$O/summary.txt" 2>&1); note "summary rc=$?"
  note "DONE $1 $(du -sh "$O" | cut -f1)"     # nothing is written into out/ after the manifest
  (cd "$O" && find . -type f ! -name 'SHA256SUMS*' ! -name .gpu.lock -print0 | sort -z \
     | xargs -0 sha256sum > SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS)
  exit 0
}

note "start $(hostname -s | sha256sum | cut -c1-8) python=$PY mode=$MODE commit=$(cat "$J/src/COMMIT")"
(cd "$J" && sha256sum -c --quiet MANIFEST.sha256) || { note "MANIFEST mismatch"; finish manifest; }
step env 300 env_record.py

# p0: the probe's command (850,320 queries, 87 s on the H100), then the gate
step p0 1200 "$J/src/tools/record_heuristics.py" --grid c5,evo2,evo2_small_m --ws 0,1,4,32 --out "$O/p0/heur.npz"
(cd "$J/sweep" && SWEEP_OUT=$O/p0 "$PY" drift.py "$O/p0/heur.npz" >> "$O/p0/stdout.log" 2>&1); rc=$?
note "drift rc=$rc $(grep -o 'drift gate: [A-Z-]*' "$O/p0/stdout.log" | tail -1)"
if [ $rc -ne 0 ] && [ "$MODE" = h100 ]; then finish NO-GO; fi

# the envelope (p1) and its sealed test (p4) first; then K dense (p2) and model rows (p3)
for p in torch te ws0; do step p1 2400 q.py p1 --profile $p; done
for p in torch te ws0; do step p4_sealed 900 q.py p4 --profile $p; done
for p in torch te; do step p2 1200 q.py p2 --profile $p; done
step p3 1800 q.py p3 --profile torch; step p3 1800 q.py p3 --profile torch --align 256
step p3 1800 q.py p3 --profile te;    step p3 1800 q.py p3 --profile te --align none

(cd "$J/sweep" && SWEEP_OUT=$O/p5_jobs timeout 1800 "$PY" live_jobs.py "$O" "$O/p5_jobs" > "$O/p5_jobs.log" 2>&1)
note "p5_jobs rc=$? $(tail -1 "$O/p5_jobs.log" | cut -c1-300)"
for f in "$O"/p5_jobs/{a,bc,other,fill}_*.json; do                   # pass 1: every job
  [ -f "$f" ] && step p5_live/p1 900 live_run.py "$f" "p1_$(basename "$f" .json)"
done
for f in "$O"/p5_jobs/a_*.json; do                                   # pass 2: the A split jobs again
  [ -f "$f" ] && step p5_live/p2 900 live_run.py "$f" "p2_$(basename "$f" .json)"
done
finish complete
