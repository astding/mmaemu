#!/bin/bash
# H100 bit-exactness probe, Phase A (also the dry runs on the RTX 4060 / RTX 5070 Ti). Every GPU step runs under the
# host's GPU lock with a 10-minute limit; a failed step does not stop the others. Outputs: out/<step>/ (raw bits,
# per-call cuBLASLt configs, logs), out/status.txt (one line per step: rc and wall seconds), out/SHA256SUMS last.
# usage: run_all.sh <python> <data dir> [setup | none | <TE python>]
#   data dir: fp8-inputs/callNN.pt + truth/evo2_1b_w2k.json (the gh200 bundle's data/)
#   TE: setup (default) = te_setup.sh into <python>'s venv first; none = no TE steps;
#       a path = an existing TE venv's python (dry runs)
# env: PROBE_GPU_WRAP  GPU lock prefix (default: flock on out/.gpu.lock); PROBE_CUDA_HOME (nvcc for the mma.sync kit)
set -u
J=$(cd "$(dirname "$0")" && pwd); PY=$1; DATA=$(cd "$2" && pwd); TE=${3:-setup}
O=$J/out; mkdir -p "$O"; ST=$O/status.txt
export PROBE_SRC=$J/src PYTHONPATH=$J/src PYTHONDONTWRITEBYTECODE=1 MMAEMU_CUBLASLT_HEADER=$J/include/cublasLt.h
WRAP=${PROBE_GPU_WRAP:-flock $O/.gpu.lock}
SP=$($PY -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
note() { echo "$(date -u +%FT%TZ) $*" | tee -a "$ST"; }

step() {  # step <name> <python> <script> [args...]: GPU step under the lock, 600 s
  local n=$1 py=$2; shift 2; local d=$O/$n; mkdir -p "$d"; local t0=$(date +%s)
  (cd "$J/probe" && PROBE_OUT=$d BRINGUP_OUT=$d BRINGUP_CACHE=$O/.cache CUBLASLT_LOG_LEVEL=2 \
     CUBLASLT_LOG_FILE=$d/lt.log $WRAP timeout 600 "$py" "$@" > "$d/stdout.log" 2>&1)
  note "$n rc=$? $(( $(date +%s) - t0 ))s"
}

te_env() {  # environment of the TE steps
  local tsp; tsp=$($1 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
  export NVTE_DISABLE_NVRTC=1 LD_LIBRARY_PATH=$(ls -d "$tsp"/nvidia/*/lib | tr '\n' ':')
  export CUDA_PATH=$tsp/nvidia/cuda_nvrtc   # TE loads libnvrtc from CUDA_HOME / CUDA_PATH first: the venv wheel
}

note "start $(hostname -s | sha256sum | cut -c1-8) python=$PY te=$TE"
if [ "$TE" = setup ]; then          # before any measurement: nothing installs into the venv while steps run
  t0=$(date +%s); timeout 1200 bash "$J/te_setup.sh" "$PY" "$O/te_setup"; rc=$?; echo "rc=$rc" > "$O/te_setup/done"
  note "te_setup rc=$rc $(( $(date +%s) - t0 ))s"
  TEPY=$PY
elif [ "$TE" != none ]; then TEPY=$TE; fi

step env "$PY" env_record.py
step m1_torch "$PY" m1_cvec.py
step m2_torch "$PY" m2_evo2.py --data "$DATA"
step m3_torch "$PY" m3_promo.py
step m1_kit "$PY" "$J/src/tools/bringup/kits/c6/run_cublas_kit.py" --set MM_cublas64 --out "$O/m1_kit/kit.json"

if [ "${TEPY:-}" ]; then
  if [ "$TE" != setup ] || grep -qs "rc=0" "$O/te_setup/done"; then
    ( te_env "$TEPY"
      step m2_te "$TEPY" m2_evo2.py --data "$DATA" --te
      step m1_te "$TEPY" m1_cvec.py --te
      step m3_te "$TEPY" m3_promo.py --te )
  fi
fi

# M6: heuristic queries (no kernels; no per-call log), then the smallest live row of every split combination
d=$O/m6_heur; mkdir -p "$d"; t0=$(date +%s)
(cd "$J/probe" && $WRAP timeout 600 "$PY" "$J/src/tools/record_heuristics.py" --grid c5,evo2,evo2_small_m \
   --ws 0,1,4,32 --out "$d/heur.npz" > "$d/stdout.log" 2>&1); note "m6_heur rc=$? $(( $(date +%s) - t0 ))s"
(cd "$J/probe" && PROBE_OUT=$O/m6_heur "$PY" m6_split_jobs.py "$d/heur.npz" "$d/split_jobs.json" >> "$d/stdout.log" 2>&1)
note "m6_jobs rc=$?"
[ -f "$d/split_jobs.json" ] && step m6_live/p1 "$PY" m4_run.py "$d/split_jobs.json" p1

for f in "$J"/m4_jobs/chunk_*.json; do step "m4/p1/$(basename "$f" .json)" "$PY" m4_run.py "$f" p1; done

step m5_epilogue "$PY" "$J/src/tools/bringup/kits/epilogue_ties/record_epilogue_ties.py" "$O/m5_epilogue/epilogue_ties.pt" 8
step m5_zero "$PY" "$J/src/tools/bringup/kits/zero_specials/scripts/zero_specials.py"

# mma.sync kit (instruction level): compile outside the lock, then run
CAP=$($PY -c "import torch; print('%d%d' % torch.cuda.get_device_capability(0))")
case $CAP in 90) ARCH=9.0a; NATIVE=;; 89) ARCH=8.9; NATIVE=MM_ada;; 120) ARCH=12.0; NATIVE=MM_sm120;; *) ARCH=;; esac
if [ "$ARCH" ]; then
  CH=${PROBE_CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc || echo /usr/local/cuda-12.8/bin/nvcc)")")}
  d=$O/m1_mma; mkdir -p "$d"; t0=$(date +%s)
  (cd "$J/probe" && CUDA_HOME=$CH PATH=$CH/bin:$(dirname "$PY"):$PATH BRINGUP_OUT=$d BRINGUP_CACHE=$O/.cache timeout 900 "$PY" \
     "$J/src/tools/bringup/kits/c6/run_kit.py" --sets MM_sm90mma --arch $ARCH --compile-only > "$d/compile.log" 2>&1)
  note "m1_mma_compile rc=$? $(( $(date +%s) - t0 ))s"
  CUDA_HOME=$CH PATH=$CH/bin:$(dirname "$PY"):$PATH step m1_mma "$PY" "$J/src/tools/bringup/kits/c6/run_kit.py" \
    --sets MM_sm90mma TBX_B200 TBX_H100c0 $NATIVE --arch $ARCH --out "$O/m1_mma/kit.json"
fi

# second pass of every live recorder job (determinism across processes)
[ -f "$O/m6_heur/split_jobs.json" ] && step m6_live/p2 "$PY" m4_run.py "$O/m6_heur/split_jobs.json" p2
for f in "$J"/m4_jobs/chunk_*.json; do step "m4/p2/$(basename "$f" .json)" "$PY" m4_run.py "$f" p2; done

(cd "$J/probe" && PROBE_OUT=$O "$PY" summarize.py "$O" > "$O/summary.txt" 2>&1); note "summary rc=$?"
note "DONE $(du -sh "$O" --exclude=.cache | cut -f1)"     # nothing is written into out/ after the manifest
(cd "$O" && find . -type f ! -name 'SHA256SUMS*' ! -name .gpu.lock ! -path './.cache/*' -print0 | sort -z \
   | xargs -0 sha256sum > SHA256SUMS.tmp && mv SHA256SUMS.tmp SHA256SUMS)
