#!/usr/bin/env bash
# Run the CPU CI steps of .github/workflows/ci.yml on this machine (no GPU, nothing published).
#
#   tools/ci_local.sh                     # all jobs: lint test nvrtc dist
#   tools/ci_local.sh lint dist           # selected jobs
#   MMAEMU_CI_TORCH="2.6.0 2.7.1" tools/ci_local.sh test    # T0+T1 against these CPU torch versions (default 2.7.1)
#
# Scratch venvs (uv) live in $MMAEMU_CI_DIR (default: a new temporary directory). Each step prints PASS / FAIL /
# PENDING / SKIP; PENDING means the step's target is missing from this tree.
# Exit status: 0 = every step PASS (or SKIP), 1 = a step FAILED, 3 = no failure but a step is PENDING.
set -u -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CI="${MMAEMU_CI_DIR:-$(mktemp -d -t mmaemu-ci-XXXXXX)}"
PY_VERSION="${MMAEMU_CI_PYTHON:-3.12}"
TORCH_VERSIONS="${MMAEMU_CI_TORCH:-2.7.1}"
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"
CPU_TIERS="not gpu and not slow and not truth and not profile_device"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" PIP_DISABLE_PIP_VERSION_CHECK=1
# GitHub runners have 4 cores; on a loaded shared host, unbounded OpenMP threads can slow torch down by orders of
# magnitude (oversubscription), so the CPU tiers run with 4 threads unless told otherwise.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
unset PYTHONPATH CUDA_HOME
mkdir -p "$CI"
UV="$(command -v uv || true)"
if [ -z "$UV" ]; then echo "ci_local.sh needs uv (https://docs.astral.sh/uv/)" >&2; exit 2; fi

declare -a RESULTS=()
record() { RESULTS+=("$1|$2|$3"); printf '%-8s %-12s %s\n' "$1" "$2" "$3"; }
step() {  # step <job> <name> <command...>
  local job="$1" name="$2"; shift 2
  local log="$CI/log_${job}_${name//[^A-Za-z0-9]/_}.txt"
  if "$@" >"$log" 2>&1; then record PASS "$job" "$name"; else record FAIL "$job" "$name (log: $log)"; tail -20 "$log"; fi
}
pending() { record PENDING "$1" "$2"; }

venv() {  # venv <dir> <packages...>: create a uv venv with packages; prints nothing on success
  local dir="$1"; shift
  [ -x "$dir/bin/python" ] || "$UV" venv -q --python "$PY_VERSION" "$dir" || return 1
  [ $# -eq 0 ] || "$UV" pip install -q --python "$dir/bin/python" "$@"
}
torch_venv() {  # torch_venv <dir> <torch version|latest> <extra packages...>
  local dir="$1" tv="$2"; shift 2
  local spec="torch==$tv"; [ "$tv" = latest ] && spec=torch
  venv "$dir" && "$UV" pip install -q --python "$dir/bin/python" "$spec" --index-url "$TORCH_CPU_INDEX" \
    && { [ $# -eq 0 ] || "$UV" pip install -q --python "$dir/bin/python" "$@"; }
}

job_lint() {
  local V="$CI/venv-lint"
  if ! venv "$V" ruff==0.17.0 actionlint-py==1.7.12.25 pyyaml==6.0.3 shellcheck-py; then
    record FAIL lint "tool install"; return; fi
  cd "$REPO" || return
  step lint ruff "$V/bin/ruff" check .
  step lint actionlint env PATH="$V/bin:$PATH" "$V/bin/actionlint" .github/workflows/ci.yml .github/workflows/release.yml
  step lint check_docs "$V/bin/python" tools/check_docs.py
  if [ -f mmaemu/backends/cuda/bin/manifest.json ]; then
    step lint "build_fatbin --check" "$V/bin/python" tools/build_fatbin.py --check
    step lint "manifest sha256" "$V/bin/python" tools/check_dist.py --verify-manifest
  else
    record SKIP lint "build_fatbin --check: no committed fatbins (committed at the release-candidate freeze)"
  fi
  if grep -rnE 'torch[.]equal[(]' tests tools >"$CI/banned_equal.txt"; then
    record FAIL lint "banned torch.equal call in tests/ or tools/: $(wc -l <"$CI/banned_equal.txt") hit(s), see $CI/banned_equal.txt"
  else record PASS lint "no torch.equal call in tests/ and tools/"; fi
  if [ ! -f tools/bench_kernels.py ]; then pending lint "perf_counter ban: tools/bench_kernels.py missing"
  elif grep -nE 'perf_counter' tools/bench_kernels.py; then record FAIL lint "perf_counter in tools/bench_kernels.py"
  else record PASS lint "no perf_counter in tools/bench_kernels.py"; fi
  step lint "workflow policy" "$V/bin/python" tests/test_packaging.py --review-workflows
}

job_test() {
  local tv
  for tv in $TORCH_VERSIONS; do
    local V="$CI/venv-test-torch$tv"
    if ! torch_venv "$V" "$tv" numpy pytest pyyaml; then record FAIL test "torch $tv install"; continue; fi
    cd "$REPO" || return
    step test "T0+T1 torch $tv" env PYTHONPATH="$REPO" "$V/bin/python" -m pytest -q -p no:cacheprovider tests \
      -m "$CPU_TIERS" --durations=10
  done
}

job_nvrtc() {
  local V="$CI/venv-nvrtc"
  if ! torch_venv "$V" 2.7.1 numpy pytest "nvidia-cuda-nvrtc-cu12==12.8.*"; then record FAIL nvrtc "install"; return; fi
  cd "$REPO" || return
  if [ -f tests/test_nvrtc_compile.py ]; then
    step nvrtc "NVRTC compile test" env PYTHONPATH="$REPO" "$V/bin/python" -m pytest -q -p no:cacheprovider \
      tests/test_nvrtc_compile.py
  else pending nvrtc "tests/test_nvrtc_compile.py missing"; fi
}

job_dist() {
  local B="$CI/venv-build" D="$CI/dist" W="$CI/sdist-wheel" T="$CI/venv-wheel"
  if ! venv "$B" build==1.6.1 twine==7.0.0 packaging pip; then record FAIL dist "tool install"; return; fi
  rm -rf "$D" "$W"
  cd "$REPO" || return
  if [ ! -f mmaemu/_version.py ]; then pending dist "mmaemu/_version.py missing: no build"; return; fi
  step dist "build --sdist --wheel" "$B/bin/python" -m build --sdist --wheel --installer uv --outdir "$D"
  [ -n "$(ls "$D"/*.whl 2>/dev/null)" ] || { record FAIL dist "no wheel built"; return; }
  step dist "twine check --strict" "$B/bin/twine" check --strict "$D"/*
  local flag=--allow-missing-fatbins
  [ -f mmaemu/backends/cuda/bin/manifest.json ] && flag=--require-fatbins
  step dist "check_dist $flag" "$B/bin/python" tools/check_dist.py "$D"/* "$flag"
  step dist "wheel from the sdist" "$B/bin/python" -m pip wheel -q --no-deps -w "$W" "$D"/*.tar.gz
  step dist "check_dist --from-sdist" "$B/bin/python" tools/check_dist.py "$W"/*.whl "$D"/*.tar.gz --from-sdist
  rm -rf "$T"
  if ! torch_venv "$T" 2.7.1 "$(ls "$D"/*.whl)" pytest pyyaml; then record FAIL dist "fresh venv install"; return; fi
  cd "$CI" || return
  step dist "python -m mmaemu info (wheel)" "$T/bin/python" -m mmaemu info
  step dist "T0+T1 against the wheel" env MMAEMU_EXPECT_INSTALLED=1 "$T/bin/python" -m pytest -q -p no:cacheprovider \
    "$REPO/tests" -m "$CPU_TIERS"
  cd "$REPO" || return
}

JOBS=("$@"); [ ${#JOBS[@]} -eq 0 ] && JOBS=(lint test nvrtc dist)
echo "mmaemu CI (local) in $CI; jobs: ${JOBS[*]}"
for j in "${JOBS[@]}"; do
  case "$j" in lint|test|nvrtc|dist) "job_$j" ;; *) echo "unknown job $j" >&2; exit 2 ;; esac
done

echo; echo "summary ($CI):"
fail=0; pend=0
for r in "${RESULTS[@]}"; do
  IFS='|' read -r st job name <<<"$r"; printf '  %-8s %-6s %s\n' "$st" "$job" "$name"
  [ "$st" = FAIL ] && fail=$((fail + 1)); [ "$st" = PENDING ] && pend=$((pend + 1))
done
if [ $fail -gt 0 ]; then echo "ci_local: FAIL ($fail failed, $pend pending)"; exit 1; fi
if [ $pend -gt 0 ]; then echo "ci_local: PENDING ($pend step(s) with a missing target)"; exit 3; fi
echo "ci_local: PASS"
