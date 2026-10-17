#!/bin/bash
# Build the H100 probe bundle: mmaemu source tree at COMMIT (mmaemu/, tools/, pyproject.toml) + the c6 audit's
# MM_cublas64 / MM_sm90mma kit sets, the probe scripts, cublasLt.h (12.8.3.14 wheel), the TE 2.3.0 files, the M4 job
# lists; MANIFEST.sha256 inside, tarball h100_probe.tar.gz. usage: build_bundle.sh [--src-only]
set -eu
H=$(cd "$(dirname "$0")" && pwd); B=$H/bundle; WT=/data/quant/mmaemu_work/wt/audit-fix; COMMIT=a79668a
SETS=/data/quant/tcemu_audit/work/c6-paper-identifiability/kit/sets
rm -rf "$B"; mkdir -p "$B/src"
git -C $WT archive $COMMIT mmaemu tools pyproject.toml | tar -x -C "$B/src"
git -C $WT rev-parse $COMMIT > "$B/src/COMMIT"
for s in MM_cublas64 MM_sm90mma; do cp -a $SETS/$s "$B/src/tools/bringup/kits/c6/sets/"; done
[ "${1:-}" = "--src-only" ] && exit 0
cp -a "$H/probe" "$H/include" "$H/te_setup.sh" "$H/run_all.sh" "$H/te_torch_2.3.0_ada.tar.gz" "$H/local_te_so.sha256" \
  "$H/m4_jobs" "$B/"
find "$B" -name __pycache__ -prune -exec rm -rf {} +
find "$B" -name .ruff_cache -prune -exec rm -rf {} +
(cd "$B" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
tar -C "$H" -czf "$H/h100_probe.tar.gz" bundle
sha256sum "$H/h100_probe.tar.gz" | tee "$H/h100_probe.tar.gz.sha256"
du -sh "$H/h100_probe.tar.gz"
