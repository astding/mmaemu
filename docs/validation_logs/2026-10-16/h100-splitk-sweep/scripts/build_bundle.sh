#!/bin/bash
# Build the H100 split-K sweep bundle: the mmaemu source tree at COMMIT (mmaemu/, tools/, pyproject.toml), the sweep
# scripts, the shape lists (sweep/make_shapes.py, deterministic), the probe's heuristic recording (ref/: the source
# of heur_h100pcie_lt12.8.3.14, for the drift gate), cublasLt.h (12.8, tile/stage names of the logged configs) and
# run_all.sh; MANIFEST.sha256 inside; tarball h100_sweep.tar.gz. usage: build_bundle.sh
set -eu
H=$(cd "$(dirname "$0")" && pwd); B=$H/bundle; WT=/data/quant/mmaemu_work/wt/hopper; COMMIT=57bcf0d
rm -rf "$B"; mkdir -p "$B/src"
git -C $WT archive $COMMIT mmaemu tools pyproject.toml | tar -x -C "$B/src"
git -C $WT rev-parse $COMMIT > "$B/src/COMMIT"
cp -a "$H/sweep" "$H/shapes" "$H/include" "$H/ref" "$H/run_all.sh" "$B/"
find "$B" -name __pycache__ -prune -exec rm -rf {} +
find "$B" -name .ruff_cache -prune -exec rm -rf {} +
(cd "$B" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
tar -C "$H" -czf "$H/h100_sweep.tar.gz" bundle
sha256sum "$H/h100_sweep.tar.gz" | tee "$H/h100_sweep.tar.gz.sha256"
du -sh "$H/h100_sweep.tar.gz"
