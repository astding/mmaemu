#!/bin/bash
# Transformer Engine 2.3.0 in the probe venv, as for the H100 runs of 2026-10-14/15: the PyPI binary wheels plus the
# RTX 4060 environment's own transformer_engine_torch build (te_torch_2.3.0_ada.tar.gz, checked against
# local_te_so.sha256); it needs GLIBCXX_3.4.32 (Ubuntu: toolchain PPA, only if missing). torch must stay unchanged.
# usage: te_setup.sh <venv python> <out dir>      (pip from the venv, else uv)
set -u
PY=$1; O=$2; J=$(cd "$(dirname "$0")" && pwd); mkdir -p "$O"
SP=$($PY -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
exec > "$O/te-setup.log" 2>&1
set -x
if $PY -m pip --version; then pipx() { $PY -m pip "$@"; }; else pipx() { ${UV:-uv} pip "$1" --python "$PY" "${@:2}"; }; fi
TV=$($PY -c "import torch; print(torch.__version__)")
pipx freeze > "$O/pip-freeze-before.txt"
pipx install --no-deps transformer_engine==2.3.0 transformer_engine_cu12==2.3.0 || exit 1
pipx install pydantic importlib-metadata einops || exit 1
tar -C "$SP" -xzf "$J/te_torch_2.3.0_ada.tar.gz" || exit 1
(cd "$SP" && sha256sum -c "$J/local_te_so.sha256") || exit 1
pipx freeze > "$O/pip-freeze-after.txt"
diff "$O/pip-freeze-before.txt" "$O/pip-freeze-after.txt"
$PY -c "import torch; assert torch.__version__ == '$TV', torch.__version__; print('torch unchanged', torch.__version__)" || exit 1
LIBSTD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
if ! strings "$LIBSTD" | grep -qx GLIBCXX_3.4.32; then
  sudo -n add-apt-repository -y ppa:ubuntu-toolchain-r/test || exit 1
  sudo -n apt-get -o DPkg::Lock::Timeout=600 install -y libstdc++6 || exit 1
  strings "$LIBSTD" | grep -x GLIBCXX_3.4.32 || exit 1
fi
export NVTE_DISABLE_NVRTC=1 LD_LIBRARY_PATH=$(ls -d "$SP"/nvidia/*/lib | tr '\n' ':')
export CUDA_PATH=$SP/nvidia/cuda_nvrtc   # TE loads libnvrtc from CUDA_HOME / CUDA_PATH first: the venv wheel
$PY -c "import transformer_engine, transformer_engine.pytorch as te; print('TE OK', transformer_engine.__version__)" || exit 1
echo SETUP_DONE
