#!/bin/bash
# Transformer Engine 2.3.0 in the task venv, as for the H100 baseline of 2026-10-14: the PyPI binary wheels and the
# RTX 4060 environment's own transformer_engine_torch build (it needs GLIBCXX_3.4.32: Ubuntu toolchain PPA), and the
# evo2_1b_base checkpoint. usage: te_setup.sh <task root> <job dir>
set -u
R=$1; J=$2; PIP=$R/venv/bin/pip; PY=$R/venv/bin/python; SP=$R/venv/lib/python3.12/site-packages
exec > $J/out/te-setup.log 2>&1
set -x
$PIP freeze > $J/out/pip-freeze-before.txt
$PIP install --no-deps transformer_engine==2.3.0 transformer_engine_cu12==2.3.0 || exit 1
$PIP install pydantic importlib-metadata || exit 1
tar -C $SP -xzf $J/te_torch_2.3.0_ada.tar.gz || exit 1
(cd $SP && sha256sum -c $J/local_te_so.sha256) || exit 1
$PIP freeze > $J/out/pip-freeze-after.txt
diff $J/out/pip-freeze-before.txt $J/out/pip-freeze-after.txt
$PY -c "import torch; assert torch.__version__ == '2.7.1+cu128', torch.__version__; print('torch unchanged')" || exit 1
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test || exit 1
sudo apt-get -o DPkg::Lock::Timeout=600 install -y libstdc++6 || exit 1
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep -x GLIBCXX_3.4.32 || exit 1
HF_HOME=$R/cache/hf $PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('arcinstitute/evo2_1b_base'))" || exit 1
sha256sum $R/cache/hf/hub/models--arcinstitute--evo2_1b_base/snapshots/*/evo2_1b_base.pt
export NVTE_DISABLE_NVRTC=1 LD_LIBRARY_PATH=$(ls -d $SP/nvidia/*/lib | tr '\n' ':')
$PY -c "import transformer_engine, transformer_engine.pytorch as te; print('TE OK', transformer_engine.__version__)" || exit 1
echo SETUP_DONE
