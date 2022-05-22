#!/bin/bash

echo Setup Python
export PATH=/opt/python/$PYTHON_ALIAS/bin/:$PATH
export PYTHON_INCLUDE_DIRS=$(python -c "from sysconfig import get_paths as gp; print(gp()['include'])")
export PYTHON_SITE_DIR=$(python -c "import site; print(site.getsitepackages()[0])")

echo Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
export PATH=$HOME/.cargo/bin:$PATH

echo Install PyTorch $TORCH_VERSION+$CUDA_VERSION
pip install numpy typing-extensions dataclasses
pip install --no-index --no-cache-dir torch==$TORCH_VERSION -f https://download.pytorch.org/whl/$CUDA_VERSION/torch_stable.html
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA:', torch.version.cuda)"

echo "PATH=$PATH" >> $GITHUB_ENV
echo "PYTHON_INCLUDE_DIRS=$PYTHON_INCLUDE_DIRS" >> $GITHUB_ENV
echo "LIBTORCH=$PYTHON_SITE_DIR/torch" >> $GITHUB_ENV
echo "LD_LIBRARY_PATH=$PYTHON_SITE_DIR/torch/lib:$LD_LIBRARY_PATH" >> $GITHUB_ENV