#!/bin/bash

echo Install Conda Build
#micromamba install conda-build conda-verify --yes
#micromamba install boa -c conda-forge

echo Setup Cuda Args
export TORCH_CUDA_VERSION=$(echo $CUDA_VERSION | sed "s/cuda/cu/g")
echo "TORCH_CUDA_VERSION=$TORCH_CUDA_VERSION" >> $GITHUB_ENV