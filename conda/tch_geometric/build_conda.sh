#!/bin/bash

export PYTHON_VERSION=$1
export TORCH_VERSION=$2
export CUDA_VERSION=$3

echo Setup Cuda Args
export TORCH_CUDA_VERSION=$(echo $CUDA_VERSION | sed "s/cuda/cu/g")
echo "TORCH_CUDA_VERSION=$TORCH_CUDA_VERSION" >> $GITHUB_ENV

export CONDA_PYTORCH_CONSTRAINT="pytorch==${TORCH_VERSION%.*}.*"

if [ "${TORCH_CUDA_VERSION}" = "cpu" ]; then
  export CONDA_CUDATOOLKIT_CONSTRAINT="cpuonly  # [not osx]"
else
  case $TORCH_CUDA_VERSION in
    cu115)
      export CONDA_CUDATOOLKIT_CONSTRAINT="cudatoolkit==11.5.*"
      ;;
    cu113)
      export CONDA_CUDATOOLKIT_CONSTRAINT="cudatoolkit==11.3.*"
      ;;
    cu111)
      export CONDA_CUDATOOLKIT_CONSTRAINT="cudatoolkit==11.1.*"
      ;;
    cu102)
      export CONDA_CUDATOOLKIT_CONSTRAINT="cudatoolkit==10.2.*"
      ;;
    cu101)
      export CONDA_CUDATOOLKIT_CONSTRAINT="cudatoolkit==10.1.*"
      ;;
    *)
      echo "Unrecognized CUDA_VERSION=$TORCH_CUDA_VERSION"
      exit 1
      ;;
  esac
fi

echo "PyTorch $TORCH_VERSION+$TORCH_CUDA_VERSION"
echo "- $CONDA_PYTORCH_CONSTRAINT"
echo "- $CONDA_CUDATOOLKIT_CONSTRAINT"

conda mambabuild . -c pytorch -c default -c nvidia -c conda-forge --output-folder "$HOME/conda-bld"