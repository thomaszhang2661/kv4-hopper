#!/bin/bash
# Build the extra Python environments on a CPU node (11-12, approved by Jian; keeps pip off the login node).
# Existing envs (kv4base, fp4) are not touched.
#   sbatch slurm/setup_envs.sbatch foscar   # FOSCAR (MIT, FutureMLS-Lab) with the same torch / FlashInfer as kv4base
#   sbatch slurm/setup_envs.sbatch vllm     # vLLM 0.33.0 for end-to-end serving baselines
#   sbatch slurm/setup_envs.sbatch lmdeploy # LMDeploy (TurboMind) INT4 / INT8 KV baseline (11-14)
#   sbatch slurm/setup_envs.sbatch compat   # CUDA 13 forward-compatibility libs: vLLM 0.33.0 ships only cu130 wheels,
#                                           # the H200 nodes run driver 570 (CUDA 12.8); data-center GPUs can use these
#SBATCH -p short
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
set -euxo pipefail
which=$1
E=/scratch/$USER/envs
PY=$E/kv4base/bin/python   # Python 3.11, the interpreter kv4base uses
export PIP_CACHE_DIR=/scratch/$USER/pip_cache
case $which in
  foscar)
    mkdir -p /scratch/$USER/ext
    [ -d /scratch/$USER/ext/FOSCAR ] || git clone https://github.com/FutureMLS-Lab/FOSCAR.git /scratch/$USER/ext/FOSCAR
    git -C /scratch/$USER/ext/FOSCAR log -1 --format='FOSCAR commit %H %ci'
    $PY -m venv $E/foscar
    $E/foscar/bin/pip install -U pip
    $E/foscar/bin/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
    $E/foscar/bin/pip install ninja numpy pyyaml
    $E/foscar/bin/pip install -e "/scratch/$USER/ext/FOSCAR[cu12,bench]"
    $E/foscar/bin/pip list | grep -iE "^torch |flashinfer|cutlass|tvm|foscar|cuda-python"
    ;;
  vllm)
    $PY -m venv $E/vllm
    $E/vllm/bin/pip install -U pip
    $E/vllm/bin/pip install vllm==0.33.0
    $E/vllm/bin/pip list | grep -iE "^torch |^vllm|flashinfer|triton"
    ;;
  kv4plugin)  # our own vLLM plugin (integrations/vllm_plugin), editable, no dependencies pulled
    $E/vllm/bin/pip install --no-deps -e $SLURM_SUBMIT_DIR/integrations/vllm_plugin
    $E/vllm/bin/pip list | grep -i kv4
    ;;
  lmdeploy)  # 11-14: LMDeploy (TurboMind) for the INT4 / INT8 KV baseline on H200 (its own CUDA 12 wheels and torch)
    $PY -m venv $E/lmdeploy
    $E/lmdeploy/bin/pip install -U pip
    # 11-14 (docs/lit/lmdeploy_int4_kv.md): a bare `pip install lmdeploy` pulls PyPI torch 2.12.1 (CUDA 13.0, needs
    # driver >= 580); the H200 SXM nodes run 570 (CUDA 12.8). LMDeploy 0.18.0's wheel is built with CUDA 12.8, so pin a
    # cu128 torch first (the last cu128 build is 2.11.0)
    $E/lmdeploy/bin/pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
    $E/lmdeploy/bin/pip install lmdeploy==0.18.0
    $E/lmdeploy/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
    mkdir -p /scratch/$USER/ext
    [ -d /scratch/$USER/ext/lmdeploy ] || git clone --depth 1 --branch v0.18.0 https://github.com/InternLM/lmdeploy.git /scratch/$USER/ext/lmdeploy
    $E/lmdeploy/bin/pip list | grep -iE "^torch |lmdeploy|triton|nvidia-cuda-runtime"
    $E/lmdeploy/bin/python -c "import lmdeploy; print('lmdeploy', lmdeploy.__version__)"
    ;;
  compat)
    module load miniconda3/25.9.1
    conda create -y -p $E/cuda13compat -c conda-forge --override-channels cuda-compat=13.0.2
    find $E/cuda13compat -name "libcuda.so*" | head
    ;;
esac
echo done
