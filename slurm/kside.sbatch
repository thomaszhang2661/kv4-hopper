#!/bin/bash
# K-side CUDA prototype: build (JIT, sm_90a), correctness, bandwidth sweep (bench/kside_proto.py).
#   sbatch -J kv4kside -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 --export=ALL,KV4_COMMIT=<hash> slurm/kside.sbatch [--tag v1]
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:40:00
#SBATCH -o logs/%x-%j.out

set -uo pipefail
module load cuda/12.8.0 2>/dev/null || true
source /scratch/$USER/envs/kv4base/bin/activate
export PYTHONPATH=$SLURM_SUBMIT_DIR
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext
export TORCH_CUDA_ARCH_LIST=9.0a
export MAX_JOBS=8
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "KV4_COMMIT=${KV4_COMMIT:-unset}"
python bench/kside_proto.py "$@"
