#!/bin/bash
# FOSCAR comparison (11-12): bench/foscar_compare.py in the foscar env (same torch 2.14.1 + FlashInfer 0.7.3 as
# kv4base); our CUDA extensions are rebuilt for this env in their own extension dir.
#   sbatch -J kv4fos -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 --export=ALL,KV4_COMMIT=<hash> slurm/foscar.sbatch
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail
module load cuda/12.8.0 2>/dev/null || true
source /scratch/$USER/envs/foscar/bin/activate
export PYTHONPATH=$SLURM_SUBMIT_DIR TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_foscar TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=8
export FLASHINFER_WORKSPACE_BASE=/scratch/$USER/flashinfer_cache_foscar
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "KV4_COMMIT=${KV4_COMMIT:-unset}"
python bench/foscar_compare.py --foscar /scratch/$USER/ext/FOSCAR "$@"
