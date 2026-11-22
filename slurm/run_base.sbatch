#!/bin/bash
# Like run.sbatch, but in the reference-kernel env (/scratch/$USER/envs/kv4base: newer torch, Triton, FlashInfer).
#   sbatch -J fi --gres=gpu:h200:1 slurm/run_base.sbatch bench/some_script.py
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
module load cuda/12.8.0 2>/dev/null || true   # nvcc for FlashInfer's JIT
source /scratch/$USER/envs/kv4base/bin/activate
export HF_HOME=/scratch/$USER/hf_cache
export FLASHINFER_WORKSPACE_BASE=/scratch/$USER/flashinfer_cache
export PYTHONPATH=$SLURM_SUBMIT_DIR
cd "$SLURM_SUBMIT_DIR"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "no GPU"
python "$@"
