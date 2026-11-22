#!/bin/bash
# Run a Python script on one Explorer GPU.
#   sbatch -J bench --gres=gpu:h200:1 slurm/run.sbatch bench/bench_decode.py
# Override resources on the command line, e.g. sbatch --gres=gpu:a100:1 -t 08:00:00 ...
# CPU-only (unit tests): sbatch -p short --gres=none -t 00:20:00 slurm/run.sbatch -m pytest -q
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
module load miniconda3/24.11.1
source activate fp4  # the FP4 project's env: torch 2.6 + triton

export HF_HOME=/scratch/$USER/hf_cache
export PYTHONPATH=$SLURM_SUBMIT_DIR
cd "$SLURM_SUBMIT_DIR"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no GPU"
python "$@"
