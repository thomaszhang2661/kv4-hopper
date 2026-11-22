#!/bin/bash
# Kernel checks (11-11): sink in the merge (bench/sink_check.py), then G = 8 (bench/gqa8_check.py); KV4_SCRIPTS overrides.
#   sbatch -J kv4kside -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 --export=ALL,KV4_COMMIT=<hash> slurm/decode_fp4.sbatch [--tag v1]
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:50:00
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
export FLASHINFER_WORKSPACE_BASE=/scratch/$USER/flashinfer_cache
for s in ${KV4_SCRIPTS:-bench/sink_check.py bench/gqa8_check.py}; do python $s; done
