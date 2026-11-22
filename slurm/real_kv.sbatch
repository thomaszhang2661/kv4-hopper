#!/bin/bash
# Real-KV kernel check (11-11): capture Llama-3.1-8B decode-step tensors with transformers (fp4 conda env), then check
# our CUDA kernels on them (kv4base env). Any Hopper GPU; the sharing H100 is fine.
#   sbatch -J kv4realkv -p sharing --gres=gpu:h100:1 --export=ALL,KV4_COMMIT=<hash> slurm/real_kv.sbatch
#SBATCH -p sharing
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:40:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail
module load cuda/12.8.0 2>/dev/null || true
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/scratch/$USER/hf_cache PYTHONPATH=$SLURM_SUBMIT_DIR
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "KV4_COMMIT=${KV4_COMMIT:-unset}"
OUT=/scratch/$USER/kv4_realkv/llama31_8b.pt
[ -f $OUT ] || ~/.conda/envs/fp4/bin/python bench/capture_kv.py --out $OUT
source /scratch/$USER/envs/kv4base/bin/activate
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext TORCH_CUDA_ARCH_LIST=9.0a
python bench/real_kv_check.py --data $OUT
