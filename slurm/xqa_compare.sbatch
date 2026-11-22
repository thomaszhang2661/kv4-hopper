#!/bin/bash
# 11-18: XQA FP8 (TensorRT-LLM decode kernel via FlashInfer) vs FlashInfer fa2 FP8 vs our NVFP4, one job (bench/xqa_compare.py).
#   sbatch -J kv4xqa -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 --export=ALL,KV4_COMMIT=<hash> slurm/xqa_compare.sbatch --tag v1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:55:00
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
# 11-21: FLUSHES="write read" times every shape under both L2 flush modes in this job (bench_decode.timeit, KV4_FLUSH);
# the default is one run with the write flush, as in every earlier job. XTAG names the output; the mode is appended.
if [ -n "${FLUSHES:-}" ]; then
  for f in $FLUSHES; do KV4_FLUSH=$f python bench/xqa_compare.py "$@" --tag "${XTAG:-v1}_$f"; done
else
  python bench/xqa_compare.py "$@"
fi
