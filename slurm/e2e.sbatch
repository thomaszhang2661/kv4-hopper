#!/bin/bash
# End-to-end generation with the real FP4 kernel (bench/e2e_generate.py): needs transformers (fp4 conda env) and the
# CUDA extension, built here against the fp4 env's torch in its own extension dir.
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -o logs/%x-%j.out
set -uo pipefail
module load cuda/12.8.0 2>/dev/null || true
export HF_HOME=/scratch/$USER/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PATH=$PATH:/scratch/$USER/envs/kv4base/bin  # ninja for the JIT build (the fp4 env has none); appended, so python is unchanged
export PYTHONPATH=$SLURM_SUBMIT_DIR TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_fp4 TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=8
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "KV4_COMMIT=${KV4_COMMIT:-unset}"
~/.conda/envs/fp4/bin/python bench/e2e_generate.py "$@"
