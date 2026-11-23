#!/bin/bash
# vLLM with the kv4-hopper plugin (11-12): our kernels JIT-built for the vLLM env's torch (CUDA 13) with nvcc 13.2,
# run through the CUDA 13 forward-compat libs. Runs `python $KV4_SCRIPT "$@"` (default bench/vllm_kv4_check.py).
# 11-22: KV4_ENV=fa4 runs in /scratch/$USER/envs/vllm_fa4 (vLLM 0.34 dev + FA #206 + vLLM #63139, slurm/fa4_env.sh)
# instead of the vLLM 0.33.0 env; separate torch extension dir; the plugin is pip-installed there (editable, no deps)
# on first use. Unset: unchanged.
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 00:45:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
EXT=/scratch/$USER/torch_ext_vllm$([ "$(basename $SLURM_SUBMIT_DIR)" = kv4-hopper ] || echo _$(basename $SLURM_SUBMIT_DIR))
if [ "${KV4_ENV:-}" = fa4 ]; then
  source slurm/fa4_env.sh
  EXT=/scratch/$USER/torch_ext_vllmfa4_$(basename $SLURM_SUBMIT_DIR)
  pip show -q kv4-vllm 2>/dev/null || pip install -q --no-deps --no-build-isolation -e $SLURM_SUBMIT_DIR/integrations/vllm_plugin
else
  module load cuda/13.2.0 2>/dev/null || true
  export HF_HOME=/scratch/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_CACHE_ROOT=/scratch/$USER/vllm_cache
  COMPAT=$(dirname $(find /scratch/$USER/envs/cuda13compat -name "libcuda.so.1" | head -1))
  # the compat libs only for drivers older than CUDA 13 (570 on the SXM nodes); with a newer driver (595 on d4077)
  # they must not be loaded (error 803)
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  if [ "${DRV:-0}" -lt 580 ]; then export LD_LIBRARY_PATH=$COMPAT:${LD_LIBRARY_PATH:-}; fi
  source /scratch/$USER/envs/vllm/bin/activate
fi
export KV4_VLLM=${KV4_VLLM:-1}  # the check script sets it per run itself
export PYTHONPATH=$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR/integrations/vllm_plugin TORCH_EXTENSIONS_DIR=$EXT TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=16
echo "host $(hostname) job ${SLURM_JOB_ID:-none} env ${KV4_ENV:-vllm033} $(python -c 'import vllm; print("vllm", vllm.__version__)' 2>&1 | tail -1)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "KV4_COMMIT=${KV4_COMMIT:-unset} nvcc $(nvcc --version | tail -1) ext $TORCH_EXTENSIONS_DIR"
python ${KV4_SCRIPT:-bench/vllm_kv4_check.py} "$@"
