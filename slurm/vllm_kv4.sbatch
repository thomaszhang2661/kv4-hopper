#!/bin/bash
# vLLM with the kv4-hopper plugin (11-12): our kernels JIT-built for the vLLM env's torch (CUDA 13) with nvcc 13.2,
# run through the CUDA 13 forward-compat libs. Runs `python $KV4_SCRIPT "$@"` (default bench/vllm_kv4_check.py).
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 00:45:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail
module load cuda/13.2.0 2>/dev/null || true
export HF_HOME=/scratch/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_CACHE_ROOT=/scratch/$USER/vllm_cache
COMPAT=$(dirname $(find /scratch/$USER/envs/cuda13compat -name "libcuda.so.1" | head -1))
# the compat libs only for drivers older than CUDA 13 (570 on the SXM nodes); with a newer driver (595 on d4077)
# they must not be loaded (error 803)
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "${DRV:-0}" -lt 580 ]; then export LD_LIBRARY_PATH=$COMPAT:${LD_LIBRARY_PATH:-}; fi
source /scratch/$USER/envs/vllm/bin/activate
export KV4_VLLM=${KV4_VLLM:-1}  # the check script sets it per run itself
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR/integrations/vllm_plugin TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_vllm$([ "$(basename $SLURM_SUBMIT_DIR)" = kv4-hopper ] || echo _$(basename $SLURM_SUBMIT_DIR)) TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=16
echo "host $(hostname) job ${SLURM_JOB_ID:-none}"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "KV4_COMMIT=${KV4_COMMIT:-unset} nvcc $(nvcc --version | tail -1)"
python ${KV4_SCRIPT:-bench/vllm_kv4_check.py} "$@"
