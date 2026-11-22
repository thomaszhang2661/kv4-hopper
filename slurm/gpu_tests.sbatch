#!/bin/bash
# GPU test suite (11-19), on one Hopper GPU (11-20: plugin first, 1:45; pytest alone took 42 min in 11995516):
#   1. bench/plugin_cached_kv_check.py  in the vLLM env (vllm 0.33, CUDA 13): the plugin with the full recipe on
#      prefix-cache hits and chunked prefill (the vllm#46461 / #67789 triggers), Llama-3.1-8B
#   2. python -m pytest tests  in the kernel env (kv4base: torch cu126, Triton, nvcc 12.8): formats, Triton kernels,
#      CUDA kernels (tests/gpu/test_cuda_kernels.py runs the bench/*_check.py correctness sections)
# Last line: GPU TESTS PASSED / FAILED (exit status likewise).
#   sbatch -J kv4test -p gpu-short --gres=gpu:h200:1 --export=ALL,KV4_COMMIT=$(git rev-parse --short HEAD) slurm/gpu_tests.sbatch
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 01:45:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} KV4_COMMIT=${KV4_COMMIT:-unset}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
rc=0
(
  module load cuda/13.2.0 2>/dev/null || true
  export HF_HOME=/scratch/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_CACHE_ROOT=/scratch/$USER/vllm_cache
  COMPAT=$(dirname $(find /scratch/$USER/envs/cuda13compat -name "libcuda.so.1" | head -1))
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  if [ "${DRV:-0}" -lt 580 ]; then export LD_LIBRARY_PATH=$COMPAT:${LD_LIBRARY_PATH:-}; fi  # as slurm/vllm_kv4.sbatch
  source /scratch/$USER/envs/vllm/bin/activate
  export PYTHONPATH=$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR/integrations/vllm_plugin TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_vllm \
         TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=16
  python bench/plugin_cached_kv_check.py
) || rc=1
(
  module load cuda/12.8.0 2>/dev/null || true
  source /scratch/$USER/envs/kv4base/bin/activate
  export PYTHONPATH=$SLURM_SUBMIT_DIR TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=16
  python -m pytest -q -rs tests
) || rc=1
echo "GPU TESTS $([ $rc = 0 ] && echo PASSED || echo FAILED)"
exit $rc
