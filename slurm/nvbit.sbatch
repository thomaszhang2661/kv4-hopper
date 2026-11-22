#!/bin/bash
# Executed instructions per kernel, by opcode, with NVBit's opcode_hist (no performance counters needed;
# Nsight Compute is blocked on Explorer). One decode call per implementation (bench/ncu_target.py).
#   sbatch -J kv4nvbit -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 slurm/nvbit.sbatch [impl ...]
# Logs go to logs/nvbit/<impl>.log; bench/nvbit_summary.py makes the table.
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:45:00
#SBATCH -o logs/%x-%j.out

set -uo pipefail
module load cuda/12.8.0 2>/dev/null || true
source /scratch/$USER/envs/kv4base/bin/activate
export FLASHINFER_WORKSPACE_BASE=/scratch/$USER/flashinfer_cache
export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext
export PYTHONPATH=$SLURM_SUBMIT_DIR
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/nvbit
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "KV4_COMMIT=${KV4_COMMIT:-unset}"
TOOL=$(find /scratch/$USER/src/nvbit_release_x86_64/tools/opcode_hist/build -name "opcode_hist.so" | head -1)
echo "tool $TOOL"

IMPLS=${*:-ours_bf16 ours_fp8 ours_int4_tok ours_int4a_tok ours_nvfp4 ours_mxfp4 fi_bf16 fi_fp8 fi_nvfp4 bd_int4}
for impl in $IMPLS; do
    echo "== $impl $(date +%T)"
    # PyTorch can overwrite LD_PRELOAD; CUDA_INJECTION64_PATH is NVBit's documented alternative.
    CUDA_INJECTION64_PATH=$TOOL MANGLED_NAMES=0 python bench/ncu_target.py --impl "$impl" > "logs/nvbit/$impl.log" 2>&1
    echo "rc=$? $(grep -c '^kernel ' logs/nvbit/$impl.log) kernels logged"
    grep -A3 "==KV4 BEGIN" "logs/nvbit/$impl.log" | head -4
done
python bench/nvbit_summary.py logs/nvbit/*.log
