#!/bin/bash
# 11-15: end-to-end UltraQuant (and TurboQuant) check of vllm-project/vllm#67789 on CUDA: stock vLLM 0.33, then a copy
# of the installed package with the PR's two attention-backend changes applied (bench/debug/pr67789_attn.diff).
#   sbatch -J kv4pr -p sharing --gres=gpu:h200:1 -t 00:55:00 --export=ALL,KV4_COMMIT=<hash> slurm/pr67789_check.sbatch
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -o logs/%x-%j.out
set -uo pipefail
module load cuda/13.2.0 2>/dev/null || true
export HF_HOME=/scratch/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_CACHE_ROOT=/scratch/$USER/vllm_cache
COMPAT=$(dirname $(find /scratch/$USER/envs/cuda13compat -name "libcuda.so.1" | head -1))
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "${DRV:-0}" -lt 580 ]; then export LD_LIBRARY_PATH=$COMPAT:${LD_LIBRARY_PATH:-}; fi
source /scratch/$USER/envs/vllm/bin/activate
cd "$SLURM_SUBMIT_DIR"
export VLLM_PLUGINS=""   # no kv4 plugin: stock backends only
unset PYTHONPATH
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "KV4_COMMIT=${KV4_COMMIT:-unset}"
S=bench/debug/pr67789_check.py
for dt in auto turboquant_k8v4 ultraquant_4bit; do for m in prefix chunked; do
  PR_TAG=base python $S $dt $m 2>&1 | grep -E "^RESULT|^vllm |Error|error:" | grep -v Warning
done; done
# the PR's diff on a private copy of the installed package (nothing in the env changes)
SP=$(python -c "import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))")
P=${TMPDIR:-/tmp}/pr67789_$SLURM_JOB_ID; mkdir -p $P && cp -r $SP/vllm $P/
(cd $P && patch -p1 --forward < $SLURM_SUBMIT_DIR/bench/debug/pr67789_attn.diff) || { echo "PATCH FAILED"; exit 1; }
for dt in turboquant_k8v4 ultraquant_4bit; do for m in prefix chunked; do
  PYTHONPATH=$P PR_TAG=pr67789 python $S $dt $m 2>&1 | grep -E "^RESULT|^vllm |Error|error:" | grep -v Warning
done; done
rm -rf $P
echo done
