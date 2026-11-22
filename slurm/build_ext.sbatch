#!/bin/bash
# Pre-build the CUDA extensions on a CPU node (11-10), so GPU jobs only load them. No dependency is needed: a GPU job
# that starts while this one compiles waits on torch's build lock, and one that starts later loads the cached build.
#   sbatch -J kv4build slurm/build_ext.sbatch [module ...]      (default: decode_int4 decode_fp4 merge)
#SBATCH -p short
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out

set -uo pipefail
module load cuda/12.8.0 2>/dev/null || true
source /scratch/$USER/envs/kv4base/bin/activate
export PYTHONPATH=$SLURM_SUBMIT_DIR TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext MAX_JOBS=16 TORCH_CUDA_ARCH_LIST=9.0a
cd "$SLURM_SUBMIT_DIR"
for m in ${*:-decode_int4 decode_fp4 merge}; do
  echo "== $m $(date +%T)"
  python -c "import importlib; importlib.import_module('kv4.cuda.$m').ext(); print('built $m')" 2>&1 | grep -v "^\[" | tail -5
done
