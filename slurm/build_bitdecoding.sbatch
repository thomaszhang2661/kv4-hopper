#!/bin/bash
# Fetch BitDecoding (branch hopper) and CUTLASS at pinned commits and compile our binding, on a CPU node.
#   sbatch slurm/build_bitdecoding.sbatch
# Sources go to /scratch/$USER/src (not into the repo); the build to /scratch/$USER/src/build_bitdecoding.
#SBATCH -J kv4bdbuild
#SBATCH -p short
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
module load cuda/12.8.0 2>/dev/null || true
source /scratch/$USER/envs/kv4base/bin/activate
export PYTHONPATH=$SLURM_SUBMIT_DIR
export MAX_JOBS=16
cd "$SLURM_SUBMIT_DIR"
ROOT=/scratch/$USER/src
mkdir -p "$ROOT"
BD_COMMIT=$(python -c "from baselines.bitdecoding.build import BITDECODING_COMMIT as c; print(c)")
CT_COMMIT=$(python -c "from baselines.bitdecoding.build import CUTLASS_COMMIT as c; print(c)")

if [ ! -d "$ROOT/BitDecoding-hopper/.git" ]; then
    git clone -q -b hopper https://github.com/OpenBitSys/BitDecoding.git "$ROOT/BitDecoding-hopper"
fi
git -C "$ROOT/BitDecoding-hopper" checkout -q "$BD_COMMIT"
if [ ! -d "$ROOT/cutlass/.git" ]; then
    git clone -q https://github.com/NVIDIA/cutlass.git "$ROOT/cutlass"
fi
git -C "$ROOT/cutlass" checkout -q "$CT_COMMIT"
echo "BitDecoding $(git -C "$ROOT/BitDecoding-hopper" rev-parse --short HEAD), CUTLASS $(git -C "$ROOT/cutlass" rev-parse --short HEAD), nvcc $(nvcc --version | tail -1)"

time python -m baselines.bitdecoding.build
