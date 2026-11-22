#!/bin/bash
# Build NVBit's opcode_hist tool (CPU only). NVBit 1.8.1 must already be unpacked in /scratch/$USER/src:
#   curl -sL https://github.com/NVlabs/NVBit/releases/download/v1.8.1/nvbit-Linux-x86_64-1.8.1.tar.bz2 | tar xj
#   sbatch -J kv4nvbitbuild slurm/build_nvbit.sbatch
#SBATCH -p short
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
module load cuda/12.8.0 cmake/3.30.2
T=/scratch/$USER/src/nvbit_release_x86_64/tools
cmake -S $T/opcode_hist -B $T/opcode_hist/build -DNVBIT_TOOL_CUDA_ARCHITECTURES=90a
cmake --build $T/opcode_hist/build -j 8
find $T/opcode_hist/build -name "*.so"
