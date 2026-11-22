#!/bin/bash
# Compile one CUDA source with nvcc and print ptxas messages (registers, spills, wgmma serialization reasons), 11-11.
#   sbatch -J kv4ptxas slurm/ptxas_info.sbatch kv4/cuda/decode_fp4.cu
#SBATCH -p short
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail
module load cuda/12.8.0 2>/dev/null || true
source /scratch/$USER/envs/kv4base/bin/activate
cd "$SLURM_SUBMIT_DIR"
INC=$(python -c "import torch.utils.cpp_extension as c, sysconfig; print(' '.join('-I'+p for p in c.include_paths()) + ' -I' + sysconfig.get_paths()['include'])")
src=${1:-kv4/cuda/decode_fp4.cu}
nvcc -std=c++20 -O3 --expt-relaxed-constexpr -gencode=arch=compute_90a,code=sm_90a --use_fast_math -Xptxas=-v -Ikv4/cuda $INC \
  -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ \
  -DTORCH_EXTENSION_NAME=probe -D_GLIBCXX_USE_CXX11_ABI=1 -c "$src" -o /tmp/probe_$USER.o 2>&1 | grep -v "^$" | grep -i -E "serializ|registers|spill|Function properties|Compiling entry|error" | head -60
