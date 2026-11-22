#!/bin/bash
# Compile one CUDA source for sm_90a, print ptxas register/spill info, and dump the SASS of every kernel to
# logs/sass-<jobid>/<n>.sass (one file per function, names in index.txt), 11-11. CPU only.
#   sbatch -J kv4sass slurm/sass_dump.sbatch kv4/cuda/decode_fp4.cu
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
obj=/tmp/sass_${USER}_${SLURM_JOB_ID}.o
nvcc -std=c++20 -O3 --expt-relaxed-constexpr -gencode=arch=compute_90a,code=sm_90a --use_fast_math -Xptxas=-v -Ikv4/cuda $INC \
  -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ \
  -DTORCH_EXTENSION_NAME=probe -D_GLIBCXX_USE_CXX11_ABI=1 -c "$src" -o $obj 2>&1 | grep -v "^$" | grep -i -E "registers|spill|Compiling entry|error" | head -60
out=logs/sass-$SLURM_JOB_ID; mkdir -p $out
cuobjdump -sass $obj | awk -v d=$out '
  /Function : / { n++; f = d "/" n ".sass"; print n, $3 >> (d "/index.txt") }
  n { print > f }'
for f in $out/*.sass; do echo "$f $(grep -cE '^\s+/\*[0-9a-f]{4}\*/' $f) instructions"; done
rm -f $obj
