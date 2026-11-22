#!/bin/bash
# Build the env for the reference implementations (FlashInfer first), on a compute node.
#   sbatch slurm/setup_baselines.sbatch
# Lives in /scratch/$USER/envs/kv4base (a venv on the fp4 env's Python); the fp4 env is untouched.
#
# Drivers (11-10): H200 nodes 570.86 (CUDA 12.8), the H100 on `sharing` 545.23 (CUDA 12.3).
# So torch must be a CUDA 12.x build (cu126: runs on any 12.x driver >= 525 by minor-version
# compatibility) and must stay pinned: the first build let flashinfer-python pull PyPI's default
# torch (CUDA 13.2, needs driver >= 580) and nothing started. Triton and FlashInfer's JIT use the
# cluster's CUDA 12.8 ptxas/nvcc (set in the venv's activate), not a CUDA 13 one.
# Builds into a new directory and swaps it in at the end, so jobs that start meanwhile see a whole env.
#SBATCH -J kv4setup
#SBATCH -p short
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail
module load miniconda3/24.11.1
source activate fp4
ENV=/scratch/$USER/envs/kv4base
NEW=$ENV.new
CUDA=/shared/EL9/explorer/cuda/12.8.0
TORCH="torch==2.14.1+cu126"
export PIP_CACHE_DIR=/scratch/$USER/pip_cache
rm -rf "$NEW"
python -m venv "$NEW"
cat >> "$NEW/bin/activate" <<EOF
export CUDA_HOME=$CUDA
export PATH=$CUDA/bin:\$PATH
export TRITON_PTXAS_PATH=$CUDA/bin/ptxas
EOF
source "$NEW/bin/activate"
python -m pip install -q --upgrade pip
printf '%s\n' "$TORCH" "cuda-python<13" > "$NEW/constraints.txt"
python -m pip install -q "$TORCH" --index-url https://download.pytorch.org/whl/cu126
python -m pip install -q flashinfer-python numpy pytest -c "$NEW/constraints.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu126
python - <<'EOF'
import torch, triton
print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__)
assert torch.version.cuda.startswith("12."), "torch must be a CUDA 12 build"
import flashinfer
print("flashinfer", flashinfer.__version__)
EOF
pip freeze > "$NEW/freeze.txt"
# swap in; keep the previous one until the new one is in place
deactivate
rm -rf "$ENV.old"
[ -d "$ENV" ] && mv "$ENV" "$ENV.old"
mv "$NEW" "$ENV"
# a venv records its own path: fix it after the move
sed -i "s#$NEW#$ENV#g" "$ENV/bin/activate" "$ENV"/bin/pip* "$ENV"/bin/python-config 2>/dev/null || true
grep -rl "$NEW" "$ENV/bin" | xargs -r sed -i "s#$NEW#$ENV#g"
source "$ENV/bin/activate" && python -c "import torch, flashinfer; print('swapped in:', torch.__version__, flashinfer.__version__)"
