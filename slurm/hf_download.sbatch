#!/bin/bash
# Download a model / dataset into the shared HF cache on a CPU node (11-11; keeps downloads off the login node).
#   sbatch slurm/hf_download.sbatch model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
#   sbatch slurm/hf_download.sbatch dataset HuggingFaceH4/MATH-500
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
export HF_HOME=/scratch/$USER/hf_cache
kind=$1; repo=$2
~/.conda/envs/fp4/bin/python - "$kind" "$repo" <<'PY'
import sys
from huggingface_hub import snapshot_download
kind, repo = sys.argv[1], sys.argv[2]
p = snapshot_download(repo, repo_type="dataset" if kind == "dataset" else "model",
                      allow_patterns=None if kind == "dataset" else ["*.json", "*.safetensors", "*.txt", "*.model"])
print("downloaded", repo, "to", p)
PY
