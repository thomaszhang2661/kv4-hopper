#!/bin/bash
# K smoothing follow-up (11-11 night): mean from the first 512 tokens only (deployable stand-in), and the scale
# constant c under smoothing, on three models, sink 0 (as UltraQuant) and sink 4 (our recipe).
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -o logs/%x-%j.out
set -euo pipefail
export HF_HOME=/scratch/$USER/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$SLURM_SUBMIT_DIR
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name --format=csv,noheader
PY=~/.conda/envs/fp4/bin/python
CF=bf16,nvfp4_khs,nvfp4_khp,int4a_khs,int4a_khp,mxc0.156_khs,mxc0.177_khs,mxc0.200_khs,mxc0.200_khp
for m in Qwen/Qwen3-8B-Base meta-llama/Llama-3.1-8B mistralai/Mistral-7B-v0.3; do
  t=$(basename $m | tr 'A-Z.' 'a-z_')
  for s in 0 4; do
    $PY bench/accuracy_kl.py --model $m --sink $s --configs $CF --out results/acc_kl_${t}_ctx4096_sink${s}_smooth2.csv
  done
done
