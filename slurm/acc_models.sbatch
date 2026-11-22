#!/bin/bash
# KL / perplexity on more models (11-11): Qwen3-8B-Base and Mistral-7B-v0.3, sink 0 and 4.
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -o logs/%x-%j.out
set -euo pipefail
export HF_HOME=/scratch/$USER/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$SLURM_SUBMIT_DIR
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name --format=csv,noheader
CF=bf16,fp8,int4a,int4a_kh,int4a_kvh,nvfp4,nvfp4_kh,nvfp4_kvh,mxfp4,mxfp4_kh,mxfp4_kvh,mxfp4_uq_kh,mxfp4_uq_kvh,mxc0.200_kh,mxc0.200_kvh
for m in Qwen/Qwen3-8B-Base mistralai/Mistral-7B-v0.3; do
  tag=$(basename $m | tr 'A-Z.' 'a-z_')
  for sink in 0 4; do
    ~/.conda/envs/fp4/bin/python bench/accuracy_kl.py --model $m --ctx 4096 --windows 16 --sink $sink --configs $CF \
      --out results/acc_kl_${tag}_ctx4096_sink${sink}.csv
  done
done
