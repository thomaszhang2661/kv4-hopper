#!/bin/bash
# Qwen3 rotation diagnosis (11-11 night): K/V statistics, then KL with K smoothing on three models.
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -o logs/%x-%j.out
set -euo pipefail
export HF_HOME=/scratch/$USER/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$SLURM_SUBMIT_DIR
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name --format=csv,noheader
PY=~/.conda/envs/fp4/bin/python
$PY bench/kv_stats.py
CF=bf16,nvfp4,nvfp4_kh,nvfp4_ks,nvfp4_khs,int4a_kh,int4a_ks,int4a_khs,mxc0.200,mxc0.200_kh,mxc0.200_ks,mxc0.200_khs
$PY bench/accuracy_kl.py --model Qwen/Qwen3-8B-Base --sink 4 --configs $CF --out results/acc_kl_qwen3-8b-base_ctx4096_sink4_smooth.csv
$PY bench/accuracy_kl.py --model Qwen/Qwen3-8B-Base --sink 0 --configs $CF --out results/acc_kl_qwen3-8b-base_ctx4096_sink0_smooth.csv
$PY bench/accuracy_kl.py --model meta-llama/Llama-3.1-8B --sink 4 --configs $CF --out results/acc_kl_llama31_8b_ctx4096_sink4_smooth.csv
$PY bench/accuracy_kl.py --model mistralai/Mistral-7B-v0.3 --sink 4 --configs $CF --out results/acc_kl_mistral-7b-v0_3_ctx4096_sink4_smooth.csv
