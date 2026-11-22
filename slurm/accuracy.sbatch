#!/bin/bash
# KV-format accuracy (11-11): KL from BF16 and perplexity on wikitext-2 with fake-quantized K/V (bench/accuracy_kl.py),
# fp4 conda env (transformers 5.21); KV4_SCRIPT=bench/needle.py runs the needle test instead. Any GPU with BF16 and >= 40 GB; the H200 partitions are usually quickest.
#   sbatch -J kv4acc -p gpu-short,gpu,gpu-interactive --gres=gpu:h200:1 --exclude=d4077 \
#     --export=ALL,KV4_COMMIT=<hash> slurm/accuracy.sbatch --ctx 4096 --windows 16 --out results/acc_kl_llama31_8b.csv
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:50:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/scratch/$USER/hf_cache PYTHONPATH=$SLURM_SUBMIT_DIR HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "KV4_COMMIT=${KV4_COMMIT:-unset}"
~/.conda/envs/fp4/bin/python ${KV4_SCRIPT:-bench/accuracy_kl.py} "$@"
