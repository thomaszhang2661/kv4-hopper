#!/bin/bash
# UltraQuant scale-rule follow-up (11-11): real-KV MSE over c, then KL over c and with K / V separated.
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -o logs/%x-%j.out
set -euo pipefail
module load miniconda3/24.11.1
source activate fp4
export HF_HOME=/scratch/$USER/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$SLURM_SUBMIT_DIR
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name --format=csv,noheader
python bench/scale_rule_mse.py --out results/scale_rule_mse_llama31_8b.csv
CF=bf16,mxfp4_kh,mxfp4_uq_kh,mx_uqK_ocpV_kh,mx_ocpK_uqV_kh,mxfp4_kvh,mxfp4_uq_kvh
for c in 0.125 0.140 0.156 0.177 0.200 0.220 0.250; do CF=$CF,mxc${c}_kh,mxc${c}_kvh; done
python bench/accuracy_kl.py --ctx 4096 --windows 16 --configs $CF --out results/acc_kl_llama31_8b_ctx4096_csweep.csv
