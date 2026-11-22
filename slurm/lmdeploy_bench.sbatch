#!/bin/bash
# 11-14: LMDeploy (TurboMind) INT4 / INT8 KV against its own BF16 KV, and the PyTorch engine's FP8 KV, on one H200
# SXM in one job (docs/lit/lmdeploy_int4_kv.md). Main measurement: bench/lmdeploy_step.py (steady decode window);
# cross-check: LMDeploy's own profile_throughput.py (ITL P50) at B=8, S=16K.
#   sbatch -J kv4lmd -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 --export=ALL,KV4_COMMIT=<hash> slurm/lmdeploy_bench.sbatch
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source /scratch/$USER/envs/lmdeploy/bin/activate
export HF_HOME=/scratch/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
M=${MODEL:-meta-llama/Llama-3.1-8B}
# 11-15: pass the local snapshot directory, not the repo id: with HF_HUB_OFFLINE=1, LMDeploy's snapshot_download
# rejects our cache as incomplete (README / LICENSE / .gitattributes not downloaded; job 11867311 failed in every section).
case "$M" in /*) ;; *) M=$(ls -d /scratch/$USER/hf_cache/hub/models--${M//\//--}/snapshots/* | head -1) ;; esac
L=/scratch/$USER/ext/lmdeploy/benchmark
D=$(find /scratch/$USER/hf_cache/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered -name ShareGPT_V3_unfiltered_cleaned_split.json | head -1)
mkdir -p results/lmdeploy
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python -c "import torch, lmdeploy; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'lmdeploy', lmdeploy.__version__, torch.cuda.get_device_name())"
echo "KV4_COMMIT=${KV4_COMMIT:-unset} model $M"
# 11-15: ONLY=turbomind | pytorch | profile runs one section (default: all)
O=${ONLY:-all}
[ $O = all -o $O = turbomind ] && for qp in 0 8 4; do
  echo "=== turbomind qp $qp"
  python bench/lmdeploy_step.py $M --backend turbomind --qp $qp --bs 1 8 32 --ctx 16384 32768 --out 512 \
    --json results/lmdeploy/step_tm_qp${qp}.json
done
[ $O = all -o $O = pytorch ] && for qp in 0 16; do  # PyTorch engine: 0 = BF16, 16 = FP8 (e4m3); TurboMind has no FP8 KV
  echo "=== pytorch qp $qp"
  python bench/lmdeploy_step.py $M --backend pytorch --qp $qp --bs 1 8 32 --ctx 16384 32768 --out 512 --cache 0.85 \
    --json results/lmdeploy/step_pt_qp${qp}.json
done
[ $O = all -o $O = profile ] && for qp in 0 4; do
  echo "=== profile_throughput turbomind qp $qp B=8 S=16K"
  python $L/profile_throughput.py "$D" $M --backend turbomind --quant-policy $qp --concurrency 8 --num-prompts 8 \
    --dataset-name random --random-input-len 16384 --random-output-len 512 --random-range-ratio 1.0 \
    --cache-max-entry-count 0.9 --top-k 1 --csv results/lmdeploy/pt_check_tm_qp${qp}.csv 2>&1 | tail -25
done
echo done
