#!/bin/bash
# Nsight Compute on one decode call of each implementation (bench/ncu_target.py), kv4base env.
#   sbatch -J kv4ncu -p gpu-short,gpu --gres=gpu:h200:1 slurm/ncu.sbatch [impl ...]
# Raw metrics per kernel go to logs/ncu/<impl>.csv; bench/ncu_summary.py makes the table.
# Clocks are not locked (--clock-control none) so issue rates are those of normal runs;
# caches are flushed between replay passes (ncu's default), as in our timings.
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:45:00
#SBATCH -o logs/%x-%j.out

set -uo pipefail
module load cuda/12.8.0 2>/dev/null || true
source /scratch/$USER/envs/kv4base/bin/activate
export FLASHINFER_WORKSPACE_BASE=/scratch/$USER/flashinfer_cache
export PYTHONPATH=$SLURM_SUBMIT_DIR
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/ncu
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
ncu --version | tail -1

IMPLS=${*:-ours_bf16 ours_fp8 ours_int4_tok ours_int4a_tok ours_nvfp4 ours_mxfp4 fi_bf16 fi_fp8 fi_nvfp4 bd_int4}
for impl in $IMPLS; do
    echo "== $impl"
    ncu --profile-from-start off --target-processes all --clock-control none \
        --section SpeedOfLight --section Occupancy --section LaunchStats --section SchedulerStats \
        --section WarpStateStats --section InstructionStats --section ComputeWorkloadAnalysis \
        --metrics dram__bytes_read.sum,smsp__inst_executed.sum,smsp__thread_inst_executed.sum,sm__inst_issued.avg.pct_of_peak_sustained_active,smsp__issue_active.avg.pct_of_peak_sustained_active \
        --csv --page raw python bench/ncu_target.py --impl "$impl" > "logs/ncu/$impl.csv" 2> "logs/ncu/$impl.err"
    echo "rc=$? $(grep -c . logs/ncu/$impl.csv) lines"; grep -i "err\|warn" "logs/ncu/$impl.err" | head -3
done
python bench/ncu_summary.py logs/ncu/*.csv
