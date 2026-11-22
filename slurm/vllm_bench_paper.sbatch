#!/bin/bash
# 11-17: paper v1 reruns with every configuration in ONE job on ONE node (audit docs/paper_audit/vllm_e2e.md S8, S10,
# S12, S17: the 11-13 end-to-end pairs were two jobs, the decode-led ratios three jobs on two commits, 64 x 32K two
# concurrent jobs). Runs slurm/vllm_bench.sbatch's body once per configuration, sequentially, on this allocation.
#   sbatch -J kv4paper -p gpu --gres=gpu:h200:1 --exclude=d4077 --export=ALL,KV4_COMMIT=<hash> slurm/vllm_bench_paper.sbatch
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "host $(hostname) job $SLURM_JOB_ID commit ${KV4_COMMIT:-unset}"
R="KV4_VLLM=1 KV4_CALIB=results/calib/Llama-3.1-8B.pt KV4_FORMAT=nvfp4 KV4_KMEAN=1 KV4_SINKS=1 KV4_BOUNDARY=0"
run() { echo "=== $1 | $2 | $(date +%H:%M:%S)"; env $2 bash slurm/vllm_bench.sbatch $1; }
# 11-17: PART=A|B|C runs one part (each part's comparison stays inside one job); unset = all three.
PART=${PART:-ABC}
if [[ $PART == *A* ]]; then
# A: latency (B 1, 8, 32 x 16K / 32K, 32 x 32K skipped as before) + offline throughput (8K, 16K in, 512 out)
run ultraquant_4bit "$R RTAG=_paperA"
run fp8 "ABACKEND=FLASHINFER RTAG=_paperA"
fi
if [[ $PART == *B* ]]; then
# B: decode-led throughput (16K in, 2,048 out, 256 prompts), recipe vs FlashInfer FP8 vs default FA3 FP8
for c in "ultraquant_4bit|$R" "fp8|ABACKEND=FLASHINFER" "fp8|"; do
  run "${c%%|*}" "${c#*|} BS= THR=1 THRIL=16384 OUT=2048 RTAG=_paperB"
done
fi
if [[ $PART == *C* ]]; then
# C: 64 x 32K latency (the FP8 cache cannot hold all 64 sequences at once)
run ultraquant_4bit "$R BS=64 IL=32768 THR=0 ALLSHAPES=1 RTAG=_paperC"
run fp8 "ABACKEND=FLASHINFER BS=64 IL=32768 THR=0 ALLSHAPES=1 RTAG=_paperC"
fi
echo "all done $(date +%H:%M:%S)"
