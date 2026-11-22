#!/bin/bash
# Submit a short GPU job to every partition that can run it, with the commit recorded.
#   slurm/submit.sh NAME GPU MINUTES [PARTITIONS] -- SCRIPT.sbatch ARGS...
#   slurm/submit.sh kv4fiH200a h200 30 -- slurm/run_base.sbatch bench/bench_flashinfer.py --batch 32
# Defaults: PARTITIONS = gpu-short,gpu,sharing when MINUTES <= 60, else gpu-short,gpu (<= 120) or gpu.
# Short time limits matter: the backfill scheduler fits short jobs into gaps between large ones.
# Limits per user (11-10): gpu-short 2 running; sharing 2 running and 4 submitted; gpu 4 running.
# Run on the login node from ~/kv4-hopper; it only calls sbatch. KV4_COMMIT is taken from the
# environment (set it to the local commit you synced), since the cluster copy is not a git checkout.
set -euo pipefail
name=$1 gpu=$2 minutes=$3
shift 3
if [ "${1:-}" != "--" ]; then parts=$1; shift; else
  if [ "$minutes" -le 60 ]; then parts=gpu-short,gpu,sharing
  elif [ "$minutes" -le 120 ]; then parts=gpu-short,gpu
  else parts=gpu; fi
fi
[ "${1:-}" = "--" ] && shift
sbatch --export=ALL -J "$name" -p "$parts" --gres="gpu:$gpu:1" --mem=32G -t "$minutes" "$@"
