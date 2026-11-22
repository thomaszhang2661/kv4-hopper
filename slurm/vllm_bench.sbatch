#!/bin/bash
# vLLM 0.33.0 end-to-end baselines (11-12): latency (fixed batch, one generation) and offline throughput, for one
# --kv-cache-dtype per job, Llama-3.1-8B base on one H200. JSON per run in results/vllm/.
#   BS="1 8" THR=0 sbatch -J kv4vb -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 --export=ALL,KV4_COMMIT=<hash> slurm/vllm_bench.sbatch fp8
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail
dt=$1
module load cuda/13.2.0 2>/dev/null || true  # nvcc for the plugin's JIT build (KV4_VLLM=1)
MODEL=${MODEL:-meta-llama/Llama-3.1-8B}
export HF_HOME=/scratch/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT=/scratch/$USER/vllm_cache
# vLLM 0.33.0 ships only cu130 wheels; the H200 nodes run driver 570, so use the CUDA 13 forward-compatibility libs
COMPAT=$(dirname $(find /scratch/$USER/envs/cuda13compat -name "libcuda.so.1" | head -1))
# the compat libs only for drivers older than CUDA 13 (570 on the SXM nodes); with a newer driver (595 on d4077)
# they must not be loaded (error 803)
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "${DRV:-0}" -lt 580 ]; then export LD_LIBRARY_PATH=$COMPAT:${LD_LIBRARY_PATH:-}; fi
source /scratch/$USER/envs/vllm/bin/activate
export KV4_VLLM=${KV4_VLLM:-0}  # our plugin (integrations/vllm_plugin) off unless asked for
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR/integrations/vllm_plugin TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_vllm$([ "$(basename $SLURM_SUBMIT_DIR)" = kv4-hopper ] || echo _$(basename $SLURM_SUBMIT_DIR)) TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=16
mkdir -p results/vllm
echo "host $(hostname) job ${SLURM_JOB_ID:-none}"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "KV4_COMMIT=${KV4_COMMIT:-unset} vllm $(python -c 'import vllm;print(vllm.__version__)') dtype $dt"
tag=$(basename $MODEL)_$dt${RTAG:-}  # RTAG: e.g. _nvl for runs on the H200 NVL node d4077
# 11-12: MAXLEN, IL (latency prompt lengths), WQ (weight quantization, e.g. fp8), OUT (throughput output length)
MAXLEN=${MAXLEN:-33792}; IL=${IL:-16384 32768}; OUT=${OUT:-512}
common="--model $MODEL --kv-cache-dtype $dt --max-model-len $MAXLEN --gpu-memory-utilization 0.90"
[ -n "${WQ:-}" ] && common="$common --quantization $WQ" && tag=${tag}_w$WQ
# 11-12: ABACKEND forces the attention backend of a non-UltraQuant run, e.g. FLASHINFER for the FP8 baseline
# (vLLM #66590: on H100, FlashInfer FP8 KV is ~11% faster than the default FA3 FP8 path)
[ -n "${ABACKEND:-}" ] && [ "$dt" != ultraquant_4bit ] && common="$common --attention-backend $ABACKEND" && tag=${tag}_$ABACKEND
# 11-12: UltraQuant is never auto-selected on CUDA, and with explicit selection every layer must use it, so all layers
# are quantized (KV4_NO_BOUNDARY=1, a switch in our plugin package; same for stock and plugin runs). KV4_VLLM=1 swaps
# in our store/decode kernels.
if [ "$dt" = ultraquant_4bit ]; then
  export KV4_NO_BOUNDARY=${KV4_NO_BOUNDARY:-0}
  # 11-12 night: no forced backend (the plugin package auto-selects ULTRAQUANT), boundary layers native unless
  # KV4_NO_BOUNDARY=1, and 64-token blocks (vLLM picks 32 when native layers share the pool; our pages are 64)
  common="$common --block-size 64"
  tag=${tag}_plugin${KV4_VLLM}
fi
# 11-12: split into jobs under the 1 h sharing limit: BS (batch sizes for latency, "" = none), THR=0 skips throughput
BS=${BS-1 8 32}; THR=${THR:-1}
# latency: one batch, prompt then 256 generated tokens (decode dominates at these lengths)
for bs in $BS; do
  for il in $IL; do
    [ -z "${ALLSHAPES:-}" ] && [ $bs -eq 32 ] && [ $il -eq 32768 ] && continue
    for ol in 256 1; do  # 11-12: output 1 too, so decode time per token = (t256 - t1) / 255
      vllm bench latency $common --batch-size $bs --input-len $il --output-len $ol --num-iters-warmup 1 --num-iters 3 \
        --output-json results/vllm/latency_${tag}_b${bs}_i${il}_o${ol}.json 2>&1 | grep -E "Avg latency|Error|error|KV cache size|Maximum concurrency|reempt|decode_backend=|artifactory" | tail -n 8
    done
  done
done
# throughput: many requests, the scheduler batches as memory allows (more KV capacity -> bigger batches)
for il in $([ "$THR" = 1 ] && echo ${THRIL:-8192 16384}); do
  # 11-12: vLLM 0.33 ignores --input-len / --output-len here (getattr(args, "random_input_len", input_len) finds the
  # random dataset's own default of 1024), so pass the random-dataset arguments directly
  vllm bench throughput $common --dataset-name random --random-input-len $il --random-output-len $OUT --num-prompts 256 \
    --output-json results/vllm/throughput_${tag}_i${il}_o${OUT}.json 2>&1 | grep -E "Throughput|Total num|Error|error|KV cache size|Maximum concurrency|reempt|decode_backend=|artifactory" | tail -n 8
done
echo done
