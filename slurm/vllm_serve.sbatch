#!/bin/bash
# Online serving (11-13): `vllm serve` + `vllm bench serve` with Poisson arrivals, full recipe vs fp8 FlashInfer, one
# after the other in the same job (same node). Random dataset IN / OUT tokens, NP requests per rate, RATES requests/s
# ("inf" = all at once). P50 / P99 TTFT, TPOT, ITL, E2E per rate; JSON in results/vllm_serve/.
#   sbatch -J kv4srv -p gpu-short,gpu --gres=gpu:h200:1 --exclude=d4077 --export=ALL,KV4_COMMIT=<hash> slurm/vllm_serve.sbatch
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 01:50:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail
module load cuda/13.2.0 2>/dev/null || true
MODEL=${MODEL:-meta-llama/Llama-3.1-8B}
CALIB=${CALIB:-results/calib/$(basename $MODEL).pt}
IN=${IN:-16384}; OUT=${OUT:-512}; NP=${NP:-200}; RATES=${RATES:-1 2 4 8 inf}; CONFIGS=${CONFIGS:-recipe fp8_xqa}
# 11-13: DATASET random (IN / OUT with RANGE, e.g. 0.5 = uniform +-50%) or sharegpt (DPATH = the json); W = workload tag;
# GOODPUT = vLLM's SLO spec (MLPerf Llama-3.1-8B server: TTFT 2 s, TPOT 100 ms)
DATASET=${DATASET:-random}; RANGE=${RANGE:-0}; W=${W:-}; GOODPUT=${GOODPUT:-ttft:2000 tpot:100}
# 11-18: SEED (default 0) = vllm bench serve --seed (prompts and arrival times); for repeats set RTAG=_s<SEED> too
# 11-13: NPT = seconds of arrivals per rate; the request count per rate is then max(NP, rate * NPT), so that the system
# reaches a steady state (concurrency can never exceed the number of requests, and KV only fills with enough of them)
NPT=${NPT:-0}
MAXLEN=${MAXLEN:-$(python3 -c "import math;print(int(math.ceil(($IN + $OUT) * (1 + $RANGE))) + 512)")}
# 11-13: the cluster's http(s)_proxy also catches requests to 127.0.0.1 (403 Forbidden in the smoke run 11852169);
# everything here is local and offline, so drop the proxy
unset http_proxy https_proxy ftp_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
export HF_HOME=/scratch/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_CACHE_ROOT=/scratch/$USER/vllm_cache
COMPAT=$(dirname $(find /scratch/$USER/envs/cuda13compat -name "libcuda.so.1" | head -1))
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "${DRV:-0}" -lt 580 ]; then export LD_LIBRARY_PATH=$COMPAT:${LD_LIBRARY_PATH:-}; fi
source /scratch/$USER/envs/vllm/bin/activate
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR/integrations/vllm_plugin TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_vllm$([ "$(basename $SLURM_SUBMIT_DIR)" = kv4-hopper ] || echo _$(basename $SLURM_SUBMIT_DIR)) TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=16
# 11-18: this script unsets the proxy (above), so vLLM's has_nvidia_artifactory() failed and every fp8_fi server
# decoded with FlashInfer's native kernel ("decode_backend=flashinfer-native"), not the XQA kernel vLLM picks on a
# connected H200. fp8_xqa passes the gate with VLLM_HAS_FLASHINFER_CUBIN=1 (SM90 XQA is JIT-built, no cubin needed) and
# is the default FP8 config from now on. New runs go to SDIR (default results/vllm_serve_x), so the earlier
# fp8_fi-native results in results/vllm_serve are never overwritten.
SDIR=${SDIR:-results/vllm_serve_x}
mkdir -p $SDIR
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "KV4_COMMIT=${KV4_COMMIT:-unset} vllm $(python -c 'import vllm;print(vllm.__version__)') model $MODEL in $IN out $OUT np $NP rates $RATES"
# 11-17: the port was 20000 + job id % 20000 (+1 per server), so jobs with adjacent ids on one node picked the same
# port. The second server failed to bind (Errno 98), but the health check only asked "does something answer on this
# port", found the other job's server, and the benchmark sent all its requests there: 11895234's "recipe" requests
# went to 11895233's fp8_fi server (1365 requests logged there = 755 own + 610). Now: a kernel-chosen free port, the
# health check stops if our server process died and requires our own model name in /v1/models, requests use that
# name (another server answers 404), and exactly one socket must listen on the port before and after each rate.
nlisten() { ss -Hltn "sport = :$1" 2>/dev/null | wc -l; }
for cfg in $CONFIGS; do
  PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')
  SNAME=kv4srv-$SLURM_JOB_ID-$cfg
  common="--max-model-len $MAXLEN --gpu-memory-utilization 0.90 --max-num-seqs 512 --port $PORT --served-model-name $SNAME"
  # 11-19: APISRV > 1 = vLLM's --api-server-count (several frontend processes). With one, the API server used 92-97%
  # of a core while the recipe's requests missed the TTFT target with an empty scheduler queue (11990702, seed 1).
  [ "${APISRV:-1}" != 1 ] && common="$common --api-server-count $APISRV"
  case $cfg in
    recipe) envs="KV4_VLLM=1 KV4_CALIB=$CALIB KV4_FORMAT=nvfp4 KV4_KMEAN=1 KV4_SINKS=1 KV4_BOUNDARY=0"
            args="--kv-cache-dtype ultraquant_4bit --block-size 64" ;;
    fp8_fi) envs="KV4_VLLM=0"; args="--kv-cache-dtype fp8 --attention-backend FLASHINFER" ;;
    fp8_xqa) envs="KV4_VLLM=0 VLLM_HAS_FLASHINFER_CUBIN=1"; args="--kv-cache-dtype fp8 --attention-backend FLASHINFER" ;;
    fp8)    envs="KV4_VLLM=0"; args="--kv-cache-dtype fp8" ;;
    bf16)   envs="KV4_VLLM=0"; args="--kv-cache-dtype auto" ;;
  esac
  echo "=== $cfg: $envs vllm serve $MODEL $args $common"
  # 11-16: own process group (setsid), so the cleanup below kills only this server and its engine-core children.
  # The old `pkill -u $USER -f "vllm serve"` also killed the servers of our other jobs on the same node (d4054,
  # 11-15: W-reason 0.75 fp8_fi and 1.25 recipe lost 440 / 203 requests).
  setsid env $envs vllm serve $MODEL $args $common > logs/serve-$SLURM_JOB_ID-$cfg.log 2>&1 &
  srv=$!
  ok=0
  for i in $(seq 1 120); do
    kill -0 $srv 2>/dev/null || break  # our server died (e.g. the port was taken): do not talk to someone else's
    curl -sf http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q "\"$SNAME\"" && ok=1 && break; sleep 5
  done
  if [ $ok != 1 ]; then echo "RESULT $cfg: server not healthy after 10 min"; tail -20 logs/serve-$SLURM_JOB_ID-$cfg.log; kill -9 -- -$srv; continue; fi
  grep -E "KV cache size|Maximum concurrency|kv4-hopper" logs/serve-$SLURM_JOB_ID-$cfg.log | head -5
  echo "RESULT $cfg: backend $(grep -oE "decode_backend=[a-z_-]+, kv_cache_dtype=[a-z0-9_.]+" logs/serve-$SLURM_JOB_ID-$cfg.log | sort -u | tr '\n' ' ')"  # 11-18
  for r in $RATES; do
    np=$NP; [ "$NPT" != 0 ] && [ "$r" != inf ] && np=$(python3 -c "print(max($NP, int($r * $NPT)))")
    # 11-13: server side, every 2 s: KV cache usage, running / waiting requests, preemptions (counter) -> csv
    mcsv=$SDIR/metrics_$(basename $MODEL)_${cfg}${W}_r${r}${RTAG:-}.csv
    # 11-19: + node load (1-min loadavg) and CPU ticks (utime + stime, 100 / s) of the API server ($srv) and of the
    # benchmark client, to tell a CPU-bound frontend or client from a GPU-side delay (recipe 1.35: 142 requests with
    # TTFT > 2 s while 0 requests waited and 0 were preempted). Plus the full /metrics before and after each rate
    # (histograms: time_to_first_token, request_queue_time, request_prefill_time) in $SDIR/prom/.
    echo "t,kv_cache_usage_perc,num_requests_running,num_requests_waiting,num_preemptions_total,prefix_cache_queries_total,prefix_cache_hits_total,load1,api_cpu_ticks,client_cpu_ticks" > $mcsv
    mkdir -p $SDIR/prom; ptag=$SDIR/prom/$(basename $MODEL)_${cfg}${W}_r${r}${RTAG:-}
    curl -s http://127.0.0.1:$PORT/metrics > ${ptag}_before.txt
    # api ticks: all processes of the server's process group except the engine core (= every API server process)
    ( ticks() { [ -n "$1" ] && [ -r /proc/$1/stat ] && awk '{print $14 + $15}' /proc/$1/stat || echo 0; }
      apiticks() { local t=0 q; for q in $(pgrep -g $srv); do grep -q EngineCor /proc/$q/comm 2>/dev/null && continue
          t=$((t + $(ticks $q))); done; echo $t; }
      while true; do
        m=$(curl -s http://127.0.0.1:$PORT/metrics)
        g() { echo "$m" | grep -E "^$1(_total)?[{ ]" | awk '{s += $NF} END {print s + 0}'; }
        cl=$(pgrep -n -f "vllm bench serve .*127.0.0.1:$PORT" || true)
        echo "$(date +%s.%N),$(g vllm:kv_cache_usage_perc),$(g vllm:num_requests_running),$(g vllm:num_requests_waiting),$(g vllm:num_preemptions),$(g vllm:prefix_cache_queries),$(g vllm:prefix_cache_hits),$(cut -d' ' -f1 /proc/loadavg),$(apiticks),$(ticks $cl)" >> $mcsv
        sleep 2
      done ) &
    mon=$!
    if [ "$DATASET" = sharegpt ]; then
      data="--dataset-name sharegpt --dataset-path $DPATH"; dtag=sharegpt
    elif [ "$DATASET" = prefix ]; then  # 11-13: NPFX shared prefixes of IN tokens, SUF-token suffixes, OUT output
      data="--dataset-name prefix_repetition --prefix-repetition-prefix-len $IN --prefix-repetition-suffix-len ${SUF:-256} --prefix-repetition-num-prefixes ${NPFX:-140} --prefix-repetition-output-len $OUT"
      dtag=prefix${NPFX:-140}_i${IN}_o${OUT}
    else
      data="--dataset-name random --random-input-len $IN --random-output-len $OUT --random-range-ratio $RANGE --ignore-eos"
      dtag=i${IN}_o${OUT}_x${RANGE}
    fi
    n0=$(nlisten $PORT)
    vllm bench serve --backend vllm --base-url http://127.0.0.1:$PORT --model $MODEL --served-model-name $SNAME $data \
      --num-prompts $np --request-rate $r --seed ${SEED:-0} --num-warmups ${WARMUP:-5} --goodput $GOODPUT --save-detailed \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 --save-result --result-dir $SDIR \
      --result-filename $(basename $MODEL)_${cfg}${W}_${dtag}_r${r}${RTAG:-}.json 2>&1 \
      | grep -E "Request throughput|Request goodput|Output token throughput|Total token throughput|(P50|P99|Mean) (TTFT|TPOT|ITL|E2EL)|Successful|Failed|Error" \
      | sed "s/^/RESULT $cfg rate $r: /"
    kill $mon 2>/dev/null
    curl -s http://127.0.0.1:$PORT/metrics > ${ptag}_after.txt
    n1=$(nlisten $PORT)
    echo "RESULT $cfg rate $r: listeners on port $PORT before / after: $n0 / $n1"
    if [ "$n0" != 1 ] || [ "$n1" != 1 ]; then echo "RESULT $cfg rate $r: INVALID (port shared with another server)"; fi
    echo "RESULT $cfg rate $r: max KV usage / max waiting / preemptions / prefix hit rate $(awk -F, 'NR > 1 {if ($2 > k) k = $2; if ($4 > w) w = $4; if (NR == 2) {p0 = $5; q0 = $6; h0 = $7}; p1 = $5; q1 = $6; h1 = $7} END {print k, w, p1 - p0, (q1 > q0 ? (h1 - h0) / (q1 - q0) : 0)}' $mcsv)"
  done
  kill -- -$srv 2>/dev/null; wait $srv 2>/dev/null; sleep 5; kill -9 -- -$srv 2>/dev/null; sleep 15  # whole group
done
