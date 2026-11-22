#!/bin/bash
# Does the cu130 vLLM env run on the H200 nodes' driver 570 with the CUDA 13 forward-compatibility libs? (11-12)
# Prints torch's view of the GPU, then generates a few tokens with vLLM (BF16 and each 4-bit dtype briefly).
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail
export HF_HOME=/scratch/$USER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_CACHE_ROOT=/scratch/$USER/vllm_cache
COMPAT=$(dirname $(find /scratch/$USER/envs/cuda13compat -name "libcuda.so.1" | head -1))
# the compat libs only for drivers older than CUDA 13 (570 on the SXM nodes); with a newer driver (595 on d4077)
# they must not be loaded (error 803)
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "${DRV:-0}" -lt 580 ]; then export LD_LIBRARY_PATH=$COMPAT:${LD_LIBRARY_PATH:-}; fi
source /scratch/$USER/envs/vllm/bin/activate
export KV4_VLLM=${KV4_VLLM:-0}  # our plugin (integrations/vllm_plugin) off unless asked for
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "compat libs: $COMPAT"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print(torch.cuda.get_device_name()); x=torch.randn(1024,1024,device='cuda'); print('matmul ok', (x@x).sum().item() is not None)"
for dt in ${DTYPES:-auto fp8 nvfp4 ultraquant_4bit}; do
  echo "=== $dt"
  python - "$dt" <<'PY' 2>&1 | grep -vE "it/s\]|%\|" | tail -n ${TAIL:-15}
import sys
from vllm import LLM, SamplingParams
dt = sys.argv[1]
llm = LLM("meta-llama/Llama-3.1-8B", kv_cache_dtype=dt, max_model_len=8192, gpu_memory_utilization=0.85,
          enforce_eager=True)
out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=16, temperature=0))
print(dt, "->", repr(out[0].outputs[0].text))
PY
done
