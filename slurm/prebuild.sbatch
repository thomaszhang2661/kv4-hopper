#!/bin/bash
# Build the JIT extensions on a CPU node (11-13): catches compile errors without holding an H200, and fills the same
# torch-extension cache the GPU jobs use, so they start without compiling. nvcc compiles sm_90a without a GPU; the
# final import may fail here (no driver), which is fine: the build is cached.
#   sbatch -J kv4build -p short --export=ALL,ENV=kv4base slurm/prebuild.sbatch   # run_base.sbatch jobs
#   sbatch -J kv4build -p short --export=ALL,ENV=vllm slurm/prebuild.sbatch      # vllm_*.sbatch jobs
#SBATCH -p short
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
ENV=${ENV:-kv4base}
if [ "$ENV" = vllm ]; then
  module load cuda/13.2.0 2>/dev/null || true
  source /scratch/$USER/envs/vllm/bin/activate
  export TORCH_EXTENSIONS_DIR=/scratch/$USER/torch_ext_vllm$([ "$(basename $SLURM_SUBMIT_DIR)" = kv4-hopper ] || echo _$(basename $SLURM_SUBMIT_DIR))
  exts="kv4.cuda.store_fp4 kv4.cuda.decode_fp4 kv4.cuda.merge"
else
  module load cuda/12.8.0 2>/dev/null || true
  source /scratch/$USER/envs/kv4base/bin/activate
  exts="kv4.cuda.store_fp4 kv4.cuda.decode_fp4 kv4.cuda.merge kv4.cuda.decode_int4 pr53060"
fi
export PYTHONPATH=$SLURM_SUBMIT_DIR TORCH_CUDA_ARCH_LIST=9.0a MAX_JOBS=16
echo "KV4_COMMIT=${KV4_COMMIT:-unset} env $ENV nvcc $(nvcc --version | tail -1)"
for e in $exts; do
  python - "$e" <<'PY'
import importlib, sys, traceback
name = sys.argv[1]
try:
    if name == "pr53060":
        from bench.final_table_recipe import pr53060_ext as f
    else:
        f = importlib.import_module(name).ext
    f()
    print(f"BUILD {name}: built and imported")
except Exception as ex:  # noqa: BLE001
    msg = "".join(traceback.format_exception_only(type(ex), ex)).strip().splitlines()
    err = [l for l in str(ex).splitlines() if "error" in l.lower()][:5]
    print(f"BUILD {name}: {'compile error' if 'Error building' in str(ex) else 'built, import failed (no GPU?)'}: "
          f"{msg[-1][:200]}")
    for l in err:
        print("   ", l[:300])
PY
done
