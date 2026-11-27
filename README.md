# kv4-hopper

Fast 4-bit KV-cache decoding on NVIDIA Hopper GPUs, and a vLLM plugin that uses it.

Code for the report *Fast 4-bit KV-Cache Decoding on Hopper GPUs: Why Existing Kernels Are Slow and How to Fix Them*
(Jian Zhang, 2026; [arXiv:2611.30027](https://arxiv.org/abs/2611.30027)). The LaTeX source is in `paper/`.

Hopper has no FP4 hardware, so a 4-bit KV cache has to be decoded in software. This repo contains:

- a Hopper-native (sm_90a) decode-attention kernel for NVFP4 and INT4 KV caches: a TMA producer warp feeding two
  compute warpgroups, a cache layout written so decoded values go to the tensor cores straight from registers, and
  swapped matrix operands so that small query groups need no padding rows;
- a vLLM 0.33 plugin that runs it behind `--kv-cache-dtype ultraquant_4bit`, with the accuracy recipe used in the
  report (K mean removal, K Hadamard rotation, four BF16 sink tokens, layer 0 in BF16);
- the benchmark, accuracy and analysis scripts behind the report's numbers, and the result files they produced.

## Results at a glance

From the report (version 1); H200, vLLM 0.33.0, Llama-3.1-8B unless noted. The baseline is vLLM's FP8 KV cache on
FlashInfer with XQA decode, the faster of vLLM's FP8 paths in our runs.

| | Plugin with the recipe, against FP8 KV |
|---|---|
| Tokens the KV cache holds | 1.63× as many |
| Decode step, batch 8 and above | 1.12–1.64× faster |
| Decode step, batch 1 | level (1.01–1.05× per step; 0.96–1.03× in end-to-end time per output token) |
| Decode step, Qwen3-32B and Qwen2.5-72B | 1.04–1.18× faster (the weight read dominates the step) |
| Time to first token | 1.5–6% longer (prefill about 7% slower) |
| Accuracy against BF16 KV | no significant difference on five 7B–32B models with repeated runs; the 95% intervals exclude losses larger than about 3.5 points |
| Decode kernel alone, 11 shapes | 1.12–1.49× faster than the faster of TensorRT-LLM's XQA and FlashAttention-4's proposed native FP8 kernels |

It helps most when decoding dominates and KV memory is the limit: batch 8 and above, long outputs, many requests in
flight. At batch 1 it is level with FP8, and streams with long inputs and short outputs are slower, because prefill,
not the cache, limits them.

**Serving capacity: use the revised numbers.** Version 1 of the report also gave the highest request rate one H200
sustains within the latency targets as 1.23–1.32× (Llama-3.1-8B), 1.45–1.80× (Qwen3-32B) and 1.67–2.20×
(Qwen2.5-72B) FP8's. Those brackets came from 600-s runs, mostly with one seed. Measured again with longer
steady-state runs under rules fixed in advance, several of their end points did not hold up and the planned comparison
was inconclusive, so please do not rely on those three ranges. The revised report (version 2) gives the re-measured
results.

### Notes for vLLM users on Hopper

From the same measurements (vLLM 0.33, H200):

- vLLM's NVFP4 KV cache (`--kv-cache-dtype nvfp4`) saves memory but decodes 1.4–2.3× slower than FP8 at batch 8 and
  above and about 1.1× slower at batch 1. Its UltraQuant format falls back to Triton decode kernels on NVIDIA and is
  1.8–3.9× slower than FP8 per step.
- With FlashInfer and FP8 KV, vLLM uses the faster XQA decode kernel only if its network check to NVIDIA's artifact
  server succeeds; when the check fails (behind a proxy, for example), it falls back to FlashInfer's own decode kernel
  without an error. Check the log for `decode_backend=xqa`.
- Qwen2-family reasoning models can collapse under FP8 KV without any error. Keeping layer 0 in BF16
  (`--kv-cache-dtype-skip-layers 0`, 1/28 more KV memory on a 28-layer model) restored them in our runs, but skipping
  layers crashes FlashAttention-3 for some models (vllm-project/vllm#65821; fix proposed in vllm-project/vllm#65891).

## Scope

Kernels tested on H200 (SXM and NVL) and H100 SXM; the vLLM plugin on H200, with vLLM 0.33.0, PyTorch's CUDA 13
build and nvcc 13.2. Head dimension 128,
1–8 query heads per KV head. Not supported: MLA models, head dimension 64 or 256, sliding-window layers, speculative
decoding. Tensor parallelism was checked for correctness only, at TP = 2. See Section 9 of the report for details.

## Layout

| Path | Contents |
|---|---|
| `kv4/cuda/` | CUDA kernels and their PyTorch wrappers (JIT-built): `decode_fp4.cu`, `decode_int4.cu`, `store_fp4.cu` (cache write), `merge.cu` (split and sink merge), `kside.cu` |
| `kv4/formats.py`, `kv4/reference.py` | storage formats with reference dequantization; float32 reference attention |
| `kv4/kernels/` | Triton baselines (BF16 and 4-bit) |
| `integrations/vllm_plugin/` | the vLLM plugin (`kv4_vllm`) |
| `baselines/` | bindings for baselines that are not pip packages (BitDecoding, vLLM PR #53060) |
| `bench/` | kernel benchmarks, vLLM benchmarks, accuracy evaluations, analysis scripts |
| `slurm/` | the batch scripts used to run everything (see below) |
| `results/` | result files (CSV / TXT / small JSON) the report's tables are built from |
| `tests/` | format tests (CPU) and kernel-vs-reference tests (GPU) |
| `paper/` | LaTeX source, bibliography and figure script |

## Quick start

```bash
python -m pytest -q tests/test_formats.py        # CPU: formats against the reference
python -m pytest -q tests/test_kernels.py        # GPU (sm_90a): kernels against the reference
```

The vLLM plugin, in an environment with vLLM 0.33.0:

```bash
pip install --no-deps -e integrations/vllm_plugin
export PYTHONPATH=$PWD:$PWD/integrations/vllm_plugin TORCH_CUDA_ARCH_LIST=9.0a

# once per model: calibration constants (tensor scales and the static K mean). Files for the ten models of the
# report are in results/calib/ (see its README); this regenerates one or makes one for another model:
python bench/calib_kv.py --model meta-llama/Llama-3.1-8B --out results/calib/Llama-3.1-8B.pt

# the recipe used for every accuracy number in the report
export KV4_VLLM=1 KV4_CALIB=results/calib/Llama-3.1-8B.pt KV4_FORMAT=nvfp4 KV4_KMEAN=1 KV4_SINKS=1 KV4_BOUNDARY=0
vllm serve meta-llama/Llama-3.1-8B --kv-cache-dtype ultraquant_4bit --block-size 64
```

`KV4_VLLM=0` turns the plugin off. `bench/vllm_kv4_check.py` is a quick end-to-end correctness check.

## Reproducing the report

The runs were made on a Slurm cluster. The scripts in `slurm/` load modules and keep environments and model caches
under `/scratch/$USER`; adapt those lines to your machine. Every speed comparison in the report was run with both
configurations in the same job on the same node, and the scripts are written that way.

| Report | Scripts |
|---|---|
| Instruction counts and SASS (Section 3) | `slurm/nvbit.sbatch`, `bench/nvbit_summary.py`; `slurm/sass_dump.sbatch`, `bench/sass_count.py` |
| Kernel speed (Section 6) | `slurm/final_table.sbatch`, `bench/final_table_recipe.py`, `bench/foscar_compare.py`, `bench/lmdeploy_step.py` |
| Pure decode in vLLM (Table 6, Figure 3) | `bench/vllm_forward_matrix.py` |
| End to end, throughput, 64 × 32K (Section 7) | `slurm/vllm_bench.sbatch`, `slurm/vllm_bench_paper.sbatch` |
| Per-kernel profile (Section 7) | `bench/debug/vllm_decode_profile.py` |
| Serving goodput (Section 7) | `slurm/vllm_serve.sbatch`, `bench/analyze_serve.py` |
| Capacity on one GPU, 8B to 72B (Table 8) | `slurm/vllm_serve.sbatch` (`MODEL=`, `APISRV=4`), `bench/analyze_serve.py`, `bench/serve_miss_timeline.py`; per-step and perplexity with FP8 weights: `bench/vllm_forward_matrix.py`, `bench/vllm_prompt_ppl.py` |
| FP8-tensor-core prototype (limitations) | `slurm/k8_proto.sbatch`, `bench/k8_proto.py`, `kv4/cuda/decode_fp4_k8.cu` |
| Accuracy offline (Section 8) | `slurm/accuracy.sbatch`, `bench/accuracy_kl.py`, `bench/longbench.py`, `bench/ruler_lite.py`, `bench/reasoning_eval.py` |
| Accuracy in vLLM (Section 8) | `bench/vllm_reasoning_check.py`, `bench/vllm_ruler.py`, `bench/vllm_prompt_ppl.py`, `bench/paired.py`, `bench/analyze_reasoning_runs.py` |
| Findings reported to vLLM (Appendix A) | `slurm/pr67789_check.sbatch`, `bench/debug/pr67789_check.py`, `bench/debug/uq_prefix_bug.py` |

For every table and figure, `repro/README.md` lists the result files, the analysis script and the Slurm jobs
behind its numbers, with each job's `sbatch` line as Slurm recorded it; `repro/reproduce.sh <label>` prints
(or, with `--submit`, submits) the jobs for one label, e.g. `tab:xqa` (the 11-21 additions, including `tab:capacity`, are listed in
`repro/README.md` only).

Comments in the scripts carry the date they were written or changed (e.g. `11-17:`).

## Tests

| Suite | Where | What it checks |
|---|---|---|
| `python -m pytest` | any machine (CI: every push and PR) | storage formats, packing, reference attention; GPU tests skip |
| CI `cuda` job | GitHub runner, no GPU | every CUDA kernel compiles for sm_90a with nvcc 13.2 and torch 2.14 cu130 (`tools/build_kernels.py`) |
| `slurm/gpu_tests.sbatch` | one H100 / H200 | Triton and CUDA kernels against FP32 references (ragged lengths, shuffled pages, sinks, 4 and 8 query heads per KV head, quantize-on-write byte-exact against the offline packing); the vLLM plugin with the full recipe on prefix-cache hits and chunked prefill (the vLLM #46461 / #67789 triggers, 4 needles each) |

```bash
sbatch -J kv4test -p gpu-short --gres=gpu:h200:1 --export=ALL,KV4_COMMIT=$(git rev-parse --short HEAD) slurm/gpu_tests.sbatch
# last line of logs/kv4test-<job>.out: GPU TESTS PASSED / FAILED
```

## License

Apache License 2.0 (see `LICENSE`). `baselines/pr53060/` holds unmodified files from vLLM PR #53060, also
Apache-2.0 (see `NOTICE`).

## Citation

```bibtex
@misc{zhang2026kv4hopper,
  title         = {Fast 4-bit {KV}-Cache Decoding on Hopper {GPUs}: Why Existing Kernels Are Slow and How to Fix Them},
  author        = {Zhang, Jian},
  year          = {2026},
  eprint        = {2611.30027},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2611.30027}
}
```
