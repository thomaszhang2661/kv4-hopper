# Calibration files for the vLLM plugin's recipe

One file per model: the constants a cache written token by token needs (`bench/calib_kv.py`; 8 wikitext-2 test
prompts of 2048 tokens, headroom 2.0). Pass it as `KV4_CALIB=results/calib/<model>.pt`.

Per layer (`layers[i]`): `mu_k`, the static K channel mean per KV head (post-RoPE, Hadamard-rotated, positions >= 4);
`ts_k`, `ts_k_nomean`, `ts_v`, the NVFP4 tensor scales (FP32) with and without mean removal. Files from 11-13 on
also carry `_s` variants of the scales and the calibration text name. Plain tensors and numbers: load with
`torch.load(path, weights_only=True)`.

| File | Model (Hugging Face) | Slurm job |
|---|---|---|
| `Llama-3.1-8B.pt` | meta-llama/Llama-3.1-8B | 11844479 |
| `Qwen2.5-7B-Instruct.pt` | Qwen/Qwen2.5-7B-Instruct | 11844480 |
| `DeepSeek-R1-Distill-Qwen-7B.pt` | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 11846899 |
| `Llama-3-8B-ProLong-512k-Instruct.pt` | princeton-nlp/Llama-3-8B-ProLong-512k-Instruct | 11846900 |
| `Qwen3-32B.pt` | Qwen/Qwen3-32B | 11849500 |
| `Qwen3-30B-A3B.pt` | Qwen/Qwen3-30B-A3B | 11849501 |
| `phi-4.pt` | microsoft/phi-4 | 11849502 |
| `OLMo-2-1124-7B.pt` | allenai/OLMo-2-1124-7B | 11849503 |
| `DeepSeek-R1-Distill-Llama-8B.pt` | deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 11852172 |
| `Qwen2.5-72B-Instruct.pt` | Qwen/Qwen2.5-72B-Instruct | not recorded |

For another model, run `python bench/calib_kv.py --model <id> --out results/calib/<name>.pt` (one GPU, a few minutes
for 8B; see `repro/README.md` for the jobs above).
