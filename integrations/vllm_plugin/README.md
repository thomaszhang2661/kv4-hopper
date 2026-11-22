# kv4-vllm: vLLM plugin for the kv4-hopper kernels

Runs vLLM 0.33's `--kv-cache-dtype ultraquant_4bit` on Hopper with the kv4-hopper kernels instead of vLLM's Triton
fallback:

| Part | Stock UltraQuant on NVIDIA | This plugin |
|---|---|---|
| cache write | Triton | `kv4/cuda/store_fp4.cu`, a per-(page, head) layout for the tensor cores |
| decode | Triton unified attention | `decode_fp4_paged` (wgmma, TMA) + split merge (with the BF16 sinks) |
| prefill | FlashAttention-2 | FlashAttention-3 in BF16 |
| continued prefill | Triton / dequantization + FlashAttention | dequantization of our pages + FlashAttention |

The plugin keeps vLLM's 64-token blocks and reinterprets each block's bytes as its own page layout, so vLLM's cache
allocation is unchanged.

Install into the vLLM 0.33.0 environment with `pip install --no-deps -e integrations/vllm_plugin`, put the repo root
on `PYTHONPATH` (the kernels are JIT-built with nvcc 13.x for sm_90a on first use), and set the recipe's switches as
in the top-level README. `KV4_VLLM=0` disables the plugin. Limits: head dimension 128, 4 or 8 query heads per KV head
(others are zero-padded to 8), no sliding window.
