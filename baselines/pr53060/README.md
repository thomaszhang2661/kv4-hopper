# vLLM PR #53060 decode kernel (baseline, unmodified)

Source: https://github.com/vllm-project/vllm/pull/53060 ("turboquant decode attetnion optimization", by xupinjie),
head commit ae702ab8da5ed577583098272425df1a71ebf2b4, files `csrc/attention/turboquant/{main.cpp,kernel.cu,kernel.h,
mma_kernel.cu}`, Apache-2.0 (vLLM's license). Copied unmodified for a kernel-level timing comparison
(`bench/final_table_recipe.py --pr53060`), built as a standalone torch extension with the PR's flags
(`--use_fast_math`, sm_90a here).

Format: TurboQuant K4V4-NC, 4-bit K and V codes plus 6 bytes of norms per token and head (134 bytes at head_dim 128;
our NVFP4 recipe 144, FP8 256). Timing only: the cache is random bytes (speed does not depend on the values).
