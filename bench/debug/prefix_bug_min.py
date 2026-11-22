"""11-12: the minimal repro for vllm-project/vllm#46461. Run without our plugin package:
    VLLM_PLUGINS= python bench/debug/prefix_bug_min.py ultraquant_4bit
v2 (11-12 night): the clean-env run of v1 failed for both quantized dtypes: forcing attention_backend also forces
it on the native-dtype boundary layers ("kv_cache_dtype not supported"). TurboQuant is in the CUDA priority list, so
it is now auto-selected; UltraQuant is not, so the repro appends it to the list.
"""
import os, sys
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
from vllm import LLM, SamplingParams

dt = sys.argv[1]  # auto | turboquant_k8v4 | ultraquant_4bit
if dt == "ultraquant_4bit":
    # UltraQuant is not in the CUDA backend priority list, and forcing it with attention_backend fails on the
    # boundary layers kept at the native dtype, so add it to the list (in-process engine, so this reaches the worker).
    from vllm.platforms import cuda
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    _orig = cuda._get_backend_priorities
    cuda._get_backend_priorities = lambda *a, **k: [*_orig(*a, **k), AttentionBackendEnum.ULTRAQUANT]
llm = LLM("meta-llama/Llama-3.1-8B", max_model_len=4096, gpu_memory_utilization=0.6,
          enable_prefix_caching=True, kv_cache_dtype=dt)
filler = " ".join(f"Line {i}: the weather was mild and nothing else happened that day." for i in range(40))
prefix = f"Remember this: the secret code is 4721. {filler}"
a = prefix + " Question: what is the secret code? Answer: the secret code is"
b = " ".join(f"Entry {i}: an unrelated note about gardening, soil and rain." for i in range(110))
sp, warm = SamplingParams(max_tokens=8, temperature=0), SamplingParams(max_tokens=1, temperature=0)

llm.generate([prefix], warm)                      # cache the prefix
alone = llm.generate([a], sp)[0].outputs[0].text
llm.reset_prefix_cache()
llm.generate([prefix], warm)
batched = llm.generate([a, b], sp)[0].outputs[0].text
print(f"{dt}: alone {alone!r} | batched {batched!r}")
