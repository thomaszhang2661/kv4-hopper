"""11-12 night: correctness of the plugin's full recipe in vLLM where its state is per sequence (sink pools keyed by the
first block id, static K mean, continuation prefill). One engine per config, in-process (deterministic batching).

  1. prefix cache: a request whose prefix (with a code) is cached, alone and batched with a longer fresh request
     (the #46461 situation; the plugin checks first chunks per request);
  2. prefix-cache hit after the cache was filled by *another* request with the same first block, so the hit's sinks
     come from a slot captured by a different request (the slot lookup via the first block id);
  3. chunked prefill: a ~6000-token prompt with the code at its start, max_num_batched_tokens 1024, so 5 continuation
     chunks read the 4-bit cache (and the BF16 sinks swapped into the dequantized prefix);
  4. agreement with auto: greedy 48-token continuations of 8 wikitext prompts, the fraction of tokens equal to auto's
     up to the first divergence, and the number of exact matches.

    KV4_CALIB=results/calib/Llama-3.1-8B.pt python bench/debug/recipe_correctness.py auto recipe
    MODEL=Qwen/Qwen2.5-72B-Instruct TP=2 GMU=0.85 KV4_CALIB=results/calib/Qwen2.5-72B-Instruct.pt ... auto fp8 recipe
"""
import json
import os
import subprocess
import sys

M = os.environ.get("MODEL", "meta-llama/Llama-3.1-8B")
# 11-14: TP and memory fraction for the 72B TP=2 check (defaults = the earlier 8B runs)
TP = int(os.environ.get("TP", "1"))
GMU = float(os.environ.get("GMU", "0.6"))
CONFIGS = {"auto": {}, "fp8": {},
           "recipe": {"KV4_VLLM": "1", "KV4_FORMAT": "nvfp4", "KV4_KMEAN": "1", "KV4_SINKS": "1", "KV4_BOUNDARY": "0"},
           "nvfp4_l0": {"KV4_VLLM": "1", "KV4_FORMAT": "nvfp4", "KV4_BOUNDARY": "0"}}


def child(name):
    from vllm import LLM, SamplingParams
    dt = {"auto": "auto", "fp8": "fp8"}.get(name, "ultraquant_4bit")
    kw = dict(kv_cache_dtype=dt)
    if dt == "ultraquant_4bit":
        kw["block_size"] = 64
    sp = SamplingParams(max_tokens=8, temperature=0)
    warm = SamplingParams(max_tokens=1, temperature=0)
    filler = " ".join(f"Line {i}: the weather was mild and nothing else happened that day." for i in range(40))
    prefix = f"Remember this: the secret code is 4721. {filler}"
    a = prefix + " Question: what is the secret code? Answer: the secret code is"
    a2 = prefix + " Question: please repeat the secret code. Answer: the secret code is"
    b = " ".join(f"Entry {i}: an unrelated note about gardening, soil and rain." for i in range(110))
    res = {}
    llm = LLM(M, max_model_len=8192, gpu_memory_utilization=GMU, tensor_parallel_size=TP, enable_prefix_caching=True, **kw)
    llm.generate([prefix], warm)
    res["1_alone"] = llm.generate([a], sp)[0].outputs[0].text
    llm.reset_prefix_cache()
    llm.generate([prefix], warm)
    res["1_batched"] = llm.generate([a, b], sp)[0].outputs[0].text
    # 2: the cache is filled by request a; request a2 shares the first blocks and must find a's sinks
    llm.reset_prefix_cache()
    llm.generate([a], warm)
    r = llm.generate([a2], sp)[0]
    res["2_hit_other_request"] = r.outputs[0].text
    res["2_cached_tokens"] = r.num_cached_tokens
    del llm
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    # 3 and 4: chunked prefill, no prefix cache
    llm = LLM(M, max_model_len=8192, gpu_memory_utilization=GMU, tensor_parallel_size=TP, enable_prefix_caching=False,
              max_num_batched_tokens=1024, **kw)
    long_filler = " ".join(f"Note {i}: the river was calm and the market was quiet." for i in range(480))
    c = f"Remember this: the secret code is 4721. {long_filler} Question: what is the secret code? Answer: the secret code is"
    res["3_prompt_tokens"] = len(llm.get_tokenizer()(c).input_ids)
    res["3_chunked"] = llm.generate([c], sp)[0].outputs[0].text
    tok = llm.get_tokenizer()
    text = open(__file__).read()  # fallback text if wikitext is not readable here
    try:
        from bench.vllm_prompt_ppl import wikitext
        text = wikitext()
    except Exception as ex:  # noqa: BLE001
        print("wikitext unavailable:", type(ex).__name__, flush=True)
    ids = tok(text).input_ids
    prompts = [{"prompt_token_ids": ids[20000 + i * 3000: 20000 + i * 3000 + 2500]} for i in range(8)]
    outs = llm.generate(prompts, SamplingParams(max_tokens=48, temperature=0))
    res["4_tokens"] = [list(o.outputs[0].token_ids) for o in outs]
    print("JSON " + json.dumps({"config": name, **res}), flush=True)


def main():
    if sys.argv[1] == "--child":
        return child(sys.argv[2])
    rows = {}
    for name in sys.argv[1:]:
        env = {k: v for k, v in os.environ.items()
               if k not in ("KV4_NO_BOUNDARY", "KV4_BOUNDARY", "KV4_KMEAN", "KV4_SINKS", "KV4_FORMAT")}
        env.setdefault("KV4_VLLM", "0")
        env.update(CONFIGS[name])
        env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        p = subprocess.run([sys.executable, __file__, "--child", name], env=env, capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("JSON ")]
        if not line:
            print(f"--- {name} FAILED\n" + "\n".join((p.stdout + p.stderr).splitlines()[-25:]), flush=True)
            continue
        rows[name] = json.loads(line[-1][5:])
        r = rows[name]
        print(f"RESULT {name}: 1 alone {r['1_alone']!r} | batched {r['1_batched']!r} | 2 hit from another request "
              f"{r['2_hit_other_request']!r} (cached {r['2_cached_tokens']}) | 3 chunked ({r['3_prompt_tokens']} tokens) "
              f"{r['3_chunked']!r}", flush=True)
    if "auto" in rows:
        ref = rows["auto"]["4_tokens"]
        for name, r in rows.items():
            if name == "auto":
                continue
            pre, exact = [], 0
            for x, y in zip(ref, r["4_tokens"]):
                n = next((i for i, (u, w) in enumerate(zip(x, y)) if u != w), min(len(x), len(y)))
                pre.append(n / len(x))
                exact += x == y
            print(f"AGREE {name} vs auto: tokens up to the first divergence {sum(pre) / len(pre):.2f} (per prompt "
                  f"{', '.join(f'{v:.2f}' for v in pre)}); exact {exact}/{len(ref)}", flush=True)


if __name__ == "__main__":
    main()
