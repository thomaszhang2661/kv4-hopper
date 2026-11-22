"""11-12: does stock vLLM TurboQuant / UltraQuant ignore a request's cached prefix when it shares a prefill batch with a
longer fresh prompt? `_prefill_attention` takes its no-cache fast path when max_query_len == max_seq_len. With prefix
caching, request A (shared prefix cached, only its question new: ql ~ 20, sl ~ 700) batched with a fresh request B of
1500 tokens gives max_query_len = max_seq_len = 1500, so A would attend to its question only.

Request A asks for a code stated only in the cached prefix. Runs (greedy, 8 tokens), per kv dtype, one engine each:
  alone:   warm the cache with the prefix, then A alone          (continuation path: the cache is read)
  batched: warm the cache with the prefix, then [A, B] together  (fast path if the bug is real)

    python bench/debug/uq_prefix_bug.py auto ultraquant_4bit ultraquant_4bit+plugin
"""
import os
import subprocess
import sys

M = "meta-llama/Llama-3.1-8B"


def patch_log():
    """11-12 (run 2): run 1 (11834801) answered correctly in both modes, but did not show that A hit the cache or shared a
    step with B. Log every prefill call of the first attention layer seen (engine in-process,
    VLLM_ENABLE_V1_MULTIPROCESSING=0): which path, max_query_len, max_seq_len, per-request query / seq lengths."""
    from vllm.v1.attention.backends import turboquant_attn as ta
    from vllm.v1.attention.backends import ultraquant_attn as ua
    first = []

    def wrap(cls, tag):
        orig = cls._prefill_attention

        def f(self, query, key, value, kv_cache, md, *a, **k):
            if not first:
                first.append(id(self))
            if id(self) == first[0] and md.query_start_loc.numel() <= 5:   # real requests, not the dummy run
                qsl = md.query_start_loc_cpu if getattr(md, "query_start_loc_cpu", None) is not None else md.query_start_loc
                sl = md.seq_lens_cpu if getattr(md, "seq_lens_cpu", None) is not None else md.seq_lens
                q = qsl.tolist()
                print(f"PREFILL {tag}: max_q {md.max_query_len} max_s {md.max_seq_len} fast "
                      f"{md.max_query_len == md.max_seq_len} q {[q[i + 1] - q[i] for i in range(len(q) - 1)]} "
                      f"s {sl.tolist()}", flush=True)
            return orig(self, query, key, value, kv_cache, md, *a, **k)
        cls._prefill_attention = f
    wrap(ua.UltraQuantAttentionImpl, "ultraquant")
    wrap(ta.TurboQuantAttentionImpl, "turboquant")


def child(dt):
    from vllm import LLM, SamplingParams
    if dt.startswith(("ultraquant", "turboquant")):
        patch_log()
    kw = dict(kv_cache_dtype=dt, attention_backend="ULTRAQUANT") if dt.startswith("ultraquant") else \
        dict(kv_cache_dtype=dt, attention_backend="TURBOQUANT") if dt.startswith("turboquant") else {}
    llm = LLM(M, max_model_len=4096, gpu_memory_utilization=0.6, enable_prefix_caching=True, **kw)
    filler = " ".join(f"Line {i}: the weather was mild and nothing else happened that day." for i in range(40))
    prefix = f"Remember this: the secret code is 4721. {filler}"
    a = prefix + " Question: what is the secret code? Answer: the secret code is"
    b = " ".join(f"Entry {i}: an unrelated note about gardening, soil and rain." for i in range(110))
    sp = SamplingParams(max_tokens=8, temperature=0)
    tok = llm.get_tokenizer()
    print(f"LEN {dt}: prefix {len(tok(prefix).input_ids)} A {len(tok(a).input_ids)} B {len(tok(b).input_ids)} tokens")
    llm.generate([prefix], SamplingParams(max_tokens=1, temperature=0))          # warm the prefix cache
    print("RUN alone", flush=True)
    ra = llm.generate([a], sp)[0]
    llm.reset_prefix_cache()
    llm.generate([prefix], SamplingParams(max_tokens=1, temperature=0))
    print("RUN batched", flush=True)
    rb = llm.generate([a, b], sp)[0]
    print(f"RESULT {dt}: alone {ra.outputs[0].text!r} (cached {ra.num_cached_tokens}) | batched with a longer fresh "
          f"prompt {rb.outputs[0].text!r} (cached {rb.num_cached_tokens})", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        child(sys.argv[2])
    else:
        for dt in sys.argv[1:]:  # "ultraquant_4bit+plugin": the same with our kernels (KV4_VLLM=1)
            env = dict(os.environ, KV4_VLLM="1" if dt.endswith("+plugin") else "0", KV4_NO_BOUNDARY="1",
                       VLLM_ENABLE_V1_MULTIPROCESSING="0")
            subprocess.run([sys.executable, __file__, "--child", dt.removesuffix("+plugin")], env=env)
