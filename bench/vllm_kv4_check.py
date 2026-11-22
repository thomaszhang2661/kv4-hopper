"""vLLM with ultraquant_4bit, stock backend vs the kv4-hopper plugin, and the BF16 cache, on the same prompts (11-12).

Each configuration runs in its own process (vLLM state, plugin registration). Prompts: short factual ones and a
passkey retrieval at about 5K and 14K tokens (one 16K prompt also exercises a multi-chunk prefill when
max_num_batched_tokens is smaller). Greedy, 24 new tokens. Reports the outputs and whether the passkeys are found.

    python bench/vllm_kv4_check.py
"""
import json
import os
import subprocess
import sys

MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.1-8B")
CHILD = r'''
import json, os, random, sys
from vllm import LLM, SamplingParams
dt, chunk = sys.argv[1], int(sys.argv[2])
random.seed(0)
filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
def passkey(n_rep, key):
    return ("There is important info hidden in a lot of irrelevant text. Find it and memorize it. " + filler * n_rep
            + f"The pass key is {key}. Remember it. {key} is the pass key. " + filler * n_rep
            + "What is the pass key? The pass key is")
prompts = ["The capital of France is", "def fibonacci(n):", passkey(110, 68431), passkey(300, 20917)]
# 11-12: vLLM 0.33's CUDA platform never auto-selects ULTRAQUANT (its priority list has no entry for
# ultraquant_4bit), so it has to be chosen explicitly; validation then runs for that backend alone
kw = {"attention_config": {"backend": "ULTRAQUANT"}} if dt == "ultraquant_4bit" else {}
llm = LLM(os.environ["MODEL"], kv_cache_dtype=dt, max_model_len=32768, gpu_memory_utilization=0.85,
          max_num_batched_tokens=chunk, enable_prefix_caching=False, **kw)
outs = llm.generate(prompts, SamplingParams(max_tokens=24, temperature=0))
print("RESULT " + json.dumps([o.outputs[0].text for o in outs]), flush=True)
'''


def run(dt, plugin, chunk):
    env = dict(os.environ, MODEL=MODEL, KV4_VLLM="1" if plugin else "0", KV4_NO_BOUNDARY="1")
    p = subprocess.run([sys.executable, "-c", CHILD, dt, str(chunk)], env=env, capture_output=True, text=True)
    line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
    if not line:
        print(f"--- {dt} plugin={plugin} chunk={chunk} FAILED; stderr tail:\n" + "\n".join(p.stderr.splitlines()[-40:]))
        return None
    return json.loads(line[0][7:])


def main():
    runs = [("auto", False, 32768), ("ultraquant_4bit", False, 32768), ("ultraquant_4bit", True, 32768),
            ("ultraquant_4bit", True, 4096)]  # the last: 16K prompt prefilled in 4K chunks (continuation path)
    keys = ["", "", "68431", "20917"]
    for dt, plugin, chunk in runs:
        res = run(dt, plugin, chunk)
        if res is None:
            continue
        found = [k in r for k, r in zip(keys, res) if k]
        print(f"=== {dt} plugin={plugin} max_num_batched_tokens={chunk}: passkeys found {found}")
        for r in res:
            print("   ", repr(r[:90]))


if __name__ == "__main__":
    main()
