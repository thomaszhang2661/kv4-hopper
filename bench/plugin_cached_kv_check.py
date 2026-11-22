"""11-19: end-to-end regression test of the kv4 plugin (full recipe) on the two paths where vLLM's TurboQuant and
UltraQuant backends lost cached KV (vllm#46460 / #46461, fixed by #67789): prefix-cache hits and chunked prefill.
Our plugin decides per request whether a prefill chunk reads cached KV (backend.py _prefill_attention), so it should
pass both; this is the test that keeps it so. Same triggers as bench/debug/pr67789_check.py (4 needles, greedy):
  prefix:  A = cached prefix + question, batched with a fresh prompt B longer than A.
  chunked: prefix caching off, max_num_batched_tokens 2048; [F, A, B] so that A's continuation shares a step with
           the fresh, longer B.
  control: A alone, in one step.
Prints one RESULT line per mode; exit status 0 iff control and trigger find every needle in every mode.
Runs in the vLLM env with the plugin on PYTHONPATH (slurm/vllm_kv4.sbatch, or tests/gpu/test_vllm_plugin.py):
    KV4_SCRIPT=bench/plugin_cached_kv_check.py sbatch ... slurm/vllm_kv4.sbatch [prefix|chunked]
"""
import os
import subprocess
import sys

MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.1-8B")
CALIB = os.environ.get("KV4_CALIB", "results/calib/Llama-3.1-8B.pt")
RECIPE = dict(KV4_VLLM="1", KV4_CALIB=CALIB, KV4_FORMAT="nvfp4", KV4_KMEAN="1", KV4_SINKS="1", KV4_BOUNDARY="0")

CHILD = r'''
import os, sys
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
mode, model = sys.argv[1], sys.argv[2]
M, LF, LA, LB = 2048, 1500, 1000, 1300
CODES = ["4721", "8306", "5194", "2687"]


def build(tok, code):
    head = tok.encode(f"Remember this: the secret code is {code}.", add_special_tokens=True)
    q = tok.encode(" Question: what is the secret code? Answer: the secret code is", add_special_tokens=False)
    fill = tok.encode(" ".join(f"Line {i}: the weather was mild and nothing else happened that day."
                               for i in range(200)), add_special_tokens=False)
    a = head + fill[:LA - len(head) - len(q)] + q
    other = tok.encode(" ".join(f"Entry {i}: an unrelated note about gardening, soil and rain." for i in range(400)),
                       add_special_tokens=False)
    f = [tok.bos_token_id] + other[:LF - 1]
    b = [tok.bos_token_id] + other[LF:LF + LB - 1]
    assert len(a) == LA and len(f) == LF and len(b) == LB
    return a, f, b


sp, one = SamplingParams(max_tokens=8, temperature=0), SamplingParams(max_tokens=1, temperature=0)
kw = dict(max_model_len=4096, gpu_memory_utilization=0.6, kv_cache_dtype="ultraquant_4bit", block_size=64)
if mode == "prefix":
    llm = LLM(model, enable_prefix_caching=True, **kw)
else:
    llm = LLM(model, enable_prefix_caching=False, enable_chunked_prefill=True, max_num_batched_tokens=M, **kw)
tok = llm.get_tokenizer()
ok_ctl = ok = 0
outs = []
for code in CODES:
    a, f, b = build(tok, code)
    llm.reset_prefix_cache()
    ctl = llm.generate([TokensPrompt(prompt_token_ids=a)], sp)[0].outputs[0].text
    llm.reset_prefix_cache()
    if mode == "prefix":
        llm.generate([TokensPrompt(prompt_token_ids=a[:576])], one)  # 9 blocks of 64: cached, A then hits them
        got = llm.generate([TokensPrompt(prompt_token_ids=a), TokensPrompt(prompt_token_ids=b)], sp)[0].outputs[0].text
    else:
        got = llm.generate([TokensPrompt(prompt_token_ids=x) for x in (f, a, b)], [one, sp, sp])[1].outputs[0].text
    ok_ctl += code in ctl
    ok += code in got
    outs.append(got)
print(f"RESULT plugin recipe {mode}: control {ok_ctl}/4, trigger {ok}/4; trigger outputs {outs!r}", flush=True)
'''


def run(mode):
    """One engine per process (vLLM state, plugin registration). Returns (passed, RESULT line or stderr tail)."""
    env = dict(os.environ, MODEL=MODEL, **RECIPE)
    p = subprocess.run([sys.executable, "-c", CHILD, mode, MODEL], env=env, capture_output=True, text=True)
    line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
    if p.returncode or not line:
        return False, f"{mode}: exit {p.returncode}; stderr tail:\n" + "\n".join(p.stderr.splitlines()[-30:])
    return "control 4/4, trigger 4/4" in line[0], line[0]


def main():
    ok = True
    for mode in sys.argv[1:] or ["prefix", "chunked"]:
        good, msg = run(mode)
        ok &= good
        print(msg, "-> ok" if good else "-> FAIL", flush=True)
    print("plugin cached-KV checks", "ok" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
