"""11-15: end-to-end check of vllm-project/vllm#67789 (skip the prefill fast path when a request has cached KV) for
UltraQuant on CUDA, which the PR did not run. Stock vLLM, no kv4 plugin (VLLM_PLUGINS=""), in-process engine.
    VLLM_PLUGINS= python bench/debug/pr67789_check.py ultraquant_4bit prefix   # dtype: auto | turboquant_k8v4 | ultraquant_4bit;
    mode: prefix | chunked (one engine per process)
Two triggers, 4 needles each (a different 4-digit code per trial), greedy, Llama-3.1-8B:
  prefix: the #46461 case. A = cached prefix + question, batched with a fresh prompt B longer than A.
  chunked: the #67789 case, prefix caching off, max_num_batched_tokens 2048. Submitted as [F, A, B]: step 1 holds F
           (1500 tokens, 1 output token) and A's first 548 tokens; step 2 holds A's 452-token continuation (seq 1000)
           and the fresh B (1300 tokens), so max_query_len == max_seq_len == 1300 while A has cached KV.
  control: A alone, in one step.
Slurm: slurm/pr67789_check.sbatch runs it on base 0.33 and on a copy of the package with the PR's diff applied.
"""
import os, sys
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import vllm
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

dt, mode = sys.argv[1], sys.argv[2]
tag = os.environ.get("PR_TAG", "base")
if dt == "ultraquant_4bit":
    # ULTRAQUANT is not in the CUDA backend priority list in 0.33; add it so each layer picks by its own KV dtype
    from vllm.platforms import cuda
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    _orig = cuda._get_backend_priorities
    cuda._get_backend_priorities = lambda *a, **k: [*_orig(*a, **k), AttentionBackendEnum.ULTRAQUANT]
print(f"vllm {vllm.__version__} from {os.path.dirname(vllm.__file__)} [{tag}]", flush=True)

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


sp = SamplingParams(max_tokens=8, temperature=0)
one = SamplingParams(max_tokens=1, temperature=0)
if True:
    kw = dict(max_model_len=4096, gpu_memory_utilization=0.6, kv_cache_dtype=dt)
    if mode == "prefix":
        llm = LLM("meta-llama/Llama-3.1-8B", enable_prefix_caching=True, **kw)
    else:
        llm = LLM("meta-llama/Llama-3.1-8B", enable_prefix_caching=False, enable_chunked_prefill=True,
                  max_num_batched_tokens=M, **kw)
    tok = llm.get_tokenizer()
    ok_ctl = ok = 0
    outs = []
    for code in CODES:
        a, f, b = build(tok, code)
        llm.reset_prefix_cache()
        ctl = llm.generate([TokensPrompt(prompt_token_ids=a)], sp)[0].outputs[0].text
        llm.reset_prefix_cache()
        if mode == "prefix":
            pre = a[:576]                                   # 9 blocks of 64: cached, A then hits it
            llm.generate([TokensPrompt(prompt_token_ids=pre)], one)
            o = llm.generate([TokensPrompt(prompt_token_ids=a), TokensPrompt(prompt_token_ids=b)], sp)
            got = o[0].outputs[0].text
        else:
            o = llm.generate([TokensPrompt(prompt_token_ids=x) for x in (f, a, b)], [one, sp, sp])
            got = o[1].outputs[0].text
        ok_ctl += code in ctl
        ok += code in got
        outs.append(got)
    print(f"RESULT [{tag}] {dt} {mode}: control {ok_ctl}/4, trigger {ok}/4; trigger outputs {outs!r}", flush=True)
