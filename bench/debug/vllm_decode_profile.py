"""11-12: why is decode with our plugin 16 ms/token at B=8, 32K in vLLM when the kernels predict ~7 ms? In-process
engine (VLLM_ENABLE_V1_MULTIPROCESSING=0) so torch.profiler sees the GPU kernels: B=8 random 16K prompts, then time
prefill-only and prefill + 64 tokens, and profile a 32-token generation. One child per config:
  plugin  : ultraquant_4bit, ULTRAQUANT backend, KV4_VLLM=1
  stock   : ultraquant_4bit, ULTRAQUANT backend, KV4_VLLM=0
  fp8     : fp8, default backend
"""
import json, os, subprocess, sys, time

CONFIGS = {"plugin": ("ultraquant_4bit", "1"), "stock": ("ultraquant_4bit", "0"), "fp8": ("fp8", "0"),
           # 11-13: the full recipe (boundary layer 0 native, block 64, no forced backend) and FlashInfer fp8
           "recipe": ("ultraquant_4bit", "1"), "fp8_fi": ("fp8", "0")}
RECIPE = {"KV4_FORMAT": "nvfp4", "KV4_KMEAN": "1", "KV4_SINKS": "1", "KV4_BOUNDARY": "0"}

CHILD = r'''
import os, sys, time, torch
from vllm import LLM, SamplingParams
name, dt = sys.argv[1], sys.argv[2]
kw = {"attention_config": {"backend": "ULTRAQUANT"}} if dt == "ultraquant_4bit" else {}
if name == "recipe":
    kw = {"block_size": 64}
if name == "fp8_fi":
    kw = {"attention_backend": "FLASHINFER"}
PB, PS = int(os.environ.get("PROF_B", "8")), int(os.environ.get("PROF_S", "16000"))  # 11-12: shape switches
if os.environ.get("PROF_WQ"):
    kw["quantization"] = os.environ["PROF_WQ"]
llm = LLM("meta-llama/Llama-3.1-8B", kv_cache_dtype=dt, max_model_len=PS + 1000, gpu_memory_utilization=0.90,
          enable_prefix_caching=False, **kw)
cfg = llm.llm_engine.vllm_config.compilation_config
print("CFG cudagraph_mode", getattr(cfg, "cudagraph_mode", None), "capture sizes", getattr(cfg, "cudagraph_capture_sizes", None)[:6] if getattr(cfg, "cudagraph_capture_sizes", None) else None, flush=True)
g = torch.Generator().manual_seed(0)
prompts = [{"prompt_token_ids": torch.randint(1000, 30000, (PS,), generator=g).tolist()} for _ in range(PB)]
def run(n):
    torch.cuda.synchronize(); t = time.time()
    llm.generate(prompts, SamplingParams(max_tokens=n, min_tokens=n, temperature=0, ignore_eos=True), use_tqdm=False)
    torch.cuda.synchronize(); return time.time() - t
run(4)
t1 = min(run(1) for _ in range(2)); t65 = min(run(65) for _ in range(2))
print(f"TIME {name}: B={PB} S={PS} prefill {t1:.3f} s, decode {(t65 - t1) / 64 * 1000:.2f} ms/token", flush=True)
from torch.profiler import profile, ProfilerActivity
if os.environ.get("PROF_MODE") == "diff":
    # 11-14: decode-only kernel time = profile(prefill + 33 tokens) - profile(prefill + 1 token), per kernel name,
    # divided by the 32 extra steps: is B=1's gap to fp8 the per-layer small kernels (prep_q_rot, merge_sink_pool)?
    def kt():
        with profile(activities=[ProfilerActivity.CUDA]) as pr:
            yield
        d = {}
        for e in pr.key_averages():
            if e.device_time_total > 0:
                d[e.key] = d.get(e.key, [0.0, 0])
                d[e.key][0] += e.device_time_total; d[e.key][1] += e.count
        kt.last = d
    # 11-17: PROF_DIFF=a,b (default 1,33). At B > 1 with chunked prefill, 1,33 is wrong: in the 33-token run the
    # sequences that finish prefill first decode while the others still prefill, and the decode tokens take part of
    # the token budget, so the 33-token run has one more prefill chunk (job 11942370, B=8 x 16K: 128 extra large-M
    # GEMMs, decode-kernel counts 37.8 per step instead of 31). Use a >= the number of mixed steps (e.g. 33,65), so
    # both runs contain the same prefill and the difference is pure decode.
    import contextlib
    na, nb = (int(x) for x in os.environ.get("PROF_DIFF", "1,33").replace(":", ",").split(","))  # "33:65" inside sbatch --export
    ctx = contextlib.contextmanager(kt)
    with ctx():
        run(na)
    a = kt.last
    with ctx():
        run(nb)
    b = kt.last
    st = nb - na
    if (na, nb) != (1, 33):
        ta = min(run(na) for _ in range(2)); tb = min(run(nb) for _ in range(2))
        step = (tb - ta) / st * 1000
    else:
        step = (t65 - t1) / 64 * 1000
    rows = sorted(((b[k][0] - a.get(k, [0, 0])[0], b[k][1] - a.get(k, [0, 0])[1], k) for k in b), reverse=True)
    tot = sum(r[0] for r in rows if r[0] > 0)
    for us, n, k in rows[:24]:
        print(f"KERNEL {name}: {us / st:9.1f} us/token  n/token={n / st:7.1f}  {k[:100]}", flush=True)
    print(f"TIME {name}: decode kernel time {tot / st / 1000:.3f} ms/token (sum over kernels, steps {na + 1}-{nb}) "
          f"vs step {step:.3f} ms/token", flush=True)
    sys.exit(0)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    run(1 if os.environ.get("PROF_MODE") == "prefill" else 33)  # 11-13: PROF_MODE=prefill profiles the prefill only
ka = prof.key_averages()
cuda = sorted([e for e in ka if e.device_time_total > 0], key=lambda e: -e.device_time_total)[:28]
tot = sum(e.device_time_total for e in ka if e.device_type.name == "CUDA") if hasattr(ka[0], "device_type") else 0
for e in cuda:
    print(f"KERNEL {name}: {e.device_time_total / 1000:9.2f} ms  n={e.count:6d}  {e.key[:110]}", flush=True)
print(f"CPU {name}: self cpu total {sum(e.self_cpu_time_total for e in ka) / 1000:.1f} ms", flush=True)
'''


def main():
    for name in (sys.argv[1].split(",") if len(sys.argv) > 1 else CONFIGS):
        dt, plug = CONFIGS[name]
        env = dict(os.environ, KV4_VLLM=plug, KV4_NO_BOUNDARY="1", VLLM_ENABLE_V1_MULTIPROCESSING="0")
        if name == "recipe":
            env.pop("KV4_NO_BOUNDARY")
            env.update(RECIPE)
        p = subprocess.run([sys.executable, "-c", CHILD, name, dt], env=env, capture_output=True, text=True)
        keep = [l for l in p.stdout.splitlines() if l.split(" ")[0] in ("CFG", "TIME", "KERNEL", "CPU")]
        print("\n".join(keep) if keep else f"--- {name} FAILED\n" + "\n".join(p.stderr.splitlines()[-30:]), flush=True)


if __name__ == "__main__":
    main()
