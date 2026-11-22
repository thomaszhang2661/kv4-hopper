"""11-14: LMDeploy decode step time at batch B, context S: the steady window
where all B sequences decode and none prefills, i.e. [last sequence's first token, first sequence's last token].
Checks: n_ok == B, tokens_per_tick ~ 1, window_tokens in the hundreds. All shapes run in one event loop.

    python bench/lmdeploy_step.py meta-llama/Llama-3.1-8B --backend turbomind --qp 4 --bs 1 8 32 --ctx 16384 32768
"""
import argparse, asyncio, json, time
import numpy as np
from lmdeploy.messages import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig

sid = [0]

async def run(B, S, out):
    rng = np.random.default_rng(B * 100003 + S)
    prompts = [rng.integers(1000, 30000, S).tolist() for _ in range(B)]
    tr = [[] for _ in range(B)]
    gc = GenerationConfig(max_new_tokens=out, ignore_eos=True, top_k=1)
    async def one(i):
        sid[0] += 1
        inst = eng.create_instance()
        gen = inst.async_stream_infer(sid[0], input_ids=prompts[i], gen_config=gc, stream_output=True)
        n = 0
        try:
            async for o in gen:
                n += len(o.token_ids)                   # incremental tokens, as in profile_throughput
                tr[i].append((time.perf_counter(), n))
        finally:
            await gen.aclose()
    await asyncio.gather(*[one(i) for i in range(B)])
    t0 = max(t[0][0] for t in tr); t1 = min(t[-1][0] for t in tr)   # window in which all B sequences are decoding
    steps, ticks = [], []
    for t in tr:
        w = [x for x in t if t0 <= x[0] <= t1]
        if len(w) >= 2 and w[-1][1] > w[0][1]:
            steps.append((w[-1][0] - w[0][0]) / (w[-1][1] - w[0][1]))
            ticks.append((w[-1][1] - w[0][1]) / (len(w) - 1))
    return dict(B=B, S=S, out=out, step_ms=1e3 * float(np.median(steps)) if steps else None,
                window_tokens=int(min(x[-1][1] for x in tr) - max(x[0][1] for x in tr)),
                tokens_per_tick=float(np.mean(ticks)) if ticks else None, n_ok=len(steps))



async def main():
    await run(1, 1024, 16)                              # warmup
    rows = []
    for S in a.ctx:
        for B in a.bs:
            r = await run(B, S, a.out)
            r.update(backend=a.backend, qp=a.qp)
            rows.append(r)
            print("RESULT " + json.dumps(r), flush=True)
    if a.json:
        json.dump(rows, open(a.json, 'w'), indent=1)

# 11-15: the PyTorch engine starts worker processes with spawn, which re-imports this file: everything that
# parses arguments or builds an engine must sit under the main guard (job 11881741 failed without it).
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('model'); p.add_argument('--backend', default='turbomind', choices=['turbomind', 'pytorch'])
    p.add_argument('--qp', type=int, default=0)            # TM: 0/4/8; PyTorch: 0/4/8/16(fp8)
    p.add_argument('--bs', type=int, nargs='+', default=[1, 8, 32])
    p.add_argument('--ctx', type=int, nargs='+', default=[16384, 32768])
    p.add_argument('--out', type=int, default=512)         # must be much larger than the prefill offset (B*S/8192 steps)
    p.add_argument('--cache', type=float, default=0.9)
    p.add_argument('--json', default=None)
    a = p.parse_args()
    kw = dict(quant_policy=a.qp, max_batch_size=max(a.bs), session_len=max(a.ctx) + a.out + 64,
              cache_max_entry_count=a.cache)
    if a.backend == 'turbomind':
        from lmdeploy.turbomind import TurboMind
        eng = TurboMind.from_pretrained(a.model, engine_config=TurbomindEngineConfig(**kw))
    else:
        from lmdeploy.pytorch.engine import Engine
        eng = Engine.from_pretrained(a.model, engine_config=PytorchEngineConfig(**kw))
    asyncio.run(main())
