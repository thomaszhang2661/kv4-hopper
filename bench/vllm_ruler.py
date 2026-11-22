"""11-12 night: the RULER-lite long-context tasks (bench/ruler_lite.py: multikey, multivalue, vt) through vLLM, with the
real KV paths instead of fake quantization: auto (BF16), fp8, the plugin's full recipe. Same generators, the same
seeded trials (crc32 of task-length-index, so trial i is the trial i of the offline runs) and the same score
(string_match_all: fraction of reference strings in the output). Greedy. One engine per config (child process).

    KV4_CALIB=results/calib/Llama-3-8B-ProLong-512k-Instruct.pt python bench/vllm_ruler.py \
        --configs auto,fp8,recipe --lens 32768,65536,122880 --trials 50 --out results/vllm_ruler_prolong.csv
"""
import argparse
import csv
import os
import random
import subprocess
import sys
import zlib

M = "princeton-nlp/Llama-3-8B-ProLong-512k-Instruct"
CONFIGS = {"auto": ("auto", {}), "fp8": ("fp8", {}),
           "recipe": ("ultraquant_4bit", {"KV4_VLLM": "1", "KV4_FORMAT": "nvfp4", "KV4_KMEAN": "1", "KV4_SINKS": "1",
                                          "KV4_BOUNDARY": "0"}),
           "nvfp4_l0": ("ultraquant_4bit", {"KV4_VLLM": "1", "KV4_FORMAT": "nvfp4", "KV4_BOUNDARY": "0"})}


def child(a):
    from vllm import LLM, SamplingParams
    from bench.ruler_lite import NEW, make_trial
    from bench.vllm_prompt_ppl import wikitext
    dt, _ = CONFIGS[a.child]
    kw = dict(kv_cache_dtype=dt)
    if dt == "ultraquant_4bit":
        kw["block_size"] = 64
    lens = [int(x) for x in a.lens.split(",")]
    llm = LLM(a.model, max_model_len=max(lens) + 128, gpu_memory_utilization=0.90, enable_prefix_caching=False,
              **kw)
    tok = llm.get_tokenizer()
    wiki = tok(wikitext(), add_special_tokens=False).input_ids
    rows = []
    for task in a.tasks.split(","):
        for L in lens:
            idx = range(a.first, a.first + a.trials)
            trials = [make_trial(task, tok, L, random.Random(zlib.crc32(f"{task}-{L}-{i}".encode())), wiki)
                      for i in idx]
            outs = llm.generate([{"prompt_token_ids": ids} for ids, _ in trials],
                                SamplingParams(max_tokens=NEW[task], temperature=0), use_tqdm=False)
            score = 0.0
            for i, (ids, refs), o in zip(idx, trials, outs):
                text = o.outputs[0].text
                s = sum(r.lower() in text.lower() for r in refs) / len(refs)
                score += s
                rows.append(dict(model=a.model, task=task, ctx=L, config=a.child, trial=i, score=f"{s:.2f}",
                                 prompt_tokens=len(ids), output=text.strip().replace("\n", " ")[:80]))
            print(f"RESULT {task:10s} ctx {L:6d} {a.child:9s}: {score / len(trials):6.1%}", flush=True)
    out = a.out.replace(".csv", f"_{a.child}.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved", out, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="auto,fp8,recipe")
    p.add_argument("--model", default=M)
    p.add_argument("--tasks", default="multikey,multivalue,vt")
    p.add_argument("--lens", default="32768,65536,122880")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--child", default="")
    a = p.parse_args()
    if a.child:
        return child(a)
    for c in a.configs.split(","):
        env = {k: v for k, v in os.environ.items()
               if k not in ("KV4_NO_BOUNDARY", "KV4_BOUNDARY", "KV4_KMEAN", "KV4_SINKS", "KV4_FORMAT")}
        env.setdefault("KV4_VLLM", "0")
        env.update(CONFIGS[c][1])
        subprocess.run([sys.executable, __file__, "--child", c] + [x for k, v in vars(a).items()
                       if k not in ("child", "configs") for x in (f"--{k}", str(v))], env=env)


if __name__ == "__main__":
    main()
