"""11-12 night: does real vLLM FP8 KV break DeepSeek-R1-Distill-Qwen-7B as fake-quant FP8 K does (fp8 0.5% vs bf16
73.5% on MATH-500)? Greedy MATH-500 first N problems through vLLM with kv_cache_dtype auto and fp8, scored with the
same grader as bench/reasoning_eval.py. One child process per dtype (a fresh engine each).

    python bench/vllm_reasoning_check.py --dtypes auto,fp8 --n 40 --new 4096 --out results/vllm_reasoning_r1q7b.csv

A dtype spec `fp8@0+27` also keeps layers 0 and 27 at native precision (`kv_cache_dtype_skip_layers`).
"""
import argparse
import csv
import json
import os
import subprocess
import sys

M = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


def repetitive(ids, tail=2048, n=32, thresh=0.5):
    """11-13: a degenerate loop in the last `tail` tokens: fewer than `thresh` of the n-grams there are distinct
    (normal text has nearly all 32-grams distinct; a loop repeats the same ones). Returns (flag, distinct ratio)."""
    t = list(ids)[-tail:]
    if len(t) < 4 * n:
        return 0, 1.0
    grams = [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
    r = len(set(grams)) / len(grams)
    return int(r < thresh), r


def child(M, dt, n, new, out):
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    from bench.reasoning_eval import PROMPT, last_boxed, norm
    root = snapshot_download("HuggingFaceH4/MATH-500", repo_type="dataset")
    probs = [json.loads(l) for l in open(os.path.join(root, "test.jsonl")) if l.strip()][:n]
    seed = int(os.environ.get("KV4_SHUFFLE_SEED", "-1"))  # 11-13: another submission order = another batching
    if seed >= 0:
        import random
        random.Random(seed).shuffle(probs)
    kv, _, skip = dt.partition("@")
    extra = dict(kv_cache_dtype_skip_layers=skip.split("+")) if skip else {}
    # 11-12: KV4_ATTN_BACKEND (e.g. FLASHINFER) to avoid the FA3 crash with skip layers (#65821, hit on R1-1.5B);
    # KV4_MAX_LEN for models with a shorter context (Qwen2.5-Math-7B-Instruct: 4096)
    if kv == "ultraquant_4bit":  # 11-12: our pages are 64 tokens; with native boundary layers vLLM would pick 32
        extra["block_size"] = 64
    if os.environ.get("KV4_ATTN_BACKEND"):
        extra["attention_backend"] = os.environ["KV4_ATTN_BACKEND"]
    if os.environ.get("KV4_TOKFIX", "1") == "1":
        # 11-14: transformers 5 loads DeepSeek-R1-Distill-Llama as LlamaTokenizer (Metaspace pre-tokenizer and decoder
        # over a ByteLevel vocab: spaces dropped when encoding, Ġ / Ċ left when decoding) unless config._name_or_path
        # matches its MODEL_IDS_TO_TOKENIZERS_BACKEND list; vLLM loads from the local snapshot path, which does not
        # match (huggingface/transformers#45488, the local-directory case). Widen the pattern to paths.
        import transformers.models.auto.tokenization_auto as _ta
        _ta.MODEL_IDS_TO_TOKENIZERS_BACKEND.append("*deepseek-r1-distill-llama*")
    llm = LLM(M, kv_cache_dtype=kv, max_model_len=int(os.environ.get("KV4_MAX_LEN", "8192")),
              gpu_memory_utilization=0.85, **extra)
    msgs = [[{"role": "user", "content": PROMPT.format(problem=p["problem"])}] for p in probs]
    if os.environ.get("KV4_TOKDBG"):  # 11-14: which tokenizer / decoder does the engine actually use (Ġ markers)
        t = llm.get_tokenizer()
        raw = getattr(t, "_tokenizer", None)
        print(f"TOKDBG llm.get_tokenizer: {type(t).__module__}.{type(t).__name__}, decoder "
              f"{str(raw.decoder)[:120] if raw is not None else None}", flush=True)
        eng = getattr(llm, "llm_engine", None)
        for name in ("tokenizer", "renderer", "output_processor", "input_processor"):
            o = getattr(eng, name, None)
            if o is not None:
                tt = getattr(o, "tokenizer", o)
                rr = getattr(tt, "_tokenizer", None)
                print(f"TOKDBG engine.{name}: {type(o).__name__} -> {type(tt).__name__}, decoder "
                      f"{str(rr.decoder)[:120] if rr is not None else None}", flush=True)
    # 11-13: KV4_NO_THINK=1 turns off Qwen3's thinking mode (chat template enable_thinking=False): short answers
    kw = dict(chat_template_kwargs={"enable_thinking": False}) if os.environ.get("KV4_NO_THINK") == "1" else {}
    res = llm.chat(msgs, SamplingParams(max_tokens=new, temperature=0), **kw)
    if os.environ.get("KV4_TOKDBG"):
        o0 = res[0].outputs[0]
        print(f"TOKDBG text {o0.text[:80]!r}\nTOKDBG get_tokenizer().decode(ids) {t.decode(list(o0.token_ids))[:80]!r}"
              f"\nTOKDBG prompt ids {list(res[0].prompt_token_ids)[:12]} ... ({len(res[0].prompt_token_ids)})", flush=True)
        import tokenizers
        from huggingface_hub import hf_hub_download
        ref = tokenizers.Tokenizer.from_file(hf_hub_download(M, "tokenizer.json"))  # the file's own ByteLevel pipeline
        pids = list(res[0].prompt_token_ids)
        print(f"TOKDBG prompt decoded by tokenizer.json: {ref.decode(pids, skip_special_tokens=False)[:300]!r}", flush=True)
    rows = []
    for p, r in zip(probs, res):
        o = r.outputs[0]
        text = o.text
        # 11-13: R1-Distill-Llama-8B's text comes back with byte-level BPE markers (Ġ space, Ċ newline) in vLLM 0.33;
        # answers with spaces then failed to match. Undo them before grading (texts without the markers unchanged).
        if text.count("Ġ") > 10:
            text = text.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")
        pred = last_boxed(text)
        rows.append(dict(model=M, config=dt, id=p["unique_id"], answer=p["answer"], pred=pred or "",
                         correct=int(pred is not None and norm(pred) == norm(p["answer"])), gen_tokens=len(o.token_ids),
                         finished=int(o.finish_reason == "stop"), hit_cap=int(o.finish_reason == "length"),
                         repetitive=repetitive(o.token_ids)[0], distinct32=round(repetitive(o.token_ids)[1], 3),
                         head=o.text[:120].replace("\n", " ")))
    # 11-13: full texts next to the csv, to inspect loops by hand
    with open(out.rsplit(".", 1)[0] + "_text.jsonl", "w") as f:
        for p, r in zip(probs, res):
            d = dict(id=p["unique_id"], text=r.outputs[0].text)
            if os.environ.get("KV4_SAVE_IDS"):  # 11-14: token ids too, to check the Ġ / Ċ markers against HF decode
                d["ids"] = list(r.outputs[0].token_ids)
            f.write(json.dumps(d) + "\n")
    acc = sum(r["correct"] for r in rows) / len(rows)
    tagenv = ",".join(f"{k}={os.environ[k]}" for k in ("KV4_VLLM", "KV4_BOUNDARY", "KV4_KCENTER") if k in os.environ)
    print(f"RESULT {dt} [{tagenv}]: {acc:.1%} correct, finished {sum(r['finished'] for r in rows)}/{len(rows)}, mean "
          f"{sum(r['gen_tokens'] for r in rows) / len(rows):.0f} tokens, hit the cap {sum(r['hit_cap'] for r in rows)}, "
          f"repetitive {sum(r['repetitive'] for r in rows)}; first: {rows[0]['head']!r}", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtypes", default="auto,fp8")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--new", type=int, default=4096)
    p.add_argument("--out", required=True)
    p.add_argument("--child", default="")
    p.add_argument("--model", default=M)  # 11-12: other Qwen2.5-family models
    p.add_argument("--shuffle-seed", type=int, default=-1, help="11-13: shuffle the submission order (rows by id)")
    a = p.parse_args()
    if a.child:
        return child(a.model, a.child, a.n, a.new, a.out)
    parts = []
    # 11-12: a spec may carry env overrides for its child after "!", e.g. "ultraquant_4bit!KV4_VLLM=1;KV4_BOUNDARY=0"
    # (";" between pairs, since "," separates specs)
    for spec in a.dtypes.split(","):
        dt, _, envs = spec.partition("!")
        env = dict(os.environ, **dict(kv.split("=", 1) for kv in envs.split(";") if kv))
        env["KV4_SHUFFLE_SEED"] = str(a.shuffle_seed)
        tag = spec.replace("@", "_skip").replace("+", "-").replace("!", "_").replace(";", "_").replace("=", "").replace("/", "-")  # 11-16: "/" from a KV4_CALIB path
        part = a.out.replace(".csv", f"_{tag}.csv")
        subprocess.run([sys.executable, __file__, "--model", a.model, "--child", dt, "--n", str(a.n), "--new",
                        str(a.new), "--out", part], env=env)
        if os.path.exists(part):
            parts.append(part)
    print("parts:", parts)


if __name__ == "__main__":
    main()
