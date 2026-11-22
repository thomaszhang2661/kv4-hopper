"""Reasoning accuracy with a fake-quantized KV cache (11-11): DeepSeek-R1-Distill-Qwen-7B on MATH-500, greedy, long
chains of thought, where quantization errors can accumulate over thousands of generated tokens.

Decode mode as vLLM: the prompt is prefilled in BF16, the cache is then quantized (with `:4` in a config name the first
4 real tokens of each row stay BF16), and every generated token is quantized as it is written. Batched over problems
with left padding. Score: the last \\boxed{...} of the output against the reference answer after light normalization
(the same grader for every config, so paired differences are what matters).

Note: this model has 28 query heads and 4 KV heads (G = 7), which our kernels do not support; the formats are what is
evaluated here.

    python bench/reasoning_eval.py --configs bf16,nvfp4_kh,nvfp4_kh:4 --n 200 --out results/reasoning_r1q7b.csv

K mean removal (`_khs` configs, 11-11 night): the per-channel K mean is taken from each row's prompt at prefill and
reused for every generated token, as a deployment would do.
"""
import argparse
import csv
import os
import re
import time

import torch

from bench.accuracy_kl import CONFIGS, fake, make_cache_cls

PROMPT = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
        k += 1
    return None


def norm(s):
    if s is None:
        return None
    s = s.strip()
    for a, b in (("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""), ("dfrac", "frac"),
                 ("tfrac", "frac"), ("^\\circ", ""), ("^{\\circ}", ""), ("\\%", ""), ("$", ""), (" ", ""),
                 ("\\text{", "{"), ("\\mbox{", "{")):
        s = s.replace(a, b)
    s = s.rstrip(".")
    s = re.sub(r"\{([^{}]*)\}", lambda m: "{" + m.group(1) + "}", s)
    if s.startswith("{") and s.endswith("}") and s.count("{") == 1:
        s = s[1:-1]
    try:
        return str(float(s.replace(",", "")))
    except ValueError:
        return s


def run_config(model, tok, problems, kf, vf, sink, bs, new, Cache, skip=()):
    outs = []
    for i0 in range(0, len(problems), bs):
        chunk = problems[i0:i0 + bs]
        texts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(problem=p["problem"])}],
                                         tokenize=False, add_generation_prompt=True) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        ids, am = enc.input_ids, enc.attention_mask
        pads = (am == 0).sum(-1).tolist()
        with torch.no_grad():
            cache = Cache(None, None, 0)
            pos = (am.cumsum(-1) - 1).clamp(min=0)
            out = model(ids, attention_mask=am, position_ids=pos, use_cache=True, past_key_values=cache,
                        logits_to_keep=1)
            if kf is not None:
                smooth = kf.endswith("~s")
                for li, layer in enumerate(cache.layers):
                    if li in skip:  # kept unquantized, as vLLM's --kv-cache-dtype-skip-layers
                        continue
                    mus = []
                    for r, p in enumerate(pads):
                        s0 = p + sink  # the first `sink` real tokens of row r stay BF16; padding is masked anyway
                        k = layer.keys[r:r + 1, :, s0:]
                        # `~s`: K mean over this row's prompt (after the sink), kept for every generated token. The
                        # BF16 sink keys stay raw here: in fake quant the mean is added back, so nothing differs.
                        mu = k.float().mean(2, keepdim=True) if smooth else None
                        mus.append(mu)
                        layer.keys[r:r + 1, :, s0:] = fake(k, kf, 0, mu)
                        layer.values[r:r + 1, :, s0:] = fake(layer.values[r:r + 1, :, s0:], vf, 0)
                    if smooth:
                        cache.kmu[li] = torch.cat(mus, 0)
                cache.kfmt, cache.vfmt = kf, vf
                cache.skip = set(skip)
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            gen = [nxt]
            done = nxt[:, 0] == tok.eos_token_id
            cur = pos[:, -1:] + 1
            for _ in range(new - 1):
                am = torch.cat([am, torch.ones_like(nxt)], -1)
                out = model(nxt, attention_mask=am, position_ids=cur, use_cache=True, past_key_values=cache)
                nxt = out.logits[:, -1].argmax(-1, keepdim=True)
                nxt = torch.where(done[:, None], torch.full_like(nxt, tok.eos_token_id), nxt)
                gen.append(nxt)
                done |= nxt[:, 0] == tok.eos_token_id
                cur = cur + 1
                if bool(done.all()):
                    break
            g = torch.cat(gen, -1).tolist()
        for row in g:
            n = row.index(tok.eos_token_id) if tok.eos_token_id in row else len(row)
            outs.append((tok.decode(row[:n], skip_special_tokens=True), n, n < len(row)))
        del cache, out
        torch.cuda.empty_cache()
    return outs


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    p.add_argument("--configs", default="bf16,nvfp4_kh")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--bs", type=int, default=100)
    p.add_argument("--new", type=int, default=4096)
    p.add_argument("--out", required=True)
    p.add_argument("--skip-layers", default="", help="e.g. 0 or 0,27: these layers keep a BF16 cache")
    args = p.parse_args()
    skip = tuple(int(x) for x in args.skip_layers.split(",") if x)
    commit = os.environ.get("KV4_COMMIT", "unset")
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    # 11-12: the dataset was fetched with snapshot_download (slurm/hf_download.sbatch), not into the datasets cache, so
    # load_dataset fails offline; read the snapshot's test.jsonl directly
    import json
    from huggingface_hub import snapshot_download
    root = snapshot_download("HuggingFaceH4/MATH-500", repo_type="dataset")
    with open(os.path.join(root, "test.jsonl")) as f:
        problems = [json.loads(line) for line in f if line.strip()][: args.n]
    Cache = make_cache_cls()
    print(f"{args.model} MATH-500 first {len(problems)}, greedy, max {args.new} new tokens, decode mode, commit "
          f"{commit}, {torch.cuda.get_device_name()}", flush=True)
    rows, t0 = [], time.time()
    for spec in args.configs.split(","):
        name, _, sk = spec.partition(":")
        sink = int(sk or 0)
        kf, vf = CONFIGS[name]
        outs = run_config(model, tok, problems, kf, vf, sink, args.bs, args.new, Cache, skip)
        right = 0
        for prob, (text, n, finished) in zip(problems, outs):
            pred = last_boxed(text)
            ok = pred is not None and norm(pred) == norm(prob["answer"])
            right += ok
            rows.append(dict(model=args.model, config=spec + (f"@{args.skip_layers}" if skip else ""), k_fmt=kf or "bf16", v_fmt=vf or "bf16", sink=sink,
                             id=prob["unique_id"], level=prob["level"], answer=prob["answer"], pred=pred or "",
                             correct=int(ok), gen_tokens=n, finished=int(finished), commit=commit))
        fin = sum(o[2] for o in outs)
        print(f"{spec:14s}: {right}/{len(problems)} = {right / len(problems):.1%} correct, finished {fin}, mean "
              f"{sum(o[1] for o in outs) / len(outs):.0f} tokens ({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved", args.out)


if __name__ == "__main__":
    main()
