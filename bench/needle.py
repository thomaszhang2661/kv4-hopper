"""Needle in a haystack with a fake-quantized KV cache (11-11; env with transformers, e.g. the fp4 conda env).

Haystack: wikitext-2 test tokens. Needle: "The special magic number for <city> is <7 digits>." inserted at a depth
(fraction of the context). Completion-style prompt, so the base model answers without instruction tuning:
    <haystack with needle> \\n\\n The special magic number for <city> is
Greedy decoding of 12 tokens; a trial passes if the 7 digits appear in the output.

Two modes (`--mode`):
  all     the prompt is processed through the quantized cache too (as bench/accuracy_kl.py): every layer of every
          prompt token attends to quantized K/V, so errors compound through the prompt; a lower bound on accuracy.
  decode  the prompt is prefilled in BF16, the whole cache is then quantized, and generation reads the quantized cache
          (new tokens are quantized as they are written); what a decode-only kernel like ours sees.
Quantized values come from kv4.formats (the same functions behind our kernels' inputs); see accuracy_kl.py.

    python bench/needle.py --configs bf16,nvfp4_kh --lens 4096,8192 --out results/needle_x.csv
"""
import argparse
import csv
import os
import random
import time

import torch

from bench.accuracy_kl import CONFIGS, fake, make_cache_cls

CITIES = ["Lisbon", "Nairobi", "Osaka", "Quito", "Tbilisi", "Winnipeg"]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--configs", default="bf16")
    p.add_argument("--mode", choices=["all", "decode"], default="decode")
    p.add_argument("--lens", default="4096,8192,16384,32768,65536")
    p.add_argument("--depths", type=int, default=11, help="evenly spaced depths from 0 to 1")
    p.add_argument("--needles", type=int, default=3)
    p.add_argument("--new", type=int, default=12, help="tokens to generate")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    commit = os.environ.get("KV4_COMMIT", "unset")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    from datasets import load_dataset
    hay = tok("\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]),
              add_special_tokens=False).input_ids
    rng = random.Random(0)
    needles = [(CITIES[i], f"{rng.randrange(10**6, 10**7)}") for i in range(args.needles)]
    Cache = make_cache_cls()
    lens = [int(x) for x in args.lens.split(",")]
    depths = [i / (args.depths - 1) for i in range(args.depths)]
    print(f"{args.model} needle, mode {args.mode}, lens {lens}, {len(depths)} depths, {len(needles)} needles, "
          f"commit {commit}, {torch.cuda.get_device_name()}", flush=True)
    rows = []
    t0 = time.time()
    for name in args.configs.split(","):
        kf, vf = CONFIGS[name]
        for L in lens:
            hits = 0
            for city, num in needles:
                needle = tok(f" The special magic number for {city} is {num}.", add_special_tokens=False).input_ids
                question = tok(f"\n\nThe special magic number for {city} is", add_special_tokens=False).input_ids
                nhay = L - len(needle) - len(question) - 1  # 1: BOS
                for d in depths:
                    pos = int(d * nhay)
                    ids = [tok.bos_token_id] + hay[:pos] + needle + hay[pos:nhay] + question
                    x = torch.tensor([ids], device="cuda")
                    with torch.no_grad():
                        if args.mode == "all":
                            cache = Cache(kf, vf, 0)
                            out = model(x, use_cache=True, past_key_values=cache, logits_to_keep=1)
                        else:
                            cache = Cache(None, None, 0)
                            out = model(x, use_cache=True, past_key_values=cache, logits_to_keep=1)
                            for layer in cache.layers:  # quantize the prefilled cache, then quantize new tokens too
                                layer.keys, layer.values = fake(layer.keys, kf, 0), fake(layer.values, vf, 0)
                            cache.kfmt, cache.vfmt = kf, vf
                        gen = []
                        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
                        for _ in range(args.new):
                            gen.append(nxt.item())
                            out = model(nxt, use_cache=True, past_key_values=cache)
                            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
                    text = tok.decode(gen)
                    ok = num in text
                    hits += ok
                    rows.append(dict(model=args.model, mode=args.mode, config=name, k_fmt=kf or "bf16",
                                     v_fmt=vf or "bf16", ctx=L, depth=f"{d:.2f}", needle=city, ok=int(ok),
                                     output=text.strip().replace("\n", " ")[:60], commit=commit))
                    del cache, out
            n = len(needles) * len(depths)
            print(f"{name:12s} mode {args.mode:6s} ctx {L:6d}: {hits}/{n} = {hits / n:.0%} ({time.time() - t0:.0f} s)",
                  flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("saved", args.out)


if __name__ == "__main__":
    main()
