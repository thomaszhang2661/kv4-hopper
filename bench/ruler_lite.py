"""Harder long-context tasks than single-needle retrieval, after RULER (Hsieh et al. 2024, arXiv 2404.06654), with a
fake-quantized KV cache (11-11; env with transformers, e.g. the fp4 conda env). Our own generators, RULER's base-model
prompt templates and its string_match_all score (fraction of reference strings found in the output).

Tasks:
  multikey    RULER niah_multikey_3: the whole context is "One of the special magic uuids for <uuid> is: <uuid>."
              lines; one key is queried, the answer is its 36-character UUID value.
  multivalue  RULER niah_multivalue: wikitext haystack, one key with 4 values (7-digit numbers) at random depths;
              all 4 must be returned.
  vt          RULER variable tracking: 1 chain of 4 hops (VAR A = 12345, VAR B = VAR A, ...) in repeated noise
              sentences, with a one-shot example; all 5 variable names must be returned.

Mode `decode` (default; how vLLM UltraQuant runs: BF16 prefill, then generation reads the quantized cache) or `all`
(the prompt also goes through the quantized cache); see bench/needle.py.

    python bench/ruler_lite.py --configs bf16,nvfp4_kh --lens 4096,16384,65536 --trials 20 --out results/ruler_x.csv
"""
import argparse
import csv
import os
import random
import string
import time
import uuid
import zlib

import torch

from bench.accuracy_kl import CONFIGS, fake, make_cache_cls

NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
NIAH_TMPL = ("Some special magic {t} are hidden within the following text. Make sure to memorize it. I will quiz you "
             "about the {t} afterwards.\n{ctx}\nWhat are all the special magic {t} for {q} mentioned in the provided "
             "text? The special magic {t} for {q} mentioned in the provided text are")
VT_TMPL = ("Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{ctx}\nQuestion: "
           "Find all variables that are assigned the value {q} in the text above. Answer: According to the chain(s) of "
           "variable assignment in the text above, {n} variables are assigned the value {q}, they are: ")
NEW = {"multikey": 32, "multivalue": 40, "vt": 24}


def _bos(tok):
    """[BOS] if the tokenizer has one (Qwen's base tokenizer has none)."""
    return [tok.bos_token_id] if tok.bos_token_id is not None else []


def fill_insert(tok, base_ids, inserts, budget, rng):
    """base_ids truncated so that base + inserts fit `budget` tokens; inserts (token lists) at random sorted depths."""
    n = budget - sum(len(x) for x in inserts)
    hay = base_ids[:n]
    pos = sorted(rng.randrange(0, n + 1) for _ in inserts)
    out, prev = [], 0
    for p, ins in zip(pos, inserts):
        out += hay[prev:p] + ins
        prev = p
    return out + hay[prev:]


def make_trial(task, tok, L, rng, wiki):
    """-> (prompt token ids incl. BOS, list of reference strings)."""
    enc = lambda s: tok(s, add_special_tokens=False).input_ids  # noqa: E731
    if task == "multikey":
        key = str(uuid.UUID(int=rng.getrandbits(128)))
        val = str(uuid.UUID(int=rng.getrandbits(128)))
        q = enc(NIAH_TMPL.format(t="uuids", ctx="", q=key))
        target = enc(f"One of the special magic uuids for {key} is: {val}.\n")
        budget = L - len(q) - 1 - len(target)
        lines, used = [], 0
        while True:
            k, v = str(uuid.UUID(int=rng.getrandbits(128))), str(uuid.UUID(int=rng.getrandbits(128)))
            t = enc(f"One of the special magic uuids for {k} is: {v}.\n")
            if used + len(t) > budget:
                break
            lines.append(t)
            used += len(t)
        lines.insert(rng.randrange(0, len(lines) + 1), target)
        ctx = [x for ln in lines for x in ln]
        pre, post = enc(NIAH_TMPL.format(t="uuids", ctx="\x00", q=key).split("\x00")[0]), \
            enc(NIAH_TMPL.format(t="uuids", ctx="\x00", q=key).split("\x00")[1])
        return _bos(tok) + pre + ctx + post, [val]
    if task == "multivalue":
        key = "".join(rng.choice(string.ascii_lowercase) for _ in range(8))
        vals = [str(rng.randrange(10 ** 6, 10 ** 7)) for _ in range(4)]
        pre, post = NIAH_TMPL.format(t="numbers", ctx="\x00", q=key).split("\x00")
        pre, post = enc(pre), enc(post)
        needles = [enc(f" One of the special magic numbers for {key} is: {v}.") for v in vals]
        off = rng.randrange(0, len(wiki) - L)
        ctx = fill_insert(tok, wiki[off:], needles, L - len(pre) - len(post) - 1, rng)
        return _bos(tok) + pre + ctx + post, vals
    if task == "vt":
        def chain(r):
            names = ["".join(r.choice(string.ascii_uppercase) for _ in range(5)) for _ in range(5)]
            val = str(r.randrange(10 ** 4, 10 ** 5))
            stm = [f" VAR {names[0]} = {val}."] + [f" VAR {names[i]} = VAR {names[i - 1]}." for i in range(1, 5)]
            return names, val, stm
        ex_names, ex_val, ex_stm = chain(rng)
        example = VT_TMPL.format(ctx=(NOISE * 3).join(ex_stm), q=ex_val, n=5) + " ".join(ex_names) + "\n\n"
        names, val, stm = chain(rng)
        pre, post = VT_TMPL.format(ctx="\x00", q=val, n=5).split("\x00")
        pre, post = enc(example + pre), enc(post)
        noise = enc(NOISE * (L // 10))
        ctx = fill_insert(tok, noise, [enc(s) for s in stm], L - len(pre) - len(post) - 1, rng)
        return _bos(tok) + pre + ctx + post, names
    raise ValueError(task)


def generate(model, x, kf, vf, mode, new, Cache, sink=0):
    with torch.no_grad():
        if mode == "all":
            cache = Cache(kf, vf, sink)
            out = model(x, use_cache=True, past_key_values=cache, logits_to_keep=1)
            cache.sink = 0  # generated tokens come one at a time and are all quantized
        else:
            cache = Cache(None, None, 0)
            out = model(x, use_cache=True, past_key_values=cache, logits_to_keep=1)
            for li, layer in enumerate(cache.layers):
                mu = None
                if kf is not None and kf.endswith("~s"):  # K mean over the prompt (after the sink), kept for decode
                    mu = layer.keys[:, :, sink:].float().mean(2, keepdim=True)
                    cache.kmu[li] = mu
                layer.keys, layer.values = fake(layer.keys, kf, sink, mu), fake(layer.values, vf, sink)
            cache.kfmt, cache.vfmt = kf, vf
        gen = []
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        for _ in range(new):
            gen.append(nxt.item())
            out = model(nxt, use_cache=True, past_key_values=cache)
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    return gen


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--configs", default="bf16")
    p.add_argument("--tasks", default="multikey,multivalue,vt")
    p.add_argument("--mode", choices=["all", "decode"], default="decode")
    p.add_argument("--lens", default="4096,16384,65536")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--first", type=int, default=0, help="index of the first trial (to extend an earlier run)")
    p.add_argument("--out", required=True)
    p.add_argument("--yarn", type=float, default=0.0,
                   help="YaRN factor over the native window (Qwen3: 4 -> 131072 positions, as its model card says)")
    p.add_argument("--yarn-native", type=int, default=32768)
    args = p.parse_args()
    commit = os.environ.get("KV4_COMMIT", "unset")
    tok = AutoTokenizer.from_pretrained(args.model)
    kw = {}
    if args.yarn:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model)
        native = args.yarn_native  # Qwen3's config says 40960 positions, but its YaRN recipe starts from 32768
        yarn = {"rope_type": "yarn", "factor": args.yarn, "original_max_position_embeddings": native}
        if getattr(cfg, "rope_parameters", None):   # transformers 5: rope_theta lives in the same dict
            cfg.rope_parameters = {**cfg.rope_parameters, **yarn}
        else:
            cfg.rope_scaling = yarn
        cfg.max_position_embeddings = int(native * args.yarn)
        kw["config"] = cfg
        print(f"YaRN factor {args.yarn}: {native} -> {cfg.max_position_embeddings} positions", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda", **kw)
    model.eval()
    from datasets import load_dataset
    wiki = tok("\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]),
               add_special_tokens=False).input_ids
    Cache = make_cache_cls()
    lens = [int(x) for x in args.lens.split(",")]
    print(f"{args.model} RULER-lite, mode {args.mode}, tasks {args.tasks}, lens {lens}, {args.trials} trials, "
          f"(from index {args.first}), commit {commit}, {torch.cuda.get_device_name()}", flush=True)
    rows, t0 = [], time.time()
    for task in args.tasks.split(","):
        for L in lens:
            # the same trials for every config (seeded per task and length)
            # (crc32, not hash(): str hashes are randomized per process, so jobs would get different trials)
            idx = range(args.first, args.first + args.trials)
            trials = [make_trial(task, tok, L, random.Random(zlib.crc32(f"{task}-{L}-{i}".encode())), wiki)
                      for i in idx]
            for name in args.configs.split(","):  # "name:4" keeps the first 4 tokens in BF16 (attention sink)
                base, _, sk = name.partition(":")
                kf, vf = CONFIGS[base]
                score = 0.0
                for i, (ids, refs) in zip(idx, trials):
                    x = torch.tensor([ids], device="cuda")
                    text = tok.decode(generate(model, x, kf, vf, args.mode, NEW[task], Cache, int(sk or 0)))
                    s = sum(r.lower() in text.lower() for r in refs) / len(refs)
                    score += s
                    rows.append(dict(model=args.model, mode=args.mode, task=task, ctx=L, config=name, trial=i,
                                     score=f"{s:.2f}", prompt_tokens=len(ids),
                                     output=text.strip().replace("\n", " ")[:80], commit=commit))
                print(f"{task:10s} ctx {L:6d} {name:12s} mode {args.mode:6s}: {score / len(trials):6.1%} "
                      f"({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("saved", args.out)


if __name__ == "__main__":
    main()
