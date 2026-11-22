"""LongBench (v1, English, non-ROUGE tasks) with a fake-quantized KV cache in decode mode (11-12).

Per item: one BF16 prefill, then for every config a copy of that cache is fake-quantized (decode mode, as vLLM: the
prompt is prefilled in BF16 and every cached and generated token is stored quantized; `name:4` keeps each sequence's
first 4 tokens in BF16; `_khs` formats take the K mean from the prompt) and the answer is generated greedily. So all
configs are paired on identical prefills, and the BF16 config is the reference.

Prompts, generation lengths and scoring follow THUDM/LongBench (LongBench/config/*.json copied to
bench/longbench_cfg/, metrics re-implemented from LongBench/metrics.py and eval.py): chat template except for trec,
triviaqa, samsum, lsht, lcc, repobench-p; first line only for trec / triviaqa; middle truncation to --max-len tokens.
One deviation: code similarity uses difflib.SequenceMatcher instead of fuzzywuzzy's fuzz.ratio (the same ratio
definition, 2 * matches / total length; fuzzywuzzy is not installed).

    python bench/longbench.py --model princeton-nlp/Llama-3-8B-ProLong-512k-Instruct --configs bf16,fp8,nvfp4_khs:4 \
        --tasks hotpotqa,qasper --n 50 --out results/longbench_prolong.csv
"""
import argparse
import copy
import csv
import difflib
import json
import os
import re
import string
import time
import zipfile
from collections import Counter
from pathlib import Path

import torch

from bench.accuracy_kl import CONFIGS, fake, make_cache_cls

CFG = Path(__file__).resolve().parent / "longbench_cfg"
NO_CHAT = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
FIRST_LINE = {"trec", "triviaqa", "samsum", "lsht"}
TASKS = ("narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,trec,triviaqa,passage_count,"
         "passage_retrieval_en,lcc,repobench-p")


# --- metrics (LongBench/metrics.py) -------------------------------------------------------------------------------
def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred_tokens, gt_tokens):
    common = Counter(pred_tokens) & Counter(gt_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    p, r = same / len(pred_tokens), same / len(gt_tokens)
    return 2 * p * r / (p + r)


def qa_f1(pred, gt, **_):
    return f1(normalize_answer(pred).split(), normalize_answer(gt).split())


def count_score(pred, gt, **_):
    nums = re.findall(r"\d+", pred)
    return 0.0 if not nums else sum(n == str(gt) for n in nums) / len(nums)


def retrieval_score(pred, gt, **_):
    gid = re.findall(r"Paragraph (\d+)", gt)[0]
    nums = re.findall(r"\d+", pred)
    return 0.0 if not nums else sum(n == gid for n in nums) / len(nums)


def code_sim(pred, gt, **_):
    line = ""
    for ln in pred.lstrip("\n").split("\n"):
        if "`" not in ln and "#" not in ln and "//" not in ln:
            line = ln
            break
    return round(difflib.SequenceMatcher(None, line, gt).ratio() * 100) / 100  # fuzz.ratio rounds to an integer


def classification(pred, gt, all_classes=None, **_):
    em = [c for c in all_classes if c in pred]
    em = [c for c in em if not (c in gt and c != gt)]
    return 1.0 / len(em) if gt in em else 0.0


METRIC = {"narrativeqa": qa_f1, "qasper": qa_f1, "multifieldqa_en": qa_f1, "hotpotqa": qa_f1, "2wikimqa": qa_f1,
          "musique": qa_f1, "trec": classification, "triviaqa": qa_f1, "passage_count": count_score,
          "passage_retrieval_en": retrieval_score, "lcc": code_sim, "repobench-p": code_sim}


def score(task, pred, answers, all_classes):
    if task in FIRST_LINE:
        pred = pred.lstrip("\n").split("\n")[0]
    return max(METRIC[task](pred, gt, all_classes=all_classes) for gt in answers)


# --- data ---------------------------------------------------------------------------------------------------------
def load_task(task):
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download("zai-org/LongBench", repo_type="dataset", allow_patterns=["data.zip"]))
    with zipfile.ZipFile(root / "data.zip") as z:
        name = next(n for n in z.namelist() if n.endswith(f"/{task}.jsonl") or n == f"{task}.jsonl")
        return [json.loads(line) for line in z.read(name).decode().splitlines() if line.strip()]


def build_ids(tok, task, item, prompt_fmt, max_len):
    prompt = prompt_fmt.format(**item)
    ids = tok(prompt, add_special_tokens=False).input_ids
    if len(ids) > max_len:  # LongBench truncates in the middle
        half = max_len // 2
        prompt = tok.decode(ids[:half], skip_special_tokens=True) + tok.decode(ids[-half:], skip_special_tokens=True)
    if task not in NO_CHAT and tok.chat_template:
        return tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True)["input_ids"]
    return tok(prompt, return_tensors="pt").input_ids  # with BOS if the tokenizer adds one


# --- generation ---------------------------------------------------------------------------------------------------
def quantize_cache(cache, kf, vf, sink):
    for li, layer in enumerate(cache.layers):
        mu = None
        if kf is not None and kf.endswith("~s"):
            mu = layer.keys[:, :, sink:].float().mean(2, keepdim=True)
            cache.kmu[li] = mu
        layer.keys = fake(layer.keys, kf, sink, mu)
        layer.values = fake(layer.values, vf, sink)
    cache.kfmt, cache.vfmt = kf, vf


def greedy(model, cache, first_logits, new, eos, start):
    nxt = first_logits.argmax(-1, keepdim=True)
    out = []
    pos = start
    for _ in range(new):
        t = nxt.item()
        if t in eos:
            break
        out.append(t)
        o = model(nxt, use_cache=True, past_key_values=cache, position_ids=torch.tensor([[pos]], device=nxt.device))
        nxt = o.logits[:, -1].argmax(-1, keepdim=True)
        pos += 1
    return out


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="princeton-nlp/Llama-3-8B-ProLong-512k-Instruct")
    p.add_argument("--configs", default="bf16,fp8,nvfp4_khs:4")
    p.add_argument("--tasks", default=TASKS)
    p.add_argument("--n", type=int, default=50, help="items per task (the first n)")
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--max-len", type=int, default=65536)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    commit = os.environ.get("KV4_COMMIT", "unset")
    prompts = json.loads((CFG / "dataset2prompt.json").read_text())
    maxgen = json.loads((CFG / "dataset2maxlen.json").read_text())
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) | {tok.eos_token_id}
    Cache = make_cache_cls()
    specs = [(s, *CONFIGS[s.partition(":")[0]], int(s.partition(":")[2] or 0)) for s in args.configs.split(",")]
    print(f"{args.model} LongBench {args.tasks}, n {args.n} from {args.first}, configs {args.configs}, commit {commit},"
          f" {torch.cuda.get_device_name()}", flush=True)
    rows, t0 = [], time.time()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for task in args.tasks.split(","):
        data = load_task(task)[args.first:args.first + args.n]
        sums = Counter()
        for i, item in enumerate(data):
            ids = build_ids(tok, task, item, prompts[task], args.max_len).to("cuda")
            with torch.no_grad():
                base = Cache(None, None, 0)
                out = model(ids, use_cache=True, past_key_values=base, logits_to_keep=1)
                first = out.logits[:, -1]
                for spec, kf, vf, sink in specs:
                    cache = copy.deepcopy(base)
                    if kf is not None:
                        quantize_cache(cache, kf, vf, sink)
                    gen = greedy(model, cache, first, maxgen[task], eos, ids.shape[1])
                    pred = tok.decode(gen, skip_special_tokens=True)
                    sc = score(task, pred, item["answers"], item.get("all_classes"))
                    sums[spec] += sc
                    rows.append(dict(model=args.model, task=task, idx=args.first + i, config=spec, k_fmt=kf or "bf16",
                                     v_fmt=vf or "bf16", sink=sink, prompt_tokens=ids.shape[1], score=f"{sc:.4f}",
                                     pred=pred[:300], commit=commit))
                    del cache
                del base, out
            torch.cuda.empty_cache()
        line = "  ".join(f"{s} {100 * sums[s] / len(data):.1f}" for s, *_ in specs)
        print(f"{task:22s} n={len(data):3d}: {line} ({time.time() - t0:.0f} s)", flush=True)
        with open(args.out, "w", newline="") as f:  # rewrite after every task, so a time-out keeps finished tasks
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print("saved", args.out)


if __name__ == "__main__":
    main()
