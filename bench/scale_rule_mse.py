"""Why UltraQuant's MXFP4 scale constant loses on Llama (11-11): reconstruction error of real K/V (post-RoPE, all
layers, one wikitext-2 window) under s = c amax (continuous scale) and s = 2^round(log2(c amax)) (power of two, E8M0,
what UltraQuant stores), over c; OCP's 2^(floor(log2 amax) - 2) for reference.

UltraQuant (arXiv 2606.20474) calibrates c = 0.156 by minimizing E[(z - c m q(z / (c m)))^2]; on Gaussian data that
minimum is the continuous-scale one, while with the power-of-two rounding the minimum moves to about c = 0.19 to 0.20
(local check, 11-11). This measures both on real KV.

    python bench/scale_rule_mse.py --out results/scale_rule_mse_llama31_8b.csv
"""
import argparse
import csv
import os

import torch

from kv4.formats import e2m1_decode, e2m1_encode, rotate

CS = (0.125, 0.14, 0.15, 0.156, 0.166, 0.177, 0.19, 0.2, 0.21, 0.22, 0.25)


def rel_mse(x, c, mode):
    b = x.reshape(*x.shape[:-1], -1, 32)
    m = b.abs().amax(-1, keepdim=True).clamp(min=torch.finfo(torch.float32).tiny)
    if mode == "continuous":
        s = c * m
    elif mode == "pow2":
        s = torch.exp2(torch.round(torch.log2(c * m)))
    elif mode == "best2":  # 11-20 (#67176 alternative): per group, the better of the OCP exponent and the next one
        eo = torch.floor(torch.log2(m)) - 2
        ys = [e2m1_decode(e2m1_encode(b / torch.exp2(e))) * torch.exp2(e) for e in (eo, eo + 1)]
        err = torch.stack([(y - b).pow(2).sum(-1, keepdim=True) for y in ys])
        y = torch.where(err[0] <= err[1], ys[0], ys[1])
        return ((y - b).pow(2).sum() / b.pow(2).sum()).item()
    else:  # ocp
        s = torch.exp2(torch.floor(torch.log2(m)) - 2)
    y = e2m1_decode(e2m1_encode(b / s)) * s
    return ((y - b).pow(2).sum() / b.pow(2).sum()).item()


def main():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")  # 11-20: cpu works too (one 4096-token forward)
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=args.device).eval()
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    x = tok(text, return_tensors="pt").input_ids[0][: args.ctx].unsqueeze(0).to(args.device)
    cache = DynamicCache()
    with torch.no_grad():
        model(x, use_cache=True, past_key_values=cache, logits_to_keep=1)
    ks = torch.cat([l.keys.float() for l in cache.layers], 1)    # (1, layers * HKV, S, D)
    vs = torch.cat([l.values.float() for l in cache.layers], 1)
    tensors = {"K rotated": rotate(ks, 128), "K": ks, "V": vs, "V rotated": rotate(vs, 128)}
    rows = []
    commit = os.environ.get("KV4_COMMIT", "unset")
    for name, t in tensors.items():
        ocp = rel_mse(t, 0, "ocp")
        b2 = rel_mse(t, 0, "best2")
        line = [f"{name:10s} OCP {ocp:.4f} best2 {b2:.4f} |"]
        rows.append(dict(model=args.model, tensor=name, rule="ocp", c="", rel_mse=f"{ocp:.5f}", commit=commit))
        rows.append(dict(model=args.model, tensor=name, rule="best2", c="", rel_mse=f"{b2:.5f}", commit=commit))
        for mode in ("continuous", "pow2"):
            for c in CS:
                e = rel_mse(t, c, mode)
                rows.append(dict(model=args.model, tensor=name, rule=mode, c=c, rel_mse=f"{e:.5f}", commit=commit))
            best = min((r for r in rows if r["tensor"] == name and r["rule"] == mode), key=lambda r: float(r["rel_mse"]))
            uq = next(r for r in rows if r["tensor"] == name and r["rule"] == mode and r["c"] == 0.156)
            line.append(f"{mode}: best c {best['c']} ({best['rel_mse']}), c=0.156 {uq['rel_mse']} |")
        print(" ".join(line), flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved", args.out)


if __name__ == "__main__":
    main()
