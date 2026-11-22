"""Why K rotation helps NVFP4 on Llama / Mistral but hurts it 10x on Qwen3 (11-11 night): per-model statistics of the
real post-RoPE K and V (one 4096-token wikitext-2 window, all layers), and the reconstruction error of each format with
and without rotation and with and without subtracting the per-channel mean over tokens (K smoothing).

  mean energy share = ||mean over tokens||^2 / mean over tokens of ||k||^2 (per layer and kv head; 1 = every token's K
  is the same vector); top-channel share = largest per-channel mean square / total.

    python bench/kv_stats.py --models meta-llama/Llama-3.1-8B,mistralai/Mistral-7B-v0.3,Qwen/Qwen3-8B-Base
"""
import argparse
import os

import torch

from kv4.formats import dequantize, quantize


def rel(x, fmt, smooth):
    mu = x.mean(-2, keepdim=True) if smooth else 0.0
    y = dequantize(quantize(x - mu, fmt)) + mu
    return ((y - x).pow(2).sum() / x.pow(2).sum()).item()


def main():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="meta-llama/Llama-3.1-8B,mistralai/Mistral-7B-v0.3,Qwen/Qwen3-8B-Base")
    p.add_argument("--ctx", type=int, default=4096)
    args = p.parse_args()
    print("commit", os.environ.get("KV4_COMMIT", "unset"), torch.cuda.get_device_name(), flush=True)
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    for name in args.models.split(","):
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
        x = tok(text, return_tensors="pt").input_ids[0][: args.ctx].unsqueeze(0).cuda()
        cache = DynamicCache()
        with torch.no_grad():
            model(x, use_cache=True, past_key_values=cache, logits_to_keep=1)
        print(f"== {name}", flush=True)
        for kind in ("K", "V"):
            t = torch.cat([(l.keys if kind == "K" else l.values)[:, :, 4:].float() for l in cache.layers], 1)  # skip sink
            mu = t.mean(2, keepdim=True)
            share = (mu.pow(2).sum(-1) / t.pow(2).sum(-1).mean(2, keepdim=True)).flatten()
            top = (mu.pow(2).amax(-1) / t.pow(2).mean(2).sum(-1, keepdim=True)).flatten()
            amax = t.abs().amax().item()
            print(f"  {kind}: mean energy share median {share.median():.2f}, max {share.max():.2f}; top channel share "
                  f"median {top.median():.2f}, max {top.max():.2f}; max |x| {amax:.0f}, rms {t.pow(2).mean().sqrt():.2f}",
                  flush=True)
            for fmt in ("nvfp4", "mxfp4_c0.200", "int4a_tok"):
                r = {f"{'rot' if h else 'raw'}{'+smooth' if sm else ''}": rel(t, fmt + ("_h" if h else ""), sm)
                     for h in (0, 1) for sm in (0, 1)}
                print(f"    {fmt:13s} rel MSE " + "  ".join(f"{k} {v:.4f}" for k, v in r.items()), flush=True)
        del model, cache
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
