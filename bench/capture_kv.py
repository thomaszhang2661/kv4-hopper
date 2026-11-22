"""Capture real decode-step tensors from Llama-3.1-8B for kernel checks (11-11; runs in an env with transformers).

For a few layers: the KV cache of an 8192-token text (post-RoPE K and V, as the cache stores them) and the query of
the last token (q_proj output with RoPE), i.e. exactly the inputs of one decode step attending to all 8192 positions.
This is NOT the accuracy evaluation (KL / needle are separate): it only gives our kernels real data distributions (outlier
channels, attention sinks) to be checked against a reference computed from the same quantized values.

    python bench/capture_kv.py --out /scratch/$USER/kv4_realkv/llama31_8b.pt
"""
import argparse
from pathlib import Path

import torch


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--tokens", type=int, default=8192)
    p.add_argument("--layers", default="0,1,2,15,31")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    try:
        from datasets import load_dataset
        text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
        src = "wikitext-2 test"
    except Exception as ex:  # noqa: BLE001
        print("datasets unavailable, using repeated text:", type(ex).__name__)
        text = open(__file__).read() * 200
        src = "this file repeated"
    ids = tok(text, return_tensors="pt").input_ids[:, : args.tokens].cuda()
    assert ids.shape[1] == args.tokens
    layers = [int(x) for x in args.layers.split(",")]
    cap = {}
    hooks = []
    for li in layers:
        attn = model.model.layers[li].self_attn
        hooks.append(attn.q_proj.register_forward_hook(lambda m, i, o, li=li: cap.__setitem__(("q", li), o[:, -1:].detach())))
    hooks.append(model.model.rotary_emb.register_forward_hook(lambda m, i, o: cap.__setitem__("rope", (o[0][:, -1:], o[1][:, -1:]))))
    with torch.no_grad():
        out = model(ids, use_cache=True)
    for h in hooks:
        h.remove()
    cache = out.past_key_values
    cfg = model.config
    hd = cfg.hidden_size // cfg.num_attention_heads
    cos, sin = cap["rope"]
    res = {"source": src, "model": args.model, "tokens": args.tokens}
    for li in layers:
        if hasattr(cache, "layers"):
            k, v = cache.layers[li].keys, cache.layers[li].values
        else:
            k, v = cache.key_cache[li], cache.value_cache[li]
        q = cap[("q", li)].view(1, 1, cfg.num_attention_heads, hd).transpose(1, 2)        # (1, 32, 1, 128)
        q = q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)
        res[li] = dict(q=q[:, :, 0].float().cpu(), k=k.cpu(), v=v.cpu())
        print(f"layer {li}: q {tuple(q.shape)} k {tuple(k.shape)} v {tuple(v.shape)} |k|max {k.abs().max().item():.1f} "
              f"|v|max {v.abs().max().item():.1f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(res, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
