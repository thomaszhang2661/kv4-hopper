"""11-12 night: offline calibration for the NVFP4 KV recipe in the vLLM plugin (runs in an env with transformers).

A cache written token by token needs fixed per-layer constants that the offline formats take from the tensor at hand:
  ts_k, ts_v     NVFP4 tensor scales (fp32): amax over 16-value blocks / (6 * 448), times HEADROOM
  ts_k_nomean    the same for K without mean removal
  mu_k           per-layer, per-kv-head K channel mean (post-RoPE, positions >= 4, i.e. without the sinks), as in
                 BitDecoding-FP4's static mean (8 prompts)
K is taken post-RoPE (as the cache stores it) and rotated (Hadamard-128) like the writer does. Text: wikitext-2 test.

    python bench/calib_kv.py --model meta-llama/Llama-3.1-8B --out results/calib/Llama-3.1-8B.pt
"""
import argparse
import os
from pathlib import Path

import torch

from kv4.formats import rotate

HEADROOM = 2.0


def block_amax(x, block=16):
    return x.reshape(*x.shape[:-1], -1, block).abs().amax(-1).amax().item()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--prompts", type=int, default=8)
    p.add_argument("--tokens", type=int, default=2048)
    p.add_argument("--out", required=True)
    # 11-12 night: calibration text. "math": MATH-500 problems 200..499 (disjoint from the first 200 used in the
    # reasoning checks) in the chat template, for an in-domain vs cross-domain (wikitext) comparison
    p.add_argument("--text", default="wikitext", choices=["wikitext", "math"])
    # 11-13: --device cpu runs the calibration on a CPU node (short partition; BF16 weights, fp32 statistics), e.g.
    # for 14B-72B models that would otherwise hold one or two H200s
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    # 11-14: float32 for GPUs without BF16 (V100 on courses-gpu); the plugin's calibrations are bfloat16
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map=a.device).eval()
    dev = a.device
    if a.text == "math":
        import json as _json
        from huggingface_hub import snapshot_download
        from bench.reasoning_eval import PROMPT
        root = snapshot_download("HuggingFaceH4/MATH-500", repo_type="dataset")
        probs = [_json.loads(l) for l in open(os.path.join(root, "test.jsonl")) if l.strip()][200:]
        text = "".join(tok.apply_chat_template([{"role": "user", "content": PROMPT.format(problem=q["problem"])}],
                                               tokenize=False, add_generation_prompt=True) for q in probs)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        assert len(ids) >= a.prompts * a.tokens, f"math text has {len(ids)} tokens"
    else:
        from datasets import load_dataset
        text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
        ids = tok(text, return_tensors="pt").input_ids[0]
    n_layers = model.config.num_hidden_layers
    s_k = [None] * n_layers
    cnt = 0
    keep = []
    for i in range(a.prompts):  # pass 1: means (positions >= 4); keep the caches for pass 2
        x = ids[i * a.tokens: (i + 1) * a.tokens][None].to(dev)
        if tok.bos_token_id is not None and x[0, 0] != tok.bos_token_id:
            x = torch.cat([torch.tensor([[tok.bos_token_id]], device=dev), x[:, :-1]], 1)
        with torch.no_grad():
            cache = model(x, use_cache=True).past_key_values
        kv = []
        for li in range(n_layers):
            lay = cache.layers[li] if hasattr(cache, "layers") else None
            k, v = (lay.keys, lay.values) if lay is not None else cache[li]
            k, v = k[0].float(), v[0].float()            # (Hk, S, D)
            s = k[:, 4:].sum(1)
            s_k[li] = s if s_k[li] is None else s_k[li] + s
            kv.append((k.cpu(), v.cpu()))
        cnt += x.shape[1] - 4
        keep.append(kv)
    mu = [s / cnt for s in s_k]
    out = {"model": a.model, "text": a.text, "prompts": a.prompts, "tokens": a.tokens, "headroom": HEADROOM, "layers": []}
    for li in range(n_layers):
        ak = akn = av = 0.0
        akr = avr = akrs = 0.0  # 11-14: raw post-RoPE K / V maxima (no rotation, no mean): what vLLM's FP8 cache
        # stores with scale 1.0; above 448 it saturates (the calibrated-FP8-scale question)
        aks = akns = avs = 0.0  # 11-13: the same without the first 4 tokens (when the sinks are kept in BF16, the
        # 4-bit scale should not be set by them; offline, formats take the scale from the non-sink tokens too)
        for kv in keep:
            k, v = kv[li][0].to(dev), kv[li][1].to(dev)
            kr, krn = rotate(k - mu[li][:, None], 128), rotate(k, 128)
            ak, akn, av = max(ak, block_amax(kr)), max(akn, block_amax(krn)), max(av, block_amax(v))
            aks, akns = max(aks, block_amax(kr[:, 4:])), max(akns, block_amax(krn[:, 4:]))
            avs = max(avs, block_amax(v[:, 4:]))
            akr, avr = max(akr, k.abs().amax().item()), max(avr, v.abs().amax().item())
            akrs = max(akrs, k[:, 4:].abs().amax().item())
        c = HEADROOM / (6 * 448)
        out["layers"].append({"mu_k": mu[li].cpu(), "ts_k": ak * c, "ts_k_nomean": akn * c, "ts_v": av * c,
                              "ts_k_s": aks * c, "ts_k_nomean_s": akns * c, "ts_v_s": avs * c,
                              "amax_k_raw": akr, "amax_v_raw": avr, "amax_k_raw_s": akrs})
        print(f"layer {li:2d}: |mu| {mu[li].norm(dim=-1).mean():8.3f}  amax K {ak:8.2f} (no mean {akn:8.2f})  "
              f"V {av:8.2f} | without the first 4 tokens: K {aks:8.2f} (no mean {akns:8.2f}) V {avs:8.2f} | raw K {akr:8.2f} (no sinks {akrs:8.2f}) V {avr:8.2f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)
    print("saved", a.out, "commit", os.environ.get("KV4_COMMIT", "unset"), flush=True)


if __name__ == "__main__":
    main()
