"""End-to-end greedy generation on Llama-3.1-8B with the real Hopper FP4 decode kernel (11-11), against the fake-quant
path the accuracy numbers come from (bench/ruler_lite.py `decode` mode).

How a decode step's attention is computed here, per layer (LlamaAttention.forward patched):
  - tokens [4, P) of the BF16-prefilled prompt: quantized once (kv4.formats, the same call as the fake-quant path) and
    read by the paged FP4 kernel (decode_fp4_paged, 64-token pages, length P - 4, K rotated: q rotated to match);
  - tokens [0, 4) (the attention sink) stay BF16, and the tokens generated so far are fake-quantized one by one exactly
    as the reference does (on the GPU they would sit in the same 4-bit pages; here they are joined in FP32 torch);
  - the kernel's partials (base-2 m, l carrying 2^10) and the extra tokens are merged by log-sum-exp.
The reference: the same prompt, prefill in BF16, cache fake-quantized with the first 4 tokens kept BF16, new tokens
fake-quantized as written, attention by SDPA.

    python bench/e2e_generate.py --config nvfp4_kh --lens 16384,32768 --trials 10 --out results/e2e_generate.csv
"""
import argparse
import csv
import math
import os
import random
import time
import zlib

import torch

from bench.accuracy_kl import CONFIGS, fake, make_cache_cls
from bench.ruler_lite import NEW, make_trial
from kv4.cuda import decode_fp4 as df
from kv4.formats import FORMATS, quantize, rotate

NS, HKV, HQ, D, GQA = 4, 8, 32, 128, 4
LOG2E = 1.4426950408889634
P_SHIFT = 10.0


class KernelAttention:
    """Per-layer state: kernel args for the quantized prompt, BF16 sink, fake-quantized generated tokens."""

    def __init__(self, keys, values, kf, vf, splits):
        P = keys.shape[2]
        self.kf, self.vf, self.splits = kf, vf, splits
        self.krot = FORMATS[kf].hadamard
        assert not FORMATS[vf].hadamard, "V rotation needs the output rotated back; not wired here"
        n = P - NS
        pad = (-n) % 256  # the packers work on 256-token blocks; the pages past the length are never read
        kb = torch.nn.functional.pad(keys[:, :, NS:].float(), (0, 0, 0, pad))
        vb = torch.nn.functional.pad(values[:, :, NS:].float(), (0, 0, 0, pad))
        # quantize exactly the tensor the reference quantizes (padding zeros change no block and no tensor-wide max)
        kq, vq = quantize(kb, kf), quantize(vb, vf)
        self.a = df.args_for(kq, vq, 1 / math.sqrt(D))
        self.pages, self.bt, self.sl = df.pack_pages(self.a[0], [n] * keys.shape[0], poison=False, shuffle=True)
        self.extra_k = [keys[:, :, :NS].float()]  # (B, HKV, t, D), unrotated basis
        self.extra_v = [values[:, :, :NS].float()]

    def __call__(self, q, k_new, v_new):
        """q (B, HQ, 1, D), k_new / v_new (B, HKV, 1, D) post-RoPE -> (B, HQ, D) fp32."""
        b = q.shape[0]
        self.extra_k.append(fake(k_new, self.kf, 0).float())
        self.extra_v.append(fake(v_new, self.vf, 0).float())
        qf = q[:, :, 0].float()                                    # (B, HQ, D)
        qk = rotate(qf, self.krot) if self.krot else qf             # the kernel's K is in the rotated basis
        e, a = df.ext(), self.a
        o, ml = e.decode_fp4_paged(self.pages, self.bt, self.sl, df.prep_q(qk, HKV), a[1], a[2], a[3], a[4],
                                   self.splits, a[5])                # (B, HKV, splits, 4, D), (B, HKV, splits, 4, 2)
        m, l = ml[..., 0], ml[..., 1]                               # base 2; l carries 2^P_SHIFT
        ek, ev = torch.cat(self.extra_k, 2), torch.cat(self.extra_v, 2)   # (B, HKV, t, D)
        qe = qf.half().float().reshape(b, HKV, GQA, D)              # the q rounding the kernel uses
        s2 = torch.einsum("bhgd,bhtd->bhgt", qe, ek) * (LOG2E / math.sqrt(D))
        mx = torch.maximum(m.amax(2), s2.amax(-1))                  # (B, HKV, 4)
        w = torch.exp2(m - mx[:, :, None]) * l                      # (B, HKV, splits, 4)
        we = torch.exp2(s2 - mx[..., None] + P_SHIFT)               # (B, HKV, 4, t)
        num = (o * w[..., None]).sum(2) + torch.einsum("bhgt,bhtd->bhgd", we, ev)
        den = w.sum(2) + we.sum(-1)
        return (num / den[..., None]).reshape(b, HQ, D)


def patch(model, state):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    def make_forward(attn):
        def forward(hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kw):
            shp = (*hidden_states.shape[:-1], -1, attn.head_dim)
            q = attn.q_proj(hidden_states).view(shp).transpose(1, 2)
            k = attn.k_proj(hidden_states).view(shp).transpose(1, 2)
            v = attn.v_proj(hidden_states).view(shp).transpose(1, 2)
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            out = state[attn.layer_idx](q, k, v)                     # (B, HQ, D)
            return attn.o_proj(out.to(hidden_states.dtype).reshape(*hidden_states.shape[:-1], -1)), None
        return forward

    originals = {}
    for layer in model.model.layers:
        originals[layer.self_attn] = layer.self_attn.forward
        layer.self_attn.forward = make_forward(layer.self_attn)
    return originals


def gen_kernel(model, x, kf, vf, new, splits):
    from transformers import DynamicCache
    P = x.shape[1]
    with torch.no_grad():
        cache = DynamicCache()
        out = model(x, use_cache=True, past_key_values=cache, logits_to_keep=1)
        state = {i: KernelAttention(l.keys, l.values, kf, vf, splits) for i, l in enumerate(cache.layers)}
        del cache
        originals = patch(model, state)
        try:
            gen, nxt = [], out.logits[:, -1].argmax(-1, keepdim=True)
            for t in range(new):
                gen.append(nxt.item())
                pos = torch.tensor([[P + t]], device=x.device)
                out = model(nxt, position_ids=pos, use_cache=False)
                nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        finally:
            for attn, f in originals.items():
                attn.forward = f
    return gen


def gen_fake(model, x, kf, vf, new, Cache):
    with torch.no_grad():
        cache = Cache(None, None, 0)
        out = model(x, use_cache=True, past_key_values=cache, logits_to_keep=1)
        for layer in cache.layers:   # sink kept BF16, then new tokens quantized as written (sink 0 for them)
            layer.keys, layer.values = fake(layer.keys, kf, NS), fake(layer.values, vf, NS)
        cache.kfmt, cache.vfmt = kf, vf
        gen, nxt = [], out.logits[:, -1].argmax(-1, keepdim=True)
        for _ in range(new):
            gen.append(nxt.item())
            out = model(nxt, use_cache=True, past_key_values=cache)
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    return gen


def main():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--config", default="nvfp4_kh")
    p.add_argument("--tasks", default="multikey,multivalue,vt")
    p.add_argument("--lens", default="16384,32768")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--splits", type=int, default=4)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    commit = os.environ.get("KV4_COMMIT", "unset")
    kf, vf = CONFIGS[args.config]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    wiki = tok("\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]),
               add_special_tokens=False).input_ids
    Cache = make_cache_cls()
    print(f"{args.model} e2e generate, real FP4 kernel vs fake quant, config {args.config} ({kf}, {vf}), sink {NS}, "
          f"commit {commit}, {torch.cuda.get_device_name()}", flush=True)
    rows, t0 = [], time.time()
    for task in args.tasks.split(","):
        for L in map(int, args.lens.split(",")):
            same_tok = same_seq = sk = sf = 0
            for i in range(args.trials):
                ids, refs = make_trial(task, tok, L, random.Random(zlib.crc32(f"{task}-{L}-{i}".encode())), wiki)
                x = torch.tensor([ids], device="cuda")
                gk = gen_kernel(model, x, kf, vf, NEW[task], args.splits)
                gf = gen_fake(model, x, kf, vf, NEW[task], Cache)
                tk, tf = tok.decode(gk), tok.decode(gf)
                score = lambda t: sum(r.lower() in t.lower() for r in refs) / len(refs)  # noqa: E731
                n_same = next((j for j, (a, c) in enumerate(zip(gk, gf)) if a != c), len(gk))
                same_tok += n_same / len(gk)
                same_seq += gk == gf
                sk += score(tk)
                sf += score(tf)
                rows.append(dict(task=task, ctx=L, trial=i, config=args.config, first_diff=n_same, n=len(gk),
                                 score_kernel=f"{score(tk):.2f}", score_fake=f"{score(tf):.2f}",
                                 out_kernel=tk.strip().replace("\n", " ")[:60],
                                 out_fake=tf.strip().replace("\n", " ")[:60], commit=commit))
            n = args.trials
            print(f"{task:10s} ctx {L:6d}: identical outputs {same_seq}/{n}, tokens before first difference "
                  f"{same_tok / n:.0%}, score kernel {sk / n:.1%} vs fake quant {sf / n:.1%} ({time.time() - t0:.0f} s)",
                  flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved", args.out)


if __name__ == "__main__":
    main()
