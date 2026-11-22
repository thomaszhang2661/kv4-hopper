"""11-14: how many rotary pairs does the per-position (rotating) K-bias correction need? Input for the rotating kernel
version (not implemented here; this script only measures).

A static center removes only the position average of RoPE(b, p). For rotate_half pair i (dims i and i + D/2,
theta_i = base^(-2i/D)) the energy left after the window's own mean over positions p in [4, T) is
    E_i(T) = |b_i|^2 (1 - |mean_p exp(i p theta_i)|^2)
(RoPE preserves norms). High-frequency pairs keep all of |b_i|^2, very low-frequency pairs (that barely turn within T)
keep ~0; the middle grows with T. Correcting a set P of pairs per position (subtract RoPE(b, p) on P at store, add
q . RoPE(b, p) on P back when scoring) costs 2 |P| extra multiply-adds per (query, key) and |P| cos / sin of p theta_i
per key (shared by the heads), against 128 for q . k; #59674 corrects all 64 pairs.

Per layer, pairs ranked by E_i(T) summed over KV heads (one pair set per layer, shared by its heads):
  rot       total rotating energy per token and KV head
  k90 / k99 pairs needed for 90 % / 99 % of it
  with --lst (accuracy_kl.py --layer-stats JSON at 2048 context): content = window-centered |K - mu|^2 per token and
  KV head minus the rotating energy at that JSON's ctx (the rotating part is deterministic, the rest is content);
  rot/content, and left/content after correcting the top 4 / 8 / 16 / 24 / 32 / 48 pairs. The content estimate comes from the 2048
  context and is reused at longer T (assumes content energy per token does not depend on length).
NVFP4's own error is ~0.009 of the centered K (11-14 layer stats), so "left/content" well below 1 means the 4-bit
scales see mostly content.

Weights only (k_proj biases from safetensors); CPU, no forward pass. Default RoPE only (no rope_scaling).

    python bench/kbias_pair_budget.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --T 2048 8192 32768 \
        [--lst results/lst_r1q7b.json]
"""
import argparse
import json
import os

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

KS = (4, 8, 16, 24, 32, 48)  # 11-14: 24 / 32 / 48 added after the first run (16 pairs left 2.6x the content in Math-7B layer 0)


def kneed(e, share):
    c = torch.cumsum(e.sort(descending=True).values, 0)
    return int((c < share * c[-1]).sum().item()) + 1 if c[-1] > 0 else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--T", type=int, nargs="+", default=[2048, 8192, 32768])
    p.add_argument("--lst", default=None)
    a = p.parse_args()
    root = snapshot_download(a.model, allow_patterns=["*.json"])
    cfg = json.load(open(os.path.join(root, "config.json")))
    theta, L = cfg.get("rope_theta", 10000.0), cfg["num_hidden_layers"]
    hk = cfg["num_key_value_heads"]
    d = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    h = d // 2
    idx = os.path.join(root, "model.safetensors.index.json")
    wmap = json.load(open(idx))["weight_map"] if os.path.exists(idx) else None
    lst, lctx = None, None
    if a.lst:
        lst = json.load(open(a.lst))
        meta = lst.pop("_meta")
        lctx, ntok = meta["ctx"], meta["windows"] * (meta["ctx"] - meta["sink"])
        lst = lst["nvfp4_kh"]
    print(f"{a.model}: rope_theta {theta}, rope_scaling {cfg.get('rope_scaling')}, {L} layers, {hk} KV heads, D {d}"
          + (f"; content from {a.lst} (ctx {lctx})" if lst else ""))
    i = torch.arange(h, dtype=torch.float64)
    th = theta ** (-2 * i / d)

    def rot_pairs(b2, T):  # (Hk, h) pair energies -> (h,) rotating energy per pair, summed over heads
        pos = torch.arange(4, T, dtype=torch.float64)
        m2 = torch.cos(pos[:, None] * th).mean(0) ** 2 + torch.sin(pos[:, None] * th).mean(0) ** 2
        return (b2 * (1 - m2)).sum(0)

    out = {}
    for li in range(L):
        key = f"model.layers.{li}.self_attn.k_proj.bias"
        f = wmap[key] if wmap else "model.safetensors"
        path = snapshot_download(a.model, allow_patterns=[f])
        with safe_open(os.path.join(path, f), "pt") as fh:
            b = fh.get_tensor(key).double().reshape(hk, d)
        b2 = b[:, :h] ** 2 + b[:, h:] ** 2
        content = None
        if lst:
            content = lst[str(li)]["sig_centered"] / (ntok * hk) - rot_pairs(b2, lctx).sum().item() / hk
        for T in a.T:
            e = rot_pairs(b2, T)
            tot = e.sum().item() / hk
            srt = e.sort(descending=True)
            r = dict(rot=tot, k90=kneed(e, 0.9), k99=kneed(e, 0.99), top=srt.indices[:8].tolist())
            if content is not None:
                r["content"] = content
                r["rot_c"] = tot / content
                for k in KS:
                    r[f"left{k}_c"] = (tot - srt.values[:k].sum().item() / hk) / content
            out.setdefault(T, []).append(r)
    for T in a.T:
        rows = out[T]
        print(f"\n== T = {T}")
        hd = f"{'L':>3s} {'rot':>9s} {'k90':>4s} {'k99':>4s}" + (
            f" {'content':>9s} {'rot/c':>7s}" + "".join(f" {'left' + str(k) + '/c':>9s}" for k in KS) if lst else "")
        print(hd + "  top pairs (0 = highest frequency)")
        for li, r in enumerate(rows):
            s = f"{li:3d} {r['rot']:9.2f} {r['k90']:4d} {r['k99']:4d}"
            if lst:
                s += f" {r['content']:9.2f} {r['rot_c']:7.3f}" + "".join(f" {r[f'left{k}_c']:9.4f}" for k in KS)
            print(s + "  " + ",".join(map(str, r["top"])))
        k90 = sorted(r["k90"] for r in rows)
        msg = (f"RESULT {a.model} T={T}: k90 layer 0 {rows[0]['k90']}, median {k90[len(k90) // 2]}, max {k90[-1]}; "
               f"k99 layer 0 {rows[0]['k99']}, max {max(r['k99'] for r in rows)}")
        if lst:
            for k in KS:
                v = [r[f"left{k}_c"] for r in rows]
                msg += f"; left/content top{k}: layer 0 {v[0]:.4f}, max {max(v):.4f} (layer {v.index(max(v))})"
            msg += f"; rot/content layer 0 {rows[0]['rot_c']:.3f}, max {max(r['rot_c'] for r in rows):.3f}"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
