"""11-14: how well does the K bias work as a fixed post-RoPE K center, against the calibrated K mean? (Preparation for
the bias-center FP8 prototype after 11-18; rtp-llm #1583 uses the bias as a fixed center for FP8.)

The calibration file's mu_k is the post-RoPE, unrotated K channel mean per layer and KV head (positions >= 4). The
bias b is added before RoPE, so after RoPE each frequency pair of b is rotated by position * theta_i: averaged over
positions, high-frequency pairs of b average out, low-frequency pairs stay. So b should match mu in the low-frequency
half and over-subtract in the high-frequency half. Per layer (mean over KV heads): |mu|, |b|, cos(mu, b), the
fraction of |mu|^2 that subtracting b removes, 1 - |mu - b|^2 / |mu|^2 (1 = as good as the mean, < 0 = worse than no
center), the same for the low- and high-frequency halves (normalized by that half's own |mu|^2, so a half with
little energy can show large negative values) (HF rotate_half layout: pair i = dims i and i + D/2,
theta_i = base^(-2i/D), small i = high frequency), and the best scalar multiple of b.
11-14 (after the first run): also the RoPE-averaged bias, c = mean over positions p in [4, T) of RoPE(b, p), which
keeps the low-frequency pairs of b and shrinks the high-frequency ones toward 0. It needs only rope_theta, no
calibration (column "ropeavg").
11-14: column "rot" = mean over p of |RoPE(b, p) - c|^2 = |b|^2 - |c|^2 (RoPE preserves norms), over |mu|^2: the
position-dependent part of the bias that no static center can remove (a static center is free because q . mu is the
same for every key and drops out of the softmax; a per-position center is not).

Reads only weights (safetensors) and the calibration file: CPU, no forward pass.

    python bench/kbias_vs_mean.py --model Qwen/Qwen2.5-7B-Instruct --calib results/calib/Qwen2.5-7B-Instruct.pt
"""
import argparse
import json
import os

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


def frac(mu, c):
    return 1 - ((mu - c) ** 2).sum(-1) / (mu ** 2).sum(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--T", type=int, default=None, help="11-14: average RoPE(b, p) over p in [4, T) instead of the "
                   "calibration length (sensitivity of c to the context range; mu stays the calibration mean)")
    p.add_argument("--ref-calib", default=None, help="11-14: a second calibration file (e.g. another context length); "
                   "column 'vs ref' = fraction of |mu|^2 removed by subtracting the other file's mu_k")
    a = p.parse_args()
    root = snapshot_download(a.model, allow_patterns=["*.json"])
    idx = os.path.join(root, "model.safetensors.index.json")
    wmap = json.load(open(idx))["weight_map"] if os.path.exists(idx) else None
    cal = torch.load(a.calib, map_location="cpu")
    ref = torch.load(a.ref_calib, map_location="cpu") if a.ref_calib else None
    cfg = json.load(open(os.path.join(root, "config.json")))
    theta = cfg.get("rope_theta", 10000.0)
    T = a.T or cal.get("tokens", 2048)
    print(f"rope_theta {theta} rope_scaling {cfg.get('rope_scaling')}")
    print(f"model {a.model} calib {a.calib} ({cal.get('text')}, {cal.get('prompts')} x {cal.get('tokens')}), c averaged over [4, {T})")
    print(f"{'layer':>5s} {'|mu|':>8s} {'|b|':>8s} {'cos':>6s} {'removed':>8s} {'low-f':>7s} {'high-f':>7s} "
          f"{'best a':>7s} {'removed@a':>9s} {'ropeavg':>8s} {'vs ref':>7s} {'rot':>8s}")
    rows = []
    for li, lay in enumerate(cal["layers"]):
        key = f"model.layers.{li}.self_attn.k_proj.bias"
        f = wmap[key] if wmap else "model.safetensors"
        path = snapshot_download(a.model, allow_patterns=[f])
        with safe_open(os.path.join(path, f), "pt") as fh:
            b = fh.get_tensor(key).float()
        mu = lay["mu_k"].float()                           # (Hk, D)
        hk, d = mu.shape
        b = b.reshape(hk, d)
        h = d // 2
        lo = torch.zeros(d, dtype=torch.bool)
        lo[h // 2:h] = True
        lo[h + h // 2:] = True                             # the low-frequency half of each rotate_half pair
        i = torch.arange(h, dtype=torch.float64)
        th = theta ** (-2 * i / d)                          # rotate_half pair i: dims i and i + h
        pos = torch.arange(4, T, dtype=torch.float64)
        C, S = torch.cos(pos[:, None] * th).mean(0).float(), torch.sin(pos[:, None] * th).mean(0).float()
        b1, b2 = b[:, :h], b[:, h:]
        rb = torch.cat([b1 * C - b2 * S, b2 * C + b1 * S], -1)  # mean over positions of RoPE(b, p)
        cos = torch.nn.functional.cosine_similarity(mu, b, dim=-1)
        al = (mu * b).sum(-1) / (b * b).sum(-1)            # best scalar per head
        r = dict(layer=li, mu=mu.norm(dim=-1).mean().item(), b=b.norm(dim=-1).mean().item(), cos=cos.mean().item(),
                 removed=frac(mu, b).mean().item(), low=frac(mu[:, lo], b[:, lo]).mean().item(),
                 high=frac(mu[:, ~lo], b[:, ~lo]).mean().item(), alpha=al.mean().item(),
                 removed_a=frac(mu, al[:, None] * b).mean().item(),
                 ropeavg=frac(mu, rb).mean().item(),
                 rot=((b * b).sum(-1) - (rb * rb).sum(-1)).mean().item() / (mu * mu).sum(-1).mean().item(),
                 ref=frac(mu, ref["layers"][li]["mu_k"].float()).mean().item() if ref else float("nan"))
        rows.append(r)
        print(f"{li:5d} {r['mu']:8.2f} {r['b']:8.2f} {r['cos']:6.3f} {r['removed']:8.3f} {r['low']:7.3f} "
              f"{r['high']:7.3f} {r['alpha']:7.3f} {r['removed_a']:9.3f} {r['ropeavg']:8.3f} {r['ref']:7.3f} {r['rot']:8.5f}", flush=True)
    l0 = rows[0]
    print(f"RESULT {a.model} T={T}: layer 0 cos {l0['cos']:.3f}, removed {l0['removed']:.3f} (low-f {l0['low']:.3f}, "
          f"high-f {l0['high']:.3f}), RoPE-averaged {l0['ropeavg']:.3f}; last layer removed {rows[-1]['removed']:.3f}, "
          f"RoPE-averaged {rows[-1]['ropeavg']:.3f}; median over layers removed "
          f"{sorted(r['removed'] for r in rows)[len(rows) // 2]:.3f}, RoPE-averaged "
          f"{sorted(r['ropeavg'] for r in rows)[len(rows) // 2]:.3f}"
          + f"; rotating bias energy / |mu|^2 layer 0 {l0['rot']:.5f}, last {rows[-1]['rot']:.5f}"
          + (f"; vs ref calib layer 0 {l0['ref']:.3f}, last {rows[-1]['ref']:.3f}, median "
             f"{sorted(r['ref'] for r in rows)[len(rows) // 2]:.3f}" if ref else ""), flush=True)


if __name__ == "__main__":
    main()
