"""Accuracy of each KV-cache format on Llama (11-11): KL divergence of next-token distributions from BF16, and
perplexity, on wikitext-2 test (runs in an env with transformers, e.g. the fp4 conda env).

How the cache is quantized: a DynamicCache subclass quantizes and dequantizes K and V in `update()`, i.e. after RoPE,
exactly what the cache stores; attention then uses those values for every query. The values come from
kv4.formats.quantize / dequantize, the same functions that produce the inputs of our CUDA kernels (the kernels match
an FP32 reference over these values within 2.5e-4 on random data and 1.7e-4 on real Llama KV, bench/real_kv_check.py),
so the formats measured here are the formats the kernels decode. Dequantized values are cast to BF16 for the model's
attention (relative rounding 2^-9, far below the 4-bit quantization error).

Each window is one teacher-forced forward pass of `--ctx` tokens (prefill), so every position attends to a quantized
cache of its whole prefix. Tensor-wide scales (fp8, nvfp4) are taken per layer over the window, i.e. dynamic; real
serving would fix them (vLLM's FP8 default is scale 1.0), which can only be worse for fp8.

    python bench/accuracy_kl.py --model meta-llama/Llama-3.1-8B --ctx 4096 --windows 16 --out results/acc_kl.csv
"""
import argparse
import csv
import math
import os
import re
import time

import torch

from kv4.formats import dequantize, quantize

# name -> (K format, V format); "_kh" = Hadamard rotation on K only (SAW-INT4; also vLLM UltraQuant for MXFP4)
CONFIGS = {
    "bf16": (None, None),
    "fp8": ("fp8", "fp8"),
    "int4a": ("int4a_tok", "int4a_tok"),
    "int4a_kh": ("int4a_tok_h", "int4a_tok"),
    "nvfp4": ("nvfp4", "nvfp4"),
    "nvfp4_kh": ("nvfp4_h", "nvfp4"),
    "mxfp4": ("mxfp4", "mxfp4"),
    "mxfp4_kh": ("mxfp4_h", "mxfp4"),
    # 11-11 follow-ups: UltraQuant's scale rule (mxfp4_uq_kh is exactly vLLM UltraQuant's format), and V rotated too
    # ("_kvh"; in a kernel the output is rotated back, or the rotation is folded into W_o, at no speed cost)
    "mxfp4_uq": ("mxfp4_uq", "mxfp4_uq"),
    "mxfp4_uq_kh": ("mxfp4_uq_h", "mxfp4_uq"),
    "int4a_kvh": ("int4a_tok_h", "int4a_tok_h"),
    "nvfp4_kvh": ("nvfp4_h", "nvfp4_h"),
    "mxfp4_kvh": ("mxfp4_h", "mxfp4_h"),
    "mxfp4_uq_kvh": ("mxfp4_uq_h", "mxfp4_uq_h"),
    # 11-11 evening: which side loses with UltraQuant's rule, and a sweep of its constant c (OCP = 0.177)
    "mx_uqK_ocpV_kh": ("mxfp4_uq_h", "mxfp4"),
    "mx_ocpK_uqV_kh": ("mxfp4_h", "mxfp4_uq"),
}
from kv4.formats import MX_C_SWEEP  # noqa: E402

# 11-11 night: Qwen3 (rotation hurts block-scaled FP4 10x there): unrotated MX with c = 0.2, and K smoothing (~s)
CONFIGS.update({
    # 11-18: plain baselines (one scale per tensor; NVFP4 with vLLM's default global scale 1.0)
    "fp4t": ("fp4_tensor", "fp4_tensor"), "int4t": ("int4_tensor", "int4_tensor"), "nvfp4_g1": ("nvfp4_g1", "nvfp4_g1"),
    "mxc0.200": ("mxfp4_c0.200", "mxfp4_c0.200"),
    "nvfp4_ks": ("nvfp4~s", "nvfp4"), "nvfp4_khs": ("nvfp4_h~s", "nvfp4"),
    "int4a_ks": ("int4a_tok~s", "int4a_tok"), "int4a_khs": ("int4a_tok_h~s", "int4a_tok"),
    "mxc0.200_ks": ("mxfp4_c0.200~s", "mxfp4_c0.200"), "mxc0.200_khs": ("mxfp4_c0.200_h~s", "mxfp4_c0.200"),
    "mxc0.156_khs": ("mxfp4_c0.156_h~s", "mxfp4_c0.156"), "mxc0.177_khs": ("mxfp4_c0.177_h~s", "mxfp4_c0.177"),
    "nvfp4_khp": ("nvfp4_h~p", "nvfp4"), "int4a_khp": ("int4a_tok_h~p", "int4a_tok"),
    "mxc0.200_khp": ("mxfp4_c0.200_h~p", "mxfp4_c0.200"),
    # 11-12 night: FP8 K with mean removal (R1-Distill-Qwen-7B collapses with plain FP8 K; its K has large channel means)
    "fp8_ks": ("fp8~s", "fp8"),
    # 11-14: per-position bias centering (`~b`, see fake()); models with a k_proj bias only
    "nvfp4_khb": ("nvfp4_h~b", "nvfp4"), "fp8_kb": ("fp8~b", "fp8"),
    # 11-14 (afternoon): the rotating version's cost / precision. `nvfp4_khbN`: correct only the top N rotary pairs per
    # layer (ranked by their rotating energy over the window, as bench/kbias_pair_budget.py; the other pairs keep only
    # the window mean); `nvfp4_khbr`: all pairs, but the add-back RoPE(b, p) rounded to BF16 (a stand-in for a BF16
    # add-back in the kernel; the subtraction at store stays fp32)
    "nvfp4_khb8": ("nvfp4_h~b", "nvfp4"), "nvfp4_khb16": ("nvfp4_h~b", "nvfp4"), "nvfp4_khb32": ("nvfp4_h~b", "nvfp4"),
    "nvfp4_khb48": ("nvfp4_h~b", "nvfp4"), "nvfp4_khbr": ("nvfp4_h~B", "nvfp4"),
    # 11-21 (XQA borrow, C1): K with one scale per 32 dims (what an FP8 wgmma k-step needs), V unchanged (NVFP4 g16)
    "nvfp4g32K_khs": ("nvfp4_g32_h~s", "nvfp4"), "mxc0.200K_khs": ("mxfp4_c0.200_h~s", "nvfp4"),
    # 11-21 (C2): V with its NVFP4 scales along tokens (one E4M3 scale per 16 / 32 tokens of one channel), what P V on
    # FP8 wgmma needs (the MMA reduces over tokens); "nvfp4g32K_khs_vt32" = C1's K g32 too, i.e. K and V both on FP8
    "nvfp4_khs_vt16": ("nvfp4_h~s", "nvfp4_t16"), "nvfp4_khs_vt32": ("nvfp4_h~s", "nvfp4_t32"),
    "nvfp4g32K_khs_vt32": ("nvfp4_g32_h~s", "nvfp4_t32"),
    # V with one E4M3 scale per token row (folds into P)
    "nvfp4_khs_vrow": ("nvfp4_h~s", "nvfp4_row"), "nvfp4g32K_khs_vrow": ("nvfp4_g32_h~s", "nvfp4_row"),
})

for _c in MX_C_SWEEP:
    CONFIGS[f"mxc{_c:.3f}_kh"] = (f"mxfp4_c{_c:.3f}_h", f"mxfp4_c{_c:.3f}")
    CONFIGS[f"mxc{_c:.3f}_kvh"] = (f"mxfp4_c{_c:.3f}_h", f"mxfp4_c{_c:.3f}_h")


PREFIX = 512


def cfg_base(n):
    """config name without the layer selectors: `@` skip layers, `#` only these layers, `^` (11-14) the `~b` K format on
    these layers only (the others use `~s`, e.g. nvfp4_khb^0 = per-position bias centering on layer 0 only)"""
    return n.partition("@")[0].partition("#")[0].partition("^")[0]


def fake(x, fmt, sink, mu=None, pb=None):
    """(B, H, S, D) bf16 -> the values a cache in `fmt` holds, bf16; the first `sink` tokens stay BF16.
    `fmt~p`: the same, but the mean comes from the first PREFIX real tokens of the call only (a stand-in for a
    mean taken from the prompt at prefill).
    `fmt~s` (11-11, K smoothing as in SageAttention): subtract the per-channel mean over the tokens of this call before
    quantizing and add it back after. q . mean is the same for every token, so attention is unchanged and only the
    quantization error changes. Over a whole window the mean includes later tokens; a deployable version takes it
    from the prefill and keeps it per (layer, head) in BF16.
    `fmt~b` (11-14, per-position bias centering): subtract RoPE(b, p) of the k_proj bias b at every position p
    (`pb`, (1, Hk, S, D)), then the remaining mean over the call as in `~s`, quantize, add both back. A static center
    can only remove the positional average of RoPE(b, p); with rope_theta 1e4 the rest rotates within the context and
    stays in the quantized K (11-14 early morning). In a kernel the add-back is q . RoPE(b, p), one extra dot product per key
    (it is not the same for all keys, so unlike the mean it does not cancel in the softmax). Prefill only: the call
    covers positions 0..S-1."""
    if fmt is None:
        return x
    if fmt is not None and fmt.endswith("~B"):  # 11-14: `~b` with the add-back rounded to BF16 (nvfp4_khbr)
        y = fake(x, fmt[:-2] + "~b", sink, mu, pb)
        if pb is not None:
            p = pb[:, :, sink:x.shape[2]].to(x.device, torch.float32)
            y[:, :, sink:] = (y[:, :, sink:].float() - p + p.to(torch.bfloat16).float()).to(x.dtype)
        return y
    smooth = fmt[-2:] in ("~s", "~p", "~b")
    prefix = fmt.endswith("~p")
    pb_s = 0.0
    y = x.clone()
    xs = x[:, :, sink:].float()
    if fmt.endswith("~b"):
        assert pb is not None and mu is None, "~b needs the RoPE'd k_proj bias and a prefill call"
        assert pb.shape[2] >= x.shape[2], "~b: precomputed positions shorter than the call"
        pb_s = pb[:, :, sink:x.shape[2]].to(xs.device, xs.dtype)
        xs = xs - pb_s
    fmt = fmt[:-2] if smooth else fmt
    if smooth and mu is not None:  # a stored mean (decode mode: taken from the prompt at prefill)
        y[:, :, sink:] = (dequantize(quantize(xs - mu, fmt)) + mu).to(x.dtype)
        return y
    mu = 0.0
    if smooth:
        # a mean over a handful of tokens would quantize them almost losslessly (decode mode calls this per token)
        assert xs.shape[2] >= 64, "smoothing needs a stored mean in decode mode; not implemented"
    if smooth:  # `~p`: the mean of the first PREFIX tokens only, as a deployment would take it from the prefill
        mu = (xs[:, :, :PREFIX] if prefix else xs).mean(2, keepdim=True)
    y[:, :, sink:] = (dequantize(quantize(xs - mu, fmt)) + mu + pb_s).to(x.dtype)
    return y


def rope_bias(model, S):
    """11-14: per layer, RoPE(b, p) of the k_proj bias for p in [0, S), (1, Hk, S, D) fp32 (HF rotate_half layout;
    default RoPE only). {} if the model has no k_proj bias."""
    cfg = model.config
    theta = getattr(cfg, "rope_theta", None) or (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta")
    sc = getattr(cfg, "rope_scaling", None)
    assert theta and (not sc or sc.get("rope_type", sc.get("type")) == "default"), f"unsupported RoPE {theta} {sc}"
    d = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    h = d // 2
    ang = torch.arange(S, dtype=torch.float64)[:, None] * float(theta) ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    cos, sin = ang.cos(), ang.sin()                                    # (S, h)
    out = {}
    for li, layer in enumerate(model.model.layers):
        bias = getattr(layer.self_attn.k_proj, "bias", None)
        if bias is None:
            return {}
        b = bias.detach().double().cpu().reshape(-1, 1, d)             # (Hk, 1, D)
        b1, b2 = b[..., :h], b[..., h:]
        out[li] = torch.cat([b1 * cos - b2 * sin, b2 * cos + b1 * sin], -1).float().unsqueeze(0)
    print(f"rope_bias: theta {theta}, {len(out)} layers, head_dim {d}", flush=True)
    return out


def pb_topk(PB, k, sink):
    """11-14: RoPE(b, p) restricted to the top k rotary pairs of each layer (rotate_half: dims i and i + D/2), ranked by
    their rotating energy over positions [sink, S) summed over KV heads; the other pairs are zeroed (they keep only the
    window mean, like `~s`)."""
    out = {}
    for li, pb in PB.items():
        x = pb[0, :, sink:]                                           # (Hk, S', D)
        h = x.shape[-1] // 2
        e = (x - x.mean(1, keepdim=True)).pow(2).sum((0, 1))         # (D,)
        top = (e[:h] + e[h:]).topk(k).indices
        m = torch.zeros(h, dtype=pb.dtype)
        m[top] = 1
        out[li] = pb * torch.cat([m, m])
    return out


def make_cache_cls():
    from transformers import DynamicCache

    class FakeQuantCache(DynamicCache):
        def __init__(self, kfmt, vfmt, sink, **kw):
            super().__init__(**kw)
            self.kfmt, self.vfmt, self.sink = kfmt, vfmt, sink
            self.bonly = None  # 11-14: layers that use `~b` (None = all); see cfg_base()
            self.pb = {}  # 11-14: layer -> RoPE(b, p) for `~b` formats
            self.kmu = {}  # layer -> (B, H, 1, D) stored K mean for `~s` formats in decode mode
            self.skip = set()  # 11-12: layers kept unquantized (vLLM's --kv-cache-dtype-skip-layers)
            # 11-12 night (interpretive experiment A): squared quantization error and signal of the quantized
            # layers' K and V, summed (relative error energy = err / sig)
            self.err = dict(k=0.0, v=0.0, ks=0.0, vs=0.0)
            # 11-14: per layer, over the non-sink tokens: K error energy, raw K energy, and the energy of K minus its
            # mean over the call (what attention uses; the mean cancels in the softmax) -> --layer-stats
            self.lst = {}

        def update(self, key_states, value_states, layer_idx, *args, **kwargs):
            if layer_idx in self.skip:
                return super().update(key_states, value_states, layer_idx, *args, **kwargs)
            kf = self.kfmt
            if kf and kf[-2:] in ("~b", "~B") and self.bonly is not None and layer_idx not in self.bonly:
                kq = fake(key_states, kf[:-2] + "~s", self.sink, self.kmu.get(layer_idx))
            else:
                kq = fake(key_states, kf, self.sink, self.kmu.get(layer_idx), self.pb.get(layer_idx))
            vq = fake(value_states, self.vfmt, self.sink)
            if self.kfmt is not None:
                kd, vd = key_states.float(), value_states.float()
                self.err["k"] += (kq.float() - kd).pow(2).sum().item()
                self.err["v"] += (vq.float() - vd).pow(2).sum().item()
                self.err["ks"] += kd.pow(2).sum().item()
                self.err["vs"] += vd.pow(2).sum().item()
                ks, kqs = kd[:, :, self.sink:], kq.float()[:, :, self.sink:]
                L = self.lst.setdefault(layer_idx, [0.0, 0.0, 0.0])
                L[0] += (kqs - ks).pow(2).sum().item()
                L[1] += ks.pow(2).sum().item()
                L[2] += (ks - ks.mean(2, keepdim=True)).pow(2).sum().item()
            return super().update(kq, vq, layer_idx, *args, **kwargs)

    return FakeQuantCache


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--windows", type=int, default=16)
    p.add_argument("--configs", default=",".join(CONFIGS))
    p.add_argument("--sink", type=int, default=0, help="keep the first N tokens of each window in BF16")
    p.add_argument("--chunk", type=int, default=1024, help="tokens per KL chunk (fp32 log-softmax scratch)")
    p.add_argument("--out", required=True)
    # 11-12 night (interpretive experiment C): fp32 removes the BF16 KL floor F (the quantized cache is still made from
    # the fp32 K / V, so only the format's error is measured); also for GPUs without BF16 (V100)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    # 11-14: --device cpu for FP32 runs of 7-8B models on the CPU short partition (interpretive experiments)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    # 11-14: `~s` configs use the static K mean of a calibration file (bench/calib_kv.py, mu_k per layer) instead of
    # the window's own mean, as the vLLM plugin does; tests a calibration length that differs from the context
    p.add_argument("--kmu-calib", default=None)
    # 11-14: write per-layer K error energy / raw K energy / centered K energy (summed over windows) to this JSON
    p.add_argument("--layer-stats", default=None)
    args = p.parse_args()
    commit = os.environ.get("KV4_COMMIT", "unset")
    dt = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    if args.model.startswith("QwQZh/gated_attention/"):
        # 11-14 (experiment B): the gated-attention 1B checkpoints, rebuilt on stock Qwen3 by ../quant's fp4diag.gated
        # (weights_only load, no remote code); FP4DIAG_SRC points at that repo's src (on kk ~/fp4-activation-quant/src)
        import sys
        sys.path.insert(0, os.environ.get("FP4DIAG_SRC", os.path.expanduser("~/fp4-activation-quant/src")))
        from fp4diag.gated import load_gated
        model, tok = load_gated(args.model, dt, args.device)
    else:
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt, device_map=args.device)
    model.eval()
    from datasets import load_dataset
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    nwin = min(args.windows, ids.numel() // args.ctx)
    print(f"{args.model}: {ids.numel()} wikitext-2 test tokens, {nwin} windows of {args.ctx}, sink {args.sink}, "
          f"commit {commit}, {torch.cuda.get_device_name() if args.device == 'cuda' else 'cpu'}", flush=True)
    Cache = make_cache_cls()
    names = args.configs.split(",")
    PB = rope_bias(model, args.ctx) if any(CONFIGS[cfg_base(n)][0] and
                                          CONFIGS[cfg_base(n)][0][-2:] in ("~b", "~B")
                                          for n in names) else {}
    PBK = {}  # 11-14: nvfp4_khbN -> RoPE(b, p) on the top N pairs only
    for n in names:
        m = re.fullmatch(r"nvfp4_khb(\d+)", cfg_base(n))
        if m and PB:
            PBK[n] = pb_topk(PB, int(m.group(1)), args.sink)
    KMU = {}
    if args.kmu_calib:
        cal = torch.load(args.kmu_calib, map_location="cpu")
        KMU = {li: L["mu_k"].float()[None, :, None, :].to(args.device) for li, L in enumerate(cal["layers"])}
        print(f"static K mean from {args.kmu_calib} ({cal.get('prompts')} x {cal.get('tokens')} tokens, "
              f"{cal.get('text')})", flush=True)
    LST = {}
    acc = {n: dict(kl=0.0, nll=0.0, top1=0, kl_max=0.0, n=0, ek=0.0, ev=0.0, sk=0.0, sv=0.0) for n in names}
    t0 = time.time()
    with torch.no_grad():
        for w in range(nwin):
            x = ids[w * args.ctx:(w + 1) * args.ctx].unsqueeze(0).to(args.device)
            tgt = x[0, 1:]
            base = model(x, use_cache=True, past_key_values=Cache(None, None, 0)).logits[0, :-1]  # bf16 (S-1, V)
            for n in names:
                kf, vf = CONFIGS[cfg_base(n)]
                c = Cache(kf, vf, args.sink)
                c.pb = PBK.get(n, PB)
                if kf and kf.endswith("~s"):
                    c.kmu = KMU
                # 11-12: `name@0+27` keeps those layers unquantized (vLLM's --kv-cache-dtype-skip-layers);
                # `name#5` quantizes layer 5 only (per-layer sensitivity)
                c.skip = {int(i) for i in n.partition("@")[2].partition("^")[0].partition("#")[0].split("+") if i}
                if "^" in n:
                    c.bonly = {int(i) for i in n.partition("^")[2].partition("@")[0].partition("#")[0].split("+") if i}
                if "#" in n:
                    only = {int(i) for i in n.partition("#")[2].split("+") if i}
                    c.skip = set(range(model.config.num_hidden_layers)) - only
                lg = base if kf is None else model(x, use_cache=True, past_key_values=c).logits[0, :-1]
                a = acc[n]
                for li, v in c.lst.items():
                    t3 = LST.setdefault(n, {}).setdefault(li, [0.0, 0.0, 0.0])
                    for j in range(3):
                        t3[j] += v[j]
                a["ek"] += c.err["k"]; a["ev"] += c.err["v"]; a["sk"] += c.err["ks"]; a["sv"] += c.err["vs"]  # noqa: E702
                for c in range(0, lg.shape[0], args.chunk):
                    lb = torch.log_softmax(base[c:c + args.chunk].float(), -1)
                    lq = torch.log_softmax(lg[c:c + args.chunk].float(), -1)
                    kl = (lb.exp() * (lb - lq)).sum(-1)                       # KL(P_bf16 || P_fmt) per token
                    a["kl"] += kl.sum().item()
                    a["kl_max"] = max(a["kl_max"], kl.max().item())
                    t = tgt[c:c + args.chunk]
                    a["nll"] -= lq.gather(-1, t[:, None]).sum().item()
                    a["top1"] += (lq.argmax(-1) == lb.argmax(-1)).sum().item()
                    a["n"] += t.numel()
                del lg
            print(f"window {w + 1}/{nwin} done ({time.time() - t0:.0f} s)", flush=True)
    rows = []
    b_ppl = math.exp(acc["bf16"]["nll"] / acc["bf16"]["n"]) if "bf16" in acc else float("nan")
    print(f"\n{'config':10s} {'K fmt':12s} {'V fmt':10s} {'mean KL':>10s} {'max KL':>8s} {'PPL':>8s} {'dPPL':>8s} "
          f"{'top-1 agree':>11s}")
    for n in names:
        a = acc[n]
        ppl = math.exp(a["nll"] / a["n"])
        kf, vf = CONFIGS[cfg_base(n)]
        r = dict(model=args.model, ctx=args.ctx, windows=nwin, sink=args.sink, config=n, k_fmt=kf or "bf16",
                 v_fmt=vf or "bf16", mean_kl=f"{a['kl'] / a['n']:.3e}", max_kl=f"{a['kl_max']:.3f}",
                 ppl=f"{ppl:.4f}", dppl=f"{ppl - b_ppl:+.4f}", top1_agree=f"{a['top1'] / a['n']:.4f}",
                 dtype=args.dtype, rel_err_k=f"{a['ek'] / a['sk']:.4e}" if a["sk"] else "",
                 rel_err_v=f"{a['ev'] / a['sv']:.4e}" if a["sv"] else "", commit=commit)
        rows.append(r)
        print(f"{n:10s} {r['k_fmt']:12s} {r['v_fmt']:10s} {r['mean_kl']:>10s} {r['max_kl']:>8s} {r['ppl']:>8s} "
              f"{r['dppl']:>8s} {r['top1_agree']:>11s}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("saved", args.out)
    if args.layer_stats:
        import json
        with open(args.layer_stats, "w") as f:
            json.dump({n: {str(li): dict(err=v[0], sig=v[1], sig_centered=v[2]) for li, v in sorted(d.items())}
                       for n, d in LST.items()} | {"_meta": dict(model=args.model, ctx=args.ctx, windows=nwin,
                                                                 sink=args.sink, dtype=args.dtype, commit=commit)},
                      f, indent=1)
        print("saved", args.layer_stats)


if __name__ == "__main__":
    main()
