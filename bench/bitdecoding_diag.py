"""Why is BitDecoding's error 5x ours and unlike per-token fake quantization? (11-10)

Separates the possible causes on small shapes:
  * GQA or not: H_q = H_kv (its own test) against H_q = 4 H_kv (ours);
  * head mapping: query head h -> kv head h // g (standard) or h % H_kv;
  * V alone: K = 0 makes attention uniform, so the output is the mean of the dequantized V;
  * rounding of the codes: nearest or floor.
Prints output and error magnitudes for each case.
"""
import math

import torch

from baselines.bitdecoding.build import load
from bench.bench_bitdecoding import bd_inputs

ext = load()
D = 128
sm = 1 / math.sqrt(D)


def fq(x, mode):
    x = x.float()
    lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    scale = (hi - lo).clamp_min(1e-8) / 15
    t = (x - lo) / scale
    t = torch.round(t) if mode == "round" else torch.floor(t)
    return t.clamp(0, 15) * scale + lo


def attn(q, k, v, mapping):
    """q (B, Hq, D), k/v (B, S, Hkv, D)."""
    hq, hkv = q.shape[1], k.shape[2]
    idx = torch.arange(hq, device=q.device)
    idx = idx // (hq // hkv) if mapping == "div" else idx % hkv
    kk, vv = k.float()[:, :, idx].permute(0, 2, 1, 3), v.float()[:, :, idx].permute(0, 2, 1, 3)
    p = torch.softmax(torch.einsum("bhd,bhsd->bhs", q.float(), kk) * sm, -1)
    return torch.einsum("bhs,bhsd->bhd", p, vv)


def run(b, s, hq, hkv, k_zero=False, v_shift=0.0):
    torch.manual_seed(0)
    q = torch.randn(b, 1, hq, D, device="cuda", dtype=torch.float16)
    k = torch.randn(b, s, hkv, D, device="cuda", dtype=torch.float16)
    if k_zero:
        k.zero_()
    v = torch.randn_like(k) + v_shift
    kp, kpar, vp, vpar, cu = bd_inputs(k, v)
    ext.pack(k.reshape(b * s, hkv, D), kp, kpar, v.reshape(b * s, hkv, D), vp, vpar, cu, s)
    out = ext.fwd(q, k, kp, kpar, v, vp, vpar, sm, 0).float()[:, 0]
    tag = f"B={b} S={s} Hq={hq} Hkv={hkv}{' K=0' if k_zero else ''}{f' V+{v_shift}' if v_shift else ''}"
    print(f"{tag}: |out| mean {out.abs().mean():.3e}, nan {torch.isnan(out).any().item()}", flush=True)
    for mapping in (["div", "mod"] if hq != hkv else ["div"]):
        for name, kk, vv in [("orig", k, v), ("fq-round", fq(k, "round"), fq(v, "round")),
                             ("fq-floor", fq(k, "floor"), fq(v, "floor"))]:
            ref = attn(q[:, 0], kk, vv, mapping)
            e = (out - ref).abs()
            print(f"   vs {name:8s} map={mapping}: |ref| mean {ref.abs().mean():.3e}, err mean {e.mean():.3e} max {e.max():.3e}",
                  flush=True)
    # per-head error against the standard reference (which heads are off?)
    ref = attn(q[:, 0], fq(k, "round"), fq(v, "round"), "div")
    ph = (out - ref).abs().mean(dim=(0, 2))
    print("   per-head err (fq-round, div):", " ".join(f"{x:.1e}" for x in ph.tolist()), flush=True)


print(torch.cuda.get_device_name(), flush=True)
run(1, 1024, 32, 32)            # its own test's shape
run(1, 4096, 32, 32)
run(2, 4096, 32, 8)             # ours
run(2, 4096, 32, 8, k_zero=True)
run(2, 4096, 32, 8, k_zero=True, v_shift=3.0)
run(1, 4096, 8, 8, k_zero=True)
