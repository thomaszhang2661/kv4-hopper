"""K mean removal in the real kernel path (11-11 night): store k - mu in the 4-bit cache, keep the sink keys as
sink_k - mu in BF16, and run the unchanged decode kernel + merge_sink. Since q.mu is the same for every token, softmax
gives the same attention as with the raw keys, so the output should match exact attention over the raw BF16 cache
better than rotation alone, at zero read cost (the kernel and its inputs have the same shapes and types).

Data: Qwen3-like keys, a per-channel offset shared by all tokens with one huge channel, plus a strong sink at token 0.
Reference: FP32 attention over the unquantized BF16 K/V. The mean comes from the first 512 tokens (as from a prefill).

    python bench/smooth_check.py
"""
import math
import os

import torch

from kv4.cuda import decode_fp4 as df
from kv4.cuda import merge as mg
from kv4.formats import FORMATS, quantize, rotate

HQ, HKV, D, GQA, NS = 32, 8, 128, 4, 4
SM = 1 / math.sqrt(D)


def make(b, s, big, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(b, HQ, D, device="cuda")
    mu = torch.randn(1, HKV, 1, D, device="cuda") * 2.0
    mu[..., 5] = big                                       # one outlier channel, the same for every token
    k = (mu + torch.randn(b, HKV, NS + s, D, device="cuda")).to(torch.bfloat16)
    v = torch.randn_like(k)
    qg = q.reshape(b, HKV, GQA, D)
    u = torch.nn.functional.normalize(qg.sum(2), dim=-1)
    proj = (qg * u[:, :, None]).sum(-1).mean(-1, keepdim=True).clamp(min=1.0)
    k[:, :, 0] = (u * 8.0 / (SM * proj) + mu[:, :, 0]).to(k.dtype)    # strong sink on top of the offset
    return q, k, v


def exact(q, k, v):
    qe = q.reshape(q.shape[0], HKV, GQA, D).half().float()
    p = torch.softmax(torch.einsum("bhgd,bhsd->bhgs", qe, k.float()) * SM, -1)
    return torch.einsum("bhgs,bhsd->bhgd", p, v.float()).reshape(q.shape[0], HQ, D)


def run(q, k, v, kf, smooth):
    b = q.shape[0]
    rest = k[:, :, NS:].float()
    mu = rest[:, :, :512].mean(2, keepdim=True) if smooth else torch.zeros_like(rest[:, :, :1])
    kq, vq = quantize(rest - mu, kf), quantize(v[:, :, NS:].float(), "nvfp4")
    a = df.args_for(kq, vq, SM)
    hb = FORMATS[kf].hadamard
    qk = rotate(q, hb) if hb else q                       # the cache's K is in the rotated basis
    o, ml = df.ext().decode_fp4(*a[0], df.prep_q(qk, HKV), a[1], a[2], a[3], a[4], 1, 2, a[5])
    qe = q.reshape(b, HKV, GQA, D).half().float().reshape(b, HQ, D)
    sk = (k[:, :, :NS].float() - mu).to(torch.bfloat16).contiguous()  # the sink keys in the same (shifted) basis
    return mg.merge_sink(o, ml, qe, sk, v[:, :, :NS].contiguous(), SM).float()


def rel(out, r):
    return ((out - r).norm(dim=-1) / r.norm(dim=-1)).mean().item()


def main():
    print(torch.cuda.get_device_name(), "commit", os.environ.get("KV4_COMMIT", "unset"), flush=True)
    ok = True
    for big in (0.0, 20.0, 60.0, 200.0):
        q, k, v = make(4, 4096, big)
        r = exact(q, k, v)
        res = {}
        for kf in ("nvfp4", "nvfp4_h"):
            for smooth in (False, True):
                res[(kf, smooth)] = rel(run(q, k, v, kf, smooth), r)
        line = "  ".join(f"{kf}{'+mean' if s else ''} {e:.4f}" for (kf, s), e in res.items())
        print(f"offset channel {big:5.0f}: relative output error vs exact BF16 attention: {line}", flush=True)
        if big >= 20:  # with a large offset, mean removal + rotation must beat rotation alone clearly
            good = res[("nvfp4_h", True)] < 0.7 * res[("nvfp4_h", False)]
            ok &= good
    print("smooth check", "ok" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
