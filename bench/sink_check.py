"""Attention sink in BF16 with the 4-bit decode kernels (11-11): the first NS tokens stay out of the 4-bit cache and are
joined in the merge kernel (kv4/cuda/merge.cu merge_sink); the decode kernels read tokens [NS, len) only.

Correctness: against an FP32 reference over [sink in BF16 | dequantized rest], with a strong sink (about half of the
attention mass, as in real Llama layers). Speed: one whole decode step (kernel + merge_sink) against kernel + merge and
the kernel alone, at the best splits for each.

    python bench/sink_check.py
"""
import math
import os
import sys

import torch

from bench.bench_decode import quantize_chunked, timeit
from kv4.cuda import decode_fp4 as df
from kv4.cuda import decode_int4 as di
from kv4.cuda import kside
from kv4.cuda import merge as mg
from kv4.formats import dequantize, quantize

HQ, HKV, D, GQA, NS = 32, 8, 128, 4, 4
SM = 1 / math.sqrt(D)


def make(b, s, seed=0):
    """q (B, HQ, D) fp32; k, v (B, HKV, NS + s, D) bf16 with a strong sink at token 0."""
    torch.manual_seed(seed)
    q = torch.randn(b, HQ, D, device="cuda")
    k = torch.randn(b, HKV, NS + s, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    qg = q.reshape(b, HKV, GQA, D)
    u = torch.nn.functional.normalize(qg.sum(2), dim=-1)                     # (B, HKV, D)
    proj = (qg * u[:, :, None]).sum(-1).mean(-1, keepdim=True).clamp(min=1.0)  # mean q.u per kv head
    k[:, :, 0] = (u * 8.0 / (SM * proj)).to(k.dtype)                         # score about 8 -> about half the mass
    return q, k, v


def ref(qe, k, v, kq, vq, lens=None):
    """qe (B, HKV, 4, D) the q the kernel effectively uses; sink from k/v (BF16), the rest dequantized."""
    kd, vd = dequantize(kq, rotate_back=False).float(), dequantize(vq, rotate_back=False).float()
    outs = []
    for b in range(qe.shape[0]):
        L = kd.shape[2] if lens is None else lens[b]
        kk = torch.cat([k[b, :, :NS].float(), kd[b, :, :L]], 1)
        vv = torch.cat([v[b, :, :NS].float(), vd[b, :, :L]], 1)
        p = torch.softmax(torch.einsum("hgd,hsd->hgs", qe[b], kk) * SM, -1)
        outs.append(torch.einsum("hgs,hsd->hgd", p, vv))
    return torch.stack(outs)


def err_of(out, r):
    out = out.float().reshape(r.shape)
    bad = (~torch.isfinite(out)).sum().item()
    return max(((out[i] - r[i]).abs().max() / r[i].abs().max()).item() for i in range(r.shape[0])), bad


def main():
    print(torch.cuda.get_device_name(), "commit", os.environ.get("KV4_COMMIT", "unset"), flush=True)
    e, e4, em = df.ext(), di.ext(), mg.ext()
    ok = True
    # ---- correctness
    for fmt in ("nvfp4", "mxfp4", "int4a_tok"):
        b, s = 3, 2048
        q, k, v = make(b, s)
        kq, vq = quantize(k[:, :, NS:], fmt), quantize(v[:, :, NS:], fmt)
        sk, sv = k[:, :, :NS].contiguous(), v[:, :, :NS].contiguous()
        if fmt == "int4a_tok":
            c4 = di.prepare(kq, vq)
            qp, qc = di.prep_q(q, HKV)
            qe = qp[:, :, :GQA].float() * kside.dim_factor(q.device)
            r = ref(qe, k, v, kq, vq)
            runs = [(f"dense splits={sp}", lambda sp=sp: e4.decode_int4(*c4, qp, qc, SM, sp, 2, True)) for sp in (1, 2, 8)]
        else:
            a = df.args_for(kq, vq, SM)
            qh = df.prep_q(q, HKV)
            qe = q.reshape(b, HKV, GQA, D).half().float()
            r = ref(qe, k, v, kq, vq)
            runs = [(f"dense splits={sp}", lambda sp=sp: e.decode_fp4(*a[0], qh, a[1], a[2], a[3], a[4], sp, 2, a[5]))
                    for sp in (1, 2, 8)]
        for name, fn in runs:
            o, ml = fn()
            out = mg.merge_sink(o, ml, qe.reshape(b, HQ, D), sk, sv, SM)
            err, bad = err_of(out, r)
            o0 = mg.ext().merge(o, ml)                       # the same kernel output without the sink, for contrast
            err0, _ = err_of(o0, r)
            good = bad == 0 and err < 5e-3                   # bf16 output: rounding 2^-9 = 2e-3 of max |out|
            ok &= good
            print(f"check {fmt} {name}: max err {err:.2e} (without the sink {err0:.2e}), non-finite {bad} -> "
                  f"{'ok' if good else 'FAIL'}", flush=True)
        if fmt != "int4a_tok":  # paged, ragged lengths (quantized part = len - NS)
            lens = [300, 2048, 1000, 257, 64, 1]
            bp = len(lens)
            q, k, v = make(bp, 2048, seed=1)
            kq, vq = quantize(k[:, :, NS:], fmt), quantize(v[:, :, NS:], fmt)
            a = df.args_for(kq, vq, SM)
            pages, bt, sl = df.pack_pages(a[0], lens)
            qh = df.prep_q(q, HKV)
            qe = q.reshape(bp, HKV, GQA, D).half().float()
            r = ref(qe, k, v, kq, vq, lens)
            for sp in (1, 3, 8):
                o, ml = e.decode_fp4_paged(pages, bt, sl, qh, a[1], a[2], a[3], a[4], sp, a[5])
                out = mg.merge_sink(o, ml, qe.reshape(bp, HQ, D), k[:, :, :NS].contiguous(), v[:, :, :NS].contiguous(), SM)
                err, bad = err_of(out, r)
                good = bad == 0 and err < 5e-3
                ok &= good
                print(f"check {fmt} paged lens={lens} splits={sp}: max err {err:.2e}, non-finite {bad} -> "
                      f"{'ok' if good else 'FAIL'}", flush=True)
    print("all sink checks", "ok" if ok else "FAIL", flush=True)
    if os.environ.get("KV4_CHECK_ONLY") == "1":  # tests/gpu: correctness only, exit status = result
        sys.exit(0 if ok else 1)
    # ---- speed: one whole decode step
    for b, s in [(32, 16384), (128, 16384), (8, 65536), (1, 65536), (32, 4096)]:
        torch.manual_seed(0)
        q = torch.randn(b, HQ, D, device="cuda")
        sk = torch.randn(b, HKV, NS, D, device="cuda", dtype=torch.bfloat16)
        sv = torch.randn_like(sk)
        kb = torch.randn(b, HKV, s, D, device="cuda", dtype=torch.bfloat16)
        for fmt in ("nvfp4", "int4a_tok"):
            kq, vq = quantize_chunked(kb, fmt, chunk=1), quantize_chunked(torch.randn_like(kb), fmt, chunk=1)
            if fmt == "nvfp4":
                a = df.args_for(kq, vq, SM)
                qh = df.prep_q(q, HKV)
                pages, bt, _ = df.pack_pages(a[0], [s] * b, poison=False)
                sl_full = torch.full((b,), s, dtype=torch.int32, device="cuda")
                sl_sink = sl_full - NS
                kern = {"dense": lambda sp: e.decode_fp4(*a[0], qh, a[1], a[2], a[3], a[4], sp, 2, a[5]),
                        "paged": lambda sp: e.decode_fp4_paged(pages, bt, sl_full, qh, a[1], a[2], a[3], a[4], sp, a[5]),
                        "paged_sink": lambda sp: e.decode_fp4_paged(pages, bt, sl_sink, qh, a[1], a[2], a[3], a[4], sp,
                                                                    a[5])}
            else:
                c4 = di.prepare(kq, vq)
                qp, qc = di.prep_q(q, HKV)
                kern = {"dense": lambda sp: e4.decode_int4(*c4, qp, qc, SM, sp, 2, True)}
            for path in [p for p in kern if not p.endswith("_sink")]:
                ks = kern.get(path + "_sink", kern[path])  # paged: the kernel reads len - NS tokens
                res = {}
                for sp in (1, 2, 4, 8, 16):
                    if path == "dense" and s % (sp * 256):
                        continue
                    t_k = timeit(lambda: kern[path](sp))
                    t_m = timeit(lambda: em.merge(*kern[path](sp)))
                    t_s = timeit(lambda: mg.merge_sink(*ks(sp), q, sk, sv, SM))
                    for key, t in (("kernel", t_k), ("kernel+merge", t_m), ("kernel+merge_sink", t_s)):
                        if key not in res or t < res[key][0]:
                            res[key] = (t, sp)
                base = min(res["kernel"][0] if res["kernel"][1] == 1 else 1e9, res["kernel+merge"][0])
                print(f"{fmt:9s} {path:5s} B={b:3d} S={s:6d}: kernel {res['kernel'][0]:.4f} ms (splits "
                      f"{res['kernel'][1]}), kernel+merge {res['kernel+merge'][0]:.4f} (splits {res['kernel+merge'][1]}), "
                      f"kernel+merge_sink {res['kernel+merge_sink'][0]:.4f} (splits {res['kernel+merge_sink'][1]}) -> "
                      f"sink step / best step without sink {res['kernel+merge_sink'][0] / base:.3f}x", flush=True)
            del kq, vq
        del kb
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
