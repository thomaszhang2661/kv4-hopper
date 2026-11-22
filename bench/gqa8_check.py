"""G = 8 query heads per kv head in the FP4 decode kernel (11-11 evening; e.g. Llama-3.1-70B: 64 query heads, 8 kv
heads, head_dim 128). The wgmma N = 8 rows that G = 4 pads with zeros are all real here, so the tensor-core work per
byte doubles for free and the softmax work per lane doubles.

Correctness (dense and paged, NVFP4 and MXFP4, merge_sink) against the FP32 reference; speed at the same (B, HKV, S)
for G = 4 and G = 8, ours against FlashInfer fa2 BF16 / FP8 at the same G.

    python bench/gqa8_check.py
"""
import math
import os
import sys

import torch

from bench.bench_decode import PEAK_GBS, quantize_chunked, timeit
from kv4.cuda import decode_fp4 as df
from kv4.cuda import decode_int4 as di
from kv4.cuda import merge as mg
from kv4.formats import dequantize, quantize

HKV, D = 8, 128
SM = 1 / math.sqrt(D)


def ref(q, kq, vq, lens=None):
    b, hq, _ = q.shape
    g = hq // HKV
    k, v = dequantize(kq, rotate_back=False).float(), dequantize(vq, rotate_back=False).float()
    outs = []
    for i in range(b):
        L = k.shape[2] if lens is None else lens[i]
        qe = q[i].reshape(HKV, g, D).half().float()
        p = torch.softmax(torch.einsum("hgd,hsd->hgs", qe, k[i, :, :L]) * SM, -1)
        outs.append(torch.einsum("hgs,hsd->hgd", p, v[i, :, :L]))
    return torch.stack(outs)


def err_of(out, r):
    out = out.float().reshape(r.shape)
    bad = (~torch.isfinite(out)).sum().item()
    return max(((out[i] - r[i]).abs().max() / r[i].abs().max()).item() for i in range(r.shape[0])), bad


def flashinfer(b, s, hq, k, v, page):
    from bench.bench_flashinfer import make_wrapper, page_table
    indptr, indices, last = page_table([s] * b, page)
    q = torch.randn(b, hq, D, device="cuda", dtype=torch.bfloat16)
    paged = lambda x: x.reshape(b, HKV, s // page, page, D).permute(0, 2, 1, 3, 4).reshape(-1, HKV, page, D).contiguous()  # noqa: E731
    kb, vb = paged(k), paged(v)
    w, _ = make_wrapper("fa2", True, indptr, indices, last, hq, HKV, D, page, torch.bfloat16)
    k8, v8 = kb.to(torch.float8_e4m3fn), vb.to(torch.float8_e4m3fn)
    w8, _ = make_wrapper("fa2", True, indptr, indices, last, hq, HKV, D, page, torch.float8_e4m3fn)
    return {"bf16": lambda: w.run(q, (kb, vb)), "fp8": lambda: w8.run(q, (k8, v8), k_scale=1.0, v_scale=1.0)}


def main():
    gpu = torch.cuda.get_device_name()
    peak = next((v for k, v in PEAK_GBS.items() if k in gpu), float("nan"))
    print(gpu, "commit", os.environ.get("KV4_COMMIT", "unset"), flush=True)
    e = df.ext()
    ok = True
    for fmt in ("nvfp4", "mxfp4"):
        for g in (4, 8):
            torch.manual_seed(0)
            b, s = 2, 4096
            q = torch.randn(b, HKV * g, D, device="cuda")
            kq = quantize(torch.randn(b, HKV, s, D, device="cuda", dtype=torch.bfloat16), fmt)
            vq = quantize(torch.randn(b, HKV, s, D, device="cuda", dtype=torch.bfloat16), fmt)
            a = df.args_for(kq, vq, SM)
            qh = df.prep_q(q, HKV)
            r = ref(q, kq, vq)
            for sp in (1, 2, 8):
                o, ml = e.decode_fp4(*a[0], qh, a[1], a[2], a[3], a[4], sp, 2, a[5], gqa=g)
                err, bad = err_of(di.combine(o, ml), r)
                err_m, _ = err_of(mg.ext().merge(o, ml), r)
                good = bad == 0 and err < 2e-3 and err_m < 5e-3
                ok &= good
                print(f"check {fmt} G={g} dense S={s} splits={sp}: max err {err:.2e} (merge kernel {err_m:.2e}), "
                      f"non-finite {bad} -> {'ok' if good else 'FAIL'}", flush=True)
            lens = [300, 4096, 1000, 257, 64, 1]
            torch.manual_seed(1)
            bp = len(lens)
            q = torch.randn(bp, HKV * g, D, device="cuda")
            kq = quantize(torch.randn(bp, HKV, s, D, device="cuda", dtype=torch.bfloat16), fmt)
            vq = quantize(torch.randn(bp, HKV, s, D, device="cuda", dtype=torch.bfloat16), fmt)
            a = df.args_for(kq, vq, SM)
            pages, bt, sl = df.pack_pages(a[0], lens)
            qh = df.prep_q(q, HKV)
            r = ref(q, kq, vq, lens)
            for sp in (1, 3, 8):
                o, ml = e.decode_fp4_paged(pages, bt, sl, qh, a[1], a[2], a[3], a[4], sp, a[5], gqa=g)
                err, bad = err_of(di.combine(o, ml), r)
                good = bad == 0 and err < 2e-3
                ok &= good
                print(f"check {fmt} G={g} paged lens={lens} splits={sp}: max err {err:.2e}, non-finite {bad} -> "
                      f"{'ok' if good else 'FAIL'}", flush=True)
    print("all G checks", "ok" if ok else "FAIL", flush=True)
    if os.environ.get("KV4_CHECK_ONLY") == "1":  # tests/gpu: correctness only, exit status = result
        sys.exit(0 if ok else 1)
    for b, s in [(32, 16384), (128, 16384), (8, 65536), (1, 65536), (16, 32768)]:
        torch.manual_seed(0)
        k = torch.randn(b, HKV, s, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        nbytes = b * HKV * s * 2 * 72
        line = [f"B={b:3d} S={s:6d}:"]
        kq, vq = quantize_chunked(k, "nvfp4", chunk=1), quantize_chunked(v, "nvfp4", chunk=1)
        a = df.args_for(kq, vq, SM)
        del kq, vq
        pages, bt, _ = df.pack_pages(a[0], [s] * b, poison=False)
        sl = torch.full((b,), s, dtype=torch.int32, device="cuda")
        res = {}
        for g in (4, 8):
            q = torch.randn(b, HKV * g, D, device="cuda")
            qh = df.prep_q(q, HKV)
            t, sp = min((timeit(lambda: mg.ext().merge(*e.decode_fp4_paged(pages, bt, sl, qh, a[1], a[2], a[3], a[4], sp,
                                                                          a[5], gqa=g))), sp) for sp in (1, 2, 4, 8, 16))
            fi = flashinfer(b, s, HKV * g, k, v, 16)
            tb, tf = timeit(fi["bf16"]), timeit(fi["fp8"])
            res[g] = t
            line.append(f"G={g}: ours paged NVFP4 {t:.4f} ms ({nbytes / t / 1e6 / peak:.0%} BW, splits {sp}), "
                        f"FlashInfer FP8 {tf:.4f} ({tf / t:.2f}x), BF16 {tb:.4f} ({tb / t:.2f}x) |")
        line.append(f"G=8 / G=4 time {res[8] / res[4]:.3f}")
        print(" ".join(line), flush=True)
        del k, v, pages, a
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
