"""11-21: C1 prototype (kv4/cuda/decode_fp4_k8.cu, K on FP8 tensor cores) against the current NVFP4 kernel.

1. Correctness: max abs error of each kernel against its FP32 reference (bench/real_kv_check.py style tolerance:
   ~2.5e-4 on random data), and for k8 also the error that q in E4M3 adds (reference with q exact vs q rounded).
2. Speed: both kernels with merge, and (11-21 v2) XQA FP8 in the same job, best split per shape, same timer as the paper tables (bench_decode.timeit), the
   11 shapes of the kernel tables. Bandwidth = bytes the format stores (k8: K 68 + V 72 per token and kv head;
   nvfp4: 72 + 72).
    python bench/k8_proto.py --tag v1
"""
import argparse
import csv
import math
import os
from pathlib import Path

import torch

from bench.bench_decode import PEAK_GBS, quantize_chunked, timeit
from kv4.cuda import decode_fp4 as df
from kv4.cuda import decode_fp4_k8 as k8
from kv4.cuda import merge as mg
from kv4.formats import quantize

HQ, HKV, D = 32, 8, 128
SM = 1 / math.sqrt(D)
SHAPES = "1x16384,1x65536,1x131072,4x32768,8x65536,16x32768,16x131072,32x4096,32x16384,64x16384,128x16384"


def correctness(ef, ek, em):
    torch.manual_seed(0)
    for b, s in ((2, 4096), (1, 16384)):
        q = torch.randn(b, HQ, D, device="cuda")
        k = torch.randn(b, HKV, s, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        kq8, kq, vq = quantize(k, "nvfp4_g32"), quantize(k, "nvfp4"), quantize(v, "nvfp4")
        cache, smul, omul = k8.args_for(kq8, vq, SM)
        q8, qs = k8.prep_q8(q, HKV)
        for sp in (1, 4):
            out = em.merge(*ek.decode_k8(*cache, q8, smul * qs, omul, sp)).reshape(b, HKV, 4, D)
            r8 = k8.reference(q, kq8, vq, SM, q_e4m3=True)
            rx = k8.reference(q, kq8, vq, SM, q_e4m3=False)
            print(f"k8    B={b} S={s} splits={sp}: max|kernel - ref(q e4m3)| {(out - r8).abs().max().item():.2e}, "
                  f"max|ref(q e4m3) - ref(q exact)| {(r8 - rx).abs().max().item():.2e}, "
                  f"max|ref_g32 - ref_g16| {(rx - df.reference(q, kq, vq, SM)).abs().max().item():.2e}", flush=True)
        a = df.args_for(kq, vq, SM)
        c, sm2, om2, nbk, nbv, mx = a
        out = em.merge(*ef.decode_fp4(*c, df.prep_q(q, HKV), sm2, om2, nbk, nbv, 1, 2, mx)).reshape(b, HKV, 4, D)
        print(f"nvfp4 B={b} S={s}: max|kernel - ref| {(out - df.reference(q, kq, vq, SM)).abs().max().item():.2e}",
              flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="v1")
    p.add_argument("--shapes", default=SHAPES)
    args = p.parse_args()
    gpu = torch.cuda.get_device_name()
    peak = next((v for k, v in PEAK_GBS.items() if k in gpu), float("nan"))
    commit = os.environ.get("KV4_COMMIT", "unset")
    print(gpu, "commit", commit, flush=True)
    global WS
    WS = torch.zeros(512 * 2**20, dtype=torch.uint8, device="cuda")  # XQA workspace
    ef, ek, em = df.ext(), k8.ext(), mg.ext()
    correctness(ef, ek, em)
    rows = []
    for shape in args.shapes.split(","):
        b, s = map(int, shape.split("x"))
        torch.manual_seed(0)
        q = torch.randn(b, HQ, D, device="cuda")
        k = torch.randn(b, HKV, s, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        splits_list = [sp for sp in (1, 2, 4, 8, 16, 32, 64) if s % (sp * 256) == 0 and sp * HKV * b <= 32768]
        vq = quantize_chunked(v, "nvfp4", chunk=1)
        kq = quantize_chunked(k, "nvfp4", chunk=1)
        c, smul, omul, nbk, nbv, mx = df.args_for(kq, vq, SM)
        qh = df.prep_q(q, HKV)
        del kq
        t16 = min((timeit(lambda: em.merge(*ef.decode_fp4(*c, qh, smul, omul, nbk, nbv, sp, 2, mx))), sp)
                  for sp in splits_list)
        del c
        kq8 = quantize_chunked(k, "nvfp4_g32", chunk=1)
        c8, smul8, omul8 = k8.args_for(kq8, vq, SM)
        q8, qs = k8.prep_q8(q, HKV)
        del kq8, vq
        t8 = min((timeit(lambda: em.merge(*ek.decode_k8(*c8, q8, smul8 * qs, omul8, sp))), sp) for sp in splits_list)
        del c8
        # 11-21: XQA FP8 in the same job (as bench/xqa_compare.py: HND pages, best of pages 16 / 32 / 64 / 128)
        from bench.xqa_compare import paged
        from flashinfer.decode import xqa_batch_decode_with_kv_cache
        k8c, v8c = k.to(torch.float8_e4m3fn), v.to(torch.float8_e4m3fn)
        qb = q.to(torch.bfloat16)
        tx = None
        for page in (16, 32, 64, 128):
            kpg, vpg = paged(k8c, b, s, page), paged(v8c, b, s, page)
            bt = torch.arange(b * (s // page), dtype=torch.int32, device="cuda").reshape(b, s // page)
            sl = torch.full((b,), s, dtype=torch.uint32, device="cuda")
            ob = torch.empty(b, HQ, D, dtype=torch.bfloat16, device="cuda")
            ms = timeit(lambda: xqa_batch_decode_with_kv_cache(qb, (kpg, vpg), WS, bt, sl, s, bmm1_scale=SM,
                                                               bmm2_scale=1.0, out=ob, kv_layout="HND"))
            tx = min(tx or (9e9, 0), (ms, page))
            del kpg, vpg, bt, sl, ob
        del k8c, v8c
        for impl, (ms, sp), nb in (("cuda_nvfp4", t16, b * HKV * s * 144), ("cuda_k8", t8, b * HKV * s * 140),
                                   ("xqa_fp8", tx, b * HKV * s * 256)):
            gbs = nb / ms / 1e6
            rows.append(dict(gpu=gpu, commit=commit, tag=args.tag, impl=impl, splits=sp, batch=b, ctx=s,
                             ms=f"{ms:.4f}", gbs=f"{gbs:.0f}", peak_frac=f"{gbs / peak:.3f}"))
        print(f"B={b:3d} S={s:6d}  nvfp4 {t16[0]:.4f} ms (splits {t16[1]})  k8 {t8[0]:.4f} ms (splits {t8[1]})  "
              f"nvfp4/k8 {t16[0] / t8[0]:.3f}  k8 {b * HKV * s * 140 / t8[0] / 1e6 / peak:.0%} of peak  "
              f"xqa {tx[0]:.4f} ms (page {tx[1]})  xqa/nvfp4 {tx[0] / t16[0]:.2f}  xqa/k8 {tx[0] / t8[0]:.2f}", flush=True)
        del q, k, v
        torch.cuda.empty_cache()
    gtag = next((k for k in PEAK_GBS if k in gpu), gpu.replace(" ", "_"))
    out = Path(__file__).resolve().parent.parent / "results" / f"k8_proto_{gtag}_{args.tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
