"""11-18: TensorRT-LLM's FP8 decode kernel (XQA, through FlashInfer) as the FP8 baseline at the kernel level.

The kernel tables (bench/final_table.py) time FlashInfer FP8 through BatchDecodeWithPagedKVCacheWrapper (fa2, tensor
cores). vLLM 0.33 on SM90 decodes an FP8 cache with XQA instead (vllm/utils/flashinfer.py: "SM90/SM12x + FP8 KV
cache: prefer the XQA decode kernel"), and FOSCAR's FP8 baseline is XQA. This script times, in one job at the
11 shapes of the kernel tables (Llama-3.1-8B heads: 32 query, 8 KV, d = 128, same seeds and timer):
  flashinfer_fp8         fa2 wrapper, page 16 (the kernel tables' FP8 column)
  xqa_fp8_p{16,32,64,128} flashinfer.decode.xqa_batch_decode_with_kv_cache, FP8 E4M3 cache, BF16 query (as vLLM
                         calls it: HND, bmm1_scale = sm_scale, bmm2_scale = 1); on SM90 this runs xqa_mha_sm90
  cuda_nvfp4             ours, best split count; cuda_nvfp4_rule: the plugin's fixed rule
Correctness: every XQA output is compared with the fa2 FP8 output on the same FP8 cache (printed max abs diff).

    python bench/xqa_compare.py --tag v1
"""
import argparse
import csv
import math
import os
from pathlib import Path

import torch

from bench.bench_decode import PEAK_GBS, quantize_chunked, timeit
from bench.bench_flashinfer import make_wrapper, page_table
from kv4.cuda import decode_fp4 as df
from kv4.cuda import merge as mg

HQ, HKV, D = 32, 8, 128
SM = 1 / math.sqrt(D)
SHAPES = "1x16384,1x65536,1x131072,4x32768,8x65536,16x32768,16x131072,32x4096,32x16384,64x16384,128x16384"
PAGES = (16, 32, 64, 128)


def paged(x, b, s, page):
    """(B, HKV, S, D) -> (pages, HKV, page, D) (HND), pages of one request contiguous."""
    return x.reshape(b, HKV, s // page, page, D).permute(0, 2, 1, 3, 4).reshape(-1, HKV, page, D).contiguous()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="v1")
    p.add_argument("--shapes", default=SHAPES)
    p.add_argument("--pages", default=",".join(map(str, PAGES)))
    args = p.parse_args()
    from flashinfer.decode import xqa_batch_decode_with_kv_cache
    import flashinfer
    gpu = torch.cuda.get_device_name()
    peak = next((v for k, v in PEAK_GBS.items() if k in gpu), float("nan"))
    commit = os.environ.get("KV4_COMMIT", "unset")
    print(gpu, "commit", commit, "flashinfer", flashinfer.__version__, flush=True)
    ef, em = df.ext(), mg.ext()
    ws = torch.zeros(512 * 2**20, dtype=torch.uint8, device="cuda")  # XQA: 8 MB semaphores + scratch
    rows, table = [], []
    for shape in args.shapes.split(","):
        b, s = map(int, shape.split("x"))
        torch.manual_seed(0)
        q = torch.randn(b, HQ, D, device="cuda")
        k = torch.randn(b, HKV, s, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        res = {}

        def record(impl, ms, nbytes, cfg=""):
            gbs = nbytes / ms / 1e6
            res[impl] = ms
            rows.append(dict(gpu=gpu, commit=commit, tag=args.tag, impl=impl, config=cfg, batch=b, ctx=s,
                             ms=f"{ms:.4f}", gbs=f"{gbs:.0f}", peak_frac=f"{gbs / peak:.3f}"))
            print(f"{impl:18s} B={b:3d} S={s:6d} {cfg:12s} {ms:8.4f} ms {gbs:6.0f} GB/s ({gbs / peak:.0%})", flush=True)

        fp8_bytes = b * HKV * s * 2 * 128
        qb = q.to(torch.bfloat16)
        k8, v8 = k.to(torch.float8_e4m3fn), v.to(torch.float8_e4m3fn)  # (B, HKV, S, D), scale 1 as in final_table
        # fa2 FP8, page 16 (the kernel tables' baseline)
        indptr, indices, last = page_table([s] * b, 16)
        kp, vp = paged(k8, b, s, 16), paged(v8, b, s, 16)
        w8, resolved = make_wrapper("fa2", True, indptr, indices, last, HQ, HKV, D, 16, torch.float8_e4m3fn)
        ref = w8.run(qb, (kp, vp), k_scale=1.0, v_scale=1.0)
        record("flashinfer_fp8", timeit(lambda: w8.run(qb, (kp, vp), k_scale=1.0, v_scale=1.0)), fp8_bytes,
               f"fa2/{resolved}")
        del kp, vp, w8
        for page in map(int, args.pages.split(",")):
            if s % page:
                continue
            kp, vp = paged(k8, b, s, page), paged(v8, b, s, page)
            bt = torch.arange(b * (s // page), dtype=torch.int32, device="cuda").reshape(b, s // page)
            sl = torch.full((b,), s, dtype=torch.uint32, device="cuda")
            out = torch.empty(b, HQ, D, dtype=torch.bfloat16, device="cuda")

            def run():
                return xqa_batch_decode_with_kv_cache(qb, (kp, vp), ws, bt, sl, s, bmm1_scale=SM, bmm2_scale=1.0,
                                                      out=out, kv_layout="HND")
            try:
                run()
                torch.cuda.synchronize()
                err = (out.float() - ref.float()).abs().max().item()
                record(f"xqa_fp8_p{page}", timeit(run), fp8_bytes, f"err={err:.1e}")
            except Exception as ex:  # noqa: BLE001
                print(f"xqa page {page} failed: {str(ex).splitlines()[0][:160]}", flush=True)
            del kp, vp, bt, sl, out
        del k8, v8, ref
        torch.cuda.empty_cache()
        # ours: NVFP4, best split and the plugin's fixed rule (as bench/final_table.py)
        splits_list = [sp for sp in (1, 2, 4, 8, 16, 32, 64) if s % (sp * 256) == 0 and sp * HKV * b <= 32768]
        kq, vq = quantize_chunked(k, "nvfp4", chunk=1), quantize_chunked(v, "nvfp4", chunk=1)
        a = df.args_for(kq, vq, SM)
        qh = df.prep_q(q, HKV)
        del kq, vq
        cache, smul, omul, nbk, nbv, mx = a
        nb = b * HKV * s * 2 * 72
        best = min((timeit(lambda: em.merge(*ef.decode_fp4(*cache, qh, smul, omul, nbk, nbv, sp, 2, mx))), sp)
                   for sp in splits_list)
        record("cuda_nvfp4", best[0], nb, f"splits={best[1]}")
        rule = max(1, min(16, (2 * 132) // max(1, b * HKV)))
        record("cuda_nvfp4_rule", timeit(lambda: em.merge(*ef.decode_fp4(*cache, qh, smul, omul, nbk, nbv, rule, 2, mx))),
               nb, f"splits={rule}")
        del a, cache
        table.append((b, s, dict(res)))
        del q, k, v
        torch.cuda.empty_cache()
    gtag = next((k for k in PEAK_GBS if k in gpu), gpu.replace(" ", "_"))
    out = Path(__file__).resolve().parent.parent / "results" / f"xqa_compare_{gtag}_{args.tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, flush=True)
    print("\nshape        fa2_fp8   best_xqa(page)      nvfp4  nvfp4_rule   fa2/xqa  xqa/nvfp4  xqa/nvfp4_rule")
    for b, s, r in table:
        xq = {kk: vv for kk, vv in r.items() if kk.startswith("xqa_fp8")}
        if not xq:
            print(f"{b}x{s}: no xqa")
            continue
        bk = min(xq, key=xq.get)
        x = xq[bk]
        print(f"{b:3d}x{s:<7d} {r['flashinfer_fp8']:8.4f}  {x:8.4f} ({bk[9:]:>4s})  {r['cuda_nvfp4']:8.4f}  "
              f"{r['cuda_nvfp4_rule']:8.4f}   {r['flashinfer_fp8'] / x:6.2f}x  {x / r['cuda_nvfp4']:6.2f}x  "
              f"{x / r['cuda_nvfp4_rule']:6.2f}x", flush=True)


if __name__ == "__main__":
    main()
