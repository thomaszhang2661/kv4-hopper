"""Fairness check for the speed table (11-11): FlashInfer at page 16 (what final_table.py uses) and page 64 (the page
size of our paged kernel and of vLLM UltraQuant), next to our dense and paged (page 64) NVFP4 / MXFP4, same inputs
(B, S, 32 query heads, 8 KV heads, D = 128) in one job.

    python bench/fairness_page.py
"""
import torch

from bench.bench_decode import PEAK_GBS, quantize_chunked, timeit
from bench.decode_fp4_bench import flashinfer_nvfp4
from bench.decode_int4_bench import flashinfer_fns
from kv4.cuda import decode_fp4 as df

HQ, HKV, D = 32, 8, 128
SM = 1.0 / D ** 0.5


def main():
    gpu = torch.cuda.get_device_name()
    peak = next((v for k, v in PEAK_GBS.items() if k in gpu), float("nan"))
    print(gpu, flush=True)
    e = df.ext()
    for b, s in [(32, 16384), (128, 16384), (8, 65536)]:
        torch.manual_seed(0)
        k = torch.randn(b, HKV, s, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        res = {}
        for page in (16, 64):
            for name, (fn, nb) in flashinfer_fns(b, s, k, v, page).items():
                res[f"{name} p{page}"] = (timeit(fn), nb)
            fn, nb = flashinfer_nvfp4(b, s, page)
            res[f"flashinfer_nvfp4 p{page}"] = (timeit(fn), nb)
        q = torch.randn(b, HQ, D, device="cuda")
        qh = df.prep_q(q, HKV)
        for fmt in ("nvfp4", "mxfp4"):
            kq, vq = quantize_chunked(k, fmt, chunk=1), quantize_chunked(v, fmt, chunk=1)
            a = df.args_for(kq, vq, SM)
            nb = b * HKV * s * 2 * (72 if fmt == "nvfp4" else 68)
            res[f"ours_{fmt} dense"] = (timeit(lambda: e.decode_fp4(*a[0], qh, a[1], a[2], a[3], a[4], 1, 2, a[5])), nb)
            pages, bt, sl = df.pack_pages(a[0], [s] * b, poison=False, shuffle=True)
            res[f"ours_{fmt} paged p64 shuffled"] = (
                timeit(lambda: e.decode_fp4_paged(pages, bt, sl, qh, a[1], a[2], a[3], a[4], 1, a[5])), nb)
            del kq, vq, a, pages, bt
        fp8 = res["flashinfer_fp8 p16"][0]
        for name, (ms, nb) in res.items():
            print(f"B={b:3d} S={s:6d} {name:32s} {ms:8.4f} ms {nb / ms / 1e6 / peak:5.0%} of peak  "
                  f"{fp8 / ms:5.2f}x vs FlashInfer FP8 p16", flush=True)
        del k, v
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
