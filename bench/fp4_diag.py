"""Where does the H200 NVFP4 kernel's time go? (11-11) Times decode_fp4 with diag 0 (normal), 1 (no reconstruction:
raw code words to the tensor cores), 2 (no matrix multiplies; reconstruction kept), plus our INT4 for reference.

    python bench/fp4_diag.py
"""
import math

import torch

from bench.bench_decode import PEAK_GBS, quantize_chunked, timeit
from kv4.cuda import decode_fp4 as df
from kv4.cuda import decode_int4 as di

HQ, HKV, D = 32, 8, 128
SM = 1 / math.sqrt(D)


def main():
    gpu = torch.cuda.get_device_name()
    peak = next((v for k, v in PEAK_GBS.items() if k in gpu), float("nan"))
    print(gpu, flush=True)
    ef, e4 = df.ext(), di.ext()
    for b, s in [(32, 16384), (128, 16384)]:
        torch.manual_seed(0)
        q = torch.randn(b, HQ, D, device="cuda")
        k = torch.randn(b, HKV, s, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        kq, vq = quantize_chunked(k, "nvfp4", chunk=1), quantize_chunked(v, "nvfp4", chunk=1)
        a = df.args_for(kq, vq, SM)
        qh = df.prep_q(q, HKV)
        nb = b * HKV * s * 2 * 72
        for diag, name in [(0, "normal"), (1, "no reconstruction"), (2, "no MMA")]:
            ms = timeit(lambda: ef.decode_fp4(*a[0], qh, a[1], a[2], a[3], a[4], 1, 2, False, False, diag))
            print(f"nvfp4 diag={diag} ({name:17s}) B={b:3d} S={s}: {ms:.4f} ms ({nb / ms / 1e6 / peak:.0%} of peak "
                  f"at 72 B/row)", flush=True)
        kq4, vq4 = quantize_chunked(k, "int4a_tok", chunk=1), quantize_chunked(v, "int4a_tok", chunk=1)
        c4 = di.prepare(kq4, vq4)
        qp, qc = di.prep_q(q, HKV)
        ms = timeit(lambda: e4.decode_int4(*c4, qp, qc, SM, 1, 2, True))
        print(f"int4a                         B={b:3d} S={s}: {ms:.4f} ms ({b * HKV * s * 136 / ms / 1e6 / peak:.0%})",
              flush=True)
        del q, k, v, kq, vq, kq4, vq4, c4, a
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
