"""FP8 conversion-path variants of the Triton decode kernel (pass 3, step 2).

CONV 0 = per-element scale (pass 1), 1 = direct fp8->bf16, 2 = via fp32, 3 = via fp16, 4 = fp16 operands for the dots
(1-3 fold the per-tensor scales into the scores and the output). Prints ms per variant.
"""
import math

import torch
import triton

from kv4.formats import quantize
from kv4.kernels.decode_triton import decode_attention

HQ, HKV, D = 32, 8, 128
for b, s in [(32, 16384), (64, 65536)]:
    torch.manual_seed(0)
    q = torch.randn(b, HQ, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, HKV, s, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    seq = torch.full((b,), s, device="cuda", dtype=torch.int32)
    kq, vq = quantize(k, "fp8"), quantize(v, "fp8")
    sm = 1 / math.sqrt(D)
    ref = decode_attention(q, kq, vq, seq, sm, fmt_name="fp8", fp8_conv=0)
    t_bf = triton.testing.do_bench(lambda: decode_attention(q, k, v, seq, sm, fmt_name="bf16"), warmup=25, rep=100)
    print(f"B={b} S={s} bf16 {t_bf:.3f} ms", flush=True)
    for conv in range(5):
        fn = lambda: decode_attention(q, kq, vq, seq, sm, fmt_name="fp8", fp8_conv=conv)
        err = (fn() - ref).abs().max().item()
        print(f"  CONV={conv}: {triton.testing.do_bench(fn, warmup=25, rep=100):.3f} ms (max diff vs CONV=0: {err:.2e})", flush=True)
    del k, v, kq, vq
