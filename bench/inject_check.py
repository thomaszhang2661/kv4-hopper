"""Mantissa injection (0x4300 | code) vs the current unpack: numerical error first, then speed.

inject=1: correct for the +128 after the dot (fp32); inject=2: subtract 136 in bf16 before the dot (exact).

Error is measured against the float32 reference on the dequantized cache, for the injected and the
current kernel; the injected one must not be noticeably worse. Run: python bench/inject_check.py
"""
import math

import torch
import triton

from kv4.formats import quantize
from kv4.kernels.decode_triton import decode_attention
from kv4.reference import decode_attention_ref

HQ, HKV, D = 32, 8, 128
print(torch.cuda.get_device_name(), flush=True)
for name in ["int4_tok", "int4a_tok"]:
    for b, s in [(2, 1000), (32, 16384), (8, 65536)]:
        torch.manual_seed(0)
        q = torch.randn(b, HQ, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(b, HKV, s, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        if name == "int4a_tok":  # a large offset makes the zero-point term matter
            k, v = k + 3.0, v + 3.0
        seq = torch.full((b,), s, device="cuda", dtype=torch.int32)
        sm = 1 / math.sqrt(D)
        kq, vq = quantize(k, name), quantize(v, name)
        ref = decode_attention_ref(q, kq, vq, seq, sm) if b * s <= 2_000_000 else None
        outs, ms = {}, {}
        for inj in (0, 1, 2):
            fn = lambda: decode_attention(q, kq, vq, seq, sm, fmt_name=name, block_n=128, num_warps=4, inject=inj)
            outs[inj] = fn()
            ms[inj] = triton.testing.do_bench(fn, warmup=10, rep=40)
        line = f"{name:9s} B={b:3d} S={s:6d}: ms current/inj1/inj2 = {ms[0]:.3f}/{ms[1]:.3f}/{ms[2]:.3f}"
        line += ", max|inj-cur| = " + "/".join(f"{(outs[i] - outs[0]).abs().max().item():.1e}" for i in (1, 2))
        if ref is not None:
            line += ", err vs ref current/inj1/inj2 = " + "/".join(f"{(outs[i] - ref).abs().max().item():.1e}" for i in (0, 1, 2))
        print(line, flush=True)
        del k, v, kq, vq
