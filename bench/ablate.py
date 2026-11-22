"""Ablation + register report for the decode kernels (separates the H1/H2/H3 hypotheses in docs/plan_*.md).

For token-wise INT4 (the folded kernel), at each BLOCK_N and a few shapes:
  abl=0 full kernel, abl=1 loads only (no dot, no softmax), abl=2 no nibble unpacking,
  abl=3 no scale loads or fix-ups.  abl=1..3 give wrong numbers: timing only.
Also prints registers per thread and spills for several formats at each BLOCK_N.
"""
import math

import torch
import triton

from kv4.formats import quantize
from kv4.kernels import decode_triton as dt

HQ, HKV, D = 32, 8, 128
NAMES = {0: "full", 1: "loads_only", 2: "no_unpack", 3: "no_scales"}
print(torch.cuda.get_device_name(), flush=True)
for b, s in [(32, 16384), (8, 65536)]:
    torch.manual_seed(0)
    q = torch.randn(b, HQ, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, HKV, s, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    seq = torch.full((b,), s, device="cuda", dtype=torch.int32)
    sm = 1 / math.sqrt(D)
    print(f"== B={b} S={s}", flush=True)
    kq, vq = quantize(k, "int4_tok"), quantize(v, "int4_tok")
    for bn in [64, 128, 256]:
        row = []
        for abl in range(4):
            fn = lambda: dt.decode_attention(q, kq, vq, seq, sm, fmt_name="int4_tok", block_n=bn, num_warps=4, abl=abl)
            try:
                ms = triton.testing.do_bench(fn, warmup=10, rep=40)
            except Exception as e:
                row.append(f"{NAMES[abl]} failed({type(e).__name__})")
                continue
            row.append(f"{NAMES[abl]} {ms:.3f}")
        print(f"  int4_tok bn={bn} nw=4: " + " | ".join(row) + f"   [regs={dt.LAST_META.get('regs')} spills={dt.LAST_META.get('spills')} for abl=3]", flush=True)
    del kq, vq
    print("  -- registers / spills (full kernels, nw=4) --", flush=True)
    for name in ["bf16", "int4_tok", "int4_g32", "mxfp4"]:
        kq, vq = (k, v) if name == "bf16" else (quantize(k, name), quantize(v, name))
        for bn in [64, 128, 256, 512]:
            try:
                dt.decode_attention(q, kq, vq, seq, sm, fmt_name=name, block_n=bn, num_warps=4)
                print(f"  {name:9s} bn={bn:3d}: regs={dt.LAST_META['regs']} spills={dt.LAST_META['spills']}", flush=True)
            except Exception as e:
                print(f"  {name:9s} bn={bn:3d}: failed ({type(e).__name__})", flush=True)
        if name != "bf16":
            del kq, vq
