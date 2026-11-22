"""Pass 3: sweep BLOCK_N, num_warps, num_stages for a few formats (B=32, S=16K, and B=8, S=64K).

Pipelining hypothesis was refuted (num_stages irrelevant); this sweep now extends BLOCK_N to 256/512. Prints ms per setting and the best one.
"""
import itertools
import math

import torch
import triton

from kv4.formats import quantize
from kv4.kernels.decode_triton import decode_attention

HQ, HKV, D = 32, 8, 128
FORMATS = ["bf16", "fp8", "int4_tok", "int4_g32", "nvfp4", "mxfp4"]
for b, s in [(32, 16384), (8, 65536)]:
    torch.manual_seed(0)
    q = torch.randn(b, HQ, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, HKV, s, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    seq = torch.full((b,), s, device="cuda", dtype=torch.int32)
    sm = 1 / math.sqrt(D)
    print(f"== B={b} S={s}", flush=True)
    for name in FORMATS:
        kq, vq = (k, v) if name == "bf16" else (quantize(k, name), quantize(v, name))
        try:
            base = triton.testing.do_bench(lambda: decode_attention(q, kq, vq, seq, sm, fmt_name=name), warmup=10, rep=40)
        except Exception as e:  # e.g. fp8 on a GPU without fp8 support
            print(f"  >> {name}: skipped ({type(e).__name__}: {str(e)[:80]})", flush=True)
            continue
        best = (base, "default")
        for bn, nw, ns in itertools.product([64, 128, 256, 512], [4, 8], [2, 3]):
            try:
                fn = lambda: decode_attention(q, kq, vq, seq, sm, fmt_name=name, block_n=bn, num_warps=nw, num_stages=ns)
                ms = triton.testing.do_bench(fn, warmup=10, rep=40)
            except Exception as e:  # e.g. out of shared memory for this combination
                print(f"  {name} bn={bn} nw={nw} ns={ns}: failed ({type(e).__name__})", flush=True)
                continue
            print(f"  {name} bn={bn} nw={nw} ns={ns}: {ms:.3f} ms", flush=True)
            best = min(best, (ms, f"bn={bn} nw={nw} ns={ns}"))
        print(f"  >> {name}: default {base:.3f} ms, best {best[0]:.3f} ms ({best[1]})", flush=True)
        if name != "bf16":
            del kq, vq
