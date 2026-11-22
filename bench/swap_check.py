""""Swap A and B" for token-wise INT4: same result as the current kernel? faster? does the round trip go?

For int4_tok and int4a_tok: max |swap - current| (should be ~0: same math, different dot order),
error of both against the torch reference, median ms of both, registers/spills, and from the
compiled IR: wgmma use (warp_group_dot in ttgir, wgmma in PTX), local_alloc and convert_layout
counts (the write-back of unpacked bf16 to shared memory shows up as local_alloc of a bf16 tile).
Runs in either env; it prints the Triton version, so the same script answers for Triton 3.2 and 3.9.
"""
import math
import re

import torch
import triton

from bench.bench_decode import timeit
from kv4.formats import quantize
from kv4.kernels import decode_triton as dt
from kv4.reference import decode_attention_ref

print(torch.cuda.get_device_name(), "triton", triton.__version__, "torch", torch.__version__, flush=True)
HQ, HKV, D = 32, 8, 128
sm = 1 / math.sqrt(D)


def ir_counts(km) -> str:
    ttgir, ptx = km.asm.get("ttgir", ""), km.asm.get("ptx", "")
    n = lambda pat, text: len(re.findall(pat, text))  # noqa: E731
    bf16_alloc = n(r"local_alloc %\S+ : \(tensor<\d+x\d+xbf16", ttgir)
    wgd, wgmma, mma = n("warp_group_dot", ttgir), n(r"wgmma\.mma_async", ptx), n(r"mma\.sync", ptx)
    la, cl = n("local_alloc", ttgir), n("convert_layout", ttgir)
    return (f"regs={km.n_regs} spills={km.n_spills} warp_group_dot={wgd} ptx_wgmma={wgmma} mma_sync={mma} "
            f"local_alloc={la} (bf16 tiles {bf16_alloc}) convert_layout={cl}")


for name in ["int4_tok", "int4a_tok"]:
    for b, s in [(2, 1000), (8, 16384), (32, 16384), (1, 65536), (8, 65536)]:
        torch.manual_seed(0)
        q = torch.randn(b, HQ, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(b, HKV, s, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        if name == "int4a_tok":
            k, v = k + 3.0, v + 3.0  # off-centre values, where the zero point matters
        seq = torch.full((b,), s, device="cuda", dtype=torch.int32)
        kq, vq = quantize(k, name), quantize(v, name)
        line = f"{name:9s} B={b:3d} S={s:6d}:"
        try:
            cur = dt.decode_attention(q, kq, vq, seq, sm, fmt_name=name)
            ir_cur = ir_counts(dt.LAST_META["kernel"])
            swp = dt.decode_attention(q, kq, vq, seq, sm, fmt_name=name, swap=True)
            ir_swp = ir_counts(dt.LAST_META["kernel"])
        except Exception as e:
            print(f"{line} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        line += f" max|swap-cur| {(swp - cur).abs().max().item():.1e}"
        if b * s <= 600_000:
            ref = decode_attention_ref(q, kq, vq, seq, sm)
            line += f", err vs ref cur/swap {(cur - ref).abs().max().item():.1e}/{(swp - ref).abs().max().item():.1e}"
        if s >= 16384:  # timings for every large shape (before 11-10 this was an else, so nothing was timed)
            t_cur = timeit(lambda: dt.decode_attention(q, kq, vq, seq, sm, fmt_name=name))
            t_swp = timeit(lambda: dt.decode_attention(q, kq, vq, seq, sm, fmt_name=name, swap=True))
            line += f", ms cur/swap {t_cur:.3f}/{t_swp:.3f}"
        print(line, flush=True)
        if b == 2:
            print(f"   IR current: {ir_cur}\n   IR swap   : {ir_swp}", flush=True)
        del k, v, kq, vq
        torch.cuda.empty_cache()
