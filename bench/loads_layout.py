"""How fast can the packed INT4 cache be read at all? Clean loads-only kernel, two layouts, plus a copy floor.

The loads-only ablation in decode_triton (abl=1) reduces across rows every tile, which costs shared-memory
traffic; here the per-tile work is an element-wise add, reduced once at the end. Layouts:
  BHSD (ours): (B, H, S, D/2) bytes, one head's tokens contiguous (128 tokens = 8 KB).
  BSHD       : (B, S, H, D/2), tokens of one head 8 * 64 B apart (as in token-major / paged caches).
Reports GB/s for K+V bytes (+ scales) and the fraction of spec peak.
"""
import torch
import triton
import triton.language as tl

from kv4.formats import quantize
from kv4.kernels.decode_triton import _plan_splits

PEAK_GBS = {"H200": 4800, "H100": 3350, "A100": 2039}


@triton.jit
def _loads(k_ptr, ks_ptr, v_ptr, vs_ptr, out_ptr, S, CHUNK, NSPLIT, H_KV: tl.constexpr, HD: tl.constexpr,
           BLOCK_N: tl.constexpr, BSHD: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    start = pid_s * CHUNK
    end = tl.minimum(start + CHUNK, S)
    d2 = tl.arange(0, HD)
    if BSHD:
        dbase = pid_b.to(tl.int64) * S * H_KV * HD + pid_h * HD
        rstride = H_KV * HD
        sbase = pid_b.to(tl.int64) * S * H_KV + pid_h
        sstride = H_KV
    else:
        dbase = (pid_b.to(tl.int64) * H_KV + pid_h) * S * HD
        rstride = HD
        sbase = (pid_b.to(tl.int64) * H_KV + pid_h) * S
        sstride = 1
    acc = tl.zeros([BLOCK_N, HD], tl.int32)
    zs = tl.zeros([BLOCK_N], tl.float32)
    for n0 in range(start, end, BLOCK_N):
        rows = tl.minimum(n0 + tl.arange(0, BLOCK_N), end - 1)
        off = dbase + rows[:, None] * rstride + d2[None, :]
        acc += tl.load(k_ptr + off).to(tl.int32) + tl.load(v_ptr + off).to(tl.int32)
        zs += tl.load(ks_ptr + sbase + rows * sstride).to(tl.float32) + tl.load(vs_ptr + sbase + rows * sstride).to(tl.float32)
    tot = tl.sum(acc, 0).to(tl.float32) + tl.sum(zs, 0)
    tl.store(out_ptr + ((pid_b * H_KV + pid_h) * NSPLIT + pid_s) * HD + d2, tot)


gpu = torch.cuda.get_device_name()
peak = next((val for key, val in PEAK_GBS.items() if key in gpu), float("nan"))
print(gpu, flush=True)
HKV, D = 8, 128
for b, s in [(32, 16384), (8, 65536)]:
    torch.manual_seed(0)
    k = torch.randn(b, HKV, s, D, device="cuda", dtype=torch.bfloat16)
    kq, vq = quantize(k, "int4_tok"), quantize(torch.randn_like(k), "int4_tok")
    del k
    nbytes = sum(t.numel() * t.element_size() for t in (kq.data, vq.data, kq.scales, vq.scales))
    layouts = {
        "BHSD": (kq.data, kq.scales, vq.data, vq.scales),
        "BSHD": tuple(t.permute(0, 2, 1, 3).contiguous() for t in (kq.data, kq.scales, vq.data, vq.scales)),
    }
    print(f"== B={b} S={s}  K+V+scales {nbytes / 2**20:.0f} MiB", flush=True)
    dst = [torch.empty_like(kq.data), torch.empty_like(vq.data)]
    ms = triton.testing.do_bench(lambda: (dst[0].copy_(kq.data), dst[1].copy_(vq.data)), warmup=10, rep=40)
    cbytes = 2 * (kq.data.numel() + vq.data.numel())  # read + write
    print(f"  torch copy of the packed bytes: {ms:.3f} ms, {cbytes / ms / 1e6:.0f} GB/s ({cbytes / ms / 1e6 / peak:.0%} of peak, read+write)", flush=True)
    for lay, (kd, ks, vd, vs) in layouts.items():
        for bn in (64, 128, 256):
            for nw in (4, 8):
                chunk, nsplit = _plan_splits(b, HKV, s, bn, kd.device)
                out = torch.empty(b * HKV * nsplit * (D // 2), device="cuda", dtype=torch.float32)
                fn = lambda: _loads[(b, HKV, nsplit)](kd, ks, vd, vs, out, s, chunk, nsplit, H_KV=HKV, HD=D // 2,
                                                      BLOCK_N=bn, BSHD=lay == "BSHD", num_warps=nw)
                try:
                    ms = triton.testing.do_bench(fn, warmup=10, rep=40)
                except Exception as e:
                    print(f"  {lay} bn={bn} nw={nw}: failed ({type(e).__name__})", flush=True)
                    continue
                gbs = nbytes / ms / 1e6
                print(f"  {lay} bn={bn:3d} nw={nw}: {ms:.3f} ms, {gbs:.0f} GB/s ({gbs / peak:.0%} of peak)", flush=True)
