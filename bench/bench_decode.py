"""Decode-attention speed: time per call and achieved bandwidth, over batch size x context length.

Every kernel in kv4.kernels.KERNELS is timed next to two baselines:
  sdpa    : torch's scaled_dot_product_attention on the BF16 cache (what you beat first)
  copy_kv : device-to-device copy of K and V, counted as read + write bytes: the
            practical bandwidth ceiling (a plain sum does not reach it)
Bandwidth = bytes of K and V the format stores (bytes_per_row) / time; the
roofline says a memory-bound kernel cannot beat bytes / peak bandwidth. Small
shapes are latency-bound instead (a launch costs ~10 us): look at ms there.

    python bench/bench_decode.py                      # Llama-3.1-8B heads: H_q 32, H_kv 8, D 128
    python bench/bench_decode.py --batch 1 8 --ctx 4096 --tag quick
"""

from __future__ import annotations

import argparse
import csv
import os
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from kv4.formats import FORMATS, QTensor, quantize
from kv4.kernels import KERNELS

PEAK_GBS = {"H200": 4800, "H100": 3350, "A100": 2039}  # HBM spec sheets (SXM)


def timeit(fn, warmup=5, reps=50) -> float:
    """Median milliseconds per call (CUDA events; the cache flush keeps K, V out of L2)."""
    flush = torch.zeros(256 * 2**20, dtype=torch.int8, device="cuda")  # > H200's 60 MB L2
    # 11-21: KV4_FLUSH=read evicts L2 by reading the buffer, so L2 holds clean lines when the timed kernel starts.
    # The default (write, every paper table so far) leaves ~60 MB of dirty lines that the timed kernel's reads must
    # write back first, a cost common to every kernel timed this way (xqa_vs_decode_fp4.md, E1).
    read_flush = os.environ.get("KV4_FLUSH", "write") == "read"
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        if read_flush:
            flush.max()
        else:
            flush.zero_()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return sorted(times)[len(times) // 2]


def quantize_chunked(x: torch.Tensor, name: str, chunk: int = 2) -> QTensor:
    """kv4.formats.quantize in batch chunks: the torch reference needs ~10x the cache size as scratch.

    For a *speed* benchmark only: tensor-wide scales (fp8, nvfp4) are taken from the first chunk,
    so the values are not exactly what quantize(x) would give; the bytes and layout are identical.
    """
    parts = [quantize(x[i:i + chunk], name) for i in range(0, x.shape[0], chunk)]
    cat = lambda fs: None if fs[0] is None else torch.cat(fs)
    return QTensor(parts[0].fmt, cat([t.data for t in parts]), cat([t.scales for t in parts]),
                   parts[0].tensor_scale, cat([t.zeros for t in parts]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32, 64])
    p.add_argument("--ctx", type=int, nargs="+", default=[1024, 4096, 16384, 65536])
    p.add_argument("--hq", type=int, default=32)
    p.add_argument("--hkv", type=int, default=8)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--mem-gb", type=float, default=100, help="skip shapes whose BF16 cache exceeds this")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    gpu = torch.cuda.get_device_name()
    peak = next((v for k, v in PEAK_GBS.items() if k in gpu), float("nan"))
    sm_scale = 1 / math.sqrt(args.d)
    rows = []
    for b in args.batch:
        for s in args.ctx:
            bf16_bytes = 2 * b * args.hkv * s * args.d * 2
            if bf16_bytes > args.mem_gb * 2**30:
                continue
            torch.manual_seed(0)
            q = torch.randn(b, args.hq, args.d, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(b, args.hkv, s, args.d, device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)
            seq_lens = torch.full((b,), s, device="cuda", dtype=torch.int32)
            k2, v2 = torch.empty_like(k), torch.empty_like(v)
            runs = {  # name: (format, function, bytes moved per byte of K and V)
                "sdpa": ("bf16", lambda: F.scaled_dot_product_attention(q[:, :, None], k, v, enable_gqa=True), 1),
                "copy_kv": ("bf16", lambda: (k2.copy_(k), v2.copy_(v)), 2),
            }
            for name, kernel in KERNELS.items():
                kq, vq = (k, v) if name == "bf16" else (quantize_chunked(k, name), quantize_chunked(v, name))
                runs[f"kernel_{name}"] = (name, lambda kernel=kernel, kq=kq, vq=vq: kernel(q, kq, vq, seq_lens, sm_scale), 1)
            for run, (fmt, fn, traffic) in runs.items():
                try:
                    ms = timeit(fn)
                except NotImplementedError:
                    continue
                except Exception as e:  # e.g. fp8 / nvfp4 do not compile on A100
                    print(f"B={b} S={s} {run}: skipped ({type(e).__name__})", flush=True)
                    continue
                nbytes = traffic * 2 * b * args.hkv * s * FORMATS[fmt].bytes_per_row(args.d)
                gbs = nbytes / ms / 1e6
                rows.append(dict(gpu=gpu, commit=os.environ.get("KV4_COMMIT", "unknown"), run=run, format=fmt, batch=b, ctx=s, ms=round(ms, 4),
                                 kv_mb=round(nbytes / 2**20, 1), gbs=round(gbs), peak_frac=round(gbs / peak, 3)))
                print(f"B={b:3d} S={s:6d} {run:14s} {ms:8.3f} ms {gbs:7.0f} GB/s ({gbs / peak:.0%} of peak)", flush=True)
            del k, v, k2, v2, runs
            torch.cuda.empty_cache()
            write_csv(rows, args.tag, gpu)  # keep partial results if a later shape fails

    write_csv(rows, args.tag, gpu)


def write_csv(rows, tag, gpu):
    gpu_tag = next((k for k in PEAK_GBS if k in gpu), gpu.replace(" ", "_"))
    out = Path("results") / (f"bench_decode_{gpu_tag}" + (f"_{tag}" if tag else "") + ".csv")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
