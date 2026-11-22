"""BitDecoding's Hopper INT4 decode kernel against ours, on the same shapes and with the same timer.

BitDecoding (branch `hopper`) is an FA3-style kernel: a producer warp loads packed K/V with TMA,
the consumer warpgroups unpack them in registers, write FP16 back to shared memory and run wgmma
(QK: both operands from shared memory; PV: P from registers). Its format: K and V quantized per
token, asymmetric, one FP16 (scale, zero) pair per 128 values: 64 + 4 = 68 bytes per row, the same
as our int4a_tok. It is FP16 only (ours run in BF16; the bytes are the same).

1. Correctness on a small shape: BitDecoding's output against an FP32 reference on (a) the
   original K/V and (b) K/V fake-quantized per token (asymmetric, 15 levels), with our int4a_tok's
   error on the same data for scale.
2. Timings: num_splits from its own heuristic ("auto") and the best over a sweep, our int4a_tok,
   int4_tok and bf16 in the same job; bandwidth = 2 * B * H_kv * S * 68 bytes / time.

    python bench/bench_bitdecoding.py --batch 8 --ctx 16384 65536 --tag b8
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import torch

from baselines.bitdecoding.build import BITDECODING_COMMIT, load
from bench.bench_decode import quantize_chunked, timeit
from kv4.formats import FORMATS
from kv4.kernels import KERNELS

PEAK_GBS = {"H200": 4800, "H100": 3350}
BYTES_PER_ROW = 68  # 128 4-bit codes + one fp16 (scale, zero) pair
SPLITS = [2, 4, 8, 16, 32, 64]


def bd_inputs(k, v):
    """BSHD fp16 K/V -> BitDecoding's packed tensors (k-tensor mode, group 128)."""
    b, s, hkv, d = k.shape
    dev = k.device
    k_pack = torch.empty(b, s, hkv, d // 4, dtype=torch.uint16, device=dev)
    v_pack = torch.empty_like(k_pack)
    k_params = torch.empty(b, d // 128, hkv, s, dtype=torch.float32, device=dev)
    v_params = torch.empty_like(k_params)
    cu = torch.arange(0, (b + 1) * s, s, dtype=torch.int32, device=dev)
    return k_pack, k_params, v_pack, v_params, cu


def fake_quant_tok(x):
    """Per-token asymmetric 4-bit over the last dim (min/max, 15 levels, round to nearest)."""
    x = x.float()
    lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    scale = (hi - lo).clamp_min(1e-8) / 15
    return (torch.round((x - lo) / scale).clamp(0, 15) * scale + lo)


def ref_attn(q, k, v, sm):
    """q (B, Hq, D), k/v (B, S, Hkv, D) -> (B, Hq, D), fp32, GQA head h -> kv head h // (Hq/Hkv)."""
    g = q.shape[1] // k.shape[2]
    kk = k.float().permute(0, 2, 1, 3).repeat_interleave(g, 1)
    vv = v.float().permute(0, 2, 1, 3).repeat_interleave(g, 1)
    p = torch.softmax(torch.einsum("bhd,bhsd->bhs", q.float(), kk) * sm, -1)
    return torch.einsum("bhs,bhsd->bhd", p, vv)


def check(ext, hq, hkv, d):
    torch.manual_seed(0)
    b, s = 2, 4096
    sm = 1 / math.sqrt(d)
    q = torch.randn(b, 1, hq, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, s, hkv, d, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    kp, kpar, vp, vpar, cu = bd_inputs(k, v)
    ext.pack(k.reshape(b * s, hkv, d), kp, kpar, v.reshape(b * s, hkv, d), vp, vpar, cu, s)
    out = ext.fwd(q, k, kp, kpar, v, vp, vpar, sm, 0).float()[:, 0]
    ref = ref_attn(q[:, 0], k, v, sm)
    ref_fq = ref_attn(q[:, 0], fake_quant_tok(k), fake_quant_tok(v), sm)
    line = (f"check B={b} S={s}: BitDecoding max|err| vs original {(out - ref).abs().max().item():.2e} "
            f"(mean {(out - ref).abs().mean().item():.2e}), vs fake-quant {(out - ref_fq).abs().max().item():.2e}")
    if "int4a_tok" in KERNELS:
        kb, vb = k.permute(0, 2, 1, 3).bfloat16().contiguous(), v.permute(0, 2, 1, 3).bfloat16().contiguous()
        seq = torch.full((b,), s, device="cuda", dtype=torch.int32)
        o = KERNELS["int4a_tok"](q[:, 0].bfloat16(), quantize_chunked(kb, "int4a_tok"),
                                 quantize_chunked(vb, "int4a_tok"), seq, sm).float()
        line += f"; ours int4a_tok max|err| vs original {(o - ref).abs().max().item():.2e} (mean {(o - ref).abs().mean().item():.2e})"
    print(line, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32, 64])
    p.add_argument("--ctx", type=int, nargs="+", default=[4096, 16384, 65536])
    p.add_argument("--hq", type=int, default=32)
    p.add_argument("--hkv", type=int, default=8)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--mem-gb", type=float, default=60, help="skip shapes whose fp16 K+V exceed this")
    p.add_argument("--no-ours", action="store_true")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    gpu = torch.cuda.get_device_name()
    peak = next((v for k, v in PEAK_GBS.items() if k in gpu), float("nan"))
    commit = os.environ.get("KV4_COMMIT", "?")
    ext = load()
    print(gpu, "BitDecoding", BITDECODING_COMMIT, "torch", torch.__version__, flush=True)
    check(ext, args.hq, args.hkv, args.d)

    rows = []
    sm = 1 / math.sqrt(args.d)

    def record(impl, fmt, b, s, ms, splits, nbytes_row):
        gbs = 2 * b * args.hkv * s * nbytes_row / ms / 1e6
        rows.append(dict(gpu=gpu, commit=commit, impl=impl, format=fmt, splits=splits, batch=b, ctx=s,
                         ms=round(ms, 4), gbs=round(gbs), peak_frac=round(gbs / peak, 3)))
        print(f"B={b:3d} S={s:6d} {impl:18s} {fmt:9s} splits={splits!s:4s} {ms:8.3f} ms "
              f"{gbs:6.0f} GB/s ({gbs / peak:.0%})", flush=True)

    for b in args.batch:
        for s in args.ctx:
            if 2 * b * s * args.hkv * args.d * 2 / 2**30 > args.mem_gb:
                print(f"B={b} S={s}: skipped (fp16 K+V over {args.mem_gb} GB)", flush=True)
                continue
            torch.manual_seed(0)
            q = torch.randn(b, 1, args.hq, args.d, device="cuda", dtype=torch.float16)
            k = torch.randn(b, s, args.hkv, args.d, device="cuda", dtype=torch.float16)
            v = torch.randn_like(k)
            kp, kpar, vp, vpar, cu = bd_inputs(k, v)
            ext.pack(k.reshape(b * s, args.hkv, args.d), kp, kpar, v.reshape(b * s, args.hkv, args.d),
                     vp, vpar, cu, s)
            try:
                record("bitdecoding", "int4a_tok", b, s,
                       timeit(lambda: ext.fwd(q, k, kp, kpar, v, vp, vpar, sm, 0)), "auto", BYTES_PER_ROW)
                best = min((timeit(lambda n=n: ext.fwd(q, k, kp, kpar, v, vp, vpar, sm, n)), n) for n in SPLITS)
                record("bitdecoding_best", "int4a_tok", b, s, best[0], best[1], BYTES_PER_ROW)
            except Exception as e:
                print(f"B={b} S={s} bitdecoding: {type(e).__name__}: {str(e)[:200]}", flush=True)
            if not args.no_ours:
                qb = q[:, 0].bfloat16()
                kb = k.permute(0, 2, 1, 3).bfloat16().contiguous()
                del k
                vb = v.permute(0, 2, 1, 3).bfloat16().contiguous()
                del v
                seq = torch.full((b,), s, device="cuda", dtype=torch.int32)
                for name in ["bf16", "int4_tok", "int4a_tok"]:
                    if name not in KERNELS:
                        continue
                    kq, vq = (kb, vb) if name == "bf16" else (quantize_chunked(kb, name), quantize_chunked(vb, name))
                    record("ours_triton", name, b, s, timeit(lambda: KERNELS[name](qb, kq, vq, seq, sm)), "-",
                           FORMATS[name].bytes_per_row(args.d))
                    del kq, vq
                del kb, vb
            torch.cuda.empty_cache()
            write_csv(rows, args.tag, gpu)


def write_csv(rows, tag, gpu):
    if not rows:
        return
    gpu_tag = next((k for k in PEAK_GBS if k in gpu), gpu.replace(" ", "_"))
    out = Path("results") / (f"bench_bitdecoding_{gpu_tag}" + (f"_{tag}" if tag else "") + ".csv")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
