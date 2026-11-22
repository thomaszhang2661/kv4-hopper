"""Hopper INT4 decode kernel (kv4/cuda/decode_int4.cu, 11-10): correctness, then speed against FlashInfer fa2 BF16 and
FP8 (tensor cores) and our Triton int4a_tok, all in the same job at the same B and S.

Bytes for the CUDA kernel: K and V codes plus (scale, zero) fp16 = 2 x 68 per token and kv head.

    python bench/decode_int4_bench.py --tag v1
"""
import argparse
import csv
import math
import os
from pathlib import Path

import torch

from bench.bench_decode import PEAK_GBS, quantize_chunked, timeit
from kv4.cuda import decode_int4 as di
from kv4.formats import quantize

HQ, HKV, D = 32, 8, 128
SM = 1 / math.sqrt(D)


def make(b, s, fmt, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(b, HQ, D, device="cuda")
    k = torch.randn(b, HKV, s, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    small = b * s <= 65536
    kq = quantize(k, fmt) if small else quantize_chunked(k, fmt, chunk=1)
    vq = quantize(v, fmt) if small else quantize_chunked(v, fmt, chunk=1)
    return q, k, v, kq, vq


def bf16p_error(q, kq, vq, ref):
    """Error of an ordinary kernel that stores P in bf16 (as FlashAttention-style kernels do), for context."""
    from kv4.cuda import kside
    from kv4.formats import dequantize
    qp, _ = kside.prep_q(q, kq.data.shape[1])
    qe = qp[:, :, :4].float() * kside.dim_factor(q.device)
    k, v = dequantize(kq).float(), dequantize(vq).float()
    p = torch.softmax(torch.einsum("bhgd,bhsd->bhgs", qe, k) * SM, -1)
    out = torch.einsum("bhgs,bhsd->bhgd", p.to(torch.bfloat16).float(), v)
    return (out - ref).abs().max().item() / ref.abs().max().item()


def check(e):
    ok = True
    for fmt in ("int4a_tok", "int4_tok"):
        for b, s, splits, nst in [(2, 4096, 1, 2), (2, 4096, 4, 3), (1, 2048, 2, 2), (3, 1536, 3, 3), (1, 16384, 1, 2)]:
            q, _, _, kq, vq = make(b, s, fmt)
            cache = di.prepare(kq, vq)
            qp, qc = di.prep_q(q, HKV)
            ref = di.reference(q, kq, vq, SM)
            eb = bf16p_error(q, kq, vq, ref)
            for center in (True, False):
                o, ml = e.decode_int4(*cache, qp, qc, SM, splits, nst, center)
                out = di.combine(o, ml)
                err = (out - ref).abs().max().item() / ref.abs().max().item()
                bad = (~torch.isfinite(out)).sum().item()
                good = err < 2e-3 and bad == 0
                ok &= good or not center
                print(f"check {fmt} center={int(center)} B={b} S={s} splits={splits} nst={nst}: max err {err:.2e} "
                      f"(rel. to max |out|; bf16-P kernel {eb:.2e}), non-finite {bad} -> {'ok' if good else 'FAIL'}",
                      flush=True)
            if not good:
                print("  kernel", out[0, 0, 0, :6].tolist(), "\n  ref   ", ref[0, 0, 0, :6].tolist(), flush=True)
    return ok


def flashinfer_fns(b, s, k, v, page=16):
    """FlashInfer fa2 decode with tensor cores, HND (as bench/bench_flashinfer.py): BF16 and FP8; page 16 by default."""
    from bench.bench_flashinfer import make_wrapper, page_table
    indptr, indices, last = page_table([s] * b, page)
    q = torch.randn(b, HQ, D, device="cuda", dtype=torch.bfloat16)

    def paged(x):  # (B, HKV, S, D) -> (pages, HKV, page, D) in page order
        return x.reshape(b, HKV, s // page, page, D).permute(0, 2, 1, 3, 4).reshape(-1, HKV, page, D).contiguous()

    out = {}
    kb, vb = paged(k), paged(v)
    w, _ = make_wrapper("fa2", True, indptr, indices, last, HQ, HKV, D, page, torch.bfloat16)
    out["flashinfer_bf16"] = (lambda: w.run(q, (kb, vb)), b * HKV * s * 2 * 256)
    k8, v8 = kb.to(torch.float8_e4m3fn), vb.to(torch.float8_e4m3fn)
    w8, _ = make_wrapper("fa2", True, indptr, indices, last, HQ, HKV, D, page, torch.float8_e4m3fn)
    out["flashinfer_fp8"] = (lambda: w8.run(q, (k8, v8), k_scale=1.0, v_scale=1.0), b * HKV * s * 2 * 128)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="v1")
    p.add_argument("--shapes", default="32x16384,8x65536,64x16384,128x16384,16x32768,1x65536")
    p.add_argument("--splits", default="1,2,4,8,16,32")
    p.add_argument("--nst", default="2,3")
    p.add_argument("--no-baselines", action="store_true")
    args = p.parse_args()
    gpu = torch.cuda.get_device_name()
    peak = next((v for k, v in PEAK_GBS.items() if k in gpu), float("nan"))
    commit = os.environ.get("KV4_COMMIT", "unset")
    print(gpu, "commit", commit, flush=True)
    e = di.ext()
    if not check(e):
        print("correctness failed; timing anyway", flush=True)
    rows = []

    def record(impl, b, s, ms, nbytes, extra=""):
        gbs = nbytes / ms / 1e6
        rows.append(dict(gpu=gpu, commit=commit, tag=args.tag, impl=impl, config=extra, batch=b, ctx=s,
                         ms=f"{ms:.4f}", gbs=f"{gbs:.0f}", peak_frac=f"{gbs / peak:.3f}"))
        print(f"{impl:18s} B={b:3d} S={s:6d} {extra:14s} {ms:8.4f} ms {gbs:6.0f} GB/s ({gbs / peak:.0%})", flush=True)

    for shape in args.shapes.split(","):
        b, s = map(int, shape.split("x"))
        q, k, v, kq, vq = make(b, s, "int4a_tok")
        cache = di.prepare(kq, vq)
        qp, qc = di.prep_q(q, HKV)
        nbytes = b * HKV * s * 2 * 68
        best = None
        for center in (True, False):
            for splits in map(int, args.splits.split(",")):
                if s % (splits * 256):
                    continue
                for nst in map(int, args.nst.split(",")):
                    ms = timeit(lambda: e.decode_int4(*cache, qp, qc, SM, splits, nst, center))
                    record("cuda_int4a" + ("" if center else "_nocenter"), b, s, ms, nbytes, f"splits={splits},nst={nst}")
                    if center and (best is None or ms < best[0]):
                        best = (ms, splits, nst)
        # the split merge is a separate step when splits > 1; time it at the best config for honesty
        o, ml = e.decode_int4(*cache, qp, qc, SM, best[1], best[2])
        ms_c = timeit(lambda: (e.decode_int4(*cache, qp, qc, SM, best[1], best[2]), di.combine(o, ml)))
        record("cuda_int4a+merge", b, s, ms_c, nbytes, f"splits={best[1]},nst={best[2]}")
        del cache
        if not args.no_baselines:
            try:  # 11-10: FlashInfer's JIT cache was built on the H200 nodes (driver 570); the H100 node (545) rejects it
                for name, (fn, nb) in flashinfer_fns(b, s, k, v).items():
                    record(name, b, s, timeit(fn), nb)
            except RuntimeError as ex:
                print(f"flashinfer skipped: {str(ex).splitlines()[0][:120]}", flush=True)
            if b * s <= 32 * 16384:
                from kv4.kernels import KERNELS
                seq = torch.full((b,), s, device="cuda", dtype=torch.int32)
                kq2, vq2 = quantize_chunked(k, "int4a_tok"), quantize_chunked(v, "int4a_tok")
                qb = q.to(torch.bfloat16)
                record("triton_int4a", b, s, timeit(lambda: KERNELS["int4a_tok"](qb, kq2, vq2, seq, SM)), nbytes)
                del kq2, vq2
        print(f"BEST cuda_int4a B={b} S={s}: splits={best[1]} nst={best[2]} {best[0]:.4f} ms "
              f"({nbytes / best[0] / 1e6 / peak:.0%})", flush=True)
        del q, k, v, kq, vq
        torch.cuda.empty_cache()
    gtag = next((k for k in PEAK_GBS if k in gpu), gpu.replace(" ", "_"))
    out = Path(__file__).resolve().parent.parent / "results" / f"decode_int4_{gtag}_{args.tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
