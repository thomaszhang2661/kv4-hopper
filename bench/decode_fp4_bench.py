"""Hopper NVFP4 decode kernel (kv4/cuda/decode_fp4.cu, 11-10 night): correctness, then speed against FlashInfer fa2
NVFP4, FP8, BF16 and our CUDA INT4 kernel, all in the same job at the same B and S.

Bytes for our NVFP4 kernel: K and V codes plus E4M3 scales = 2 x 72 per token and kv head (tensor scales ignored).

    python bench/decode_fp4_bench.py --tag v1
"""
import argparse
import csv
import math
import os
from pathlib import Path

import torch

from bench.bench_decode import PEAK_GBS, quantize_chunked, timeit
from bench.decode_int4_bench import flashinfer_fns
from kv4.cuda import decode_fp4 as df
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


def run(e, a, qh, splits, nst, ilp=False):
    cache, smul, omul, nbk, nbv, mx = a
    return e.decode_fp4(*cache, qh, smul, omul, nbk, nbv, splits, nst, mx, ilp)


BYTES = {"nvfp4": 72, "mxfp4": 68, "mxfp4_uq": 68}  # bytes per (token, head) for K or V


def check(e, fmts):
    ok = True
    for fmt, (b, s, splits, nst), ilp in [(f, c, i) for f in fmts for i in (False, True) for c in
                                          [(2, 4096, 1, 2), (2, 4096, 4, 2), (1, 2048, 2, 2), (3, 1536, 3, 2),
                                           (1, 16384, 1, 2)]]:
        q, _, _, kq, vq = make(b, s, fmt)
        a = df.args_for(kq, vq, SM)
        qh = df.prep_q(q, HKV)
        o, ml = run(e, a, qh, splits, nst, ilp)
        out = di.combine(o, ml)
        ref = df.reference(q, kq, vq, SM)
        err = (out - ref).abs().max().item() / ref.abs().max().item()
        bad = (~torch.isfinite(out)).sum().item()
        good = err < 2e-3 and bad == 0
        ok &= good
        print(f"check {fmt} ilp={int(ilp)} B={b} S={s} splits={splits} nst={nst}: max err {err:.2e} (rel. to max |out|), "
              f"non-finite {bad} -> {'ok' if good else 'FAIL'}", flush=True)
        if not good:
            print("  kernel", out[0, 0, 0, :6].tolist(), "\n  ref   ", ref[0, 0, 0, :6].tolist(), flush=True)
    return ok


def flashinfer_nvfp4(b, s, page=16):
    """FlashInfer fa2 NVFP4 decode (tensor cores), random codes and unit scales, as bench/ncu_target.py."""
    from bench.bench_flashinfer import make_wrapper, page_table
    indptr, indices, last = page_table([s] * b, page)
    npages = int(indptr[-1])
    q = torch.randn(b, HQ, D, device="cuda", dtype=torch.bfloat16)
    shape = (npages, HKV, page, D)
    one = torch.tensor(1.0).to(torch.float8_e4m3fn).view(torch.uint8).item()
    sf = shape[:-1] + (D // 16,)
    cache = (torch.randint(0, 256, shape[:-1] + (D // 2,), device="cuda", dtype=torch.uint8),
             torch.randint(0, 256, shape[:-1] + (D // 2,), device="cuda", dtype=torch.uint8))
    sfs = (torch.full(sf, one, device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn),
           torch.full(sf, one, device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn))
    w, _ = make_wrapper("fa2", True, indptr, indices, last, HQ, HKV, D, page, torch.uint8)
    return lambda: w.run(q, cache, k_scale=0.01, v_scale=0.01, kv_cache_sf=sfs), b * HKV * s * 2 * 72  # noqa: E731


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="v1")
    p.add_argument("--shapes", default="32x16384,8x65536,64x16384,128x16384,16x32768,1x65536")
    p.add_argument("--splits", default="1,2,4,8,16,32")
    p.add_argument("--nst", default="2")
    p.add_argument("--ilp", default="0")
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--fmts", default="nvfp4,mxfp4")
    args = p.parse_args()
    gpu = torch.cuda.get_device_name()
    peak = next((v for k, v in PEAK_GBS.items() if k in gpu), float("nan"))
    commit = os.environ.get("KV4_COMMIT", "unset")
    print(gpu, "commit", commit, flush=True)
    e = df.ext()
    fmts = args.fmts.split(",")
    if not check(e, fmts):
        print("correctness failed; timing anyway", flush=True)
    rows = []

    def record(impl, b, s, ms, nbytes, extra=""):
        gbs = nbytes / ms / 1e6
        rows.append(dict(gpu=gpu, commit=commit, tag=args.tag, impl=impl, config=extra, batch=b, ctx=s,
                         ms=f"{ms:.4f}", gbs=f"{gbs:.0f}", peak_frac=f"{gbs / peak:.3f}"))
        print(f"{impl:18s} B={b:3d} S={s:6d} {extra:14s} {ms:8.4f} ms {gbs:6.0f} GB/s ({gbs / peak:.0%})", flush=True)

    for shape in args.shapes.split(","):
        b, s = map(int, shape.split("x"))
        q = k = v = None
        for fmt in fmts:
            q, k, v, kq, vq = make(b, s, fmt)
            a = df.args_for(kq, vq, SM)
            qh = df.prep_q(q, HKV)
            nbytes = b * HKV * s * 2 * BYTES[fmt]
            best = None
            for ilp in map(int, args.ilp.split(",")):
                for splits in map(int, args.splits.split(",")):
                    if s % (splits * 256):
                        continue
                    for nst in map(int, args.nst.split(",")):
                        ms = timeit(lambda: run(e, a, qh, splits, nst, bool(ilp)))
                        record(f"cuda_{fmt}" + ("_ilp" if ilp else ""), b, s, ms, nbytes, f"splits={splits},nst={nst}")
                        if not ilp and (best is None or ms < best[0]):
                            best = (ms, splits, nst)
            if best[1] > 1:
                o, ml = run(e, a, qh, best[1], best[2])
                ms_c = timeit(lambda: (run(e, a, qh, best[1], best[2]), di.combine(o, ml)))
                record(f"cuda_{fmt}+merge", b, s, ms_c, nbytes, f"splits={best[1]},nst={best[2]}")
            print(f"BEST cuda_{fmt} B={b} S={s}: splits={best[1]} nst={best[2]} {best[0]:.4f} ms "
                  f"({nbytes / best[0] / 1e6 / peak:.0%})", flush=True)
            del a, kq, vq
        if not args.no_baselines:
            try:
                fn, nb = flashinfer_nvfp4(b, s)
                record("flashinfer_nvfp4", b, s, timeit(fn), nb)
                for name, (fn, nb) in flashinfer_fns(b, s, k, v).items():
                    record(name, b, s, timeit(fn), nb)
            except RuntimeError as ex:
                print(f"flashinfer skipped: {str(ex).splitlines()[0][:120]}", flush=True)
            # our CUDA INT4 at the same shape, for the 4-bit comparison
            kq4 = quantize_chunked(k, "int4a_tok", chunk=1)
            vq4 = quantize_chunked(v, "int4a_tok", chunk=1)
            c4 = di.prepare(kq4, vq4)
            qp, qc = di.prep_q(q, HKV)
            e4 = di.ext()
            best4 = min((timeit(lambda: e4.decode_int4(*c4, qp, qc, SM, sp, 2, True)), sp)
                        for sp in (1, 2, 4, 8, 16, 32) if s % (sp * 256) == 0)
            record("cuda_int4a", b, s, best4[0], b * HKV * s * 2 * 68, f"splits={best4[1]},nst=2")
            del kq4, vq4, c4
        del q, k, v
        torch.cuda.empty_cache()
    gtag = next((k for k in PEAK_GBS if k in gpu), gpu.replace(" ", "_"))
    out = Path(__file__).resolve().parent.parent / "results" / f"decode_fp4_{gtag}_{args.tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
