"""K-side CUDA prototype (kv4/cuda/kside.cu, 11-10): correctness, then bandwidth vs splits and pipeline stages.

Gate: K side above ~60% of H200 bandwidth at B=32, S=16K. Bytes = codes + fp16 scales = 66 per token row.

    python bench/kside_proto.py --tag v1
"""
import argparse
import csv
import math
import os
from pathlib import Path

import torch

from bench.bench_decode import PEAK_GBS, quantize_chunked, timeit
from kv4.cuda import kside
from kv4.formats import quantize

HQ, HKV, D = 32, 8, 128
SM = 1 / math.sqrt(D)
NAMES = {0: "v1", 1: "v2", 2: "v4", 3: "v3"}


def inputs(b, s, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(b, HQ, D, device="cuda")
    k = torch.randn(b, HKV, s, D, device="cuda", dtype=torch.bfloat16)
    qt = quantize(k, "int4_tok") if b * s <= 65536 else quantize_chunked(k, "int4_tok", chunk=1)
    del k
    kp = kside.pack_kside(qt.data)
    ks = qt.scales[..., 0].contiguous()
    qp, k0 = kside.prep_q(q, HKV)
    return q, qt, kp, ks, qp, k0


def call(e, var, kp, ks, qp, k0, splits, nst, scores=None, mode=0):
    """var 0 v1, 1 v2, 2 v4 (lean): grid = (splits, HKV, B). var 3 = v3: v1's compute with persistent CTAs."""
    if var == 3:
        return e.kside_persistent(kp, ks, kside.core_matrix(qp) if qp.dim() == 4 else qp, k0, SM, splits, nst, scores)
    return e.kside(kp, ks, qp, k0, SM, splits, nst, scores, var, mode)


def check(e):
    ok = True
    for (b, s, splits, nst), var in [(c, v) for c in [(2, 4096, 1, 4), (2, 4096, 4, 2), (1, 2048, 2, 6), (3, 1536, 3, 3)]
                                      for v in (0, 1, 2, 3)]:
        q, qt, kp, ks, qp, k0 = inputs(b, s)
        scores = torch.full((b, HKV, 4, s), float("nan"), device="cuda")
        ml = call(e, var, kp, ks, qp, k0, splits, nst, scores)
        torch.cuda.synchronize()
        ref_s, ref_ml = kside.reference(q, qt.data, qt.scales, SM, splits)
        err = (scores - ref_s).abs().max().item() / ref_s.abs().max().item()
        merr = (ml[..., 0] - ref_ml[..., 0]).abs().max().item()
        lerr = ((ml[..., 1] - ref_ml[..., 1]).abs() / ref_ml[..., 1]).max().item()
        nan = torch.isnan(scores).sum().item()
        good = err < 1e-3 and merr < 1e-2 and lerr < 1e-3 and nan == 0
        ok &= good
        print(f"check {NAMES[var]} B={b} S={s} splits={splits} nst={nst}: score err {err:.2e} (rel. to max), m err {merr:.2e}, "
              f"l rel err {lerr:.2e}, unwritten {nan} -> {'ok' if good else 'FAIL'}", flush=True)
        if not good:
            print("  kernel", scores[0, 0, 0, :6].tolist(), "\n  ref   ", ref_s[0, 0, 0, :6].tolist(), flush=True)
    return ok


def diag(e, peak):
    """Where does a CTA's time go? mode 1 = load only, mode 2 = compute only (no loads). Per-CTA GB/s with 1 CTA per
    SM (B=8, S=64K, splits=1: 64 CTAs) and the full shape (B=32, S=16K)."""
    print("== diagnosis: mode 0 normal, 1 load only, 2 compute only", flush=True)
    for b, s, splits_list in [(8, 65536, [1]), (32, 16384, [1, 4])]:
        _, _, kp, ks, qp, k0 = inputs(b, s)
        nbytes = b * HKV * s * 66
        for splits in splits_list:
            for var in (0, 1):
                for nst in (2, 4):
                    for mode in (0, 1, 2):
                        ms = timeit(lambda: e.kside(kp, ks, qp, k0, SM, splits, nst, None, var, mode))
                        gbs = nbytes / ms / 1e6
                        ctas = splits * HKV * b
                        print(f"diag B={b:2d} S={s:6d} splits={splits} v{var + 1} nst={nst} mode={mode}: {ms:.4f} ms "
                              f"{gbs:6.0f} GB/s ({gbs / peak:.0%}); per CTA {gbs / min(ctas, 132):.1f} GB/s "
                              f"if one wave", flush=True)
        del kp, ks
        torch.cuda.empty_cache()


def steady(e, peak, b=128, s=16384):
    """v1 and the diagnosis switches on a large problem (B=128, S=16K: 1.1 GB of K) to amortize launch, pipeline
    fill and drain."""
    print(f"== steady state B={b} S={s}", flush=True)
    _, _, kp, ks, qp, k0 = inputs(b, s)
    nbytes = b * HKV * s * 66
    for var in (0, 2):
        for splits in (1, 2, 4):
            for nst in (2, 3, 4):
                for mode in (0, 1, 2):
                    ms = timeit(lambda: e.kside(kp, ks, qp, k0, SM, splits, nst, None, var, mode))
                    gbs = nbytes / ms / 1e6
                    print(f"steady {NAMES[var]} B={b} S={s} splits={splits} nst={nst} mode={mode}: {ms:.4f} ms "
                          f"{gbs:6.0f} GB/s ({gbs / peak:.0%})", flush=True)
    del kp, ks
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="v1")
    p.add_argument("--shapes", default="32x16384,8x65536,1x65536,64x16384")
    p.add_argument("--vars", default="0,2")  # v1 and v4
    p.add_argument("--splits", default="1,2,4,8,16,32,64,128")
    p.add_argument("--nst", default="2,3,4,6")
    p.add_argument("--diag", action="store_true")
    p.add_argument("--steady", action="store_true")
    args = p.parse_args()
    gpu = torch.cuda.get_device_name()
    peak = next((v for k, v in PEAK_GBS.items() if k in gpu), float("nan"))
    commit = os.environ.get("KV4_COMMIT", "unset")
    print(gpu, "commit", commit, flush=True)
    e = kside.ext()
    if not check(e):
        print("correctness failed; timing anyway, to see the speed of the structure", flush=True)
    if args.diag:
        diag(e, peak)
    if args.steady:
        steady(e, peak)
        if not args.shapes:
            return
    rows = []
    for shape in filter(None, args.shapes.split(",")):
        b, s = map(int, shape.split("x"))
        _, _, kp, ks, qp, k0 = inputs(b, s)
        qcm = kside.core_matrix(qp)
        nbytes = b * HKV * s * 66
        for var in map(int, args.vars.split(",")):
            best = None
            for splits in map(int, args.splits.split(",")):
                if s % (splits * 256) or splits * HKV * b > 32768:
                    continue
                for nst in map(int, args.nst.split(",")):
                    ms = timeit(lambda: call(e, var, kp, ks, qcm if var == 3 else qp, k0, splits, nst))
                    gbs = nbytes / ms / 1e6
                    rows.append(dict(gpu=gpu, commit=commit, tag=args.tag, var=NAMES[var], batch=b, ctx=s,
                                     splits=splits, nst=nst, ctas=splits * HKV * b, ms=f"{ms:.4f}", gbs=f"{gbs:.0f}",
                                     peak_frac=f"{gbs / peak:.3f}"))
                    print(f"{NAMES[var]} B={b:3d} S={s:6d} splits={splits:3d} nst={nst} ctas={splits * HKV * b:5d}: "
                          f"{ms:.4f} ms {gbs:6.0f} GB/s ({gbs / peak:.0%})", flush=True)
                    if best is None or ms < best[0]:
                        best = (ms, splits, nst, gbs)
            print(f"BEST {NAMES[var]} B={b} S={s}: splits={best[1]} nst={best[2]} {best[0]:.4f} ms {best[3]:.0f} GB/s "
                  f"({best[3] / peak:.0%} of peak)", flush=True)
        del kp, ks
        torch.cuda.empty_cache()
    gtag = next((k for k in PEAK_GBS if k in gpu), gpu.replace(" ", "_"))
    out = Path(__file__).resolve().parent.parent / "results" / f"kside_proto_{gtag}_{args.tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
