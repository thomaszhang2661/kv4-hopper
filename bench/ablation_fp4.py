"""11-18: removal ablation of our NVFP4 decode kernel (review item: which ingredients matter on Hopper), one job.

At each shape the split count is the best one for the full kernel and is then kept fixed. Rows:
  copy            device copy of the same K + V bytes (torch), the bandwidth ceiling for this many bytes
  nvfp4           full kernel (TMA producer, two warpgroups, register-A wgmma, write-time layout, bit placement)
  nvfp4_ilp       + ILP variant (reconstruct subtile 1 while subtile 0's wgmmas run)
  nvfp4_noreco    DIAG 1: no reconstruction (raw code words go to the tensor cores): what decoding costs
  nvfp4_nomma     DIAG 2: no matrix multiplies (reconstruction kept alive): what the tensor-core work costs
  mxfp4           MXFP4 (E8M0 per 32: half the scale bytes and multiplies)
  int4a           INT4 asymmetric (magic-number decode, 0.625 instructions per value; same bytes as MXFP4)
DIAG rows give wrong outputs by design; they are timings only. Llama-3.1-8B heads (32 / 8, d = 128).

    python bench/ablation_fp4.py --tag v1
"""
import argparse
import csv
import math
import os
from pathlib import Path

import torch

from bench.bench_decode import PEAK_GBS, quantize_chunked, timeit
from kv4.cuda import decode_fp4 as df
from kv4.cuda import decode_int4 as di
from kv4.cuda import merge as mg

HQ, HKV, D = 32, 8, 128
SM = 1 / math.sqrt(D)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="v1")
    p.add_argument("--shapes", default="1x65536,8x65536,32x16384,128x16384")
    args = p.parse_args()
    gpu = torch.cuda.get_device_name()
    peak = next((v for k, v in PEAK_GBS.items() if k in gpu), float("nan"))
    commit = os.environ.get("KV4_COMMIT", "unset")
    print(gpu, "commit", commit, flush=True)
    ef, e4, em = df.ext(), di.ext(), mg.ext()
    rows = []
    for shape in args.shapes.split(","):
        b, s = map(int, shape.split("x"))
        torch.manual_seed(0)
        q = torch.randn(b, HQ, D, device="cuda")
        k = torch.randn(b, HKV, s, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        base = {}

        def record(impl, ms, nbytes, cfg=""):
            gbs = nbytes / ms / 1e6
            base.setdefault("nvfp4", ms)
            rows.append(dict(gpu=gpu, commit=commit, tag=args.tag, impl=impl, config=cfg, batch=b, ctx=s,
                             ms=f"{ms:.4f}", gbs=f"{gbs:.0f}", peak_frac=f"{gbs / peak:.3f}",
                             vs_full=f"{ms / base['nvfp4']:.3f}"))
            print(f"{impl:14s} B={b:3d} S={s:6d} {cfg:10s} {ms:8.4f} ms {gbs:6.0f} GB/s ({gbs / peak:.0%}) "
                  f"{ms / base['nvfp4']:.2f}x full", flush=True)

        splits_list = [sp for sp in (1, 2, 4, 8, 16, 32, 64) if s % (sp * 256) == 0 and sp * HKV * b <= 32768]
        kq, vq = quantize_chunked(k, "nvfp4", chunk=1), quantize_chunked(v, "nvfp4", chunk=1)
        cache, smul, omul, nbk, nbv, mx = df.args_for(kq, vq, SM)
        qh = df.prep_q(q, HKV)
        del kq, vq
        nb = b * HKV * s * 2 * 72

        def run(sp, ilp=False, diag=0):
            return lambda: em.merge(*ef.decode_fp4(*cache, qh, smul, omul, nbk, nbv, sp, 2, False, ilp, diag))
        best = min((timeit(run(sp)), sp) for sp in splits_list)
        sp = best[1]
        record("nvfp4", best[0], nb, f"splits={sp}")
        record("nvfp4_ilp", timeit(run(sp, ilp=True)), nb, f"splits={sp}")
        record("nvfp4_noreco", timeit(run(sp, diag=1)), nb, f"splits={sp}")
        record("nvfp4_nomma", timeit(run(sp, diag=2)), nb, f"splits={sp}")
        src = torch.empty(nb, dtype=torch.uint8, device="cuda")
        dst = torch.empty_like(src)
        record("copy", timeit(lambda: dst.copy_(src)), 2 * nb, "read+write")  # bytes moved: read + write
        del src, dst, cache
        torch.cuda.empty_cache()
        kq, vq = quantize_chunked(k, "mxfp4", chunk=1), quantize_chunked(v, "mxfp4", chunk=1)
        cm, smm, omm, nbkm, nbvm, mxm = df.args_for(kq, vq, SM)
        del kq, vq
        record("mxfp4", timeit(lambda: em.merge(*ef.decode_fp4(*cm, qh, smm, omm, nbkm, nbvm, sp, 2, mxm))),
               b * HKV * s * 2 * 68, f"splits={sp}")
        del cm
        kq, vq = quantize_chunked(k, "int4a_tok", chunk=1), quantize_chunked(v, "int4a_tok", chunk=1)
        c4 = di.prepare(kq, vq)
        qp, qc = di.prep_q(q, HKV)
        del kq, vq
        record("int4a", timeit(lambda: em.merge(*e4.decode_int4(*c4, qp, qc, SM, sp, 2, True))),
               b * HKV * s * 2 * 68, f"splits={sp}")
        del c4, q, k, v
        torch.cuda.empty_cache()
    gtag = next((k for k in PEAK_GBS if k in gpu), gpu.replace(" ", "_"))
    out = Path(__file__).resolve().parent.parent / "results" / f"ablation_fp4_{gtag}_{args.tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
