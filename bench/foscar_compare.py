"""Our Hopper decode kernels against FOSCAR (arXiv 2611.09722, github.com/FutureMLS-Lab/FOSCAR, MIT) on one GPU, in one
process, on the same K/V (11-12).

FOSCAR INT4 here is its own benchmark configuration (benchmarks/kernels.py build_ours): token-wise asymmetric INT4, one
group per (token, head) of 128 values, 64-token pages in random order. That is 68 bytes per token and head, the same
bytes as our int4a_tok (and 4 fewer than NVFP4's 72).

Columns: FlashInfer fa2 BF16 / FP8 (faster of 16- and 64-token pages), FOSCAR INT4, ours INT4 (dense kernel + merge),
ours NVFP4 (paged kernel + merge), ours NVFP4 with the full accuracy recipe (fused q rotation, merge.prep_q_rot + paged kernel over S - 4
tokens + merge_sink with 4 BF16 sink tokens; mean removal is write-side only).

Two timers, both reported:
  warm: bench.bench_decode.timeit (CUDA events around repeated calls, L2 warm, host dispatch included);
  cold: FOSCAR's benchmarks/timer.measure (sum of the call's kernel times from CUPTI, 1 GiB L2 flush before each call,
        host dispatch excluded), i.e. the number its README and paper report.

Correctness: FOSCAR's output against exact attention over its own dequantized cache, and against BF16 attention.

    python bench/foscar_compare.py --foscar /scratch/$USER/ext/FOSCAR
"""
import argparse
import csv
import math
import os
import sys
from pathlib import Path

import torch

from bench.bench_decode import PEAK_GBS, quantize_chunked, timeit
from bench.decode_int4_bench import flashinfer_fns
from kv4.cuda import decode_fp4 as df
from kv4.cuda import decode_int4 as di
from kv4.cuda import merge as mg
from kv4.formats import rotate

HQ, HKV, D, NS, PAGE = 32, 8, 128, 4, 64
SM = 1 / math.sqrt(D)
SHAPES = ("1x16384,1x65536,1x131072,4x32768,8x65536,16x32768,16x131072,32x4096,32x16384,64x16384,128x16384,"
          "1x262144,1x524288,8x131072,8x262144")


def foscar_fn(foscar, k, v, q, b, s, check):
    """FOSCAR INT4 as its benchmark builds it. k, v (B, HKV, S, D) bf16; q (B, HQ, D) bf16."""
    pages = s // PAGE
    dev = k.device
    layout = foscar.QuantKVLayout.create(bits=4, head_dim=D, num_kv_heads=HKV, k_quant=("token", D),
                                         v_quant=("token", D), device=dev)
    cache = layout.allocate(b * pages)
    g = torch.Generator(device=dev).manual_seed(0)
    indices = torch.randperm(b * pages, device=dev, generator=g).to(torch.int32)
    physical = indices.view(b, pages).long()
    slots = (physical[:, :, None] * PAGE + torch.arange(PAGE, device=dev)[None, None]).reshape(-1)
    tok = lambda x: x.permute(0, 2, 1, 3).reshape(b * s, HKV, D).contiguous()  # noqa: E731
    cache.append(tok(k), tok(v), slots)
    indptr = torch.arange(b + 1, device=dev, dtype=torch.int32) * pages
    seq_lens = torch.full((b,), s, device=dev, dtype=torch.int32)
    w = foscar.PagedDecodeWrapper(torch.empty(256 << 20, dtype=torch.uint8, device=dev))
    w.plan(indptr, indices, seq_lens, num_qo_heads=HQ, layout=layout)
    out = torch.empty(b, HQ, D, device=dev, dtype=torch.bfloat16)

    def fn():
        w.run(q, cache, out=out, sm_scale=SM)
        return out

    if check:
        fn()
        torch.cuda.synchronize()
        qe = q.float().reshape(b, HKV, HQ // HKV, D)
        p = torch.softmax(torch.einsum("bhgd,bhsd->bhgs", qe, k.float()) * SM, -1)
        ref = torch.einsum("bhgs,bhsd->bhgd", p, v.float()).reshape(b, HQ, D)
        err = ((out.float() - ref).norm(dim=-1) / ref.norm(dim=-1)).mean().item()
        print(f"check FOSCAR INT4 B={b} S={s}: relative error vs BF16 attention {err:.4f} "
              f"(INT4 quantization error, about 0.1 expected for random K/V)", flush=True)
    return fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--foscar", required=True, help="FOSCAR checkout (for benchmarks/timer.py)")
    p.add_argument("--shapes", default=SHAPES)
    p.add_argument("--tag", default="v1")
    args = p.parse_args()
    sys.path.insert(0, str(Path(args.foscar) / "benchmarks"))
    import foscar
    from timer import measure
    gpu = torch.cuda.get_device_name()
    peak = next((v for k, v in PEAK_GBS.items() if k in gpu), float("nan"))
    commit = os.environ.get("KV4_COMMIT", "unset")
    fcommit = os.popen(f"git -C {args.foscar} log -1 --format=%h").read().strip()
    print(gpu, "commit", commit, "FOSCAR", fcommit, "torch", torch.__version__, flush=True)
    ef, e4, em = df.ext(), di.ext(), mg.ext()
    rows, table = [], []
    for i, shape in enumerate(args.shapes.split(",")):
        b, s = map(int, shape.split("x"))
        torch.manual_seed(0)
        k = torch.randn(b, HKV, s, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        q = torch.randn(b, HQ, D, device="cuda", dtype=torch.bfloat16)
        fns = {}
        fi = {}
        for page in (16, 64):
            try:
                for name, (fn, nb) in flashinfer_fns(b, s, k, v, page=page).items():
                    t = timeit(fn)
                    if name not in fi or t < fi[name][0]:
                        fi[name] = (t, nb, fn)
            except Exception as ex:  # noqa: BLE001  (e.g. out of memory at the largest shapes)
                print(f"flashinfer page {page} B={b} S={s} skipped: {str(ex).splitlines()[0][:100]}", flush=True)
        for name, (_t, nb, fn) in fi.items():
            fns[name] = (fn, nb)
        fns["foscar_int4"] = (foscar_fn(foscar, k, v, q, b, s, check=i < 2), b * HKV * s * 2 * 68)
        # ours INT4 (dense, as in final_table.py)
        splits = [sp for sp in (1, 2, 4, 8, 16, 32, 64) if sp * HKV * b <= 32768]
        kq, vq = quantize_chunked(k, "int4a_tok", chunk=1), quantize_chunked(v, "int4a_tok", chunk=1)
        c4 = di.prepare(kq, vq)
        qp, qc = di.prep_q(q.float(), HKV)
        del kq, vq
        dense = [sp for sp in splits if s % (sp * 256) == 0]
        sp4 = min((timeit(lambda: em.merge(*e4.decode_int4(*c4, qp, qc, SM, sp, 2, True))), sp) for sp in dense)[1]
        fns["ours_int4a"] = (lambda: em.merge(*e4.decode_int4(*c4, qp, qc, SM, sp4, 2, True)), b * HKV * s * 2 * 68)
        # ours NVFP4, paged; plain and with the full recipe
        kq, vq = quantize_chunked(k, "nvfp4_h", chunk=1), quantize_chunked(v, "nvfp4", chunk=1)
        a = df.args_for(kq, vq, SM)
        del kq, vq
        pages, bt, _ = df.pack_pages(a[0], [s] * b, poison=False)
        sl_all = torch.full((b,), s, dtype=torch.int32, device="cuda")
        sl_rest = sl_all - NS
        qh = df.prep_q(rotate(q.float(), 128), HKV)
        sk, sv = k[:, :, :NS].contiguous(), v[:, :, :NS].contiguous()

        def plain(sp):
            return em.merge(*ef.decode_fp4_paged(pages, bt, sl_all, qh, a[1], a[2], a[3], a[4], sp, a[5]))

        def recipe(sp):
            o, ml = ef.decode_fp4_paged(pages, bt, sl_rest, mg.prep_q_rot(q, HKV), a[1], a[2], a[3], a[4], sp, a[5])
            return mg.merge_sink(o, ml, q, sk, sv, SM)

        spp = min((timeit(lambda: plain(sp)), sp) for sp in splits)[1]
        spr = min((timeit(lambda: recipe(sp)), sp) for sp in splits)[1]
        fns["ours_nvfp4"] = (lambda: plain(spp), b * HKV * s * 2 * 72)
        fns["ours_nvfp4_recipe"] = (lambda: recipe(spr), b * HKV * s * 2 * 72)
        res = {}
        for name, (fn, nb) in fns.items():
            warm = timeit(fn)
            try:
                m = measure(fn)
                cold, nk = m.median_us / 1e3, m.n_kernels
            except Exception as ex:  # noqa: BLE001
                print(f"cold timer failed for {name}: {str(ex)[:120]}", flush=True)
                cold, nk = float("nan"), 0
            res[name] = (warm, cold)
            rows.append(dict(gpu=gpu, commit=commit, foscar=fcommit, tag=args.tag, impl=name, batch=b, ctx=s,
                             warm_ms=f"{warm:.4f}", cold_ms=f"{cold:.4f}", kernels=nk,
                             cold_bw_frac=f"{nb / cold / 1e6 / peak:.3f}"))
            print(f"{name:18s} B={b:3d} S={s:6d} warm {warm:8.4f} ms  cold {cold:8.4f} ms ({nk} kernels, "
                  f"{nb / cold / 1e6 / peak:.0%} of peak)", flush=True)
        table.append((b, s, res))
        del k, v, q, c4, pages, a, fns, fi
        torch.cuda.empty_cache()
    gtag = next((k for k in PEAK_GBS if k in gpu), gpu.replace(" ", "_"))
    out = Path(__file__).resolve().parent.parent / "results" / f"foscar_compare_{gtag}_{args.tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, flush=True)
    impls = ["flashinfer_bf16", "flashinfer_fp8", "foscar_int4", "ours_int4a", "ours_nvfp4", "ours_nvfp4_recipe"]
    for which, j in (("cold L2 (FOSCAR's timer)", 1), ("warm L2 (our timer)", 0)):
        print(f"\nspeedup vs FlashInfer FP8, {which}; ms in parentheses")
        print("| B | S | " + " | ".join(impls) + " |")
        print("|" + "---|" * (len(impls) + 2))
        for b, s, res in table:
            base = res.get("flashinfer_fp8", (float("nan"),) * 2)[j]
            cells = [f"{base / res[n][j]:.2f}x ({res[n][j]:.3f})" if n in res else "-" for n in impls]
            print(f"| {b} | {s} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
