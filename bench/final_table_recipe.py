"""Decode-attention speed with the final accuracy recipe (11-12), on the 11 shapes of bench/final_table.py.

The recipe whose accuracy we report: NVFP4, K mean-removed (mean from the prefill) and Hadamard-rotated, the first 4
tokens in BF16. Per decode step and layer that is, all timed together:
  q rotation (Hadamard 128, K side) + prep_q + paged NVFP4 kernel over S - 4 tokens (64-token pages, shuffled)
  + merge_sink with the 4 BF16 sink keys (stored mean-removed) and values.
Mean removal happens at write time only (the cache holds k - mu), so it adds nothing here; the cache is built from
mean-removed keys anyway, so the bytes are exactly those of the recipe.

Also timed, for the breakdown: the paged kernel + plain merge over S tokens (the number in final_table.py), and
FlashInfer fa2 BF16 / FP8 (best of 16- and 64-token pages). INT4 (asym, K rotated + mean removal) uses the dense kernel
over S tokens plus merge_sink, since it has no paged kernel.

    python bench/final_table_recipe.py --tag v1

v2 (11-12): v1 showed a nearly constant ~28 us of extra time for the recipe (rotate + prep_q + merge_sink as ~10 small
PyTorch launches), so the recipe also runs with the fused q preparation (merge.prep_q_rot, one launch) and every
implementation is also timed as a CUDA graph replay (what a server with CUDA graphs pays: no host launch cost).
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
from kv4.cuda import merge as mg
from kv4.formats import rotate

HQ, HKV, D, NS = 32, 8, 128, 4
SM = 1 / math.sqrt(D)
SHAPES = "1x16384,1x65536,1x131072,4x32768,8x65536,16x32768,16x131072,32x4096,32x16384,64x16384,128x16384"


def graph_time(fn):
    """Time fn as one CUDA-graph replay; None if it cannot be captured."""
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        t = timeit(g.replay)
        del g
        return t
    except Exception as ex:  # noqa: BLE001
        print(f"graph capture failed: {str(ex).splitlines()[0][:100]}", flush=True)
        torch.cuda.synchronize()
        return None


def pr53060_ext():
    """11-13: vLLM PR #53060's TurboQuant decode kernel (baselines/pr53060, unmodified), built with its flags."""
    from torch.utils.cpp_extension import load
    d = Path(__file__).resolve().parent.parent / "baselines" / "pr53060"
    return load(name="pr53060_decode", sources=[str(d / f) for f in ("main.cpp", "kernel.cu", "mma_kernel.cu")],
                # 11-13: C++20 (this torch's headers need it; the PR builds inside vLLM's CMake), and vLLM's -U flags
                extra_cflags=["-O3", "-std=c++20"],
                extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a",
                                   "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                                   "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="v1")
    p.add_argument("--shapes", default=SHAPES)
    p.add_argument("--pr53060", action="store_true", help="11-13: also time vLLM PR #53060's TurboQuant decode")
    p.add_argument("--no-int4", action="store_true")
    args = p.parse_args()
    gpu = torch.cuda.get_device_name()
    peak = next((v for k, v in PEAK_GBS.items() if k in gpu), float("nan"))
    commit = os.environ.get("KV4_COMMIT", "unset")
    print(gpu, "commit", commit, flush=True)
    ef, e4, em = df.ext(), di.ext(), mg.ext()
    rows, table = [], []
    for shape in args.shapes.split(","):
        b, s = map(int, shape.split("x"))
        torch.manual_seed(0)
        mu = torch.randn(1, HKV, 1, D, device="cuda") * 2
        k = (torch.randn(b, HKV, s, D, device="cuda") + mu).to(torch.bfloat16)
        v = torch.randn_like(k)
        q = torch.randn(b, HQ, D, device="cuda", dtype=torch.bfloat16)
        res = {}

        graph = {}

        def record(impl, ms, nbytes, cfg="", fn=None):
            if fn is not None:
                tg = graph_time(fn)
                if tg is not None:
                    graph[impl] = tg
                    rows.append(dict(gpu=gpu, commit=commit, tag=args.tag, impl=impl + "_graph", config=cfg, batch=b,
                                     ctx=s, ms=f"{tg:.4f}", gbs=f"{nbytes / tg / 1e6:.0f}",
                                     peak_frac=f"{nbytes / tg / 1e6 / peak:.3f}"))
            gbs = nbytes / ms / 1e6
            res[impl] = ms
            rows.append(dict(gpu=gpu, commit=commit, tag=args.tag, impl=impl, config=cfg, batch=b, ctx=s,
                             ms=f"{ms:.4f}", gbs=f"{gbs:.0f}", peak_frac=f"{gbs / peak:.3f}"))
            print(f"{impl:22s} B={b:3d} S={s:6d} {cfg:10s} {ms:8.4f} ms {gbs:6.0f} GB/s ({gbs / peak:.0%})", flush=True)

        fi = {}
        for page in (16, 64):
            for name, (fn, nb) in flashinfer_fns(b, s, k, v, page=page).items():
                t = timeit(fn)
                if name not in fi or t < fi[name][0]:
                    fi[name] = (t, nb, page, fn)
        for name, (t, nb, page, fn) in fi.items():
            record(name, t, nb, f"page={page}", fn)
        torch.cuda.empty_cache()
        if args.pr53060:  # 11-13: random K4V4-NC cache (134 B per token and head), best of its page sizes
            ext53 = pr53060_ext()
            rot = torch.linalg.qr(torch.randn(D, D))[0].float().cuda().contiguous()
            cen = torch.linspace(-2.5, 2.5, 16, device="cuda")
            best = None
            for page in (16, 32, 64, 128):
                npg = (s + page - 1) // page
                cache = torch.randint(0, 256, (b * npg, HKV, 1, page * (D + 6)), dtype=torch.uint8, device="cuda")
                pt = torch.randperm(b * npg, device="cuda").int().view(b, npg).contiguous()
                sl = torch.full((b,), s, dtype=torch.int32, device="cuda")
                ws = torch.empty(max(16, ext53.workspace_size(b, HQ, HKV, npg, page, D, s, 0)), dtype=torch.uint8,
                                 device="cuda")
                out53 = torch.empty_like(q)

                def f53(cache=cache, pt=pt, sl=sl, ws=ws, out53=out53, page=page):
                    ext53.run_with_workspace(q, cache, pt, sl, rot, cen, ws, out53, page, s)

                t = timeit(f53)
                if best is None or t < best[0]:
                    best = (t, page, f53)
            record("pr53060_turboquant", best[0], b * HKV * s * (D + 6), f"page={best[1]}", best[2])
            del cache, pt, ws, best
            torch.cuda.empty_cache()
        # the recipe's cache: K mean-removed (mean over the first 512 tokens, as from a prefill), sink kept apart
        kmu = k[:, :, :512].float().mean(2, keepdim=True)
        sk = (k[:, :, :NS].float() - kmu).to(torch.bfloat16).contiguous()
        sv = v[:, :, :NS].contiguous()
        kc = (k.float() - kmu).to(torch.bfloat16)
        del k
        splits_list = [sp for sp in (1, 2, 4, 8, 16, 32) if sp * HKV * b <= 32768]
        kq, vq = quantize_chunked(kc, "nvfp4_h", chunk=1), quantize_chunked(v, "nvfp4", chunk=1)
        a = df.args_for(kq, vq, SM)
        del kq, vq
        pages, bt, _ = df.pack_pages(a[0], [s] * b, poison=False)
        sl_all = torch.full((b,), s, dtype=torch.int32, device="cuda")
        sl_rest = sl_all - NS  # with a real cache the 4-bit part starts at token 4 on a page boundary; same bytes
        qh_plain = df.prep_q(rotate(q.float(), 128), HKV)

        def plain(sp):
            return em.merge(*ef.decode_fp4_paged(pages, bt, sl_all, qh_plain, a[1], a[2], a[3], a[4], sp, a[5]))

        def recipe(sp):
            qr = rotate(q.float(), 128)
            o, ml = ef.decode_fp4_paged(pages, bt, sl_rest, df.prep_q(qr, HKV), a[1], a[2], a[3], a[4], sp, a[5])
            return mg.merge_sink(o, ml, q, sk, sv, SM)

        def fused(sp):  # the same recipe with q preparation in one launch and bf16 q read directly by merge_sink
            o, ml = ef.decode_fp4_paged(pages, bt, sl_rest, mg.prep_q_rot(q, HKV), a[1], a[2], a[3], a[4], sp, a[5])
            return mg.merge_sink(o, ml, q, sk, sv, SM)

        if not table:  # correctness of the fused path against the Python one, once
            d_q = (mg.prep_q_rot(q, HKV).float() - qh_plain.float()).abs().max().item()
            d_o = (fused(1).float() - recipe(1).float()).abs().max().item()
            print(f"check fused q prep: max |diff| {d_q:.2e} (fp16 rounding); recipe output max |diff| {d_o:.2e} -> "
                  f"{'ok' if d_q < 2e-3 and d_o < 2e-2 else 'FAIL'}", flush=True)

        nb4 = b * HKV * s * 2 * 72
        t, sp = min((timeit(lambda: plain(sp)), sp) for sp in splits_list)
        record("ours_nvfp4_kernel", t, nb4, f"splits={sp}", lambda sp=sp: plain(sp))
        t, sp = min((timeit(lambda: recipe(sp)), sp) for sp in splits_list)
        record("ours_nvfp4_recipe", t, nb4, f"splits={sp}", lambda sp=sp: recipe(sp))
        t, sp = min((timeit(lambda: fused(sp)), sp) for sp in splits_list)
        record("ours_nvfp4_fused", t, nb4, f"splits={sp}", lambda sp=sp: fused(sp))
        del pages, a
        torch.cuda.empty_cache()
        if args.no_int4:
            del kc, v
            torch.cuda.empty_cache()
            table.append((b, s, dict(res), dict(graph)))
            continue
        kq, vq = quantize_chunked(kc, "int4a_tok_h", chunk=1), quantize_chunked(v, "int4a_tok", chunk=1)
        c4 = di.prepare(kq, vq)
        del kq, vq

        def int4_recipe(sp):
            qr = rotate(q.float(), 128)
            qp, qc = di.prep_q(qr, HKV)
            return mg.merge_sink(*e4.decode_int4(*c4, qp, qc, SM, sp, 2, True), q, sk, sv, SM)

        dense = [sp for sp in splits_list if s % (sp * 256) == 0]
        t, sp = min((timeit(lambda: int4_recipe(sp)), sp) for sp in dense)
        record("ours_int4a_recipe", t, b * HKV * s * 2 * 68, f"splits={sp}", lambda sp=sp: int4_recipe(sp))
        del c4, kc, v
        torch.cuda.empty_cache()
        table.append((b, s, dict(res), dict(graph)))
    gtag = next((k for k in PEAK_GBS if k in gpu), gpu.replace(" ", "_"))
    out = Path(__file__).resolve().parent.parent / "results" / f"final_table_recipe_{gtag}_{args.tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, flush=True)
    impls = ["flashinfer_bf16", "flashinfer_fp8", "ours_nvfp4_kernel", "ours_nvfp4_recipe", "ours_nvfp4_fused",
             "ours_int4a_recipe", "pr53060_turboquant"]
    for which, j in (("eager (host launches included)", 2), ("CUDA graph replay", 3)):
        print(f"\nspeedup vs FlashInfer FP8, {which}; ms in parentheses")
        print("| B | S | " + " | ".join(impls) + " | fused / kernel |")
        print("|" + "---|" * (len(impls) + 3))
        for row in table:
            b, s, res = row[0], row[1], row[j]
            fp8 = res.get("flashinfer_fp8", float("nan"))
            cells = [f"{fp8 / res[i]:.2f}x ({res[i]:.3f})" if i in res else "-" for i in impls]
            r = res.get("ours_nvfp4_fused", float("nan")) / res.get("ours_nvfp4_kernel", float("nan"))
            print(f"| {b} | {s} | " + " | ".join(cells) + f" | {r:.3f} |")


if __name__ == "__main__":
    main()
