"""Final decode-attention comparison on one GPU (11-11): FlashInfer fa2 (tensor cores) BF16 / FP8 / NVFP4 against our
CUDA kernels INT4 (asymmetric, centered), NVFP4, MXFP4, all in one job at the same B and S.

Our times include the split merge kernel (kv4/cuda/merge.cu), which also produces the final (B, HQ, 128) bf16 output
as FlashInfer does; the best split count is chosen per shape. Bytes per token and kv head: BF16 512, FP8 256,
NVFP4 144, MXFP4 136, INT4 (asym) 136.

    python bench/final_table.py --tag v1
"""
import argparse
import csv
import math
import os
from pathlib import Path

import torch

from bench.bench_decode import PEAK_GBS, quantize_chunked, timeit
from bench.decode_fp4_bench import flashinfer_nvfp4
from bench.decode_int4_bench import flashinfer_fns
from kv4.cuda import decode_fp4 as df
from kv4.cuda import decode_int4 as di
from kv4.cuda import merge as mg

HQ, HKV, D = 32, 8, 128
SM = 1 / math.sqrt(D)
SHAPES = "1x16384,1x65536,1x131072,4x32768,8x65536,16x32768,16x131072,32x4096,32x16384,64x16384,128x16384"


def check_merge(em):
    torch.manual_seed(0)
    o = torch.randn(3, HKV, 5, 4, D, device="cuda")
    ml = torch.stack([torch.randn(3, HKV, 5, 4, device="cuda") * 3, torch.rand(3, HKV, 5, 4, device="cuda") + 0.1], -1)
    ref = di.combine(o, ml).reshape(3, HQ, D)
    out = em.merge(o.contiguous(), ml.contiguous()).float()
    err = (out - ref).abs().max().item() / ref.abs().max().item()
    print(f"check merge: max err {err:.2e} (bf16 output) -> {'ok' if err < 1e-2 else 'FAIL'}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="v1")
    p.add_argument("--shapes", default=SHAPES)
    p.add_argument("--no-flashinfer", action="store_true", help="e.g. on the H100 node, whose driver rejects the cache")
    p.add_argument("--rule", action="store_true", help="11-18: also time our kernels with the plugin's fixed split rule")
    args = p.parse_args()
    gpu = torch.cuda.get_device_name()
    peak = next((v for k, v in PEAK_GBS.items() if k in gpu), float("nan"))
    commit = os.environ.get("KV4_COMMIT", "unset")
    print(gpu, "commit", commit, flush=True)
    e4, ef, em = di.ext(), df.ext(), mg.ext()
    check_merge(em)
    rows, table = [], []
    for shape in args.shapes.split(","):
        b, s = map(int, shape.split("x"))
        torch.manual_seed(0)
        q = torch.randn(b, HQ, D, device="cuda")
        k = torch.randn(b, HKV, s, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        res = {}

        def record(impl, ms, nbytes, cfg=""):
            gbs = nbytes / ms / 1e6
            res[impl] = ms
            rows.append(dict(gpu=gpu, commit=commit, tag=args.tag, impl=impl, config=cfg, batch=b, ctx=s,
                             ms=f"{ms:.4f}", gbs=f"{gbs:.0f}", peak_frac=f"{gbs / peak:.3f}"))
            print(f"{impl:18s} B={b:3d} S={s:6d} {cfg:10s} {ms:8.4f} ms {gbs:6.0f} GB/s ({gbs / peak:.0%})", flush=True)

        try:
            if args.no_flashinfer:
                raise RuntimeError("--no-flashinfer")
            for name, (fn, nb) in flashinfer_fns(b, s, k, v).items():
                record(name, timeit(fn), nb)
            fn, nb = flashinfer_nvfp4(b, s)
            record("flashinfer_nvfp4", timeit(fn), nb)
        except Exception as ex:  # noqa: BLE001
            print(f"flashinfer skipped: {str(ex).splitlines()[0][:120]}", flush=True)
        torch.cuda.empty_cache()
        splits_list = [sp for sp in (1, 2, 4, 8, 16, 32, 64) if s % (sp * 256) == 0 and sp * HKV * b <= 32768]
        # ours: INT4 (asymmetric, centered)
        kq, vq = quantize_chunked(k, "int4a_tok", chunk=1), quantize_chunked(v, "int4a_tok", chunk=1)
        c4 = di.prepare(kq, vq)
        qp, qc = di.prep_q(q, HKV)
        del kq, vq
        best = min((timeit(lambda: em.merge(*e4.decode_int4(*c4, qp, qc, SM, sp, 2, True))), sp) for sp in splits_list)
        record("cuda_int4a", best[0], b * HKV * s * 2 * 68, f"splits={best[1]}")
        # 11-18: --rule also times our kernels with the vLLM plugin's fixed split rule (no per-shape tuning;
        # integrations/vllm_plugin/kv4_vllm/backend.py), since the baselines use their own heuristics
        rule = max(1, min(16, (2 * 132) // max(1, b * HKV)))
        if args.rule:
            record("cuda_int4a_rule", timeit(lambda: em.merge(*e4.decode_int4(*c4, qp, qc, SM, rule, 2, True))),
                   b * HKV * s * 2 * 68, f"splits={rule}")
        del c4
        for fmt in ("nvfp4", "mxfp4"):
            kq, vq = quantize_chunked(k, fmt, chunk=1), quantize_chunked(v, fmt, chunk=1)
            a = df.args_for(kq, vq, SM)
            qh = df.prep_q(q, HKV)
            del kq, vq
            cache, smul, omul, nbk, nbv, mx = a
            best = min((timeit(lambda: em.merge(*ef.decode_fp4(*cache, qh, smul, omul, nbk, nbv, sp, 2, mx))), sp)
                       for sp in splits_list)
            record(f"cuda_{fmt}", best[0], b * HKV * s * 2 * (72 if fmt == "nvfp4" else 68), f"splits={best[1]}")
            if args.rule:
                record(f"cuda_{fmt}_rule", timeit(lambda: em.merge(*ef.decode_fp4(*cache, qh, smul, omul, nbk, nbv, rule, 2, mx))),
                       b * HKV * s * 2 * (72 if fmt == "nvfp4" else 68), f"splits={rule}")
            del a, cache
        fp8 = res.get("flashinfer_fp8", float("nan"))
        table.append((b, s, {kk: vv for kk, vv in res.items()}, fp8))
        del q, k, v
        torch.cuda.empty_cache()
    gtag = next((k for k in PEAK_GBS if k in gpu), gpu.replace(" ", "_"))
    out = Path(__file__).resolve().parent.parent / "results" / f"final_table_{gtag}_{args.tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, flush=True)
    impls = ["flashinfer_bf16", "flashinfer_fp8", "flashinfer_nvfp4", "cuda_int4a", "cuda_nvfp4", "cuda_mxfp4"]
    impls += [i + "_rule" for i in impls[3:]] if args.rule else []
    print("\n| B | S | " + " | ".join(impls) + " |")
    print("|" + "---|" * (len(impls) + 2))
    for b, s, res, fp8 in table:
        cells = [f"{res[i]:.3f} ({fp8 / res[i]:.2f}x)" if i in res else "-" for i in impls]
        print(f"| {b} | {s} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
