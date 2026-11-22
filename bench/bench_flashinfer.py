"""FlashInfer decode attention on BF16, FP8 and NVFP4 KV caches: the existing implementations to beat.

Runs in the reference-kernel env (slurm/run_base.sbatch). Same timing as bench_decode.py (L2 flush
before every call, median of 50 after 5 warm-ups), same shapes, same bytes accounting, so the
numbers sit next to ours. For every format it tries every decode backend FlashInfer offers here
(fa2 CUDA-core, fa2 tensor-core, fa3, auto) and records which kernel actually ran
(`resolved_backend`): on SM90, NVFP4 KV is only accepted by fa2, and fa2 has two kernels.

Before timing, a correctness check on a small ragged batch compares FlashInfer against a torch
reference built from our own reading of its cache layout (packed E2M1, low nibble = even element,
E4M3 block scales per 16, one global scale). A large error there means the layout reading is wrong,
and the speed numbers are then still valid but the "same bytes" claim needs a second look.

  sbatch -J kv4fiH200 --gres=gpu:h200:1 slurm/run_base.sbatch bench/bench_flashinfer.py
"""
import argparse
import csv
import math
import os
from pathlib import Path

import torch

import flashinfer

PEAK_GBS = {"H200": 4800, "H100": 3350, "A100": 2039}
E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
BYTES_PER_ROW = {"bf16": 256, "fp8": 128, "nvfp4": 64 + 8}  # D = 128; NVFP4: packed codes + E4M3 scale per 16
KV_DTYPE = {"bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn, "nvfp4": torch.uint8}
# (backend, use_tensor_cores) combinations to try for every format
CONFIGS = [("fa2", False), ("fa2", True), ("fa3", True), ("auto", False), ("auto", True)]


def timeit(fn, warmup=5, reps=50) -> float:
    flush = torch.empty(256 * 2**20, dtype=torch.int8, device="cuda")
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        flush.zero_()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return sorted(times)[len(times) // 2]


def page_table(seq_lens: list[int], page: int):
    """Contiguous pages per request: indptr, indices, last_page_len."""
    npages = [-(-s // page) for s in seq_lens]
    indptr = torch.tensor([0] + list(torch.tensor(npages).cumsum(0)), dtype=torch.int32, device="cuda")
    indices = torch.arange(int(indptr[-1]), dtype=torch.int32, device="cuda")
    last = torch.tensor([s - (n - 1) * page for s, n in zip(seq_lens, npages)], dtype=torch.int32, device="cuda")
    return indptr, indices, last


def make_wrapper(backend, tc, indptr, indices, last, hq, hkv, d, page, kv_dtype):
    ws = torch.zeros(256 * 2**20, dtype=torch.uint8, device="cuda")
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "HND", use_tensor_cores=tc, backend=backend)
    w.plan(indptr, indices, last, hq, hkv, d, page, q_data_type=torch.bfloat16, kv_data_type=kv_dtype,
           o_data_type=torch.bfloat16)
    resolved = getattr(w, "resolved_backend", None)
    return w, (resolved if isinstance(resolved, str) else str(resolved))


def dequant_nvfp4(data: torch.Tensor, sf: torch.Tensor, gscale: float) -> torch.Tensor:
    """(.., D/2) uint8 + (.., D/16) E4M3 bits -> (.., D) float32, low nibble = even element."""
    data = data.view(torch.uint8)
    codes = torch.stack([data & 0xF, data >> 4], -1).flatten(-2).long()
    grid = E2M1_GRID.to(data.device)
    vals = torch.where(codes >= 8, -grid[codes & 7], grid[codes & 7])
    scale = sf.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
    return vals * scale * gscale


def paged_to_dense(cache: torch.Tensor, indptr, last, page: int, b: int) -> list[torch.Tensor]:
    """HND paged (pages, H, page, X) -> per request (H, S, X)."""
    out = []
    for i in range(b):
        p0, p1 = int(indptr[i]), int(indptr[i + 1])
        x = cache[p0:p1].permute(1, 0, 2, 3).reshape(cache.shape[1], (p1 - p0) * page, -1)
        out.append(x[:, : (p1 - p0 - 1) * page + int(last[i])])
    return out


def reference(q: torch.Tensor, ks: list[torch.Tensor], vs: list[torch.Tensor]) -> torch.Tensor:
    outs = []
    for i, (k, v) in enumerate(zip(ks, vs)):
        g = q.shape[1] // k.shape[0]
        kk, vv = k.float().repeat_interleave(g, 0), v.float().repeat_interleave(g, 0)
        w = torch.softmax(torch.einsum("hd,hsd->hs", q[i].float(), kk) / math.sqrt(q.shape[-1]), -1)
        outs.append(torch.einsum("hs,hsd->hd", w, vv))
    return torch.stack(outs)


def check(hq, hkv, d, page):
    """Small ragged batch, every format and config: max |FlashInfer - torch reference on the stored values|."""
    seq_lens = [1000, 37, 4096]
    b = len(seq_lens)
    indptr, indices, last = page_table(seq_lens, page)
    npages = int(indptr[-1])
    torch.manual_seed(0)
    q = torch.randn(b, hq, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(npages, hkv, page, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    # FP8 with a per-tensor scale, as vLLM does
    ksc, vsc = k.abs().amax().item() / 448, v.abs().amax().item() / 448
    k8, v8 = (k / ksc).to(torch.float8_e4m3fn), (v / vsc).to(torch.float8_e4m3fn)
    # NVFP4 through FlashInfer's own quantize-and-append kernel, from the same BF16 rows
    kd = torch.zeros(npages, hkv, page, d // 2, dtype=torch.uint8, device="cuda")
    vd = torch.zeros_like(kd)
    # FlashInfer wants the scale caches typed as E4M3 (the bytes are the same)
    ksf = torch.zeros(npages, hkv, page, d // 16, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
    vsf = torch.zeros_like(ksf)
    kdense, vdense = paged_to_dense(k, indptr, last, page, b), paged_to_dense(v, indptr, last, page, b)
    rows_k = torch.cat([x.permute(1, 0, 2) for x in kdense])  # (nnz, H, D)
    rows_v = torch.cat([x.permute(1, 0, 2) for x in vdense])
    bidx = torch.cat([torch.full((s,), i, dtype=torch.int32) for i, s in enumerate(seq_lens)]).cuda()
    pos = torch.cat([torch.arange(s, dtype=torch.int32) for s in seq_lens]).cuda()
    kg, vg = rows_k.abs().amax().item() / (6 * 448), rows_v.abs().amax().item() / (6 * 448)
    flashinfer.nvfp4_quantize_append_paged_kv_cache(rows_k.contiguous(), rows_v.contiguous(), bidx, pos, (kd, vd),
                                                   (ksf, vsf), indices, indptr, last, kg, vg, kv_layout="HND")
    stored = {
        "bf16": ((k, v), {}, k.float(), v.float()),
        "fp8": ((k8, v8), dict(k_scale=ksc, v_scale=vsc), k8.float() * ksc, v8.float() * vsc),
        "nvfp4": ((kd, vd), dict(k_scale=kg, v_scale=vg, kv_cache_sf=(ksf, vsf)),
                  dequant_nvfp4(kd, ksf, kg), dequant_nvfp4(vd, vsf, vg)),
    }
    ref_bf16 = reference(q, kdense, vdense)
    for fmt, (cache, kw, kf, vf) in stored.items():
        ref = reference(q, paged_to_dense(kf, indptr, last, page, b), paged_to_dense(vf, indptr, last, page, b))
        for backend, tc in CONFIGS:
            try:
                w, resolved = make_wrapper(backend, tc, indptr, indices, last, hq, hkv, d, page, KV_DTYPE[fmt])
                out = w.run(q, cache, **kw).float()
                print(f"check {fmt:5s} {backend:4s} tc={int(tc)} -> {resolved:10s} max err vs stored values "
                      f"{(out - ref).abs().max().item():.1e}, vs bf16 original {(out - ref_bf16).abs().max().item():.1e}",
                      flush=True)
            except Exception as e:
                print(f"check {fmt:5s} {backend:4s} tc={int(tc)}: {type(e).__name__}: {str(e)[:120]}", flush=True)


def ours(b, s, hq, hkv, d, peak, rows, gpu, commit):
    """Our Triton kernels in the same job (this env runs a newer Triton than the fp4 env)."""
    from bench.bench_decode import quantize_chunked
    from kv4.kernels import KERNELS
    from kv4.formats import FORMATS
    torch.manual_seed(0)
    q = torch.randn(b, hq, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, hkv, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    seq = torch.full((b,), s, device="cuda", dtype=torch.int32)
    for name in ["bf16", "fp8", "int4_tok", "nvfp4", "mxfp4"]:
        if name not in KERNELS:
            continue
        try:
            kq, vq = (k, v) if name == "bf16" else (quantize_chunked(k, name), quantize_chunked(v, name))
            ms = timeit(lambda: KERNELS[name](q, kq, vq, seq, 1 / math.sqrt(d)))
        except Exception as e:
            print(f"B={b} S={s} ours {name}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        gbs = 2 * b * hkv * s * FORMATS[name].bytes_per_row(d) / ms / 1e6
        rows.append(dict(gpu=gpu, commit=commit, impl="ours_triton" + triton_version(), format=name, backend="-",
                         tensor_cores="-", resolved="-", batch=b, ctx=s, page=0, ms=round(ms, 4), gbs=round(gbs),
                         peak_frac=round(gbs / peak, 3)))
        print(f"B={b:3d} S={s:6d} ours {name:9s} {ms:8.3f} ms {gbs:6.0f} GB/s ({gbs / peak:.0%})", flush=True)


def triton_version():
    import triton
    return triton.__version__


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32, 64])
    p.add_argument("--ctx", type=int, nargs="+", default=[4096, 16384, 65536])
    p.add_argument("--hq", type=int, default=32)
    p.add_argument("--hkv", type=int, default=8)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--page", type=int, default=16, help="vLLM's default page size")
    p.add_argument("--mem-gb", type=float, default=40, help="skip shapes whose BF16 cache exceeds this")
    p.add_argument("--no-ours", action="store_true")
    p.add_argument("--skip-check", action="store_true", help="skip the layout check (run it in one job of a split)")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    assert args.d == 128, "BYTES_PER_ROW assumes D = 128"

    gpu = torch.cuda.get_device_name()
    peak = next((v for k, v in PEAK_GBS.items() if k in gpu), float("nan"))
    commit = os.environ.get("KV4_COMMIT", "unknown")
    print(gpu, "torch", torch.__version__, "flashinfer", flashinfer.__version__, "triton", triton_version(), flush=True)
    if not args.skip_check:
        try:
            check(args.hq, args.hkv, args.d, args.page)
        except Exception as e:  # a broken check must not cost the timings
            print(f"check failed: {type(e).__name__}: {str(e)[:200]}", flush=True)

    rows = []
    for b in args.batch:
        for s in args.ctx:
            if 2 * b * args.hkv * s * args.d * 2 > args.mem_gb * 2**30:
                continue
            indptr, indices, last = page_table([s] * b, args.page)
            npages = int(indptr[-1])
            torch.manual_seed(0)
            q = torch.randn(b, args.hq, args.d, device="cuda", dtype=torch.bfloat16)
            shape = (npages, args.hkv, args.page, args.d)
            caches = {
                "bf16": ((torch.randn(shape, device="cuda", dtype=torch.bfloat16),
                          torch.randn(shape, device="cuda", dtype=torch.bfloat16)), {}),
                "fp8": ((torch.randn(shape, device="cuda").to(torch.float8_e4m3fn),
                         torch.randn(shape, device="cuda").to(torch.float8_e4m3fn)), dict(k_scale=0.01, v_scale=0.01)),
            }
            sf_shape = (npages, args.hkv, args.page, args.d // 16)
            one = torch.tensor(1.0).to(torch.float8_e4m3fn).view(torch.uint8).item()  # E4M3 bits of 1.0
            caches["nvfp4"] = (
                (torch.randint(0, 256, shape[:-1] + (args.d // 2,), device="cuda", dtype=torch.uint8),
                 torch.randint(0, 256, shape[:-1] + (args.d // 2,), device="cuda", dtype=torch.uint8)),
                dict(k_scale=0.01, v_scale=0.01,
                     kv_cache_sf=(torch.full(sf_shape, one, device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn),
                                  torch.full(sf_shape, one, device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn))))
            for fmt, (cache, kw) in caches.items():
                for backend, tc in CONFIGS:
                    try:
                        w, resolved = make_wrapper(backend, tc, indptr, indices, last, args.hq, args.hkv, args.d,
                                                   args.page, KV_DTYPE[fmt])
                        ms = timeit(lambda: w.run(q, cache, **kw))
                    except Exception as e:
                        print(f"B={b} S={s} {fmt} {backend} tc={int(tc)}: {type(e).__name__}: {str(e)[:100]}", flush=True)
                        continue
                    gbs = 2 * b * args.hkv * s * BYTES_PER_ROW[fmt] / ms / 1e6
                    rows.append(dict(gpu=gpu, commit=commit, impl="flashinfer" + flashinfer.__version__, format=fmt,
                                     backend=backend, tensor_cores=int(tc), resolved=resolved, batch=b, ctx=s,
                                     page=args.page, ms=round(ms, 4), gbs=round(gbs), peak_frac=round(gbs / peak, 3)))
                    print(f"B={b:3d} S={s:6d} fi {fmt:5s} {backend:4s} tc={int(tc)} -> {resolved:10s} {ms:8.3f} ms "
                          f"{gbs:6.0f} GB/s ({gbs / peak:.0%})", flush=True)
            del caches
            torch.cuda.empty_cache()
            if not args.no_ours:
                ours(b, s, args.hq, args.hkv, args.d, peak, rows, gpu, commit)
                torch.cuda.empty_cache()
            write_csv(rows, args.tag, gpu)
    write_csv(rows, args.tag, gpu)


def write_csv(rows, tag, gpu):
    if not rows:
        return
    gpu_tag = next((k for k in PEAK_GBS if k in gpu), gpu.replace(" ", "_"))
    out = Path("results") / (f"bench_flashinfer_{gpu_tag}" + (f"_{tag}" if tag else "") + ".csv")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
