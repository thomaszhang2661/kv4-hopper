"""One decode call of one implementation, inside a profiler range, for Nsight Compute (11-10).

Question: are the 4-bit kernels (all near 25% of bandwidth) limited by waiting (serial
reconstruction and MMA leave the SM idle; overlap would help) or by the instruction issue budget
(the SM issues at full rate; only fewer instructions per value would help)?

    ncu --profile-from-start off ... python bench/ncu_target.py --impl ours_int4_tok
Implementations: ours_{bf16,fp8,int4_tok,int4a_tok,nvfp4,mxfp4}, fi_{bf16,fp8,nvfp4}, bd_int4,
                 xqa_p{16,64} (11-21, TRT-LLM XQA FP8 via FlashInfer).
"""
import argparse
import math

import torch

p = argparse.ArgumentParser()
p.add_argument("--impl", required=True)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--ctx", type=int, default=16384)
args = p.parse_args()
B, S, HQ, HKV, D = args.batch, args.ctx, 32, 8, 128
sm = 1 / math.sqrt(D)
torch.manual_seed(0)
kind, fmt = args.impl.split("_", 1)

if kind == "ours":
    from bench.bench_decode import quantize_chunked
    from kv4.kernels import KERNELS
    q = torch.randn(B, HQ, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, HKV, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    seq = torch.full((B,), S, device="cuda", dtype=torch.int32)
    kq, vq = (k, v) if fmt == "bf16" else (quantize_chunked(k, fmt), quantize_chunked(v, fmt))
    fn = lambda: KERNELS[fmt](q, kq, vq, seq, sm)  # noqa: E731
elif kind == "fi":
    from bench.bench_flashinfer import KV_DTYPE, make_wrapper, page_table
    page = 16
    indptr, indices, last = page_table([S] * B, page)
    npages = int(indptr[-1])
    q = torch.randn(B, HQ, D, device="cuda", dtype=torch.bfloat16)
    shape = (npages, HKV, page, D)
    kw = {}
    if fmt == "bf16":
        cache = (torch.randn(shape, device="cuda", dtype=torch.bfloat16),
                 torch.randn(shape, device="cuda", dtype=torch.bfloat16))
    elif fmt == "fp8":
        cache = (torch.randn(shape, device="cuda").to(torch.float8_e4m3fn),
                 torch.randn(shape, device="cuda").to(torch.float8_e4m3fn))
        kw = dict(k_scale=0.01, v_scale=0.01)
    else:
        one = torch.tensor(1.0).to(torch.float8_e4m3fn).view(torch.uint8).item()
        sf_shape = shape[:-1] + (D // 16,)
        cache = (torch.randint(0, 256, shape[:-1] + (D // 2,), device="cuda", dtype=torch.uint8),
                 torch.randint(0, 256, shape[:-1] + (D // 2,), device="cuda", dtype=torch.uint8))
        kw = dict(k_scale=0.01, v_scale=0.01,
                  kv_cache_sf=(torch.full(sf_shape, one, device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn),
                               torch.full(sf_shape, one, device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn)))
    w, resolved = make_wrapper("fa2", True, indptr, indices, last, HQ, HKV, D, page, KV_DTYPE[fmt])
    fn = lambda: w.run(q, cache, **kw)  # noqa: E731
elif kind == "bd":
    from baselines.bitdecoding.build import load
    from bench.bench_bitdecoding import bd_inputs
    ext = load()
    q = torch.randn(B, 1, HQ, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, S, HKV, D, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    kp, kpar, vp, vpar, cu = bd_inputs(k, v)
    ext.pack(k.reshape(B * S, HKV, D), kp, kpar, v.reshape(B * S, HKV, D), vp, vpar, cu, S)
    fn = lambda: ext.fwd(q, k, kp, kpar, v, vp, vpar, sm, 0)  # noqa: E731
elif kind == "kside":  # K-side CUDA prototype, fmt = v1 / v2 (kv4/cuda/kside.cu); best H200 config splits=4, nst=3
    from bench.kside_proto import inputs
    from kv4.cuda import kside
    e = kside.ext()
    _, _, kp, ks, qp, k0 = inputs(B, S)
    var = {"v1": 0, "v2": 1, "v4": 2}[fmt]
    fn = lambda: e.kside(kp, ks, qp, k0, sm, 4, 3, None, var)  # noqa: E731
elif kind == "cuda":  # full CUDA decode kernels (11-10 night): cuda_int4a, cuda_nvfp4, cuda_mxfp4; splits=1, nst=2
    from kv4.formats import quantize as _q
    from bench.bench_decode import quantize_chunked as _qc
    q = torch.randn(B, HQ, D, device="cuda")
    k = torch.randn(B, HKV, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    if fmt == "int4a":
        from kv4.cuda import decode_int4 as di
        kq, vq = _qc(k, "int4a_tok", chunk=1), _qc(v, "int4a_tok", chunk=1)
        cache = di.prepare(kq, vq)
        qp, qc = di.prep_q(q, HKV)
        e = di.ext()
        fn = lambda: e.decode_int4(*cache, qp, qc, sm, 1, 2, True)  # noqa: E731
    else:
        from kv4.cuda import decode_fp4 as df
        kq, vq = _qc(k, fmt, chunk=1), _qc(v, fmt, chunk=1)
        a = df.args_for(kq, vq, sm)
        qh = df.prep_q(q, HKV)
        e = df.ext()
        fn = lambda: e.decode_fp4(*a[0], qh, a[1], a[2], a[3], a[4], 1, 2, a[5])  # noqa: E731
    del k, v
elif kind == "k8":  # 11-21: C1 prototype (K on FP8 tensor cores), kv4/cuda/decode_fp4_k8.cu; splits = 1
    from bench.bench_decode import quantize_chunked as _qc
    from kv4.cuda import decode_fp4_k8 as k8m
    q = torch.randn(B, HQ, D, device="cuda")
    k = torch.randn(B, HKV, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    c8, smul8, omul8 = k8m.args_for(_qc(k, "nvfp4_g32", chunk=1), _qc(v, "nvfp4", chunk=1), sm)
    q8, qs = k8m.prep_q8(q, HKV)
    e = k8m.ext()
    fn = lambda: e.decode_k8(*c8, q8, smul8 * qs, omul8, 1)  # noqa: E731
    del k, v
elif kind == "xqa":  # 11-21: TRT-LLM's FP8 decode (XQA) through FlashInfer, as bench/xqa_compare.py calls it; fmt = p<page>
    from flashinfer.decode import xqa_batch_decode_with_kv_cache
    page = int(fmt[1:])
    q = torch.randn(B, HQ, D, device="cuda", dtype=torch.bfloat16)
    k8 = torch.randn(B * (S // page), HKV, page, D, device="cuda").to(torch.float8_e4m3fn)  # HND pages
    v8 = torch.randn_like(k8, dtype=torch.float32).to(torch.float8_e4m3fn)
    cache = (k8, v8)
    ws = torch.zeros(512 * 2**20, dtype=torch.uint8, device="cuda")
    bt = torch.arange(B * (S // page), dtype=torch.int32, device="cuda").reshape(B, S // page)
    sl = torch.full((B,), S, dtype=torch.uint32, device="cuda")
    out = torch.empty(B, HQ, D, dtype=torch.bfloat16, device="cuda")
    fn = lambda: xqa_batch_decode_with_kv_cache(q, cache, ws, bt, sl, S, bmm1_scale=sm, bmm2_scale=1.0,  # noqa: E731
                                                out=out, kv_layout="HND")
else:
    raise SystemExit(f"unknown impl {args.impl}")

if kind == "ours":
    from kv4.formats import FORMATS
    nbytes = B * S * HKV * 2 * FORMATS[fmt].bytes_per_row(D)
elif kind == "kside":
    nbytes = B * HKV * S * 66  # K only: codes + fp16 scale
elif kind == "cuda":
    nbytes = B * HKV * S * 2 * {"int4a": 68, "nvfp4": 72, "mxfp4": 68}[fmt]
elif kind == "k8":
    nbytes = B * HKV * S * 140  # K 64 + 4, V 64 + 8
elif kind == "xqa":
    nbytes = B * HKV * S * 2 * D  # FP8 K and V, no scales
elif kind == "fi":
    nbytes = sum(t.numel() * t.element_size() for t in cache) + sum(
        t.numel() * t.element_size() for t in kw.get("kv_cache_sf", ()))
else:
    nbytes = sum(t.numel() * t.element_size() for t in (kp, kpar, vp, vpar))


def mark(text):
    """Marker in stdout between NVBit's own printf output (11-10, bench/nvbit_summary.py): flush C stdio first,
    then write straight to fd 1, so the order in the log is the order of events."""
    import ctypes
    import os
    ctypes.CDLL(None).fflush(None)
    os.write(1, (text + "\n").encode())


for _ in range(3):  # compile / JIT / autotune outside the profiled range
    fn()
torch.cuda.synchronize()
mark(f"==KV4 BEGIN impl={args.impl} B={B} S={S} HKV={HKV} D={D} bytes={nbytes}")
torch.cuda.profiler.start()
fn()
torch.cuda.synchronize()
torch.cuda.profiler.stop()
mark("==KV4 END")
print(f"profiled {args.impl} B={B} S={S}", flush=True)
