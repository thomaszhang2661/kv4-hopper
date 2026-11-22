"""One Llama-3.1-8B decode step on H200, end to end by parts (11-11): the non-attention work of all 32 layers plus the
LM head, measured as one CUDA graph with random BF16 weights of the real shapes, plus 32 x the measured attention time
of each KV format at the same (B, S). Gives tokens/s and the speedup a 4-bit cache brings to the whole step, and whether
the whole cache fits next to the weights in 141 GB.

Attention per layer: FlashInfer fa2 BF16 and FP8 (best of 16- and 64-token pages); ours: q rotation (Hadamard, K-side),
paged NVFP4 kernel over S - 4 tokens (64-token pages, shuffled) and merge_sink with the 4 BF16 sink tokens, best splits.
Not included (same for every format, or small): the cache write of the new token, sampling, the CPU side of a server.

    python bench/e2e_step_time.py
"""
import math
import os

import torch

from bench.bench_decode import quantize_chunked, timeit
from bench.decode_fp4_bench import flashinfer_nvfp4  # noqa: F401  (kept importable together)
from bench.decode_int4_bench import flashinfer_fns
from kv4.cuda import decode_fp4 as df
from kv4.cuda import merge as mg
from kv4.formats import rotate

H, HQ, HKV, D, FF, V, LAYERS, NS = 4096, 32, 8, 128, 14336, 128256, 32, 4
SM = 1 / math.sqrt(D)
WEIGHTS_GB = 16.06
HBM_GB = 141.0
KV_BYTES = {"bf16": 2 * D * 2, "fp8": 2 * D, "nvfp4": 2 * 72}  # per token, KV head and layer (K + V)


def rest_of_step(b):
    """CUDA-graph time (ms) of everything but the attention core for B tokens: per layer 2 RMSNorm, QKV, O, gate/up,
    SiLU*mul, down, 2 residual adds; then the final norm and the LM head."""
    dev, dt = "cuda", torch.bfloat16
    ws = [dict(qkv=torch.randn(H + 2 * HKV * D, H, device=dev, dtype=dt) * 0.02,
               o=torch.randn(H, H, device=dev, dtype=dt) * 0.02,
               gu=torch.randn(2 * FF, H, device=dev, dtype=dt) * 0.02,
               down=torch.randn(H, FF, device=dev, dtype=dt) * 0.02,
               n1=torch.ones(H, device=dev, dtype=dt), n2=torch.ones(H, device=dev, dtype=dt)) for _ in range(LAYERS)]
    head = torch.randn(V, H, device=dev, dtype=dt) * 0.02
    x0 = torch.randn(b, H, device=dev, dtype=dt)
    nrm = lambda t, w: torch.nn.functional.rms_norm(t, (H,), w, 1e-5)  # noqa: E731

    def step():
        x = x0
        for w in ws:
            qkv = torch.nn.functional.linear(nrm(x, w["n1"]), w["qkv"])
            x = x + torch.nn.functional.linear(qkv[:, :H].contiguous(), w["o"])   # attention core replaced by q
            g, u = torch.nn.functional.linear(nrm(x, w["n2"]), w["gu"]).chunk(2, -1)
            x = x + torch.nn.functional.linear(torch.nn.functional.silu(g) * u, w["down"])
        return torch.nn.functional.linear(nrm(x, ws[0]["n1"]), head)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    t = timeit(g.replay)
    del ws, head, g
    torch.cuda.empty_cache()
    return t


def attn_times(b, s):
    torch.manual_seed(0)
    k = torch.randn(b, HKV, s, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out = {}
    for page in (16, 64):
        for name, (fn, _) in flashinfer_fns(b, s, k, v, page=page).items():
            t = timeit(fn)
            out[name] = min(out.get(name, 1e9), t)
    kq, vq = quantize_chunked(k, "nvfp4_h", chunk=1), quantize_chunked(v, "nvfp4", chunk=1)
    del k, v
    a = df.args_for(kq, vq, SM)
    del kq, vq
    pages, bt, _ = df.pack_pages(a[0], [s] * b, poison=False)
    sl = torch.full((b,), s - NS, dtype=torch.int32, device="cuda")
    q = torch.randn(b, HQ, D, device="cuda", dtype=torch.bfloat16)
    sk = torch.randn(b, HKV, NS, D, device="cuda", dtype=torch.bfloat16)
    sv = torch.randn_like(sk)
    e = df.ext()

    def ours(sp):
        qr = rotate(q.float(), 128)
        o, ml = e.decode_fp4_paged(pages, bt, sl, df.prep_q(qr, HKV), a[1], a[2], a[3], a[4], sp, a[5])
        return mg.merge_sink(o, ml, q, sk, sv, SM)

    out["ours_nvfp4_kh_sink"], out["ours_splits"] = min((timeit(lambda: ours(sp)), sp) for sp in (1, 2, 4, 8, 16, 32))
    del pages, a
    torch.cuda.empty_cache()
    return out


def main():
    print(torch.cuda.get_device_name(), "commit", os.environ.get("KV4_COMMIT", "unset"), flush=True)
    rest = {}
    for b, s in [(1, 131072), (8, 32768), (16, 65536), (32, 16384), (32, 32768), (64, 16384), (128, 16384),
                 (256, 8192)]:
        if b not in rest:
            rest[b] = rest_of_step(b)
        at = attn_times(b, s)
        line = [f"B={b:3d} S={s:6d}: rest of step {rest[b]:.3f} ms |"]
        steps = {}
        for name, key, fmt in (("BF16", "flashinfer_bf16", "bf16"), ("FP8", "flashinfer_fp8", "fp8"),
                               ("NVFP4+sink (ours)", "ours_nvfp4_kh_sink", "nvfp4")):
            att = at[key]
            step = rest[b] + LAYERS * att
            steps[name] = step
            kv_gb = b * s * HKV * LAYERS * KV_BYTES[fmt] / 1e9
            fits = "fits" if kv_gb + WEIGHTS_GB < HBM_GB * 0.92 else "does NOT fit"
            line.append(f"{name}: attn {att:.4f} ms/layer, step {step:.2f} ms, {b / step * 1e3:.0f} tok/s, "
                        f"KV {kv_gb:.0f} GB ({fits}) |")
        line.append(f"ours vs FP8 {steps['FP8'] / steps['NVFP4+sink (ours)']:.2f}x, vs BF16 "
                    f"{steps['BF16'] / steps['NVFP4+sink (ours)']:.2f}x (splits {at['ours_splits']})")
        print(" ".join(line), flush=True)


if __name__ == "__main__":
    main()
