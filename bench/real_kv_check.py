"""Our CUDA decode kernels on real Llama-3.1-8B decode-step tensors (bench/capture_kv.py), 11-11.

Per layer and format: kernel error against an FP32 reference over the same quantized cache (the kernel check), and,
for context only, that reference against unquantized attention (the quantization error itself; the accuracy
evaluation proper is separate). Rotated formats (_h): K rotated at write time and q rotated before the call (K only, as
SAW-INT4); V unrotated.

    python bench/real_kv_check.py --data /scratch/$USER/kv4_realkv/llama31_8b.pt
"""
import argparse
import math

import torch

from kv4.cuda import decode_fp4 as df
from kv4.cuda import decode_int4 as di
from kv4.formats import dequantize, quantize, rotate

D = 128
SM = 1 / math.sqrt(D)


def attn(q, k, v):
    """q (B, HQ, D) fp32, k, v (B, HKV, S, D) fp32 -> (B, HKV, 4, D)."""
    b, hq, d = q.shape
    hkv = k.shape[1]
    qg = q.reshape(b, hkv, hq // hkv, d)
    p = torch.softmax(torch.einsum("bhgd,bhsd->bhgs", qg, k) * SM, -1)
    return torch.einsum("bhgs,bhsd->bhgd", p, v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--splits", default="1,8")
    args = p.parse_args()
    data = torch.load(args.data)
    print(data["model"], data["tokens"], "tokens of", data["source"], flush=True)
    e4, ef = di.ext(), df.ext()
    print(f"{'layer':>5s} {'format':12s} {'splits':>6s} {'kernel err':>10s} {'quant err':>10s}  (both rel. to max |out|)")
    worst = 0.0
    for li in [k for k in data if isinstance(k, int)]:
        q = data[li]["q"].cuda()
        k = data[li]["k"].cuda().float()
        v = data[li]["v"].cuda().float()
        exact = attn(q.half().float(), k, v)
        for fmt in ("int4a_tok", "int4a_tok_h", "nvfp4", "nvfp4_h", "mxfp4", "mxfp4_h"):
            kq = quantize(k, fmt)
            vq = quantize(v, fmt.removesuffix("_h"))
            qk = rotate(q, 128) if fmt.endswith("_h") else q
            ref = attn(qk.half().float(), dequantize(kq, rotate_back=False).float(), dequantize(vq).float())
            qerr = (ref - exact).abs().max().item() / exact.abs().max().item()
            for sp in map(int, args.splits.split(",")):
                if fmt.startswith("int4"):
                    cache = di.prepare(kq, vq)
                    qp, qc = di.prep_q(qk, k.shape[1])
                    o, ml = e4.decode_int4(*cache, qp, qc, SM, sp, 2, True)
                else:
                    a = df.args_for(kq, vq, SM)
                    o, ml = ef.decode_fp4(*a[0], df.prep_q(qk, k.shape[1]), a[1], a[2], a[3], a[4], sp, 2, a[5])
                out = di.combine(o, ml)
                kerr = (out - ref).abs().max().item() / ref.abs().max().item()
                worst = max(worst, kerr)
                print(f"{li:5d} {fmt:12s} {sp:6d} {kerr:10.2e} {qerr:10.2e}", flush=True)
    print(f"worst kernel error {worst:.2e} -> {'ok' if worst < 2e-3 else 'FAIL'}")


if __name__ == "__main__":
    main()
