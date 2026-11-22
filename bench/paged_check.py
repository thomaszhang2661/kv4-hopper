"""Paged, variable-length NVFP4 / MXFP4 decode (kv4/cuda/decode_fp4.cu PAGED, 11-11): correctness with ragged
lengths, shuffled 64-token pages and poisoned tail bytes; speed against the dense kernel.

    python bench/paged_check.py
"""
import math
import os
import sys

import torch

from bench.bench_decode import PEAK_GBS, quantize_chunked, timeit
from kv4.cuda import decode_fp4 as df
from kv4.cuda import decode_int4 as di
from kv4.formats import dequantize, quantize

HQ, HKV, D = 32, 8, 128
SM = 1 / math.sqrt(D)


def ref(q, kq, vq, lens):
    k, v = dequantize(kq, rotate_back=False).float(), dequantize(vq, rotate_back=False).float()
    outs = []
    for b, L in enumerate(lens):
        qe = q[b].reshape(HKV, 4, D).half().float()
        p = torch.softmax(torch.einsum("hgd,hsd->hgs", qe, k[b, :, :L]) * SM, -1)
        outs.append(torch.einsum("hgs,hsd->hgd", p, v[b, :, :L]))
    return torch.stack(outs)


def main():
    gpu = torch.cuda.get_device_name()
    peak = next((v for k, v in PEAK_GBS.items() if k in gpu), float("nan"))
    print(gpu, flush=True)
    e = df.ext()
    ok = True
    for fmt in ("nvfp4", "mxfp4"):
        torch.manual_seed(0)
        lens = [300, 4096, 1000, 257, 64, 1]
        b, s = len(lens), 4096
        q = torch.randn(b, HQ, D, device="cuda")
        kq = quantize(torch.randn(b, HKV, s, D, device="cuda", dtype=torch.bfloat16), fmt)
        vq = quantize(torch.randn(b, HKV, s, D, device="cuda", dtype=torch.bfloat16), fmt)
        a = df.args_for(kq, vq, SM)
        pages, bt, sl = df.pack_pages(a[0], lens)
        qh = df.prep_q(q, HKV)
        r = ref(q, kq, vq, lens)
        for splits in (1, 3, 8):
            o, ml = e.decode_fp4_paged(pages, bt, sl, qh, a[1], a[2], a[3], a[4], splits, a[5])
            out = di.combine(o, ml)
            bad = (~torch.isfinite(out)).sum().item()
            err = max(((out[i] - r[i]).abs().max() / r[i].abs().max()).item() for i in range(b))
            good = bad == 0 and err < 2e-3
            ok &= good
            print(f"check paged {fmt} lens={lens} splits={splits}: max err {err:.2e}, non-finite {bad} -> "
                  f"{'ok' if good else 'FAIL'}", flush=True)
    print("all paged checks", "ok" if ok else "FAIL", flush=True)
    if os.environ.get("KV4_CHECK_ONLY") == "1":  # tests/gpu: correctness only, exit status = result
        sys.exit(0 if ok else 1)
    for fmt in ("nvfp4", "mxfp4"):
        for b, s, splits in [(32, 16384, 1), (8, 65536, 4), (128, 16384, 1)]:
            torch.manual_seed(0)
            q = torch.randn(b, HQ, D, device="cuda")
            k = torch.randn(b, HKV, s, D, device="cuda", dtype=torch.bfloat16)
            kq, vq = quantize_chunked(k, fmt, chunk=1), quantize_chunked(torch.randn_like(k), fmt, chunk=1)
            del k
            a = df.args_for(kq, vq, SM)
            qh = df.prep_q(q, HKV)
            nb = b * HKV * s * 2 * (72 if fmt == "nvfp4" else 68)
            md = timeit(lambda: e.decode_fp4(*a[0], qh, a[1], a[2], a[3], a[4], splits, 2, a[5]))
            res = []
            for shuffle in (False, True):
                pages, bt, sl = df.pack_pages(a[0], [s] * b, poison=False, shuffle=shuffle)
                mp = timeit(lambda: e.decode_fp4_paged(pages, bt, sl, qh, a[1], a[2], a[3], a[4], splits, a[5]))
                res.append(mp)
                if not shuffle:  # diagnostics, valid only for in-order pages at full length
                    o0 = e.decode_fp4_paged(pages, bt, sl, qh, a[1], a[2], a[3], a[4], splits, a[5])[0]
                    ph = pages.transpose(0, 1).contiguous()
                    for dg, pg in ((3, pages), (5, ph)):
                        od = e.decode_fp4_paged(pg, bt, sl, qh, a[1], a[2], a[3], a[4], splits, a[5], dg)[0]
                        same = torch.equal(od, o0)
                        res.append(timeit(lambda: e.decode_fp4_paged(pg, bt, sl, qh, a[1], a[2], a[3], a[4], splits,
                                                                     a[5], dg)))
                        print(f"  diag {dg} output identical to paged: {same}", flush=True)
                    del ph
                del pages, bt
            print(f"{fmt} B={b:3d} S={s}: dense {md:.4f} ms ({nb / md / 1e6 / peak:.0%}), paged in order "
                  f"{res[0]:.4f} ms ({res[0] / md:.3f}x), no block-table loads {res[1] / md:.3f}x, head-major pool "
                  f"{res[2] / md:.3f}x, paged shuffled {res[3]:.4f} ms ({res[3] / md:.3f}x)", flush=True)
            del a, kq, vq
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
