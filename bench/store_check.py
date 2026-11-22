"""Check the MXFP4 quantize-on-write kernel (kv4/cuda/store_fp4.cu, 11-12) against the offline packing path.

1. Bytes: K/V written token by token, in random order, into pages pre-filled with garbage, against
   formats.quantize + decode_fp4.args_for + pack_pages (in-order pages, lengths multiple of 64 so every byte of a used
   page belongs to a real token). Unrotated K must match exactly; with the Hadamard the writer's butterflies and
   formats.rotate's matmul round differently, so a few codes at rounding boundaries may differ (counted).
2. Decode: the written pages through decode_fp4_paged with the fixed exponent offset (StoreMXFP4.decode_args) against
   FP32 attention over the reference dequantized cache; also with K mean removal.
3. Speed of the writer for one decode step (B tokens) and for a prefill chunk.

    python bench/store_check.py
"""
import math
import os
import sys

import torch

from bench.bench_decode import timeit
from kv4.cuda import decode_fp4 as df
from kv4.cuda import merge as mg
from kv4.cuda.store_fp4 import StoreMXFP4
from kv4.formats import dequantize, quantize, rotate

HQ, HKV, D = 32, 8, 128
SM = 1 / math.sqrt(D)


def reference_pages(k, v, lens, kf, vf):
    kq, vq = quantize(k, kf), quantize(v, vf)
    a = df.args_for(kq, vq, SM)
    pages, bt, sl = df.pack_pages(a[0], lens, poison=False, shuffle=False)
    return pages, bt, sl, kq, vq


def main():
    print(torch.cuda.get_device_name(), "commit", os.environ.get("KV4_COMMIT", "unset"), flush=True)
    torch.manual_seed(0)
    lens = [320, 1024, 64]
    B, S = len(lens), 1024
    off = torch.randn(1, HKV, 1, D, device="cuda") * 2
    off[..., 7] = 30.0                                     # a large shared channel offset, as in real K
    k = (torch.randn(B, HKV, S, D, device="cuda") + off).to(torch.bfloat16)
    v = torch.randn(B, HKV, S, D, device="cuda", dtype=torch.bfloat16) * 0.5
    q = torch.randn(B, HQ, D, device="cuda", dtype=torch.bfloat16)
    ok = True
    for rot in (False, True):
        for smooth in (False, True):
            kf = "mxfp4_c0.200_h" if rot else "mxfp4_c0.200"
            mu = k[:, :, :512].float().mean((0, 2)) if smooth else None          # (HKV, D)
            kin = k.float() - mu[None, :, None] if smooth else k.float()   # no bf16 rounding: the writer has none
            ref, bt, sl, kq, vq = reference_pages(kin, v.float(), lens, kf, "mxfp4_c0.200")
            st = StoreMXFP4("cuda", rot_k=rot, c=0.2)
            pages = torch.full_like(ref, 0xAB)
            toks = [(b, t) for b in range(B) for t in range(lens[b])]
            order = torch.randperm(len(toks)).tolist()
            for i0 in range(0, len(order), 257):           # several calls, each with tokens of many pages
                sel = [toks[i] for i in order[i0:i0 + 257]]
                bi = torch.tensor([b for b, _ in sel], device="cuda")
                ti = torch.tensor([t for _, t in sel], device="cuda")
                slots = (bt[bi, ti // 64].long() * 64 + ti % 64)
                st(k[bi, :, ti], v[bi, :, ti], slots, pages, mu)   # the writer subtracts mu itself
            used = torch.unique(bt[bt >= 0].long())
            used = used[used < ref.shape[0]]
            diff = (pages[used] != ref[used])
            nbytes = int(diff.sum())
            frac = nbytes / diff.numel()
            # the decode check: the written cache against FP32 attention over the reference quantized values
            qh = mg.prep_q_rot(q, HKV, rot)
            o, ml = df.ext().decode_fp4_paged(pages, bt, sl, qh, *StoreMXFP4.decode_args(SM), 4, True)
            out = mg.ext().merge(o, ml).float()
            kd = dequantize(kq, rotate_back=False).float()
            vd = dequantize(vq, rotate_back=False).float()
            qe = (rotate(q.float(), 128) if rot else q.float()).half().float().reshape(B, HKV, HQ // HKV, D)
            errs = []
            for b in range(B):
                L = lens[b]
                p = torch.softmax(torch.einsum("hgd,hsd->hgs", qe[b], kd[b, :, :L]) * SM, -1)
                r = torch.einsum("hgs,hsd->hgd", p, vd[b, :, :L]).reshape(HQ, D)
                errs.append(((out[b] - r).abs().max() / r.abs().max()).item())
            err = max(errs)
            good = err < 5e-3 and (frac == 0 if not rot else frac < 1e-3)
            ok &= good
            print(f"rot={rot} mean_removal={smooth}: {nbytes} of {diff.numel()} page bytes differ ({frac:.2e}); "
                  f"decode max err {err:.2e} -> {'ok' if good else 'FAIL'}", flush=True)
    print("store checks", "ok" if ok else "FAIL", flush=True)
    st = StoreMXFP4("cuda", rot_k=True, c=0.2)
    pool = torch.zeros(4096, HKV, 8704, dtype=torch.uint8, device="cuda")
    for n in (1, 32, 128, 8192):
        kk = torch.randn(n, HKV, D, device="cuda", dtype=torch.bfloat16)
        slots = torch.randperm(4096 * 64, device="cuda")[:n]
        t = timeit(lambda: st(kk, kk, slots, pool))
        print(f"store {n:5d} tokens x {HKV} heads: {t * 1000:.1f} us", flush=True)
    # 4. (11-12) the CUDA dequant for vLLM's continuation prefill against the torch reference, then its speed
    from kv4.cuda.store_fp4 import dequant_pages, dequant_pages_cuda
    n = 3000                                                # not a multiple of 64: a partial last page
    ids = torch.randperm(4096, device="cuda")[: math.ceil(n / 64)].int()
    kk = (torch.randn(n, HKV, D, device="cuda") * 3 + 20 * (torch.arange(D, device="cuda") == 5)).to(torch.bfloat16)
    vv = torch.randn(n, HKV, D, device="cuda", dtype=torch.bfloat16)
    slots = (ids.long()[torch.arange(n, device="cuda") // 64] * 64 + torch.arange(n, device="cuda") % 64)
    st(kk, vv, slots, pool)
    kr, vr = dequant_pages(pool, ids, n, st.tabs, rot_k=True)
    kc, vc = dequant_pages_cuda(pool, ids, n, st.tabs, rot_k=True)
    v_ok = torch.equal(vc, vr.to(torch.bfloat16))
    k_bad = (kc != kr.to(torch.bfloat16)).float().mean().item()
    k_err = ((kc.float() - kr).abs().max() / kr.abs().max()).item()
    good = v_ok and k_err < 1e-2
    ok &= good
    print(f"dequant_pages_cuda: V identical {v_ok}; K {k_bad:.2e} of values differ, max rel err {k_err:.2e} "
          f"-> {'ok' if good else 'FAIL'}", flush=True)
    for n in (4096, 32768):
        ids = torch.arange(n // 64, device="cuda", dtype=torch.int32)
        t1 = timeit(lambda: dequant_pages_cuda(pool, ids, n, st.tabs))
        t0 = timeit(lambda: dequant_pages(pool, ids, n, st.tabs))
        print(f"dequant {n:5d} tokens x {HKV} heads: cuda {t1 * 1000:.0f} us, torch {t0 * 1000:.0f} us", flush=True)
    ok &= nvfp4_checks(k, v, q, lens)
    ok &= sink_pool_check(k, v, q, lens)
    ok &= sink_pool_check(k, v, q, lens, mean=False)              # 11-13: sinks without mean removal
    ok &= sink_pool_check(k, v, q, lens, mean=False, kscale=4.0)  # and with a larger K (bigger channel offset)
    print("all checks", "ok" if ok else "FAIL", flush=True)
    if os.environ.get("KV4_CHECK_ONLY") == "1":  # tests/gpu: correctness only, exit status = result
        sys.exit(0 if ok else 1)


def sink_pool_check(k, v, q, lens, mean=True, kscale=1.0):
    """6. (11-12 night) merge_sink_pool, the vLLM plugin's sinks: NVFP4 pages with K mean removal, a strong attention
    sink at token 0, BF16 sinks for sequences 0 and 1 and an invalid slot for sequence 2 (it must merge as plain 4-bit).
    Reference: FP32 attention whose first 4 keys / values are the BF16 ones (K - mu) and the rest the 4-bit values."""
    from kv4.cuda.store_fp4 import StoreNVFP4
    B, NSK = len(lens), 4
    k = k.clone()
    k[:, :, 0] = k[:, :, 0] * 4 + 20                        # a dominant sink key
    k = k * kscale                                          # 11-13: also a large shared K offset (Qwen-like)
    mu = k[:, :, 4:512].float().mean((0, 2)) if mean else torch.zeros(HKV, D, device="cuda")
    kin = k.float() - mu[None, :, None]
    ref, bt, sl, kq, vq = reference_pages(kin, v.float(), lens, "nvfp4_h", "nvfp4")
    st = StoreNVFP4("cuda", kq.tensor_scale, vq.tensor_scale, rot_k=True)
    pages = torch.full_like(ref, 0xAB)
    for b in range(B):
        L = lens[b]
        ti = torch.arange(L, device="cuda")
        st(k[b, :, :L].transpose(0, 1), v[b, :, :L].transpose(0, 1), bt[b, ti // 64].long() * 64 + ti % 64, pages,
           mu if mean else None)
    P = 8
    sk = torch.zeros(P, HKV, NSK, D, device="cuda", dtype=torch.bfloat16)
    sv, k4, v4 = torch.zeros_like(sk), torch.zeros_like(sk), torch.zeros_like(sk)
    owner = torch.full((P,), -1, device="cuda", dtype=torch.int32)
    smap = torch.full((1 << 12,), -1, device="cuda", dtype=torch.int32)
    for b in range(B):
        slot = 3 + b
        sk[slot] = kin[b, :, :NSK].to(torch.bfloat16)
        sv[slot] = v[b, :, :NSK]
        kr, vr = st.dequant_raw(pages, bt[b, :1], 64)
        k4[slot] = kr[:NSK].transpose(0, 1)
        v4[slot] = vr[:NSK].transpose(0, 1)
        owner[slot] = bt[b, 0] if b < 2 else -7               # sequence 2: a reused slot -> plain 4-bit
        smap[bt[b, 0].long()] = slot
    qh = mg.prep_q_rot(q, HKV, True)
    o, ml = df.ext().decode_fp4_paged(pages, bt, sl, qh, *st.decode_args(SM), 4, False)
    G = HQ // HKV
    out = mg.ext().merge_sink_pool(o, ml, q, qh, bt, smap, owner, sk, sv, k4, v4, G, SM * 1.4426950408889634,
                                   st.ts_k, st.ts_v, mg.P_SHIFT).float()
    kd, vd = dequantize(kq).float(), dequantize(vq).float()       # original basis (K rotated back)
    qe = q.float().reshape(B, HKV, G, D)
    errs = []
    for b in range(B):
        L = lens[b]
        kk, vv = kd[b, :, :L].clone(), vd[b, :, :L].clone()
        if b < 2:
            kk[:, :NSK], vv[:, :NSK] = sk[3 + b].float(), sv[3 + b].float()
        p = torch.softmax(torch.einsum("hgd,hsd->hgs", qe[b], kk) * SM, -1)
        r = torch.einsum("hgs,hsd->hgd", p, vv).reshape(HQ, D)
        errs.append(((out[b] - r).abs().max() / r.abs().max()).item())
    # what the BF16 sinks change: the same reference without them (all 4-bit) for sequence 0
    p0 = torch.softmax(torch.einsum("hgd,hsd->hgs", qe[0], kd[0, :, :lens[0]]) * SM, -1)
    r0 = torch.einsum("hgs,hsd->hgd", p0, vd[0, :, :lens[0]]).reshape(HQ, D)
    pk = torch.softmax(torch.einsum("hgd,hsd->hgs", qe[0], torch.cat([sk[3].float(), kd[0, :, NSK:lens[0]]], 1)) * SM, -1)
    gap = ((out[0] - r0).abs().max() / r0.abs().max()).item()
    good = max(errs) < 5e-3
    print(f"merge_sink_pool (mean removal {mean}, K scale {kscale}): max err {max(errs):.2e} (per sequence {', '.join(f'{e:.1e}' for e in errs)}; sequence 2 "
          f"is the fallback); sinks vs all-4-bit differ by {gap:.2e}; sink weight {pk[..., :NSK].sum(-1).mean():.2f} "
          f"-> {'ok' if good else 'FAIL'}", flush=True)
    return good


def nvfp4_checks(k, v, q, lens):
    """5. (11-12 night) NVFP4 pages (StoreNVFP4, 9216 bytes per page and head): bytes against formats.quantize
    ("nvfp4" / "nvfp4_h") + pack_pages with the same tensor scales, decode through decode_fp4_paged(mx=False), and the
    NVFP4 dequant for the continuation prefill against formats.dequantize."""
    from kv4.cuda.store_fp4 import StoreNVFP4
    B = len(lens)
    ok = True
    for rot, smooth in ((False, False), (True, False), (True, True)):
        kf = "nvfp4_h" if rot else "nvfp4"
        mu = k[:, :, :512].float().mean((0, 2)) if smooth else None
        kin = k.float() - mu[None, :, None] if smooth else k.float()
        ref, bt, sl, kq, vq = reference_pages(kin, v.float(), lens, kf, "nvfp4")
        st = StoreNVFP4("cuda", kq.tensor_scale, vq.tensor_scale, rot_k=rot)
        pages = torch.full_like(ref, 0xAB)
        assert pages.shape[-1] == StoreNVFP4.PAGE_BYTES
        toks = [(b, t) for b in range(B) for t in range(lens[b])]
        order = torch.randperm(len(toks)).tolist()
        for i0 in range(0, len(order), 257):
            sel = [toks[i] for i in order[i0:i0 + 257]]
            bi = torch.tensor([b for b, _ in sel], device="cuda")
            ti = torch.tensor([t for _, t in sel], device="cuda")
            st(k[bi, :, ti], v[bi, :, ti], bt[bi, ti // 64].long() * 64 + ti % 64, pages, mu)
        used = torch.unique(bt[bt >= 0].long())
        used = used[used < ref.shape[0]]
        diff = pages[used] != ref[used]
        frac = int(diff.sum()) / diff.numel()
        qh = mg.prep_q_rot(q, HKV, rot)
        o, ml = df.ext().decode_fp4_paged(pages, bt, sl, qh, *st.decode_args(SM), 4, False)
        out = mg.ext().merge(o, ml).float()
        kd, vd = dequantize(kq, rotate_back=False).float(), dequantize(vq, rotate_back=False).float()
        qe = (rotate(q.float(), 128) if rot else q.float()).half().float().reshape(B, HKV, HQ // HKV, D)
        errs = []
        for b in range(B):
            L = lens[b]
            p = torch.softmax(torch.einsum("hgd,hsd->hgs", qe[b], kd[b, :, :L]) * SM, -1)
            r = torch.einsum("hgs,hsd->hgd", p, vd[b, :, :L]).reshape(HQ, D)
            errs.append(((out[b] - r).abs().max() / r.abs().max()).item())
        # dequant of sequence 1 (1024 tokens) against formats.dequantize (K back in the original basis)
        ids = bt[1, : lens[1] // 64]
        kc, vc = st.dequant(pages, ids, lens[1])
        kr = dequantize(kq).float()[1].transpose(0, 1)          # (S, HKV, D)
        vr = dequantize(vq).float()[1].transpose(0, 1)
        dk = ((kc.float() - kr[: lens[1]]).abs().max() / kr.abs().max()).item()
        dv = ((vc.float() - vr[: lens[1]]).abs().max() / vr.abs().max()).item()
        good = max(errs) < 5e-3 and (frac == 0 if not rot else frac < 1e-3) and dk < 1e-2 and dv < 1e-2
        ok &= good
        print(f"NVFP4 rot={rot} mean_removal={smooth}: page bytes differ {frac:.2e}; decode max err {max(errs):.2e}; "
              f"dequant K {dk:.2e} V {dv:.2e} -> {'ok' if good else 'FAIL'}", flush=True)
    return ok


if __name__ == "__main__":
    main()
