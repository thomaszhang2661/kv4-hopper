"""Python side of the C1 prototype (kv4/cuda/decode_fp4_k8.cu, 11-21): NVFP4 decode with the K side on FP8 tensor
cores. K: kv4.formats "nvfp4_g32" (E2M1, one E4M3 scale per 32 dims); V: "nvfp4" (decode_fp4.py's layouts); q: E4M3
with one scale per tensor.

    from kv4.cuda import decode_fp4_k8 as k8
    cache, score_mul, out_mul = k8.args_for(kq, vq, sm)     # kq nvfp4_g32, vq nvfp4
    q8, qs = k8.prep_q8(q, hkv)
    o, ml = k8.ext().decode_k8(*cache, q8, score_mul * qs, out_mul, splits)
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from kv4.cuda import decode_fp4 as df
from kv4.formats import QTensor, dequantize, unpack_nibbles

NPAD, D = 8, 128
E4M3_MAX = 448.0
_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        here = Path(__file__).parent
        # 11-22: KV4_K8V selects the kernel variant switch K8V of decode_fp4_k8.cu (default: the .cu's own default, 3);
        # e.g. KV4_K8V=7 applies the G = 8 register fix to G = 4 too. Each variant gets its own extension name.
        v = os.environ.get("KV4_K8V", "")
        assert v == "" or v.isdigit(), f"KV4_K8V {v}"
        _EXT = load(name="kv4_decode_fp4_k8" + (f"_v{v}" if v else ""), sources=[str(here / "decode_fp4_k8.cu")],
                    extra_include_paths=[str(here)], extra_cflags=["-O3", "-std=c++20"],
                    extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a",
                                       "--use_fast_math", "-lineinfo",
                                       "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                                       "-U__CUDA_NO_HALF2_OPERATORS__"] + ([f"-DK8V={v}"] if v else []),
                    verbose=True)
    return _EXT


def _k8_layout():
    """Per token row: 16 words (lane t = word // 4, k-step i = word % 4); bits (m, e0, e1, s) of each of 8 values."""
    idx, bits = [], []
    for t in range(4):
        for i in range(4):
            a, b = [], []
            for j in range(4):
                a.append(32 * i + 4 * t + j)            # slot L: (s, e1, e0, m) at (7, 4, 3, 2)
                b.append([8 * j + 2, 8 * j + 3, 8 * j + 4, 8 * j + 7])
            for j in range(4):
                a.append(32 * i + 16 + 4 * t + j)       # slot H: (s, e1, e0, m) at (5, 6, 1, 0)
                b.append([8 * j + 0, 8 * j + 1, 8 * j + 6, 8 * j + 5])
            idx.append(a)
            bits.append(b)
    idx, bits = torch.tensor(idx), torch.tensor(bits)
    assert sorted(idx.flatten().tolist()) == list(range(D))
    assert all(sorted(sum(w, [])) == list(range(32)) for w in bits.tolist())
    return idx, bits


def pack_k8(data: torch.Tensor, chunk: int = 1) -> torch.Tensor:
    idx, bits = _k8_layout()
    idx, bits = idx.to(data.device), bits.to(data.device)
    out = torch.empty(*data.shape[:-1], 16, dtype=torch.int32, device=data.device)
    for i in range(0, data.shape[0], chunk):
        out[i:i + chunk] = df._pack(unpack_nibbles(data[i:i + chunk]), idx, bits)
    return out


def args_for(kq: QTensor, vq: QTensor, sm: float):
    """(kp, ksc, vp, vsc), score_mul (without the q scale), out_mul."""
    assert kq.fmt.block == 32 and kq.fmt.scale == "e4m3", "K must be nvfp4_g32"
    assert vq.fmt.name.startswith("nvfp4") and vq.fmt.block == 16
    cache = (pack_k8(kq.data), df.pack_scales(kq.scales, df._k_scale_perm_mx()),
             df.pack_v(vq.data), df.pack_scales(vq.scales, df._v_scale_perm()))
    # E4M3 placement is the value * 2^-6; V's FP16 placement is * 2^-14 (decode_fp4.cu)
    return cache, sm * 1.4426950408889634 * kq.tensor_scale * 64.0, vq.tensor_scale * 16384.0


def prep_q8(q: torch.Tensor, hkv: int):
    """(B, HQ, D) -> ((B, HKV, 8, D) uint8 E4M3, rows 4..7 zero; the FP32 scale qs, q ~ q8 * qs)."""
    b, hq, d = q.shape
    assert hq // hkv == 4, "the prototype supports 4 query heads per kv head"
    qs = max(q.float().abs().amax().item() / E4M3_MAX, 1e-12)
    q8 = torch.zeros(b, hkv, NPAD, d, device=q.device, dtype=torch.float8_e4m3fn)
    q8[:, :, :4] = (q.float() / qs).reshape(b, hkv, 4, d).to(torch.float8_e4m3fn)
    return q8.view(torch.uint8).contiguous(), qs


def reference(q: torch.Tensor, kq: QTensor, vq: QTensor, sm: float, q_e4m3: bool) -> torch.Tensor:
    """FP32 attention over the dequantized cache (rotated basis kept); q exact or rounded to E4M3 as the kernel sees
    it. (B, HKV, 4, D)."""
    b, hq, _ = q.shape
    hkv = kq.data.shape[1]
    qe = q.float()
    if q_e4m3:
        q8, qs = prep_q8(q, hkv)
        qe = q8.view(torch.float8_e4m3fn)[:, :, :4].float().reshape(b, hq, -1) * qs
    qe = qe.reshape(b, hkv, hq // hkv, -1)
    k, v = dequantize(kq, rotate_back=False).float(), dequantize(vq, rotate_back=False).float()
    p = torch.softmax(torch.einsum("bhgd,bhsd->bhgs", qe, k) * sm, -1)
    return torch.einsum("bhgs,bhsd->bhgd", p, v)


# ---- 11-21: paged K8 (C1 in the vLLM plugin) -------------------------------------------------------------------
PAGE_BYTES = 9216  # K codes 4096 | K scales 512 (first 256 used) | V codes 4096 | V scales 512


def pack_pages_k8(cache, lens, seed: int = 0, poison: bool = True, shuffle: bool = True):
    """Dense K8 layouts (kp, ksc, vp, vsc) from args_for -> a pool of 64-token K8 pages as store_fp4.cu's K8 store
    writes them: (pages (num_pages, HKV, 9216) uint8, block_table (B, max_pages) int32, seq_lens (B) int32), as
    decode_fp4.pack_pages. With `poison`, code rows and scale bytes of tokens past each sequence's end are 0xFF
    (E4M3 NaN scales), so a kernel that fails to mask them shows NaN; the unused K scale bytes 256..511 are 0xFF."""
    kp, ksc, vp, vsc = cache
    B, HKV, S, _ = kp.shape
    assert ksc.shape[-1] == 4 and vsc.shape[-1] == 8
    kperm, vperm = df._k_scale_perm_mx().to(kp.device), df._v_scale_perm().to(kp.device)
    npg = [(int(l) + 63) // 64 for l in lens]
    total = sum(npg)
    g = torch.Generator().manual_seed(seed)
    num_pages = total + 7
    phys = (torch.randperm(num_pages, generator=g)[:total] if shuffle else torch.arange(total)).to(kp.device)
    pages = torch.full((num_pages, HKV, PAGE_BYTES), 0xFF, dtype=torch.uint8, device=kp.device)
    bt = torch.zeros(B, max(npg), dtype=torch.int32, device=kp.device)
    k = 0
    for b in range(B):
        n, L = npg[b], int(lens[b])
        kc = kp[b, :, : n * 64].contiguous().view(torch.uint8).reshape(HKV, n * 64, 64).clone()
        ks = ksc[b, :, : n * 64].contiguous().reshape(HKV, -1).clone()
        vc = vp[b, :, : n * 64].contiguous().view(torch.uint8).reshape(HKV, n, 4096)
        vs = vsc[b, :, : n * 64].contiguous().reshape(HKV, -1).clone()
        if poison and L < n * 64:
            kc[:, L:] = 0xFF
            pk = torch.arange(n * 256, device=kp.device)
            ks[:, ((pk // 1024) * 256 + kperm[pk % 1024] // 4) >= L] = 0xFF
            pv = torch.arange(n * 512, device=kp.device)
            vs[:, ((pv // 2048) * 256 + vperm[pv % 2048] // 8) >= L] = 0xFF
        pad = torch.full((HKV, n, 256), 0xFF, dtype=torch.uint8, device=kp.device)
        ids = phys[k: k + n]
        k += n
        pages[ids] = torch.cat([kc.reshape(HKV, n, 4096), ks.reshape(HKV, n, 256), pad, vc, vs.reshape(HKV, n, 512)],
                               -1).transpose(0, 1)
        bt[b, :n] = ids.to(torch.int32)
    return pages.contiguous(), bt.contiguous(), torch.tensor([int(x) for x in lens], dtype=torch.int32,
                                                              device=kp.device)


def decode_paged(pages, block_table, seq_lens, qh, score_mul, out_mul, splits, gqa):
    """qh (B, HKV, 8, 128) fp16 (merge.prep_q_rot) -> (o, ml) of decode_k8_paged and qd, the dequantized E4M3 q the
    kernel used (fp16, same shape as qh). score_mul without the q scale (StoreK8.decode_args)."""
    q8, qs, qd = ext().prep_q8(qh)
    o, ml = ext().decode_k8_paged(pages, block_table, seq_lens, q8, qs, score_mul, out_mul, splits, gqa)
    return o, ml, qd


def decode_paged_q8(pages, block_table, seq_lens, q8, qs, score_mul, out_mul, splits, gqa):
    """11-21: as decode_paged with q already in E4M3: (q8, qs) from merge.prep_q_rot8 (rotation + E4M3 in one launch,
    the plugin's KV4_K8 path). Returns (o, ml)."""
    return ext().decode_k8_paged(pages, block_table, seq_lens, q8, qs, score_mul, out_mul, splits, gqa)
