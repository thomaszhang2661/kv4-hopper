"""Python side of the C1 prototype (kv4/cuda/decode_fp4_k8.cu, 11-21): NVFP4 decode with the K side on FP8 tensor
cores. K: kv4.formats "nvfp4_g32" (E2M1, one E4M3 scale per 32 dims); V: "nvfp4" (decode_fp4.py's layouts); q: E4M3
with one scale per tensor.

    from kv4.cuda import decode_fp4_k8 as k8
    cache, score_mul, out_mul = k8.args_for(kq, vq, sm)     # kq nvfp4_g32, vq nvfp4
    q8, qs = k8.prep_q8(q, hkv)
    o, ml = k8.ext().decode_k8(*cache, q8, score_mul * qs, out_mul, splits)
"""
from __future__ import annotations

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
        _EXT = load(name="kv4_decode_fp4_k8", sources=[str(here / "decode_fp4_k8.cu")],
                    extra_include_paths=[str(here)], extra_cflags=["-O3", "-std=c++20"],
                    extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a",
                                       "--use_fast_math", "-lineinfo",
                                       "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                                       "-U__CUDA_NO_HALF2_OPERATORS__"],
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
