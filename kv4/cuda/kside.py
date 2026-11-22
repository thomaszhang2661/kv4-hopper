"""Python side of the K-side CUDA prototype (kv4/cuda/kside.cu, 11-10): build, write-time layout, q preparation,
reference.

    from kv4.cuda import kside
    kp = kside.pack_kside(qt.data)              # int4_tok codes (B, HKV, S, 64) uint8 -> (B, HKV, S, 16) int32
    qp, k0 = kside.prep_q(q, hkv)               # q (B, HQ, 128)
    ml = kside.ext().kside(kp, qt.scales[..., 0].contiguous(), qp, k0, sm, splits, nst)
"""
from __future__ import annotations

from pathlib import Path

import torch

from kv4.formats import unpack_nibbles

GQA, NPAD, D = 4, 8, 128
_EXT = None


def ext():
    """JIT-build the extension (sm_90a). Register and spill counts are printed by ptxas (-v) in the build log."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        _EXT = load(name="kv4_kside", sources=[str(Path(__file__).with_name("kside.cu"))],
                    extra_cflags=["-O3", "-std=c++20"],
                    extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a",
                                       "--use_fast_math", "-lineinfo", "-Xptxas=-v",
                                       "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__"],
                    verbose=True)
    return _EXT


def _perm():
    """For each of the 16 words of a packed row (thread t = word // 4, word i = word % 4): the 8 (dim, bit shift)
    pairs it holds, matching the wgmma A-fragment layout described in kside.cu."""
    dims, shifts = [], []
    for t in range(4):
        for i in range(4):
            e = [(0, 32 * i + 2 * t), (16, 32 * i + 2 * t + 1), (4, 32 * i + 2 * t + 8), (20, 32 * i + 2 * t + 9),
                 (8, 32 * i + 16 + 2 * t), (24, 32 * i + 17 + 2 * t), (12, 32 * i + 24 + 2 * t),
                 (28, 32 * i + 25 + 2 * t)]
            shifts.append([s for s, _ in e])
            dims.append([d for _, d in e])
    dims, shifts = torch.tensor(dims), torch.tensor(shifts)
    assert sorted(dims.flatten().tolist()) == list(range(D))
    return dims, shifts


def pack_kside(data: torch.Tensor, chunk: int = 1) -> torch.Tensor:
    """int4_tok packed bytes (..., 64) (element 2i in the low nibble) -> permuted words (..., 16) int32.
    Done once when the cache is written; chunked along dim 0 to bound memory."""
    dims, shifts = _perm()
    dims, shifts = dims.flatten().to(data.device), shifts.to(data.device)
    out = torch.empty(*data.shape[:-1], 16, dtype=torch.int32, device=data.device)
    for i in range(0, data.shape[0], chunk):
        codes = unpack_nibbles(data[i:i + chunk]).to(torch.int64)
        g = codes[..., dims].reshape(*codes.shape[:-1], 16, 8)
        w = (g << shifts).sum(-1)
        out[i:i + chunk] = torch.where(w >= 2**31, w - 2**32, w).to(torch.int32)
    return out


def dim_factor(device=None) -> torch.Tensor:
    """m_d: 16 for dims with d % 16 >= 8 (their codes come out of the fp16 trick multiplied by 16), else 1."""
    d = torch.arange(D, device=device)
    return torch.where(d % 16 >= 8, 16.0, 1.0)


def prep_q(q: torch.Tensor, hkv: int):
    """q (B, HQ, D) -> qp (B, HKV, 8, D) fp16 = q / m_d with heads 4..7 zero, and K0 (B, HKV, 8) fp32."""
    b, hq, d = q.shape
    assert d == D and hq == hkv * GQA
    m = dim_factor(q.device)
    qp = torch.zeros(b, hkv, NPAD, D, device=q.device, dtype=torch.float16)
    qp[:, :, :GQA] = (q.float().reshape(b, hkv, GQA, D) / m).half()
    k0 = (qp.float() * (1024 + 8 * m)).sum(-1)
    return qp.contiguous(), k0.contiguous()


def core_matrix(qp: torch.Tensor) -> torch.Tensor:
    """qp (B, HKV, 8, D) -> the shared-memory order the kernel's wgmma descriptor expects (core matrix kc = dims
    8kc..8kc+7 of all 8 heads, 128 bytes each), so it can be copied with one bulk copy: (B, HKV, D/8, 8, 8)."""
    b, h, n, d = qp.shape
    return qp.reshape(b, h, n, d // 8, 8).permute(0, 1, 3, 2, 4).contiguous()


def reference(q: torch.Tensor, data: torch.Tensor, scales: torch.Tensor, sm: float, splits: int):
    """Scores (B, HKV, 4, S) in base 2 and per-split (m, l) (B, HKV, splits, 4, 2), with q rounded exactly as the
    kernel sees it (fp16 q' times m_d) and K dequantized in fp32."""
    b, hq, _ = q.shape
    hkv = data.shape[1]
    qp, _ = prep_q(q, hkv)
    qe = qp[:, :, :GQA].float() * dim_factor(q.device)                        # (B, HKV, 4, D)
    k = (unpack_nibbles(data).float() - 8) * scales.float()                    # (B, HKV, S, D)
    s2 = torch.einsum("bhgd,bhsd->bhgs", qe, k) * (sm * 1.4426950408889634)    # base 2
    S = s2.shape[-1]
    parts = s2.reshape(b, hkv, GQA, splits, S // splits)
    m = parts.amax(-1)
    l = torch.exp2(parts - m[..., None]).sum(-1)
    return s2, torch.stack([m, l], -1).permute(0, 1, 3, 2, 4).contiguous()
