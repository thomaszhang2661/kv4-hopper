"""Python side of the Hopper INT4 decode kernel (kv4/cuda/decode_int4.cu, 11-10).

Formats: int4a_tok (asymmetric, x = code * s + z) and int4_tok (symmetric, passed as z = -8 s); rotation (the _h
formats) is applied outside the kernel (K at write time, q before the call, the output rotated back if V is rotated),
so the kernel is the same.

    from kv4.cuda import decode_int4 as di
    cache = di.prepare(kq, vq)                  # QTensor int4_tok / int4a_tok (B, HKV, S, 64) -> kernel layout
    qp, qc = di.prep_q(q, hkv)                  # q (B, HQ, 128)
    o, ml = di.ext().decode_int4(*cache, qp, qc, sm, splits, nst)
    out = di.combine(o, ml)                     # (B, HKV, 4, 128) -> reshape to (B, HQ, 128)
"""
from __future__ import annotations

from pathlib import Path

import torch

from kv4.cuda import kside
from kv4.formats import QTensor, dequantize, unpack_nibbles

GQA, NPAD, D, STAGE = 4, 8, 128, 256
_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        here = Path(__file__).parent
        _EXT = load(name="kv4_decode_int4", sources=[str(here / "decode_int4.cu")], extra_include_paths=[str(here)],
                    extra_cflags=["-O3", "-std=c++20"],
                    extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a",
                                       "--use_fast_math", "-lineinfo",
                                       "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                                       "-U__CUDA_NO_HALF2_OPERATORS__"],
                    verbose=True)
    return _EXT


def _v_perm():
    """For each of the 4096 words of a 256-token V block, the 8 (token, dim, bit shift) entries (see the .cu header)."""
    tok, dim, sh = [], [], []
    for wg in range(2):
        for kh in range(2):
            for mt in range(2):
                for w in range(4):
                    for lane in range(32):
                        g, t = lane // 4, lane % 4
                        for kk in range(4):
                            T, Dm = 128 * wg + 16 * (4 * kh + kk), 64 * mt + 16 * w
                            e = [(0, Dm + g, T + 2 * t), (16, Dm + g, T + 2 * t + 1),
                                 (8, Dm + g + 8, T + 2 * t), (24, Dm + g + 8, T + 2 * t + 1),
                                 (4, Dm + g, T + 2 * t + 8), (20, Dm + g, T + 2 * t + 9),
                                 (12, Dm + g + 8, T + 2 * t + 8), (28, Dm + g + 8, T + 2 * t + 9)]
                            sh.append([s for s, _, _ in e])
                            dim.append([d for _, d, _ in e])
                            tok.append([x for _, _, x in e])
    tok, dim, sh = torch.tensor(tok), torch.tensor(dim), torch.tensor(sh)
    flat = (tok * D + dim).flatten()
    assert torch.equal(flat.sort().values, torch.arange(STAGE * D)), "V layout must cover each (token, dim) once"
    return tok, dim, sh


def pack_v(data: torch.Tensor, chunk: int = 1) -> torch.Tensor:
    """V codes (B, HKV, S, 64) uint8 -> (B, HKV, S, 16) int32 whose 256-token blocks are in the PV fragment layout."""
    b, h, s, _ = data.shape
    assert s % STAGE == 0
    tok, dim, sh = _v_perm()
    idx = (tok * D + dim).flatten().to(data.device)
    sh = sh.to(data.device)
    out = torch.empty(b, h, s, 16, dtype=torch.int32, device=data.device)
    for i in range(0, b, chunk):
        codes = unpack_nibbles(data[i:i + chunk]).to(torch.int64)                   # (c, h, s, 128)
        blocks = codes.reshape(codes.shape[0], h, s // STAGE, STAGE * D)            # token-major within a block
        g = blocks[..., idx].reshape(codes.shape[0], h, s // STAGE, 4096, 8)
        w = (g << sh).sum(-1)
        out[i:i + chunk] = torch.where(w >= 2**31, w - 2**32, w).to(torch.int32).reshape(codes.shape[0], h, s, 16)
    return out


def scale_zero(qt: QTensor) -> torch.Tensor:
    """(B, HKV, S, 2) fp16 (scale, zero) with x = code * scale + zero; symmetric int4 gets zero = -8 * scale."""
    assert qt.fmt.element == "int4" and qt.fmt.block == 0, "token-wise int4 only"
    s = qt.scales[..., 0]
    z = qt.zeros[..., 0] if qt.fmt.asym else (-8 * s.float()).half()
    return torch.stack([s.half(), z.half()], -1).contiguous()


def prepare(kq: QTensor, vq: QTensor):
    """Write-time layouts: (kp, ksz, vp, vsz) for the kernel."""
    return (kside.pack_kside(kq.data), scale_zero(kq), pack_v(vq.data), scale_zero(vq))


def prep_q(q: torch.Tensor, hkv: int):
    """qp (B, HKV, 8, D) fp16 = q / m_d (heads 4..7 zero); qc (B, HKV, 8, 2) fp32 = (1024 sum q'_d, sum q^_d) where
    q^ = q' m_d is the q the kernel effectively uses."""
    qp, _ = kside.prep_q(q, hkv)
    m = kside.dim_factor(q.device)
    qe = qp.float() * m
    qc = torch.stack([1024 * qp.float().sum(-1), qe.sum(-1)], -1)
    return qp, qc.contiguous()


def combine(out_o: torch.Tensor, out_ml: torch.Tensor) -> torch.Tensor:
    """Merge splits: (B, HKV, splits, 4, D) normalized partials and (m, l) in base 2 -> (B, HKV, 4, D)."""
    m, l = out_ml[..., 0], out_ml[..., 1]
    w = torch.exp2(m - m.amax(2, keepdim=True)) * l
    return (out_o * w[..., None]).sum(2) / w.sum(2)[..., None]


def reference(q: torch.Tensor, kq: QTensor, vq: QTensor, sm: float) -> torch.Tensor:
    """FP32 attention over the dequantized cache with q rounded as the kernel sees it: (B, HKV, 4, D)."""
    b, hq, _ = q.shape
    hkv = kq.data.shape[1]
    qp, _ = kside.prep_q(q, hkv)
    qe = qp[:, :, :GQA].float() * kside.dim_factor(q.device)
    k, v = dequantize(kq).float(), dequantize(vq).float()
    p = torch.softmax(torch.einsum("bhgd,bhsd->bhgs", qe, k) * sm, -1)
    return torch.einsum("bhgs,bhsd->bhgd", p, v)
