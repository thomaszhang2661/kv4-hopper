"""Python side of the Hopper NVFP4 decode kernel (kv4/cuda/decode_fp4.cu, 11-10 night): build, write-time layouts
(codes with the FP16 bit-placement slots, scales arranged per thread), q, reference.

    from kv4.cuda import decode_fp4 as df
    kp, ksc, vp, vsc = df.prepare(kq, vq)        # QTensor nvfp4 (B, HKV, S, 64) -> kernel layout
    o, ml = df.ext().decode_nvfp4(kp, ksc, vp, vsc, df.prep_q(q, hkv), sm, kq.tensor_scale, vq.tensor_scale, splits, nst)
    out = di.combine(o, ml)                      # kv4.cuda.decode_int4.combine
"""
from __future__ import annotations

from pathlib import Path

import torch

from kv4.formats import QTensor, dequantize, unpack_nibbles

GQA, NPAD, D, STAGE = 4, 8, 128, 256
_EXT = None
# slot -> (value-bit positions for (m, e0, e1), sign position), low half; the high half is +16
SLOTS = {"A": ((9, 10, 11), 15), "B": ((6, 7, 8), 12), "C": ((0, 1, 2), 13), "D": ((3, 4, 5), 14)}


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        here = Path(__file__).parent
        _EXT = load(name="kv4_decode_fp4", sources=[str(here / "decode_fp4.cu")], extra_include_paths=[str(here)],
                    extra_cflags=["-O3", "-std=c++20"],
                    extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a",
                                       "--use_fast_math", "-lineinfo",
                                       "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                                       "-U__CUDA_NO_HALF2_OPERATORS__"],
                    verbose=True)
    return _EXT


def _word_bits(entries):
    """entries: list of (slot, half, flat index) -> (index list, bit list) for the 4 bits of each of the 8 values."""
    idx, bits = [], []
    for slot, half, f in entries:
        (p0, p1, p2), ps = SLOTS[slot]
        off = 16 * half
        idx.append(f)
        bits.append([p0 + off, p1 + off, p2 + off, ps + off])
    return idx, bits


def _pack(codes: torch.Tensor, idx: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    """codes (..., N) E2M1 codes (s e1 e0 m) -> words (..., W) int32; idx (W, 8) value indices, bits (W, 8, 4) positions
    of (m, e0, e1, s)."""
    c = codes[..., idx.flatten()].reshape(*codes.shape[:-1], *idx.shape).to(torch.int64)   # (..., W, 8)
    fields = torch.stack([c & 1, (c >> 1) & 1, (c >> 2) & 1, (c >> 3) & 1], -1)              # (..., W, 8, 4)
    w = (fields << bits).sum((-1, -2))
    return torch.where(w >= 2**31, w - 2**32, w).to(torch.int32)


def _k_layout():
    """Per token row: 16 words (lane t = word // 4, word i = word % 4) of dims (see the .cu header)."""
    idx, bits = [], []
    for t in range(4):
        for i in range(4):
            e = [("A", 0, 32 * i + 2 * t), ("A", 1, 32 * i + 2 * t + 1),
                 ("B", 0, 32 * i + 2 * t + 8), ("B", 1, 32 * i + 2 * t + 9),
                 ("C", 0, 32 * i + 16 + 2 * t), ("C", 1, 32 * i + 17 + 2 * t),
                 ("D", 0, 32 * i + 24 + 2 * t), ("D", 1, 32 * i + 25 + 2 * t)]
            a, b = _word_bits(e)
            idx.append(a)
            bits.append(b)
    idx, bits = torch.tensor(idx), torch.tensor(bits)
    assert sorted(idx.flatten().tolist()) == list(range(D))
    return idx, bits


def _v_layout():
    """Per 256-token block: 4096 words, flat index token * 128 + dim (see the .cu header)."""
    idx, bits = [], []
    for wg in range(2):
        for kh in range(2):
            for mt in range(2):
                for w in range(4):
                    for lane in range(32):
                        g, t = lane // 4, lane % 4
                        for kk in range(4):
                            T, Dm = 128 * wg + 16 * (4 * kh + kk), 64 * mt + 16 * w
                            f = lambda tok, dim: tok * D + dim  # noqa: E731
                            e = [("A", 0, f(T + 2 * t, Dm + g)), ("A", 1, f(T + 2 * t + 1, Dm + g)),
                                 ("B", 0, f(T + 2 * t, Dm + g + 8)), ("B", 1, f(T + 2 * t + 1, Dm + g + 8)),
                                 ("C", 0, f(T + 2 * t + 8, Dm + g)), ("C", 1, f(T + 2 * t + 9, Dm + g)),
                                 ("D", 0, f(T + 2 * t + 8, Dm + g + 8)), ("D", 1, f(T + 2 * t + 9, Dm + g + 8))]
                            a, b = _word_bits(e)
                            idx.append(a)
                            bits.append(b)
    idx, bits = torch.tensor(idx), torch.tensor(bits)
    assert sorted(idx.flatten().tolist()) == list(range(STAGE * D))
    return idx, bits


def _k_scale_perm():
    """Per 256-token block: 2048 byte positions -> flat index token * 8 + group."""
    out = []
    for wg in range(2):
        for sub in range(2):
            for w in range(4):
                for g in range(8):
                    r0 = 128 * wg + 64 * sub + 16 * w + g
                    for k in range(4):
                        out += [r0 * 8 + 2 * k, (r0 + 8) * 8 + 2 * k, r0 * 8 + 2 * k + 1, (r0 + 8) * 8 + 2 * k + 1]
    out = torch.tensor(out)
    assert torch.equal(out.sort().values, torch.arange(STAGE * 8))
    return out


def _v_scale_perm():
    out = []
    for wg in range(2):
        for kh in range(2):
            for mt in range(2):
                for w in range(4):
                    for t in range(4):
                        for kk in range(4):
                            for j in range(2):
                                for e in range(2):
                                    tok = 128 * wg + 16 * (4 * kh + kk) + 2 * t + 8 * j + e
                                    out.append(tok * 8 + 4 * mt + w)
    out = torch.tensor(out)
    assert torch.equal(out.sort().values, torch.arange(STAGE * 8))
    return out


def _k_scale_perm_mx():
    """MXFP4 K scales per 256-token block: 1024 byte positions -> flat index token * 4 + group (see the .cu header)."""
    out = []
    for wg in range(2):
        for sub in range(2):
            for w in range(4):
                for g in range(8):
                    r0 = 128 * wg + 64 * sub + 16 * w + g
                    for k in range(2):
                        out += [r0 * 4 + 2 * k, r0 * 4 + 2 * k + 1, (r0 + 8) * 4 + 2 * k, (r0 + 8) * 4 + 2 * k + 1]
    out = torch.tensor(out)
    assert torch.equal(out.sort().values, torch.arange(STAGE * 4))
    return out


def _v_scale_perm_mx():
    out = []
    for wg in range(2):
        for kh in range(2):
            for mt in range(2):
                for wp in range(2):
                    for t in range(4):
                        for kk in range(4):
                            T, grp = 128 * wg + 16 * (4 * kh + kk), 2 * mt + wp
                            out += [(T + 2 * t) * 4 + grp, (T + 2 * t + 8) * 4 + grp,
                                    (T + 2 * t + 1) * 4 + grp, (T + 2 * t + 9) * 4 + grp]
    out = torch.tensor(out)
    assert torch.equal(out.sort().values, torch.arange(STAGE * 4))
    return out


def mx_rebase(scales: torch.Tensor):
    """E8M0 bytes -> (B, negB, factor): FP16 exponent field F = e - B with B = emax - 30, so F <= 30; the kernel's
    scale is 2^(e - B - 15) = true scale * 2^(112 - B), and its products carry 2^-14 from the E2M1 placement."""
    emax, emin = int(scales.max()), int(scales.min())
    B = emax - 30
    assert emin - B >= 1, f"MXFP4 scale range {emax - emin} > 29 binades: not representable in one FP16 rebase"
    negB = (-(B * 1024 * 65537)) % 2**32
    if negB >= 2**31:
        negB -= 2**32
    return B, negB, 2.0 ** (14 + B - 112)


def args_for(kq: QTensor, vq: QTensor, sm: float):
    """Everything decode_fp4() needs after q: (kp, ksc, vp, vsc), score_mul, out_mul, negbk, negbv, mx."""
    mx = kq.fmt.scale.startswith("e8m0")  # OCP MX or UltraQuant scale rule: same bytes, same decode
    if mx:
        _, nbk, fk = mx_rebase(kq.scales)
        _, nbv, fv = mx_rebase(vq.scales)
        cache = (pack_k(kq.data), pack_scales(kq.scales, _k_scale_perm_mx()),
                 pack_v(vq.data), pack_scales(vq.scales, _v_scale_perm_mx()))
        return cache, sm * 1.4426950408889634 * fk, fv, nbk, nbv, True
    cache = prepare(kq, vq)
    return cache, sm * 1.4426950408889634 * kq.tensor_scale * 16384.0, vq.tensor_scale * 16384.0, 0, 0, False


def pack_k(data: torch.Tensor, chunk: int = 1) -> torch.Tensor:
    idx, bits = _k_layout()
    idx, bits = idx.to(data.device), bits.to(data.device)
    out = torch.empty(*data.shape[:-1], 16, dtype=torch.int32, device=data.device)
    for i in range(0, data.shape[0], chunk):
        out[i:i + chunk] = _pack(unpack_nibbles(data[i:i + chunk]), idx, bits)
    return out


def pack_v(data: torch.Tensor, chunk: int = 1) -> torch.Tensor:
    b, h, s, _ = data.shape
    assert s % STAGE == 0
    idx, bits = _v_layout()
    idx, bits = idx.to(data.device), bits.to(data.device)
    out = torch.empty(b, h, s, 16, dtype=torch.int32, device=data.device)
    for i in range(0, b, chunk):
        codes = unpack_nibbles(data[i:i + chunk])
        blocks = codes.reshape(codes.shape[0], h, s // STAGE, STAGE * D)
        out[i:i + chunk] = _pack(blocks, idx, bits).reshape(codes.shape[0], h, s, 16)
    return out


def pack_scales(scales: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """(B, HKV, S, 8) uint8 E4M3 bits -> same shape, 256-token blocks permuted."""
    b, h, s, n = scales.shape
    blocks = scales.reshape(b, h, s // STAGE, STAGE * n)
    return blocks[..., perm.to(scales.device)].reshape(b, h, s, n).contiguous()


def prepare(kq: QTensor, vq: QTensor):
    """NVFP4 layouts (MXFP4: use args_for)."""
    assert kq.fmt.name.startswith("nvfp4") and vq.fmt.name.startswith("nvfp4")
    return (pack_k(kq.data), pack_scales(kq.scales, _k_scale_perm()),
            pack_v(vq.data), pack_scales(vq.scales, _v_scale_perm()))


def prep_q(q: torch.Tensor, hkv: int) -> torch.Tensor:
    """(B, HQ, D) -> (B, HKV, 8, D) fp16; G = HQ / HKV query heads per kv head (4: rows 4..7 zero; 8: all rows)."""
    b, hq, d = q.shape
    g = hq // hkv
    assert g in (4, 8), "the kernel supports 4 or 8 query heads per kv head"
    qh = torch.zeros(b, hkv, NPAD, d, device=q.device, dtype=torch.float16)
    qh[:, :, :g] = q.reshape(b, hkv, g, d).half()
    return qh.contiguous()


def reference(q: torch.Tensor, kq: QTensor, vq: QTensor, sm: float) -> torch.Tensor:
    """FP32 attention over the dequantized cache (rotated basis kept, as the kernel sees it), q rounded to fp16."""
    b, hq, _ = q.shape
    hkv = kq.data.shape[1]
    qe = q.reshape(b, hkv, hq // hkv, -1).half().float()
    k, v = dequantize(kq, rotate_back=False).float(), dequantize(vq, rotate_back=False).float()
    p = torch.softmax(torch.einsum("bhgd,bhsd->bhgs", qe, k) * sm, -1)
    return torch.einsum("bhgs,bhsd->bhgd", p, v)


def pack_pages(cache, lens, seed: int = 0, poison: bool = True, shuffle: bool = True):
    """Dense kernel layouts (kp, ksc, vp, vsc) of shape (B, HKV, S, .) -> a shuffled pool of 64-token pages for
    decode_fp4_paged: (pages (num_pages, HKV, PAGE_BYTES) uint8, block_table (B, max_pages) int32, seq_lens (B) int32).
    Each (page, head) = K codes 4096 | K scales | V codes 4096 | V scales, i.e. one quarter of a 256-token stage block
    (decode_fp4.cu). With `poison`, the scale bytes of tokens past each sequence's end are set to 0xFF (NaN for E4M3)
    and their K code rows too, so a kernel that fails to mask them shows NaN."""
    kp, ksc, vp, vsc = cache
    B, HKV, S, _ = kp.shape
    nscale = ksc.shape[-1]                     # 8 (NVFP4) or 4 (MXFP4) scale bytes per token
    scq = 64 * nscale
    kperm = (_k_scale_perm() if nscale == 8 else _k_scale_perm_mx()).to(kp.device)
    vperm = (_v_scale_perm() if nscale == 8 else _v_scale_perm_mx()).to(kp.device)
    npg = [(int(l) + 63) // 64 for l in lens]
    total = sum(npg)
    g = torch.Generator().manual_seed(seed)
    num_pages = total + 7                      # a few unused pages in the pool
    phys = (torch.randperm(num_pages, generator=g)[:total] if shuffle else torch.arange(total)).to(kp.device)
    pages = torch.full((num_pages, HKV, 2 * (4096 + scq)), 0xFF, dtype=torch.uint8, device=kp.device)
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
            # scale byte position -> token: flat index (token * nscale + group) within its 256-token block
            pos = torch.arange(n * scq, device=kp.device)
            blk, off = pos // (256 * nscale), pos % (256 * nscale)
            ks[:, (blk * 256 + kperm[off] // nscale) >= L] = 0xFF
            vs[:, (blk * 256 + vperm[off] // nscale) >= L] = 0xFF
        ids = phys[k: k + n]
        k += n
        pages[ids] = torch.cat([kc.reshape(HKV, n, 4096), ks.reshape(HKV, n, scq), vc, vs.reshape(HKV, n, scq)],
                               -1).transpose(0, 1)
        bt[b, :n] = ids.to(torch.int32)
    return pages.contiguous(), bt.contiguous(), torch.tensor([int(x) for x in lens], dtype=torch.int32,
                                                              device=kp.device)
