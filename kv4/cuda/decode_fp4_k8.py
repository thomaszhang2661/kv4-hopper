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


# ---- 11-22: C2, P V on FP8 tensor cores (docs/c2_log.md) --------------------------------------------------------
# V: E2M1 with one E4M3 scale per (channel, 32 tokens of a page) times a per-layer FP32 tensor scale ts_v (formats
# "nvfp4_t32" with a given ts). Page (9216 bytes): K as K8 | V codes 4096 in the C2 word layout | V scales 512 (256
# used). P is rounded to E4M3 at P = exp2(s - m + 8); the kernel's l carries 2^8.
P_SHIFT_C2 = 8.0
E2M1_MAX = 6.0


def _c2_v_layout():
    """C2 V words of one 64-token page: idx (1024, 8) flat value index o * 128 + d (token o of the page, channel d),
    bits (1024, 8, 4) positions of (m, e0, e1, s). Word of (channel d = 64 mt + 16 wi + 8 h8 + g, group gi = o // 32,
    lane t): ((mt * 4 + wi) * 32 + 4 g + t) * 4 + 2 gi + h8; it holds group positions p = 4t + j (slot L, byte j, bits
    2, 3, 4, 7) and p = 16 + 4t + j (slot H, byte j, bits 0, 1, 6, 5), i.e. C1's K word with dims read as tokens."""
    idx = [[-1] * 8 for _ in range(1024)]
    bits = [[None] * 8 for _ in range(1024)]
    for o in range(64):
        gi, p = divmod(o, 32)
        t, j, hs = (p % 16) // 4, p % 4, p // 16
        for d in range(D):
            mt, wi, h8, g = d // 64, (d // 16) % 4, (d // 8) % 2, d % 8
            w = ((mt * 4 + wi) * 32 + 4 * g + t) * 4 + 2 * gi + h8
            idx[w][j + 4 * hs] = o * D + d
            bits[w][j + 4 * hs] = [8 * j + 2, 8 * j + 3, 8 * j + 4, 8 * j + 7] if hs == 0 else \
                [8 * j + 0, 8 * j + 1, 8 * j + 6, 8 * j + 5]
    idx, bits = torch.tensor(idx), torch.tensor(bits)
    assert sorted(idx.flatten().tolist()) == list(range(64 * D))
    assert all(sorted(sum(w, [])) == list(range(32)) for w in bits.tolist())
    return idx, bits


def c2_vscale_pos():
    """(2 * 128) int64: byte position (0..255) in the page's V scale region of the scale of (group gi, channel d) at
    index gi * 128 + d: (wi * 8 + g) * 8 + mt * 4 + 2 gi + h8."""
    pos = []
    for gi in range(2):
        for d in range(D):
            mt, wi, h8, g = d // 64, (d // 16) % 4, (d // 8) % 2, d % 8
            pos.append((wi * 8 + g) * 8 + mt * 4 + 2 * gi + h8)
    assert sorted(pos) == list(range(256))
    return torch.tensor(pos)


def e4m3_scale(amax: torch.Tensor, ts: float) -> torch.Tensor:
    """E4M3 of min(amax / 6 / ts, 448) with IEEE divisions, as the CUDA stores compute it (ts rounded to float32 like the
    kernel's float argument). Not `amax / 6.0 / ts`: PyTorch computes a CUDA tensor divided by a Python scalar as a
    multiply by the reciprocal, which can land one ulp off an exact E4M3 tie (11-22, job 12065757: amax / 6 / ts =
    2.625 gave 0x43 where the store's round-to-even gives 0x42; seen only with V spanning many binades)."""
    return (amax / torch.full_like(amax, E2M1_MAX) / torch.full_like(amax, ts)).clamp(max=E4M3_MAX).to(
        torch.float8_e4m3fn)


def quantize_v_c2(v: torch.Tensor, ts: float, lens=None):
    """V (B, HKV, S, D), S a multiple of 32 -> (codes (B, HKV, S, D) uint8 E2M1, scales (B, HKV, D, S // 32) uint8 E4M3)
    with one scale per (channel, 32 tokens) and the GIVEN tensor scale ts, the rule of formats "nvfp4_t32" (and of the
    C2 store when a group is written in one call). Tokens >= lens[b] count as zero (their codes are 0), as the store
    sees a partly filled last group. One batch row at a time (memory)."""
    from kv4.formats import e2m1_encode
    B, H, S, _ = v.shape
    assert S % 32 == 0
    tiny = torch.finfo(torch.float32).tiny
    codes = torch.empty(B, H, S, D, dtype=torch.uint8, device=v.device)
    scales = torch.empty(B, H, D, S // 32, dtype=torch.uint8, device=v.device)
    for b in range(B):
        x = v[b].float()
        if lens is not None and int(lens[b]) < S:
            x = x.clone()
            x[:, int(lens[b]):] = 0
        xb = x.reshape(H, S // 32, 32, D)
        s8 = e4m3_scale(xb.abs().amax(2), ts)                                                 # (H, G, D)
        sc = (s8.float() * ts).clamp(min=tiny)
        codes[b] = e2m1_encode(xb / sc[:, :, None, :]).reshape(H, S, D).to(torch.uint8)
        scales[b] = s8.view(torch.uint8).transpose(1, 2)
    return codes, scales


def dequant_v_c2(codes: torch.Tensor, scales: torch.Tensor, ts: float) -> torch.Tensor:
    """(codes, scales) of quantize_v_c2 -> V (B, HKV, S, D) fp32 as the kernel sees it (code x E4M3 scale x ts)."""
    from kv4.formats import e2m1_decode
    B, H, S, _ = codes.shape
    s = scales.view(torch.float8_e4m3fn).float().transpose(2, 3)          # (B, H, G, D)
    return (e2m1_decode(codes).reshape(B, H, S // 32, 32, D) * (s * ts)[:, :, :, None, :]).reshape(B, H, S, D)


def args_for_c2(kq: QTensor, ts_v: float, sm: float):
    """K (nvfp4_g32) -> (kp, ksc) dense K8 layouts, score_mul (without the q scale), out_mul (V bytes are E4M3 values
    x 2^-6, so out_mul = ts_v * 64)."""
    assert kq.fmt.block == 32 and kq.fmt.scale == "e4m3", "K must be nvfp4_g32"
    kc = (pack_k8(kq.data), df.pack_scales(kq.scales, df._k_scale_perm_mx()))
    return kc, sm * 1.4426950408889634 * kq.tensor_scale * 64.0, ts_v * 64.0


def pack_pages_c2(kc, vcodes, vscales, lens, seed: int = 0, poison: bool = True, shuffle: bool = True):
    """Dense K8 K layouts kc = (kp, ksc) and C2 V (vcodes (B, HKV, S, D), vscales (B, HKV, D, S // 32) from
    quantize_v_c2) -> a pool of 64-token C2 pages as store_fp4.cu's C2 store writes them: (pages (num_pages, HKV, 9216)
    uint8, block_table (B, max_pages) int32, seq_lens (B) int32). With `poison`: K code rows and K scale bytes of
    tokens past each sequence's end are 0xFF, V codes of such tokens are random (finite in the E4M3 bit placement), and
    V scale bytes of groups that start past the end are 0xFF (E4M3 NaN), so a kernel that fails to mask shows NaN or a
    wrong result. The unused scale bytes (K 256..511, V 256..511 of their regions) are 0xFF."""
    kp, ksc = kc
    B, HKV, S, _ = kp.shape
    dev = kp.device
    kperm = df._k_scale_perm_mx().to(dev)
    idx, bits = (x.to(dev) for x in _c2_v_layout())
    vpos = c2_vscale_pos().to(dev)
    npg = [(int(l) + 63) // 64 for l in lens]
    total = sum(npg)
    g = torch.Generator().manual_seed(seed)
    num_pages = total + 7
    phys = (torch.randperm(num_pages, generator=g)[:total] if shuffle else torch.arange(total)).to(dev)
    pages = torch.full((num_pages, HKV, PAGE_BYTES), 0xFF, dtype=torch.uint8, device=dev)
    bt = torch.zeros(B, max(npg), dtype=torch.int32, device=dev)
    k = 0
    for b in range(B):
        n, L = npg[b], int(lens[b])
        assert n * 64 <= S, "S must cover whole pages"
        kc_ = kp[b, :, : n * 64].contiguous().view(torch.uint8).reshape(HKV, n * 64, 64).clone()
        ks = ksc[b, :, : n * 64].contiguous().reshape(HKV, -1).clone()
        vc = vcodes[b, :, : n * 64].clone()                                   # (HKV, n * 64, D)
        if poison and L < n * 64:
            kc_[:, L:] = 0xFF
            pk = torch.arange(n * 256, device=dev)
            ks[:, ((pk // 1024) * 256 + kperm[pk % 1024] // 4) >= L] = 0xFF
            vc[:, L:] = torch.randint(0, 16, vc[:, L:].shape, device=dev, dtype=torch.uint8)
        vcr = vc.reshape(HKV, n, 64 * D)
        words = torch.cat([df._pack(vcr[:, i: i + 128], idx, bits) for i in range(0, n, 128)], 1)  # (HKV, n, 1024)
        vs = torch.full((HKV, n, 512), 0xFF, dtype=torch.uint8, device=dev)
        sc = vscales[b, :, :, : 2 * n].reshape(HKV, D, n, 2).permute(0, 2, 3, 1).reshape(HKV, n, 256)  # gi * 128 + d
        if poison:
            gstart = torch.arange(2 * n, device=dev).reshape(n, 2) * 32
            sc = torch.where((gstart >= L)[None, :, :, None].expand(HKV, n, 2, D).reshape(HKV, n, 256),
                             torch.full_like(sc, 0xFF), sc)
        vs[:, :, vpos] = sc
        pad = torch.full((HKV, n, 256), 0xFF, dtype=torch.uint8, device=dev)
        ids = phys[k: k + n]
        k += n
        pages[ids] = torch.cat([kc_.reshape(HKV, n, 4096), ks.reshape(HKV, n, 256), pad,
                                words.contiguous().view(torch.uint8).reshape(HKV, n, 4096), vs], -1).transpose(0, 1)
        bt[b, :n] = ids.to(torch.int32)
    return pages.contiguous(), bt.contiguous(), torch.tensor([int(x) for x in lens], dtype=torch.int32, device=dev)


_EMPTY = {}


def _empty_i32(dev):
    if dev not in _EMPTY:
        _EMPTY[dev] = torch.empty(0, dtype=torch.int32, device=dev)
    return _EMPTY[dev]


def decode_paged_c2(pages, block_table, seq_lens, q8, qs, score_mul, out_mul, splits, gqa, sink_map=None,
                    sink_owner=None, nsk: int = 0):
    """C2 decode: (q8, qs) from merge.prep_q_rot8 (or prep_q8); score_mul without the q scale, out_mul = ts_v * 64
    (StoreC2.decode_args). With sink_map (and sink_owner, None = timing mode) and nsk = 4, the first 4 tokens of
    sequences with a valid sink slot are masked; merge with ext().merge_sink_pool_c2. Returns (o, ml)."""
    e = _empty_i32(pages.device)
    return ext().decode_c2_paged(pages, block_table, seq_lens, q8, qs, score_mul, out_mul, splits, gqa,
                                 e if sink_map is None else sink_map, e if sink_owner is None else sink_owner,
                                 nsk if sink_map is not None else 0)
