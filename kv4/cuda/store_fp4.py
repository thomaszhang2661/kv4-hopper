"""Quantize-on-write into 64-token MXFP4 pages (kv4/cuda/store_fp4.cu, 11-12), for a serving cache (vLLM).

    st = StoreMXFP4(device, rot_k=True, c=0.2)            # c <= 0: OCP rule
    st(key, value, slot_mapping, pages)                    # key/value (N, HKV, 128) bf16, slots int64, pages uint8
    o, ml = df.ext().decode_fp4_paged(pages, block_table, seq_lens, qh, *st.decode_args(sm), splits, True)

The decode kernel turns E8M0 scales into FP16 exponents with an offset B (decode_fp4.py mx_rebase). Offline it takes B
from the cache's largest scale; a cache written token by token needs a fixed B, so here B = FIXED_B for K and V:
scales 2^(e - 127) with e - B in [1, 30], i.e. 2^-20 .. 2^9 (group maxima up to about 3000 with E2M1's max of 6).
"""
from __future__ import annotations

from pathlib import Path

import torch

from kv4.cuda import decode_fp4 as df

_EXT = None
PAGE, D, PAGE_BYTES = 64, 128, 8704
FIXED_B = 106


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        here = Path(__file__).parent
        # no --use_fast_math: log2f / exp2f must match torch's so the codes agree with formats.quantize
        _EXT = load(name="kv4_store_fp4", sources=[str(here / "store_fp4.cu")], extra_cflags=["-O3", "-std=c++20"],
                    extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a",
                                       "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"], verbose=True)
    return _EXT


def _pack_bits(b):
    return int(b[0]) | int(b[1]) << 5 | int(b[2]) << 10 | int(b[3]) << 15


def tables_c2():
    """11-22 C2: K as tables(nv=True, k8=True); V words / bits per (page token o, channel d) from
    decode_fp4_k8._c2_v_layout (64 * 128 each) and the V scale position per (group o // 32, channel d) (2 * 128).
    The C2 store kernel computes these positions itself; the dequant kernel reads them from here."""
    from kv4.cuda import decode_fp4_k8 as k8m
    kword, kbits, _, _, kspos, _ = tables(nv=True, k8=True)
    idx, bits = k8m._c2_v_layout()
    vword, vbits = [0] * (PAGE * D), [0] * (PAGE * D)
    for w in range(1024):
        for j in range(8):
            f = int(idx[w, j])
            vword[f], vbits[f] = w, _pack_bits(bits[w, j])
    t = lambda v: torch.tensor(v, dtype=torch.int32)  # noqa: E731
    return kword, kbits, t(vword), t(vbits), kspos, k8m.c2_vscale_pos().to(torch.int32)


def tables(nv: bool = False, k8: bool = False):
    """Page-local positions from decode_fp4.py's layout functions; asserts that all 4 pages of a 256-token stage share
    one local layout. Returns int32 tensors (CPU): kword (128), kbits (128), vword (64*128), vbits (64*128),
    kspos (64*ng), vspos (64*ng), ng = 4 scale groups per token (MXFP4) or 8 (NVFP4, 11-12 night).
    11-21 k8 (with nv): K words from decode_fp4_k8.py's _k8_layout and K scales per 32 dims at the MXFP4 K scale
    positions (kspos 64*4); V as NVFP4."""
    assert nv or not k8
    if k8:
        from kv4.cuda import decode_fp4_k8 as k8m
        idx, bits = k8m._k8_layout()
    else:
        idx, bits = df._k_layout()
    kword, kbits = [0] * D, [0] * D
    for w in range(16):
        for j in range(8):
            d = int(idx[w, j])
            kword[d], kbits[d] = w, _pack_bits(bits[w, j])
    vidx, vb = df._v_layout()
    vloc = {}
    for w in range(4096):
        for j in range(8):
            tok, d = divmod(int(vidx[w, j]), D)
            q, o = divmod(tok, PAGE)
            assert w // 1024 == q
            vloc[(q, o, d)] = (w % 1024, _pack_bits(vb[w, j]))
    assert all(vloc[(q, o, d)] == vloc[(0, o, d)] for q in range(4) for o in range(PAGE) for d in range(D))
    vword = [vloc[(0, o, d)][0] for o in range(PAGE) for d in range(D)]
    vbits = [vloc[(0, o, d)][1] for o in range(PAGE) for d in range(D)]

    ng = 8 if nv else 4

    def scale_pos(perm, ng=ng):
        inv = {}
        for pos, v in enumerate(perm.tolist()):
            tok, g = divmod(v, ng)
            q, o = divmod(tok, PAGE)
            assert pos // (PAGE * ng) == q
            inv[(q, o, g)] = pos % (PAGE * ng)
        assert all(inv[(q, o, g)] == inv[(0, o, g)] for q in range(4) for o in range(PAGE) for g in range(ng))
        return [inv[(0, o, g)] for o in range(PAGE) for g in range(ng)]

    t = lambda v: torch.tensor(v, dtype=torch.int32)  # noqa: E731
    kp, vp = (df._k_scale_perm(), df._v_scale_perm()) if nv else (df._k_scale_perm_mx(), df._v_scale_perm_mx())
    if k8:
        return (t(kword), t(kbits), t(vword), t(vbits), t(scale_pos(df._k_scale_perm_mx(), 4)), t(scale_pos(vp)))
    return (t(kword), t(kbits), t(vword), t(vbits), t(scale_pos(kp)), t(scale_pos(vp)))


class StoreMXFP4:
    def __init__(self, device, rot_k: bool = True, c: float = 0.2):
        self.tabs = [x.to(device) for x in tables()]
        self.rot_k, self.c = rot_k, float(c)
        self.empty = torch.empty(0, device=device, dtype=torch.float32)

    def __call__(self, key, value, slots, pages, mu=None):
        """key, value (N, HKV, 128) bf16; slots (N) int64 = page * 64 + offset (-1: skip); pages
        (num_pages, HKV, 8704) uint8; mu: optional fp32 (HKV, 128) K mean to remove (the sink keys must then be
        stored as k - mu too)."""
        ext().store_mxfp4(key.contiguous(), value.contiguous(), slots.contiguous(), pages, *self.tabs,
                          self.empty if mu is None else mu.float().contiguous(), self.rot_k, self.c)

    @staticmethod
    def decode_args(sm: float):
        """(score_mul, out_mul, negbk, negbv) for decode_fp4_paged with the fixed B (see mx_rebase)."""
        negb = (-(FIXED_B * 1024 * 65537)) % 2**32
        if negb >= 2**31:
            negb -= 2**32
        f = 2.0 ** (14 + FIXED_B - 112)
        return sm * 1.4426950408889634 * f, f, negb, negb


class StoreNVFP4:
    """11-12 night: quantize-on-write into 64-token NVFP4 pages (9216 bytes per page and head = 64 x 144), the main
    recipe's format. ts_k / ts_v: fixed per-layer fp32 tensor scales (calibrated offline; the offline formats take
    them from the tensor at hand)."""
    PAGE_BYTES = 9216

    def __init__(self, device, ts_k: float, ts_v: float, rot_k: bool = True):
        self.tabs = [x.to(device) for x in tables(nv=True)]
        self.rot_k, self.ts_k, self.ts_v = rot_k, float(ts_k), float(ts_v)
        self.empty = torch.empty(0, device=device, dtype=torch.float32)

    def __call__(self, key, value, slots, pages, mu=None):
        ext().store_fp4(key.contiguous(), value.contiguous(), slots.contiguous(), pages, *self.tabs,
                        self.empty if mu is None else mu.float().contiguous(), self.rot_k, 0.0, True, self.ts_k,
                        self.ts_v)

    def decode_args(self, sm: float):
        """(score_mul, out_mul, negbk, negbv) for decode_fp4_paged(..., mx=False), as decode_fp4.cu decode_nvfp4."""
        return sm * 1.4426950408889634 * self.ts_k * 16384.0, self.ts_v * 16384.0, 0, 0

    def dequant(self, pages, block_ids, n_tokens):
        return ext().dequant_fp4(pages, block_ids.int().contiguous(), int(n_tokens), *self.tabs, self.rot_k, True,
                                 self.ts_k, self.ts_v)

    def dequant_raw(self, pages, block_ids, n_tokens):
        """K (rotated basis) and V as code x E4M3 scale, without the tensor scales: exact in bf16 (for the sink pools
        of the vLLM plugin, kv4/cuda/merge.cu merge_sink_pool)."""
        return ext().dequant_fp4(pages, block_ids.int().contiguous(), int(n_tokens), *self.tabs, False, True, 1.0, 1.0)


class StoreK8(StoreNVFP4):
    """11-21: C1 in the plugin. Pages of 9216 bytes like NVFP4 (vLLM's spec unchanged), but K in decode_fp4_k8's
    layout with one E4M3 scale per 32 dims (bytes 0..255 of the 512-byte K scale region; 256..511 unused), so the
    decode kernel runs QK^T on FP8 tensor cores. V is NVFP4 as in StoreNVFP4."""

    def __init__(self, device, ts_k: float, ts_v: float, rot_k: bool = True):
        self.tabs = [x.to(device) for x in tables(nv=True, k8=True)]
        self.rot_k, self.ts_k, self.ts_v = rot_k, float(ts_k), float(ts_v)
        self.empty = torch.empty(0, device=device, dtype=torch.float32)

    def __call__(self, key, value, slots, pages, mu=None):
        ext().store_fp4_k8(key.contiguous(), value.contiguous(), slots.contiguous(), pages, *self.tabs,
                           self.empty if mu is None else mu.float().contiguous(), self.rot_k, self.ts_k, self.ts_v)

    def decode_args(self, sm: float):
        """(score_mul without the q scale, out_mul) for decode_fp4_k8.cu's paged entry: the K bytes are E4M3 values
        x 2^-6 (decode_fp4_k8.args_for), V's FP16 placement x 2^-14 as in NVFP4."""
        return sm * 1.4426950408889634 * self.ts_k * 64.0, self.ts_v * 16384.0

    def dequant(self, pages, block_ids, n_tokens):
        return ext().dequant_fp4_k8(pages, block_ids.int().contiguous(), int(n_tokens), *self.tabs, self.rot_k,
                                    self.ts_k, self.ts_v)

    def dequant_raw(self, pages, block_ids, n_tokens):
        return ext().dequant_fp4_k8(pages, block_ids.int().contiguous(), int(n_tokens), *self.tabs, False, 1.0, 1.0)


class StoreC2(StoreK8):
    """11-22: C2 (docs/c2_log.md). K as StoreK8; V with one E4M3 scale per (channel, 32 tokens of a page) in
    decode_fp4_k8.cu's C2 layout, so P V also runs on FP8 tensor cores. Pages stay 9216 bytes (V scales use 256 of
    their 512 bytes). Requires in-order appends per sequence (a group's scale is a running maximum; see store_fp4.cu)."""

    def __init__(self, device, ts_k: float, ts_v: float, rot_k: bool = True):
        self.tabs = [x.to(device) for x in tables_c2()]
        self.rot_k, self.ts_k, self.ts_v = rot_k, float(ts_k), float(ts_v)
        self.empty = torch.empty(0, device=device, dtype=torch.float32)

    def __call__(self, key, value, slots, pages, mu=None):
        ext().store_fp4_c2(key.contiguous(), value.contiguous(), slots.contiguous(), pages, *self.tabs,
                           self.empty if mu is None else mu.float().contiguous(), self.rot_k, self.ts_k, self.ts_v)

    def decode_args(self, sm: float):
        """(score_mul without the q scale, out_mul) for decode_fp4_k8.decode_paged_c2: K bytes and V bytes are both
        E4M3 values x 2^-6."""
        return sm * 1.4426950408889634 * self.ts_k * 64.0, self.ts_v * 64.0

    def dequant(self, pages, block_ids, n_tokens):
        return ext().dequant_fp4_c2(pages, block_ids.int().contiguous(), int(n_tokens), *self.tabs, self.rot_k,
                                    self.ts_k, self.ts_v)

    def dequant_raw(self, pages, block_ids, n_tokens):
        return ext().dequant_fp4_c2(pages, block_ids.int().contiguous(), int(n_tokens), *self.tabs, False, 1.0, 1.0)


_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def dequant_pages_cuda(pages: torch.Tensor, block_ids: torch.Tensor, n_tokens: int, tabs, rot_k: bool = True):
    """dequant_pages with the CUDA kernel (11-12): K, V (n_tokens, HKV, 128) bf16 in the original basis."""
    return ext().dequant_mxfp4(pages, block_ids.int().contiguous(), int(n_tokens), *tabs, bool(rot_k))


def dequant_pages(pages: torch.Tensor, block_ids: torch.Tensor, n_tokens: int, tabs, rot_k: bool = True):
    """Torch inverse of the writer (slow path, e.g. vLLM continuation prefill): the first n_tokens of a sequence whose
    pages are block_ids -> K, V (n_tokens, HKV, 128) fp32 in the original basis (K rotated back if rot_k). tabs are
    StoreMXFP4.tabs (on the pages' device). Mean removal is not undone (attention does not need it)."""
    kword, kbits, vword, vbits, kspos, vspos = tabs
    pg = pages[block_ids.long()]                                     # (n, HKV, 8704) uint8
    n, h = pg.shape[0], pg.shape[1]
    grid = torch.tensor(_GRID, device=pages.device)

    def codes_of(words, bits):                                        # words (..., 128) int64, bits (128) packed
        c = torch.zeros_like(words)
        for i in range(4):
            pos = ((bits.long() >> (5 * i)) & 31)
            c |= ((words >> pos) & 1) << i
        v = grid[c & 7]
        return torch.where(c >= 8, -v, v)

    kw = pg[..., :4096].contiguous().view(torch.int32).reshape(n, h, 64, 16).long() & 0xFFFFFFFF
    kv = codes_of(torch.gather(kw, 3, kword.long().view(1, 1, 1, 128).expand(n, h, 64, 128)), kbits)
    kse = pg[..., 4096:4352].long()[..., kspos.long().view(64, 4)]   # (n, HKV, 64, 4)
    k = (kv.view(n, h, 64, 4, 32) * torch.exp2(kse.float() - 127)[..., None]).view(n, h, 64, 128)
    vw = pg[..., 4352:8448].contiguous().view(torch.int32).long() & 0xFFFFFFFF   # (n, HKV, 1024)
    vv = codes_of(torch.gather(vw, 2, vword.long().view(1, 1, -1).expand(n, h, 64 * 128)).view(n, h, 64, 128),
                  vbits.view(64, 128))
    vse = pg[..., 8448:8704].long()[..., vspos.long().view(64, 4)]
    v = (vv.view(n, h, 64, 4, 32) * torch.exp2(vse.float() - 127)[..., None]).view(n, h, 64, 128)
    k = k.permute(0, 2, 1, 3).reshape(n * 64, h, 128)[:n_tokens]
    v = v.permute(0, 2, 1, 3).reshape(n * 64, h, 128)[:n_tokens]
    if rot_k:
        from kv4.formats import rotate
        k = rotate(k, 128)
    return k, v
