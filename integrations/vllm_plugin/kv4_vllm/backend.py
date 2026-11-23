"""UltraQuant backend with the kv4-hopper write and decode kernels (11-12). See __init__.py.

11-12 night, the recipe's format: KV4_FORMAT=nvfp4 (default when KV4_CALIB is set) stores NVFP4 + Hadamard K in
144-byte slots (64-token pages of 9216 bytes per kv head) with fixed per-layer tensor scales from the calibration
file KV4_CALIB (bench/calib_kv.py). KV4_KMEAN=1 also removes the calibrated static K mean (exact for attention).
KV4_FORMAT=mxfp4 keeps UltraQuant's own bytes (136, MXFP4 + Hadamard K): not the recipe.
KV4_SINKS=1 (11-12 night): the first 4 tokens of each sequence also kept in BF16 (per-layer pools of KV4_SINK_SLOTS
slots, found through the sequence's first block id) and swapped in for their 4-bit copies in the merge
(merge.cu merge_sink_pool); a sequence whose slot was reused falls back to plain 4-bit. The recipe is
KV4_CALIB=... KV4_KMEAN=1 KV4_SINKS=1 KV4_BOUNDARY=0.
KV4_K8=1 (11-21, C1, default off; with KV4_FORMAT=nvfp4): K is stored with one E4M3 scale per 32 dims in the layout of
kv4/cuda/decode_fp4_k8.cu and QK^T runs on FP8 tensor cores (q in E4M3 with one scale per sequence and kv head); V,
the page size (9216) and everything else as the recipe. docs/c1_plugin_log.md.
KV4_K8=2 (11-22, C2, default off): C1 plus P V on FP8 tensor cores: V stored with one E4M3 scale per (channel, 32
tokens of a page) (StoreC2), P rounded to E4M3 in the kernel (decode_fp4_k8.cu C2). Same 9216-byte pages. With sinks,
the C2 kernel masks the first 4 tokens of sequences with a valid sink slot and merge_sink_pool_c2 adds the BF16 sinks
(no subtraction of 4-bit copies). docs/c2_log.md.
"""
import math
import os
import re
from dataclasses import replace

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.ultraquant_attn import UltraQuantAttentionBackend, UltraQuantAttentionImpl

logger = init_logger(__name__)
NUM_SMS = 132
NSK = 4             # sink tokens
MAP_SIZE = 1 << 18  # block ids covered by the sink map (16.7M tokens at 64 per block)


_MAX_SPLITS = int(os.environ.get("KV4_MAX_SPLITS", "16"))
# 11-22 (docs/smallbatch_merge_log.md): KV4_MERGE=v2 runs the sink merge with its splits spread over warps
# (merge.cu merge_sink_pool_v2_kernel, KV4_MERGE_WARPS warps at most, default 8); default v1 = the old kernel.
# KV4_SPLIT_CTAS: the CTA target of the split rule splits = min(KV4_MAX_SPLITS, KV4_SPLIT_CTAS // (B * H_kv)), default
# 2 * 132 as before (B = 1 can then reach 33 splits, B = 4 at most 8).
_MERGE_V2 = os.environ.get("KV4_MERGE", "v1") == "v2"
_MERGE_WARPS = int(os.environ.get("KV4_MERGE_WARPS", "8"))
_SPLIT_CTAS = int(os.environ.get("KV4_SPLIT_CTAS", str(2 * NUM_SMS)))
# 11-21 (KV4_K8 only): KV4_K8_PREP=two keeps the old q prep in two launches (prep_q_rot + prep_q8); default one launch
_K8_PREP_TWO = os.environ.get("KV4_K8_PREP", "one") == "two"


def _rank_heads(mu, n_local):
    """11-13: the calibration holds every KV head (H, 128); under tensor parallelism this rank owns n_local of them.
    vLLM splits KV heads contiguously by rank when H >= TP, and replicates them when H < TP (rank r then holds head
    r // (TP / H))."""
    h = mu.shape[0]
    if h == n_local:
        return mu
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    r, tp = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
    if h >= tp:
        assert h == n_local * tp, f"kv4: {h} calibrated KV heads, {n_local} per rank at TP {tp}"
        return mu[r * n_local:(r + 1) * n_local]
    assert n_local == 1 and tp % h == 0, f"kv4: {h} calibrated KV heads at TP {tp}"
    return mu[r // (tp // h): r // (tp // h) + 1]


def _h2d(x, dtype, dev):
    """11-13: a small host list to the GPU without a stream sync (torch.tensor(x, device=cuda) copies from pageable
    memory and synchronizes; the profile's prefill path did several per layer)."""
    return torch.tensor(x, dtype=dtype, pin_memory=True).to(dev, non_blocking=True)


def _fmt():
    f = os.environ.get("KV4_FORMAT") or ("nvfp4" if os.environ.get("KV4_CALIB") else "mxfp4")
    assert f in ("nvfp4", "mxfp4"), f"KV4_FORMAT {f}"
    return f


_CALIB = None


def _calib():
    global _CALIB
    if _CALIB is None:
        path = os.environ.get("KV4_CALIB")
        assert path, "KV4_FORMAT=nvfp4 needs KV4_CALIB (per-layer tensor scales, bench/calib_kv.py)"
        _CALIB = torch.load(path, map_location="cpu")
        logger.info("kv4-hopper: calibration %s (%s, %d layers)", path, _CALIB.get("model"), len(_CALIB["layers"]))
    return _CALIB


class KV4UltraQuantBackend(UltraQuantAttentionBackend):
    @staticmethod
    def get_impl_cls():
        return KV4UltraQuantImpl

    @classmethod
    def customize_spec(cls, spec):
        spec = super().customize_spec(spec)
        if _fmt() == "nvfp4" and spec.state_content_bytes is not None and spec.kv_quant_mode.is_ultraquant:
            spec = replace(spec, state_content_bytes=144 * spec.head_size // 128)  # NVFP4: 72 B for K, 72 for V
        return spec


class KV4UltraQuantImpl(UltraQuantAttentionImpl):
    _kv4_store = None
    _kv4_layer = None
    _kv4_mu = None

    def _kv4_set_layer(self, layer):
        if self._kv4_layer is None and layer is not None:
            m = re.search(r"layers\.(\d+)\.", getattr(layer, "layer_name", "") or "")
            if m:
                self._kv4_layer = int(m.group(1))

    def _kv4(self, device, layer=None):
        self._kv4_set_layer(layer)
        if self._kv4_store is None:
            from kv4.cuda.store_fp4 import StoreC2, StoreK8, StoreMXFP4, StoreNVFP4
            assert self.head_size == 128 and self.num_heads % self.num_kv_heads == 0 and \
                1 <= self.num_heads // self.num_kv_heads <= 8, "kv4 kernels: head_dim 128, 1..8 query heads per kv head"
            self._kv4_nv = _fmt() == "nvfp4"
            if self._kv4_nv:
                assert self._kv4_layer is not None, "kv4: layer index unknown (needed for the calibration)"
                cal = _calib()["layers"][self._kv4_layer]
                mean = os.environ.get("KV4_KMEAN") == "1"
                if mean:
                    self._kv4_mu = _rank_heads(cal["mu_k"].float(), self.num_kv_heads).to(device).contiguous()
                # 11-13: with BF16 sinks the 4-bit tensor scales come from the tokens after the sinks (`_s` entries
                # of newer calibration files), as the offline formats do
                sfx = "_s" if os.environ.get("KV4_SINKS") == "1" and "ts_k_s" in cal and \
                    os.environ.get("KV4_SINK_SCALE", "1") == "1" else ""
                ts_k = cal["ts_k" + sfx] if mean else cal["ts_k_nomean" + sfx]
                k8env = os.environ.get("KV4_K8", "0")
                self._kv4_c2 = k8env == "2"  # 11-22: C2 = K8 + P V on FP8 tensor cores
                self._kv4_k8 = k8env == "1"
                store_cls = StoreC2 if self._kv4_c2 else StoreK8 if self._kv4_k8 else StoreNVFP4
                self._kv4_store = store_cls(device, ts_k, cal["ts_v" + sfx], rot_k=True)
                self._kv4_args = self._kv4_store.decode_args(self.scale)
                logger.info_once("kv4-hopper: ultraquant_4bit slots as NVFP4 + Hadamard K (recipe format), K mean %s, "
                                 "K8 (K on FP8 tensor cores) %s, C2 (PV on FP8 tensor cores) %s", mean, self._kv4_k8,
                                 self._kv4_c2)
            else:
                c = float(os.environ.get("KV4_SCALE_C", "0.156"))
                self._kv4_store = StoreMXFP4(device, rot_k=True, c=c)
                self._kv4_args = StoreMXFP4.decode_args(self.scale)
                logger.info_once("kv4-hopper: ultraquant_4bit on Hopper kernels (MXFP4, UltraQuant bytes), c=%s", c)
            self._kv4_sinks = os.environ.get("KV4_SINKS") == "1"
            if self._kv4_sinks:  # allocated at the first call (vLLM's profile run), so the KV budget accounts for it
                P, hk = int(os.environ.get("KV4_SINK_SLOTS", "1024")), self.num_kv_heads
                bf = dict(device=device, dtype=torch.bfloat16)
                self._sk, self._sv, self._k4, self._v4 = (torch.zeros(P, hk, NSK, 128, **bf) for _ in range(4))
                self._owner = torch.full((P,), -1, device=device, dtype=torch.int32)
                self._smap = torch.full((MAP_SIZE,), -1, device=device, dtype=torch.int32)
                self._next = 0
                self._owner_arg = self._owner
                if os.environ.get("KV4_SINK_TIMING") == "1":  # timing only (vllm bench forward has no prefill):
                    # every block maps to some slot and the owner check is off, so every sequence takes the sink path
                    self._smap.copy_(torch.arange(MAP_SIZE, device=device, dtype=torch.int32) % P)
                    self._owner_arg = torch.empty(0, device=device, dtype=torch.int32)
                logger.info_once("kv4-hopper: BF16 sinks (%d tokens), %d slots per layer", NSK, P)
        return self._kv4_store

    def _kv4_capture_sinks(self, key, value, pages, attn_metadata, qsl, lens):
        """First-chunk prefill: put each new sequence's first NSK tokens into the next pool slots (vectorized; eager,
        outside CUDA graphs), with their 4-bit copies as the cache now holds them."""
        first = [i for i in range(len(qsl) - 1) if lens[i] == qsl[i + 1] - qsl[i] and lens[i] > 0]
        if not first:
            return
        dev = key.device
        short = [i for i in first if lens[i] < NSK]
        if short:
            self._smap[attn_metadata.block_table[_h2d(short, torch.long, dev), 0].long()] = -1
        rows = [i for i in first if lens[i] >= NSK]
        if not rows:
            return
        n, P = len(rows), self._owner.numel()
        bt0 = attn_metadata.block_table[_h2d(rows, torch.long, dev), 0].long()
        torch._assert_async(bt0.max() < MAP_SIZE, "kv4: block id beyond the sink map")
        starts = _h2d([qsl[i] for i in rows], torch.long, dev)
        tok = starts[:, None] + torch.arange(NSK, device=dev)[None]                     # (n, NSK)
        kk, vv = key[tok].to(torch.bfloat16), value[tok].to(torch.bfloat16)            # (n, NSK, Hk, D)
        if self._kv4_mu is not None:
            kk = (kk.float() - self._kv4_mu).to(torch.bfloat16)
        st = self._kv4_store
        if self._kv4_nv:
            k4, v4 = st.dequant_raw(pages, bt0.int(), n * 64)
        else:
            from kv4.cuda.store_fp4 import dequant_pages_cuda
            k4, v4 = dequant_pages_cuda(pages, bt0.int(), n * 64, st.tabs, rot_k=False)
        hk = self.num_kv_heads
        k4 = k4.view(n, 64, hk, -1)[:, :NSK]
        v4 = v4.view(n, 64, hk, -1)[:, :NSK]
        slots = (torch.arange(n, device=dev) + self._next) % P
        self._next = (self._next + n) % P
        self._sk[slots] = kk.transpose(1, 2).contiguous()
        self._sv[slots] = vv.transpose(1, 2).contiguous()
        self._k4[slots] = k4.transpose(1, 2).to(torch.bfloat16).contiguous()
        self._v4[slots] = v4.transpose(1, 2).to(torch.bfloat16).contiguous()
        self._owner[slots] = bt0.int()
        self._smap[bt0] = slots.int()

    def _kv4_sink_slot(self, blk):
        """(slot, valid) for first-block ids blk (long tensor)."""
        s = self._smap[blk].long()
        sc = s.clamp(min=0)
        return sc, (s >= 0) & (self._owner[sc] == blk.int())

    def _kv4_pages(self, kv_cache):
        """kv_cache as the TurboQuant scaffolding passes it, (num_blocks, 64, Hk, slot) uint8 in physical order ->
        our pool (num_blocks, Hk, 64 * slot); the block's bytes are ours alone, so they are simply reinterpreted.
        slot = 136 (MXFP4) or 144 (NVFP4)."""
        nb = kv_cache.shape[0]
        pb = 9216 if self._kv4_nv else 8704
        ok = kv_cache.is_contiguous() and kv_cache.numel() * kv_cache.element_size() == nb * self.num_kv_heads * pb
        if not ok and kv_cache.dim() == 4 and kv_cache.shape[1] != 64:
            raise RuntimeError(f"kv4: blocks of {kv_cache.shape[1]} tokens; our pages are 64 tokens. With native "
                               "boundary layers vLLM lowers the block size: pass --block-size 64 (block_size=64)")
        if not ok:  # 11-12: seen with boundary layers kept native (mixed KV specs); report the layout we got
            raise RuntimeError(f"kv4: unexpected kv_cache layout shape {tuple(kv_cache.shape)} stride "
                               f"{kv_cache.stride()} dtype {kv_cache.dtype} contiguous {kv_cache.is_contiguous()}; "
                               f"expected {nb} x {self.num_kv_heads} x {pb} bytes")
        return kv_cache.view(torch.uint8).view(nb, self.num_kv_heads, -1)

    def _store_kv(self, key, value, kv_cache, slot_mapping, layer):
        st = self._kv4(key.device, layer)
        st(key.to(torch.bfloat16), value.to(torch.bfloat16), slot_mapping.long(), self._kv4_pages(kv_cache),
           self._kv4_mu)

    def _decode_attention(self, query, kv_cache, attn_metadata, Pi, centroids, PiT=None, layer=None):
        from kv4.cuda import decode_fp4 as df
        from kv4.cuda import merge as mg
        self._kv4(query.device, layer)
        b = query.shape[0]
        # 11-13: KV4_MAX_SPLITS (default 16); with few KV heads (Qwen3-30B-A3B: 4) B=1 gave only 64 CTAs on 132 SMs
        splits = max(1, min(_MAX_SPLITS, _SPLIT_CTAS // max(1, b * self.num_kv_heads)))
        hk, g = self.num_kv_heads, self.num_heads // self.num_kv_heads
        gk = 4 if g <= 4 else 8  # 11-12: the kernel runs 4 or 8 heads per kv head; q is zero-padded to that
        if getattr(self, "_kv4_c2", False):  # 11-22 C2: q prep as C1; the kernel masks valid sinks, the merge adds them
            from kv4.cuda import decode_fp4_k8 as k8
            q8, qs, _ = mg.prep_q_rot8(query, hk, True)
            pages = self._kv4_pages(kv_cache)
            if self._kv4_sinks:
                o, ml = k8.decode_paged_c2(pages, attn_metadata.block_table, attn_metadata.seq_lens, q8, qs,
                                           *self._kv4_args, splits, gk, self._smap, self._owner_arg, NSK)
                out = k8.ext().merge_sink_pool_c2(o, ml, query.to(torch.bfloat16).contiguous(),
                                                  attn_metadata.block_table, self._smap, self._owner_arg, self._sk,
                                                  self._sv, g, self.scale * 1.4426950408889634, k8.P_SHIFT_C2)
                return out.to(query.dtype)
            o, ml = k8.decode_paged_c2(pages, attn_metadata.block_table, attn_metadata.seq_lens, q8, qs,
                                       *self._kv4_args, splits, gk)
            out = mg.ext().merge(o, ml)
            if gk != g:
                out = out.view(b, hk, gk, -1)[:, :, :g].reshape(b, hk * g, -1)
            return out.to(query.dtype)
        if getattr(self, "_kv4_k8", False):  # 11-21 C1: q in E4M3 per (sequence, kv head); the sink merge below must
            from kv4.cuda import decode_fp4_k8 as k8  # use the q the kernel used (qd) so the 4-bit sink copies cancel
            if _K8_PREP_TWO:  # the 11-21 two-launch prep (prep_q_rot, then prep_q8), kept for timing comparisons
                qh = mg.prep_q_rot(query, hk, True)
                o, ml, qh = k8.decode_paged(self._kv4_pages(kv_cache), attn_metadata.block_table,
                                            attn_metadata.seq_lens, qh, *self._kv4_args, splits, gk)
            else:  # 11-21: rotation + E4M3 q + its scale + qd in one launch (bit-identical to the two launches)
                q8, qs, qh = mg.prep_q_rot8(query, hk, True)
                o, ml = k8.decode_paged_q8(self._kv4_pages(kv_cache), attn_metadata.block_table,
                                           attn_metadata.seq_lens, q8, qs, *self._kv4_args, splits, gk)
        else:
            qh = mg.prep_q_rot(query, hk, True)
            o, ml = df.ext().decode_fp4_paged(self._kv4_pages(kv_cache), attn_metadata.block_table,
                                              attn_metadata.seq_lens, qh, *self._kv4_args, splits, not self._kv4_nv,
                                              0, gk)
        if self._kv4_sinks:  # swap the 4-bit copies of the first NSK tokens for the BF16 sinks (real heads only)
            st = self._kv4_store
            k4m, v4m = (st.ts_k, st.ts_v) if self._kv4_nv else (1.0, 1.0)
            margs = (o, ml, query.to(torch.bfloat16).contiguous(), qh, attn_metadata.block_table, self._smap,
                     self._owner_arg, self._sk, self._sv, self._k4, self._v4, g, self.scale * 1.4426950408889634, k4m,
                     v4m, mg.P_SHIFT)
            if _MERGE_V2:  # 11-22: same result up to fp32 summation order (bench/merge_v2_check.py)
                out = mg.ext().merge_sink_pool_v2(*margs, _MERGE_WARPS, False)
            else:
                out = mg.ext().merge_sink_pool(*margs)
            return out.to(query.dtype)
        out = mg.ext().merge(o, ml)
        if gk != g:  # drop the padded heads (e.g. Qwen2.5-7B: 7 of 8)
            out = out.view(b, hk, gk, -1)[:, :, :g].reshape(b, hk * g, -1)
        return out.to(query.dtype)

    def _flash_attn_varlen(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, out=None):
        """Prefill attention (first chunks via the parent's fast path, and our continuation). 11-12: the profile
        showed the parent's call on FA2 (_vllm_fa2_C::varlen_fwd), ~1.6x the time of vLLM fp8's FA3-FP8 prefill.
        KV4_PREFILL=bf16 (default): FA3 in BF16 (exact; the 11-12 microbench 11840255 found FA3 FP8 no faster than FA3 BF16, 3.60 vs 3.26 ms at 16K). KV4_PREFILL=fp8: FA3 in FP8 like vLLM's fp8 KV, but with
        the per-head K mean removed first (exact for softmax: q . mu is the same for every key of a query; guards the
        Qwen2-style layer-0 K that is ~98% mean) and dynamic per-head scales (amax / 448) for q, k, v."""
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
        kw = dict(cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_q,
                  max_seqlen_k=max_seqlen_k, softmax_scale=self.scale, causal=True, fa_version=3)
        if os.environ.get("KV4_PREFILL", "bf16") != "fp8":
            if out is not None and q.dtype == out.dtype:  # 11-13: write the request's rows of the output in place
                flash_attn_varlen_func(q=q, k=k, v=v, out=out, **kw)
                return out
            r = flash_attn_varlen_func(q=q, k=k, v=v, **kw)
            if out is not None:
                out.copy_(r)
            return r
        f8 = torch.float8_e4m3fn
        nb = cu_seqlens_q.numel() - 1
        hk = k.shape[1]
        k = k - k.float().mean(0, keepdim=True).to(k.dtype)

        def quant(x, groups):  # per kv-head scale over all tokens (and the q heads of that group)
            a = x.view(x.shape[0], hk, groups, x.shape[-1]).abs().amax(dim=(0, 2, 3)).float().clamp_min(1e-6) / 448.0
            xq = (x.view(x.shape[0], hk, groups, -1) / a.view(1, hk, 1, 1)).to(f8).view(x.shape)
            return xq, a.view(1, hk).expand(nb, hk).contiguous()

        qq, qd = quant(q, q.shape[1] // hk)
        kq, kd = quant(k, 1)
        vq, vd = quant(v, 1)
        r = flash_attn_varlen_func(q=qq, k=kq, v=vq, q_descale=qd, k_descale=kd, v_descale=vd, **kw)
        if out is not None:
            out.copy_(r)
        return r

    def _prefill_attention(self, query, key, value, kv_cache, attn_metadata, Pi, centroids, PiT=None, layer=None):
        qsl = (attn_metadata.query_start_loc_cpu if attn_metadata.query_start_loc_cpu is not None
               else attn_metadata.query_start_loc).tolist()
        lens = (attn_metadata.seq_lens_cpu if attn_metadata.seq_lens_cpu is not None
                else attn_metadata.seq_lens).tolist()
        # 11-12: first chunks only (no cache read) checked per request; the stock test max_query_len == max_seq_len
        # also passes when a continuing request is shorter than another request's first chunk
        self._kv4(query.device, layer)
        if self._kv4_sinks:
            self._kv4_capture_sinks(key, value, self._kv4_pages(kv_cache), attn_metadata, qsl, lens)
        if all(lens[i] == qsl[i + 1] - qsl[i] for i in range(len(qsl) - 1)):
            return super()._prefill_attention(query, key, value, kv_cache, attn_metadata, Pi, centroids, PiT, layer)
        from kv4.cuda.store_fp4 import dequant_pages, dequant_pages_cuda
        deq = dequant_pages if os.environ.get("KV4_TORCH_DEQUANT") == "1" else dequant_pages_cuda
        st = self._kv4(query.device, layer)
        pages = self._kv4_pages(kv_cache)
        dev = query.device
        out = torch.empty(query.shape, dtype=query.dtype, device=dev)
        if qsl[-1] < out.shape[0]:
            out[qsl[-1]:].zero_()  # padding rows
        # 11-13: all cu_seqlens in one non-blocking copy (two torch.tensor(..., device) per request synchronized)
        cus = _h2d([x for i in range(len(qsl) - 1) for x in (0, qsl[i + 1] - qsl[i], 0, lens[i])], torch.int32, dev)
        from kv4.cuda.store_fp4 import ext as store_ext
        nv = self._kv4_nv
        ts = (st.ts_k, st.ts_v) if nv else (1.0, 1.0)
        for i in range(len(qsl) - 1):
            q0, q1 = qsl[i], qsl[i + 1]
            ql, sl = q1 - q0, lens[i]
            if ql <= 0:
                continue
            k, v = key[q0:q1], value[q0:q1]
            cached = sl - ql
            if cached > 0:  # continuation: the prefix from our pages, the chunk itself fresh
                ids = attn_metadata.block_table[i, : math.ceil(cached / 64)]
                if os.environ.get("KV4_TORCH_DEQUANT") == "1" and not nv:
                    kp, vp = deq(pages, ids, cached, st.tabs, rot_k=True)
                    kf, vf = torch.cat([kp.to(k.dtype), k]), torch.cat([vp.to(v.dtype), v])
                else:
                    # 11-13: one (sl, Hk, D) buffer per request; the CUDA dequant writes the prefix rows directly
                    # (the profile showed ~480 ms of cat / copies at B=8 x 32K before)
                    kf = torch.empty(sl, *k.shape[1:], device=dev, dtype=torch.bfloat16)
                    vf = torch.empty_like(kf)
                    if getattr(self, "_kv4_c2", False):  # 11-22: C2 pages
                        store_ext().dequant_fp4_c2_into(pages, ids.int().contiguous(), cached, *st.tabs, True, ts[0],
                                                        ts[1], kf, vf)
                    elif getattr(self, "_kv4_k8", False):  # 11-21: K8 pages
                        store_ext().dequant_fp4_k8_into(pages, ids.int().contiguous(), cached, *st.tabs, True, ts[0],
                                                        ts[1], kf, vf)
                    else:
                        store_ext().dequant_fp4_into(pages, ids.int().contiguous(), cached, *st.tabs, True, nv, ts[0],
                                                     ts[1], kf, vf)
                    if self._kv4_mu is not None:  # cached keys are k - mu: shift the fresh ones too (exact for softmax)
                        torch.sub(k, self._kv4_mu, out=kf[cached:])  # fp32 math, one bf16 rounding (as before)
                    else:
                        kf[cached:].copy_(k)
                    vf[cached:].copy_(v)
                if self._kv4_sinks and cached >= NSK:  # the prefix's first tokens from the BF16 sinks when valid
                    sc, ok = self._kv4_sink_slot(attn_metadata.block_table[i, :1].long())
                    ok = ok.view(1, 1, 1)  # 11-13: on the device (bool(ok) synchronized every request and layer)
                    kf[:NSK] = torch.where(ok, self._sk[sc[:1]][0].transpose(0, 1), kf[:NSK])  # (1,) index: no sync
                    vf[:NSK] = torch.where(ok, self._sv[sc[:1]][0].transpose(0, 1), vf[:NSK])
                k, v = kf.to(key.dtype), vf.to(value.dtype)
            cq, ck = cus[4 * i: 4 * i + 2], cus[4 * i + 2: 4 * i + 4]
            self._flash_attn_varlen(q=query[q0:q1], k=k, v=v, cu_seqlens_q=cq, cu_seqlens_k=ck, max_seqlen_q=ql,
                                    max_seqlen_k=sl, out=out[q0:q1])
        return out
