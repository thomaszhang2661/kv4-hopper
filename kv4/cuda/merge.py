"""Split-KV merge kernel (kv4/cuda/merge.cu, 11-11): (out_o, out_ml) of the CUDA decode kernels -> (B, HQ, 128) bf16.

prep_q_rot (11-12) does q rotation + padding + fp16 in one launch:
    qh = prep_q_rot(q, hkv)                           # == decode_fp4.prep_q(rotate(q, 128), hkv)

merge_sink (11-11) also joins the attention-sink tokens kept in BF16 (not in the 4-bit cache):
    out = merge_sink(o, ml, q, sink_k, sink_v, sm)   # q (B, HQ, 128) in sink_k's basis; sink_k/v (B, HKV, ns, 128) bf16
"""
from pathlib import Path

import torch

_EXT = None
P_SHIFT = 10.0  # the decode kernels scale P (and so l) by 2^P_SHIFT; decode_fp4.cu / decode_int4.cu


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        here = Path(__file__).parent
        _EXT = load(name="kv4_merge", sources=[str(here / "merge.cu")], extra_cflags=["-O3", "-std=c++20"],
                    extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a",
                                       "--use_fast_math", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"], verbose=True)
    return _EXT


def merge_sink(o, ml, q: torch.Tensor, sink_k: torch.Tensor, sink_v: torch.Tensor, sm: float) -> torch.Tensor:
    """q bf16 or fp32 (11-12: bf16 is read directly, no conversion launch)."""
    if q.dtype not in (torch.bfloat16, torch.float32):
        q = q.float()
    return ext().merge_sink(o, ml, q.contiguous(), sink_k.contiguous(), sink_v.contiguous(),
                            sm * 1.4426950408889634, P_SHIFT)


def prep_q_rot(q: torch.Tensor, hkv: int, rot: bool = True) -> torch.Tensor:
    """One launch for decode_fp4.prep_q(formats.rotate(q, 128) if rot else q, hkv): (B, HQ, 128) bf16/fp32 ->
    (B, HKV, 8, 128) fp16 (11-12)."""
    if q.dtype not in (torch.bfloat16, torch.float32):
        q = q.float()
    return ext().prep_q_rot(q.contiguous(), hkv, rot)


def prep_q_rot8(q: torch.Tensor, hkv: int, rot: bool = True):
    """C1 (KV4_K8, 11-21): prep_q_rot and decode_fp4_k8's prep_q8 in one launch. (B, HQ, 128) bf16/fp32 ->
    (q8 (B, HKV, 8, 128) uint8 E4M3, qs (B * HKV) fp32 scale per (sequence, kv head), qd (B, HKV, 8, 128) fp16 the
    dequantized q), bit-identical to decode_fp4_k8.ext().prep_q8(prep_q_rot(q, hkv, rot))."""
    if q.dtype not in (torch.bfloat16, torch.float32):
        q = q.float()
    return ext().prep_q_rot8(q.contiguous(), hkv, rot)
