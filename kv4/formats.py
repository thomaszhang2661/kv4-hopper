"""KV-cache storage formats: real packed bytes, as a decode kernel reads them.

A cache tensor has shape (B, H_kv, S, D); every format quantizes along D (one
token's head vector), so a kernel that loads one token's row has all it needs.

  bf16    : no quantization.
  fp8     : FP8 E4M3 elements, one FP32 scale per tensor (vLLM's kv-cache scale).
  int4    : symmetric integers -7..7 stored as code = q + 8, one FP16 scale per
            group of `block` elements (block 0 = the whole row, "token-wise").
  int4 asym: codes 0..15, x = code * scale + zero, FP16 scale = (max - min) / 15 and
            FP16 zero = min per group (SAW-INT4, arXiv 2604.19157: per token and head).
  e2m1    : FP4 E2M1 elements, code = sign << 3 | index into E2M1_GRID; scale
            per block of 16 as E4M3 times an FP32 tensor scale (NVFP4) or per
            block of 32 as a power of two, E8M0 (MXFP4).
Packing: two 4-bit codes per byte, element 2i in the low nibble, 2i+1 in the high.

`hadamard` > 0 multiplies each row by a block-diagonal Walsh-Hadamard matrix
(blocks of that size) before quantizing. It is orthogonal and symmetric, so
q.k = (qH).(kH): the kernel rotates q the same way, and for V rotates the output
back (out = (P V H) H).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, replace

import torch

E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MAX = 6.0
E4M3_MAX = 448.0
INT4_MAX = 7


@dataclass(frozen=True)
class KVFormat:
    name: str
    element: str  # "bf16", "fp8", "int4", "e2m1"
    block: int = 0  # elements per scale along D; 0 = one scale per row
    scale: str = "none"  # "none", "tensor" (fp8), "fp16" (int4), "e4m3" (+ fp32 tensor scale), "e8m0"
    hadamard: int = 0  # block size of the Walsh-Hadamard rotation; 0 = none
    asym: bool = False  # int4 only: scale and zero point instead of a symmetric scale
    # 11-21 (C2): > 0 = scales along the TOKEN axis instead of D: one scale per `tok_block` consecutive tokens of one
    # channel (`block` unused); see _quantize_tok
    tok_block: int = 0

    def bytes_per_row(self, d: int) -> float:
        """Bytes a kernel reads per token per head (scales included, tensor scales ignored)."""
        if self.element == "bf16":
            return 2 * d
        if self.element == "fp8":
            return d
        if self.scale == "tensor":  # 11-18: one scale per tensor, no per-row bytes
            return d / 2
        if self.tok_block:  # 11-21 (C2): D scale bytes per tok_block tokens, amortized per token
            return d / 2 + d / self.tok_block
        blocks = math.ceil(d / (self.block or d))
        return d / 2 + blocks * (2 if self.scale == "fp16" else 1) * (2 if self.asym else 1)


FORMATS = {f.name: f for f in [
    KVFormat("bf16", "bf16"),
    KVFormat("fp8", "fp8", scale="tensor"),
    KVFormat("int4_tok", "int4", block=0, scale="fp16"),
    KVFormat("int4_tok_h", "int4", block=0, scale="fp16", hadamard=128),
    KVFormat("int4_g32", "int4", block=32, scale="fp16"),
    KVFormat("int4a_tok", "int4", block=0, scale="fp16", asym=True),
    KVFormat("int4a_tok_h", "int4", block=0, scale="fp16", hadamard=128, asym=True),  # SAW-INT4 BDR-128
    KVFormat("nvfp4", "e2m1", block=16, scale="e4m3"),
    KVFormat("nvfp4_h", "e2m1", block=16, scale="e4m3", hadamard=128),
    KVFormat("mxfp4", "e2m1", block=32, scale="e8m0"),
    KVFormat("mxfp4_h", "e2m1", block=32, scale="e8m0", hadamard=128),
    # vLLM UltraQuant's scale rule (uq_format.py, vLLM main 5a0a462220): s = 2^round(log2(0.156 amax)) instead of
    # OCP's 2^(floor(log2 amax) - 2); same bytes and the same decode (code * 2^(e - 127)), so the MXFP4 kernel reads it
    KVFormat("mxfp4_uq", "e2m1", block=32, scale="e8m0_uq"),
    KVFormat("mxfp4_uq_h", "e2m1", block=32, scale="e8m0_uq", hadamard=128),
    # 11-18 (baselines): the plainest 4-bit caches, one FP32 scale per tensor (here per layer and
    # call: amax / 6 for E2M1, amax / 7 for INT4), no blocks, no rotation
    KVFormat("fp4_tensor", "e2m1", block=0, scale="tensor"),
    KVFormat("int4_tensor", "int4", block=0, scale="tensor"),
    # 11-18: NVFP4 as vLLM's --kv-cache-dtype nvfp4 stores it by default: E4M3 block scales of 16 with the FP32
    # global scale left at 1.0 (not calibrated), so block scales of small blocks fall into E4M3 subnormals or to 0
    KVFormat("nvfp4_g1", "e2m1", block=16, scale="e4m3_g1"),
    # 11-21 (XQA borrow, change C1): E2M1 with one E4M3 scale per 32 values, so one FP8 wgmma k-step (32 dims) has one
    # scale per token and the scale can be applied after the MMA (offline accuracy only; no kernel reads it yet)
    KVFormat("nvfp4_g32", "e2m1", block=32, scale="e4m3"),
    KVFormat("nvfp4_g32_h", "e2m1", block=32, scale="e4m3", hadamard=128),
    # 11-21 (C2): NVFP4 V with its E4M3 block scales along tokens (one per 16 / 32 consecutive tokens of one channel,
    # times the FP32 tensor scale), so P V on FP8 wgmma (which reduces over tokens, k32) can apply the scale after the
    # MMA, per output channel. A page holds 64 tokens, so 16- and 32-token blocks never span pages. Offline only.
    KVFormat("nvfp4_t16", "e2m1", block=0, scale="e4m3", tok_block=16),
    KVFormat("nvfp4_t32", "e2m1", block=0, scale="e4m3", tok_block=32),
    # 11-21 (C2, coordinator): one E4M3 scale per token row (all D dims) times the FP32 tensor scale; the scale folds
    # into P (P[t] * s_v[t]), so P V runs on FP8 with V token-major
    KVFormat("nvfp4_row", "e2m1", block=0, scale="e4m3"),
]}
# 11-11: the one-parameter family behind both rules, s = 2^round(log2(c amax)): UltraQuant is c = 0.156 and OCP's
# 2^(floor(log2 amax) - 2) equals c = 1/(4 sqrt 2) = 0.177 (round(log2 amax - 2.5) = floor(log2 amax) - 2 except at
# exact ties). For sweeping c (bench/accuracy_kl.py, bench/scale_rule_mse.py); names mxfp4_c0.156, mxfp4_c0.156_h.
MX_C_SWEEP = (0.125, 0.14, 0.156, 0.177, 0.2, 0.22, 0.25)
for _c in MX_C_SWEEP:
    for _h in (0, 128):
        _n = f"mxfp4_c{_c:.3f}" + ("_h" if _h else "")
        FORMATS[_n] = KVFormat(_n, "e2m1", block=32, scale=f"e8m0_c{_c:.3f}", hadamard=_h)


@dataclass
class QTensor:
    """A quantized cache: `data` is (..., D) for bf16/fp8 and (..., D/2) uint8 for 4-bit;
    `scales` is (..., n_blocks): fp16 (int4) or raw uint8 bits (e4m3, e8m0);
    `zeros` (..., n_blocks) fp16 for asymmetric int4."""
    fmt: KVFormat
    data: torch.Tensor
    scales: torch.Tensor | None = None
    tensor_scale: float = 1.0
    zeros: torch.Tensor | None = None
    length: int = 0  # 11-21 (C2): token-axis formats only, the number of real tokens (data is padded to the block)


# --- rotation -----------------------------------------------------------------

def hadamard_matrix(n: int, device=None) -> torch.Tensor:
    """Sylvester Walsh-Hadamard matrix of size n (a power of 2), scaled to be orthonormal."""
    assert n > 0 and n & (n - 1) == 0, "Hadamard size must be a power of 2"
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(n)).to(device)


@functools.lru_cache(maxsize=None)
def _hadamard_cached(n: int, device: torch.device) -> torch.Tensor:
    # 11-11: built once per device (rebuilding it on the CPU and copying it over on every call made per-token
    # fake quantization in bench/needle.py about 0.6 s per generated token)
    return hadamard_matrix(n, device).to(torch.float32)


@functools.lru_cache(maxsize=None)
def _e2m1_grid(device: torch.device) -> torch.Tensor:
    return torch.tensor(E2M1_GRID, device=device)


def rotate(x: torch.Tensor, block: int) -> torch.Tensor:
    """x (..., D) times the block-diagonal Hadamard matrix; its own inverse."""
    if not block:
        return x
    d = x.shape[-1]
    h = _hadamard_cached(block, x.device)
    return (x.float().reshape(*x.shape[:-1], d // block, block) @ h).reshape(x.shape).to(x.dtype)


# --- element codes ------------------------------------------------------------

def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """(..., D) codes in 0..15 -> (..., D/2) uint8."""
    c = codes.to(torch.uint8)
    return c[..., 0::2] | (c[..., 1::2] << 4)


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """(..., D/2) uint8 -> (..., D) codes in 0..15."""
    return torch.stack([packed & 0xF, packed >> 4], dim=-1).flatten(-2)


def e2m1_encode(x: torch.Tensor) -> torch.Tensor:
    """Codes of the nearest E2M1 values (ties to even mantissa), saturating at +-6."""
    grid = _e2m1_grid(x.device)
    mids = (grid[1:] + grid[:-1]) / 2
    a = x.float().abs().clamp(max=E2M1_MAX)
    lo, hi = torch.bucketize(a, mids, right=False), torch.bucketize(a, mids, right=True)
    idx = torch.where(lo % 2 == 0, lo, hi)  # grid index parity is the mantissa bit
    return idx | ((x < 0) & (idx > 0)).to(idx.dtype) << 3


def e2m1_decode(codes: torch.Tensor) -> torch.Tensor:
    grid = _e2m1_grid(codes.device)
    v = grid[(codes & 7).long()]
    return torch.where(codes >= 8, -v, v)


def _blocks(x: torch.Tensor, block: int) -> torch.Tensor:
    d = x.shape[-1]
    block = block or d
    assert d % block == 0
    return x.reshape(*x.shape[:-1], d // block, block)


# --- quantize / dequantize ----------------------------------------------------

def quantize(x: torch.Tensor, fmt: KVFormat | str) -> QTensor:
    """x (..., D) in any float dtype -> QTensor in `fmt`."""
    fmt = FORMATS[fmt] if isinstance(fmt, str) else fmt
    if fmt.tok_block:
        return _quantize_tok(x, fmt)
    x = rotate(x.float(), fmt.hadamard)
    if fmt.element == "bf16":
        return QTensor(fmt, x.to(torch.bfloat16))
    if fmt.element == "fp8":
        ts = max(x.abs().amax().item() / E4M3_MAX, 1e-12)
        return QTensor(fmt, (x / ts).to(torch.float8_e4m3fn), tensor_scale=ts)

    tiny = torch.finfo(torch.float32).tiny
    if fmt.scale == "tensor":  # 11-18: one scale for the whole tensor
        if fmt.element == "e2m1":
            ts = max(x.abs().amax().item() / E2M1_MAX, tiny)
            codes = e2m1_encode(x / ts)
        else:
            ts = max(x.abs().amax().item() / INT4_MAX, tiny)
            codes = (torch.round(x / ts).clamp(-INT4_MAX, INT4_MAX) + 8).to(torch.uint8)
        return QTensor(fmt, pack_nibbles(codes), None, ts)
    blocks = _blocks(x, fmt.block)
    amax = blocks.abs().amax(-1)
    ts = 1.0
    if fmt.element == "int4" and fmt.asym:
        lo = blocks.amin(-1).to(torch.float16).float()
        scale = ((blocks.amax(-1) - lo) / 15).to(torch.float16).float()
        codes = torch.round((blocks - lo[..., None]) / scale.clamp(min=tiny)[..., None]).clamp(0, 15).to(torch.uint8)
        return QTensor(fmt, pack_nibbles(codes.flatten(-2)), scale.to(torch.float16), zeros=lo.to(torch.float16))
    if fmt.element == "int4":
        scale = (amax / INT4_MAX).to(torch.float16).float()  # stored as fp16; quantize with the stored value
        q = torch.round(blocks / scale.clamp(min=tiny)[..., None]).clamp(-INT4_MAX, INT4_MAX)
        codes, stored = (q + 8).to(torch.uint8), scale.to(torch.float16)
    elif fmt.scale in ("e4m3", "e4m3_g1"):
        ts = max(amax.amax().item() / (E2M1_MAX * E4M3_MAX), tiny) if fmt.scale == "e4m3" else 1.0
        s8 = (amax / E2M1_MAX / ts).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
        scale = s8.float() * ts
        codes = e2m1_encode(blocks / scale.clamp(min=tiny)[..., None])
        stored = s8.view(torch.uint8)
    elif fmt.scale == "e8m0":  # OCP MX: 2^(floor(log2 amax) - 2), 2 = emax of E2M1
        e = (torch.floor(torch.log2(amax.clamp(min=tiny))) - 2).clamp(-127, 127)
        scale = torch.exp2(e)
        codes = e2m1_encode(blocks / scale[..., None])
        stored = (e + 127).to(torch.uint8)
    elif fmt.scale == "e8m0_uq":  # UltraQuant: MSE-motivated constant c = 0.156, codes saturate at +-6
        e = torch.round(torch.log2((0.156 * amax).clamp(min=tiny))).clamp(-127, 127)
        scale = torch.exp2(e)
        codes = e2m1_encode(blocks / scale[..., None])
        stored = (e + 127).to(torch.uint8)
    elif fmt.scale.startswith("e8m0_c"):  # s = 2^round(log2(c amax)), c from the name
        e = torch.round(torch.log2((float(fmt.scale[6:]) * amax).clamp(min=tiny))).clamp(-127, 127)
        scale = torch.exp2(e)
        codes = e2m1_encode(blocks / scale[..., None])
        stored = (e + 127).to(torch.uint8)
    else:
        raise ValueError(f"unknown format {fmt}")
    return QTensor(fmt, pack_nibbles(codes.flatten(-2)), stored, ts)


def dequantize(t: QTensor, rotate_back: bool = True) -> torch.Tensor:
    """Float32 values; `rotate_back=False` keeps them in the rotated basis, as a kernel sees them."""
    fmt = t.fmt
    if fmt.tok_block:  # 11-21 (C2)
        x = dequantize(QTensor(_tok_inner(fmt), t.data, t.scales, t.tensor_scale), rotate_back=False)
        return x[..., :t.length].transpose(-1, -2)
    if fmt.element in ("bf16", "fp8"):
        x = t.data.float() * t.tensor_scale
    else:
        codes = unpack_nibbles(t.data)
        if fmt.scale == "tensor":  # 11-18
            x = (e2m1_decode(codes) if fmt.element == "e2m1" else codes.float() - 8) * t.tensor_scale
            return rotate(x, fmt.hadamard) if rotate_back else x
        if fmt.element == "int4" and fmt.asym:
            x = (_blocks(codes.float(), fmt.block) * t.scales.float()[..., None] + t.zeros.float()[..., None]).flatten(-2)
            return rotate(x, fmt.hadamard) if rotate_back else x
        if fmt.element == "int4":
            vals, scale = codes.float() - 8, t.scales.float()
        else:
            vals = e2m1_decode(codes)
            if fmt.scale in ("e4m3", "e4m3_g1"):
                scale = t.scales.view(torch.float8_e4m3fn).float() * t.tensor_scale
            else:
                scale = torch.exp2(t.scales.float() - 127)
        x = (_blocks(vals, fmt.block) * scale[..., None]).flatten(-2)
    return rotate(x, fmt.hadamard) if rotate_back else x


# --- 11-21 (C2): scales along the token axis -------------------------------------

def _tok_inner(fmt: KVFormat) -> KVFormat:
    """The D-axis format that quantizes the transposed (..., D, S) tensor in blocks of tok_block."""
    return replace(fmt, name=fmt.name + "_inner", block=fmt.tok_block, tok_block=0)


def _quantize_tok(x: torch.Tensor, fmt: KVFormat) -> QTensor:
    """x (..., S, D) -> one scale per tok_block consecutive tokens of each channel (tensor scale as in nvfp4: global
    amax / (6 * 448), the same whatever the block axis). The token axis is moved last and zero-padded to a multiple of
    tok_block; zeros do not change a block's amax, so a tail of r < tok_block tokens is a PARTIAL block whose scale
    comes from its r real tokens only (like the last, partly filled block of a page if it were quantized as is).
    Blocks start at the first token of x. data (..., D, S_pad / 2) uint8, scales (..., D, S_pad / tok_block)."""
    assert not fmt.hadamard, "token-axis formats rotate nothing"
    s = x.shape[-2]
    xt = torch.nn.functional.pad(x.float().transpose(-1, -2), (0, -s % fmt.tok_block))
    q = quantize(xt, _tok_inner(fmt))
    return QTensor(fmt, q.data, q.scales, q.tensor_scale, length=s)
