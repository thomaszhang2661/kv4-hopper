"""Decode-attention kernels, by the KV format they read (see kv4.formats.FORMATS).

Each entry takes (q, k, v, seq_lens, sm_scale) where k and v are what
kv4.formats.quantize returns for that format (plain bf16 tensors for "bf16").
Add a stage's kernel here when it exists; tests and the benchmark pick it up.

Pass 1 (11-08): decode_triton covers every format with one generic implementation.
decode_bf16 is the original stub; the BF16 entry now points at decode_triton too.
"""

from functools import partial

from kv4.formats import FORMATS
from kv4.kernels import decode_triton

# 11-20: formats that exist only for offline accuracy baselines (one scale per tensor; NVFP4 with vLLM's global scale
# 1.0) have no decode kernel; the Triton kernel does not read them (gpu_tests 11995516 ran them and failed)
OFFLINE_ONLY = {n for n, f in FORMATS.items() if f.element in ("int4", "e2m1") and f.scale in ("tensor", "e4m3_g1")}
# 11-22: C2's token-axis V formats (nvfp4_t16 / nvfp4_t32) are offline accuracy formats here; the generic Triton
# kernel assumes D-axis scales and does not read them
OFFLINE_ONLY |= {n for n, f in FORMATS.items() if f.tok_block}
KERNELS = {name: partial(decode_triton.decode_attention, fmt_name=name) for name in FORMATS if name not in OFFLINE_ONLY}
