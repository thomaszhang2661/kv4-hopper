"""Is the main loop software-pipelined? Count async copies in the compiled IR / PTX of each decode kernel.

num_stages having no effect means either pipelining does not matter or the loads are not pipelined at all;
this tells the two apart. Prints Triton/torch versions, registers, spills and shared memory too.
"""
import math
import re

import torch
import triton

from kv4.formats import quantize
from kv4.kernels import decode_triton as dt

print(torch.cuda.get_device_name(), "triton", triton.__version__, "torch", torch.__version__, flush=True)
HQ, HKV, D, B, S = 32, 8, 128, 2, 4096
torch.manual_seed(0)
q = torch.randn(B, HQ, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, HKV, S, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
seq = torch.full((B,), S, device="cuda", dtype=torch.int32)
PATTERNS = {  # name: regex, searched in ttgir unless noted
    "async_copy": r"async_copy_global_to_local",
    "async_wait": r"async_wait",
    "local_alloc": r"local_alloc",
    "convert_layout": r"convert_layout",
    "tma": r"async_tma|tensormap|cp\.async\.bulk",
}
try:
    fp8_ok = torch.cuda.get_device_capability()[0] >= 9
except Exception:
    fp8_ok = False
for name in ["bf16", "fp8", "int4_tok", "int4_g32", "nvfp4", "mxfp4"]:
    if name in ("fp8", "nvfp4") and not fp8_ok:
        print(f"{name}: skipped (needs sm90)", flush=True)
        continue
    kq, vq = (k, v) if name == "bf16" else (quantize(k, name), quantize(v, name))
    for ns in (2, 3, 4):
        try:
            dt.decode_attention(q, kq, vq, seq, 1 / math.sqrt(D), fmt_name=name, block_n=128, num_warps=4,
                                num_stages=ns)
        except Exception as e:  # e.g. bf16 ns=4 needs 200 KB shared memory, above A100's 163 KB
            print(f"{name:9s} ns={ns}: failed ({type(e).__name__}: {str(e)[:90]})", flush=True)
            continue
        km = dt.LAST_META["kernel"]
        ttgir, ptx = km.asm.get("ttgir", ""), km.asm.get("ptx", "")
        counts = {key: len(re.findall(pat, ttgir)) for key, pat in PATTERNS.items()}
        counts["ptx_cp.async"] = len(re.findall(r"cp\.async", ptx))
        shared = getattr(km.metadata, "shared", None) if hasattr(km, "metadata") else None
        print(f"{name:9s} ns={ns}: regs={km.n_regs} spills={km.n_spills} shared={shared} " +
              " ".join(f"{key}={val}" for key, val in counts.items()), flush=True)
