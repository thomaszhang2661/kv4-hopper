"""Waiting or issue-bound, without performance counters (11-10: Explorer blocks them, ERR_NVGPUCTRPERM).

For each of our Triton kernels: disassemble the cubin, find the main loop (the largest span closed by a
backward branch), count its SASS instructions per iteration and group them by opcode. One iteration
reads K and V for BLOCK_N tokens of one kv head, so

    thread instructions per byte = n_loop * 32 * num_warps / (2 * BLOCK_N * bytes_per_row)

H200 can issue ~7 thread instructions per byte at peak bandwidth (132 SMs x 4 schedulers x 32 lanes x
1.98 GHz / 4.8 TB/s). With the measured bandwidth fraction f (from the Triton 3.8 main table),
implied issue utilization = inst/B * f / 7. Near 100%: issue-bound (fewer instructions per value
needed). Far below: the SM waits (overlap needed). Caveats: static count (predicated-off instructions
count as issued, which ncu would also count), prologue/epilogue ignored (long contexts), clock assumed.

    python bench/sass_count.py --csv results/bench_decode_H200_t38_b32.csv
"""
import argparse
import csv
import math
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import torch

from kv4.formats import FORMATS, quantize
from kv4.kernels import decode_triton as dt

BUDGET = 132 * 4 * 32 * 1.98e9 / 4.8e12  # ~6.97 thread instructions per byte on H200
INSTR = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(@!?U?P\w+\s+)?([A-Z0-9_]+)(\.[A-Z0-9_.]+)?\s*([^;]*);")
LABEL = re.compile(r"^\s*(\.L_x_\d+):")
TARGET = re.compile(r"`\((\.L_x_\d+)\)")


def sass(kernel) -> str:
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(kernel.asm["cubin"])
        f.flush()
        return subprocess.run(["nvdisasm", "-c", f.name], capture_output=True, text=True, check=True).stdout


def main_loop(text: str):
    """(instructions of the largest backward-branch loop, total instructions)."""
    ins, labels = [], {}
    for line in text.splitlines():
        m = LABEL.match(line)
        if m:
            labels[m.group(1)] = len(ins)
            continue
        m = INSTR.search(line)
        if m:
            ins.append((m.group(2), m.group(3) or "", line))
    best = (0, 0)
    for i, (op, _, line) in enumerate(ins):
        t = TARGET.search(line)
        if op == "BRA" and t and t.group(1) in labels and labels[t.group(1)] <= i:
            start = labels[t.group(1)]
            if i - start + 1 > best[1] - best[0]:
                best = (start, i + 1)
    return ins[best[0]:best[1]], len(ins)


def measured(csv_path, fmt, b, s):
    if not csv_path:
        return math.nan
    for r in csv.DictReader(open(csv_path)):
        if r["run"] == f"kernel_{fmt}" and int(r["batch"]) == b and int(r["ctx"]) == s:
            return float(r["peak_frac"])
    return math.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="", help="main-table CSV for the measured bandwidth fraction")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--ctx", type=int, default=16384)
    args = p.parse_args()
    import triton
    print(torch.cuda.get_device_name(), "triton", triton.__version__, f"budget {BUDGET:.2f} inst/B", flush=True)
    torch.manual_seed(0)
    b, s, hq, hkv, d = 2, 4096, 32, 8, 128
    q = torch.randn(b, hq, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, hkv, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    seq = torch.full((b,), s, device="cuda", dtype=torch.int32)
    cases = [(f, {}) for f in ["bf16", "fp8", "int4_tok", "int4a_tok", "int4_g32", "nvfp4", "mxfp4"]]
    cases.append(("int4_tok", {"swap": True}))
    for fmt, kw in cases:
        kq, vq = (k, v) if fmt == "bf16" else (quantize(k, fmt), quantize(v, fmt))
        dt.decode_attention(q, kq, vq, seq, 1 / math.sqrt(d), fmt_name=fmt, **kw)
        km = dt.LAST_META["kernel"]
        nw = km.metadata.num_warps
        loop, total = main_loop(sass(km))
        n = len(loop)
        bpr = FORMATS[fmt].bytes_per_row(d)
        per_b = n * 32 * nw / (2 * 128 * bpr)
        f = measured(args.csv, fmt, args.batch, args.ctx) if not kw else math.nan
        util = per_b * f / BUDGET
        hist = Counter(op for op, _, _ in loop).most_common(14)
        name = fmt + ("+swap" if kw else "")
        print(f"\n{name:14s} warps={nw} regs={km.n_regs} loop={n} SASS/iter (kernel {total}) "
              f"-> {per_b:.2f} inst/B ({per_b / BUDGET:.0%} of budget at peak BW); measured BW {f:.0%} "
              f"-> implied issue utilization {util:.0%}", flush=True)
        print("   " + ", ".join(f"{op} {c}" for op, c in hist), flush=True)


if __name__ == "__main__":
    main()
