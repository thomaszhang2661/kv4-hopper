"""Waiting or issue-bound, from NVBit opcode_hist logs (slurm/nvbit.sbatch; 11-10).

opcode_hist counts executed warp instructions per kernel launch (predicated-off ones included, as they
take issue slots). Only the kernels between the ==KV4 BEGIN / END markers of bench/ncu_target.py count
(one decode call after warm-up; split-KV merge kernels included).

    thread instructions per byte = sum(warp instructions) * 32 / bytes read
    implied issue utilization    = instructions per byte * measured bandwidth fraction / 7

7 = H200's issue budget at full bandwidth (132 SMs x 4 schedulers x 32 lanes x 1.98 GHz / 4.8 TB/s).
Measured fractions at B=32, S=16K come from the main-table CSVs.

    python bench/nvbit_summary.py logs/nvbit/*.log
"""
import csv
import re
import sys
from collections import Counter
from pathlib import Path

BUDGET = 132 * 4 * 32 * 1.98e9 / 4.8e12
ROOT = Path(__file__).resolve().parent.parent / "results"
KERNEL = re.compile(r"^kernel \d+ - (.*) - #thread-blocks (\d+),\s+kernel instructions (\d+)")
OPC = re.compile(r"^\s+(\S+) = (\d+)$")
BEGIN = re.compile(r"==KV4 BEGIN impl=(\S+) B=(\d+) S=(\d+) HKV=(\d+) D=(\d+) bytes=(\d+)")
# Static loop counts (bench/sass_count.py, c076ad2), thread instructions per byte.
STATIC = {"ours_bf16": 0.95, "ours_fp8": 6.23, "ours_int4_tok": 13.3, "ours_int4a_tok": 13.1,
          "ours_nvfp4": 20.6, "ours_mxfp4": 21.4}


def measured(impl, b, s):
    kind, fmt = impl.split("_", 1)
    try:
        if kind == "ours":
            for r in csv.DictReader(open(ROOT / "bench_decode_H200_t38_b32.csv")):
                if r["run"] == f"kernel_{fmt}" and int(r["batch"]) == b and int(r["ctx"]) == s:
                    return float(r["peak_frac"])
        if kind == "fi":
            for r in csv.DictReader(open(ROOT / "bench_flashinfer_H200_p32.csv")):
                if (r["impl"].startswith("flashinfer") and r["format"] == fmt and r["backend"] == "fa2"
                        and r["tensor_cores"] == "1" and int(r["batch"]) == b and int(r["ctx"]) == s):
                    return float(r["peak_frac"])
        if kind == "kside":  # K-side prototype, splits=4, nst=3 (the config bench/ncu_target.py runs)
            rows = [r for f in ("kside_proto_H200_v4.csv", "kside_proto_H200_v2.csv") if (ROOT / f).exists()
                    for r in csv.DictReader(open(ROOT / f))]
            for r in rows:
                if (r["var"] == fmt and int(r["batch"]) == b and int(r["ctx"]) == s and r["splits"] == "4"
                        and r["nst"] == "3"):
                    return float(r["peak_frac"])
        if kind == "cuda":  # full CUDA kernels: decode_int4_H200_*.csv / decode_fp4_H200_*.csv, splits=1, nst=2
            name = {"int4a": "cuda_int4a", "nvfp4": "cuda_nvfp4", "mxfp4": "cuda_mxfp4"}[fmt]
            for f in sorted(ROOT.glob("decode_*_H200_*.csv"), reverse=True):
                for r in csv.DictReader(open(f)):
                    if (r["impl"] == name and int(r["batch"]) == b and int(r["ctx"]) == s
                            and r["config"] == "splits=1,nst=2"):
                        return float(r["peak_frac"])
        if kind == "bd":
            for r in csv.DictReader(open(ROOT / "bench_bitdecoding_H200_b32.csv")):
                if r["impl"] == "bitdecoding" and int(r["batch"]) == b and int(r["ctx"]) == s:
                    return float(r["peak_frac"])
    except FileNotFoundError:
        pass
    return float("nan")


def parse(path):
    """[(kernel name, blocks, warp instructions, Counter of opcodes)] inside the markers, and the marker."""
    kernels, meta, inside = [], None, False
    for line in Path(path).read_text(errors="replace").splitlines():
        m = BEGIN.search(line)
        if m:
            meta, inside = m.groups(), True
            continue
        if "==KV4 END" in line:
            inside = False
            continue
        if not inside:
            continue
        m = KERNEL.match(line)
        if m:
            kernels.append((m.group(1), int(m.group(2)), int(m.group(3)), Counter()))
            continue
        m = OPC.match(line)
        if m and kernels:
            kernels[-1][3][m.group(1).split(".")[0]] += int(m.group(2))
    return kernels, meta


def main():
    print(f"{'impl':15s} {'kernels':>7s} {'inst/value':>10s} {'inst/B':>7s} {'static':>6s} {'budget':>6s} "
          f"{'BW':>5s} {'issue util':>10s} {'issue cap':>9s}  top opcodes per value")
    for path in sys.argv[1:]:
        kernels, meta = parse(path)
        if not meta or not kernels:
            print(f"{Path(path).stem:15s} (no kernels between markers)")
            continue
        impl, b, s, hkv, d, nbytes = meta[0], *map(int, meta[1:])
        values = b * s * hkv * d * (1 if impl.startswith("kside") else 2)  # the prototype reads K only
        warp = sum(k[2] for k in kernels)
        per_b = warp * 32 / nbytes
        per_v = warp * 32 / values
        f = measured(impl, b, s)
        ops = sum((k[3] for k in kernels), Counter())
        top = ", ".join(f"{op} {c * 32 / values:.2f}" for op, c in ops.most_common(10))
        print(f"{impl:15s} {len(kernels):7d} {per_v:10.2f} {per_b:7.2f} {STATIC.get(impl, float('nan')):6.2f} "
              f"{per_b / BUDGET:6.0%} {f:5.0%} {per_b * f / BUDGET:10.0%} {min(1, BUDGET / per_b):9.0%}  {top}")
        for name, blocks, n, _ in kernels:
            print(f"{'':15s}   {n * 32 / nbytes:6.2f} inst/B  {blocks:6d} blocks  {name[:110]}")


if __name__ == "__main__":
    main()
