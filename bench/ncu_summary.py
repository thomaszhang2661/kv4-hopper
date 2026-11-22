"""Table from Nsight Compute raw CSVs (slurm/ncu.sbatch): waiting or issue-bound?

Per kernel: duration, DRAM throughput (% of peak), issue slots busy (% of peak: near 100 means
issue-bound), achieved occupancy, registers, thread instructions per DRAM byte read (H200's budget
at full bandwidth is ~7: 132 SMs x 4 schedulers x 32 lanes x ~1.98 GHz / 4.8 TB/s), and the top
stall reasons (warps stalled per issued instruction).

    python bench/ncu_summary.py logs/ncu/*.csv
"""
import csv
import io
import sys
from pathlib import Path

UNIT = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "nsecond": 1e-3, "usecond": 1, "msecond": 1e3,
        "second": 1e6}


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return float("nan")


def rows(path):
    text = Path(path).read_text(errors="replace").splitlines()
    start = next((i for i, l in enumerate(text) if l.startswith('"ID"')), None)
    if start is None:
        return [], []
    r = list(csv.reader(io.StringIO("\n".join(text[start:]))))
    header, units, data = r[0], r[1], r[2:]
    return [dict(zip(header, d)) for d in data], dict(zip(header, units))


def scaled(row, units, key, to):
    return num(row.get(key)) * UNIT.get(units.get(key, ""), 1) / UNIT[to]


def main():
    print(f"{'impl':16s} {'kernel':34s} {'us':>8s} {'dram%':>6s} {'issue%':>7s} {'occ%':>5s} {'regs':>4s} "
          f"{'inst/B':>6s}  top stalls (warps per issue)")
    for path in sys.argv[1:]:
        data, units = rows(path)
        if not data:
            print(f"{Path(path).stem:16s} (no kernels profiled; see {Path(path).with_suffix('.err')})")
            continue
        for row in data:
            stalls = sorted(((num(v), k.split("issue_stalled_")[1].split("_per_issue")[0]) for k, v in row.items()
                             if k.startswith("smsp__average_warps_issue_stalled_") and k.endswith("_per_issue_active.ratio")),
                            reverse=True)[:4]
            nbytes = scaled(row, units, "dram__bytes_read.sum", "byte")
            inst = num(row.get("smsp__thread_inst_executed.sum"))
            print(f"{Path(path).stem:16s} {row.get('Kernel Name', '?')[:34]:34s} "
                  f"{scaled(row, units, 'gpu__time_duration.sum', 'usecond'):8.1f} "
                  f"{num(row.get('dram__throughput.avg.pct_of_peak_sustained_elapsed')):6.1f} "
                  f"{num(row.get('sm__inst_issued.avg.pct_of_peak_sustained_active')):7.1f} "
                  f"{num(row.get('sm__warps_active.avg.pct_of_peak_sustained_active')):5.1f} "
                  f"{num(row.get('launch__registers_per_thread')):4.0f} "
                  f"{inst / nbytes if nbytes else float('nan'):6.2f}  "
                  + ", ".join(f"{n} {v:.2f}" for v, n in stalls))


if __name__ == "__main__":
    main()
