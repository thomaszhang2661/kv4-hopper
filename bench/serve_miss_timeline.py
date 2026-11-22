"""11-21: where in a serving run the SLO misses fall (cold start, a burst, or spread out), for runs the paper does not
explain yet (reviewer finding: W-reason 0.25 / 0.5 seed 0 with one API server; seed 2 at 1.25 / 1.5).
Per JSON (vllm bench serve --save-detailed): requests in arrival order, split into 10 equal bins; per bin the number
of requests over the TTFT target (2 s) and over the TPOT target (100 ms), the median TTFT, and the first 10 misses
(index, TTFT, TPOT). Pure Python; run on a CPU node.
    python bench/serve_miss_timeline.py results/vllm_serve_x/<file>.json ...
"""
import json
import statistics
import sys

for f in sys.argv[1:]:
    d = json.load(open(f))
    ttft, lat, ol, err = d["ttfts"], d["latencies"], d["output_lens"], d["errors"]
    n = len(ttft)
    rows = []
    for i, (t, l, o, e) in enumerate(zip(ttft, lat, ol, err)):
        tp = (l - t) / (o - 1) if (not e and o > 1) else float("nan")
        rows.append((i, t if not e else float("nan"), tp, bool(e)))
    print(f"== {f.split('/')[-1]}: {n} requests, {sum(r[3] for r in rows)} failed, duration {d['duration']:.0f} s, "
          f"output {d['output_throughput']:.0f} tok/s")
    for b in range(10):
        part = rows[b * n // 10:(b + 1) * n // 10]
        t_miss = sum(1 for r in part if not r[3] and r[1] > 2.0)
        p_miss = sum(1 for r in part if not r[3] and r[2] > 0.1)
        med = statistics.median([r[1] for r in part if not r[3]] or [float("nan")])
        print(f"  bin {b}: requests {part[0][0]}-{part[-1][0]}  TTFT>2s {t_miss:3d}  TPOT>100ms {p_miss:3d}  "
              f"median TTFT {med:.3f} s")
    misses = [r for r in rows if r[3] or r[1] > 2.0 or r[2] > 0.1][:10]
    print("  first misses:", ", ".join(f"#{i} ttft {t:.2f} tpot {1e3 * p:.0f}{' FAILED' if e else ''}"
                                      for i, t, p, e in misses))
