"""11-16: goodput table from the online-serving runs (slurm/vllm_serve.sbatch, `vllm bench serve --save-detailed`).
Per request: TTFT, TPOT = (latency - TTFT) / (output tokens - 1). A request meets the SLO if TTFT <= 2 s and
TPOT <= 100 ms (MLPerf Llama-3.1-8B server scenario; plan_CN 11-13). Per (workload, rate, config): completed, failed,
share meeting the SLO, goodput (met requests / duration), p99 TTFT / TPOT, output tok/s; from the /metrics csv: max KV
usage, max waiting, preemptions in the run. Pure Python (json + csv); run on a CPU node (the JSONs are up to 180 MB).
    python bench/analyze_serve.py [results/vllm_serve] [--ttft 2.0 --tpot 0.1] > results/serve_goodput.csv
"""
import argparse, csv, glob, json, os, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("dir", nargs="?", default="results/vllm_serve")
ap.add_argument("--ttft", type=float, default=2.0)
ap.add_argument("--tpot", type=float, default=0.100)
# 11-18: --boot N adds a 95% interval of the SLO share from resampling requests. Requests in one run are not
# independent (they share the queue), so this understates the uncertainty; the seed repeats give the real spread.
ap.add_argument("--boot", type=int, default=0)
a = ap.parse_args()


def boot_share(ok, draws, seed=1):
    import random
    rng, n = random.Random(seed), len(ok)
    xs = sorted(sum(ok[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return round(xs[int(0.025 * draws)], 4), round(xs[int(0.975 * draws) - 1], 4)

# 11-18: _s<seed> repeats; 11-19: any RTAG (_diag rerun, _a4 = 4 API servers, _s1a4 = seed 1 with 4), kept in "tag"
pat = re.compile(r"(?P<model>.+?)_(?P<cfg>recipe_k8|recipe|fp8_fi|fp8_xqa)_(?P<w>[a-z]+)_i\d+_o\d+_x[\d.]+_r(?P<r>[\d.]+)(?P<tag>(?:_[a-z0-9]+)?)\.json$")


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = min(len(xs) - 1, max(0, int(round(q / 100 * (len(xs) - 1)))))
    return xs[i]


def metrics(model, cfg, w, r, tag):
    f = os.path.join(a.dir, f"metrics_{model}_{cfg}_{w}_r{r}{tag}.csv")
    if not os.path.exists(f):
        return None, None, None
    rows = list(csv.DictReader(open(f)))
    if not rows:
        return None, None, None
    kv = max(float(x["kv_cache_usage_perc"]) for x in rows)
    wait = max(float(x["num_requests_waiting"]) for x in rows)
    pre = float(rows[-1]["num_preemptions_total"]) - float(rows[0]["num_preemptions_total"])
    return kv, wait, pre


out = csv.writer(sys.stdout)
out.writerow(["workload", "rate", "seed", "config", "completed", "failed", "slo_share", "goodput_rps", "p99_ttft_s",
              "p99_tpot_ms", "output_tok_s", "max_kv_usage", "max_waiting", "preemptions", "duration_s"]
             + (["slo_lo", "slo_hi"] if a.boot else []) + ["tag", "model"])  # 11-21: model column (32B / 72B share a dir)
rows = []
for f in glob.glob(os.path.join(a.dir, "*.json")):
    m = pat.match(os.path.basename(f))
    if not m:
        continue
    d = json.load(open(f))
    ttft, lat, ol, err = d["ttfts"], d["latencies"], d["output_lens"], d["errors"]
    met, tpots, ttfts, ok = 0, [], [], []
    for t, l, o, e in zip(ttft, lat, ol, err):
        if e:
            ok.append(0)  # a failed request misses the SLO (the share's denominator counts it)
            continue
        tp = (l - t) / (o - 1) if o > 1 else 0.0
        tpots.append(tp); ttfts.append(t)
        met += (t <= a.ttft and tp <= a.tpot); ok.append(int(t <= a.ttft and tp <= a.tpot))
    n = d["completed"] + d["failed"]
    kv, wait, pre = metrics(m["model"], m["cfg"], m["w"], m["r"], m["tag"])
    sd = re.match(r"_s(\d+)", m["tag"])
    rows.append([m["w"], float(m["r"]), int(sd[1]) if sd else 0, m["cfg"], d["completed"], d["failed"], round(met / n, 4),
                 round(met / d["duration"], 4), round(pct(ttfts, 99), 3), round(1e3 * pct(tpots, 99), 2),
                 round(d["output_throughput"], 1), kv, wait, pre, round(d["duration"], 1)]
                + (list(boot_share(ok, a.boot)) if a.boot else []) + [m["tag"].lstrip("_"), m["model"]])
for r in sorted(rows, key=lambda x: (x[-1], x[0], x[1], x[2], x[-2], x[3])):
    out.writerow(r)
