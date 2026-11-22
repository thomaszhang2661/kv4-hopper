"""11-13: compare reasoning configs over repeated runs (vLLM greedy decoding is not reproducible across runs, so one
run per config is not enough; see docs/plan_CN.md 11-13). Input: CSVs of bench/vllm_reasoning_check.py, grouped by
config with glob patterns. Per config: accuracy of each run, mean and between-run SD; per problem: mean correctness
over that config's runs. Against the reference config: the mean difference and a paired sign-flip permutation test
on the per-problem differences (20000 draws), plus the run-to-run flip rate within each config.

    python bench/analyze_reasoning_runs.py --ref auto \\
        auto='results/vllm_reasoning_q25_7bi_500*_auto.csv' recipe='results/vllm_reasoning_q25_7bi_500*KMEAN1_KV4_SINKS1*'
"""
import argparse
import csv
import glob
import random
import statistics


def load(f):
    return {r["id"]: int(r["correct"]) for r in csv.DictReader(open(f))}


def boot2(rs_a, rs_b, ids, draws, seed=1):
    """11-18: difference of mean accuracy (a - b). Each draw resamples problems with replacement, then, per config,
    resamples its runs with replacement (the run-to-run variation of vLLM's greedy decoding); percentile interval."""
    rng = random.Random(seed)
    n = len(ids)
    ca = [[r[i] for i in ids] for r in rs_a]  # run x problem
    cb = [[r[i] for i in ids] for r in rs_b]
    ds = []
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        ra = [ca[rng.randrange(len(ca))] for _ in ca]
        rb = [cb[rng.randrange(len(cb))] for _ in cb]
        ma = sum(sum(r[j] for j in idx) for r in ra) / (n * len(ra))
        mb = sum(sum(r[j] for j in idx) for r in rb) / (n * len(rb))
        ds.append(ma - mb)
    ds.sort()
    m = sum(ds) / draws
    se = (sum((x - m) ** 2 for x in ds) / (draws - 1)) ** 0.5
    return ds[int(0.025 * draws)], ds[int(0.975 * draws) - 1], se


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", default="auto")
    p.add_argument("--draws", type=int, default=20000)
    p.add_argument("--boot", type=int, default=0,
                   help="11-18: two-level bootstrap draws for a 95%% interval of the difference (problems, then runs "
                        "within each config) and the minimum detectable difference 2.8 x SE (alpha 0.05, power 0.8)")
    p.add_argument("groups", nargs="+", help="name=glob")
    a = p.parse_args()
    runs = {}
    for g in a.groups:
        name, _, pat = g.partition("=")
        files = sorted(glob.glob(pat))
        assert files, f"no files for {name}: {pat}"
        runs[name] = [load(f) for f in files]
    ids = sorted(set.intersection(*[set(r) for rs in runs.values() for r in rs]))
    mean = {n: {i: sum(r[i] for r in rs) / len(rs) for i in ids} for n, rs in runs.items()}
    print(f"{len(ids)} problems common to all runs")
    print(f"{'config':12s} {'runs':>4s} {'acc per run':30s} {'mean':>6s} {'sd':>5s} {'flips/run pair':>14s}")
    for n, rs in runs.items():
        accs = [sum(r[i] for i in ids) / len(ids) for r in rs]
        flips = [sum(r1[i] != r2[i] for i in ids) for k, r1 in enumerate(rs) for r2 in rs[k + 1:]]
        sd = statistics.stdev(accs) if len(accs) > 1 else float("nan")
        fl = f"{statistics.mean(flips):.1f}" if flips else "-"
        print(f"{n:12s} {len(rs):4d} {' '.join(f'{x:.3f}' for x in accs):30s} {statistics.mean(accs):6.3f} {sd:5.3f} "
              f"{fl:>14s}")
    if a.ref not in mean:
        return
    rng = random.Random(0)
    print(f"\nagainst {a.ref} (per-problem mean correctness, paired sign-flip permutation test):")
    for n in runs:
        if n == a.ref:
            continue
        d = [mean[n][i] - mean[a.ref][i] for i in ids]
        obs = sum(d) / len(d)
        nz = [x for x in d if x != 0]
        hits = sum(abs(sum(x if rng.random() < 0.5 else -x for x in nz)) / len(d) >= abs(obs) - 1e-12
                   for _ in range(a.draws))
        print(f"  {n:12s} diff {obs * 100:+.2f} points, p = {(hits + 1) / (a.draws + 1):.3f} "
              f"({len(nz)} problems differ)")
        if a.boot:
            lo, hi, se = boot2(runs[n], runs[a.ref], ids, a.boot)
            print(f"  {'':12s} 95% CI [{lo * 100:+.2f}, {hi * 100:+.2f}] points (two-level bootstrap, {a.boot} draws); "
                  f"SE {se * 100:.2f}, minimum detectable difference {2.8 * se * 100:.2f} points")


if __name__ == "__main__":
    main()
