"""Paired comparison of every config against a reference config over the same items (11-12): mean difference,
items worse / better, two-sided sign test, and a paired bootstrap 95% interval of the mean difference.

Works on any results CSV with a `config` column, a score column and item-identifying columns, e.g.
    python bench/paired.py results/longbench_prolong_g*.csv --score score --keys task,idx --by task
    python bench/paired.py results/reasoning_r1q7b_*.csv --score correct --keys id
    python bench/paired.py results/ruler_lc_prolong_*.csv --score score --keys task,ctx,trial --by task,ctx
"""
import argparse
import csv
import random
from collections import defaultdict
from math import comb


def sign_p(w, b):
    n = w + b
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(w, b) + 1)) / 2 ** n) if n else 1.0


def boot(d, reps=2000, seed=0):
    rng = random.Random(seed)
    n = len(d)
    ms = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return ms[int(0.025 * reps)], ms[int(0.975 * reps)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--score", default="score")
    p.add_argument("--keys", default="task,idx")
    p.add_argument("--ref", default="bf16")
    p.add_argument("--by", default="", help="also break down by these columns")
    p.add_argument("--scale", type=float, default=100.0, help="report differences times this (points)")
    a = p.parse_args()
    keys, by = a.keys.split(","), [x for x in a.by.split(",") if x]
    rows = [r for f in a.files for r in csv.DictReader(open(f))]
    val = {}
    for r in rows:
        val[(r["config"], tuple(r[k] for k in keys))] = float(r[a.score])
    configs = sorted({c for c, _ in val}, key=lambda c: (c != a.ref, c))
    groups = defaultdict(set)
    for r in rows:
        item = tuple(r[k] for k in keys)
        groups[("all",)].add(item)
        if by:
            groups[tuple(r[k] for k in by)].add(item)
    for g, items in sorted(groups.items()):
        ref = [val.get((a.ref, i)) for i in sorted(items)]
        print(f"== {'/'.join(g)}: {len(items)} items, {a.ref} mean {a.scale * sum(x for x in ref if x is not None) / max(1, sum(x is not None for x in ref)):.2f}")
        for c in configs:
            if c == a.ref:
                continue
            d = [val[(c, i)] - val[(a.ref, i)] for i in sorted(items) if (c, i) in val and (a.ref, i) in val]
            if not d:
                continue
            w, b = sum(x < 0 for x in d), sum(x > 0 for x in d)
            lo, hi = boot(d) if len(d) > 1 else (d[0], d[0])
            print(f"  {c:16s} n={len(d):4d} diff {a.scale * sum(d) / len(d):+7.2f}  [{a.scale * lo:+.2f}, "
                  f"{a.scale * hi:+.2f}]  worse {w:3d} better {b:3d}  sign p={sign_p(w, b):.4f}")


if __name__ == "__main__":
    main()
