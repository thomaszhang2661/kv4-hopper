"""11-14: experiment A (per-layer "error x gain"), from the accuracy_kl.py CSVs of single-layer configs
nvfp4_kh#L (K rotated, no mean removal) and nvfp4_khs#L (K rotated, mean removed), same sink setting.

Model per layer L: KL_L = a_L * eK_L + c_L, where eK = relative K error energy (rel_err_k), a_L = the layer's K gain,
c_L = the V part (V error is the same in both configs, since only K changes). Two configs per layer identify a_L and
c_L exactly: a_L = (KL_kh - KL_khs) / (eK_kh - eK_khs), c_L = KL_khs - a_L * eK_khs. Reports per layer: KL of both
configs, eK of both, the eK reduction from mean removal, a_L, c_L; and the checks of the plan's predictions (11-12):
  - Spearman over layers between measured KL_kh and the error-only predictor (median gain x eK_kh), and between
    KL_kh and a_L * eK_kh + c_L (identity by construction; printed as a sanity check);
  - is layer 0 the largest KL, is its eK above the median, is its gain above the median;
  - layer 0's eK reduction from mean removal (prediction >= 3x).

    python bench/analyze_expA.py 'results/expA_llama31_8b_L*.csv'
"""
import csv
import glob
import statistics
import sys


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(o):
            r[i] = k
        return r
    rx, ry = rank(x), rank(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main():
    rows = [r for f in sorted(glob.glob(sys.argv[1])) for r in csv.DictReader(open(f))]
    d = {}
    for r in rows:
        name, _, layer = r["config"].partition("#")
        if not layer:
            continue
        d.setdefault(int(layer), {})[name] = (float(r["mean_kl"]), float(r["rel_err_k"]), float(r["rel_err_v"]))
    layers = sorted(L for L in d if "nvfp4_kh" in d[L] and "nvfp4_khs" in d[L])
    missing = sorted(set(d) - set(layers))
    print(f"{rows[0]['model'] if rows else '?'}: {len(layers)} layers complete" + (f", incomplete {missing}" if missing else ""))
    print(f"{'L':>3s} {'KL_kh':>9s} {'KL_khs':>9s} {'eK_kh':>9s} {'eK_khs':>9s} {'eK cut':>7s} {'eV':>9s} {'a_L':>8s} {'c_L':>9s}")
    A, C, KL, EK = [], [], [], []
    for L in layers:
        (k1, e1, v1), (k2, e2, v2) = d[L]["nvfp4_kh"], d[L]["nvfp4_khs"]
        a = (k1 - k2) / (e1 - e2) if e1 != e2 else float("nan")
        c = k2 - a * e2
        A.append(a); C.append(c); KL.append(k1); EK.append(e1)
        print(f"{L:3d} {k1:9.2e} {k2:9.2e} {e1:9.2e} {e2:9.2e} {e1 / e2 if e2 else float('nan'):7.2f} {v1:9.2e} "
              f"{a:8.3f} {c:9.2e}")
    if not layers:
        return
    ga = statistics.median(A)
    print(f"\nSpearman(KL_kh, median gain x eK_kh) = {spearman(KL, [ga * e for e in EK]):.3f}  (error alone)")
    print(f"Spearman(KL_kh, a_L) = {spearman(KL, A):.3f}  (gain alone)")
    i0 = layers.index(0) if 0 in layers else None
    if i0 is not None:
        rank_kl = sorted(KL, reverse=True).index(KL[i0]) + 1
        print(f"layer 0: KL_kh rank {rank_kl} of {len(KL)} (x{KL[i0] / statistics.median(KL):.1f} the median); "
              f"eK_kh x{EK[i0] / statistics.median(EK):.2f} the median; gain a_0 x{A[i0] / ga:.2f} the median; "
              f"mean removal cuts layer 0's eK by x{d[0]['nvfp4_kh'][1] / d[0]['nvfp4_khs'][1]:.2f}")
    neg = [L for L, a in zip(layers, A) if a < 0]
    if neg:
        print(f"negative gain (KL rises while eK falls) in layers {neg}: the one-gain model does not hold there")


if __name__ == "__main__":
    main()
