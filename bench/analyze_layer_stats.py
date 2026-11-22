"""11-14: per-layer K error relative to the centered K (what attention uses), from accuracy_kl.py --layer-stats JSONs.

Per layer: mean share of K energy (1 - |K - mu|^2 / |K|^2), and for each config the error energy divided by the raw
|K|^2 (experiment A's rel_err_k, per layer) and by the centered |K - mu|^2. Optionally (--expA 'glob') the Spearman over
layers between "error / centered K" under nvfp4_kh and experiment A's single-layer KL (nvfp4_kh#L).

    python bench/analyze_layer_stats.py results/lst_q25_7bi.json --expA 'results/expA_q25_7bi_*.csv'
"""
import argparse
import csv
import glob
import json

from analyze_expA import spearman


def main():
    p = argparse.ArgumentParser()
    p.add_argument("json")
    p.add_argument("--expA", default=None)
    p.add_argument("--md", action="store_true", help="markdown table")
    a = p.parse_args()
    d = json.load(open(a.json))
    meta = d.pop("_meta")
    cfgs = list(d)
    layers = sorted(int(k) for k in d[cfgs[0]])
    ref = d[cfgs[0]]
    share = {L: 1 - ref[str(L)]["sig_centered"] / ref[str(L)]["sig"] for L in layers}
    print(f"{meta['model']} ctx {meta['ctx']} x {meta['windows']}, sink {meta['sink']}, {meta['dtype']}, commit {meta['commit']}")
    head = ["L", "mean share"] + [f"{c} /|K|^2" for c in cfgs] + [f"{c} /|K-mu|^2" for c in cfgs]
    if a.md:
        print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    else:
        print(" ".join(f"{h:>18s}" for h in head))
    rc = {c: [] for c in cfgs}
    for L in layers:
        row = [str(L), f"{share[L]:.3f}"]
        raw = [d[c][str(L)]["err"] / d[c][str(L)]["sig"] for c in cfgs]
        cen = [d[c][str(L)]["err"] / d[c][str(L)]["sig_centered"] for c in cfgs]
        for c, v in zip(cfgs, cen):
            rc[c].append(v)
        row += [f"{v:.2e}" for v in raw] + [f"{v:.2e}" for v in cen]
        print("| " + " | ".join(row) + " |" if a.md else " ".join(f"{x:>18s}" for x in row))
    for c in cfgs:
        v = sorted(rc[c])
        print(f"{c}: error / centered K over layers: min {v[0]:.2e}, median {v[len(v) // 2]:.2e}, max {v[-1]:.2e} "
              f"(layer {layers[rc[c].index(v[-1])]})")
    if a.expA and "nvfp4_kh" in d:
        kl = {}
        for f in glob.glob(a.expA):
            for r in csv.DictReader(open(f)):
                name, _, L = r["config"].partition("#")
                if name == "nvfp4_kh" and L:
                    kl[int(L)] = float(r["mean_kl"])
        common = [L for L in layers if L in kl]
        x = [rc["nvfp4_kh"][layers.index(L)] for L in common]
        print(f"Spearman(single-layer KL_kh, error / centered K under kh) over {len(common)} layers = "
              f"{spearman([kl[L] for L in common], x):.3f}")


if __name__ == "__main__":
    main()
