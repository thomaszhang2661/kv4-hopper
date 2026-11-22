"""Figures for paper/main.tex (11-17). Data only from the repo:
  fig_kernel.pdf : results/final_table_H200_v1.csv (job 11812198, H200 SXM, one job; times include the split merge)
  fig_vllm.pdf   : `vllm bench forward` medians of job 11979733 (H200 SXM, one job, FP8 = XQA decode),
                   results/vllm_forward_xqa (11-19; was job 11845880).
  fig_serving.pdf: not written by default (11-19): serving() plots results/serve_goodput_11-17.csv, whose FP8 runs
                   fell back to FlashInfer's native decode; the paper's serving numbers are tab:serve (XQA FP8).
                   KV4_SERVE_FIG=1 writes it anyway (void points left out: reason 1.25 fp8_fi, longctx 0.75 / 1.25 recipe).
    python paper/make_figures.py   (needs matplotlib)
"""
import csv
import collections
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "pdf.fonttype": 42})
BLUE, ORANGE, VERM, GREY = "#0072B2", "#E69F00", "#D55E00", "#9A9A9A"


def label(b, s):
    return f"{b}×{s // 1024}K"


def kernel():
    d = collections.defaultdict(dict)
    for x in csv.DictReader(open(os.path.join(ROOT, "results", "final_table_H200_v1.csv"))):
        d[(int(x["batch"]), int(x["ctx"]))][x["impl"]] = float(x["ms"])
    keys = sorted(d)
    series = [("flashinfer_bf16", "FlashInfer BF16", GREY), ("flashinfer_nvfp4", "FlashInfer NVFP4", ORANGE),
              ("cuda_nvfp4", "Ours NVFP4", BLUE), ("cuda_int4a", "Ours INT4", VERM)]
    fig, ax = plt.subplots(figsize=(6.5, 2.75))
    w = 0.2
    for j, (k, name, c) in enumerate(series):
        ys = [d[s]["flashinfer_fp8"] / d[s][k] for s in keys]
        ax.bar([i + (j - 1.5) * w for i in range(len(keys))], ys, w * 0.92, color=c, label=name, linewidth=0)
    ax.axhline(1.0, color="black", lw=0.6, label="FlashInfer FP8 (= 1)")
    ax.set_xticks(range(len(keys)), [label(*s) for s in keys], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Speed-up over\nFlashInfer FP8")
    ax.set_xlabel("Batch × context (Llama-3.1-8B head shape, H200)")
    ax.set_ylim(0, 2.9)
    # 11-17: legend above the axes in 3 columns (5 in one row was crowded and ran to the right edge)
    ax.legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01), fontsize=7, handlelength=1.6,
              columnspacing=1.4)
    ax.grid(axis="y", lw=0.3, color="#dddddd")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_kernel.pdf"))


def vllm():
    """11-19: job 11979733 (H200 SXM d4054, one job; FP8 resolved to XQA decode, logged), results/vllm_forward_xqa."""
    import json
    d = {c: {x["case"]: x["median_us"] / 1e3 for x in json.load(open(os.path.join(ROOT, "results", "vllm_forward_xqa",
         f"Llama-3.1-8B_{c}.json")))} for c in ("fp8_fi", "nvfp4_fi", "plugin_recipe")}
    cases = ["b1_16k", "b1_32k", "b8_16k", "b8_32k", "b32_16k", "b32_32k", "b64_16k", "b16_64k", "b128_8k"]
    lab = lambda c: c[1:].replace("_", "×").upper()
    fig, ax = plt.subplots(figsize=(6.5, 2.3))
    w = 0.27
    xs = range(len(cases))
    for k, (c, color, name) in enumerate((("fp8_fi", ORANGE, "FP8 (FlashInfer, XQA decode)"),
                                          ("nvfp4_fi", GREY, "vLLM NVFP4 (FlashInfer)"),
                                          ("plugin_recipe", BLUE, "Ours (full recipe)"))):
        ax.bar([i + (k - 1) * w for i in xs], [d[c][x] for x in cases], w * 0.92, color=color, label=name, linewidth=0)
    for i, x in enumerate(cases):
        ax.text(i + w, d["plugin_recipe"][x] + 0.5, f"{d['fp8_fi'][x] / d['plugin_recipe'][x]:.2f}×", ha="center",
                va="bottom", fontsize=6)
    ax.set_xticks(list(xs), [lab(x) for x in cases])
    ax.set_ylabel("Decode step (ms)")
    ax.set_xlabel("Batch × context (Llama-3.1-8B in vLLM 0.33.0, H200 SXM)")
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    ax.grid(axis="y", lw=0.3, color="#dddddd")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_vllm.pdf"))


SERVE_CSV = os.environ.get("KV4_SERVE_CSV", os.path.join(ROOT, "results", "serve_goodput_11-17.csv"))
SERVE_VOID = {("reason", 1.25, "fp8_fi"), ("longctx", 0.75, "recipe"), ("longctx", 1.25, "recipe")}
SERVE_FP8 = os.environ.get("KV4_SERVE_FP8", "fp8_fi")  # 11-18: fp8_xqa once the XQA reruns are in
SERVE_FP8_LABEL = {"fp8_fi": "FlashInfer FP8 (native decode)", "fp8_xqa": "FlashInfer FP8 (XQA decode)"}


def serving():
    """11-18: share of requests meeting the SLO (TTFT <= 2 s, TPOT <= 100 ms) and peak KV-cache use per request rate."""
    data = collections.defaultdict(dict)
    for r in csv.DictReader(open(SERVE_CSV)):
        key = (r["workload"], float(r["rate"]), r["config"])
        if key in SERVE_VOID or int(r.get("seed", 0) or 0) != 0:
            continue
        data[(r["workload"], r["config"])][float(r["rate"])] = (float(r["slo_share"]), float(r["max_kv_usage"]))
    panels = [("reason", "Reasoning-like: 1K in, 8K out"), ("longctx", "Long context: 16K in, 1K out"),
              ("prefill", "Prefill-heavy: 16K in, 512 out")]
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 3.4), sharex="col", gridspec_kw={"height_ratios": [1.6, 1]})
    for j, (w, title) in enumerate(panels):
        for cfg, color, marker, name in ((SERVE_FP8, ORANGE, "s", SERVE_FP8_LABEL.get(SERVE_FP8, SERVE_FP8)),
                                         ("recipe", BLUE, "o", "Ours (full recipe)")):
            pts = sorted(data[(w, cfg)].items())
            xs = [x for x, _ in pts]
            axes[0, j].plot(xs, [100 * v[0] for _, v in pts], color=color, marker=marker, ms=3.5, lw=1.4, label=name)
            axes[1, j].plot(xs, [100 * v[1] for _, v in pts], color=color, marker=marker, ms=3.5, lw=1.4)
        axes[0, j].axhline(99, color=GREY, lw=0.7, ls="--")
        axes[0, j].set_title(title, fontsize=7.5)
        axes[0, j].set_ylim(0, 104)
        axes[1, j].set_ylim(0, 105)
        axes[1, j].set_xlabel("Request rate (requests/s)")
        for i in (0, 1):
            axes[i, j].grid(lw=0.3, color="#dddddd")
            axes[i, j].set_axisbelow(True)
    axes[0, 0].text(0.27, 95.5, "99%", color=GREY, fontsize=6.5, va="top")
    axes[0, 0].set_ylabel("Requests meeting\nthe SLO (%)")
    axes[1, 0].set_ylabel("Peak KV cache\nused (%)")
    axes[0, 0].legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    fig.savefig(os.path.join(OUT, "fig_serving.pdf"))
    fig.savefig(os.path.join(OUT, "fig_serving_preview.png"), dpi=200)


if __name__ == "__main__":
    kernel()
    vllm()
    if os.environ.get("KV4_SERVE_FIG") == "1":
        serving()
    print("wrote", OUT)
