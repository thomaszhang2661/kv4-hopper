"""11-12: prefill accuracy inside vLLM. Wikitext-2 test perplexity from prompt logprobs (computed during prefill), so it
measures the prefill attention precision and, with small chunks (max_num_batched_tokens), the 4-bit cache read by the
continuation chunks. Configs (one engine each, in a child process; 11-12 night: boundary layers always native):
  auto                BF16 KV, default backend
  fp8                 vLLM fp8 KV (FA3 FP8 prefill)
  stock               ultraquant_4bit, vLLM's own kernels (vLLM's boundary: first / last 2 layers native)
  plugin_mx           ultraquant_4bit, our kernels, UltraQuant's MXFP4 bytes (not the recipe)
  plugin_nv           our kernels, NVFP4 + Hadamard K (recipe format), vLLM's boundary; needs KV4_CALIB
  plugin_nv_l0        the same, only layer 0 native (recipe)
  plugin_nv_l0_mean   + static K mean (no sinks yet: offline, mean without sinks hurt Llama)

    KV4_CALIB=results/calib/Llama-3.1-8B.pt python bench/vllm_prompt_ppl.py --windows 16 --len 8192 --chunk 2048
"""
import argparse
import math
import os
import subprocess
import sys

# 11-12 night: boundary layers always native (no all-layers-4-bit runs); env per config. KV4_CALIB is taken from the
# environment (bench/calib_kv.py output for the model) and turns the plugin's format to NVFP4 (the recipe).
CONFIGS = {"auto": ("auto", {}), "fp8": ("fp8", {}),
           "stock": ("ultraquant_4bit", {"KV4_VLLM": "0", "KV4_FORMAT": "mxfp4"}),             # vLLM boundary
           "plugin_mx": ("ultraquant_4bit", {"KV4_VLLM": "1", "KV4_FORMAT": "mxfp4"}),         # UQ bytes, ours
           "plugin_nv": ("ultraquant_4bit", {"KV4_VLLM": "1", "KV4_FORMAT": "nvfp4"}),         # recipe format
           "plugin_nv_l0": ("ultraquant_4bit", {"KV4_VLLM": "1", "KV4_FORMAT": "nvfp4", "KV4_BOUNDARY": "0"}),
           "plugin_nv_l0_mean": ("ultraquant_4bit", {"KV4_VLLM": "1", "KV4_FORMAT": "nvfp4", "KV4_BOUNDARY": "0",
                                                     "KV4_KMEAN": "1"}),
           # 11-13: the ablation's missing cell, sinks without mean removal
           "plugin_nv_l0_sinks": ("ultraquant_4bit", {"KV4_VLLM": "1", "KV4_FORMAT": "nvfp4", "KV4_BOUNDARY": "0",
                                                      "KV4_SINKS": "1"}),
           # 11-12 night: the full recipe (NVFP4 + K rotation + static K mean + 4 BF16 sinks + layer 0 BF16)
           "plugin_recipe": ("ultraquant_4bit", {"KV4_VLLM": "1", "KV4_FORMAT": "nvfp4", "KV4_BOUNDARY": "0",
                                                 "KV4_KMEAN": "1", "KV4_SINKS": "1"}),
           # 11-15: vLLM's stock UltraQuant at the scale constant proposed in #67176 (KV4_UQ_C patches vLLM's own
           # format.DEFAULT_CONSTANT_C; "stock" above is the default 0.156)
           "stock_c20": ("ultraquant_4bit", {"KV4_VLLM": "0", "KV4_FORMAT": "mxfp4", "KV4_UQ_C": "0.20"}),
           "stock_c177": ("ultraquant_4bit", {"KV4_VLLM": "0", "KV4_FORMAT": "mxfp4", "KV4_UQ_C": "0.177"}),
           # 11-16 (#41343 follow-up): FP8 with layer 0 skipped, and FP8 on all layers with a bias-derived K center
           # subtracted after RoPE (kv4_vllm._kcenter: raw bias as in rtp-llm #1583, or the RoPE-averaged bias over
           # the window length)
           "fp8_l0": ("fp8", {"KV4_SKIP_LAYERS": "0"}),
           "fp8_kbias": ("fp8", {"KV4_KCENTER": "bias"}),
           "fp8_kropeavg": ("fp8", {"KV4_KCENTER": "ropeavg"}),
           # 11-18 (baseline): vLLM's own NVFP4 KV as a user gets it (all layers, global scale 1.0, FlashInfer
           # fa2 on SM90), and the same with layer 0 at native precision
           "nvfp4": ("nvfp4", {"KV4_ATTN_BACKEND": "FLASHINFER"}),
           "nvfp4_l0": ("nvfp4", {"KV4_ATTN_BACKEND": "FLASHINFER", "KV4_SKIP_LAYERS": "0"})}


WIKI = os.environ.get("KV4_WIKI_PARQUET", os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
    "hub/datasets--Salesforce--wikitext/snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/"
    "wikitext-2-raw-v1/test-00000-of-00001.parquet"))
DATA_PY = os.environ.get("KV4_DATA_PY", os.path.expanduser("~/.conda/envs/fp4/bin/python"))


def wikitext():
    """The vLLM env has neither datasets nor pyarrow: read the cached parquet with the fp4 env's pandas."""
    try:
        from datasets import load_dataset
        return "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    except ImportError:
        code = f"import pandas, sys; sys.stdout.write(chr(10).join(pandas.read_parquet({WIKI!r})['text']))"
        lines = subprocess.run([DATA_PY, "-c", code], capture_output=True, text=True, check=True).stdout
        return "\n\n".join(lines.split("\n"))


def child(a):
    from vllm import LLM, SamplingParams
    dt, _ = CONFIGS[a.child]
    kw = dict(kv_cache_dtype=dt)
    if os.environ.get("KV4_SKIP_LAYERS"):  # 11-16
        kw["kv_cache_dtype_skip_layers"] = os.environ["KV4_SKIP_LAYERS"].split("+")
    if os.environ.get("KV4_ATTN_BACKEND"):  # 11-18
        kw["attention_backend"] = os.environ["KV4_ATTN_BACKEND"]
    if dt == "ultraquant_4bit":
        kw["block_size"] = 64  # our 64-token pages (vLLM lowers to 32 with native boundary layers)
    llm = LLM(a.model, max_model_len=a.len + 16, gpu_memory_utilization=0.85, enable_prefix_caching=False,
              max_num_batched_tokens=a.chunk, **kw)
    tok = llm.get_tokenizer()
    text = wikitext()
    ids = tok(text).input_ids
    wins = [ids[i * a.len: (i + 1) * a.len] for i in range(a.windows)]
    assert len(wins[-1]) == a.len, "not enough text"
    outs = llm.generate([{"prompt_token_ids": w} for w in wins], SamplingParams(max_tokens=1, prompt_logprobs=0))
    nll, n = 0.0, 0
    for w, o in zip(wins, outs):
        for t, lp in zip(w[1:], o.prompt_logprobs[1:]):
            nll -= lp[t].logprob
            n += 1
    print(f"RESULT {a.child} [{os.environ.get('KV4_FORMAT', '-')}, boundary {os.environ.get('KV4_BOUNDARY', 'vllm')}, "
          f"mean {os.environ.get('KV4_KMEAN', '0')}, sinks {os.environ.get('KV4_SINKS', '0')}]: ppl {math.exp(nll / n):.4f} nll {nll / n:.6f} over {n} tokens "
          f"({a.windows} x {a.len}, chunk {a.chunk})", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="auto,fp8,stock,plugin_mx,plugin_nv,plugin_nv_l0,plugin_nv_l0_mean")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--windows", type=int, default=16)
    p.add_argument("--len", type=int, default=8192)
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--child", default="")
    a = p.parse_args()
    if a.child:
        return child(a)
    for c in a.configs.split(","):
        _, extra = CONFIGS[c]
        env = {k: v for k, v in os.environ.items()
               if k not in ("KV4_NO_BOUNDARY", "KV4_BOUNDARY", "KV4_KMEAN", "KV4_SINKS", "KV4_UQ_C", "KV4_KCENTER",
                             "KV4_SKIP_LAYERS", "KV4_ATTN_BACKEND")}
        env.update(extra)
        env.setdefault("KV4_KCENTER_T", str(a.len))  # 11-16: RoPE-average over the window length
        subprocess.run([sys.executable, __file__, "--child", c, "--model", a.model, "--windows", str(a.windows),
                        "--len", str(a.len), "--chunk", str(a.chunk)], env=env)


if __name__ == "__main__":
    main()
