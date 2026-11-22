"""11-12: pure decode-step timing with `vllm bench forward` (vLLM 0.33, #61339). It builds exact batches without the
scheduler and times one model forward as a CUDA graph, so no prefill mixes into the decode numbers (unlike
(t256 - t1) / 255 from `bench latency`). Dummy weights, KV cache zero-initialized, no length jitter.

Configs (one `vllm bench forward` process each):
  bf16        auto KV, default backend
  fp8         fp8 KV, default backend (FA3, FP8 query)
  fp8_fi      fp8 KV, FLASHINFER (vLLM #66590: faster than FA3 on H100)
  stock       ultraquant_4bit, vLLM's kernels, vLLM's default boundary (first / last 2 layers native)
  plugin      ultraquant_4bit, our kernels, same boundary
  plugin_l0   ultraquant_4bit, our kernels, only layer 0 native (our recipe; FLASH_ATTN for that layer)

    python bench/vllm_forward_matrix.py --configs fp8,fp8_fi,plugin --cases decode8x32k=8*1@32768 ...
"""
import argparse
import os
import subprocess
import sys

M = "meta-llama/Llama-3.1-8B"
CASES = ["b1_16k=1*1@16384", "b1_32k=1*1@32768", "b8_16k=8*1@16384", "b8_32k=8*1@32768", "b32_16k=32*1@16384",
         "b32_32k=32*1@32768", "b64_16k=64*1@16384", "b16_64k=16*1@65536", "b128_8k=128*1@8192"]
CONFIGS = {
    "bf16": (["--kv-cache-dtype", "auto"], {}),
    "fp8": (["--kv-cache-dtype", "fp8"], {}),
    "fp8_fi": (["--kv-cache-dtype", "fp8", "--attention-backend", "FLASHINFER"], {}),
    # 11-18: on the SXM nodes (no route to NVIDIA's artifactory) vLLM's gate has_nvidia_artifactory() fails, so fp8_fi
    # decodes with FlashInfer's native kernel ("decode_backend=flashinfer-native"); on a connected machine vLLM picks
    # TensorRT-LLM's XQA. VLLM_HAS_FLASHINFER_CUBIN=1 (vLLM's switch for pre-downloaded cubins) passes the gate; SM90
    # XQA is JIT-built from source, so no cubin is needed. This is the path a connected H200 takes.
    "fp8_fi_xqa": (["--kv-cache-dtype", "fp8", "--attention-backend", "FLASHINFER"], {"VLLM_HAS_FLASHINFER_CUBIN": "1"}),
    # 11-16: vLLM's own NVFP4 KV (#60620): on SM90 FlashInfer decodes it with fa2 (tensor cores), software FP4 decode.
    # vLLM's default for this dtype (every layer NVFP4, global scale 1.0); a baseline, not our recipe.
    "nvfp4_fi": (["--kv-cache-dtype", "nvfp4", "--attention-backend", "FLASHINFER"], {}),
    # 11-12 night: no all-layers-4-bit configs (user: always keep the boundary layers at native precision).
    # stock / plugin: vLLM's default boundary (first and last 2 layers native); plugin_l0: only layer 0 native.
    # No --attention-backend: the plugin package puts ULTRAQUANT in the CUDA priority list (KV4_AUTOSELECT).
    "stock": (["--kv-cache-dtype", "ultraquant_4bit", "--block-size", "64"], {"KV4_VLLM": "0"}),
    "plugin": (["--kv-cache-dtype", "ultraquant_4bit", "--block-size", "64"], {"KV4_VLLM": "1"}),
    "plugin_l0": (["--kv-cache-dtype", "ultraquant_4bit", "--block-size", "64"], {"KV4_VLLM": "1", "KV4_BOUNDARY": "0"}),
    # 11-12 night: explicit formats (NVFP4 = the recipe's, needs KV4_CALIB; MXFP4 = UltraQuant's bytes)
    "plugin_mx": (["--kv-cache-dtype", "ultraquant_4bit", "--block-size", "64"],
                  {"KV4_VLLM": "1", "KV4_FORMAT": "mxfp4"}),
    "plugin_nv": (["--kv-cache-dtype", "ultraquant_4bit", "--block-size", "64"],
                  {"KV4_VLLM": "1", "KV4_FORMAT": "nvfp4"}),
    "plugin_nv_l0": (["--kv-cache-dtype", "ultraquant_4bit", "--block-size", "64"],
                     {"KV4_VLLM": "1", "KV4_FORMAT": "nvfp4", "KV4_BOUNDARY": "0"}),
    # the full recipe as in the accuracy runs (+ static K mean, + 4 BF16 sinks); KV4_SINK_TIMING makes every sequence
    # take the sink path, since bench forward never prefills (no sink would be captured otherwise)
    "plugin_recipe": (["--kv-cache-dtype", "ultraquant_4bit", "--block-size", "64"],
                      {"KV4_VLLM": "1", "KV4_FORMAT": "nvfp4", "KV4_BOUNDARY": "0", "KV4_KMEAN": "1", "KV4_SINKS": "1",
                       "KV4_SINK_TIMING": "1"}),
    # 11-13: the same with up to 32 splits (few KV heads at small batch)
    "plugin_recipe_s32": (["--kv-cache-dtype", "ultraquant_4bit", "--block-size", "64"],
                          {"KV4_VLLM": "1", "KV4_FORMAT": "nvfp4", "KV4_BOUNDARY": "0", "KV4_KMEAN": "1", "KV4_SINKS": "1",
                           "KV4_SINK_TIMING": "1", "KV4_MAX_SPLITS": "32"}),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="fp8,fp8_fi,bf16,stock")
    p.add_argument("--cases", nargs="*", default=CASES)
    p.add_argument("--model", default=M)
    p.add_argument("--max-model-len", default="66048")
    p.add_argument("--out-dir", default="results/vllm_forward")
    # 11-20: weight quantization for every config (e.g. fp8 = vLLM's online FP8 weights from the BF16 checkpoint, here
    # dummy weights); output files get the suffix _w<name>. Default: none (BF16 weights, unchanged file names).
    p.add_argument("--weight-quant", default="")
    a = p.parse_args()
    try:
        import importlib.metadata as md
        print(f"versions: vllm {md.version('vllm')}, flashinfer-python {md.version('flashinfer-python')}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"versions: unknown ({e})", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    for c in a.configs.split(","):
        args, env_extra = CONFIGS[c]
        env = {k: v for k, v in os.environ.items()
               if k not in ("KV4_NO_BOUNDARY", "KV4_BOUNDARY", "KV4_KMEAN", "KV4_SINKS", "KV4_SINK_TIMING")}
        env.setdefault("KV4_VLLM", "0")
        env.update(env_extra)
        cmd = ["vllm", "bench", "forward", "--model", a.model, "--max-model-len", a.max_model_len,
               "--gpu-memory-utilization", "0.90", "--kv-len-jitter", "0",
               "--output-json", f"{a.out_dir}/{os.path.basename(a.model)}_{c}{'_w' + a.weight_quant if a.weight_quant else ''}.json",
               *args, *(["--quantization", a.weight_quant] if a.weight_quant else [])]
        for case in a.cases:
            cmd += ["--case", case]
        print(f"=== {c}: {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        lines = (r.stdout + r.stderr).splitlines()
        keep = [l for l in lines if any(t in l for t in ("ms", "case", "Case", "Error", "error", "Traceback"))
                and "WARNING" not in l and "rocm" not in l]
        print("\n".join(keep[-60:]), flush=True)
        # 11-18: which decode kernel FlashInfer resolved (XQA on SM90 with an FP8 cache needs NVIDIA's artifactory)
        seen = set()
        for l in lines:
            if any(t in l for t in ("resolved query dtypes", "artifactory", "TRTLLM", "JIT module xqa")):
                key = l.split("]", 1)[-1][:200]
                if key not in seen:
                    seen.add(key)
                    print("BACKEND", l[:300], flush=True)
        print(f"=== {c} exit {r.returncode}", flush=True)


if __name__ == "__main__":
    main()
