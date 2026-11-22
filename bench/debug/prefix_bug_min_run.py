"""11-12: run bench/debug/prefix_bug_min.py (the #46461 repro) for each kv dtype in a clean environment: our vLLM plugin
package not loaded (VLLM_PLUGINS=""), no KV4_* switches."""
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
for dt in sys.argv[1:] or ["auto", "turboquant_k8v4", "ultraquant_4bit"]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("KV4_")}
    env["VLLM_PLUGINS"] = ""
    subprocess.run([sys.executable, os.path.join(here, "prefix_bug_min.py"), dt], env=env)
