"""The vLLM plugin with the full recipe, end to end, on the cached-KV paths that broke vLLM's own 4-bit backends
(vllm#46460 / #46461 / #67789): prefix-cache hits and chunked prefill. Runs bench/plugin_cached_kv_check.py with the
vLLM env's interpreter (KV4_VLLM_PYTHON; that env has vllm 0.33, the plugin on PYTHONPATH, and no pytest), so this
file is collected from the kernel env. Skipped unless KV4_VLLM_PYTHON is set (slurm/gpu_tests.sbatch sets it).
"""
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PY = os.environ.get("KV4_VLLM_PYTHON")


@pytest.mark.skipif(not PY, reason="KV4_VLLM_PYTHON (the vLLM env's python) not set")
@pytest.mark.parametrize("mode", ["prefix", "chunked"])
def test_plugin_keeps_cached_kv(mode):
    p = subprocess.run([PY, "bench/plugin_cached_kv_check.py", mode], cwd=ROOT, capture_output=True, text=True,
                       env=dict(os.environ, KV4_VLLM_PYTHON=""))
    print(p.stdout)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
