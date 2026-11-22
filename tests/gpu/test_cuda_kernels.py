"""The CUDA kernels (kv4/cuda, sm_90a) against FP32 references: every correctness check of the bench/*_check.py
scripts, run with KV4_CHECK_ONLY=1 (skips their timing sections; exit status = result). Needs a Hopper GPU and
nvcc (the kernels JIT-build on first use). Run: slurm/gpu_tests.sbatch, or `python -m pytest tests/gpu` on an H100/H200.

  paged_check   paged, variable-length NVFP4 / MXFP4 decode: ragged lengths, shuffled pages, poisoned tail bytes, splits
  sink_check    BF16 attention sinks joined in the merge kernel, dense and paged
  gqa8_check    4 and 8 query heads per KV head
  store_check   quantize-on-write (MXFP4, NVFP4) byte-exact against the offline packing, the continuation-prefill
                dequant, and the plugin's sink pools with and without K mean removal
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
hopper = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
                            reason="needs a Hopper GPU (sm_90a kernels)")


@hopper
@pytest.mark.parametrize("script", ["paged_check", "sink_check", "gqa8_check", "store_check"])
def test_check_script(script):
    env = dict(os.environ, KV4_CHECK_ONLY="1", PYTHONPATH=f"{ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    p = subprocess.run([sys.executable, f"bench/{script}.py"], cwd=ROOT, env=env, capture_output=True, text=True)
    failed = [x for x in p.stdout.splitlines() if "FAIL" in x]
    assert p.returncode == 0 and not failed, "\n".join(failed or p.stdout.splitlines()[-20:]) + p.stderr[-3000:]
