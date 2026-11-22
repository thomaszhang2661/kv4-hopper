"""Build every CUDA extension in kv4/cuda (sm_90a) without running it: the CI compile check, and a warm-up of the
JIT cache before a job. Needs nvcc and a CUDA build of torch; no GPU (set TORCH_CUDA_ARCH_LIST=9.0a).
    python tools/build_kernels.py [decode_fp4 decode_int4 kside merge store_fp4]
"""
import importlib
import sys
import time

MODULES = ["decode_fp4", "decode_int4", "kside", "merge", "store_fp4"]


def main():
    failed = []
    for name in sys.argv[1:] or MODULES:
        t = time.time()
        try:
            importlib.import_module(f"kv4.cuda.{name}").ext()
            print(f"built kv4.cuda.{name} in {time.time() - t:.0f} s", flush=True)
        except Exception as e:  # report every module, then fail
            failed.append(name)
            print(f"FAILED kv4.cuda.{name}: {type(e).__name__}: {str(e)[-2000:]}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
