"""Build (or load the cached build of) the BitDecoding Hopper binding.

Sources are not vendored: slurm/build_bitdecoding.sbatch clones BitDecoding (branch `hopper`,
pinned commit) and CUTLASS (the commit its submodule pins) into $BITDECODING_ROOT, default
/scratch/$USER/src. The build is sm_90a only and needs CUDA 12.x nvcc (the kv4base env sets it).

    python -m baselines.bitdecoding.build      # compile (a CPU node is enough)
    from baselines.bitdecoding.build import load; ext = load()
"""
from __future__ import annotations

import os
from pathlib import Path

BITDECODING_COMMIT = "584d0d3"   # branch hopper, 2025-08-04
CUTLASS_COMMIT = "3fe62887"      # the submodule pin of that commit


def _root() -> Path:
    return Path(os.environ.get("BITDECODING_ROOT", f"/scratch/{os.environ['USER']}/src"))


def load(verbose: bool = False):
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    from torch.utils.cpp_extension import load as _load

    src = _root() / "BitDecoding-hopper" / "hopper" / "src"
    cutlass = _root() / "cutlass"
    build = _root() / "build_bitdecoding"
    build.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).parent
    return _load(
        name="bitdecoding_hopper",
        sources=[str(here / "binding.cu"),
                 str(src / "genfile" / "flash_fwd_hdim128_fp16_split_sm90.cu"),
                 str(src / "genfile" / "flash_fwd_combine.cu"),
                 str(src / "genfile" / "flash_qpack_hdim128_fp16_sm80_4bit.cu")],
        extra_include_paths=[str(src), str(cutlass / "include"), str(cutlass / "tools" / "util" / "include")],
        extra_cflags=["-O3", "-std=c++20"],
        # C++20: torch >= 2.10 headers require it (BitDecoding itself used C++17).
        # The flags of BitDecoding's own hopper/CMakeLists.txt (sm_90a, maxrregcount 255, extended
        # MMA shapes) plus those of its setup.py (half/bf16 operators, relaxed constexpr, fast math).
        extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a",
                           "-maxrregcount=255", "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED", "-DNDEBUG",
                           "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                           "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                           "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math", "-w"],
        build_directory=str(build),
        verbose=verbose,
    )


if __name__ == "__main__":
    ext = load(verbose=True)
    print("built:", ext.__file__)
