"""Every kernel in kv4.kernels.KERNELS against the float32 reference (GPU only)."""

import math

import pytest
import torch

pytest.importorskip("triton")  # the Triton kernels; CPU-only machines (CI, laptops) skip this file

from kv4.formats import quantize  # noqa: E402
from kv4.kernels import KERNELS  # noqa: E402
from kv4.reference import decode_attention_ref

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

SHAPES = [  # B, H_q, H_kv, S, D: MHA, Llama-3.1-8B's GQA, ragged lengths, one long sequence
    (2, 4, 4, 64, 128),
    (4, 32, 8, 1000, 128),
    (3, 32, 8, 4096, 128),
    (1, 32, 8, 32768, 128),
]


@cuda
@pytest.mark.parametrize("name", list(KERNELS))
@pytest.mark.parametrize("shape", SHAPES)
def test_kernel_matches_reference(name, shape):
    b, hq, hkv, s, d = shape
    torch.manual_seed(0)
    q = torch.randn(b, hq, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, hkv, s, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, hkv, s, d, device="cuda", dtype=torch.bfloat16)
    seq_lens = torch.randint(1, s + 1, (b,), device="cuda", dtype=torch.int32)
    seq_lens[0] = s
    kq, vq = (k, v) if name == "bf16" else (quantize(k, name), quantize(v, name))
    try:
        out = KERNELS[name](q, kq, vq, seq_lens, 1 / math.sqrt(d))
    except NotImplementedError as e:
        pytest.skip(str(e))
    ref = decode_attention_ref(q, kq, vq, seq_lens)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
