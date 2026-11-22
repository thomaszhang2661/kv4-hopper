import math

import pytest
import torch

from kv4.formats import (E2M1_GRID, FORMATS, dequantize, e2m1_decode, e2m1_encode, hadamard_matrix, pack_nibbles,
                         quantize, rotate, unpack_nibbles)
from kv4.reference import decode_attention_ref


def test_e2m1_codes_round_trip():
    codes = torch.arange(16)
    vals = e2m1_decode(codes)
    assert vals[:8].tolist() == list(E2M1_GRID) and vals[9:].tolist() == [-v for v in E2M1_GRID[1:]]
    back = e2m1_encode(vals)
    assert torch.equal(back[codes != 8], codes[codes != 8])  # code 8 is -0, encoded as +0


def test_e2m1_rounds_to_nearest_ties_to_even():
    x = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0, -2.6])
    assert e2m1_decode(e2m1_encode(x)).tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, -3.0]


def test_nibble_packing_round_trip():
    codes = torch.randint(0, 16, (3, 5, 128), dtype=torch.uint8)
    packed = pack_nibbles(codes)
    assert packed.shape == (3, 5, 64) and packed.dtype == torch.uint8
    assert torch.equal(unpack_nibbles(packed), codes)
    assert packed[0, 0, 0] == codes[0, 0, 0] | codes[0, 0, 1] << 4  # element 0 in the low nibble


def test_hadamard_is_orthogonal_and_preserves_dot_products():
    h = hadamard_matrix(128)
    assert torch.allclose(h @ h.T, torch.eye(128), atol=1e-6)
    q, k = torch.randn(4, 128), torch.randn(4, 128)
    assert torch.allclose((rotate(q, 64) * rotate(k, 64)).sum(-1), (q * k).sum(-1), atol=1e-4)
    assert torch.allclose(rotate(rotate(q, 128), 128), q, atol=1e-5)


@pytest.mark.parametrize("name", [n for n in FORMATS if not n.startswith("mxfp4_c")])  # the c sweep spans bad c on purpose
def test_round_trip_error_is_small(name):
    torch.manual_seed(0)
    x = torch.randn(2, 4, 64, 128)
    q = quantize(x, name)
    err = (dequantize(q) - x).pow(2).mean() / x.pow(2).mean()
    # 4-bit with block scales: ~1% relative MSE on Gaussian data. One scale per tensor (11-18 baselines) is coarser on
    # purpose: the largest of 65K values sets the step (INT4 ~3.5%, E2M1 below 2%).
    bound = {"bf16": 1e-4, "fp8": 1e-3, "int4_tensor": 0.05}.get(name, 0.02)
    assert err < bound, f"{name}: relative MSE {err:.4f}"


@pytest.mark.parametrize("name", [n for n, f in FORMATS.items() if f.element in ("int4", "e2m1") and f.scale != "tensor"])
def test_four_bit_storage_shapes(name):
    fmt = FORMATS[name]
    q = quantize(torch.randn(2, 3, 10, 128), name)
    assert q.data.shape == (2, 3, 10, 64) and q.data.dtype == torch.uint8
    assert q.scales.shape == (2, 3, 10, 128 // (fmt.block or 128))
    side = 2 if fmt.asym else 1  # scale and zero point
    assert fmt.bytes_per_row(128) == 64 + side * q.scales.shape[-1] * q.scales.element_size()


def test_mxfp4_scales_are_powers_of_two_covering_the_block():
    x = torch.randn(1, 1, 8, 128) * 10
    q = quantize(x, "mxfp4")
    scale = torch.exp2(q.scales.float() - 127)
    amax = x.reshape(1, 1, 8, 4, 32).abs().amax(-1)
    assert ((amax / scale >= 4) & (amax / scale < 8)).all()  # 2^(floor(log2 amax) - 2)


def test_nvfp4_values_lie_on_the_scaled_grid():
    x = torch.randn(1, 2, 8, 128)
    q = quantize(x, "nvfp4")
    vals = dequantize(q).reshape(1, 2, 8, 8, 16)
    scale = q.scales.view(torch.float8_e4m3fn).float() * q.tensor_scale
    ratio = (vals / scale[..., None]).abs()
    grid = torch.tensor(E2M1_GRID)
    assert torch.isclose(ratio[..., None], grid, atol=1e-5).any(-1).all()


def test_int4_token_wise_has_one_scale_per_row():
    x = torch.randn(1, 1, 4, 128)
    x[0, 0, 0, 5] = 100.0  # an outlier only hurts its own row
    q = quantize(x, "int4_tok")
    err = (dequantize(q) - x).pow(2).mean(-1)[0, 0]
    assert err[0] > 10 * err[1:].max()


def test_hadamard_spreads_an_outlier():
    x = torch.randn(1, 1, 4, 128)
    x[..., 5] = 100.0
    plain = (dequantize(quantize(x, "int4_tok")) - x).pow(2).mean()
    rotated = (dequantize(quantize(x, "int4_tok_h")) - x).pow(2).mean()
    assert rotated < plain / 4


def test_reference_matches_naive_attention_with_gqa_and_padding():
    torch.manual_seed(0)
    b, hq, hkv, s, d = 2, 8, 2, 37, 64
    q, k, v = torch.randn(b, hq, d), torch.randn(b, hkv, s, d), torch.randn(b, hkv, s, d)
    seq_lens = torch.tensor([37, 20])
    out = decode_attention_ref(q, k, v, seq_lens)
    for bi in range(b):
        for h in range(hq):
            n = seq_lens[bi]
            kk, vv = k[bi, h // (hq // hkv), :n], v[bi, h // (hq // hkv), :n]
            p = torch.softmax(kk @ q[bi, h] / math.sqrt(d), -1)
            assert torch.allclose(out[bi, h], p @ vv, atol=1e-5)


def test_reference_with_rotated_cache_equals_rotating_q():
    """What a kernel will do: read K, V in the rotated basis, rotate q, rotate the output back."""
    torch.manual_seed(0)
    q, k, v = torch.randn(1, 4, 128), torch.randn(1, 2, 16, 128), torch.randn(1, 2, 16, 128)
    seq_lens = torch.tensor([16])
    kq, vq = quantize(k, "int4_tok_h"), quantize(v, "int4_tok_h")
    full = decode_attention_ref(q, kq, vq, seq_lens)
    kr, vr = dequantize(kq, rotate_back=False), dequantize(vq, rotate_back=False)
    in_basis = rotate(decode_attention_ref(rotate(q, 128), kr, vr, seq_lens), 128)
    assert torch.allclose(full, in_basis, atol=1e-4)


def test_asymmetric_int4_hits_both_ends_of_the_range():
    x = torch.rand(1, 1, 8, 128) * 3 + 1  # all positive: symmetric INT4 wastes half its codes
    q = quantize(x, "int4a_tok")
    codes = unpack_nibbles(q.data)
    assert (codes.amin(-1) == 0).all() and (codes.amax(-1) == 15).all()
    sym = (dequantize(quantize(x, "int4_tok")) - x).pow(2).mean()
    asym = (dequantize(q) - x).pow(2).mean()
    assert asym < sym / 2


def test_mx_c_family_matches_uq_and_ocp():
    """s = 2^round(log2(c amax)): c = 0.156 is UltraQuant's rule exactly; c = 0.177 is OCP's floor(log2 amax) - 2
    except where frac(log2 amax) falls in [0.498, 0.5) (0.177 vs the exact 1 / (4 sqrt 2))."""
    from kv4.formats import quantize
    x = torch.randn(64, 1024, 128) * torch.rand(64, 1024, 1) * 3
    assert torch.equal(quantize(x, "mxfp4_c0.156").scales, quantize(x, "mxfp4_uq").scales)
    assert torch.equal(quantize(x, "mxfp4_c0.156_h").data, quantize(x, "mxfp4_uq_h").data)
    same = (quantize(x, "mxfp4_c0.177").scales == quantize(x, "mxfp4").scales).float().mean().item()
    assert same > 0.99


@pytest.mark.parametrize("name", [n for n, f in FORMATS.items() if f.scale == "tensor"])
def test_per_tensor_formats_store_no_row_scales(name):
    fmt = FORMATS[name]
    q = quantize(torch.randn(2, 3, 10, 128), name)
    assert q.scales is None
    assert fmt.bytes_per_row(128) == (128 if fmt.element == "fp8" else 64)
