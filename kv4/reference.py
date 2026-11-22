"""Reference decode attention in float32: the ground truth every kernel is tested against.

Decode step: each sequence has one new query token per query head and attends
to its whole cache.
  q        (B, H_q, D)
  k, v     (B, H_kv, S, D)  tensors, or QTensors from kv4.formats
  seq_lens (B,) int: sequence b uses cache positions 0 .. seq_lens[b] - 1
  out      (B, H_q, D) float32
Grouped-query attention: query head h reads kv head h // (H_q // H_kv).
"""

from __future__ import annotations

import math

import torch

from kv4.formats import QTensor, dequantize


def decode_attention_ref(q, k, v, seq_lens, sm_scale: float | None = None) -> torch.Tensor:
    k = dequantize(k) if isinstance(k, QTensor) else k.float()
    v = dequantize(v) if isinstance(v, QTensor) else v.float()
    b, hq, d = q.shape
    hkv, s = k.shape[1], k.shape[2]
    sm_scale = sm_scale or 1 / math.sqrt(d)
    k = k.repeat_interleave(hq // hkv, dim=1)
    v = v.repeat_interleave(hq // hkv, dim=1)
    scores = torch.einsum("bhd,bhsd->bhs", q.float(), k) * sm_scale
    pad = torch.arange(s, device=q.device)[None, :] >= seq_lens.to(q.device)[:, None]  # (B, S)
    scores = scores.masked_fill(pad[:, None, :], float("-inf"))
    return torch.einsum("bhs,bhsd->bhd", scores.softmax(-1), v)
