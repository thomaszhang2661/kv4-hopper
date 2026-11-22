"""Stage 1: decode attention over a BF16 KV cache, in Triton. (Written by you.)

Contract (the same for every stage; tests/test_kernels.py checks it against
kv4.reference.decode_attention_ref):
  q        (B, H_q, D)        bf16, contiguous
  k, v     (B, H_kv, S, D)    bf16, contiguous
  seq_lens (B,)               int32, on the GPU; positions >= seq_lens[b] are padding
  sm_scale float
  returns  (B, H_q, D)        float32 (or bf16; the test casts)
Query head h reads kv head h // (H_q // H_kv).

Hints, in the order you will need them:
1. Start with one program per (batch, query head): grid = (B, H_q). Loop over S
   in tiles of BLOCK_N keys: load a (BLOCK_N, D) tile of K, compute scores
   q . k^T * sm_scale, mask positions >= seq_len to -inf, keep a running max m,
   running sum l and running output acc (online softmax, as in Triton tutorial
   06 "Fused Attention"), and divide acc by l at the end.
2. Measure it (bench/bench_decode.py) before optimizing. Decode is memory-bound:
   the number to watch is GB/s of K+V read versus the GPU's peak (H200 ~4.8 TB/s).
3. Then: one program per (batch, kv head) handling all H_q / H_kv query heads of
   the group at once (each K/V tile is loaded once instead of once per query
   head), and split-K ("flash decoding"): split S across programs and combine
   the partial (m, l, acc) in a second small kernel. Small batches need split-K
   to fill the GPU's 132 SMs.
"""

import torch


def decode_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_lens: torch.Tensor,
                     sm_scale: float) -> torch.Tensor:
    raise NotImplementedError("stage 1: write the Triton kernel here")
