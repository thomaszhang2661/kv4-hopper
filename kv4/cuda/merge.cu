// Split-KV merge for the Hopper decode kernels (11-11): out_o (B, HKV, splits, 4, 128) fp32, each split normalized
// by its own sum, out_ml (B, HKV, splits, 4, 2) = (base-2 max, sum) -> final output (B, HKV * 4, 128) bf16.
// One 128-thread block per (batch, kv head, query head); thread = dim.
#include <algorithm>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace {
constexpr int D = 128;  // query heads per kv head (G) come from o.size(3): 4 or 8

__global__ void __launch_bounds__(D) merge_kernel(const float* __restrict__ o, const float* __restrict__ ml,
                                                  __nv_bfloat16* __restrict__ out, int splits, int GQA) {
  const int head = blockIdx.x % GQA, bh = blockIdx.x / GQA, d = threadIdx.x;
  const float* m = ml + ((size_t)bh * splits * GQA + head) * 2;  // split i at m[i * GQA * 2]
  float mx = -INFINITY;
  for (int i = 0; i < splits; ++i) mx = fmaxf(mx, m[i * GQA * 2]);
  float num = 0.f, den = 0.f;
  for (int i = 0; i < splits; ++i) {
    const float w = exp2f(m[i * GQA * 2] - mx) * m[i * GQA * 2 + 1];
    num += w * o[(((size_t)bh * splits + i) * GQA + head) * D + d];
    den += w;
  }
  out[((size_t)bh * GQA + head) * D + d] = __float2bfloat16(num / den);
}

// Attention sink in BF16 (11-11): the first `ns` tokens of each sequence are not in the 4-bit cache; they are kept in
// BF16 (sk, sv: (B, HKV, ns, 128)) and joined here as one more partial, so the decode kernels' hot loops are untouched
// (they read tokens [ns, len) only, as a sequence of length len - ns that starts on a page boundary).
// q: (B, HKV * 4, 128) fp32 in the same basis as sk (with K rotation: rotated q and rotated sink keys, or neither).
// The kernels' l carries 2^P_SHIFT (decode_*.cu: P scaled by 2^10), so a sink token's weight is exp2(s - mx + lshift).
constexpr int MAXNS = 16;
__device__ __forceinline__ float load_f(const float* p) { return *p; }
__device__ __forceinline__ float load_f(const __nv_bfloat16* p) { return __bfloat162float(*p); }

template <typename QT>
__global__ void __launch_bounds__(D) merge_sink_kernel(const float* __restrict__ o, const float* __restrict__ ml,
                                                       const QT* __restrict__ q, const __nv_bfloat16* __restrict__ sk,
                                                       const __nv_bfloat16* __restrict__ sv,
                                                       __nv_bfloat16* __restrict__ out, int splits, int ns,
                                                       float score_mul, float lshift, int GQA) {
  __shared__ float red[D / 32][MAXNS];
  const int head = blockIdx.x % GQA, bh = blockIdx.x / GQA, d = threadIdx.x, w = d / 32, lane = d % 32;
  const float qd = load_f(q + ((size_t)bh * GQA + head) * D + d);
  const __nv_bfloat16* k = sk + (size_t)bh * ns * D;
  const __nv_bfloat16* v = sv + (size_t)bh * ns * D;
  for (int j = 0; j < ns; ++j) {
    float x = qd * __bfloat162float(k[j * D + d]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) x += __shfl_xor_sync(0xffffffffu, x, off);
    if (lane == 0) red[w][j] = x;
  }
  __syncthreads();
  float sc[MAXNS];
  const float* m = ml + ((size_t)bh * splits * GQA + head) * 2;
  float mx = -INFINITY;
  for (int j = 0; j < ns; ++j) {
    sc[j] = (red[0][j] + red[1][j] + red[2][j] + red[3][j]) * score_mul;
    mx = fmaxf(mx, sc[j]);
  }
  for (int i = 0; i < splits; ++i) mx = fmaxf(mx, m[i * GQA * 2]);
  float num = 0.f, den = 0.f;
  for (int i = 0; i < splits; ++i) {
    const float l = m[i * GQA * 2 + 1];
    if (l == 0.f) continue;  // an empty split (m = -inf) adds nothing
    const float wt = exp2f(m[i * GQA * 2] - mx) * l;
    num += wt * o[(((size_t)bh * splits + i) * GQA + head) * D + d];
    den += wt;
  }
  for (int j = 0; j < ns; ++j) {
    const float wt = exp2f(sc[j] - mx + lshift);
    num += wt * __bfloat162float(v[j * D + d]);
    den += wt;
  }
  out[((size_t)bh * GQA + head) * D + d] = __float2bfloat16(num / den);
}


// 11-12 night: sinks for a paged serving cache (vLLM plugin). The first NSK = 4 tokens of every sequence are written to
// the 4-bit pages as usual, and also kept per sequence in small pools: sk / sv (P, HKV, 4, 128) bf16 = K - mu (original
// basis) and V at full precision, k4 / v4 = the same tokens as the 4-bit cache holds them, raw (K rotated, code x scale
// without the tensor scale: exact in bf16). A sequence's slot is smap[block_table[b, 0]], valid if owner[slot] equals
// that block. For a valid sequence the 4-bit sink tokens are taken out of split 0 (their scores recomputed from qh, the
// kernel's own fp16 rotated q, so they match the kernel up to fp32 summation order) and the BF16 sinks added in; an
// invalid one is merged as plain 4-bit (always correct). Output: the G real query heads (rows >= G of o are padding).
constexpr int NSK = 4;
// 11-22 (KV4_MERGE=v2 work): the output type is a template parameter so that the test can read this kernel's result
// in fp32 (merge_sink_pool_f32); the bf16 instantiation is the kernel the plugin has always run.
__device__ __forceinline__ void store_out(__nv_bfloat16* p, float x) { *p = __float2bfloat16(x); }
__device__ __forceinline__ void store_out(float* p, float x) { *p = x; }

template <typename OT>
__global__ void __launch_bounds__(D) merge_sink_pool_kernel(
    const float* __restrict__ o, const float* __restrict__ ml, const __nv_bfloat16* __restrict__ q,
    const __half* __restrict__ qh, const int* __restrict__ bt, int bt_stride, const int* __restrict__ smap, int nmap,
    const int* __restrict__ owner, const __nv_bfloat16* __restrict__ sk, const __nv_bfloat16* __restrict__ sv,
    const __nv_bfloat16* __restrict__ k4, const __nv_bfloat16* __restrict__ v4, OT* __restrict__ out,
    int HKV, int splits, int GK, int G, float score_mul, float k4_mul, float v4_mul, float lshift) {
  __shared__ float red[D / 32][2 * NSK];
  const int head = blockIdx.x % G, bh = blockIdx.x / G, b = bh / HKV, h = bh % HKV, d = threadIdx.x;
  const int w = d / 32, lane = d % 32;
  const int blk = bt[(size_t)b * bt_stride];
  int slot = (blk >= 0 && blk < nmap) ? smap[blk] : -1;
  const bool valid = slot >= 0 && (owner == nullptr || owner[slot] == blk);  // no owner array: timing mode
  const float* m = ml + ((size_t)bh * splits * GK + head) * 2;  // split i at m[i * GK * 2]
  float sb[NSK], s4[NSK];
  if (valid) {
    const size_t base = ((size_t)slot * HKV + h) * NSK * D;
    const float qd = __bfloat162float(q[((size_t)b * HKV * G + h * G + head) * D + d]);
    const float qr = __half2float(qh[((size_t)bh * 8 + head) * D + d]);
    for (int j = 0; j < NSK; ++j) {
      float x = qd * __bfloat162float(sk[base + j * D + d]);
      float y = qr * __bfloat162float(k4[base + j * D + d]);
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        x += __shfl_xor_sync(0xffffffffu, x, off);
        y += __shfl_xor_sync(0xffffffffu, y, off);
      }
      if (lane == 0) red[w][j] = x, red[w][NSK + j] = y;
    }
    __syncthreads();
    for (int j = 0; j < NSK; ++j) {
      sb[j] = (red[0][j] + red[1][j] + red[2][j] + red[3][j]) * score_mul;
      s4[j] = (red[0][NSK + j] + red[1][NSK + j] + red[2][NSK + j] + red[3][NSK + j]) * score_mul * k4_mul;
    }
  }
  float mx = -INFINITY;
  for (int i = 0; i < splits; ++i) mx = fmaxf(mx, m[i * GK * 2]);
  if (valid)
    for (int j = 0; j < NSK; ++j) mx = fmaxf(mx, sb[j]);
  float num = 0.f, den = 0.f;
  for (int i = 0; i < splits; ++i) {
    float l = m[i * GK * 2 + 1];
    if (l == 0.f) continue;  // an empty split (m = -inf) adds nothing
    const float mi = m[i * GK * 2];
    float n = l * o[(((size_t)bh * splits + i) * GK + head) * D + d];  // the split's unnormalized sum
    if (valid && i == 0) {  // take the 4-bit sink tokens out (they are the first tokens of split 0)
      const size_t base = ((size_t)slot * HKV + h) * NSK * D;
      for (int j = 0; j < NSK; ++j) {
        const float p = exp2f(s4[j] - mi + lshift);
        l -= p;
        n -= __half2float(__float2half_rn(p)) * __bfloat162float(v4[base + j * D + d]) * v4_mul;
      }
      if (!(l > 0.f)) continue;  // all of split 0's weight was in the sinks
    }
    const float wt = exp2f(mi - mx);
    num += wt * n;
    den += wt * l;
  }
  if (valid) {
    const size_t base = ((size_t)slot * HKV + h) * NSK * D;
    for (int j = 0; j < NSK; ++j) {
      const float wt = exp2f(sb[j] - mx + lshift);
      num += wt * __bfloat162float(sv[base + j * D + d]);
      den += wt;
    }
  }
  store_out(out + ((size_t)b * HKV * G + h * G + head) * D + d, num / den);
}

// 11-22 (KV4_MERGE=v2, default off; docs/smallbatch_merge_log.md): merge_sink_pool_kernel with the split loop spread
// over warps instead of walked serially by every thread. Same grid (one CTA per (b, kv head, real head)), same result
// up to fp32 summation order. Before, every thread walked all splits twice (max, then sum) behind the dependent chain
// block table -> sink map -> owner -> sink dot products, so the time grew ~0.5 us per split at B = 1 (32 CTAs); the
// decode kernel's gain from more splits went back into the merge.
// Layout: lane l holds dims l, l + 32, l + 64, l + 96 (so the sink dot products reduce exactly as the old kernel's warp
// c did over dims 32c..32c+31: bit-identical sink scores). Warp 0 does the sink chain, split 0 (with the 4-bit sink
// removal) and the final combine; warps 1..nw-1 take splits 1, 2, ... round-robin, each with its own max over its
// splits (lane-parallel loads, one shuffle reduction) and its own sums; they hand (max, sum o, sum l) to warp 0 through
// shared memory, which rescales them by exp2(warp max - global max). The global max is the old kernel's (all splits'
// m and the BF16 sink scores), so only the factorization exp2(m_i - mx) = exp2(m_i - m_w) * exp2(m_w - mx) and the
// summation order differ. The chain (warp 0) and the split sums (other warps) overlap. As in FlashInfer's
// MergeStatesLargeNumIndexSetsKernel (include/flashinfer/attention/cascade.cuh, L285-353 at 253986e: threadIdx.y warps
// over index sets, then threadblock_sync_state L124-144 merges the warps in shared memory), but with one max pass per
// warp instead of an online state, and the sink swap kept.
// nw = 1 + min(max_warps - 1, splits - 1) warps, chosen at launch (splits = 1: one warp, which does everything).
constexpr int MAXW = 16;
template <typename OT>
__global__ void __launch_bounds__(MAXW * 32) merge_sink_pool_v2_kernel(
    const float* __restrict__ o, const float* __restrict__ ml, const __nv_bfloat16* __restrict__ q,
    const __half* __restrict__ qh, const int* __restrict__ bt, int bt_stride, const int* __restrict__ smap, int nmap,
    const int* __restrict__ owner, int P, const __nv_bfloat16* __restrict__ sk, const __nv_bfloat16* __restrict__ sv,
    const __nv_bfloat16* __restrict__ k4, const __nv_bfloat16* __restrict__ v4, OT* __restrict__ out,
    int HKV, int splits, int GK, int G, float score_mul, float k4_mul, float v4_mul, float lshift) {
  __shared__ float s_num[MAXW - 1][D];
  __shared__ float s_den[MAXW], s_mw[MAXW];
  const int head = blockIdx.x % G, bh = blockIdx.x / G, b = bh / HKV, h = bh % HKV;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, nw = blockDim.x / 32;
  const float* m = ml + ((size_t)bh * splits * GK + head) * 2;         // split i: m at m[i * GK * 2], l at +1
  const float* ob = o + ((size_t)bh * splits * GK + head) * D + lane;  // split i, dim lane + 32c: ob[i * GK * D + 32c]
  float m0 = 0.f, l = 0.f, n[4], sb[NSK], svr[NSK][4];  // warp 0's state across the barrier
  bool valid = false, use0 = false;
  if (warp > 0) {
    const int u = warp - 1, nwk = nw - 1;
    const int K = splits - 1 - u > 0 ? (splits - 1 - u + nwk - 1) / nwk : 0;  // this warp's splits 1 + u + nwk * k
    float mw = -INFINITY;
    for (int k = lane; k < K; k += 32) mw = fmaxf(mw, m[(size_t)(1 + u + nwk * k) * GK * 2]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) mw = fmaxf(mw, __shfl_xor_sync(0xffffffffu, mw, off));
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, den = 0.f;
    for (int k0 = 0; k0 < K; k0 += 32) {
      float mi_l = -INFINITY, li_l = 0.f;
      if (k0 + lane < K) {
        const float* mp = m + (size_t)(1 + u + nwk * (k0 + lane)) * GK * 2;
        mi_l = mp[0], li_l = mp[1];
      }
      const int cnt = min(32, K - k0);
#pragma unroll 4
      for (int k = 0; k < cnt; ++k) {
        const float mi = __shfl_sync(0xffffffffu, mi_l, k), li = __shfl_sync(0xffffffffu, li_l, k);
        const float* p = ob + (size_t)(1 + u + nwk * (k0 + k)) * GK * D;
        const float x0 = p[0], x1 = p[32], x2 = p[64], x3 = p[96];
        const bool e = li == 0.f;  // an empty split (m = -inf) adds nothing (selects: o of an empty split unused)
        const float wt = e ? 0.f : exp2f(mi - mw);
        acc[0] += e ? 0.f : wt * (li * x0);
        acc[1] += e ? 0.f : wt * (li * x1);
        acc[2] += e ? 0.f : wt * (li * x2);
        acc[3] += e ? 0.f : wt * (li * x3);
        den += e ? 0.f : wt * li;
      }
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) s_num[u][lane + 32 * c] = acc[c];
    if (lane == 0) s_mw[warp] = mw, s_den[warp] = den;
  } else {
  // warp 0: loads that do not depend on the chain first, then block table -> sink map -> {owner, sink rows}
  const int blk = bt[(size_t)b * bt_stride];
  m0 = m[0];
  const float l0 = m[1];
  float x[4], qd[4], qr[4];
  const size_t qo = ((size_t)b * HKV * G + h * G + head) * D + lane, qho = ((size_t)bh * 8 + head) * D + lane;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    x[c] = ob[32 * c];
    qd[c] = __bfloat162float(q[qo + 32 * c]);
    qr[c] = __half2float(qh[qho + 32 * c]);
  }
  const int slot = (blk >= 0 && blk < nmap) ? smap[blk] : -1;
  const bool spec = slot >= 0 && slot < P;  // rows of a slot in range are safe to load before the owner check
  float skr[NSK][4], k4r[NSK][4], v4r[NSK][4];
  const size_t base = ((size_t)(spec ? slot : 0) * HKV + h) * NSK * D + lane;
  if (spec) {
#pragma unroll
    for (int j = 0; j < NSK; ++j)
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const size_t a = base + j * D + 32 * c;
        skr[j][c] = __bfloat162float(sk[a]), k4r[j][c] = __bfloat162float(k4[a]);
        svr[j][c] = __bfloat162float(sv[a]), v4r[j][c] = __bfloat162float(v4[a]);
      }
  }
  valid = spec && (owner == nullptr || owner[slot] == blk);  // no owner array: timing mode
  float s4[NSK];
  if (valid) {
#pragma unroll
    for (int j = 0; j < NSK; ++j) {
      float xs[4], ys[4];
#pragma unroll
      for (int c = 0; c < 4; ++c) xs[c] = qd[c] * skr[j][c], ys[c] = qr[c] * k4r[j][c];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          xs[c] += __shfl_xor_sync(0xffffffffu, xs[c], off);
          ys[c] += __shfl_xor_sync(0xffffffffu, ys[c], off);
        }
#pragma unroll
      for (int c = 0; c < 4; ++c)  // lane 0's sums, which the old kernel stored as red[c][j]
        xs[c] = __shfl_sync(0xffffffffu, xs[c], 0), ys[c] = __shfl_sync(0xffffffffu, ys[c], 0);
      sb[j] = (xs[0] + xs[1] + xs[2] + xs[3]) * score_mul;
      s4[j] = (ys[0] + ys[1] + ys[2] + ys[3]) * score_mul * k4_mul;
    }
  }
  // split 0, exactly as the old kernel
  l = l0;
  use0 = l0 != 0.f;
#pragma unroll
  for (int c = 0; c < 4; ++c) n[c] = l0 * x[c];
  if (valid && use0) {
#pragma unroll
    for (int j = 0; j < NSK; ++j) {
      const float p = exp2f(s4[j] - m0 + lshift);
      l -= p;
      const float ph = __half2float(__float2half_rn(p));
#pragma unroll
      for (int c = 0; c < 4; ++c) n[c] -= ph * v4r[j][c] * v4_mul;
    }
    if (!(l > 0.f)) use0 = false;  // all of split 0's weight was in the sinks
  }
  }  // warp 0
  __syncthreads();  // the one barrier: workers' (max, sums) are in shared memory
  if (warp > 0) return;
  float mx = m0;  // the old kernel's max: all splits' m, then the BF16 sink scores
  for (int w = 1; w < nw; ++w) mx = fmaxf(mx, s_mw[w]);
  if (valid)
#pragma unroll
    for (int j = 0; j < NSK; ++j) mx = fmaxf(mx, sb[j]);
  float num[4] = {0.f, 0.f, 0.f, 0.f}, den = 0.f;
  if (use0) {
    const float wt = exp2f(m0 - mx);
#pragma unroll
    for (int c = 0; c < 4; ++c) num[c] += wt * n[c];
    den += wt * l;
  }
  for (int w = 1; w < nw; ++w) {
    const float mw = s_mw[w];
    if (mw == -INFINITY) continue;  // no non-empty split in this warp
    const float sc = exp2f(mw - mx);
#pragma unroll
    for (int c = 0; c < 4; ++c) num[c] += sc * s_num[w - 1][lane + 32 * c];
    den += sc * s_den[w];
  }
  if (valid) {
#pragma unroll
    for (int j = 0; j < NSK; ++j) {
      const float wt = exp2f(sb[j] - mx + lshift);
#pragma unroll
      for (int c = 0; c < 4; ++c) num[c] += wt * svr[j][c];
      den += wt;
    }
  }
  OT* dst = out + ((size_t)b * HKV * G + h * G + head) * D + lane;
#pragma unroll
  for (int c = 0; c < 4; ++c) store_out(dst + 32 * c, num[c] / den);
}

// q preparation for the FP4 decode kernels in one launch (11-12): (B, HQ, 128) bf16 or fp32 -> (B, HKV, 8, 128) fp16,
// rows g >= G zero, optionally times the orthonormal Sylvester Hadamard (formats.hadamard_matrix(128); symmetric, so
// q @ H = H q). Replaces rotate() + prep_q() in Python, about ten small launches (11-12: ~20 us of host time per step).
// One warp per output row, lane holds dims 4*lane .. 4*lane+3: butterflies of stride 1, 2 in registers, 4..64 by shuffle.
template <typename QT>
__global__ void __launch_bounds__(256) prep_q_rot_kernel(const QT* __restrict__ q, __half* __restrict__ out, int G,
                                                         bool rot) {
  const int bh = blockIdx.x, r = threadIdx.x / 32, lane = threadIdx.x % 32;  // bh = b * HKV + h; HQ index bh * G + r
  float x[4] = {0.f, 0.f, 0.f, 0.f};
  if (r < G) {
    const QT* src = q + ((size_t)bh * G + r) * D + lane * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) x[j] = load_f(src + j);
    if (rot) {
      float a = x[0], b = x[1], c = x[2], e = x[3];  // stride 1 then 2 (dims 4*lane + j)
      x[0] = a + b; x[1] = a - b; x[2] = c + e; x[3] = c - e;
      a = x[0]; b = x[1]; c = x[2]; e = x[3];
      x[0] = a + c; x[1] = b + e; x[2] = a - c; x[3] = b - e;
#pragma unroll
      for (int s = 1; s < 32; s <<= 1) {  // dim stride 4 * s
        const bool upper = lane & s;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float p = __shfl_xor_sync(0xffffffffu, x[j], s);
          x[j] = upper ? p - x[j] : x[j] + p;
        }
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) x[j] *= 0.08838834764831845f;  // 1 / sqrt(128)
    }
  }
  __half* dst = out + ((size_t)bh * 8 + r) * D + lane * 4;
  *reinterpret_cast<__half2*>(dst) = __floats2half2_rn(x[0], x[1]);
  *reinterpret_cast<__half2*>(dst + 2) = __floats2half2_rn(x[2], x[3]);
}

// 11-21 (C1, KV4_K8): prep_q_rot_kernel above followed by decode_fp4_k8.cu's prep_q8_kernel, in one launch (the
// non-K8 path keeps prep_q_rot_kernel unchanged). Same block / thread mapping as both: block = (b, h), warp r = row r,
// lane = dims 4*lane .. 4*lane+3 (prep_q8's threadIdx.x * 4). The rotated row is rounded to fp16 as prep_q_rot writes
// it, then: qs = amax over the 8 rows / 448 (1 if all zero), q8 = E4M3(x / qs), qd = fp16(E4M3(q8) * qs), the q the
// K8 kernel effectively uses (the sink merge needs exactly it). Bit-identical to the two launches by construction.
__device__ __forceinline__ void hadamard128_lane(float x[4], int lane) {
  float a = x[0], b = x[1], c = x[2], e = x[3];  // stride 1 then 2 (dims 4*lane + j)
  x[0] = a + b; x[1] = a - b; x[2] = c + e; x[3] = c - e;
  a = x[0]; b = x[1]; c = x[2]; e = x[3];
  x[0] = a + c; x[1] = b + e; x[2] = a - c; x[3] = b - e;
#pragma unroll
  for (int s = 1; s < 32; s <<= 1) {  // dim stride 4 * s
    const bool upper = lane & s;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float p = __shfl_xor_sync(0xffffffffu, x[j], s);
      x[j] = upper ? p - x[j] : x[j] + p;
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) x[j] *= 0.08838834764831845f;  // 1 / sqrt(128)
}

template <typename QT>
__global__ void __launch_bounds__(256) prep_q_rot8_kernel(const QT* __restrict__ q, uint8_t* __restrict__ q8,
                                                          float* __restrict__ qs, __half* __restrict__ qd, int G,
                                                          bool rot) {
  __shared__ float red[8];
  const int bh = blockIdx.x, r = threadIdx.x / 32, lane = threadIdx.x % 32;
  float x[4] = {0.f, 0.f, 0.f, 0.f};
  if (r < G) {
    const QT* src = q + ((size_t)bh * G + r) * D + lane * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) x[j] = load_f(src + j);
    if (rot) hadamard128_lane(x, lane);
  }
  float m = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) x[j] = __half2float(__float2half_rn(x[j])), m = fmaxf(m, fabsf(x[j]));
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
  if (lane == 0) red[r] = m;
  __syncthreads();
  m = red[0];
#pragma unroll
  for (int w = 1; w < 8; ++w) m = fmaxf(m, red[w]);
  const float sc = m > 0.f ? m / 448.f : 1.f;
  const size_t base = ((size_t)bh * 8 + r) * D + lane * 4;
  uint32_t packed = 0;
  float y[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const __nv_fp8_e4m3 e(x[j] / sc);
    packed |= (uint32_t)e.__x << (8 * j);
    y[j] = (float)e * sc;
  }
  *reinterpret_cast<uint32_t*>(q8 + base) = packed;
  *reinterpret_cast<__half2*>(qd + base) = __floats2half2_rn(y[0], y[1]);
  *reinterpret_cast<__half2*>(qd + base + 2) = __floats2half2_rn(y[2], y[3]);
  if (threadIdx.x == 0) qs[bh] = sc;
}
}  // namespace

torch::Tensor prep_q_rot(torch::Tensor q, int64_t hkv, bool rot) {
  TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.dim() == 3 && q.size(2) == D);
  const int B = q.size(0), HQ = q.size(1), G = HQ / hkv;
  // 11-12: any 1..8 query heads per kv head (rows G..7 are zero; the decode kernel runs G = 4 or 8 and the caller
  // drops the padded heads)
  TORCH_CHECK(HQ == G * hkv && G >= 1 && G <= 8, "1..8 query heads per kv head");
  auto out = torch::empty({B, hkv, 8, D}, q.options().dtype(torch::kFloat16));
  auto st = at::cuda::getCurrentCUDAStream();
  auto* o = reinterpret_cast<__half*>(out.data_ptr());
  if (q.scalar_type() == torch::kBFloat16)
    prep_q_rot_kernel<<<B * hkv, 256, 0, st>>>(reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), o, G, rot);
  else {
    TORCH_CHECK(q.scalar_type() == torch::kFloat32, "q must be bf16 or fp32");
    prep_q_rot_kernel<<<B * hkv, 256, 0, st>>>(q.data_ptr<float>(), o, G, rot);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "prep_q_rot launch failed");
  return out;
}

// 11-21 (C1): q (B, HQ, 128) bf16 / fp32 -> {q8 (B, HKV, 8, 128) uint8 E4M3, qs (B * HKV) fp32, qd (B, HKV, 8, 128)
// fp16}, == decode_fp4_k8 prep_q8(prep_q_rot(q, hkv, rot)) in one launch.
std::vector<torch::Tensor> prep_q_rot8(torch::Tensor q, int64_t hkv, bool rot) {
  TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.dim() == 3 && q.size(2) == D);
  const int B = q.size(0), HQ = q.size(1), G = HQ / hkv;
  TORCH_CHECK(HQ == G * hkv && G >= 1 && G <= 8, "1..8 query heads per kv head");
  auto q8 = torch::empty({B, hkv, 8, D}, q.options().dtype(torch::kUInt8));
  auto qs = torch::empty({(int64_t)B * hkv}, q.options().dtype(torch::kFloat));
  auto qd = torch::empty({B, hkv, 8, D}, q.options().dtype(torch::kFloat16));
  if (B == 0) return {q8, qs, qd};
  auto st = at::cuda::getCurrentCUDAStream();
  auto* q8p = q8.data_ptr<uint8_t>();
  auto* qdp = reinterpret_cast<__half*>(qd.data_ptr());
  if (q.scalar_type() == torch::kBFloat16)
    prep_q_rot8_kernel<<<B * hkv, 256, 0, st>>>(reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), q8p,
                                                qs.data_ptr<float>(), qdp, G, rot);
  else {
    TORCH_CHECK(q.scalar_type() == torch::kFloat32, "q must be bf16 or fp32");
    prep_q_rot8_kernel<<<B * hkv, 256, 0, st>>>(q.data_ptr<float>(), q8p, qs.data_ptr<float>(), qdp, G, rot);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "prep_q_rot8 launch failed");
  return {q8, qs, qd};
}

torch::Tensor merge_sink(torch::Tensor o, torch::Tensor ml, torch::Tensor q, torch::Tensor sk, torch::Tensor sv,
                         double score_mul, double lshift) {
  TORCH_CHECK(o.is_cuda() && o.is_contiguous() && o.dim() == 5 && (o.size(3) == 4 || o.size(3) == 8) && o.size(4) == D);
  const int GQA = o.size(3);
  TORCH_CHECK(ml.is_contiguous() && ml.dim() == 5);
  const int B = o.size(0), HKV = o.size(1), splits = o.size(2), ns = sk.size(2);
  TORCH_CHECK(q.is_contiguous() && q.numel() == (int64_t)B * HKV * GQA * D &&
              (q.scalar_type() == torch::kFloat32 || q.scalar_type() == torch::kBFloat16));
  TORCH_CHECK(sk.is_contiguous() && sv.is_contiguous() && sk.scalar_type() == torch::kBFloat16 &&
              sv.scalar_type() == torch::kBFloat16 && sk.size(0) == B && sk.size(1) == HKV && sk.size(3) == D &&
              sv.sizes() == sk.sizes() && ns <= MAXNS);
  auto out = torch::empty({B, HKV * GQA, D}, o.options().dtype(torch::kBFloat16));
  auto* skp = reinterpret_cast<const __nv_bfloat16*>(sk.data_ptr());
  auto* svp = reinterpret_cast<const __nv_bfloat16*>(sv.data_ptr());
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  auto st = at::cuda::getCurrentCUDAStream();
  if (q.scalar_type() == torch::kBFloat16)
    merge_sink_kernel<<<B * HKV * GQA, D, 0, st>>>(o.data_ptr<float>(), ml.data_ptr<float>(),
                                                   reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), skp, svp, op,
                                                   splits, ns, (float)score_mul, (float)lshift, GQA);
  else
    merge_sink_kernel<<<B * HKV * GQA, D, 0, st>>>(o.data_ptr<float>(), ml.data_ptr<float>(), q.data_ptr<float>(),
                                                   skp, svp, op, splits, ns, (float)score_mul, (float)lshift, GQA);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "merge_sink launch failed");
  return out;
}

torch::Tensor merge(torch::Tensor o, torch::Tensor ml) {
  TORCH_CHECK(o.is_cuda() && o.is_contiguous() && o.dim() == 5 && (o.size(3) == 4 || o.size(3) == 8) && o.size(4) == D);
  const int GQA = o.size(3);
  TORCH_CHECK(ml.is_contiguous() && ml.dim() == 5);
  const int B = o.size(0), HKV = o.size(1), splits = o.size(2);
  auto out = torch::empty({B, HKV * GQA, D}, o.options().dtype(torch::kBFloat16));
  merge_kernel<<<B * HKV * GQA, D, 0, at::cuda::getCurrentCUDAStream()>>>(
      o.data_ptr<float>(), ml.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), splits, GQA);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "merge launch failed");
  return out;
}


// o, ml (B, HKV, splits, GK, .) from decode_fp4_paged; q (B, HKV * G, 128) bf16 original basis; qh (B, HKV, 8, 128) fp16
// (prep_q_rot); block_table (B, >= 1) int32; smap (num_blocks) int32; owner (P) int32; sk, sv, k4, v4 (P, HKV, 4, 128)
// bf16. score_mul = softmax scale * log2(e); k4_mul / v4_mul: the 4-bit tensor scales (NVFP4) or 1 (MXFP4 raw values
// are true values). Returns (B, HKV * G, 128) bf16.
torch::Tensor merge_sink_pool_impl(torch::Tensor o, torch::Tensor ml, torch::Tensor q, torch::Tensor qh,
                                   torch::Tensor block_table, torch::Tensor smap, torch::Tensor owner, torch::Tensor sk,
                                   torch::Tensor sv, torch::Tensor k4, torch::Tensor v4, int64_t G, double score_mul,
                                   double k4_mul, double v4_mul, double lshift, bool v2, bool f32, int64_t max_warps) {
  TORCH_CHECK(o.is_cuda() && o.is_contiguous() && o.dim() == 5 && o.size(4) == D && ml.is_contiguous());
  const int B = o.size(0), HKV = o.size(1), splits = o.size(2), GK = o.size(3);
  TORCH_CHECK(G >= 1 && G <= GK, "G real heads <= kernel heads");
  TORCH_CHECK(q.is_contiguous() && q.scalar_type() == torch::kBFloat16 && q.numel() == (int64_t)B * HKV * G * D);
  TORCH_CHECK(qh.is_contiguous() && qh.scalar_type() == torch::kHalf && qh.size(2) == 8);
  TORCH_CHECK(block_table.scalar_type() == torch::kInt && block_table.stride(1) == 1 && block_table.size(0) == B);
  TORCH_CHECK(smap.is_contiguous() && smap.scalar_type() == torch::kInt && owner.is_contiguous() &&
              owner.scalar_type() == torch::kInt);
  for (auto* t : {&sk, &sv, &k4, &v4})
    TORCH_CHECK(t->is_contiguous() && t->scalar_type() == torch::kBFloat16 && t->dim() == 4 && t->size(1) == HKV &&
                t->size(2) == NSK && t->size(3) == D && (owner.numel() == 0 || t->size(0) == owner.size(0)));
  auto out = torch::empty({B, HKV * G, D}, o.options().dtype(f32 ? torch::kFloat : torch::kBFloat16));
  if (B == 0) return out;
  auto bf = [](torch::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()); };
  auto st = at::cuda::getCurrentCUDAStream();
  const float* op = o.data_ptr<float>();
  const float* mlp = ml.data_ptr<float>();
  const auto* qhp = reinterpret_cast<const __half*>(qh.data_ptr());
  const int* btp = block_table.data_ptr<int>();
  const int bts = (int)block_table.stride(0), nmap = (int)smap.numel();
  const int* own = owner.numel() ? owner.data_ptr<int>() : nullptr;
  const int nblk = B * HKV * (int)G;
  if (v2) {
    TORCH_CHECK(max_warps >= 1 && max_warps <= MAXW, "max_warps 1..16");
    const int nw = 1 + std::min<int>((int)max_warps - 1, splits - 1), P = (int)sk.size(0);
#define KV4_V2(OT) merge_sink_pool_v2_kernel<OT><<<nblk, nw * 32, 0, st>>>(                                        \
        op, mlp, bf(q), qhp, btp, bts, smap.data_ptr<int>(), nmap, own, P, bf(sk), bf(sv), bf(k4), bf(v4),          \
        reinterpret_cast<OT*>(out.data_ptr()), HKV, splits, GK, (int)G, (float)score_mul, (float)k4_mul,           \
        (float)v4_mul, (float)lshift)
    if (f32) KV4_V2(float); else KV4_V2(__nv_bfloat16);
#undef KV4_V2
  } else {
#define KV4_V1(OT) merge_sink_pool_kernel<OT><<<nblk, D, 0, st>>>(                                                   \
        op, mlp, bf(q), qhp, btp, bts, smap.data_ptr<int>(), nmap, own, bf(sk), bf(sv), bf(k4), bf(v4),             \
        reinterpret_cast<OT*>(out.data_ptr()), HKV, splits, GK, (int)G, (float)score_mul, (float)k4_mul,           \
        (float)v4_mul, (float)lshift)
    if (f32) KV4_V1(float); else KV4_V1(__nv_bfloat16);
#undef KV4_V1
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "merge_sink_pool launch failed");
  return out;
}

// the plugin's merge (unchanged behaviour: the old kernel, bf16 output)
torch::Tensor merge_sink_pool(torch::Tensor o, torch::Tensor ml, torch::Tensor q, torch::Tensor qh,
                              torch::Tensor block_table, torch::Tensor smap, torch::Tensor owner, torch::Tensor sk,
                              torch::Tensor sv, torch::Tensor k4, torch::Tensor v4, int64_t G, double score_mul,
                              double k4_mul, double v4_mul, double lshift) {
  return merge_sink_pool_impl(o, ml, q, qh, block_table, smap, owner, sk, sv, k4, v4, G, score_mul, k4_mul, v4_mul,
                              lshift, false, false, 1);
}

// 11-22: the old kernel with an fp32 output (tests only) and the v2 kernel (KV4_MERGE=v2), bf16 or fp32 output
torch::Tensor merge_sink_pool_f32(torch::Tensor o, torch::Tensor ml, torch::Tensor q, torch::Tensor qh,
                                  torch::Tensor block_table, torch::Tensor smap, torch::Tensor owner, torch::Tensor sk,
                                  torch::Tensor sv, torch::Tensor k4, torch::Tensor v4, int64_t G, double score_mul,
                                  double k4_mul, double v4_mul, double lshift) {
  return merge_sink_pool_impl(o, ml, q, qh, block_table, smap, owner, sk, sv, k4, v4, G, score_mul, k4_mul, v4_mul,
                              lshift, false, true, 1);
}

torch::Tensor merge_sink_pool_v2(torch::Tensor o, torch::Tensor ml, torch::Tensor q, torch::Tensor qh,
                                 torch::Tensor block_table, torch::Tensor smap, torch::Tensor owner, torch::Tensor sk,
                                 torch::Tensor sv, torch::Tensor k4, torch::Tensor v4, int64_t G, double score_mul,
                                 double k4_mul, double v4_mul, double lshift, int64_t max_warps, bool f32) {
  return merge_sink_pool_impl(o, ml, q, qh, block_table, smap, owner, sk, sv, k4, v4, G, score_mul, k4_mul, v4_mul,
                              lshift, true, f32, max_warps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("merge", &merge, "split-KV merge -> (B, HQ, 128) bf16");
  m.def("merge_sink", &merge_sink, "split-KV merge plus BF16 sink tokens -> (B, HQ, 128) bf16");
  m.def("merge_sink_pool", &merge_sink_pool, "split-KV merge with per-sequence BF16 sinks from pools (vLLM plugin)");
  m.def("merge_sink_pool_f32", &merge_sink_pool_f32, "merge_sink_pool with an fp32 output (tests)");
  m.def("merge_sink_pool_v2", &merge_sink_pool_v2, "11-22 KV4_MERGE=v2: merge_sink_pool with the splits over warps; "
        "extra args max_warps (1..16), f32 output");
  m.def("prep_q_rot", &prep_q_rot, "q (B, HQ, 128) bf16/fp32 -> (B, HKV, 8, 128) fp16, optionally Hadamard-rotated");
  m.def("prep_q_rot8", &prep_q_rot8, "C1: prep_q_rot + decode_fp4_k8 prep_q8 in one launch -> q8, qs, qd");
}
