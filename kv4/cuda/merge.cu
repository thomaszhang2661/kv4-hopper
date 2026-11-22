// Split-KV merge for the Hopper decode kernels (11-11): out_o (B, HKV, splits, 4, 128) fp32, each split normalized
// by its own sum, out_ml (B, HKV, splits, 4, 2) = (base-2 max, sum) -> final output (B, HKV * 4, 128) bf16.
// One 128-thread block per (batch, kv head, query head); thread = dim.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

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

__global__ void __launch_bounds__(D) merge_sink_pool_kernel(
    const float* __restrict__ o, const float* __restrict__ ml, const __nv_bfloat16* __restrict__ q,
    const __half* __restrict__ qh, const int* __restrict__ bt, int bt_stride, const int* __restrict__ smap, int nmap,
    const int* __restrict__ owner, const __nv_bfloat16* __restrict__ sk, const __nv_bfloat16* __restrict__ sv,
    const __nv_bfloat16* __restrict__ k4, const __nv_bfloat16* __restrict__ v4, __nv_bfloat16* __restrict__ out,
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
  out[((size_t)b * HKV * G + h * G + head) * D + d] = __float2bfloat16(num / den);
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
torch::Tensor merge_sink_pool(torch::Tensor o, torch::Tensor ml, torch::Tensor q, torch::Tensor qh,
                              torch::Tensor block_table, torch::Tensor smap, torch::Tensor owner, torch::Tensor sk,
                              torch::Tensor sv, torch::Tensor k4, torch::Tensor v4, int64_t G, double score_mul,
                              double k4_mul, double v4_mul, double lshift) {
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
  auto out = torch::empty({B, HKV * G, D}, o.options().dtype(torch::kBFloat16));
  auto bf = [](torch::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()); };
  merge_sink_pool_kernel<<<B * HKV * G, D, 0, at::cuda::getCurrentCUDAStream()>>>(
      o.data_ptr<float>(), ml.data_ptr<float>(), bf(q), reinterpret_cast<const __half*>(qh.data_ptr()),
      block_table.data_ptr<int>(), (int)block_table.stride(0), smap.data_ptr<int>(), (int)smap.numel(),
      owner.numel() ? owner.data_ptr<int>() : nullptr, bf(sk), bf(sv), bf(k4), bf(v4), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), HKV,
      splits, GK, (int)G, (float)score_mul, (float)k4_mul, (float)v4_mul, (float)lshift);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "merge_sink_pool launch failed");
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("merge", &merge, "split-KV merge -> (B, HQ, 128) bf16");
  m.def("merge_sink", &merge_sink, "split-KV merge plus BF16 sink tokens -> (B, HQ, 128) bf16");
  m.def("merge_sink_pool", &merge_sink_pool, "split-KV merge with per-sequence BF16 sinks from pools (vLLM plugin)");
  m.def("prep_q_rot", &prep_q_rot, "q (B, HQ, 128) bf16/fp32 -> (B, HKV, 8, 128) fp16, optionally Hadamard-rotated");
}
