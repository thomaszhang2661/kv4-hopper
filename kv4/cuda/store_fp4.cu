// Quantize-on-write for the paged MXFP4 decode kernel (11-12): K/V of new tokens (N, HKV, 128) -> their slots in a
// pool of 64-token pages, directly in the decode kernel's layout (decode_fp4.cu, pack_pages in decode_fp4.py).
//
// Page bytes per (page, kv head), MXFP4: K codes 4096 | K scales 256 | V codes 4096 | V scales 256 = 8704, the same
// size as vLLM UltraQuant's (64 tokens x 136 bytes), so a vLLM block can be reinterpreted as one of our pages.
// Every page has the same local layout (checked in store_fp4.py), so the position of a value depends only on its
// token's offset in the page (0..63) and its dim; the tables below come from decode_fp4.py's layout functions.
//
// One warp per (token, kv head); lane holds dims 4 lane .. 4 lane + 3 (a group of 32 dims = 8 lanes).
//   K: optional per-channel mean removal (mu, fp32 (HKV, 128)), optional Hadamard-128 (as formats.rotate), then
//      MXFP4 with scale 2^e per 32: OCP e = floor(log2 amax) - 2 (c <= 0) or e = round(log2(c amax)) (c > 0).
//      The token owns its 16-word K row, built in shared memory and stored whole; its 4 scale bytes are stored.
//   V: the same without mean or rotation. A V word holds 4 tokens' codes, so the token's bits are cleared with
//      atomicAnd and set with atomicOr (other tokens of the page may be written concurrently).
// E2M1: nearest value, ties to the even mantissa, saturating at 6; -0 is stored as +0 (formats.e2m1_encode).
//
// 11-12 night: NVFP4 too (template NV). Page bytes per (page, kv head): K codes 4096 | K scales 512 | V codes 4096 |
// V scales 512 = 9216 (64 tokens x 144 bytes). One E4M3 scale per 16 values (4 lanes) on top of a per-layer fp32
// tensor scale ts (fixed, calibrated offline): s8 = e4m3(min(amax / 6 / ts, 448)), codes = e2m1(x / (s8 ts)), as
// formats.quantize("nvfp4") with a given tensor scale.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {
constexpr int D = 128, PAGE = 64, PB = 8704, KS_OFF = 4096, VC_OFF = 4352, VS_OFF = 8448;
template <bool NV> struct Lay {  // page layout per format
  static constexpr int SCB = NV ? 512 : 256;            // scale bytes per (page, head) for K (same for V)
  static constexpr int PB = 2 * (4096 + SCB), KS = 4096, VC = 4096 + SCB, VS = 2 * 4096 + SCB;
  static constexpr int NG = NV ? 8 : 4;                 // scale groups per token
  static constexpr int GL = NV ? 4 : 8;                 // lanes per group (4 values per lane)
};
constexpr float TINY = 1.17549435e-38f;  // torch.finfo(float32).tiny

__device__ __forceinline__ int e2m1_code(float x) {
  const float a = fminf(fabsf(x), 6.f);
  const float mids[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.f};
  int lo = 0, hi = 0;
#pragma unroll
  for (int i = 0; i < 7; ++i) { lo += mids[i] < a; hi += mids[i] <= a; }
  const int idx = (lo & 1) ? hi : lo;  // grid index parity is the mantissa bit: ties go to the even one
  return idx | ((x < 0.f && idx > 0) ? 8 : 0);
}

__device__ __forceinline__ int scale_exp(float amax, float c) {
  float e;
  if (c <= 0.f) e = floorf(log2f(fmaxf(amax, TINY))) - 2.f;  // OCP MX
  else e = rintf(log2f(fmaxf(c * amax, TINY)));               // UltraQuant family (rint = round half to even)
  return (int)fminf(fmaxf(e, -127.f), 127.f);
}

// 4 values of one lane -> codes and the group's exponent (group = 8 lanes)
__device__ __forceinline__ void quant4(float (&x)[4], float c, int (&code)[4], int& e) {
  float m = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
#pragma unroll
  for (int off = 1; off < 8; off <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
  e = scale_exp(m, c);
  const float inv = exp2f((float)-e);
#pragma unroll
  for (int j = 0; j < 4; ++j) code[j] = e2m1_code(x[j] * inv);
}

// NVFP4: 4 values of one lane -> codes and the group's E4M3 scale byte (group = 4 lanes = 16 values)
__device__ __forceinline__ void quant4_nv(float (&x)[4], float ts, int (&code)[4], uint8_t& s8) {
  float m = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
#pragma unroll
  for (int off = 1; off < 4; off <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
  const __nv_fp8_e4m3 q(fminf(m / 6.f / ts, 448.f));
  s8 = q.__x;
  const float sc = fmaxf((float)q * ts, TINY);
#pragma unroll
  for (int j = 0; j < 4; ++j) code[j] = e2m1_code(x[j] / sc);
}

__device__ __forceinline__ uint32_t place(int code, uint32_t bits) {  // bits: positions of (m, e0, e1, s), 5 bits each
  uint32_t w = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) w |= (uint32_t)((code >> i) & 1) << ((bits >> (5 * i)) & 31);
  return w;
}

__device__ __forceinline__ uint32_t mask_of(uint32_t bits) {
  uint32_t w = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) w |= 1u << ((bits >> (5 * i)) & 31);
  return w;
}

// 11-13: V word ownership (QUAD). A V word holds 2 dims of the 4 tokens {qb, qb+1, qb+8, qb+9} of a page,
// qb = 16 (o / 16) + 2 ((o % 8) / 2) (decode_fp4.py _v_layout), and dims d, d+8 (d % 16 < 8), i.e. lanes l and l ^ 2.
// When all 4 tokens of the quad are in this batch at contiguous indices (n + slot offset), the warp of the first one
// quantizes all 4 and stores the 64 words whole (no read, no atomics); the other 3 warps skip V. Otherwise every
// token writes its own bits with atomics as before. The bytes written are the same either way.
template <bool NV, bool QUAD>
__global__ void __launch_bounds__(128) store_fp4_kernel(
    const __nv_bfloat16* __restrict__ key, const __nv_bfloat16* __restrict__ value, const int64_t* __restrict__ slots,
    uint8_t* __restrict__ pages, const int* __restrict__ kword, const uint32_t* __restrict__ kbits,
    const int* __restrict__ vword, const uint32_t* __restrict__ vbits, const int* __restrict__ kspos,
    const int* __restrict__ vspos, const float* __restrict__ mu, int N, int HKV, bool rot, float c, float tsk,
    float tsv) {
  using L = Lay<NV>;
  __shared__ uint32_t krow[4][16];
  const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int item = blockIdx.x * 4 + w;  // (token, head)
  const bool live = item < N * HKV;
  const int n = live ? item / HKV : 0, h = item % HKV;
  const int64_t slot = live ? slots[n] : -1;
  if (lane < 16) krow[w][lane] = 0;
  __syncwarp();
  if (slot < 0) return;  // padding token (vLLM: slot -1); whole warp leaves together
  const int64_t page = slot / PAGE;
  const int o = (int)(slot % PAGE);
  uint8_t* pg = pages + ((size_t)page * HKV + h) * L::PB;
  const size_t src = ((size_t)n * HKV + h) * D + lane * 4;

  // ---- K
  float x[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) x[j] = __bfloat162float(key[src + j]) - (mu ? mu[h * D + lane * 4 + j] : 0.f);
  if (rot) {  // Sylvester Hadamard / sqrt(128), the same butterflies as merge.cu prep_q_rot
    float a = x[0], b = x[1], cc = x[2], e = x[3];
    x[0] = a + b; x[1] = a - b; x[2] = cc + e; x[3] = cc - e;
    a = x[0]; b = x[1]; cc = x[2]; e = x[3];
    x[0] = a + cc; x[1] = b + e; x[2] = a - cc; x[3] = b - e;
#pragma unroll
    for (int s = 1; s < 32; s <<= 1) {
      const bool upper = lane & s;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float p = __shfl_xor_sync(0xffffffffu, x[j], s);
        x[j] = upper ? p - x[j] : x[j] + p;
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) x[j] *= 0.08838834764831845f;
  }
  int code[4];
  uint8_t sk;
  if constexpr (NV) quant4_nv(x, tsk, code, sk);
  else { int ek; quant4(x, c, code, ek); sk = (uint8_t)(ek + 127); }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int d = lane * 4 + j;
    atomicOr(&krow[w][kword[d]], place(code[j], kbits[d]));
  }
  __syncwarp();
  if (lane < 16) reinterpret_cast<uint32_t*>(pg)[o * 16 + lane] = krow[w][lane];
  if (lane % L::GL == 0) pg[L::KS + kspos[o * L::NG + lane / L::GL]] = sk;

  // ---- V
  if constexpr (QUAD) {
    const int qb = (o & ~15) | (o & 6), me = (o & 1) | ((o >> 2) & 2);
    const int64_t pbase = slot - o;
    bool full = true;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int mo = qb + (i & 1) + 8 * (i >> 1);
      const int idx = n + (mo - o);
      full = full && idx >= 0 && idx < N && slots[idx] == pbase + mo;
    }
    if (full) {
      if (me != 0) return;  // the quad's first token's warp writes it
      uint32_t acc[4] = {0u, 0u, 0u, 0u};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int mo = qb + (i & 1) + 8 * (i >> 1);
        const size_t s2 = ((size_t)(n + (mo - o)) * HKV + h) * D + lane * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) x[j] = __bfloat162float(value[s2 + j]);
        uint8_t sv;
        if constexpr (NV) quant4_nv(x, tsv, code, sv);
        else { int ev; quant4(x, c, code, ev); sv = (uint8_t)(ev + 127); }
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] |= place(code[j], vbits[mo * D + lane * 4 + j]);
        if (lane % L::GL == 0) pg[L::VS + vspos[mo * L::NG + lane / L::GL]] = sv;
      }
      uint32_t* vw = reinterpret_cast<uint32_t*>(pg + L::VC);
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[j] |= __shfl_xor_sync(0xffffffffu, acc[j], 2);
      if (!(lane & 2)) {
#pragma unroll
        for (int j = 0; j < 4; ++j) vw[vword[qb * D + lane * 4 + j]] = acc[j];
      }
      return;
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) x[j] = __bfloat162float(value[src + j]);
  uint8_t sv;
  if constexpr (NV) quant4_nv(x, tsv, code, sv);
  else { int ev; quant4(x, c, code, ev); sv = (uint8_t)(ev + 127); }
  uint32_t* vw = reinterpret_cast<uint32_t*>(pg + L::VC);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int t = o * D + lane * 4 + j;
    const uint32_t b = vbits[t];
    atomicAnd(&vw[vword[t]], ~mask_of(b));
    atomicOr(&vw[vword[t]], place(code[j], b));
  }
  if (lane % L::GL == 0) pg[L::VS + vspos[o * L::NG + lane / L::GL]] = sv;
}
}  // namespace

void store_fp4(torch::Tensor key, torch::Tensor value, torch::Tensor slots, torch::Tensor pages, torch::Tensor kword,
               torch::Tensor kbits, torch::Tensor vword, torch::Tensor vbits, torch::Tensor kspos,
               torch::Tensor vspos, torch::Tensor mu, bool rot, double c, bool nv, double tsk, double tsv) {
  TORCH_CHECK(key.is_cuda() && key.is_contiguous() && key.scalar_type() == torch::kBFloat16 && key.dim() == 3 &&
              key.size(2) == D);
  TORCH_CHECK(value.sizes() == key.sizes() && value.is_contiguous() && value.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(slots.is_contiguous() && slots.scalar_type() == torch::kInt64 && slots.numel() == key.size(0));
  const int N = key.size(0), HKV = key.size(1);
  TORCH_CHECK(pages.is_contiguous() && pages.scalar_type() == torch::kUInt8 && pages.size(1) == HKV &&
              pages.size(2) == (nv ? Lay<true>::PB : Lay<false>::PB), "pages (num_pages, HKV, 8704 MX / 9216 NV) uint8");
  TORCH_CHECK(kspos.numel() == PAGE * (nv ? 8 : 4) && vspos.numel() == kspos.numel(), "scale tables of the format");
  const float* mup = nullptr;
  if (mu.numel()) {
    TORCH_CHECK(mu.is_contiguous() && mu.scalar_type() == torch::kFloat32 && mu.numel() == HKV * D);
    mup = mu.data_ptr<float>();
  }
  if (N == 0) return;
  const int items = N * HKV;
  const bool quad = !getenv("KV4_STORE_V1");  // 11-13: KV4_STORE_V1=1 keeps the all-atomic V write
  auto kern = nv ? (quad ? store_fp4_kernel<true, true> : store_fp4_kernel<true, false>)
                 : (quad ? store_fp4_kernel<false, true> : store_fp4_kernel<false, false>);
  kern<<<(items + 3) / 4, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(key.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(value.data_ptr()),
      slots.data_ptr<int64_t>(), pages.data_ptr<uint8_t>(), kword.data_ptr<int>(),
      reinterpret_cast<const uint32_t*>(kbits.data_ptr<int>()), vword.data_ptr<int>(),
      reinterpret_cast<const uint32_t*>(vbits.data_ptr<int>()), kspos.data_ptr<int>(), vspos.data_ptr<int>(), mup, N,
      HKV, rot, (float)c, (float)tsk, (float)tsv);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "store_fp4 launch failed");
}

void store_mxfp4(torch::Tensor key, torch::Tensor value, torch::Tensor slots, torch::Tensor pages, torch::Tensor kword,
                 torch::Tensor kbits, torch::Tensor vword, torch::Tensor vbits, torch::Tensor kspos,
                 torch::Tensor vspos, torch::Tensor mu, bool rot, double c) {
  store_fp4(key, value, slots, pages, kword, kbits, vword, vbits, kspos, vspos, mu, rot, c, false, 1.0, 1.0);
}

namespace {
// 11-12: inverse of the writer for vLLM's continuation prefill (dequant_pages in store_fp4.py is the torch reference).
// One block per (page of the sequence, kv head): the page's 8704 bytes go to shared memory, then each thread writes
// dequantized K and V values (codes x 2^(e - 127)) for its (token, dim) pairs to out (n_tokens, HKV, 128) bf16, which
// is exact (E2M1 x a power of two). With rot, K is rotated back in shared memory (fp32 butterflies) before the cast.
template <bool NV>
__device__ __forceinline__ float scale_of(uint8_t b, float ts) {
  if constexpr (NV) { __nv_fp8_e4m3 q; q.__x = b; return (float)q * ts; }
  else return exp2f((float)b - 127.f);
}

template <bool NV>
__global__ void __launch_bounds__(256) dequant_fp4_kernel(
    const uint8_t* __restrict__ pages, const int* __restrict__ block_ids, int n_tokens, int HKV,
    const int* __restrict__ kword, const uint32_t* __restrict__ kbits, const int* __restrict__ vword,
    const uint32_t* __restrict__ vbits, const int* __restrict__ kspos, const int* __restrict__ vspos,
    __nv_bfloat16* __restrict__ ko, __nv_bfloat16* __restrict__ vo, bool rot, float tsk, float tsv) {
  using L = Lay<NV>;
  constexpr int GD = D / L::NG;  // dims per scale group
  __shared__ __align__(16) uint8_t sp[L::PB];
  __shared__ float kf[PAGE][D + 1];  // K rows for the in-place inverse Hadamard (rot)
  const int pi = blockIdx.x, h = blockIdx.y;
  const uint8_t* src = pages + ((size_t)block_ids[pi] * HKV + h) * L::PB;
  for (int i = threadIdx.x; i < L::PB / 16; i += blockDim.x)
    reinterpret_cast<uint4*>(sp)[i] = reinterpret_cast<const uint4*>(src)[i];
  __syncthreads();
  const float grid[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  const uint32_t* kw = reinterpret_cast<const uint32_t*>(sp);
  const uint32_t* vw = reinterpret_cast<const uint32_t*>(sp + L::VC);
  for (int e = threadIdx.x; e < PAGE * D; e += blockDim.x) {
    const int o = e / D, d = e % D, t = pi * PAGE + o;
    if (t >= n_tokens) break;  // e grows with o
    const size_t out = ((size_t)t * HKV + h) * D + d;
    uint32_t w = kw[o * 16 + kword[d]], b = kbits[d];
    int c = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) c |= ((w >> ((b >> (5 * i)) & 31)) & 1) << i;
    float x = (c & 8 ? -grid[c & 7] : grid[c & 7]) * scale_of<NV>(sp[L::KS + kspos[o * L::NG + d / GD]], tsk);
    if (rot) kf[o][d] = x;
    else ko[out] = __float2bfloat16(x);
    w = vw[vword[o * D + d]], b = vbits[o * D + d];
    c = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) c |= ((w >> ((b >> (5 * i)) & 31)) & 1) << i;
    x = (c & 8 ? -grid[c & 7] : grid[c & 7]) * scale_of<NV>(sp[L::VS + vspos[o * L::NG + d / GD]], tsv);
    vo[out] = __float2bfloat16(x);
  }
  if (!rot) return;
  // 11-12: inverse Hadamard-128 (Sylvester, its own inverse) as 7 butterfly stages in fp32, then / sqrt(128);
  // formats.rotate does the same product as an fp32 matmul, so results agree up to fp32 rounding before the bf16 cast
  const int rows = min(PAGE, n_tokens - pi * PAGE);
  for (int h2 = 1; h2 < D; h2 <<= 1) {
    __syncthreads();
    for (int e = threadIdx.x; e < PAGE * (D / 2); e += blockDim.x) {
      const int o = e / (D / 2), j = e % (D / 2);
      if (o >= rows) break;
      const int a = (j / h2) * 2 * h2 + j % h2, b = a + h2;
      const float u = kf[o][a], w = kf[o][b];
      kf[o][a] = u + w;
      kf[o][b] = u - w;
    }
  }
  __syncthreads();
  for (int e = threadIdx.x; e < rows * D; e += blockDim.x) {
    const int o = e / D, d = e % D;
    ko[((size_t)(pi * PAGE + o) * HKV + h) * D + d] = __float2bfloat16(kf[o][d] * 0.08838834764831845f);
  }
}
}  // namespace

// pages (num_pages, HKV, 8704) uint8; block_ids (n_pages) int32, the sequence's pages in order -> K, V
// (n_tokens, HKV, 128) bf16 (K rotated back if rot, mean not restored).
std::vector<torch::Tensor> dequant_fp4(torch::Tensor pages, torch::Tensor block_ids, int64_t n_tokens,
                                       torch::Tensor kword, torch::Tensor kbits, torch::Tensor vword,
                                       torch::Tensor vbits, torch::Tensor kspos, torch::Tensor vspos, bool rot, bool nv,
                                       double tsk, double tsv) {
  TORCH_CHECK(pages.is_cuda() && pages.is_contiguous() && pages.scalar_type() == torch::kUInt8 && pages.dim() == 3 &&
              pages.size(2) == (nv ? Lay<true>::PB : Lay<false>::PB), "pages (num_pages, HKV, 8704 MX / 9216 NV) uint8");
  TORCH_CHECK(block_ids.is_cuda() && block_ids.is_contiguous() && block_ids.scalar_type() == torch::kInt);
  const int HKV = pages.size(1), np = (int)((n_tokens + PAGE - 1) / PAGE);
  TORCH_CHECK(block_ids.numel() >= np, "block_ids shorter than n_tokens needs");
  auto opt = pages.options().dtype(torch::kBFloat16);
  auto k = torch::empty({n_tokens, HKV, D}, opt), v = torch::empty({n_tokens, HKV, D}, opt);
  if (n_tokens == 0) return {k, v};
  auto kern = nv ? dequant_fp4_kernel<true> : dequant_fp4_kernel<false>;
  kern<<<dim3(np, HKV), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      pages.data_ptr<uint8_t>(), block_ids.data_ptr<int>(), (int)n_tokens, HKV, kword.data_ptr<int>(),
      reinterpret_cast<const uint32_t*>(kbits.data_ptr<int>()), vword.data_ptr<int>(),
      reinterpret_cast<const uint32_t*>(vbits.data_ptr<int>()), kspos.data_ptr<int>(), vspos.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(k.data_ptr()), reinterpret_cast<__nv_bfloat16*>(v.data_ptr()), rot, (float)tsk,
      (float)tsv);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "dequant_fp4 launch failed");
  return {k, v};
}

// 11-13: the same, written into caller-provided K / V (rows 0..n_tokens-1 of contiguous (>= n_tokens, HKV, 128) bf16
// tensors, e.g. the first rows of a buffer that holds the whole sequence), so the continuation prefill needs no cat.
void dequant_fp4_into(torch::Tensor pages, torch::Tensor block_ids, int64_t n_tokens, torch::Tensor kword,
                      torch::Tensor kbits, torch::Tensor vword, torch::Tensor vbits, torch::Tensor kspos,
                      torch::Tensor vspos, bool rot, bool nv, double tsk, double tsv, torch::Tensor k, torch::Tensor v) {
  TORCH_CHECK(pages.is_cuda() && pages.is_contiguous() && pages.scalar_type() == torch::kUInt8 && pages.dim() == 3 &&
              pages.size(2) == (nv ? Lay<true>::PB : Lay<false>::PB), "pages (num_pages, HKV, 8704 MX / 9216 NV) uint8");
  TORCH_CHECK(block_ids.is_cuda() && block_ids.is_contiguous() && block_ids.scalar_type() == torch::kInt);
  const int HKV = pages.size(1), np = (int)((n_tokens + PAGE - 1) / PAGE);
  TORCH_CHECK(block_ids.numel() >= np, "block_ids shorter than n_tokens needs");
  for (auto* t : {&k, &v})
    TORCH_CHECK(t->is_contiguous() && t->scalar_type() == torch::kBFloat16 && t->dim() == 3 && t->size(0) >= n_tokens &&
                t->size(1) == HKV && t->size(2) == D, "out (>= n_tokens, HKV, 128) bf16 contiguous");
  if (n_tokens == 0) return;
  auto kern = nv ? dequant_fp4_kernel<true> : dequant_fp4_kernel<false>;
  kern<<<dim3(np, HKV), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      pages.data_ptr<uint8_t>(), block_ids.data_ptr<int>(), (int)n_tokens, HKV, kword.data_ptr<int>(),
      reinterpret_cast<const uint32_t*>(kbits.data_ptr<int>()), vword.data_ptr<int>(),
      reinterpret_cast<const uint32_t*>(vbits.data_ptr<int>()), kspos.data_ptr<int>(), vspos.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(k.data_ptr()), reinterpret_cast<__nv_bfloat16*>(v.data_ptr()), rot, (float)tsk,
      (float)tsv);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "dequant_fp4_into launch failed");
}

std::vector<torch::Tensor> dequant_mxfp4(torch::Tensor pages, torch::Tensor block_ids, int64_t n_tokens,
                                         torch::Tensor kword, torch::Tensor kbits, torch::Tensor vword,
                                         torch::Tensor vbits, torch::Tensor kspos, torch::Tensor vspos, bool rot) {
  return dequant_fp4(pages, block_ids, n_tokens, kword, kbits, vword, vbits, kspos, vspos, rot, false, 1.0, 1.0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("store_mxfp4", &store_mxfp4, "quantize-on-write of new K/V tokens into 64-token MXFP4 pages (decode layout)");
  m.def("dequant_mxfp4", &dequant_mxfp4, "pages of one sequence -> K (rotated basis), V bf16 (continuation prefill)");
  m.def("store_fp4", &store_fp4, "store_mxfp4 for MXFP4 (nv=False) or NVFP4 (nv=True, tensor scales tsk, tsv)");
  m.def("dequant_fp4", &dequant_fp4, "dequant_mxfp4 for MXFP4 or NVFP4");
  m.def("dequant_fp4_into", &dequant_fp4_into, "dequant_fp4 into caller-provided K / V rows");
}
