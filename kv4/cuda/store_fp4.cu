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
//
// 11-21: K8 (C1 in the plugin, template K8 with NV): K in decode_fp4_k8.py's layout (_k8_layout words: E2M1 bits
// placed so that a byte is an E4M3 value x 2^-6) with ONE E4M3 scale per 32 dims (group = 8 lanes, the NVFP4 scale
// rule s8 = e4m3(min(amax / 6 / ts, 448))), its 4 scale bytes per token at the MXFP4 K scale positions in the first
// 256 bytes of the 512-byte K scale region (the rest is unused). V is NVFP4 as before; the page stays 9216 bytes.
//
// 11-22: C2 (template C2 with K8; docs/c2_log.md). K as K8. V in decode_fp4_k8.cu's C2 layout: E2M1 with one E4M3
// scale per (channel, 32-token group of the page), s8 = e4m3(min(amax / 6 / ts, 448)) over the group's tokens, the
// NVFP4 rule along tokens (kv4.formats "nvfp4_t32" with a given tensor scale). Page: K codes 4096 | K scales 512 (256
// used) | V codes 4096 | V scales 512 (256 used: byte (wi * 8 + g) * 8 + mt * 4 + 2 gi + h8 for channel
// 64 mt + 16 wi + 8 h8 + g and group gi; bytes 256..511 unused). Still 9216 bytes.
// A group's scale depends on all its tokens, but decode appends one token at a time, so the V store keeps a running
// maximum: the warp of the FIRST token of a group in this batch ("leader") handles all of the group's tokens in the
// batch (consecutive slots), lane = 4 channels. If the group already holds u0 tokens (written by earlier calls, in
// order: vLLM appends), its scale becomes max(old, new) and, where it grew, the u0 old codes are re-encoded from their
// dequantized values (code x old scale) at the new scale; then the new tokens are encoded at the new scale. A group
// that is written whole in one call (prefill) is exactly nvfp4_t32; one that fills over several calls has the same
// final scale (max of the rounded maxima = rounding of the max) and differs only in the re-encoded old codes (a second
// rounding). Requires in-order appends per sequence; a token may not be rewritten after later tokens of its group.
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
// GLANES lanes per scale group: 4 (16 values, NVFP4) or 8 (32 values, K8's K)
template <int GLANES = 4>
__device__ __forceinline__ void quant4_nv(float (&x)[4], float ts, int (&code)[4], uint8_t& s8) {
  float m = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
#pragma unroll
  for (int off = 1; off < GLANES; off <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
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

// ---- 11-22 C2 V store (see the header). Bit positions of token position p (0..31) of a group inside its word: byte
// p % 4; slot L (p < 16): (m, e0, e1, s) at bits (2, 3, 4, 7); slot H (p >= 16): at (0, 1, 6, 5). The word of
// (channel c, group gi) holding p is lane t = (p % 16) / 4 of channel c's four words.
__device__ __forceinline__ uint32_t c2_place(int code, int p) {
  const uint32_t m = code & 1, e0 = (code >> 1) & 1, e1 = (code >> 2) & 1, s = (code >> 3) & 1;
  const uint32_t b = p < 16 ? (m << 2) | (e0 << 3) | (e1 << 4) | (s << 7) : m | (e0 << 1) | (e1 << 6) | (s << 5);
  return b << (8 * (p & 3));
}
__device__ __forceinline__ int c2_code(uint32_t w, int p) {
  const uint32_t b = (w >> (8 * (p & 3))) & 0xFFu;
  return p < 16 ? (int)(((b >> 2) & 1) | (((b >> 3) & 1) << 1) | (((b >> 4) & 1) << 2) | (((b >> 7) & 1) << 3))
                : (int)((b & 1) | (((b >> 1) & 1) << 1) | (((b >> 6) & 1) << 2) | (((b >> 5) & 1) << 3));
}
__device__ __forceinline__ float e2m1_val(int code) {
  const float grid[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  const float v = grid[code & 7];
  return code & 8 ? -v : v;
}

// One warp; lane = channels 4 lane .. 4 lane + 3. pg = the (page, head) bytes; n = this token's batch index, slot its
// slot (o = slot % 64). Returns at once unless this token is its group's first one in the batch.
__device__ __forceinline__ void store_v_c2(const __nv_bfloat16* __restrict__ value, const int64_t* __restrict__ slots,
                                           uint8_t* __restrict__ pg, int n, int N, int HKV, int h, int64_t slot,
                                           int o, int lane, float ts) {
  constexpr int VC = 4096 + 512, VS = 2 * 4096 + 512;
  const int u0 = o & 31, gi = o >> 5;
  if (u0 != 0 && n > 0 && slots[n - 1] == slot - 1) return;  // an earlier token of this batch leads the group
  int k = 1;                                                  // this batch's tokens of the group: n .. n + k - 1
  while (u0 + k < 32 && n + k < N && slots[n + k] == slot + k) ++k;
  const int c0 = 4 * lane, mt = c0 >> 6, wi = (c0 >> 4) & 3, h8 = (c0 >> 3) & 1, gb = c0 & 7;  // channel c0 + j: g = gb + j
  uint32_t* vw = reinterpret_cast<uint32_t*>(pg + VC);
  auto widx = [&](int j, int t) { return ((mt * 4 + wi) * 32 + 4 * (gb + j) + t) * 4 + 2 * gi + h8; };
  auto sidx = [&](int j) { return VS + (wi * 8 + gb + j) * 8 + mt * 4 + 2 * gi + h8; };
  auto load4 = [&](int i, float (&x)[4]) {  // the 4 channels of batch token n + i
    const __nv_bfloat16* src = value + ((size_t)(n + i) * HKV + h) * D + c0;
#pragma unroll
    for (int j = 0; j < 4; ++j) x[j] = __bfloat162float(src[j]);
  };
  // 1. new scale = max(old, e4m3 of the new tokens' amax / 6 / ts), per channel (positive E4M3 bytes order as values)
  float am[4] = {0.f, 0.f, 0.f, 0.f}, x[4];
  for (int i = 0; i < k; ++i) {
    load4(i, x);
#pragma unroll
    for (int j = 0; j < 4; ++j) am[j] = fmaxf(am[j], fabsf(x[j]));
  }
  uint8_t sold[4], snew[4];
  float so[4], sn[4];
  bool grow = false;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const __nv_fp8_e4m3 q(fminf(am[j] / 6.f / ts, 448.f));
    sold[j] = u0 ? pg[sidx(j)] : (uint8_t)0;
    snew[j] = (u0 && sold[j] > q.__x) ? sold[j] : q.__x;
    grow |= u0 && snew[j] != sold[j];
    __nv_fp8_e4m3 a, b;
    a.__x = sold[j], b.__x = snew[j];
    so[j] = (float)a * ts;
    sn[j] = fmaxf((float)b * ts, TINY);
  }
  // 2. the channel's 4 words of this group: old tokens' bits kept (positions < u0), the rest cleared
  uint32_t keep[4];
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    uint32_t km = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (4 * t + j < u0) km |= 0x9Cu << (8 * j);
      if (16 + 4 * t + j < u0) km |= 0x63u << (8 * j);
    }
    keep[t] = km;
  }
  uint32_t wd[4][4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int t = 0; t < 4; ++t) wd[j][t] = u0 ? vw[widx(j, t)] & keep[t] : 0u;
  // 3. where the scale grew, re-encode the old tokens at the new scale (from code x old scale)
  if (__any_sync(0xffffffffu, grow)) {
#pragma unroll
    for (int p = 0; p < 32; ++p) {
      if (p >= u0) break;
      const int t = (p & 15) >> 2;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (snew[j] == sold[j]) continue;
        const int c = e2m1_code(e2m1_val(c2_code(wd[j][t], p)) * so[j] / sn[j]);
        wd[j][t] = (wd[j][t] & ~c2_place(15, p)) | c2_place(c, p);
      }
    }
  }
  // 4. the new tokens at positions u0 .. u0 + k - 1
#pragma unroll
  for (int p = 0; p < 32; ++p) {
    if (p < u0 || p >= u0 + k) continue;
    const int t = (p & 15) >> 2;
    load4(p - u0, x);
#pragma unroll
    for (int j = 0; j < 4; ++j) wd[j][t] |= c2_place(e2m1_code(x[j] / sn[j]), p);
  }
  // 5. store (each word and scale byte belongs to one channel, i.e. one lane: no atomics)
#pragma unroll
  for (int j = 0; j < 4; ++j) {
#pragma unroll
    for (int t = 0; t < 4; ++t) vw[widx(j, t)] = wd[j][t];
    pg[sidx(j)] = snew[j];
  }
}

// 11-13: V word ownership (QUAD). A V word holds 2 dims of the 4 tokens {qb, qb+1, qb+8, qb+9} of a page,
// qb = 16 (o / 16) + 2 ((o % 8) / 2) (decode_fp4.py _v_layout), and dims d, d+8 (d % 16 < 8), i.e. lanes l and l ^ 2.
// When all 4 tokens of the quad are in this batch at contiguous indices (n + slot offset), the warp of the first one
// quantizes all 4 and stores the 64 words whole (no read, no atomics); the other 3 warps skip V. Otherwise every
// token writes its own bits with atomics as before. The bytes written are the same either way.
template <bool NV, bool QUAD, bool K8 = false, bool C2 = false>
__global__ void __launch_bounds__(128) store_fp4_kernel(
    const __nv_bfloat16* __restrict__ key, const __nv_bfloat16* __restrict__ value, const int64_t* __restrict__ slots,
    uint8_t* __restrict__ pages, const int* __restrict__ kword, const uint32_t* __restrict__ kbits,
    const int* __restrict__ vword, const uint32_t* __restrict__ vbits, const int* __restrict__ kspos,
    const int* __restrict__ vspos, const float* __restrict__ mu, int N, int HKV, bool rot, float c, float tsk,
    float tsv) {
  using L = Lay<NV>;
  static_assert(!K8 || NV, "K8 is an NVFP4 variant");
  static_assert(!C2 || K8, "C2 is a K8 variant");
  constexpr int NGK = K8 ? 4 : L::NG, GLK = K8 ? 8 : L::GL;  // K scale groups per token, lanes per K group
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
  if constexpr (NV) quant4_nv<GLK>(x, tsk, code, sk);
  else { int ek; quant4(x, c, code, ek); sk = (uint8_t)(ek + 127); }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int d = lane * 4 + j;
    atomicOr(&krow[w][kword[d]], place(code[j], kbits[d]));
  }
  __syncwarp();
  if (lane < 16) reinterpret_cast<uint32_t*>(pg)[o * 16 + lane] = krow[w][lane];
  if (lane % GLK == 0) pg[L::KS + kspos[o * NGK + lane / GLK]] = sk;

  // ---- V
  if constexpr (C2) {  // 11-22: token-axis scales, the group's first token in this batch writes the group
    store_v_c2(value, slots, pg, n, N, HKV, h, slot, o, lane, tsv);
    return;
  }
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

static void store_impl(torch::Tensor key, torch::Tensor value, torch::Tensor slots, torch::Tensor pages,
                       torch::Tensor kword, torch::Tensor kbits, torch::Tensor vword, torch::Tensor vbits,
                       torch::Tensor kspos, torch::Tensor vspos, torch::Tensor mu, bool rot, double c, bool nv,
                       double tsk, double tsv, bool k8, bool c2 = false) {
  TORCH_CHECK(key.is_cuda() && key.is_contiguous() && key.scalar_type() == torch::kBFloat16 && key.dim() == 3 &&
              key.size(2) == D);
  TORCH_CHECK(value.sizes() == key.sizes() && value.is_contiguous() && value.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(slots.is_contiguous() && slots.scalar_type() == torch::kInt64 && slots.numel() == key.size(0));
  const int N = key.size(0), HKV = key.size(1);
  TORCH_CHECK(pages.is_contiguous() && pages.scalar_type() == torch::kUInt8 && pages.size(1) == HKV &&
              pages.size(2) == (nv ? Lay<true>::PB : Lay<false>::PB), "pages (num_pages, HKV, 8704 MX / 9216 NV) uint8");
  TORCH_CHECK(!k8 || nv, "k8 needs nv");
  TORCH_CHECK(!c2 || k8, "c2 needs k8");
  // (the C2 V store computes its positions itself; vword / vbits / vspos are the C2 tables, used by the dequant)
  TORCH_CHECK(kspos.numel() == PAGE * (nv && !k8 ? 8 : 4) && vspos.numel() == (c2 ? 2 * D : PAGE * (nv ? 8 : 4)),
              "scale tables of the format");
  const float* mup = nullptr;
  if (mu.numel()) {
    TORCH_CHECK(mu.is_contiguous() && mu.scalar_type() == torch::kFloat32 && mu.numel() == HKV * D);
    mup = mu.data_ptr<float>();
  }
  if (N == 0) return;
  const int items = N * HKV;
  const bool quad = !getenv("KV4_STORE_V1");  // 11-13: KV4_STORE_V1=1 keeps the all-atomic V write
  auto kern = c2 ? store_fp4_kernel<true, false, true, true>
            : k8 ? (quad ? store_fp4_kernel<true, true, true> : store_fp4_kernel<true, false, true>)
            : nv ? (quad ? store_fp4_kernel<true, true> : store_fp4_kernel<true, false>)
                 : (quad ? store_fp4_kernel<false, true> : store_fp4_kernel<false, false>);
  kern<<<(items + 3) / 4, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(key.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(value.data_ptr()),
      slots.data_ptr<int64_t>(), pages.data_ptr<uint8_t>(), kword.data_ptr<int>(),
      reinterpret_cast<const uint32_t*>(kbits.data_ptr<int>()), vword.data_ptr<int>(),
      reinterpret_cast<const uint32_t*>(vbits.data_ptr<int>()), kspos.data_ptr<int>(), vspos.data_ptr<int>(), mup, N,
      HKV, rot, (float)c, (float)tsk, (float)tsv);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "store_fp4 launch failed");
}

void store_fp4(torch::Tensor key, torch::Tensor value, torch::Tensor slots, torch::Tensor pages, torch::Tensor kword,
               torch::Tensor kbits, torch::Tensor vword, torch::Tensor vbits, torch::Tensor kspos,
               torch::Tensor vspos, torch::Tensor mu, bool rot, double c, bool nv, double tsk, double tsv) {
  store_impl(key, value, slots, pages, kword, kbits, vword, vbits, kspos, vspos, mu, rot, c, nv, tsk, tsv, false);
}

// 11-21: K8 pages (see the header): tables from store_fp4.py tables(nv=True, k8=True)
void store_fp4_k8(torch::Tensor key, torch::Tensor value, torch::Tensor slots, torch::Tensor pages, torch::Tensor kword,
                  torch::Tensor kbits, torch::Tensor vword, torch::Tensor vbits, torch::Tensor kspos,
                  torch::Tensor vspos, torch::Tensor mu, bool rot, double tsk, double tsv) {
  store_impl(key, value, slots, pages, kword, kbits, vword, vbits, kspos, vspos, mu, rot, 0.0, true, tsk, tsv, true);
}

// 11-22: C2 pages (see the header): tables from store_fp4.py tables(nv=True, k8=True, c2=True)
void store_fp4_c2(torch::Tensor key, torch::Tensor value, torch::Tensor slots, torch::Tensor pages, torch::Tensor kword,
                  torch::Tensor kbits, torch::Tensor vword, torch::Tensor vbits, torch::Tensor kspos,
                  torch::Tensor vspos, torch::Tensor mu, bool rot, double tsk, double tsv) {
  store_impl(key, value, slots, pages, kword, kbits, vword, vbits, kspos, vspos, mu, rot, 0.0, true, tsk, tsv, true,
             true);
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

// 11-22 C2: V scale per (channel d, group o / 32) at vspos[(o / 32) * 128 + d] (store_fp4.py tables(c2=True))
template <bool NV, bool K8 = false, bool C2 = false>
__global__ void __launch_bounds__(256) dequant_fp4_kernel(
    const uint8_t* __restrict__ pages, const int* __restrict__ block_ids, int n_tokens, int HKV,
    const int* __restrict__ kword, const uint32_t* __restrict__ kbits, const int* __restrict__ vword,
    const uint32_t* __restrict__ vbits, const int* __restrict__ kspos, const int* __restrict__ vspos,
    __nv_bfloat16* __restrict__ ko, __nv_bfloat16* __restrict__ vo, bool rot, float tsk, float tsv) {
  using L = Lay<NV>;
  constexpr int GD = D / L::NG;  // dims per scale group (V; K too unless K8)
  constexpr int NGK = K8 ? 4 : L::NG, GDK = D / NGK;
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
    float x = (c & 8 ? -grid[c & 7] : grid[c & 7]) * scale_of<NV>(sp[L::KS + kspos[o * NGK + d / GDK]], tsk);
    if (rot) kf[o][d] = x;
    else ko[out] = __float2bfloat16(x);
    w = vw[vword[o * D + d]], b = vbits[o * D + d];
    c = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) c |= ((w >> ((b >> (5 * i)) & 31)) & 1) << i;
    const int vsi = C2 ? vspos[(o >> 5) * D + d] : vspos[o * L::NG + d / GD];
    x = (c & 8 ? -grid[c & 7] : grid[c & 7]) * scale_of<NV>(sp[L::VS + vsi], tsv);
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
static auto dq_kernel(bool nv, bool k8, bool c2 = false) {
  return c2 ? dequant_fp4_kernel<true, true, true> : k8 ? dequant_fp4_kernel<true, true>
       : nv ? dequant_fp4_kernel<true> : dequant_fp4_kernel<false>;
}

std::vector<torch::Tensor> dequant_fp4_impl(torch::Tensor pages, torch::Tensor block_ids, int64_t n_tokens,
                                            torch::Tensor kword, torch::Tensor kbits, torch::Tensor vword,
                                            torch::Tensor vbits, torch::Tensor kspos, torch::Tensor vspos, bool rot,
                                            bool nv, double tsk, double tsv, bool k8, bool c2 = false) {
  TORCH_CHECK(!c2 || vspos.numel() == 2 * D, "C2 V scale table (2 x 128)");
  TORCH_CHECK(pages.is_cuda() && pages.is_contiguous() && pages.scalar_type() == torch::kUInt8 && pages.dim() == 3 &&
              pages.size(2) == (nv ? Lay<true>::PB : Lay<false>::PB), "pages (num_pages, HKV, 8704 MX / 9216 NV) uint8");
  TORCH_CHECK(block_ids.is_cuda() && block_ids.is_contiguous() && block_ids.scalar_type() == torch::kInt);
  const int HKV = pages.size(1), np = (int)((n_tokens + PAGE - 1) / PAGE);
  TORCH_CHECK(block_ids.numel() >= np, "block_ids shorter than n_tokens needs");
  auto opt = pages.options().dtype(torch::kBFloat16);
  auto k = torch::empty({n_tokens, HKV, D}, opt), v = torch::empty({n_tokens, HKV, D}, opt);
  if (n_tokens == 0) return {k, v};
  auto kern = dq_kernel(nv, k8, c2);
  kern<<<dim3(np, HKV), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      pages.data_ptr<uint8_t>(), block_ids.data_ptr<int>(), (int)n_tokens, HKV, kword.data_ptr<int>(),
      reinterpret_cast<const uint32_t*>(kbits.data_ptr<int>()), vword.data_ptr<int>(),
      reinterpret_cast<const uint32_t*>(vbits.data_ptr<int>()), kspos.data_ptr<int>(), vspos.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(k.data_ptr()), reinterpret_cast<__nv_bfloat16*>(v.data_ptr()), rot, (float)tsk,
      (float)tsv);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "dequant_fp4 launch failed");
  return {k, v};
}

std::vector<torch::Tensor> dequant_fp4(torch::Tensor pages, torch::Tensor block_ids, int64_t n_tokens,
                                       torch::Tensor kword, torch::Tensor kbits, torch::Tensor vword,
                                       torch::Tensor vbits, torch::Tensor kspos, torch::Tensor vspos, bool rot, bool nv,
                                       double tsk, double tsv) {
  return dequant_fp4_impl(pages, block_ids, n_tokens, kword, kbits, vword, vbits, kspos, vspos, rot, nv, tsk, tsv,
                          false);
}

std::vector<torch::Tensor> dequant_fp4_k8(torch::Tensor pages, torch::Tensor block_ids, int64_t n_tokens,
                                          torch::Tensor kword, torch::Tensor kbits, torch::Tensor vword,
                                          torch::Tensor vbits, torch::Tensor kspos, torch::Tensor vspos, bool rot,
                                          double tsk, double tsv) {
  return dequant_fp4_impl(pages, block_ids, n_tokens, kword, kbits, vword, vbits, kspos, vspos, rot, true, tsk, tsv,
                          true);
}

std::vector<torch::Tensor> dequant_fp4_c2(torch::Tensor pages, torch::Tensor block_ids, int64_t n_tokens,
                                          torch::Tensor kword, torch::Tensor kbits, torch::Tensor vword,
                                          torch::Tensor vbits, torch::Tensor kspos, torch::Tensor vspos, bool rot,
                                          double tsk, double tsv) {
  return dequant_fp4_impl(pages, block_ids, n_tokens, kword, kbits, vword, vbits, kspos, vspos, rot, true, tsk, tsv,
                          true, true);
}

// 11-13: the same, written into caller-provided K / V (rows 0..n_tokens-1 of contiguous (>= n_tokens, HKV, 128) bf16
// tensors, e.g. the first rows of a buffer that holds the whole sequence), so the continuation prefill needs no cat.
static void dequant_into_impl(torch::Tensor pages, torch::Tensor block_ids, int64_t n_tokens, torch::Tensor kword,
                              torch::Tensor kbits, torch::Tensor vword, torch::Tensor vbits, torch::Tensor kspos,
                              torch::Tensor vspos, bool rot, bool nv, double tsk, double tsv, torch::Tensor k,
                              torch::Tensor v, bool k8, bool c2 = false) {
  TORCH_CHECK(!c2 || vspos.numel() == 2 * D, "C2 V scale table (2 x 128)");
  TORCH_CHECK(pages.is_cuda() && pages.is_contiguous() && pages.scalar_type() == torch::kUInt8 && pages.dim() == 3 &&
              pages.size(2) == (nv ? Lay<true>::PB : Lay<false>::PB), "pages (num_pages, HKV, 8704 MX / 9216 NV) uint8");
  TORCH_CHECK(block_ids.is_cuda() && block_ids.is_contiguous() && block_ids.scalar_type() == torch::kInt);
  const int HKV = pages.size(1), np = (int)((n_tokens + PAGE - 1) / PAGE);
  TORCH_CHECK(block_ids.numel() >= np, "block_ids shorter than n_tokens needs");
  for (auto* t : {&k, &v})
    TORCH_CHECK(t->is_contiguous() && t->scalar_type() == torch::kBFloat16 && t->dim() == 3 && t->size(0) >= n_tokens &&
                t->size(1) == HKV && t->size(2) == D, "out (>= n_tokens, HKV, 128) bf16 contiguous");
  if (n_tokens == 0) return;
  auto kern = dq_kernel(nv, k8, c2);
  kern<<<dim3(np, HKV), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      pages.data_ptr<uint8_t>(), block_ids.data_ptr<int>(), (int)n_tokens, HKV, kword.data_ptr<int>(),
      reinterpret_cast<const uint32_t*>(kbits.data_ptr<int>()), vword.data_ptr<int>(),
      reinterpret_cast<const uint32_t*>(vbits.data_ptr<int>()), kspos.data_ptr<int>(), vspos.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(k.data_ptr()), reinterpret_cast<__nv_bfloat16*>(v.data_ptr()), rot, (float)tsk,
      (float)tsv);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "dequant_fp4_into launch failed");
}

void dequant_fp4_into(torch::Tensor pages, torch::Tensor block_ids, int64_t n_tokens, torch::Tensor kword,
                      torch::Tensor kbits, torch::Tensor vword, torch::Tensor vbits, torch::Tensor kspos,
                      torch::Tensor vspos, bool rot, bool nv, double tsk, double tsv, torch::Tensor k, torch::Tensor v) {
  dequant_into_impl(pages, block_ids, n_tokens, kword, kbits, vword, vbits, kspos, vspos, rot, nv, tsk, tsv, k, v,
                    false);
}

void dequant_fp4_k8_into(torch::Tensor pages, torch::Tensor block_ids, int64_t n_tokens, torch::Tensor kword,
                         torch::Tensor kbits, torch::Tensor vword, torch::Tensor vbits, torch::Tensor kspos,
                         torch::Tensor vspos, bool rot, double tsk, double tsv, torch::Tensor k, torch::Tensor v) {
  dequant_into_impl(pages, block_ids, n_tokens, kword, kbits, vword, vbits, kspos, vspos, rot, true, tsk, tsv, k, v,
                    true);
}

void dequant_fp4_c2_into(torch::Tensor pages, torch::Tensor block_ids, int64_t n_tokens, torch::Tensor kword,
                         torch::Tensor kbits, torch::Tensor vword, torch::Tensor vbits, torch::Tensor kspos,
                         torch::Tensor vspos, bool rot, double tsk, double tsv, torch::Tensor k, torch::Tensor v) {
  dequant_into_impl(pages, block_ids, n_tokens, kword, kbits, vword, vbits, kspos, vspos, rot, true, tsk, tsv, k, v,
                    true, true);
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
  m.def("store_fp4_k8", &store_fp4_k8, "store_fp4 for K8 pages (K: k8 layout, one E4M3 scale per 32 dims; V: NVFP4)");
  m.def("dequant_fp4_k8", &dequant_fp4_k8, "dequant_fp4 for K8 pages");
  m.def("dequant_fp4_k8_into", &dequant_fp4_k8_into, "dequant_fp4_into for K8 pages");
  m.def("store_fp4_c2", &store_fp4_c2, "store_fp4 for C2 pages (K as K8; V: E4M3 scale per channel and 32 tokens)");
  m.def("dequant_fp4_c2", &dequant_fp4_c2, "dequant_fp4 for C2 pages");
  m.def("dequant_fp4_c2_into", &dequant_fp4_c2_into, "dequant_fp4_into for C2 pages");
}
