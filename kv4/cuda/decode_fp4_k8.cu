// rebuild tag: k8 v1 (11-21)
// Change C1 of docs/notes/xqa_vs_decode_fp4.md (11-21 prototype): decode_fp4.cu with the K side on FP8 tensor cores,
// as TRT-LLM's XQA does for an FP8 cache. Separate extension (kv4_decode_fp4_k8); decode_fp4.cu is untouched.
// Dense cache only, G = 4 query heads per kv head, NST = 2, no ILP / DIAG / paging. The V side is decode_fp4.cu's
// NVFP4 path unchanged.
//
// K format: E2M1 with one E4M3 scale per 32 dims per token (kv4.formats "nvfp4_g32"), FP32 tensor scale. One FP8
// wgmma k-step covers 32 dims, so the scale is constant over a k-step for each token row and can be applied to that
// k-step's partial accumulator after the MMA (one partial accumulator per k-step).
// E2M1 -> E4M3 by bit placement: code (s e1 e0 m) at E4M3 bits (7, 4, 3, 2) is the value * 2^-6 (exponent field
// 0 e1 e0... with E4M3's bias 7 against E2M1's 1; 0.5 becomes the E4M3 subnormal 2^-7, exact). The 2^6 goes into
// score_mul.
// wgmma m64n8k32 .e4m3 with A from registers: thread (lane 4g + t of warp w) holds per k-step R0 = (row g,
// dims 4t..4t+3), R1 = (row g + 8, same), R2 = (row g, dims 16 + 4t..+3), R3 = (row g + 8, same); one byte per dim,
// the lowest dim in the lowest byte.
// Write-time layout (decode_fp4_k8.py: pack_k8): per token row 16 words; lane t reads words 4t..4t+3, word i is
// k-step i (dims 32i..32i+31); byte j of the word holds
//   slot L: dim 32i + 4t + j,      (s, e1, e0, m) at bits (7, 4, 3, 2) -> R0 / R1 = w & 0x9C9C9C9C
//   slot H: dim 32i + 16 + 4t + j, (s, e1, e0, m) at bits (5, 6, 1, 0) -> R2 / R3 = ((w << 2) & 0x8C8C8C8C) |
//                                                                                    ((w >> 2) & 0x10101010)
// so 8 values cost 1 + 4 instructions (decode_fp4.cu: about 9, plus 4 HMUL2 and 1 cvt for the scales).
// (No layout of two 4-bit codes per byte gives both slots with one mask each: the E4M3 positions {7, 4, 3, 2} and
// their complement are not shifts of each other mod 8.)
// K scales per stage (1 KB): decode_fp4.cu's MXFP4 order, [wg 2][sub 2][warp 4][g 8][k 2][(grp 2k, row g),
// (grp 2k+1, row g), (grp 2k, row g+8), (grp 2k+1, row g+8)], bytes are E4M3 here.
// q: E4M3 (B, HKV, 8, 128) with one scale per tensor (decode_fp4_k8.py: prep_q8; folded into score_mul), rows 4..7
// zero; in shared memory K-major in 8 x 16-byte core matrices: core matrix kc (dims 16kc..16kc+15) of the 8 rows at
// byte 128 kc, so one k-step (two core matrices) advances the descriptor by 256 bytes.
#include <type_traits>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "common.cuh"

namespace {
using namespace kv4cu;

constexpr float P_SHIFT = 10.f;
constexpr int D = 128;
constexpr int ROW_BYTES = 64;
constexpr int STAGE = 256;
constexpr int HALF = 128;
constexpr int NCWARPS = 8;
constexpr int NTHREADS = NCWARPS * 32 + 32;
constexpr int NPAD = 8;
constexpr int G = 4;
constexpr int NST = 2;
constexpr int CODE_BYTES = STAGE * ROW_BYTES;  // 16 KB
constexpr int KSC_BYTES = STAGE * 4;           // E4M3 per 32 dims
constexpr int VSC_BYTES = STAGE * 8;           // E4M3 per 16 values (NVFP4 V)

constexpr size_t smem_bytes() {
  return 2 * NST * CODE_BYTES + NST * (KSC_BYTES + VSC_BYTES) + NPAD * D + 2 * NPAD * HALF * 2 + 2 * (16 * G) * 4 +
         2 * NST * 8;
}

__device__ __forceinline__ uint32_t fA(uint32_t w) { return w & 0x8E008E00u; }
__device__ __forceinline__ uint32_t fB(uint32_t w) { return (w << 3) & 0x8E008E00u; }
__device__ __forceinline__ uint32_t fC(uint32_t w) { return ((w << 9) & 0x0E000E00u) | ((w << 2) & 0x80008000u); }
__device__ __forceinline__ uint32_t fD(uint32_t w) { return ((w << 6) & 0x0E000E00u) | ((w << 1) & 0x80008000u); }
__device__ __forceinline__ uint32_t k8L(uint32_t w) { return w & 0x9C9C9C9Cu; }
__device__ __forceinline__ uint32_t k8H(uint32_t w) { return ((w << 2) & 0x8C8C8C8Cu) | ((w >> 2) & 0x10101010u); }

__device__ __forceinline__ uint32_t e4m3x2_f16x2(uint32_t lo16) {
  uint32_t r;
  asm("{\n .reg .b16 x;\n cvt.u16.u32 x, %1;\n cvt.rn.f16x2.e4m3x2 %0, x;\n}" : "=r"(r) : "r"(lo16));
  return r;
}
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
__device__ __forceinline__ float2 h2f2(uint32_t x) {
  __half2 h = *reinterpret_cast<__half2*>(&x);
  return __half22float2(h);
}

// D(64x8, fp32) = A(64x32, e4m3, registers) * B(32x8, e4m3, shared memory); accumulator overwritten (scale-d false)
__device__ __forceinline__ void mma_rs8(float (&d)[4], const uint32_t (&a)[4], uint64_t desc) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %9, 0;\n"
      " wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1;\n}"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(0));
}

__global__ void __launch_bounds__(NTHREADS, 2) decode_k8_kernel(const uint8_t* __restrict__ kp,   // (B,HKV,S,64)
                                                                const uint8_t* __restrict__ ksc,  // (B,HKV,S,4)
                                                                const uint8_t* __restrict__ vp,
                                                                const uint8_t* __restrict__ vsc,  // (B,HKV,S,8)
                                                                const uint8_t* __restrict__ q8,   // (B,HKV,8,128)
                                                                float* __restrict__ out_o, float* __restrict__ out_ml,
                                                                int S, int HKV, float score_mul, float out_mul) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* s_k = smem;
  uint8_t* s_v = s_k + NST * CODE_BYTES;
  uint8_t* s_ks = s_v + NST * CODE_BYTES;
  uint8_t* s_vs = s_ks + NST * KSC_BYTES;
  uint8_t* s_q = s_vs + NST * VSC_BYTES;
  __half* s_p = reinterpret_cast<__half*>(s_q + NPAD * D);
  float* s_wmax = reinterpret_cast<float*>(s_p + 2 * NPAD * HALF);
  float* s_lc = s_wmax + 16 * G;
  uint64_t* full = reinterpret_cast<uint64_t*>(s_lc + 16 * G);
  uint64_t* empty = full + NST;

  const int split = blockIdx.x, h = blockIdx.y, b = blockIdx.z, splits = gridDim.x;
  const int tok_per = S / splits, tok0 = split * tok_per;
  const int niter = tok_per / STAGE;
  const size_t bh = (size_t)b * HKV + h;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;

  for (int i = tid; i < NPAD * D; i += NTHREADS) {
    const int n = i / D, dim = i % D;
    s_q[(dim / 16) * 128 + n * 16 + dim % 16] = q8[bh * NPAD * D + i];
  }
  for (int i = tid; i < 2 * NPAD * HALF; i += NTHREADS) s_p[i] = __float2half(0.f);
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], NCWARPS);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  fence_proxy_async();
  __syncthreads();

  if (warp == NCWARPS) {
    if (lane == 0) {
      const size_t row0 = bh * S + tok0;
      for (int it = 0; it < niter; ++it) {
        const int s = it % NST;
        mbar_wait_sleep(&empty[s], ((it / NST) & 1) ^ 1);
        mbar_expect_tx(&full[s], 2 * CODE_BYTES + KSC_BYTES + VSC_BYTES);
        const size_t r = row0 + (size_t)it * STAGE;
        bulk_g2s(s_k + s * CODE_BYTES, kp + r * ROW_BYTES, CODE_BYTES, &full[s]);
        bulk_g2s(s_ks + s * KSC_BYTES, ksc + r * 4, KSC_BYTES, &full[s]);
        bulk_g2s(s_v + s * CODE_BYTES, vp + r * ROW_BYTES, CODE_BYTES, &full[s]);
        bulk_g2s(s_vs + s * VSC_BYTES, vsc + r * 8, VSC_BYTES, &full[s]);
      }
    }
    return;
  }

  const int wg = warp / 4, wi = warp % 4, g = lane / 4, t = lane % 4, hp = t & 1;
  const bool hi = t >= 2;
  const int hb = 2 * hp;
  constexpr int RED0 = 2;
  const uint64_t desc_q = make_desc(smem_u32(s_q));
  __half* Pw = s_p + wg * NPAD * HALF;
  const uint64_t desc_p = make_desc(smem_u32(Pw));
  float ma = -INFINITY, mb = -INFINITY, la = 0.f, lb = 0.f;
  float O[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

  auto stage_body = [&](const int s, const int it) {
    // ---- QK on FP8 tensor cores
    const uint8_t* kb = s_k + s * CODE_BYTES;
    float sc[2][2];
#pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
      const int r0 = wg * HALF + sub * 64 + wi * 16 + g;
      const uint8_t* krow = kb + r0 * ROW_BYTES;
      const uint4 w0 = *reinterpret_cast<const uint4*>(krow + t * 16);
      const uint4 w1 = *reinterpret_cast<const uint4*>(krow + 8 * ROW_BYTES + t * 16);
      const uint2 sw = *reinterpret_cast<const uint2*>(s_ks + s * KSC_BYTES + (((wg * 2 + sub) * 4 + wi) * 8 + g) * 8);
      const uint32_t x0[4] = {w0.x, w0.y, w0.z, w0.w}, x1[4] = {w1.x, w1.y, w1.z, w1.w};
      uint32_t A[4][4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        A[i][0] = k8L(x0[i]);  // row g
        A[i][1] = k8L(x1[i]);  // row g + 8
        A[i][2] = k8H(x0[i]);
        A[i][3] = k8H(x1[i]);
      }
      float P[4][4];
#pragma unroll
      for (int i = 0; i < 4; ++i) P[i][0] = P[i][1] = P[i][2] = P[i][3] = 0.f;
      wg_fence();
#pragma unroll
      for (int i = 0; i < 4; ++i) mma_rs8(P[i], A[i], desc_q + 16 * i);
      wg_commit();
      // scales of (row g, groups 0..3) and (row g + 8, groups 0..3), converted while the wgmmas run
      const float2 a01 = h2f2(e4m3x2_f16x2(sw.x)), b01 = h2f2(e4m3x2_f16x2(sw.x >> 16));
      const float2 a23 = h2f2(e4m3x2_f16x2(sw.y)), b23 = h2f2(e4m3x2_f16x2(sw.y >> 16));
      const float sg[4] = {a01.x, a01.y, a23.x, a23.y}, s8[4] = {b01.x, b01.y, b23.x, b23.y};
      wg_wait<0>();
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) fence_operand(P[i][j]);
      float dq[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        dq[0] = fmaf(sg[i], P[i][0], dq[0]);
        dq[1] = fmaf(sg[i], P[i][1], dq[1]);
        dq[2] = fmaf(s8[i], P[i][2], dq[2]);
        dq[3] = fmaf(s8[i], P[i][3], dq[3]);
      }
      // lanes t = 2, 3 take row + 8 from lane t - 2
      const float u2 = __shfl_up_sync(0xffffffffu, dq[2], 2), u3 = __shfl_up_sync(0xffffffffu, dq[3], 2);
      sc[sub][0] = (hi ? u2 : dq[0]) * score_mul;
      sc[sub][1] = (hi ? u3 : dq[1]) * score_mul;
    }

    // ---- per-warpgroup max per head
    float xa = fmaxf(sc[0][0], sc[1][0]), xb = fmaxf(sc[0][1], sc[1][1]);
#pragma unroll
    for (int off = RED0; off < 32; off <<= 1) {
      xa = fmaxf(xa, __shfl_xor_sync(0xffffffffu, xa, off));
      xb = fmaxf(xb, __shfl_xor_sync(0xffffffffu, xb, off));
    }
    float* wm = s_wmax + ((s & 1) * 2 + wg) * (4 * G);
    if (g == 0 && t < 2) wm[wi * G + hb] = xa, wm[wi * G + hb + 1] = xb;
    named_bar(2 + wg, 128);
    xa = fmaxf(fmaxf(wm[hb], wm[G + hb]), fmaxf(wm[2 * G + hb], wm[3 * G + hb]));
    xb = fmaxf(fmaxf(wm[hb + 1], wm[G + hb + 1]), fmaxf(wm[2 * G + hb + 1], wm[3 * G + hb + 1]));
    const float mna = fmaxf(ma, xa), mnb = fmaxf(mb, xb);
    const float aa = exp2f(ma - mna), ab = exp2f(mb - mnb);
    ma = mna, mb = mnb;

    // ---- P (fp16) into shared memory
    float pla = 0.f, plb = 0.f;
#pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
      const int tw = sub * 64 + wi * 16 + g + (hi ? 8 : 0);
      const float pa = exp2f(sc[sub][0] - ma + P_SHIFT), pb = exp2f(sc[sub][1] - mb + P_SHIFT);
      const int o = (tw >> 3) * 64 + (tw & 7);
      Pw[o + hb * 8] = __float2half_rn(pa);
      Pw[o + (hb + 1) * 8] = __float2half_rn(pb);
      pla += pa, plb += pb;
    }
    la = la * aa + pla, lb = lb * ab + plb;
    fence_proxy_async();
    named_bar(2 + wg, 128);

    // ---- PV (decode_fp4.cu's NVFP4 path)
    float Os[2][4];
    const uint8_t* vb = s_v + s * CODE_BYTES + wg * (CODE_BYTES / 2);
    const uint8_t* vsb = s_vs + s * VSC_BYTES + wg * (VSC_BYTES / 2);
#pragma unroll
    for (int kh = 0; kh < 2; ++kh) {
      uint32_t A[4][2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        const uint4 w = *reinterpret_cast<const uint4*>(vb + ((((kh * 2 + mt) * 4 + wi) * 32 + lane) * 16));
        const uint4 vs = *reinterpret_cast<const uint4*>(vsb + ((((kh * 2 + mt) * 4 + wi) * 4 + t) * 16));
        const uint32_t x[4] = {w.x, w.y, w.z, w.w}, sv[4] = {vs.x, vs.y, vs.z, vs.w};
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          const uint32_t sA = e4m3x2_f16x2(sv[kk]), sB = e4m3x2_f16x2(sv[kk] >> 16);
          A[kk][mt][0] = hmul2(fA(x[kk]), sA);
          A[kk][mt][1] = hmul2(fB(x[kk]), sA);
          A[kk][mt][2] = hmul2(fC(x[kk]), sB);
          A[kk][mt][3] = hmul2(fD(x[kk]), sB);
        }
      }
      wg_fence();
#pragma unroll
      for (int kk = 0; kk < 4; ++kk)
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) mma_rs(Os[mt], A[kk][mt], desc_p + 16 * (4 * kh + kk), (kh | kk) != 0);
      wg_commit();
      wg_wait<0>();
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
      for (int i = 0; i < 4; ++i) fence_operand(Os[mt][i]);
      O[mt][0] = O[mt][0] * aa + Os[mt][0], O[mt][1] = O[mt][1] * ab + Os[mt][1];
      O[mt][2] = O[mt][2] * aa + Os[mt][2], O[mt][3] = O[mt][3] * ab + Os[mt][3];
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(&empty[s]);
  };
  for (int it0 = 0; it0 < niter; it0 += NST) {
    const uint32_t ph = (it0 / NST) & 1;
#pragma unroll
    for (int s = 0; s < NST; ++s) {
      const int it = it0 + s;
      if (it >= niter) break;
      mbar_wait(&full[s], ph);
      stage_body(s, it);
    }
  }

  // ---- l per head over the warpgroup; merge the two warpgroups; write (as decode_fp4.cu, G = 4)
#pragma unroll
  for (int off = RED0; off < 32; off <<= 1) {
    la += __shfl_xor_sync(0xffffffffu, la, off), lb += __shfl_xor_sync(0xffffffffu, lb, off);
  }
  float* lc = s_lc + wg * (8 * G);
  if (g == 0 && t < 2) lc[(wi * G + hb) * 2] = la, lc[(wi * G + hb + 1) * 2] = lb;
  named_bar(2 + wg, 128);
  float L[2] = {0.f, 0.f};
#pragma unroll
  for (int w = 0; w < 4; ++w)
#pragma unroll
    for (int j = 0; j < 2; ++j) L[j] += lc[(w * G + hb + j) * 2];
  float M[2] = {ma, mb};

  named_bar(1, NCWARPS * 32);
  float* mrg = reinterpret_cast<float*>(s_k) + (wi * 32 + lane) * 12;
  if (wg == 1) {
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int i = 0; i < 4; ++i) mrg[mt * 4 + i] = O[mt][i];
    mrg[8] = M[0], mrg[9] = M[1], mrg[10] = L[0], mrg[11] = L[1];
  }
  named_bar(1, NCWARPS * 32);
  if (wg == 0 && t < 2) {
    float w0[2], w1[2], Lt[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const float m1 = mrg[8 + j], l1 = mrg[10 + j], mm = fmaxf(M[j], m1);
      w0[j] = exp2f(M[j] - mm), w1[j] = exp2f(m1 - mm);
      Lt[j] = w0[j] * L[j] + w1[j] * l1;
      M[j] = mm;
    }
    float* o = out_o + ((bh * splits + split) * G) * D;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      const int dim = 64 * mt + 16 * wi + g;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        o[(2 * t + j) * D + dim] = (w0[j] * O[mt][j] + w1[j] * mrg[mt * 4 + j]) * out_mul / Lt[j];
        o[(2 * t + j) * D + dim + 8] = (w0[j] * O[mt][2 + j] + w1[j] * mrg[mt * 4 + 2 + j]) * out_mul / Lt[j];
      }
    }
    if (wi == 0 && g == 0) {
      float* ml = out_ml + ((bh * splits + split) * G + 2 * t) * 2;
      ml[0] = M[0], ml[1] = Lt[0], ml[2] = M[1], ml[3] = Lt[1];
    }
  }
}

}  // namespace

// kp (B, HKV, S, 16) int32 (pack_k8 layout); ksc (B, HKV, S, 4) uint8 (E4M3 per 32 dims, MX scale order); vp, vsc:
// decode_fp4.py's NVFP4 V layouts; q8 (B, HKV, 8, 128) uint8 E4M3. Returns as decode_fp4().
std::vector<torch::Tensor> decode_k8(torch::Tensor kp, torch::Tensor ksc, torch::Tensor vp, torch::Tensor vsc,
                                     torch::Tensor q8, double score_mul, double out_mul, int64_t splits) {
  for (auto* x : {&kp, &vp}) TORCH_CHECK(x->is_cuda() && x->is_contiguous() && x->dim() == 4 && x->size(3) == 16);
  TORCH_CHECK(ksc.is_contiguous() && ksc.scalar_type() == torch::kUInt8 && ksc.size(-1) == 4);
  TORCH_CHECK(vsc.is_contiguous() && vsc.scalar_type() == torch::kUInt8 && vsc.size(-1) == 8);
  TORCH_CHECK(q8.is_contiguous() && q8.scalar_type() == torch::kUInt8 && q8.size(2) == NPAD && q8.size(3) == D);
  const int B = kp.size(0), HKV = kp.size(1), S = kp.size(2);
  TORCH_CHECK(S % (splits * STAGE) == 0, "S must be a multiple of splits * 256");
  auto out_o = torch::empty({B, HKV, splits, G, D}, kp.options().dtype(torch::kFloat));
  auto out_ml = torch::empty({B, HKV, splits, G, 2}, kp.options().dtype(torch::kFloat));
  constexpr size_t smem = smem_bytes();
  static bool configured = false;
  if (!configured) {
    TORCH_CHECK(cudaFuncSetAttribute(decode_k8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) ==
                cudaSuccess);
    configured = true;
  }
  dim3 grid(splits, HKV, B);
  decode_k8_kernel<<<grid, NTHREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint8_t*>(kp.data_ptr()), ksc.data_ptr<uint8_t>(),
      reinterpret_cast<const uint8_t*>(vp.data_ptr()), vsc.data_ptr<uint8_t>(), q8.data_ptr<uint8_t>(),
      out_o.data_ptr<float>(), out_ml.data_ptr<float>(), S, HKV, (float)score_mul, (float)out_mul);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "decode_k8 launch failed");
  return {out_o, out_ml};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode_k8", &decode_k8, "C1 prototype: NVFP4 decode with K on FP8 tensor cores (E4M3 scale per 32 dims)");
}
