// rebuild tag: P scaled by 2^10 (11-11); common.cuh rev 3 (producer sleeps between polls, 11-11)
// Hopper-native INT4 KV decode attention (11-10): K and V, asymmetric per-token INT4 (symmetric is z = -8s).
// Built on the K-side prototype v4 (kside.cu): TMA bulk producer warp, two compute warpgroups, wgmma with the
// reconstructed operand straight from registers (fp16 magic numbers, one LOP3 per pair), write-time layouts.
//
// Per CTA: one (batch, kv head, split); GQA 4 query heads padded to wgmma N = 8. A stage is 256 tokens; compute
// warpgroup wg owns tokens [128 wg, 128 wg + 128) of every stage and keeps its own (m, l, O); the two warpgroups
// merge once at the end.
//   QK: S^T = K q^T, wgmma m64n8k16, A = K (64 tokens x 16 dims) from registers, B = q'^T in shared memory.
//       score2 = (s_t (acc - K0) + z_t Qs) * softmax_scale * log2(e), K0 = 1024 sum q'_d, Qs = sum q^_d.
//   PV: O^T = V^T P'^T, wgmma m64n8k16, A = V^T (64 dims x 16 tokens) from registers, B = P'^T in shared memory,
//       P'_t = p_t s_t / m_t (fp16), m_t = 16 when t % 16 >= 8 (those nibbles come out as 1024 + 16c).
//       O = acc - C with C = sum_t (1024 P'_t - p_t z_t): one scalar per head, rescaled with O; it uses the
//       fp16-rounded P' so it cancels exactly what the tensor cores summed.
// Layouts (kv4/cuda/decode_int4.py): K rows permuted per token (as kside.cu); V permuted per 256-token block:
//   [wg 2][kh 2][mt 2][warp 4][lane 32][kk 4] 32-bit words; word (lane = 4g + t, k-step j = 4kh + kk, token base
//   T = 128 wg + 16 j, dim base Dm = 64 mt + 16 warp) holds nibbles
//   bits 0-3 (Dm+g, T+2t)  16-19 (Dm+g, T+2t+1)  8-11 (Dm+g+8, T+2t)  24-27 (Dm+g+8, T+2t+1)
//   bits 4-7 (Dm+g, T+2t+8) 20-23 (Dm+g, T+2t+9) 12-15 (Dm+g+8, T+2t+8) 28-31 (Dm+g+8, T+2t+9)
// so R0 = lop_lo(w), R1 = lop_lo(w >> 8), R2 = lop_hi(w), R3 = lop_hi(w >> 8) are the four A registers.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "common.cuh"

namespace {
using namespace kv4cu;

constexpr float P_SHIFT = 10.f;

constexpr int D = 128;
constexpr int ROW_BYTES = 64;
constexpr int STAGE = 256;
constexpr int HALF = 128;  // tokens per warpgroup per stage
constexpr int NCWARPS = 8;
constexpr int NTHREADS = NCWARPS * 32 + 32;
constexpr int NPAD = 8;
constexpr int GQA = 4;
constexpr int CODE_BYTES = STAGE * ROW_BYTES;  // 16 KB per tensor per stage
constexpr int SZ_BYTES = STAGE * 4;            // (scale, zero) fp16 pairs

template <int NST>
constexpr size_t smem_bytes() {
  return 2 * NST * CODE_BYTES + 2 * NST * SZ_BYTES + NPAD * D * 2 + 2 * NPAD * HALF * 2 + 64 * 4 + 64 * 4 +
         64 * 4 + 2 * NST * 8;
}

// CENTER (11-10 fix 3): subtract the fp16 offset from V's reconstructed values before the MMA (one HADD2 per
// register: 1024 + c - 1032 = c - 8, 1024 + 16c - 1152 = 16 (c - 8)), so no large offset ever enters the tensor-core
// accumulator; C then only carries the zero-point term -sum p (z + 8 s).
template <int NST, bool CENTER>
__global__ void __launch_bounds__(NTHREADS) decode_int4_kernel(const uint8_t* __restrict__ kp,    // (B,HKV,S,64)
                                                               const __half2* __restrict__ ksz,  // (B,HKV,S)
                                                               const uint8_t* __restrict__ vp,    // (B,HKV,S,64)
                                                               const __half2* __restrict__ vsz,  // (B,HKV,S)
                                                               const __half* __restrict__ qp,     // (B,HKV,8,128)
                                                               const float* __restrict__ qc,      // (B,HKV,8,2)
                                                               float* __restrict__ out_o,   // (B,HKV,splits,4,128)
                                                               float* __restrict__ out_ml,  // (B,HKV,splits,4,2)
                                                               int S, int HKV, float scale_log2) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* s_k = smem;
  uint8_t* s_v = s_k + NST * CODE_BYTES;
  __half2* s_ksz = reinterpret_cast<__half2*>(s_v + NST * CODE_BYTES);
  __half2* s_vsz = s_ksz + NST * STAGE;
  __half* s_q = reinterpret_cast<__half*>(s_vsz + NST * STAGE);
  __half* s_p = s_q + NPAD * D;                                  // 2 warpgroups x 1024 halves
  float* s_wmax = reinterpret_cast<float*>(s_p + 2 * NPAD * HALF);  // [parity 2][wg 2][warp 4][head 4]
  float* s_lc = s_wmax + 64;                                     // [wg 2][warp 4][head 4][l, c]
  float* s_cst = s_lc + 64;                                      // [parity 2][wg 2][warp 4][head 4]: stage C
  uint64_t* full = reinterpret_cast<uint64_t*>(s_cst + 64);
  uint64_t* empty = full + NST;

  const int split = blockIdx.x, h = blockIdx.y, b = blockIdx.z, splits = gridDim.x;
  const int tok_per = S / splits, tok0 = split * tok_per, niter = tok_per / STAGE;
  const size_t bh = (size_t)b * HKV + h;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;

  for (int i = tid; i < NPAD * D; i += NTHREADS) {  // q'^T in core-matrix order
    const int n = i / D, dim = i % D;
    s_q[(dim / 8) * 64 + n * 8 + dim % 8] = qp[bh * NPAD * D + i];
  }
  for (int i = tid; i < 2 * NPAD * HALF; i += NTHREADS) s_p[i] = __float2half(0.f);  // padded heads stay 0
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], NCWARPS);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  fence_proxy_async();
  __syncthreads();

  if (warp == NCWARPS) {  // producer
    if (lane == 0) {
      const size_t row0 = bh * S + tok0;
      for (int it = 0; it < niter; ++it) {
        const int s = it % NST;
        mbar_wait_sleep(&empty[s], ((it / NST) & 1) ^ 1);
        mbar_expect_tx(&full[s], 2 * (CODE_BYTES + SZ_BYTES));
        const size_t r = row0 + (size_t)it * STAGE;
        bulk_g2s(s_k + s * CODE_BYTES, kp + r * ROW_BYTES, CODE_BYTES, &full[s]);
        bulk_g2s(s_ksz + s * STAGE, ksz + r, SZ_BYTES, &full[s]);
        bulk_g2s(s_v + s * CODE_BYTES, vp + r * ROW_BYTES, CODE_BYTES, &full[s]);
        bulk_g2s(s_vsz + s * STAGE, vsz + r, SZ_BYTES, &full[s]);
      }
    }
    return;
  }

  const int wg = warp / 4, wi = warp % 4, g = lane / 4, t = lane % 4, hp = t & 1;
  const bool hi = t >= 2;  // lanes t = 2, 3 take row + 8 (and heads 2hp, 2hp + 1) after the QK shuffle
  const uint32_t magic = opaque_magic();
  const uint64_t desc_q = make_desc(smem_u32(s_q));
  __half* Pw = s_p + wg * NPAD * HALF;
  const uint64_t desc_p = make_desc(smem_u32(Pw));
  const float K0a = qc[(bh * NPAD + 2 * hp) * 2], Qa = qc[(bh * NPAD + 2 * hp) * 2 + 1];
  const float K0b = qc[(bh * NPAD + 2 * hp + 1) * 2], Qb = qc[(bh * NPAD + 2 * hp + 1) * 2 + 1];
  float ma = -INFINITY, mb = -INFINITY, la = 0.f, lb = 0.f;
  float O[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

  for (int it = 0; it < niter; ++it) {
    const int s = it % NST;
    mbar_wait(&full[s], (it / NST) & 1);

    // ---- QK: two 64-token subtiles of this warpgroup's 128 tokens
    const uint8_t* kb = s_k + s * CODE_BYTES;
    float sc[2][2];
#pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
      const int r0 = wg * HALF + sub * 64 + wi * 16 + g;
      const uint4 w0 = *reinterpret_cast<const uint4*>(kb + r0 * ROW_BYTES + t * 16);
      const uint4 w1 = *reinterpret_cast<const uint4*>(kb + (r0 + 8) * ROW_BYTES + t * 16);
      const uint32_t x0[4] = {w0.x, w0.y, w0.z, w0.w}, x1[4] = {w1.x, w1.y, w1.z, w1.w};
      uint32_t A[8][4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t a = x0[i], c = x1[i], ah = a >> 8, ch = c >> 8;
        A[2 * i][0] = lop_lo(a, magic);
        A[2 * i][1] = lop_lo(c, magic);
        A[2 * i][2] = lop_hi(a, magic);
        A[2 * i][3] = lop_hi(c, magic);
        A[2 * i + 1][0] = lop_lo(ah, magic);
        A[2 * i + 1][1] = lop_lo(ch, magic);
        A[2 * i + 1][2] = lop_hi(ah, magic);
        A[2 * i + 1][3] = lop_hi(ch, magic);
      }
      float d[4] = {0.f, 0.f, 0.f, 0.f};
      wg_fence();
#pragma unroll
      for (int j = 0; j < 8; ++j) mma_rs(d, A[j], desc_q + 16 * j, j > 0);
      wg_commit();
      wg_wait<0>();
#pragma unroll
      for (int i = 0; i < 4; ++i) fence_operand(d[i]);
      const float u2 = __shfl_up_sync(0xffffffffu, d[2], 2), u3 = __shfl_up_sync(0xffffffffu, d[3], 2);
      const float2 sz = __half22float2(s_ksz[s * STAGE + r0 + (hi ? 8 : 0)]);
      sc[sub][0] = (sz.x * ((hi ? u2 : d[0]) - K0a) + sz.y * Qa) * scale_log2;
      sc[sub][1] = (sz.x * ((hi ? u3 : d[1]) - K0b) + sz.y * Qb) * scale_log2;
    }

    // ---- one max per head over this warpgroup's 128 tokens
    float xa = fmaxf(sc[0][0], sc[1][0]), xb = fmaxf(sc[0][1], sc[1][1]);
#pragma unroll
    for (int off = 2; off < 32; off <<= 1) {
      xa = fmaxf(xa, __shfl_xor_sync(0xffffffffu, xa, off));
      xb = fmaxf(xb, __shfl_xor_sync(0xffffffffu, xb, off));
    }
    float* wm = s_wmax + ((it & 1) * 2 + wg) * 16;
    if (g == 0 && t < 2) wm[wi * 4 + 2 * hp] = xa, wm[wi * 4 + 2 * hp + 1] = xb;
    named_bar(2 + wg, 128);
    xa = fmaxf(fmaxf(wm[2 * hp], wm[4 + 2 * hp]), fmaxf(wm[8 + 2 * hp], wm[12 + 2 * hp]));
    xb = fmaxf(fmaxf(wm[2 * hp + 1], wm[5 + 2 * hp]), fmaxf(wm[9 + 2 * hp], wm[13 + 2 * hp]));
    const float mna = fmaxf(ma, xa), mnb = fmaxf(mb, xb);
    const float aa = exp2f(ma - mna), ab = exp2f(mb - mnb);
    ma = mna, mb = mnb;

    // ---- P' = p s_v / m_t into shared memory (B operand, core-matrix order), and the correction C
    float pla = 0.f, plb = 0.f, pca = 0.f, pcb = 0.f;
#pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
      const int tw = sub * 64 + wi * 16 + g + (hi ? 8 : 0);  // token within this warpgroup's 128
      const float2 vz = __half22float2(s_vsz[s * STAGE + wg * HALF + tw]);
      // P is scaled by 2^P_SHIFT (11-11): with a dominant sink most p are ~1e-4, and the INT4 path's P' = p s / 16 fell
      // into FP16 subnormals (real-KV check: 3e-3 error at 8192 tokens per CTA). l and O scale together, so O / l and
      // the split merge are unchanged; P' stays <= 2^10 s (fine for V scales < 64).
      const float pa = exp2f(sc[sub][0] - ma + P_SHIFT), pb = exp2f(sc[sub][1] - mb + P_SHIFT);
      const float fac = vz.x * (hi ? 0.0625f : 1.f);
      const __half ha = __float2half_rn(pa * fac), hb = __float2half_rn(pb * fac);
      const int o = (tw >> 3) * 64 + (tw & 7);
      Pw[o + 2 * hp * 8] = ha;
      Pw[o + (2 * hp + 1) * 8] = hb;
      pla += pa, plb += pb;
      if constexpr (CENTER) {
        pca -= pa * (vz.y + 8.f * vz.x);
        pcb -= pb * (vz.y + 8.f * vz.x);
      } else {
        pca += 1024.f * __half2float(ha) - pa * vz.y;
        pcb += 1024.f * __half2float(hb) - pb * vz.y;
      }
    }
    la = la * aa + pla, lb = lb * ab + plb;
    // this stage's C over the warpgroup's 128 tokens (11-10 fix 2): subtracted from the stage accumulator before it
    // joins O, so O never carries the 1024 sum P' offset (with it, the fp32 cancellation error grew like sqrt(S)).
#pragma unroll
    for (int off = 2; off < 32; off <<= 1) {
      pca += __shfl_xor_sync(0xffffffffu, pca, off);
      pcb += __shfl_xor_sync(0xffffffffu, pcb, off);
    }
    float* cs = s_cst + ((it & 1) * 2 + wg) * 16;
    if (g == 0 && t < 2) cs[wi * 4 + 2 * hp] = pca, cs[wi * 4 + 2 * hp + 1] = pcb;
    fence_proxy_async();
    named_bar(2 + wg, 128);
    const float Ca = cs[2 * hp] + cs[4 + 2 * hp] + cs[8 + 2 * hp] + cs[12 + 2 * hp];
    const float Cb = cs[2 * hp + 1] + cs[5 + 2 * hp] + cs[9 + 2 * hp] + cs[13 + 2 * hp];

    // ---- PV: 8 k-steps of 16 tokens, 2 tiles of 64 dims
    // A fresh accumulator per stage (11-10 fix): it carries the 1024 sum P' offset, and the tensor cores' fp32
    // accumulation lost accuracy over thousands of tokens (error grew with tokens per CTA, 2e-2 at 4096). Each stage
    // sums only 128 tokens there, then joins O in ordinary fp32: O = O * alpha + acc.
    float Os[2][4];
    const uint8_t* vb = s_v + s * CODE_BYTES + wg * (CODE_BYTES / 2);
#pragma unroll
    for (int kh = 0; kh < 2; ++kh) {
      uint32_t A[4][2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        const uint4 w = *reinterpret_cast<const uint4*>(vb + ((((kh * 2 + mt) * 4 + wi) * 32 + lane) * 16));
        const uint32_t x[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          const uint32_t x8 = x[kk] >> 8;
          A[kk][mt][0] = lop_lo(x[kk], magic);
          A[kk][mt][1] = lop_lo(x8, magic);
          A[kk][mt][2] = lop_hi(x[kk], magic);
          A[kk][mt][3] = lop_hi(x8, magic);
          if constexpr (CENTER) {  // tokens 2t, 2t+1 (m = 1): -1032 = 0xE408; tokens 2t+8, 2t+9 (m = 16): -1152 = 0xE480
            A[kk][mt][0] = hadd2_u32(A[kk][mt][0], 0xE408E408u);
            A[kk][mt][1] = hadd2_u32(A[kk][mt][1], 0xE408E408u);
            A[kk][mt][2] = hadd2_u32(A[kk][mt][2], 0xE480E480u);
            A[kk][mt][3] = hadd2_u32(A[kk][mt][3], 0xE480E480u);
          }
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
    for (int mt = 0; mt < 2; ++mt) {  // accumulator cols 2t, 2t+1 = heads 2hp, 2hp+1 for t < 2 (others padding)
#pragma unroll
      for (int i = 0; i < 4; ++i) fence_operand(Os[mt][i]);
      O[mt][0] = O[mt][0] * aa + (Os[mt][0] - Ca), O[mt][1] = O[mt][1] * ab + (Os[mt][1] - Cb);
      O[mt][2] = O[mt][2] * aa + (Os[mt][2] - Ca), O[mt][3] = O[mt][3] * ab + (Os[mt][3] - Cb);
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(&empty[s]);
  }

  // ---- per warpgroup: l and C per head over its 128 threads; O - C
#pragma unroll
  for (int off = 2; off < 32; off <<= 1) {
    la += __shfl_xor_sync(0xffffffffu, la, off), lb += __shfl_xor_sync(0xffffffffu, lb, off);
  }
  float* lc = s_lc + wg * 32;
  if (g == 0 && t < 2) lc[(wi * 4 + 2 * hp) * 2] = la, lc[(wi * 4 + 2 * hp + 1) * 2] = lb;
  named_bar(2 + wg, 128);
  float L[2] = {0.f, 0.f};
#pragma unroll
  for (int w = 0; w < 4; ++w)
#pragma unroll
    for (int j = 0; j < 2; ++j) L[j] += lc[(w * 4 + 2 * hp + j) * 2];
  float M[2] = {ma, mb};

  // ---- merge the two warpgroups (warpgroup 1 hands over through the drained K buffer) and write this split
  named_bar(1, NCWARPS * 32);  // everyone is done with the stage buffers
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
    float* o = out_o + ((bh * splits + split) * GQA) * D;  // heads 2t, 2t+1 (t < 2: hp = t)
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      const int dim = 64 * mt + 16 * wi + g;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        o[(2 * t + j) * D + dim] = (w0[j] * O[mt][j] + w1[j] * mrg[mt * 4 + j]) / Lt[j];
        o[(2 * t + j) * D + dim + 8] = (w0[j] * O[mt][2 + j] + w1[j] * mrg[mt * 4 + 2 + j]) / Lt[j];
      }
    }
    if (wi == 0 && g == 0) {
      float* ml = out_ml + ((bh * splits + split) * GQA + 2 * t) * 2;
      ml[0] = M[0], ml[1] = Lt[0], ml[2] = M[1], ml[3] = Lt[1];
    }
  }
}

template <int NST, bool CENTER>
void launch(const torch::Tensor& kp, const torch::Tensor& ksz, const torch::Tensor& vp, const torch::Tensor& vsz,
            const torch::Tensor& qp, const torch::Tensor& qc, torch::Tensor& out_o, torch::Tensor& out_ml, int splits,
            float scale_log2) {
  const int B = kp.size(0), HKV = kp.size(1), S = kp.size(2);
  constexpr size_t smem = smem_bytes<NST>();
  auto kern = decode_int4_kernel<NST, CENTER>;
  static bool configured = false;
  if (!configured) {
    TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess);
    configured = true;
  }
  dim3 grid(splits, HKV, B);
  kern<<<grid, NTHREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint8_t*>(kp.data_ptr()), reinterpret_cast<const __half2*>(ksz.data_ptr()),
      reinterpret_cast<const uint8_t*>(vp.data_ptr()), reinterpret_cast<const __half2*>(vsz.data_ptr()),
      reinterpret_cast<const __half*>(qp.data_ptr()), qc.data_ptr<float>(), out_o.data_ptr<float>(),
      out_ml.data_ptr<float>(), S, HKV, scale_log2);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "decode_int4 launch failed");
}

}  // namespace

// kp, vp (B, HKV, S, 16) int32 (permuted, see the header); ksz, vsz (B, HKV, S, 2) fp16 (scale, zero);
// qp (B, HKV, 8, 128) fp16; qc (B, HKV, 8, 2) fp32 (K0, Qs). Returns out_o (B, HKV, splits, 4, 128) fp32 (normalized
// per split) and out_ml (B, HKV, splits, 4, 2) fp32 (base-2 max, sum).
std::vector<torch::Tensor> decode_int4(torch::Tensor kp, torch::Tensor ksz, torch::Tensor vp, torch::Tensor vsz,
                                       torch::Tensor qp, torch::Tensor qc, double softmax_scale, int64_t splits,
                                       int64_t nst, bool center) {
  for (auto* x : {&kp, &vp}) TORCH_CHECK(x->is_cuda() && x->is_contiguous() && x->dim() == 4 && x->size(3) == 16);
  for (auto* x : {&ksz, &vsz})
    TORCH_CHECK(x->is_contiguous() && x->scalar_type() == torch::kHalf && x->size(-1) == 2);
  TORCH_CHECK(qp.is_contiguous() && qp.scalar_type() == torch::kHalf && qp.size(2) == NPAD && qp.size(3) == D);
  TORCH_CHECK(qc.is_contiguous() && qc.scalar_type() == torch::kFloat);
  const int B = kp.size(0), HKV = kp.size(1), S = kp.size(2);
  TORCH_CHECK(S % (splits * STAGE) == 0, "S must be a multiple of splits * 256");
  auto out_o = torch::empty({B, HKV, splits, GQA, D}, kp.options().dtype(torch::kFloat));
  auto out_ml = torch::empty({B, HKV, splits, GQA, 2}, kp.options().dtype(torch::kFloat));
  const float sl2 = (float)(softmax_scale * 1.4426950408889634);
  if (center) {
    if (nst == 2) launch<2, true>(kp, ksz, vp, vsz, qp, qc, out_o, out_ml, splits, sl2);
    else if (nst == 3) launch<3, true>(kp, ksz, vp, vsz, qp, qc, out_o, out_ml, splits, sl2);
    else TORCH_CHECK(false, "nst must be 2 or 3");
  } else {
    if (nst == 2) launch<2, false>(kp, ksz, vp, vsz, qp, qc, out_o, out_ml, splits, sl2);
    else if (nst == 3) launch<3, false>(kp, ksz, vp, vsz, qp, qc, out_o, out_ml, splits, sl2);
    else TORCH_CHECK(false, "nst must be 2 or 3");
  }
  return {out_o, out_ml};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode_int4", &decode_int4, "Hopper INT4 KV decode (wgmma RS for QK and PV, TMA bulk producer)",
        py::arg("kp"), py::arg("ksz"), py::arg("vp"), py::arg("vsz"), py::arg("qp"), py::arg("qc"),
        py::arg("softmax_scale"), py::arg("splits"), py::arg("nst"), py::arg("center") = true);
}
