// rebuild tag: G=8 (11-11 evening), paged v5 (tail after loop), paged v4 (uniform votes), paged diag 3/5, paged v3 (tail-only masking), paged v2 (page-major smem); diag modes; P scaled by 2^10 (11-11); rev 3b (restored 7c86ad7: producer warp, sleep, unrolled stages), 11-11
// Hopper-native NVFP4 KV decode attention (11-10 night). Same skeleton as decode_int4.cu (TMA bulk producer warp, two
// compute warpgroups owning half a stage each, wgmma m64n8k16 with the reconstructed K / V^T straight from registers,
// P' through shared memory); what changes is reconstruction and scaling.
//
// E2M1 -> FP16 by bit placement: code (s e1 e0 m); (e1 e0 m) at FP16 bits 11..9 and s at bit 15 is the value * 2^-14
// (0.5 becomes the FP16 subnormal 2^-15, exact). Write-time bit layout: a 32-bit word holds 4 pairs (low/high half of
// an f16x2 register, same relative position in both halves):
//   slot A: value bits 9-11, sign 15      ->  w & 0x8E008E00
//   slot B: value bits 6-8,  sign 12      ->  (w << 3) & 0x8E008E00
//   slot C: value bits 0-2,  sign 13      ->  ((w << 9) & 0x0E000E00) | ((w << 2) & 0x80008000)
//   slot D: value bits 3-5,  sign 14      ->  ((w << 6) & 0x0E000E00) | ((w << 1) & 0x80008000)
// (4 slots of the form {p, p+1, p+2, p+6} cannot tile 16 bits, so two slots need the longer form.)
// Scales: one E4M3 per 16 values (one wgmma k-step for K; one 16-dim group for V), applied with HMUL2 after
// reconstruction (Hopper tensor cores have no block scaling); cvt.rn.f16x2.e4m3x2 turns 2 scale bytes into f16x2.
// The FP32 tensor scales and the 2^14 go into the closing multiplies (scores, output).
//   K word (lane t, word i), rows g / g+8 loaded separately: A = (dims 32i+2t, +1) = R0 of k-step 2i, B = (32i+2t+8, +9)
//   = R2 of k-step 2i, C = (32i+16+2t, +1) = R0 of k-step 2i+1, D = (32i+24+2t, +1) = R2 of k-step 2i+1.
//   K scales per stage (2 KB): [wg 2][sub 2][warp 4][g 8][k 4][(grp 2k: row g, row g+8), (grp 2k+1: row g, row g+8)].
//   V word as decode_int4's [wg][kh][mt][warp][lane][kk] with A = R0 (dim g; tokens 2t, 2t+1), B = R1 (dim g+8; same),
//   C = R2 (dim g; tokens 2t+8, 2t+9), D = R3 (dim g+8; same).
//   V scales per stage (2 KB): [wg 2][kh 2][mt 2][warp 4][t 4][kk 4][j 2][e 2], token 128wg + 16(4kh+kk) + 2t + 8j + e,
//   group 4mt + warp.
#include <type_traits>
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
constexpr int HALF = 128;
constexpr int NCWARPS = 8;
constexpr int NTHREADS = NCWARPS * 32 + 32;
constexpr int NPAD = 8;
constexpr int GQA = 4;  // default group size; G = 8 (11-11) fills the wgmma N = 8 with real query heads
constexpr int CODE_BYTES = STAGE * ROW_BYTES;  // 16 KB
// scale bytes per stage: NVFP4 8 E4M3 per token row (2 KB); MXFP4 4 E8M0 per token row (1 KB)
template <bool MX>
constexpr int sc_bytes() { return STAGE * (MX ? 4 : 8); }

__device__ __forceinline__ uint32_t fA(uint32_t w) { return w & 0x8E008E00u; }
__device__ __forceinline__ uint32_t fB(uint32_t w) { return (w << 3) & 0x8E008E00u; }
__device__ __forceinline__ uint32_t fC(uint32_t w) { return ((w << 9) & 0x0E000E00u) | ((w << 2) & 0x80008000u); }
__device__ __forceinline__ uint32_t fD(uint32_t w) { return ((w << 6) & 0x0E000E00u) | ((w << 1) & 0x80008000u); }

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
// a * (s.lo, s.lo) and a * (s.hi, s.hi): written with mov.b32 {x, x} so ptxas can use HMUL2's .H0_H0 / .H1_H1
// operand selectors instead of a separate PRMT.
__device__ __forceinline__ uint32_t hmul2_lo(uint32_t a, uint32_t s) {
  uint32_t r;
  asm("{\n .reg .b16 l, h;\n .reg .b32 t;\n mov.b32 {l, h}, %2;\n mov.b32 t, {l, l};\n mul.rn.f16x2 %0, %1, t;\n}"
      : "=r"(r) : "r"(a), "r"(s));
  return r;
}
__device__ __forceinline__ uint32_t hmul2_hi(uint32_t a, uint32_t s) {
  uint32_t r;
  asm("{\n .reg .b16 l, h;\n .reg .b32 t;\n mov.b32 {l, h}, %2;\n mov.b32 t, {h, h};\n mul.rn.f16x2 %0, %1, t;\n}"
      : "=r"(r) : "r"(a), "r"(s));
  return r;
}

template <int NST, bool MX, int G = 4>
constexpr size_t smem_bytes() {
  return 2 * NST * (CODE_BYTES + sc_bytes<MX>()) + NPAD * D * 2 + 2 * NPAD * HALF * 2 + 2 * (16 * G) * 4 + 2 * NST * 8;
}

// MXFP4: two E8M0 bytes -> f16x2 bits (e - B) << 10 with one PRMT and one IMAD; B (per tensor, from Python) keeps
// the FP16 exponent field in [1, 30]. negB = -(B * 1024 * 65537) mod 2^32 (no borrow across lanes since e - B >= 1).
__device__ __forceinline__ uint32_t mx_lo(uint32_t x, uint32_t negB) { return __byte_perm(x, 0, 0x4240) * 1024u + negB; }
__device__ __forceinline__ uint32_t mx_hi(uint32_t x, uint32_t negB) { return __byte_perm(x, 0, 0x4341) * 1024u + negB; }

// MX = false: NVFP4 (E4M3 per 16, FP32 tensor scale). MX = true: MXFP4 (E8M0 per 32).
//   MXFP4 K scales per stage (1 KB): [wg 2][sub 2][warp 4][g 8][k 2][(grp 2k, row g), (grp 2k+1, row g),
//   (grp 2k, row g+8), (grp 2k+1, row g+8)] (groups of 32 dims = code word i covers k-steps 2i, 2i+1 = group i).
//   MXFP4 V scales per stage (1 KB): [wg 2][kh 2][mt 2][warp pair 2][t 4][kk 4][(tok 2t), (tok 2t+8), (tok 2t+1),
//   (tok 2t+9)], group 2mt + warp / 2.
// DIAG (11-11 diagnosis of the H200 FP4 plateau): 0 normal; 1 no reconstruction (raw code words go to the tensor
// cores); 2 no matrix multiplies (reconstruction kept alive by XOR-folding it into the accumulators).
// Paged-only diagnostics (11-11, valid only for in-order pages at full length): 3 page ids computed as
// b * max_pages + j instead of read from the block table; 5 the same with a head-major pool (HKV, num_pages,
// PAGE_BYTES), num_pages passed in S, so each head's pages are contiguous.
// PAGED (11-11): the cache is a pool of 64-token pages, each holding one (K codes, K scales, V codes, V scales) block
// per kv head in exactly the order one quarter of a stage has in shared memory, addressed through a block table, with
// a length per sequence; the last stage of a sequence is masked (scores -inf, V register halves zeroed so stale or
// uninitialized page bytes cannot inject NaN). Splits with no tokens write O = 0, l = 0, m = -inf.
// G (11-11): query heads per kv head, 4 or 8. The QK and PV wgmmas have N = 8 query rows either way; with G = 4 rows
// 4..7 are zero padding and lanes t = 2, 3 take row g + 8 of heads 0..3 (shuffle), with G = 8 every lane owns heads
// 2t, 2t + 1 over both rows g and g + 8, so the softmax work per lane doubles and the tensor-core work is unchanged.
template <int NST, bool MX, bool ILP, int DIAG = 0, bool PAGED = false, int G = 4>
__global__ void __launch_bounds__(NTHREADS, 2) decode_nvfp4_kernel(const uint8_t* __restrict__ kp,   // (B,HKV,S,64)
                                                                const uint8_t* __restrict__ ksc,  // (B,HKV,S,8)
                                                                const uint8_t* __restrict__ vp,
                                                                const uint8_t* __restrict__ vsc,
                                                                const __half* __restrict__ qh,    // (B,HKV,8,128)
                                                                float* __restrict__ out_o, float* __restrict__ out_ml,
                                                                int S, int HKV, float score_mul, float out_mul,
                                                                uint32_t negbk, uint32_t negbv,
                                                                const uint8_t* __restrict__ pages,
                                                                const int* __restrict__ block_table,
                                                                const int* __restrict__ seq_lens, int max_pages) {
  constexpr int SC_BYTES = sc_bytes<MX>();
  constexpr int SCQ = SC_BYTES / 4, PAGE_BYTES = 2 * (CODE_BYTES / 4 + SCQ);  // per page per kv head
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* s_k = smem;
  uint8_t* s_v = s_k + NST * CODE_BYTES;
  uint8_t* s_ks = s_v + NST * CODE_BYTES;
  uint8_t* s_vs = s_ks + NST * SC_BYTES;
  __half* s_q = reinterpret_cast<__half*>(s_vs + NST * SC_BYTES);
  __half* s_p = s_q + NPAD * D;
  float* s_wmax = reinterpret_cast<float*>(s_p + 2 * NPAD * HALF);
  float* s_lc = s_wmax + 16 * G;
  uint64_t* full = reinterpret_cast<uint64_t*>(s_lc + 16 * G);
  uint64_t* empty = full + NST;

  const int split = blockIdx.x, h = blockIdx.y, b = blockIdx.z, splits = gridDim.x;
  const int tok_per = S / splits, tok0 = split * tok_per;
  int niter = tok_per / STAGE, len = S, g0 = tok0 / STAGE;  // g0: first stage of this split within the sequence
  const size_t bh = (size_t)b * HKV + h;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  if constexpr (PAGED) {
    len = seq_lens[b];
    const int nstb = (len + STAGE - 1) / STAGE, per = (nstb + splits - 1) / splits;
    g0 = split * per;
    niter = max(0, min(nstb, g0 + per) - g0);
    if (niter == 0) {  // nothing for this split: a neutral partial for the merge
      float* o = out_o + ((bh * splits + split) * G) * D;
      for (int i = tid; i < G * D; i += NTHREADS) o[i] = 0.f;
      if (tid < G) {
        float* ml = out_ml + ((bh * splits + split) * G + tid) * 2;
        ml[0] = -INFINITY, ml[1] = 0.f;
      }
      return;
    }
  }

  for (int i = tid; i < NPAD * D; i += NTHREADS) {
    const int n = i / D, dim = i % D;
    s_q[(dim / 8) * 64 + n * 8 + dim % 8] = qh[bh * NPAD * D + i];
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
      int nxt[4] = {0, 0, 0, 0};  // PAGED: page ids of the next stage
      if constexpr (PAGED) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int pj = g0 * 4 + j;
          nxt[j] = pj * 64 < len ? (DIAG >= 3 ? b * max_pages + pj : block_table[(size_t)b * max_pages + pj]) : 0;
        }
      }
      for (int it = 0; it < niter; ++it) {
        const int s = it % NST;
        mbar_wait_sleep(&empty[s], ((it / NST) & 1) ^ 1);
        if constexpr (PAGED) {
          // page-major stage layout (11-11 v2): page j of slot s at s_k + (4 s + j) PAGE_BYTES, one copy per page
          // (16 small copies per stage were 12-17% slower than dense); page ids are fetched one stage ahead
          const int pg0 = (g0 + it) * 4, nvalid = min(4, (len - pg0 * 64 + 63) / 64);
          int ids[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) ids[j] = nxt[j];
          if (it + 1 < niter) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const int pj = pg0 + 4 + j;
              nxt[j] = pj * 64 < len ? (DIAG >= 3 ? b * max_pages + pj : block_table[(size_t)b * max_pages + pj]) : 0;
            }
          }
          mbar_expect_tx(&full[s], nvalid * PAGE_BYTES);
          for (int j = 0; j < nvalid; ++j)
            bulk_g2s(s_k + (s * 4 + j) * PAGE_BYTES,
                     pages + (DIAG == 5 ? (size_t)h * S + ids[j] : (size_t)ids[j] * HKV + h) * PAGE_BYTES, PAGE_BYTES,
                     &full[s]);
          continue;
        }
        mbar_expect_tx(&full[s], 2 * (CODE_BYTES + SC_BYTES));
        const size_t r = row0 + (size_t)it * STAGE;
        bulk_g2s(s_k + s * CODE_BYTES, kp + r * ROW_BYTES, CODE_BYTES, &full[s]);
        bulk_g2s(s_ks + s * SC_BYTES, ksc + r * (SC_BYTES / STAGE), SC_BYTES, &full[s]);
        bulk_g2s(s_v + s * CODE_BYTES, vp + r * ROW_BYTES, CODE_BYTES, &full[s]);
        bulk_g2s(s_vs + s * SC_BYTES, vsc + r * (SC_BYTES / STAGE), SC_BYTES, &full[s]);
      }
    }
    return;
  }

  const int wg = warp / 4, wi = warp % 4, g = lane / 4, t = lane % 4, hp = t & 1;
  const bool hi = t >= 2;
  constexpr int NR = G == 8 ? 2 : 1;            // token rows per lane in the softmax (g and g + 8 when G = 8)
  const int hb = G == 8 ? 2 * t : 2 * hp;        // this lane's two query heads: hb, hb + 1
  constexpr int RED0 = G == 8 ? 4 : 2;          // first xor offset of the over-tokens reductions
  const uint64_t desc_q = make_desc(smem_u32(s_q));
  __half* Pw = s_p + wg * NPAD * HALF;
  const uint64_t desc_p = make_desc(smem_u32(Pw));
  float ma = -INFINITY, mb = -INFINITY, la = 0.f, lb = 0.f;
  float O[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

  // 11-11: the stage loop is unrolled by NST so the stage index s is a compile-time constant: every shared-memory
  // address becomes a per-thread base plus an immediate, and the modulo / parity arithmetic disappears.
  // 11-11 paged v4: loop control on niter / len (from seq_lens in global memory) goes through warp votes, which ptxas
  // knows are uniform. Alone this did not change the SASS (slot 1 still 64 R2UR); v5 below is what fixed it.
  auto uni = [](bool c) { return PAGED ? __all_sync(0xffffffffu, c) : c; };
  // 11-11 paged v5: the last, partial stage (tail) runs once after the loop, so the hot loop holds exactly the NST
  // full-stage bodies, as in dense (v4's loop held 2 NST bodies; slot 1 lost its uniform-register descriptors: 870
  // instructions per stage vs 764 dense, 6.6% more, matching the 6.5% slowdown).
  int nfull = niter;
  bool has_tail = false;
  if constexpr (PAGED) {
    has_tail = uni((g0 + niter) * STAGE > len);
    nfull = niter - (has_tail ? 1 : 0);
  }
  // 11-11 paged v3: the stage body is compiled twice. Full stages get no masking code at all; only the last,
  // partial stage of a sequence (tail) masks scores, zeroes V halves and guards m = -inf. A runtime `tail` tested
  // inside the unrolled reconstruction split it into small blocks and cost 10-18% (in-order and shuffled pages
  // were equally slow, so page placement was not the cause).
  auto stage_body = [&](auto tail_c, const int s, const int it) {
    const int tbase = (g0 + it) * STAGE;  // first token of this stage in the sequence
    using TailT = decltype(tail_c);
    constexpr bool tail = TailT::value;

    // ---- QK
    const uint8_t* kb = s_k + s * CODE_BYTES;
    float sc[2][NR][2];
    uint32_t AQ[ILP ? 2 : 1][8][4];
    float dq[2][4];
    // one subtile's scores from its accumulator (lanes t = 2, 3 take row + 8 from lane t - 2)
    auto qk_scores = [&](int sub) {
#pragma unroll
      for (int i = 0; i < 4; ++i) fence_operand(dq[sub][i]);
      if constexpr (G == 8) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          sc[sub][r][0] = dq[sub][2 * r] * score_mul;
          sc[sub][r][1] = dq[sub][2 * r + 1] * score_mul;
          if constexpr (TailT::value) {
            if (tbase + wg * HALF + sub * 64 + wi * 16 + g + 8 * r >= len) sc[sub][r][0] = sc[sub][r][1] = -INFINITY;
          }
        }
      } else {
        const float u2 = __shfl_up_sync(0xffffffffu, dq[sub][2], 2), u3 = __shfl_up_sync(0xffffffffu, dq[sub][3], 2);
        sc[sub][0][0] = (hi ? u2 : dq[sub][0]) * score_mul;
        sc[sub][0][1] = (hi ? u3 : dq[sub][1]) * score_mul;
        if constexpr (TailT::value) {
          if (tbase + wg * HALF + sub * 64 + wi * 16 + g + (hi ? 8 : 0) >= len) sc[sub][0][0] = sc[sub][0][1] = -INFINITY;
        }
      }
    };
#pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
      uint32_t (&A)[8][4] = AQ[ILP ? sub : 0];
      const int r0 = wg * HALF + sub * 64 + wi * 16 + g;
      // PAGED: rows r0, r0 + 8 lie in page 2 wg + sub of the slot, at row (r0 % 64)
      const uint8_t* krow = PAGED ? s_k + (s * 4 + wg * 2 + sub) * PAGE_BYTES + (wi * 16 + g) * ROW_BYTES
                                  : kb + r0 * ROW_BYTES;
      const uint4 w0 = *reinterpret_cast<const uint4*>(krow + t * 16);
      const uint4 w1 = *reinterpret_cast<const uint4*>(krow + 8 * ROW_BYTES + t * 16);
      const uint8_t* kscb = PAGED ? s_k + (s * 4 + wg * 2 + sub) * PAGE_BYTES + CODE_BYTES / 4 + (wi * 8 + g) * (MX ? 8 : 16)
                                  : s_ks + s * SC_BYTES + (((wg * 2 + sub) * 4 + wi) * 8 + g) * (MX ? 8 : 16);
      const uint32_t x0[4] = {w0.x, w0.y, w0.z, w0.w}, x1[4] = {w1.x, w1.y, w1.z, w1.w};
      uint32_t sv[4];
      if constexpr (MX) {
        const uint2 sw = *reinterpret_cast<const uint2*>(kscb);
        sv[0] = mx_lo(sw.x, negbk), sv[1] = mx_hi(sw.x, negbk), sv[2] = mx_lo(sw.y, negbk), sv[3] = mx_hi(sw.y, negbk);
      } else {
        const uint4 sw = *reinterpret_cast<const uint4*>(kscb);
        sv[0] = sw.x, sv[1] = sw.y, sv[2] = sw.z, sv[3] = sw.w;
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t a = x0[i], c = x1[i];
        // (row g, row g+8) scales for k-steps 2i and 2i+1 (NVFP4: groups 2i, 2i+1; MXFP4: group i for both)
        const uint32_t s0 = MX ? sv[i] : e4m3x2_f16x2(sv[i]), s1 = MX ? sv[i] : e4m3x2_f16x2(sv[i] >> 16);
        if constexpr (DIAG == 1) {
          A[2 * i][0] = a, A[2 * i][1] = c, A[2 * i][2] = a ^ s0, A[2 * i][3] = c ^ s0;
          A[2 * i + 1][0] = a ^ s1, A[2 * i + 1][1] = c ^ s1, A[2 * i + 1][2] = a + c, A[2 * i + 1][3] = c - a;
        } else {
        A[2 * i][0] = hmul2_lo(fA(a), s0);  // row g
        A[2 * i][1] = hmul2_hi(fA(c), s0);  // row g + 8
        A[2 * i][2] = hmul2_lo(fB(a), s0);
        A[2 * i][3] = hmul2_hi(fB(c), s0);
        A[2 * i + 1][0] = hmul2_lo(fC(a), s1);
        A[2 * i + 1][1] = hmul2_hi(fC(c), s1);
        A[2 * i + 1][2] = hmul2_lo(fD(a), s1);
        A[2 * i + 1][3] = hmul2_hi(fD(c), s1);
        }
      }
      if constexpr (DIAG == 2) {
        uint32_t xr = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) xr ^= A[j][0] ^ A[j][1] ^ A[j][2] ^ A[j][3];
#pragma unroll
        for (int i = 0; i < 4; ++i) dq[sub][i] = (float)((xr >> i) & 7) * 1e-3f;
      } else {
      wg_fence();
#pragma unroll
      for (int j = 0; j < 8; ++j) mma_rs(dq[sub], A[j], desc_q + 16 * j, j > 0);
      }
      wg_commit();
      if constexpr (!ILP) {
        wg_wait<0>();
        qk_scores(sub);
      }
    }
    if constexpr (ILP) {  // 11-11: subtile 1 was reconstructed while subtile 0's wgmmas ran
      wg_wait<1>();
      qk_scores(0);
      wg_wait<0>();
      qk_scores(1);
    }

    // ---- per-warpgroup max per head
    float xa = fmaxf(sc[0][0][0], sc[1][0][0]), xb = fmaxf(sc[0][0][1], sc[1][0][1]);
    if constexpr (G == 8) {
      xa = fmaxf(xa, fmaxf(sc[0][1][0], sc[1][1][0]));
      xb = fmaxf(xb, fmaxf(sc[0][1][1], sc[1][1][1]));
    }
#pragma unroll
    for (int off = RED0; off < 32; off <<= 1) {
      xa = fmaxf(xa, __shfl_xor_sync(0xffffffffu, xa, off));
      xb = fmaxf(xb, __shfl_xor_sync(0xffffffffu, xb, off));
    }
    float* wm = s_wmax + (((NST % 2 == 0) ? (s & 1) : (it & 1)) * 2 + wg) * (4 * G);
    if (g == 0 && (G == 8 || t < 2)) wm[wi * G + hb] = xa, wm[wi * G + hb + 1] = xb;
    named_bar(2 + wg, 128);
    xa = fmaxf(fmaxf(wm[hb], wm[G + hb]), fmaxf(wm[2 * G + hb], wm[3 * G + hb]));
    xb = fmaxf(fmaxf(wm[hb + 1], wm[G + hb + 1]), fmaxf(wm[2 * G + hb + 1], wm[3 * G + hb + 1]));
    const float mna = fmaxf(ma, xa), mnb = fmaxf(mb, xb);
    float aa = exp2f(ma - mna), ab = exp2f(mb - mnb);
    if constexpr (tail) {  // a warpgroup whose tokens are all masked so far keeps m = -inf (only in the last stage)
      if (mna == -INFINITY) aa = 1.f;
      if (mnb == -INFINITY) ab = 1.f;
    }
    ma = mna, mb = mnb;

    // ---- P (fp16) into shared memory
    float pla = 0.f, plb = 0.f;
#pragma unroll
    for (int sub = 0; sub < 2; ++sub)
#pragma unroll
    for (int r = 0; r < NR; ++r) {
      const int tw = sub * 64 + wi * 16 + g + (G == 8 ? 8 * r : (hi ? 8 : 0));
      // P is scaled by 2^P_SHIFT (11-11): with a dominant sink most p are ~1e-4, and the INT4 path's P' = p s / 16 fell
      // into FP16 subnormals (real-KV check: 3e-3 error at 8192 tokens per CTA). l and O scale together, so O / l and
      // the split merge are unchanged; P' stays <= 2^10 s (fine for V scales < 64).
      float pa = exp2f(sc[sub][r][0] - ma + P_SHIFT), pb = exp2f(sc[sub][r][1] - mb + P_SHIFT);
      if constexpr (tail) {
        if (sc[sub][r][0] == -INFINITY) pa = 0.f;
        if (sc[sub][r][1] == -INFINITY) pb = 0.f;
      }
      const int o = (tw >> 3) * 64 + (tw & 7);
      Pw[o + hb * 8] = __float2half_rn(pa);
      Pw[o + (hb + 1) * 8] = __float2half_rn(pb);
      pla += pa, plb += pb;
    }
    la = la * aa + pla, lb = lb * ab + plb;
    fence_proxy_async();
    named_bar(2 + wg, 128);

    // ---- PV
    float Os[2][4];
    const uint8_t* vb = s_v + s * CODE_BYTES + wg * (CODE_BYTES / 2);
    const uint8_t* vsb = s_vs + s * SC_BYTES + wg * (SC_BYTES / 2);
    uint32_t AV[ILP ? 2 : 1][4][2][4];
#pragma unroll
    for (int kh = 0; kh < 2; ++kh) {
      uint32_t (&A)[4][2][4] = AV[ILP ? kh : 0];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        // PAGED: V block (wg, kh) is page 2 wg + kh of the slot
        const uint8_t* vpg = s_k + (s * 4 + wg * 2 + kh) * PAGE_BYTES;
        const uint4 w = *reinterpret_cast<const uint4*>(
            PAGED ? vpg + CODE_BYTES / 4 + SCQ + (((mt * 4 + wi) * 32 + lane) * 16)
                  : vb + ((((kh * 2 + mt) * 4 + wi) * 32 + lane) * 16));
        const int vsi = MX ? ((((kh * 2 + mt) * 2 + wi / 2) * 4 + t) * 16) : ((((kh * 2 + mt) * 4 + wi) * 4 + t) * 16);
        const int vsp = MX ? (((mt * 2 + wi / 2) * 4 + t) * 16) : (((mt * 4 + wi) * 4 + t) * 16);
        const uint4 vs = *reinterpret_cast<const uint4*>(PAGED ? vpg + CODE_BYTES / 2 + SCQ + vsp : vsb + vsi);
        const uint32_t x[4] = {w.x, w.y, w.z, w.w}, sv[4] = {vs.x, vs.y, vs.z, vs.w};
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          // tokens (2t, 2t+1) and (2t+8, 2t+9)
          const uint32_t sA = MX ? mx_lo(sv[kk], negbv) : e4m3x2_f16x2(sv[kk]);
          const uint32_t sB = MX ? mx_hi(sv[kk], negbv) : e4m3x2_f16x2(sv[kk] >> 16);
          if constexpr (DIAG == 1) {
            A[kk][mt][0] = x[kk], A[kk][mt][1] = x[kk] ^ sA, A[kk][mt][2] = x[kk] ^ sB, A[kk][mt][3] = sA + sB;
          } else {
          A[kk][mt][0] = hmul2(fA(x[kk]), sA);
          A[kk][mt][1] = hmul2(fB(x[kk]), sA);
          A[kk][mt][2] = hmul2(fC(x[kk]), sB);
          A[kk][mt][3] = hmul2(fD(x[kk]), sB);
          }
          if constexpr (tail) {  // zero the halves of tokens past the end (their page bytes may be stale or NaN)
            const int tb = tbase + wg * HALF + 16 * (4 * kh + kk) + 2 * t;
            const uint32_t m01 = (tb < len ? 0x0000FFFFu : 0u) | (tb + 1 < len ? 0xFFFF0000u : 0u);
            const uint32_t m23 = (tb + 8 < len ? 0x0000FFFFu : 0u) | (tb + 9 < len ? 0xFFFF0000u : 0u);
            A[kk][mt][0] &= m01, A[kk][mt][1] &= m01, A[kk][mt][2] &= m23, A[kk][mt][3] &= m23;
          }
        }
      }
      if constexpr (DIAG == 2) {
        uint32_t xr = 0;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) xr ^= A[kk][mt][0] ^ A[kk][mt][1] ^ A[kk][mt][2] ^ A[kk][mt][3];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
          for (int i = 0; i < 4; ++i) Os[mt][i] = (kh ? Os[mt][i] : 0.f) + (float)((xr >> (i + mt)) & 7) * 1e-3f;
      } else {
      wg_fence();
#pragma unroll
      for (int kk = 0; kk < 4; ++kk)
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) mma_rs(Os[mt], A[kk][mt], desc_p + 16 * (4 * kh + kk), (kh | kk) != 0);
      }
      wg_commit();
      if constexpr (!ILP) wg_wait<0>();
    }
    if constexpr (ILP) wg_wait<0>();
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
  for (int it0 = 0; !uni(it0 >= nfull); it0 += NST) {
    const uint32_t ph = (it0 / NST) & 1;
#pragma unroll
    for (int s = 0; s < NST; ++s) {
      const int it = it0 + s;
      if (uni(it >= nfull)) break;
      mbar_wait(&full[s], ph);
      stage_body(std::false_type{}, s, it);
    }
  }
  if constexpr (PAGED) {
    if (has_tail) {
      const int it = nfull, s = it % NST;
      mbar_wait(&full[s], (it / NST) & 1);
      stage_body(std::true_type{}, s, it);
    }
  }

  // ---- l per head over the warpgroup; merge the two warpgroups; write
#pragma unroll
  for (int off = RED0; off < 32; off <<= 1) {
    la += __shfl_xor_sync(0xffffffffu, la, off), lb += __shfl_xor_sync(0xffffffffu, lb, off);
  }
  float* lc = s_lc + wg * (8 * G);
  if (g == 0 && (G == 8 || t < 2)) lc[(wi * G + hb) * 2] = la, lc[(wi * G + hb + 1) * 2] = lb;
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
  if (wg == 0 && (G == 8 || t < 2)) {
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

template <int NST, bool MX, bool ILP, int DIAG = 0, int G = 4>
void launch(const torch::Tensor& kp, const torch::Tensor& ksc, const torch::Tensor& vp, const torch::Tensor& vsc,
            const torch::Tensor& qh, torch::Tensor& out_o, torch::Tensor& out_ml, int splits, float score_mul,
            float out_mul, uint32_t negbk, uint32_t negbv) {
  const int B = kp.size(0), HKV = kp.size(1), S = kp.size(2);
  constexpr size_t smem = smem_bytes<NST, MX, G>();
  auto kern = decode_nvfp4_kernel<NST, MX, ILP, DIAG, false, G>;
  static bool configured = false;
  if (!configured) {
    TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess);
    configured = true;
  }
  dim3 grid(splits, HKV, B);
  kern<<<grid, NTHREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint8_t*>(kp.data_ptr()), ksc.data_ptr<uint8_t>(),
      reinterpret_cast<const uint8_t*>(vp.data_ptr()), vsc.data_ptr<uint8_t>(),
      reinterpret_cast<const __half*>(qh.data_ptr()), out_o.data_ptr<float>(), out_ml.data_ptr<float>(), S, HKV,
      score_mul, out_mul, negbk, negbv, nullptr, nullptr, nullptr, 0);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "decode_fp4 launch failed");
}

template <bool MX, int DIAG = 0, int G = 4>
void launch_paged(const torch::Tensor& pages, const torch::Tensor& block_table, const torch::Tensor& seq_lens,
                  const torch::Tensor& qh, torch::Tensor& out_o, torch::Tensor& out_ml, int splits, float score_mul,
                  float out_mul, uint32_t negbk, uint32_t negbv) {
  constexpr int NST = 2;
  const int B = seq_lens.size(0), max_pages = block_table.size(1);
  const int HKV = DIAG == 5 ? pages.size(0) : pages.size(1), npool = DIAG == 5 ? pages.size(1) : 0;
  constexpr size_t smem = smem_bytes<NST, MX, G>();
  auto kern = decode_nvfp4_kernel<NST, MX, false, DIAG, true, G>;
  static bool configured = false;
  if (!configured) {
    TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess);
    configured = true;
  }
  dim3 grid(splits, HKV, B);
  kern<<<grid, NTHREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
      nullptr, nullptr, nullptr, nullptr, reinterpret_cast<const __half*>(qh.data_ptr()), out_o.data_ptr<float>(),
      out_ml.data_ptr<float>(), npool, HKV, score_mul, out_mul, negbk, negbv, pages.data_ptr<uint8_t>(),
      block_table.data_ptr<int>(), seq_lens.data_ptr<int>(), max_pages);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "decode_fp4_paged launch failed");
}

}  // namespace

// kp, vp (B, HKV, S, 16) int32 (permuted codes); ksc, vsc (B, HKV, S, 8 or 4) uint8 (permuted E4M3 / E8M0 scales);
// qh (B, HKV, 8, 128) fp16. score_mul multiplies the raw QK accumulator into base-2 scores, out_mul the PV output
// (decode_fp4.py computes both); negbk, negbv: MXFP4 exponent rebases. Returns out_o (B, HKV, splits, 4, 128)
// (normalized per split) and out_ml (B, HKV, splits, 4, 2).
std::vector<torch::Tensor> decode_fp4(torch::Tensor kp, torch::Tensor ksc, torch::Tensor vp, torch::Tensor vsc,
                                      torch::Tensor qh, double score_mul, double out_mul, int64_t negbk, int64_t negbv,
                                      int64_t splits, int64_t nst, bool mx, bool ilp, int64_t diag, int64_t gqa) {
  for (auto* x : {&kp, &vp}) TORCH_CHECK(x->is_cuda() && x->is_contiguous() && x->dim() == 4 && x->size(3) == 16);
  for (auto* x : {&ksc, &vsc})
    TORCH_CHECK(x->is_contiguous() && x->scalar_type() == torch::kUInt8 && x->size(-1) == (mx ? 4 : 8));
  TORCH_CHECK(qh.is_contiguous() && qh.scalar_type() == torch::kHalf && qh.size(2) == NPAD && qh.size(3) == D);
  const int B = kp.size(0), HKV = kp.size(1), S = kp.size(2);
  TORCH_CHECK(S % (splits * STAGE) == 0, "S must be a multiple of splits * 256");
  TORCH_CHECK(gqa == 4 || (gqa == 8 && diag == 0 && !ilp), "gqa must be 4, or 8 (without ilp / diag)");
  auto out_o = torch::empty({B, HKV, splits, gqa, D}, kp.options().dtype(torch::kFloat));
  auto out_ml = torch::empty({B, HKV, splits, gqa, 2}, kp.options().dtype(torch::kFloat));
  const float sm = (float)score_mul, om = (float)out_mul;
  const uint32_t bk = (uint32_t)negbk, bv = (uint32_t)negbv;
  TORCH_CHECK(nst == 2, "nst must be 2 (nst 3 never won on H200 and fits only one CTA per SM)");
  TORCH_CHECK(diag == 0 || (!mx && !ilp), "diag only for NVFP4 without ILP");
  if (gqa == 8) {
    if (mx) launch<2, true, false, 0, 8>(kp, ksc, vp, vsc, qh, out_o, out_ml, splits, sm, om, bk, bv);
    else launch<2, false, false, 0, 8>(kp, ksc, vp, vsc, qh, out_o, out_ml, splits, sm, om, bk, bv);
  } else if (diag == 1) launch<2, false, false, 1>(kp, ksc, vp, vsc, qh, out_o, out_ml, splits, sm, om, bk, bv);
  else if (diag == 2) launch<2, false, false, 2>(kp, ksc, vp, vsc, qh, out_o, out_ml, splits, sm, om, bk, bv);
  else if (mx && ilp) launch<2, true, true>(kp, ksc, vp, vsc, qh, out_o, out_ml, splits, sm, om, bk, bv);
  else if (mx) launch<2, true, false>(kp, ksc, vp, vsc, qh, out_o, out_ml, splits, sm, om, bk, bv);
  else if (ilp) launch<2, false, true>(kp, ksc, vp, vsc, qh, out_o, out_ml, splits, sm, om, bk, bv);
  else launch<2, false, false>(kp, ksc, vp, vsc, qh, out_o, out_ml, splits, sm, om, bk, bv);
  return {out_o, out_ml};
}

// Paged, variable-length decode (11-11). pages (num_pages, HKV, PAGE_BYTES) uint8, each (page, head) = K codes
// (4096) | K scales (SCQ) | V codes (4096) | V scales (SCQ) in the stage-quarter layout (decode_fp4.py:
// pack_pages); block_table (B, max_pages) int32 (64-token pages); seq_lens (B) int32. Returns as decode_fp4().
std::vector<torch::Tensor> decode_fp4_paged(torch::Tensor pages, torch::Tensor block_table, torch::Tensor seq_lens,
                                            torch::Tensor qh, double score_mul, double out_mul, int64_t negbk,
                                            int64_t negbv, int64_t splits, bool mx, int64_t diag, int64_t gqa) {
  TORCH_CHECK(pages.is_cuda() && pages.is_contiguous() && pages.dim() == 3 && pages.scalar_type() == torch::kUInt8);
  TORCH_CHECK(pages.size(2) == 2 * (CODE_BYTES / 4 + (mx ? sc_bytes<true>() : sc_bytes<false>()) / 4), "page bytes");
  TORCH_CHECK(block_table.is_contiguous() && block_table.scalar_type() == torch::kInt);
  TORCH_CHECK(seq_lens.is_contiguous() && seq_lens.scalar_type() == torch::kInt);
  TORCH_CHECK(qh.is_contiguous() && qh.scalar_type() == torch::kHalf && qh.size(2) == NPAD && qh.size(3) == D);
  // 11-22: paged DIAG 1 / 2 (no reconstruction / no MMAs; NVFP4, G = 4; wrong outputs by design) for the
  // shared-prefix diagnosis (docs/shared_prefix_log.md). The diag-0 instantiation is unchanged.
  TORCH_CHECK(diag == 0 || diag == 1 || diag == 2 || diag == 3 || diag == 5, "paged diag must be 0, 1, 2, 3 or 5");
  TORCH_CHECK((diag != 1 && diag != 2) || !mx, "paged diag 1 / 2 only for NVFP4");
  const int B = seq_lens.size(0), HKV = diag == 5 ? pages.size(0) : pages.size(1);
  TORCH_CHECK(gqa == 4 || (gqa == 8 && diag == 0), "gqa must be 4, or 8 (without diag)");
  auto out_o = torch::empty({B, HKV, splits, gqa, D}, qh.options().dtype(torch::kFloat));
  auto out_ml = torch::empty({B, HKV, splits, gqa, 2}, qh.options().dtype(torch::kFloat));
  const float sm = (float)score_mul, om = (float)out_mul;
  const uint32_t bk = (uint32_t)negbk, bv = (uint32_t)negbv;
#define KV4_PAGED(MXV, DG) launch_paged<MXV, DG>(pages, block_table, seq_lens, qh, out_o, out_ml, splits, sm, om, bk, bv)
  if (gqa == 8) { if (mx) launch_paged<true, 0, 8>(pages, block_table, seq_lens, qh, out_o, out_ml, splits, sm, om, bk, bv);
                  else launch_paged<false, 0, 8>(pages, block_table, seq_lens, qh, out_o, out_ml, splits, sm, om, bk, bv); }
  else if (mx) { if (diag == 3) KV4_PAGED(true, 3); else if (diag == 5) KV4_PAGED(true, 5); else KV4_PAGED(true, 0); }
  else { if (diag == 3) KV4_PAGED(false, 3); else if (diag == 5) KV4_PAGED(false, 5);
         else if (diag == 1) KV4_PAGED(false, 1); else if (diag == 2) KV4_PAGED(false, 2); else KV4_PAGED(false, 0); }
#undef KV4_PAGED
  return {out_o, out_ml};
}

// NVFP4 convenience: tensor scales ts_k, ts_v.
std::vector<torch::Tensor> decode_nvfp4(torch::Tensor kp, torch::Tensor ksc, torch::Tensor vp, torch::Tensor vsc,
                                        torch::Tensor qh, double softmax_scale, double ts_k, double ts_v,
                                        int64_t splits, int64_t nst) {
  return decode_fp4(kp, ksc, vp, vsc, qh, softmax_scale * 1.4426950408889634 * ts_k * 16384.0, ts_v * 16384.0, 0, 0,
                    splits, nst, false, false, 0, 4);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode_nvfp4", &decode_nvfp4, "Hopper NVFP4 KV decode (E2M1 bit placement, E4M3 scales via HMUL2)");
  m.def("decode_fp4_paged", &decode_fp4_paged, "paged, variable-length NVFP4 / MXFP4 decode", py::arg("pages"),
        py::arg("block_table"), py::arg("seq_lens"), py::arg("qh"), py::arg("score_mul"), py::arg("out_mul"),
        py::arg("negbk"), py::arg("negbv"), py::arg("splits"), py::arg("mx"), py::arg("diag") = 0, py::arg("gqa") = 4);
  m.def("decode_fp4", &decode_fp4, "Hopper NVFP4 / MXFP4 KV decode", py::arg("kp"), py::arg("ksc"), py::arg("vp"),
        py::arg("vsc"), py::arg("qh"), py::arg("score_mul"), py::arg("out_mul"), py::arg("negbk"), py::arg("negbv"),
        py::arg("splits"), py::arg("nst"), py::arg("mx"), py::arg("ilp") = false, py::arg("diag") = 0,
        py::arg("gqa") = 4);
}
