// rebuild tag: c2 v1 (11-22), k8 v3 G8 register fix (11-21), k8 v2 paged (11-21)
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
//
// 11-21 v2 (C1 in the vLLM plugin): template <PAGED, G>.
//  PAGED: pages of 9216 bytes per (page, kv head) written by store_fp4.cu's K8 store: K codes 4096 (pack_k8 rows) |
//    K scales 512 (bytes 0..255: E4M3 per 32 dims in the MXFP4 K scale order of one page, rest unused) | V codes
//    4096 | V scales 512 (NVFP4), i.e. decode_fp4.cu's NVFP4 page with a different K half. Paging, split assignment
//    by sequence length, the tail stage after the loop and the warp-vote loop control are decode_fp4.cu's PAGED code.
//  G = 4 or 8 query heads per kv head: G = 8 uses every q row; lanes then own heads 2t, 2t + 1 over rows g and g + 8
//    (decode_fp4.cu's G == 8 roles); the tensor-core work is the same. G = 8 compiles a register-lean variant of
//    the QK and PV steps (PV_INTO_O, QK_CHAIN in the kernel), without which ptxas spilled O at the 96-register cap.
//  q_scale (optional, B * HKV floats): one E4M3 scale per (sequence, kv head), multiplied into score_mul per CTA
//    (prep_q8_heads below makes q8, the scales and the dequantized q in one kernel).
//
// 11-22 C2 (template flag C2, paged only; docs/c2_log.md): P V on FP8 tensor cores too. K as in C1.
//  V format: E2M1 with ONE E4M3 scale per (channel, 32 consecutive tokens of a page) times the per-layer FP32 tensor
//    scale (kv4.formats "nvfp4_t32"). PV reduces over tokens and one wgmma m64n8k32 step covers 32 tokens, so the
//    scale is constant over a k-step for each output channel (a row of O^T) and is applied in FP32 after the MMA, from
//    a zeroed accumulator per k-step (as C1 does for K; SageAttention2's "FP22" accumulator finding).
//  MMA: O^T (128 channels x 8 heads) += V^T (channels x tokens, registers, E4M3 bit placement as K) x P^T (tokens x
//    heads, shared memory, E4M3). P = exp2(s - m + 8) in (0, 256], rounded to E4M3 (no saturation; values below 2^-9
//    flush to 0); the row sum l is taken from the unrounded P (so l carries 2^8: P_SHIFT_C2).
//  Page (still 9216 bytes): K codes 4096 | K scales 512 (first 256 used, as C1) | V codes 4096 | V scales 512 (first
//    256 used: 128 channels x 2 groups; bytes 256..511 unused). V codes of page group gi (tokens 32 gi .. 32 gi + 31)
//    use C1's K word format with "dim" read as "token in the group": the word of (channel c, group gi, lane t) holds
//    tokens 4t..4t+3 (slot L, bits 7,4,3,2 of byte j) and 16+4t..16+4t+3 (slot H, bits 5,6,1,0), so k8L / k8H give the
//    A registers R0 / R2 (channel g) and R1 / R3 (channel g + 8) of wgmma's 8-bit A fragment. Channel c = 64 mt +
//    16 wi + 8 h8 + g is read by warp wi, lanes 4g..4g+3: thread (wi, lane) loads one uint4 per m-tile at byte
//    ((mt * 4 + wi) * 32 + lane) * 16 = words {(group 0, ch g), (group 0, ch g + 8), (group 1, ch g), (group 1,
//    ch g + 8)} (the same addresses as the NVFP4 V path); its scales are 8 bytes at (wi * 8 + g) * 8: byte
//    mt * 4 + 2 gi + h8 = scale of (channel 64 mt + 16 wi + 8 h8 + g, group gi). Python: decode_fp4_k8._c2_v_layout.
//  Sinks: with the plugin's BF16 sinks, the C2 kernel masks the first nsk tokens of every sequence whose sink-pool
//    slot is valid (same test as merge.cu merge_sink_pool) and merge_sink_pool_c2 below adds the BF16 sinks. (C1's way,
//    subtracting the 4-bit sink copies in the merge, is not exact with an E4M3 P: the kernel rounds P against the
//    running max of the first stage, which the merge does not know; the error would be up to 1/16 of the sink weight.)
#include <type_traits>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "common.cuh"

namespace {
using namespace kv4cu;

constexpr float P_SHIFT = 10.f;
constexpr float P_SHIFT_C2 = 8.f;  // C2: P = exp2(s - m + 8) in (0, 256] before its E4M3 rounding
constexpr int NSK = 4;             // C2: the plugin's sink tokens (merge.cu NSK)
constexpr int D = 128;
constexpr int ROW_BYTES = 64;
constexpr int STAGE = 256;
constexpr int HALF = 128;
constexpr int NCWARPS = 8;
constexpr int NTHREADS = NCWARPS * 32 + 32;
constexpr int NPAD = 8;
constexpr int NST = 2;
constexpr int CODE_BYTES = STAGE * ROW_BYTES;  // 16 KB
constexpr int KSC_BYTES = STAGE * 4;           // E4M3 per 32 dims
constexpr int VSC_BYTES = STAGE * 8;           // E4M3 per 16 values (NVFP4 V)
constexpr int SCQ = 512;                       // PAGED: bytes of each scale region per page (K uses the first 256)
constexpr int PAGE_BYTES = 2 * (CODE_BYTES / 4 + SCQ);  // 9216 per (page, kv head)

// cache region: dense K, V, K scales, V scales per stage; PAGED 4 whole pages per stage (2 KB more)
template <bool PAGED>
constexpr size_t cache_bytes() { return PAGED ? NST * 4 * PAGE_BYTES : 2 * NST * CODE_BYTES + NST * (KSC_BYTES + VSC_BYTES); }
template <bool PAGED, int G>
constexpr size_t smem_bytes() {
  return cache_bytes<PAGED>() + NPAD * D + 2 * NPAD * HALF * 2 + 2 * (16 * G) * 4 + 2 * NST * 8;
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
// C2: two floats -> two E4M3 bytes (round to nearest even, saturating); lo in bits 0..7, hi in bits 8..15
__device__ __forceinline__ uint32_t f32x2_e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}

// D(64x8, fp32) = A(64x32, e4m3, registers) * B(32x8, e4m3, shared memory); accumulator overwritten (scale-d false)
__device__ __forceinline__ void mma_rs8(float (&d)[4], const uint32_t (&a)[4], uint64_t desc) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %9, 0;\n"
      " wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1;\n}"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(0));
}

template <bool PAGED, int G, bool C2 = false>
__global__ void __launch_bounds__(NTHREADS, 2) decode_k8_kernel(const uint8_t* __restrict__ kp,   // (B,HKV,S,64)
                                                                const uint8_t* __restrict__ ksc,  // (B,HKV,S,4)
                                                                const uint8_t* __restrict__ vp,
                                                                const uint8_t* __restrict__ vsc,  // (B,HKV,S,8)
                                                                const uint8_t* __restrict__ q8,   // (B,HKV,8,128)
                                                                const float* __restrict__ q_scale,  // (B*HKV) or null
                                                                float* __restrict__ out_o, float* __restrict__ out_ml,
                                                                int S, int HKV, float score_mul, float out_mul,
                                                                const uint8_t* __restrict__ pages,
                                                                const int* __restrict__ block_table,
                                                                const int* __restrict__ seq_lens, int max_pages,
                                                                // C2 only: the plugin's sink pool (see the header)
                                                                const int* __restrict__ sink_map, int nmap,
                                                                const int* __restrict__ sink_owner, int nsk) {
  static_assert(G == 4 || G == 8, "G = 4 or 8");
  static_assert(!C2 || PAGED, "C2 is paged only");
  // 11-21 G = 8 register fix (docs/c1_plugin_log.md, "G = 8 diagnosis"). 288 threads x 2 CTAs put 5 warps on two SM
  // sub-partitions, so every thread gets 96 registers. The G = 8 instantiation needed more and ptxas spilled the O
  // accumulators (208 B stores / 272 B loads; 8 LDL + 8 STL of O per stage): 0.97x of paged NVFP4 at G = 8. Two
  // changes, both needed, make it spill-free (1.13-1.14x of NVFP4 at 32 x 16K and 128 x 16K, job 12055287):
  //  PV_INTO_O: rescale O by alpha before the PV wgmmas and let them accumulate into O (no Os: 8 registers less);
  //  QK_CHAIN: each subtile's QK in two rounds of two k-steps (8 A + 8 accumulator registers in flight, not 16 + 16).
  // 11-22: both changes now apply to G = 4 as well (default K8V = 7, user's decision). On G = 4 they are also
  // spill-free and 2.7-3.3% faster (job 12055287); in vLLM 1.005-1.026x per step over K8V = 3 and the same
  // perplexity to four decimals (5.4858, job 12056102). Accumulating PV into a rescaled O is what FlashAttention-3
  // does (hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp: softmax.rescale_o then gemm<zero_init=false>(..., tOrO)), so
  // both G values now follow that structure. Outputs differ from K8V = 3 by about 4e-3 of the output maximum.
  // K8V is the experiment switch of bench/k8g8_variants.py: bit 0 PV_INTO_O, bit 1 QK_CHAIN, bit 2 also for G = 4;
  // 3 applies the fix to G = 8 only, 0 is the kernel before it. KV4_K8V (decode_fp4_k8.py) overrides it.
#ifndef K8V
#define K8V 7
#endif
  constexpr bool VG = G == 8 || (K8V & 4);
  constexpr bool PV_INTO_O = VG && (K8V & 1), QK_CHAIN = VG && (K8V & 2);
  // 11-22 C2: C2V is the experiment switch of the C2 PV step: bit 0 = G = 8 runs its PV wgmmas one group (k-step) per
  // round (8 A + 8 accumulator registers in flight instead of 16 + 16), bit 1 = G = 4 too. Default 0: two groups
  // (one page) per round for both. The QK side of C2 is C1's, with the K8V switch above.
#ifndef C2V
#define C2V 0
#endif
  constexpr int C2R = ((G == 8 && (C2V & 1)) || (G == 4 && (C2V & 2))) ? 1 : 2;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* s_k = smem;  // PAGED: page j of slot s at s_k + (4 s + j) PAGE_BYTES (s_v, s_ks, s_vs unused)
  uint8_t* s_v = s_k + NST * CODE_BYTES;
  uint8_t* s_ks = s_v + NST * CODE_BYTES;
  uint8_t* s_vs = s_ks + NST * KSC_BYTES;
  uint8_t* s_q = smem + cache_bytes<PAGED>();
  __half* s_p = reinterpret_cast<__half*>(s_q + NPAD * D);
  float* s_wmax = reinterpret_cast<float*>(s_p + 2 * NPAD * HALF);
  float* s_lc = s_wmax + 16 * G;
  uint64_t* full = reinterpret_cast<uint64_t*>(s_lc + 16 * G);
  uint64_t* empty = full + NST;

  const int split = blockIdx.x, h = blockIdx.y, b = blockIdx.z, splits = gridDim.x;
  const int tok_per = S / splits, tok0 = split * tok_per;
  int niter = tok_per / STAGE, len = S, g0 = tok0 / STAGE;  // g0: first stage of this split within the sequence
  const size_t bh = (size_t)b * HKV + h;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  if (q_scale) score_mul *= q_scale[bh];
  if constexpr (PAGED) {  // as decode_fp4.cu: stages of the sequence split evenly; an empty split writes a neutral partial
    len = seq_lens[b];
    const int nstb = (len + STAGE - 1) / STAGE, per = (nstb + splits - 1) / splits;
    g0 = split * per;
    niter = max(0, min(nstb, g0 + per) - g0);
    if (niter == 0) {
      float* o = out_o + ((bh * splits + split) * G) * D;
      for (int i = tid; i < G * D; i += NTHREADS) o[i] = 0.f;
      if (tid < G) {
        float* ml = out_ml + ((bh * splits + split) * G + tid) * 2;
        ml[0] = -INFINITY, ml[1] = 0.f;
      }
      return;
    }
  }
  // C2: the first nskip tokens of the sequence are masked (their BF16 copies are added by merge_sink_pool_c2); only
  // the CTA that holds stage 0 (g0 == 0) has them. Validity as merge.cu merge_sink_pool: slot = sink_map[first block],
  // valid if owner[slot] is that block (no owner array: timing mode, every mapped slot is valid).
  int nskip = 0;
  if constexpr (C2) {
    if (nsk > 0 && nmap > 0 && g0 == 0) {
      const int blk = block_table[(size_t)b * max_pages];
      const int slot = (blk >= 0 && blk < nmap) ? sink_map[blk] : -1;
      if (slot >= 0 && (sink_owner == nullptr || sink_owner[slot] == blk)) nskip = nsk;
    }
  }

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
      int nxt[4] = {0, 0, 0, 0};  // PAGED: page ids of the next stage (fetched one stage ahead)
      if constexpr (PAGED) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int pj = g0 * 4 + j;
          nxt[j] = pj * 64 < len ? block_table[(size_t)b * max_pages + pj] : 0;
        }
      }
      for (int it = 0; it < niter; ++it) {
        const int s = it % NST;
        mbar_wait_sleep(&empty[s], ((it / NST) & 1) ^ 1);
        if constexpr (PAGED) {  // one bulk copy per valid page
          const int pg0 = (g0 + it) * 4, nvalid = min(4, (len - pg0 * 64 + 63) / 64);
          int ids[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) ids[j] = nxt[j];
          if (it + 1 < niter) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const int pj = pg0 + 4 + j;
              nxt[j] = pj * 64 < len ? block_table[(size_t)b * max_pages + pj] : 0;
            }
          }
          mbar_expect_tx(&full[s], nvalid * PAGE_BYTES);
          for (int j = 0; j < nvalid; ++j)
            bulk_g2s(s_k + (s * 4 + j) * PAGE_BYTES, pages + ((size_t)ids[j] * HKV + h) * PAGE_BYTES, PAGE_BYTES,
                     &full[s]);
          continue;
        }
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
  constexpr int NR = G == 8 ? 2 : 1;      // token rows per lane in the softmax (g and g + 8 when G = 8)
  const int hb = G == 8 ? 2 * t : 2 * hp;  // this lane's two query heads: hb, hb + 1
  constexpr int RED0 = G == 8 ? 4 : 2;    // first xor offset of the over-tokens reductions
  auto uni = [](bool c) { return PAGED ? __all_sync(0xffffffffu, c) : c; };
  const uint64_t desc_q = make_desc(smem_u32(s_q));
  __half* Pw = s_p + wg * NPAD * HALF;
  const uint64_t desc_p = make_desc(smem_u32(Pw));
  float ma = -INFINITY, mb = -INFINITY, la = 0.f, lb = 0.f;
  float O[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
  // masked-stage test of a token (position in the sequence): past the end, or (C2) a sink the merge adds in BF16
  auto masked_tok = [&](int tok) { return tok >= len || (C2 && tok < nskip); };

  // the stage body is compiled twice (decode_fp4.cu paged v3/v5): full stages without masking code, and the last,
  // partial stage of a sequence (tail) with masked scores, zeroed V halves and the m = -inf guard
  auto stage_body = [&](auto tail_c, const int s, const int it) {
    constexpr bool tail = decltype(tail_c)::value;
    const int tbase = (g0 + it) * STAGE;  // first token of this stage in the sequence
    // ---- QK on FP8 tensor cores
    const uint8_t* kb = s_k + s * CODE_BYTES;
    float sc[2][NR][2];
#pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
      const int r0 = wg * HALF + sub * 64 + wi * 16 + g;
      // PAGED: rows r0, r0 + 8 lie in page 2 wg + sub of the slot, at row r0 % 64; its K scales follow its K codes
      const uint8_t* pgk = s_k + (s * 4 + wg * 2 + sub) * PAGE_BYTES;
      const uint8_t* krow = PAGED ? pgk + (wi * 16 + g) * ROW_BYTES : kb + r0 * ROW_BYTES;
      const uint4 w0 = *reinterpret_cast<const uint4*>(krow + t * 16);
      const uint4 w1 = *reinterpret_cast<const uint4*>(krow + 8 * ROW_BYTES + t * 16);
      const uint2 sw = *reinterpret_cast<const uint2*>(
          PAGED ? pgk + CODE_BYTES / 4 + (wi * 8 + g) * 8 : s_ks + s * KSC_BYTES + (((wg * 2 + sub) * 4 + wi) * 8 + g) * 8);
      const uint32_t x0[4] = {w0.x, w0.y, w0.z, w0.w}, x1[4] = {w1.x, w1.y, w1.z, w1.w};
      float dq[4] = {0.f, 0.f, 0.f, 0.f};
      if constexpr (!QK_CHAIN) {
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
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        dq[0] = fmaf(sg[i], P[i][0], dq[0]);
        dq[1] = fmaf(sg[i], P[i][1], dq[1]);
        dq[2] = fmaf(s8[i], P[i][2], dq[2]);
        dq[3] = fmaf(s8[i], P[i][3], dq[3]);
      }
      } else {  // QK_CHAIN (G = 8): two rounds of two k-steps; 8 A + 8 accumulator registers in flight, not 16 + 16
#pragma unroll
      for (int ph = 0; ph < 2; ++ph) {
        uint32_t A[2][4];
        float P[2][4];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int i = 2 * ph + j;
          A[j][0] = k8L(x0[i]), A[j][1] = k8L(x1[i]), A[j][2] = k8H(x0[i]), A[j][3] = k8H(x1[i]);
          P[j][0] = P[j][1] = P[j][2] = P[j][3] = 0.f;
        }
        wg_fence();
#pragma unroll
        for (int j = 0; j < 2; ++j) mma_rs8(P[j], A[j], desc_q + 16 * (2 * ph + j));
        wg_commit();
        // (row g, groups 2ph, 2ph + 1) and (row g + 8, same)
        const uint32_t sx = ph ? sw.y : sw.x;
        const float2 a = h2f2(e4m3x2_f16x2(sx)), b = h2f2(e4m3x2_f16x2(sx >> 16));
        wg_wait<0>();
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int e = 0; e < 4; ++e) fence_operand(P[j][e]);
        dq[0] = fmaf(a.x, P[0][0], dq[0]), dq[1] = fmaf(a.x, P[0][1], dq[1]);
        dq[2] = fmaf(b.x, P[0][2], dq[2]), dq[3] = fmaf(b.x, P[0][3], dq[3]);
        dq[0] = fmaf(a.y, P[1][0], dq[0]), dq[1] = fmaf(a.y, P[1][1], dq[1]);
        dq[2] = fmaf(b.y, P[1][2], dq[2]), dq[3] = fmaf(b.y, P[1][3], dq[3]);
      }
      }
      if constexpr (G == 8) {  // every lane: heads 2t, 2t + 1 of rows g (dq 0, 1) and g + 8 (dq 2, 3)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          sc[sub][r][0] = dq[2 * r] * score_mul;
          sc[sub][r][1] = dq[2 * r + 1] * score_mul;
          if constexpr (tail) {
            if (masked_tok(tbase + r0 + 8 * r)) sc[sub][r][0] = sc[sub][r][1] = -INFINITY;
          }
        }
      } else {  // lanes t = 2, 3 take row + 8 from lane t - 2
        const float u2 = __shfl_up_sync(0xffffffffu, dq[2], 2), u3 = __shfl_up_sync(0xffffffffu, dq[3], 2);
        sc[sub][0][0] = (hi ? u2 : dq[0]) * score_mul;
        sc[sub][0][1] = (hi ? u3 : dq[1]) * score_mul;
        if constexpr (tail) {
          if (masked_tok(tbase + r0 + (hi ? 8 : 0))) sc[sub][0][0] = sc[sub][0][1] = -INFINITY;
        }
      }
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
    float* wm = s_wmax + ((s & 1) * 2 + wg) * (4 * G);
    if (g == 0 && (G == 8 || t < 2)) wm[wi * G + hb] = xa, wm[wi * G + hb + 1] = xb;
    named_bar(2 + wg, 128);
    xa = fmaxf(fmaxf(wm[hb], wm[G + hb]), fmaxf(wm[2 * G + hb], wm[3 * G + hb]));
    xb = fmaxf(fmaxf(wm[hb + 1], wm[G + hb + 1]), fmaxf(wm[2 * G + hb + 1], wm[3 * G + hb + 1]));
    const float mna = fmaxf(ma, xa), mnb = fmaxf(mb, xb);
    float aa = exp2f(ma - mna), ab = exp2f(mb - mnb);
    if constexpr (tail) {  // a warpgroup whose tokens are all masked so far keeps m = -inf
      if (mna == -INFINITY) aa = 1.f;
      if (mnb == -INFINITY) ab = 1.f;
    }
    ma = mna, mb = mnb;

    // ---- P (fp16; C2: E4M3) into shared memory
    float pla = 0.f, plb = 0.f;
#pragma unroll
    for (int sub = 0; sub < 2; ++sub)
#pragma unroll
    for (int r = 0; r < NR; ++r) {
      const int tw = sub * 64 + wi * 16 + g + (G == 8 ? 8 * r : (hi ? 8 : 0));
      constexpr float PS = C2 ? P_SHIFT_C2 : P_SHIFT;
      float pa = exp2f(sc[sub][r][0] - ma + PS), pb = exp2f(sc[sub][r][1] - mb + PS);
      if constexpr (tail) {
        if (sc[sub][r][0] == -INFINITY) pa = 0.f;
        if (sc[sub][r][1] == -INFINITY) pb = 0.f;
      }
      if constexpr (C2) {
        // the B operand of m64n8k32 e4m3, K-major: core matrix kc = tokens 16 kc .. 16 kc + 15 of the 8 heads at
        // byte 128 kc, head n at + 16 n (the layout of q in the QK step); one k-step of 32 tokens = +256 bytes
        uint8_t* P8 = reinterpret_cast<uint8_t*>(Pw);
        const uint32_t pk = f32x2_e4m3x2(pa, pb);
        const int o = (tw >> 4) * 128 + (tw & 15);
        P8[o + hb * 16] = (uint8_t)pk;
        P8[o + (hb + 1) * 16] = (uint8_t)(pk >> 8);
      } else {
        const int o = (tw >> 3) * 64 + (tw & 7);
        Pw[o + hb * 8] = __float2half_rn(pa);
        Pw[o + (hb + 1) * 8] = __float2half_rn(pb);
      }
      pla += pa, plb += pb;  // C2: the row sum of the unrounded P
    }
    la = la * aa + pla, lb = lb * ab + plb;
    fence_proxy_async();
    named_bar(2 + wg, 128);

    if constexpr (C2) {
      // ---- PV on FP8 tensor cores (C2). O is rescaled first; each k-step (one 32-token group of a page) runs from a
      // zeroed accumulator and is added to O in FP32 with its per-channel V scale.
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) O[mt][0] *= aa, O[mt][1] *= ab, O[mt][2] *= aa, O[mt][3] *= ab;
#pragma unroll
      for (int kh = 0; kh < 2; ++kh) {  // page 2 wg + kh of the slot = this warpgroup's tokens 64 kh .. 64 kh + 63
        const uint8_t* vpg = s_k + (s * 4 + wg * 2 + kh) * PAGE_BYTES;
        const uint2 sv = *reinterpret_cast<const uint2*>(vpg + CODE_BYTES / 2 + SCQ + (wi * 8 + g) * 8);
        uint4 w[2];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
          w[mt] = *reinterpret_cast<const uint4*>(vpg + CODE_BYTES / 4 + SCQ + ((mt * 4 + wi) * 32 + lane) * 16);
#pragma unroll
        for (int gq = 0; gq < 2; gq += C2R) {  // C2R groups per round of wgmmas (2: both groups of the page)
          uint32_t A[C2R][2][4];
          float Pp[C2R][2][4];
#pragma unroll
          for (int j = 0; j < C2R; ++j)
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
              // group gq + j: words (ch g) = .x / .z, (ch g + 8) = .y / .w
              const uint32_t xg = (gq + j) ? w[mt].z : w[mt].x, x8 = (gq + j) ? w[mt].w : w[mt].y;
              A[j][mt][0] = k8L(xg), A[j][mt][1] = k8L(x8), A[j][mt][2] = k8H(xg), A[j][mt][3] = k8H(x8);
              Pp[j][mt][0] = Pp[j][mt][1] = Pp[j][mt][2] = Pp[j][mt][3] = 0.f;
            }
          wg_fence();
#pragma unroll
          for (int j = 0; j < C2R; ++j)
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) mma_rs8(Pp[j][mt], A[j][mt], desc_p + 16 * (2 * kh + gq + j));
          wg_commit();
          // scales of (ch g, ch g + 8) for each group and m-tile, converted while the wgmmas run
          float2 f[C2R][2];
#pragma unroll
          for (int j = 0; j < C2R; ++j) {
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) f[j][mt] = h2f2(e4m3x2_f16x2((mt ? sv.y : sv.x) >> (16 * (gq + j))));
            if constexpr (tail) {  // a group that starts past the end: its scale bytes are stale (maybe E4M3 NaN)
              if (tbase + wg * HALF + 64 * kh + 32 * (gq + j) >= len) f[j][0] = f[j][1] = make_float2(0.f, 0.f);
            }
          }
          wg_wait<0>();
#pragma unroll
          for (int j = 0; j < C2R; ++j)
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
              for (int i = 0; i < 4; ++i) fence_operand(Pp[j][mt][i]);
              O[mt][0] = fmaf(f[j][mt].x, Pp[j][mt][0], O[mt][0]);
              O[mt][1] = fmaf(f[j][mt].x, Pp[j][mt][1], O[mt][1]);
              O[mt][2] = fmaf(f[j][mt].y, Pp[j][mt][2], O[mt][2]);
              O[mt][3] = fmaf(f[j][mt].y, Pp[j][mt][3], O[mt][3]);
            }
        }
      }
    } else {
    // ---- PV (decode_fp4.cu's NVFP4 path)
    float Os[2][4];
    if constexpr (PV_INTO_O) {  // G = 8: rescale O first, then the PV wgmmas accumulate straight into O (no Os)
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) O[mt][0] *= aa, O[mt][1] *= ab, O[mt][2] *= aa, O[mt][3] *= ab;
    }
    const uint8_t* vb = s_v + s * CODE_BYTES + wg * (CODE_BYTES / 2);
    const uint8_t* vsb = s_vs + s * VSC_BYTES + wg * (VSC_BYTES / 2);
#pragma unroll
    for (int kh = 0; kh < 2; ++kh) {
      uint32_t A[4][2][4];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        // PAGED: V block (wg, kh) is page 2 wg + kh of the slot
        const uint8_t* vpg = s_k + (s * 4 + wg * 2 + kh) * PAGE_BYTES;
        const uint4 w = *reinterpret_cast<const uint4*>(
            PAGED ? vpg + CODE_BYTES / 4 + SCQ + (((mt * 4 + wi) * 32 + lane) * 16)
                  : vb + ((((kh * 2 + mt) * 4 + wi) * 32 + lane) * 16));
        const uint4 vs = *reinterpret_cast<const uint4*>(
            PAGED ? vpg + CODE_BYTES / 2 + SCQ + (((mt * 4 + wi) * 4 + t) * 16)
                  : vsb + ((((kh * 2 + mt) * 4 + wi) * 4 + t) * 16));
        const uint32_t x[4] = {w.x, w.y, w.z, w.w}, sv[4] = {vs.x, vs.y, vs.z, vs.w};
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          const uint32_t sA = e4m3x2_f16x2(sv[kk]), sB = e4m3x2_f16x2(sv[kk] >> 16);
          A[kk][mt][0] = hmul2(fA(x[kk]), sA);
          A[kk][mt][1] = hmul2(fB(x[kk]), sA);
          A[kk][mt][2] = hmul2(fC(x[kk]), sB);
          A[kk][mt][3] = hmul2(fD(x[kk]), sB);
          if constexpr (tail) {  // zero the halves of tokens past the end (their page bytes may be stale or NaN)
            const int tb = tbase + wg * HALF + 16 * (4 * kh + kk) + 2 * t;
            const uint32_t m01 = (tb < len ? 0x0000FFFFu : 0u) | (tb + 1 < len ? 0xFFFF0000u : 0u);
            const uint32_t m23 = (tb + 8 < len ? 0x0000FFFFu : 0u) | (tb + 9 < len ? 0xFFFF0000u : 0u);
            A[kk][mt][0] &= m01, A[kk][mt][1] &= m01, A[kk][mt][2] &= m23, A[kk][mt][3] &= m23;
          }
        }
      }
      wg_fence();
#pragma unroll
      for (int kk = 0; kk < 4; ++kk)
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          if constexpr (PV_INTO_O) mma_rs(O[mt], A[kk][mt], desc_p + 16 * (4 * kh + kk), 1);
          else mma_rs(Os[mt], A[kk][mt], desc_p + 16 * (4 * kh + kk), (kh | kk) != 0);
        }
      wg_commit();
      wg_wait<0>();
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      if constexpr (PV_INTO_O) {
#pragma unroll
        for (int i = 0; i < 4; ++i) fence_operand(O[mt][i]);
      } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) fence_operand(Os[mt][i]);
        O[mt][0] = O[mt][0] * aa + Os[mt][0], O[mt][1] = O[mt][1] * ab + Os[mt][1];
        O[mt][2] = O[mt][2] * aa + Os[mt][2], O[mt][3] = O[mt][3] * ab + Os[mt][3];
      }
    }
    }  // !C2
    __syncwarp();
    if (lane == 0) mbar_arrive(&empty[s]);
  };
  int nfull = niter;
  bool has_tail = false;
  if constexpr (PAGED) {
    has_tail = uni((g0 + niter) * STAGE > len);
    nfull = niter - (has_tail ? 1 : 0);
  }
  // C2 with masked sinks: stage 0 (a full stage here) runs the masked body first; the loop then starts at stage 1.
  // (If stage 0 is also the last, partial stage, the tail call below masks the sinks too.)
  int it_start = 0;
  if constexpr (C2) {
    if (nskip > 0 && g0 == 0 && nfull > 0) {
      mbar_wait(&full[0], 0);
      stage_body(std::true_type{}, 0, 0);
      it_start = 1;
    }
  }
  for (int it0 = 0; !uni(it0 >= nfull); it0 += NST) {
    const uint32_t ph = (it0 / NST) & 1;
#pragma unroll
    for (int s = 0; s < NST; ++s) {
      const int it = it0 + s;
      if (uni(it >= nfull)) break;
      if constexpr (C2) {
        if (it < it_start) continue;
      }
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

  // ---- l per head over the warpgroup; merge the two warpgroups; write (as decode_fp4.cu)
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

// 11-21: q for the K8 kernels. qh (B, HKV, 8, 128) fp16 as merge.cu prep_q_rot makes it (rotated, padded rows zero)
// -> q8 (same shape) uint8 E4M3 with one scale per (sequence, kv head) qs = amax / 448 (1 if all zero), qs (B * HKV)
// fp32, and qd = e4m3(q8) * qs as fp16 (the q the kernel effectively uses; the plugin's sink merge needs exactly it).
// One block of 256 threads per (b, h): 1024 values, 4 per thread.
__global__ void __launch_bounds__(256) prep_q8_kernel(const __half* __restrict__ qh, uint8_t* __restrict__ q8,
                                                      float* __restrict__ qs, __half* __restrict__ qd) {
  __shared__ float red[8];
  const size_t base = (size_t)blockIdx.x * NPAD * D + threadIdx.x * 4;
  float x[4], m = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) x[j] = __half2float(qh[base + j]), m = fmaxf(m, fabsf(x[j]));
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
  if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = m;
  __syncthreads();
  m = red[0];
#pragma unroll
  for (int w = 1; w < 8; ++w) m = fmaxf(m, red[w]);
  const float sc = m > 0.f ? m / 448.f : 1.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const __nv_fp8_e4m3 e(x[j] / sc);
    q8[base + j] = e.__x;
    qd[base + j] = __float2half_rn((float)e * sc);
  }
  if (threadIdx.x == 0) qs[blockIdx.x] = sc;
}

template <bool PAGED, int G, bool C2 = false>
void set_smem() {
  static bool configured = false;
  if (!configured) {
    TORCH_CHECK(cudaFuncSetAttribute(decode_k8_kernel<PAGED, G, C2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)smem_bytes<PAGED, G>()) == cudaSuccess);
    configured = true;
  }
}

// sink arguments (C2 only): sink_map / sink_owner may be null (no masking / timing mode)
template <int G, bool C2 = false>
void launch_paged(const torch::Tensor& pages, const torch::Tensor& block_table, const torch::Tensor& seq_lens,
                  const torch::Tensor& q8, const float* qsp, torch::Tensor& out_o, torch::Tensor& out_ml, int splits,
                  float score_mul, float out_mul, const int* smap = nullptr, int nmap = 0, const int* owner = nullptr,
                  int nsk = 0) {
  set_smem<true, G, C2>();
  const int B = seq_lens.size(0), HKV = pages.size(1), max_pages = block_table.size(1);
  dim3 grid(splits, HKV, B);
  decode_k8_kernel<true, G, C2><<<grid, NTHREADS, smem_bytes<true, G>(), at::cuda::getCurrentCUDAStream()>>>(
      nullptr, nullptr, nullptr, nullptr, q8.data_ptr<uint8_t>(), qsp, out_o.data_ptr<float>(),
      out_ml.data_ptr<float>(), 0, HKV, score_mul, out_mul, pages.data_ptr<uint8_t>(), block_table.data_ptr<int>(),
      seq_lens.data_ptr<int>(), max_pages, smap, nmap, owner, nsk);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, C2 ? "decode_c2_paged launch failed" : "decode_k8_paged launch failed");
}

// 11-22 C2: split-KV merge with the plugin's BF16 sinks for the C2 kernel, whose split 0 already EXCLUDES the first
// NSK tokens of every sequence with a valid sink slot (same test as the kernel and merge.cu merge_sink_pool). So,
// unlike merge_sink_pool, nothing is subtracted: valid sequences get the NSK BF16 sinks (sk, sv: (P, HKV, NSK, 128),
// K - mu in the original basis, scored with the bf16 query q) added as one more partial; invalid ones are merged as
// plain 4-bit. lshift = the kernel's P shift (P_SHIFT_C2 = 8: its l carries 2^8). Output: the G real query heads.
// One 128-thread block per (b, kv head, real query head); thread = dim.
__global__ void __launch_bounds__(D) merge_sink_pool_c2_kernel(
    const float* __restrict__ o, const float* __restrict__ ml, const __nv_bfloat16* __restrict__ q,
    const int* __restrict__ bt, int bt_stride, const int* __restrict__ smap, int nmap, const int* __restrict__ owner,
    const __nv_bfloat16* __restrict__ sk, const __nv_bfloat16* __restrict__ sv, __nv_bfloat16* __restrict__ out,
    int HKV, int splits, int GK, int G, float score_mul, float lshift) {
  __shared__ float red[D / 32][NSK];
  const int head = blockIdx.x % G, bh = blockIdx.x / G, b = bh / HKV, h = bh % HKV, d = threadIdx.x;
  const int w = d / 32, lane = d % 32;
  const int blk = bt[(size_t)b * bt_stride];
  const int slot = (blk >= 0 && blk < nmap) ? smap[blk] : -1;
  const bool valid = slot >= 0 && (owner == nullptr || owner[slot] == blk);
  const float* m = ml + ((size_t)bh * splits * GK + head) * 2;  // split i at m[i * GK * 2]
  float sb[NSK];
  const size_t base = ((size_t)(valid ? slot : 0) * HKV + h) * NSK * D;
  if (valid) {  // block-uniform
    const float qd = __bfloat162float(q[((size_t)b * HKV * G + h * G + head) * D + d]);
    for (int j = 0; j < NSK; ++j) {
      float x = qd * __bfloat162float(sk[base + j * D + d]);
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) x += __shfl_xor_sync(0xffffffffu, x, off);
      if (lane == 0) red[w][j] = x;
    }
    __syncthreads();
    for (int j = 0; j < NSK; ++j) sb[j] = (red[0][j] + red[1][j] + red[2][j] + red[3][j]) * score_mul;
  }
  float mx = -INFINITY;
  for (int i = 0; i < splits; ++i) mx = fmaxf(mx, m[i * GK * 2]);
  if (valid)
    for (int j = 0; j < NSK; ++j) mx = fmaxf(mx, sb[j]);
  float num = 0.f, den = 0.f;
  for (int i = 0; i < splits; ++i) {
    const float l = m[i * GK * 2 + 1];
    if (l == 0.f) continue;  // an empty (or fully masked) split adds nothing
    const float wt = exp2f(m[i * GK * 2] - mx) * l;
    num += wt * o[(((size_t)bh * splits + i) * GK + head) * D + d];
    den += wt;
  }
  if (valid)
    for (int j = 0; j < NSK; ++j) {
      const float wt = exp2f(sb[j] - mx + lshift);
      num += wt * __bfloat162float(sv[base + j * D + d]);
      den += wt;
    }
  out[((size_t)b * HKV * G + h * G + head) * D + d] = __float2bfloat16(num / den);
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
  auto out_o = torch::empty({B, HKV, splits, 4, D}, kp.options().dtype(torch::kFloat));
  auto out_ml = torch::empty({B, HKV, splits, 4, 2}, kp.options().dtype(torch::kFloat));
  set_smem<false, 4>();
  dim3 grid(splits, HKV, B);
  decode_k8_kernel<false, 4><<<grid, NTHREADS, smem_bytes<false, 4>(), at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint8_t*>(kp.data_ptr()), ksc.data_ptr<uint8_t>(),
      reinterpret_cast<const uint8_t*>(vp.data_ptr()), vsc.data_ptr<uint8_t>(), q8.data_ptr<uint8_t>(), nullptr,
      out_o.data_ptr<float>(), out_ml.data_ptr<float>(), S, HKV, (float)score_mul, (float)out_mul, nullptr, nullptr,
      nullptr, 0, nullptr, 0, nullptr, 0);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "decode_k8 launch failed");
  return {out_o, out_ml};
}

// 11-21: pages (num_pages, HKV, 9216) uint8 (store_fp4.cu K8 store); block_table (B, max_pages) int32 (64-token
// pages); seq_lens (B) int32; q8 (B, HKV, 8, 128) uint8 E4M3 and q_scale (B * HKV) fp32 from prep_q8; score_mul
// without the q scale; gqa 4 or 8. Returns out_o (B, HKV, splits, gqa, 128), out_ml (B, HKV, splits, gqa, 2).
std::vector<torch::Tensor> decode_k8_paged(torch::Tensor pages, torch::Tensor block_table, torch::Tensor seq_lens,
                                           torch::Tensor q8, torch::Tensor q_scale, double score_mul, double out_mul,
                                           int64_t splits, int64_t gqa) {
  TORCH_CHECK(pages.is_cuda() && pages.is_contiguous() && pages.scalar_type() == torch::kUInt8 && pages.dim() == 3 &&
              pages.size(2) == PAGE_BYTES, "pages (num_pages, HKV, 9216) uint8");
  TORCH_CHECK(block_table.is_contiguous() && block_table.scalar_type() == torch::kInt);
  TORCH_CHECK(seq_lens.is_contiguous() && seq_lens.scalar_type() == torch::kInt);
  const int B = seq_lens.size(0), HKV = pages.size(1);
  TORCH_CHECK(q8.is_contiguous() && q8.scalar_type() == torch::kUInt8 && q8.dim() == 4 && q8.size(0) == B &&
              q8.size(1) == HKV && q8.size(2) == NPAD && q8.size(3) == D);
  TORCH_CHECK(q_scale.is_contiguous() && q_scale.scalar_type() == torch::kFloat && q_scale.numel() == B * HKV);
  TORCH_CHECK(gqa == 4 || gqa == 8, "gqa 4 or 8");
  auto out_o = torch::empty({B, HKV, splits, gqa, D}, pages.options().dtype(torch::kFloat));
  auto out_ml = torch::empty({B, HKV, splits, gqa, 2}, pages.options().dtype(torch::kFloat));
  if (B == 0) return {out_o, out_ml};
  if (gqa == 8) launch_paged<8>(pages, block_table, seq_lens, q8, q_scale.data_ptr<float>(), out_o, out_ml, (int)splits,
                                (float)score_mul, (float)out_mul);
  else launch_paged<4>(pages, block_table, seq_lens, q8, q_scale.data_ptr<float>(), out_o, out_ml, (int)splits,
                       (float)score_mul, (float)out_mul);
  return {out_o, out_ml};
}

// 11-22 C2: as decode_k8_paged on C2 pages (store_fp4.cu C2 store / decode_fp4_k8.pack_pages_c2); out_mul = ts_v * 64
// (V bytes are E4M3 values x 2^-6). sink_map (int32, any length; empty = no masking), sink_owner (int32, the pool's
// owner array; empty = timing mode, no owner check), nsk: tokens to mask for a valid sink slot (0 = none). The l in
// out_ml carries 2^8 (P_SHIFT_C2); merge.cu merge works unchanged, the sink merge is merge_sink_pool_c2.
std::vector<torch::Tensor> decode_c2_paged(torch::Tensor pages, torch::Tensor block_table, torch::Tensor seq_lens,
                                           torch::Tensor q8, torch::Tensor q_scale, double score_mul, double out_mul,
                                           int64_t splits, int64_t gqa, torch::Tensor sink_map, torch::Tensor sink_owner,
                                           int64_t nsk) {
  TORCH_CHECK(pages.is_cuda() && pages.is_contiguous() && pages.scalar_type() == torch::kUInt8 && pages.dim() == 3 &&
              pages.size(2) == PAGE_BYTES, "pages (num_pages, HKV, 9216) uint8");
  TORCH_CHECK(block_table.is_contiguous() && block_table.scalar_type() == torch::kInt);
  TORCH_CHECK(seq_lens.is_contiguous() && seq_lens.scalar_type() == torch::kInt);
  const int B = seq_lens.size(0), HKV = pages.size(1);
  TORCH_CHECK(q8.is_contiguous() && q8.scalar_type() == torch::kUInt8 && q8.dim() == 4 && q8.size(0) == B &&
              q8.size(1) == HKV && q8.size(2) == NPAD && q8.size(3) == D);
  TORCH_CHECK(q_scale.is_contiguous() && q_scale.scalar_type() == torch::kFloat && q_scale.numel() == B * HKV);
  TORCH_CHECK(gqa == 4 || gqa == 8, "gqa 4 or 8");
  TORCH_CHECK(nsk >= 0 && nsk <= NSK, "nsk 0..4");
  for (auto* t : {&sink_map, &sink_owner})
    TORCH_CHECK(t->numel() == 0 || (t->is_cuda() && t->is_contiguous() && t->scalar_type() == torch::kInt));
  auto out_o = torch::empty({B, HKV, splits, gqa, D}, pages.options().dtype(torch::kFloat));
  auto out_ml = torch::empty({B, HKV, splits, gqa, 2}, pages.options().dtype(torch::kFloat));
  if (B == 0) return {out_o, out_ml};
  const int* smap = sink_map.numel() ? sink_map.data_ptr<int>() : nullptr;
  const int* own = sink_owner.numel() ? sink_owner.data_ptr<int>() : nullptr;
  const int nmap = (int)sink_map.numel(), ns = smap ? (int)nsk : 0;
  if (gqa == 8) launch_paged<8, true>(pages, block_table, seq_lens, q8, q_scale.data_ptr<float>(), out_o, out_ml,
                                      (int)splits, (float)score_mul, (float)out_mul, smap, nmap, own, ns);
  else launch_paged<4, true>(pages, block_table, seq_lens, q8, q_scale.data_ptr<float>(), out_o, out_ml, (int)splits,
                             (float)score_mul, (float)out_mul, smap, nmap, own, ns);
  return {out_o, out_ml};
}

// 11-22 C2: o, ml from decode_c2_paged (run with the same sink_map / sink_owner and nsk = 4); q (B, HQ, 128) bf16
// (the real heads, original basis); block_table (B, >= 1) int32 with unit stride along dim 1; sk, sv (P, HKV, 4, 128)
// bf16 sink pools; score_mul = softmax scale x log2(e); lshift = P_SHIFT_C2. Returns (B, HKV * G, 128) bf16.
torch::Tensor merge_sink_pool_c2(torch::Tensor o, torch::Tensor ml, torch::Tensor q, torch::Tensor block_table,
                                 torch::Tensor smap, torch::Tensor owner, torch::Tensor sk, torch::Tensor sv, int64_t G,
                                 double score_mul, double lshift) {
  TORCH_CHECK(o.is_cuda() && o.is_contiguous() && o.dim() == 5 && o.size(4) == D && ml.is_contiguous());
  const int B = o.size(0), HKV = o.size(1), splits = o.size(2), GK = o.size(3);
  TORCH_CHECK(G >= 1 && G <= GK, "G real heads <= kernel heads");
  TORCH_CHECK(q.is_contiguous() && q.scalar_type() == torch::kBFloat16 && q.numel() == (int64_t)B * HKV * G * D);
  TORCH_CHECK(block_table.scalar_type() == torch::kInt && block_table.stride(1) == 1 && block_table.size(0) == B);
  TORCH_CHECK(smap.is_contiguous() && smap.scalar_type() == torch::kInt && owner.scalar_type() == torch::kInt &&
              (owner.numel() == 0 || owner.is_contiguous()));
  for (auto* t : {&sk, &sv})
    TORCH_CHECK(t->is_contiguous() && t->scalar_type() == torch::kBFloat16 && t->dim() == 4 && t->size(1) == HKV &&
                t->size(2) == NSK && t->size(3) == D && (owner.numel() == 0 || t->size(0) == owner.size(0)));
  auto out = torch::empty({B, HKV * G, D}, o.options().dtype(torch::kBFloat16));
  if (B == 0) return out;
  auto bf = [](torch::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()); };
  merge_sink_pool_c2_kernel<<<B * HKV * G, D, 0, at::cuda::getCurrentCUDAStream()>>>(
      o.data_ptr<float>(), ml.data_ptr<float>(), bf(q), block_table.data_ptr<int>(), (int)block_table.stride(0),
      smap.data_ptr<int>(), (int)smap.numel(), owner.numel() ? owner.data_ptr<int>() : nullptr, bf(sk), bf(sv),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), HKV, splits, GK, (int)G, (float)score_mul, (float)lshift);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "merge_sink_pool_c2 launch failed");
  return out;
}

std::vector<torch::Tensor> prep_q8(torch::Tensor qh) {
  TORCH_CHECK(qh.is_cuda() && qh.is_contiguous() && qh.scalar_type() == torch::kHalf && qh.dim() == 4 &&
              qh.size(2) == NPAD && qh.size(3) == D, "qh (B, HKV, 8, 128) fp16 (merge.prep_q_rot)");
  const int n = qh.size(0) * qh.size(1);
  auto q8 = torch::empty(qh.sizes(), qh.options().dtype(torch::kUInt8));
  auto qs = torch::empty({n}, qh.options().dtype(torch::kFloat));
  auto qd = torch::empty_like(qh);
  if (n == 0) return {q8, qs, qd};
  prep_q8_kernel<<<n, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(qh.data_ptr()), q8.data_ptr<uint8_t>(), qs.data_ptr<float>(),
      reinterpret_cast<__half*>(qd.data_ptr()));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "prep_q8 launch failed");
  return {q8, qs, qd};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode_k8", &decode_k8, "C1 prototype: NVFP4 decode with K on FP8 tensor cores (E4M3 scale per 32 dims)");
  m.def("decode_k8_paged", &decode_k8_paged, "C1, paged (K8 pages of 9216 bytes), G = 4 or 8, q scale per (seq, head)");
  m.def("prep_q8", &prep_q8, "qh (B, HKV, 8, 128) fp16 -> q8 E4M3, scale per (seq, head), dequantized q (fp16)");
  m.def("decode_c2_paged", &decode_c2_paged, "C2: K8 + P V on FP8 tensor cores (V scale per channel and 32 tokens)");
  m.def("merge_sink_pool_c2", &merge_sink_pool_c2, "C2: split-KV merge adding the plugin's BF16 sinks (no subtraction)");
}
