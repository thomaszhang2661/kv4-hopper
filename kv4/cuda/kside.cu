// rebuild tag: common.cuh rev 2 (mbarrier wait with suspend hint, 11-11)
// K-side-only prototype of a Hopper-native 4-bit KV decode kernel (11-10). Question: with the reconstructed K fed to
// wgmma straight from registers, about 0.6 instructions per value of reconstruction, and a TMA producer warp, how close
// does reading an INT4 K cache get to H200's bandwidth? (Gate: about 60%.) No V side yet: each split writes only the
// softmax statistics (running max m and exp sum l, base 2) of its scores.
//
// Format: int4_tok (symmetric, code = q + 8, one FP16 scale per token row), rows permuted at write time to the wgmma
// A-fragment layout (kv4/cuda/kside.py: pack_kside). Per (batch, kv head): codes (S, 64 B), scales (S) fp16.
//
// Math. wgmma m64n8k16 computes S^T = K q^T: M = 64 tokens, N = 8 (4 query heads of the GQA group, padded), K = 16
// head dims per instruction, 8 instructions per 128 dims. Thread (lane = 4g + t) of warp w holds A elements
// (row 16w+g and 16w+g+8) x (cols 2t, 2t+1, 2t+8, 2t+9) of each k-step, as four f16x2 registers R0..R3:
//   R0 = (row g, cols 2t..2t+1)  R1 = (row g+8, same)  R2 = (row g, cols 2t+8..2t+9)  R3 = (row g+8, same).
// For a row, thread t's 32 values over the 8 k-steps are 16 contiguous bytes (4 words); in word i:
//   bits 0-3 | 16-19 : dims 32i+2t, +1        -> (w & 0x000F000F) | 0x64006400 = 1024 + c       (R0 of k-step 2i)
//   bits 4-7 | 20-23 : dims 32i+2t+8, +9      -> (w & 0x00F000F0) | 0x64006400 = 1024 + 16c     (R2 of k-step 2i)
//   bits 8-11| 24-27 : dims 32i+16+2t, +1     -> ((w>>8) & 0x000F000F) | ...   = 1024 + c       (R0 of k-step 2i+1)
//   bits12-15| 28-31 : dims 32i+24+2t, +1     -> ((w>>8) & 0x00F000F0) | ...   = 1024 + 16c     (R2 of k-step 2i+1)
// So 8 values cost 1 shift + 4 lop3. The factor m_d = 16 of dims with d % 16 >= 8 is folded into q (q'_d = q_d / m_d,
// exact), and the offsets into one constant per head: q.(c - 8) = acc - K0, K0 = sum_d q'_d (1024 + 8 m_d).
// Score (base 2) = (acc - K0) * scale_token * softmax_scale * log2(e).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace {

constexpr int D = 128;
constexpr int ROW_BYTES = 64;  // 128 codes
constexpr int STAGE = 256;     // tokens per pipeline stage
constexpr int NCWG = 2;        // compute warpgroups
constexpr int NCWARPS = NCWG * 4;
constexpr int NTHREADS = NCWARPS * 32 + 32;  // + one producer warp (last, so warpgroups stay aligned)
constexpr int NPAD = 8;        // wgmma N
constexpr int GQA = 4;         // real query heads per kv head
constexpr uint32_t MAGIC = 0x64006400u;  // fp16 1024 in both halves

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// --- mbarrier and bulk copy (TMA, 1D) ---------------------------------------------------------------------------
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  uint32_t addr = smem_u32(bar), done = 0;
  while (!done) {
    asm volatile(
        "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
        : "=r"(done)
        : "r"(addr), "r"(parity)
        : "memory");
  }
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                   smem_u32(dst)),
               "l"(src), "r"(bytes), "r"(smem_u32(bar))
               : "memory");
}

// --- wgmma ----------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory"); }
__device__ __forceinline__ void fence_operand(float& x) { asm volatile("" : "+f"(x)::"memory"); }

// Shared-memory matrix descriptor, no swizzle (core matrices of 8 rows x 16 bytes, rows 16 bytes apart).
// q^T is stored K-major: core matrix kc (dims 8kc..8kc+7) of all 8 heads at byte offset 128*kc. LBO (next core
// matrix along K) = 128; SBO (next 8-row group along N) is unused with N = 8 and also set to 128.
__device__ __forceinline__ uint64_t make_desc(uint32_t addr) {
  uint64_t d = 0;
  d |= (uint64_t)((addr & 0x3FFFF) >> 4);
  d |= (uint64_t)(128 >> 4) << 16;
  d |= (uint64_t)(128 >> 4) << 32;
  return d;
}

__device__ __forceinline__ void mma_rs(float (&d)[4], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %9, 0;\n"
      " wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0;\n}"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}

// (x & mask) | magic as one LOP3: SASS allows one immediate per LOP3, so magic must sit in a register (11-10: left to
// itself the compiler emitted two LOP3s per pair, 1.02 instructions per value instead of 0.5).
__device__ __forceinline__ uint32_t lop_lo(uint32_t x, uint32_t magic) {
  uint32_t r;
  asm("lop3.b32 %0, %1, 0x000F000F, %2, 0xEA;" : "=r"(r) : "r"(x), "r"(magic));
  return r;
}
__device__ __forceinline__ uint32_t lop_hi(uint32_t x, uint32_t magic) {
  uint32_t r;
  asm("lop3.b32 %0, %1, 0x00F000F0, %2, 0xEA;" : "=r"(r) : "r"(x), "r"(magic));
  return r;
}

__device__ __forceinline__ void online(float& m, float& l, float s0, float s1) {
  float mn = fmaxf(m, fmaxf(s0, s1));
  l = l * exp2f(m - mn) + exp2f(s0 - mn) + exp2f(s1 - mn);
  m = mn;
}
__device__ __forceinline__ void merge(float& m, float& l, float m2, float l2) {
  float mn = fmaxf(m, m2);
  if (mn == -INFINITY) return;
  l = l * exp2f(m - mn) + l2 * exp2f(m2 - mn);
  m = mn;
}

// VAR 0: v1 (subtiles in order). VAR 1: v2 (two wgmma groups in flight; softmax of subtile 0 overlaps subtile 1).
// mode (diagnosis): 0 normal; 1 load only (compute warps release each stage untouched); 2 compute only (no loads,
// compute warps reuse whatever is in shared memory).
template <int NST, int VAR>
__global__ void __launch_bounds__(NTHREADS) kside_kernel(const uint8_t* __restrict__ kp,   // (B, HKV, S, 64)
                                                         const __half* __restrict__ ks,    // (B, HKV, S)
                                                         const __half* __restrict__ qp,    // (B, HKV, 8, 128)
                                                         const float* __restrict__ k0,     // (B, HKV, 8)
                                                         float* __restrict__ out_ml,       // (B, HKV, splits, 4, 2)
                                                         float* __restrict__ scores,       // (B, HKV, 4, S) or null
                                                         int S, int HKV, float scale_log2, int mode) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* s_codes = smem;                                            // NST x 16 KB
  __half* s_scl = reinterpret_cast<__half*>(smem + NST * STAGE * ROW_BYTES);  // NST x 256
  __half* s_q = s_scl + NST * STAGE;                                  // 8 x 128, core-matrix order
  uint64_t* full = reinterpret_cast<uint64_t*>(s_q + NPAD * D);
  uint64_t* empty = full + NST;
  float* s_red = reinterpret_cast<float*>(empty + NST);               // NCWARPS x 4 heads x (m, l)

  const int split = blockIdx.x, h = blockIdx.y, b = blockIdx.z, splits = gridDim.x;
  const int tok_per = S / splits, tok0 = split * tok_per, niter = tok_per / STAGE;
  const size_t bh = (size_t)b * HKV + h;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;

  // q'^T into shared memory: element (head n, dim) at half offset (dim / 8) * 64 + n * 8 + dim % 8.
  for (int i = tid; i < NPAD * D; i += NTHREADS) {
    int n = i / D, dim = i % D;
    s_q[(dim / 8) * 64 + n * 8 + dim % 8] = qp[bh * NPAD * D + i];
  }
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], NCWARPS);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // q stores visible to wgmma (async proxy)
  __syncthreads();

  if (warp == NCWARPS) {  // producer warp: one thread issues the bulk copies
    if (lane == 0 && mode != 2) {
      const uint8_t* gk = kp + (bh * S + tok0) * ROW_BYTES;
      const __half* gs = ks + bh * S + tok0;
      for (int it = 0; it < niter; ++it) {
        const int s = it % NST;
        mbar_wait(&empty[s], ((it / NST) & 1) ^ 1);
        mbar_expect_tx(&full[s], STAGE * ROW_BYTES + STAGE * 2);
        bulk_g2s(s_codes + s * STAGE * ROW_BYTES, gk + (size_t)it * STAGE * ROW_BYTES, STAGE * ROW_BYTES, &full[s]);
        bulk_g2s(s_scl + s * STAGE, gs + (size_t)it * STAGE, STAGE * 2, &full[s]);
      }
    }
    return;
  }

  // compute warpgroups
  const int wg = warp / 4, wi = warp % 4, g = lane / 4, t = lane % 4;
  const bool real = t < GQA / 2;  // accumulator cols 2t, 2t+1 are heads; heads 4..7 are padding
  const float k0a = real ? k0[bh * NPAD + 2 * t] : 0.f, k0b = real ? k0[bh * NPAD + 2 * t + 1] : 0.f;
  float ma = -INFINITY, la = 0.f, mb = -INFINITY, lb = 0.f;
  const uint32_t q_addr = smem_u32(s_q);

  // One 64-token subtile: read + reconstruct (A registers), issue 8 wgmma, softmax update from the accumulators.
  auto dequant = [&](const uint8_t* base, int r0, uint32_t (&A)[8][4]) {
    const uint4 w0 = *reinterpret_cast<const uint4*>(base + r0 * ROW_BYTES + t * 16);
    const uint4 w1 = *reinterpret_cast<const uint4*>(base + (r0 + 8) * ROW_BYTES + t * 16);
    const uint32_t x0[4] = {w0.x, w0.y, w0.z, w0.w}, x1[4] = {w1.x, w1.y, w1.z, w1.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t a = x0[i], c = x1[i], ah = a >> 8, ch = c >> 8;
      A[2 * i][0] = (a & 0x000F000Fu) | MAGIC;
      A[2 * i][1] = (c & 0x000F000Fu) | MAGIC;
      A[2 * i][2] = (a & 0x00F000F0u) | MAGIC;
      A[2 * i][3] = (c & 0x00F000F0u) | MAGIC;
      A[2 * i + 1][0] = (ah & 0x000F000Fu) | MAGIC;
      A[2 * i + 1][1] = (ch & 0x000F000Fu) | MAGIC;
      A[2 * i + 1][2] = (ah & 0x00F000F0u) | MAGIC;
      A[2 * i + 1][3] = (ch & 0x00F000F0u) | MAGIC;
    }
  };
  auto mma8 = [&](float (&d)[4], const uint32_t (&A)[8][4]) {
#pragma unroll
    for (int j = 0; j < 8; ++j) mma_rs(d, A[j], make_desc(q_addr + j * 256), j > 0);
  };
  // v4 (VAR 2): lean reconstruction (one LOP3 per pair), descriptors computed once, and softmax on all 32 lanes:
  // lanes t = 2, 3 hold padded heads, so they take rows r0 + 8 of lanes t - 2 by shuffle; every lane then owns one
  // row and the head pair hp = t & 1 (heads 2hp, 2hp + 1) per subtile.
  uint32_t magic;
  asm volatile("mov.b32 %0, 0x64006400;" : "=r"(magic));
  const uint64_t desc0 = make_desc(q_addr);
  const int hp = t & 1;
  const bool hi = t >= 2;
  const float k0c = k0[bh * NPAD + 2 * hp], k0d = k0[bh * NPAD + 2 * hp + 1];
  auto dequant_lean = [&](const uint8_t* base, int r0, uint32_t (&A)[8][4]) {
    const uint4 w0 = *reinterpret_cast<const uint4*>(base + r0 * ROW_BYTES + t * 16);
    const uint4 w1 = *reinterpret_cast<const uint4*>(base + (r0 + 8) * ROW_BYTES + t * 16);
    const uint32_t x0[4] = {w0.x, w0.y, w0.z, w0.w}, x1[4] = {w1.x, w1.y, w1.z, w1.w};
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
  };
  auto mma8_lean = [&](float (&d)[4], const uint32_t (&A)[8][4]) {
#pragma unroll
    for (int j = 0; j < 8; ++j) mma_rs(d, A[j], desc0 + 16 * j, j > 0);  // +256 bytes per k-step
  };
  // this lane's two scores (heads 2hp, 2hp+1) of its row in a subtile
  auto lane_scores = [&](int s, int it, int r0, float (&d)[4], float& sa, float& sb) {
#pragma unroll
    for (int i = 0; i < 4; ++i) fence_operand(d[i]);
    const float u2 = __shfl_up_sync(0xffffffffu, d[2], 2), u3 = __shfl_up_sync(0xffffffffu, d[3], 2);
    const int row = r0 + (hi ? 8 : 0);
    const float c = __half2float(s_scl[s * STAGE + row]) * scale_log2;
    sa = ((hi ? u2 : d[0]) - k0c) * c;
    sb = ((hi ? u3 : d[1]) - k0d) * c;
    if (scores) {
      const size_t tok = (size_t)tok0 + (size_t)it * STAGE + row;
      float* pa = scores + (bh * GQA + 2 * hp) * S;
      pa[tok] = sa, pa[S + tok] = sb;
    }
  };

  auto softmax = [&](int s, int it, int r0, float (&d)[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) fence_operand(d[i]);
    if (real) {
      const float c0 = __half2float(s_scl[s * STAGE + r0]) * scale_log2;
      const float c1 = __half2float(s_scl[s * STAGE + r0 + 8]) * scale_log2;
      const float s00 = (d[0] - k0a) * c0, s01 = (d[1] - k0b) * c0, s10 = (d[2] - k0a) * c1, s11 = (d[3] - k0b) * c1;
      online(ma, la, s00, s10);
      online(mb, lb, s01, s11);
      if (scores) {
        const size_t tok = (size_t)tok0 + (size_t)it * STAGE + r0;
        float* sa = scores + (bh * GQA + 2 * t) * S;
        float* sb = sa + S;
        sa[tok] = s00, sb[tok] = s01, sa[tok + 8] = s10, sb[tok + 8] = s11;
      }
    }
  };

  for (int it = 0; it < niter; ++it) {
    const int s = it % NST;
    if (mode != 2) mbar_wait(&full[s], (it / NST) & 1);
    if (mode != 1) {
      const uint8_t* base = s_codes + s * STAGE * ROW_BYTES;
      const int r0 = wg * 128 + wi * 16 + g, r1 = r0 + 64;  // token rows of subtiles 0 and 1 (and +8)
      if constexpr (VAR == 0) {  // v1: subtiles one after the other
        uint32_t A[8][4];
        float d[4] = {0.f, 0.f, 0.f, 0.f};
        dequant(base, r0, A);
        wg_fence();
        mma8(d, A);
        wg_commit();
        wg_wait<0>();
        softmax(s, it, r0, d);
        float e[4] = {0.f, 0.f, 0.f, 0.f};
        dequant(base, r1, A);
        wg_fence();
        mma8(e, A);
        wg_commit();
        wg_wait<0>();
        softmax(s, it, r1, e);
      } else if constexpr (VAR == 2) {  // v4: v1's order, lean instructions
        uint32_t A[8][4];
        float d[4] = {0.f, 0.f, 0.f, 0.f}, e[4] = {0.f, 0.f, 0.f, 0.f};
        float sa0, sb0, sa1, sb1;
        dequant_lean(base, r0, A);
        wg_fence();
        mma8_lean(d, A);
        wg_commit();
        wg_wait<0>();
        lane_scores(s, it, r0, d, sa0, sb0);
        dequant_lean(base, r1, A);
        wg_fence();
        mma8_lean(e, A);
        wg_commit();
        wg_wait<0>();
        lane_scores(s, it, r1, e, sa1, sb1);
        online(ma, la, sa0, sa1);
        online(mb, lb, sb0, sb1);
      } else {  // v2: subtile 1 is reconstructed and its wgmma issued before subtile 0's softmax
        uint32_t A0[8][4], A1[8][4];
        float d[4] = {0.f, 0.f, 0.f, 0.f}, e[4] = {0.f, 0.f, 0.f, 0.f};
        dequant(base, r0, A0);
        wg_fence();
        mma8(d, A0);
        wg_commit();
        dequant(base, r1, A1);
        wg_fence();
        mma8(e, A1);
        wg_commit();
        wg_wait<1>();
        softmax(s, it, r0, d);
        wg_wait<0>();
        softmax(s, it, r1, e);
      }
    }
    __syncwarp();
    if (mode != 2 && lane == 0) mbar_arrive(&empty[s]);
  }

  // reduce (m, l) over the lanes holding the same heads (v1/v2: same t; v4: same t & 1), then over the 8 warps
#pragma unroll
  for (int off = (VAR == 2 ? 2 : 4); off < 32; off <<= 1) {
    merge(ma, la, __shfl_xor_sync(0xffffffffu, ma, off), __shfl_xor_sync(0xffffffffu, la, off));
    merge(mb, lb, __shfl_xor_sync(0xffffffffu, mb, off), __shfl_xor_sync(0xffffffffu, lb, off));
  }
  if (g == 0 && real) {
    float* r = s_red + (warp * GQA + 2 * t) * 2;
    r[0] = ma, r[1] = la, r[2] = mb, r[3] = lb;
  }
  asm volatile("bar.sync 1, %0;" ::"n"(NCWARPS * 32) : "memory");  // compute warps only (producer has exited)
  if (tid < GQA) {
    float m = -INFINITY, l = 0.f;
    for (int w = 0; w < NCWARPS; ++w) merge(m, l, s_red[(w * GQA + tid) * 2], s_red[(w * GQA + tid) * 2 + 1]);
    float* o = out_ml + ((bh * splits + split) * GQA + tid) * 2;
    o[0] = m, o[1] = l;
  }
}

// v3: v1's compute, persistent CTAs. grid = resident capacity; CTA c handles work items c, c + grid, ... where an item
// is (batch, kv head, split). The producer streams stages across item boundaries without draining the pipeline, and
// loads each item's q (already in core-matrix order, 2 KB) into one of two buffers with a bulk copy.
template <int NST>
__global__ void __launch_bounds__(NTHREADS) kside_persistent(const uint8_t* __restrict__ kp,  // (B, HKV, S, 64)
                                                             const __half* __restrict__ ks,   // (B, HKV, S)
                                                             const __half* __restrict__ qcm,  // (B, HKV, 16, 8, 8)
                                                             const float* __restrict__ k0,    // (B, HKV, 8)
                                                             float* __restrict__ out_ml,      // (B, HKV, splits, 4, 2)
                                                             float* __restrict__ scores,      // (B, HKV, 4, S) or null
                                                             int S, int HKV, int splits, int n_items,
                                                             float scale_log2) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* s_codes = smem;
  __half* s_scl = reinterpret_cast<__half*>(smem + NST * STAGE * ROW_BYTES);
  __half* s_q = s_scl + NST * STAGE;  // 2 buffers x 1024 halves
  uint64_t* full = reinterpret_cast<uint64_t*>(s_q + 2 * NPAD * D);
  uint64_t* empty = full + NST;
  uint64_t* qfull = empty + NST;
  uint64_t* qempty = qfull + 2;
  float* s_red = reinterpret_cast<float*>(qempty + 2);

  const int tok_per = S / splits, niter = tok_per / STAGE;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  if (tid == 0) {
    for (int s = 0; s < NST; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], NCWARPS);
    }
    for (int s = 0; s < 2; ++s) {
      mbar_init(&qfull[s], 1);
      mbar_init(&qempty[s], NCWARPS);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (warp == NCWARPS) {
    if (lane == 0) {
      int gi = 0, li = 0;
      for (int item = blockIdx.x; item < n_items; item += gridDim.x, ++li) {
        const int split = item % splits;
        const size_t bh = item / splits;
        const int tok0 = split * tok_per, qs = li & 1;
        mbar_wait(&qempty[qs], ((li >> 1) & 1) ^ 1);
        mbar_expect_tx(&qfull[qs], NPAD * D * 2);
        bulk_g2s(s_q + qs * NPAD * D, qcm + bh * NPAD * D, NPAD * D * 2, &qfull[qs]);
        const uint8_t* gk = kp + (bh * S + tok0) * ROW_BYTES;
        const __half* gs = ks + bh * S + tok0;
        for (int it = 0; it < niter; ++it, ++gi) {
          const int s = gi % NST;
          mbar_wait(&empty[s], ((gi / NST) & 1) ^ 1);
          mbar_expect_tx(&full[s], STAGE * ROW_BYTES + STAGE * 2);
          bulk_g2s(s_codes + s * STAGE * ROW_BYTES, gk + (size_t)it * STAGE * ROW_BYTES, STAGE * ROW_BYTES, &full[s]);
          bulk_g2s(s_scl + s * STAGE, gs + (size_t)it * STAGE, STAGE * 2, &full[s]);
        }
      }
    }
    return;
  }

  const int wg = warp / 4, wi = warp % 4, g = lane / 4, t = lane % 4;
  const bool real = t < GQA / 2;
  int gi = 0, li = 0;
  for (int item = blockIdx.x; item < n_items; item += gridDim.x, ++li) {
    const int split = item % splits;
    const size_t bh = item / splits;
    const int tok0 = split * tok_per, qs = li & 1;
    const float k0a = real ? k0[bh * NPAD + 2 * t] : 0.f, k0b = real ? k0[bh * NPAD + 2 * t + 1] : 0.f;
    float ma = -INFINITY, la = 0.f, mb = -INFINITY, lb = 0.f;
    const uint32_t q_addr = smem_u32(s_q + qs * NPAD * D);
    mbar_wait(&qfull[qs], (li >> 1) & 1);
    for (int it = 0; it < niter; ++it, ++gi) {
      const int s = gi % NST;
      mbar_wait(&full[s], (gi / NST) & 1);
      const uint8_t* base = s_codes + s * STAGE * ROW_BYTES;
#pragma unroll
      for (int sub = 0; sub < 2; ++sub) {
        const int r0 = wg * 128 + sub * 64 + wi * 16 + g;
        const uint4 w0 = *reinterpret_cast<const uint4*>(base + r0 * ROW_BYTES + t * 16);
        const uint4 w1 = *reinterpret_cast<const uint4*>(base + (r0 + 8) * ROW_BYTES + t * 16);
        const uint32_t x0[4] = {w0.x, w0.y, w0.z, w0.w}, x1[4] = {w1.x, w1.y, w1.z, w1.w};
        uint32_t A[8][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint32_t a = x0[i], c = x1[i], ah = a >> 8, ch = c >> 8;
          A[2 * i][0] = (a & 0x000F000Fu) | MAGIC;
          A[2 * i][1] = (c & 0x000F000Fu) | MAGIC;
          A[2 * i][2] = (a & 0x00F000F0u) | MAGIC;
          A[2 * i][3] = (c & 0x00F000F0u) | MAGIC;
          A[2 * i + 1][0] = (ah & 0x000F000Fu) | MAGIC;
          A[2 * i + 1][1] = (ch & 0x000F000Fu) | MAGIC;
          A[2 * i + 1][2] = (ah & 0x00F000F0u) | MAGIC;
          A[2 * i + 1][3] = (ch & 0x00F000F0u) | MAGIC;
        }
        float d[4] = {0.f, 0.f, 0.f, 0.f};
        wg_fence();
#pragma unroll
        for (int j = 0; j < 8; ++j) mma_rs(d, A[j], make_desc(q_addr + j * 256), j > 0);
        wg_commit();
        wg_wait<0>();
#pragma unroll
        for (int i = 0; i < 4; ++i) fence_operand(d[i]);
        if (real) {
          const float c0 = __half2float(s_scl[s * STAGE + r0]) * scale_log2;
          const float c1 = __half2float(s_scl[s * STAGE + r0 + 8]) * scale_log2;
          const float s00 = (d[0] - k0a) * c0, s01 = (d[1] - k0b) * c0, s10 = (d[2] - k0a) * c1,
                      s11 = (d[3] - k0b) * c1;
          online(ma, la, s00, s10);
          online(mb, lb, s01, s11);
          if (scores) {
            const size_t tok = (size_t)tok0 + (size_t)it * STAGE + r0;
            float* sa = scores + (bh * GQA + 2 * t) * S;
            float* sb = sa + S;
            sa[tok] = s00, sb[tok] = s01, sa[tok + 8] = s10, sb[tok + 8] = s11;
          }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty[s]);
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(&qempty[qs]);  // this warp's wgmmas on q buffer qs are complete (wait_group 0)
#pragma unroll
    for (int off = 4; off < 32; off <<= 1) {
      merge(ma, la, __shfl_xor_sync(0xffffffffu, ma, off), __shfl_xor_sync(0xffffffffu, la, off));
      merge(mb, lb, __shfl_xor_sync(0xffffffffu, mb, off), __shfl_xor_sync(0xffffffffu, lb, off));
    }
    if (g == 0 && real) {
      float* r = s_red + (warp * GQA + 2 * t) * 2;
      r[0] = ma, r[1] = la, r[2] = mb, r[3] = lb;
    }
    asm volatile("bar.sync 1, %0;" ::"n"(NCWARPS * 32) : "memory");
    if (tid < GQA) {
      float m = -INFINITY, l = 0.f;
      for (int w = 0; w < NCWARPS; ++w) merge(m, l, s_red[(w * GQA + tid) * 2], s_red[(w * GQA + tid) * 2 + 1]);
      float* o = out_ml + ((bh * splits + split) * GQA + tid) * 2;
      o[0] = m, o[1] = l;
    }
    asm volatile("bar.sync 1, %0;" ::"n"(NCWARPS * 32) : "memory");  // s_red is reused by the next item
  }
}

template <int NST>
void launch_persistent(const torch::Tensor& kp, const torch::Tensor& ks, const torch::Tensor& qcm,
                       const torch::Tensor& k0, torch::Tensor& out_ml, float* scores, int splits, float scale_log2,
                       int max_ctas) {
  const int B = kp.size(0), HKV = kp.size(1), S = kp.size(2);
  const size_t smem = NST * (STAGE * ROW_BYTES + STAGE * 2) + 2 * NPAD * D * 2 + (2 * NST + 4) * 8 +
                      NCWARPS * GQA * 2 * 4;
  auto kern = kside_persistent<NST>;
  static int resident = 0;
  if (!resident) {
    TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess);
    int per_sm = 0, sms = 0, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, NTHREADS, smem);
    resident = per_sm * sms;
  }
  const int n_items = B * HKV * splits;
  int grid = std::min(n_items, resident);
  if (max_ctas > 0) grid = std::min(grid, max_ctas);
  kern<<<grid, NTHREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint8_t*>(kp.data_ptr()), reinterpret_cast<const __half*>(ks.data_ptr()),
      reinterpret_cast<const __half*>(qcm.data_ptr()), k0.data_ptr<float>(), out_ml.data_ptr<float>(), scores, S, HKV,
      splits, n_items, scale_log2);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "kside_persistent launch failed");
}

template <int NST, int VAR>
void launch(const torch::Tensor& kp, const torch::Tensor& ks, const torch::Tensor& qp, const torch::Tensor& k0,
            torch::Tensor& out_ml, float* scores, int splits, float scale_log2, int mode) {
  const int B = kp.size(0), HKV = kp.size(1), S = kp.size(2);
  const size_t smem = NST * (STAGE * ROW_BYTES + STAGE * 2) + NPAD * D * 2 + 2 * NST * 8 + NCWARPS * GQA * 2 * 4;
  auto kern = kside_kernel<NST, VAR>;
  static bool configured = false;
  if (!configured) {
    TORCH_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess);
    configured = true;
  }
  dim3 grid(splits, HKV, B);
  kern<<<grid, NTHREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint8_t*>(kp.data_ptr()), reinterpret_cast<const __half*>(ks.data_ptr()),
      reinterpret_cast<const __half*>(qp.data_ptr()), k0.data_ptr<float>(), out_ml.data_ptr<float>(), scores, S, HKV,
      scale_log2, mode);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "kside launch failed");
}

template <int VAR>
void dispatch(int nst, const torch::Tensor& kp, const torch::Tensor& ks, const torch::Tensor& qp,
              const torch::Tensor& k0, torch::Tensor& out_ml, float* sp, int splits, float sl2, int mode) {
  switch (nst) {
    case 2: launch<2, VAR>(kp, ks, qp, k0, out_ml, sp, splits, sl2, mode); break;
    case 3: launch<3, VAR>(kp, ks, qp, k0, out_ml, sp, splits, sl2, mode); break;
    case 4: launch<4, VAR>(kp, ks, qp, k0, out_ml, sp, splits, sl2, mode); break;
    case 6: launch<6, VAR>(kp, ks, qp, k0, out_ml, sp, splits, sl2, mode); break;
    default: TORCH_CHECK(false, "nst must be 2, 3, 4 or 6");
  }
}

}  // namespace

// kp (B, HKV, S, 16) int32 permuted codes; ks (B, HKV, S) fp16; qp (B, HKV, 8, 128) fp16; k0 (B, HKV, 8) fp32.
// Returns out_ml (B, HKV, splits, 4, 2) fp32; fills `scores` (B, HKV, 4, S) fp32 (base 2) when it is defined.
torch::Tensor kside(torch::Tensor kp, torch::Tensor ks, torch::Tensor qp, torch::Tensor k0, double softmax_scale,
                    int64_t splits, int64_t nst, c10::optional<torch::Tensor> scores, int64_t var, int64_t mode) {
  TORCH_CHECK(kp.is_cuda() && kp.is_contiguous() && kp.dim() == 4 && kp.size(3) == 16 && kp.element_size() == 4);
  TORCH_CHECK(ks.is_contiguous() && ks.scalar_type() == torch::kHalf);
  TORCH_CHECK(qp.is_contiguous() && qp.scalar_type() == torch::kHalf && qp.size(2) == NPAD && qp.size(3) == D);
  TORCH_CHECK(k0.is_contiguous() && k0.scalar_type() == torch::kFloat);
  const int B = kp.size(0), HKV = kp.size(1), S = kp.size(2);
  TORCH_CHECK(S % (splits * STAGE) == 0, "S must be a multiple of splits * 256");
  auto out_ml = torch::empty({B, HKV, splits, GQA, 2}, kp.options().dtype(torch::kFloat));
  float* sp = nullptr;
  if (scores.has_value() && scores->defined()) {
    TORCH_CHECK(scores->is_contiguous() && scores->scalar_type() == torch::kFloat && scores->numel() == (int64_t)B * HKV * GQA * S);
    sp = scores->data_ptr<float>();
  }
  const float scale_log2 = (float)(softmax_scale * 1.4426950408889634);
  TORCH_CHECK(mode >= 0 && mode <= 2, "mode: 0 normal, 1 load only, 2 compute only");
  if (var == 0) dispatch<0>(nst, kp, ks, qp, k0, out_ml, sp, splits, scale_log2, mode);
  else if (var == 1) dispatch<1>(nst, kp, ks, qp, k0, out_ml, sp, splits, scale_log2, mode);
  else if (var == 2) dispatch<2>(nst, kp, ks, qp, k0, out_ml, sp, splits, scale_log2, mode);
  else TORCH_CHECK(false, "var must be 0 (v1), 1 (v2) or 2 (v4 lean)");
  return out_ml;
}

// v3 (persistent). qcm (B, HKV, 16, 8, 8) fp16 from kside.core_matrix(qp); other arguments as kside().
torch::Tensor kside_persistent_op(torch::Tensor kp, torch::Tensor ks, torch::Tensor qcm, torch::Tensor k0,
                               double softmax_scale, int64_t splits, int64_t nst, c10::optional<torch::Tensor> scores,
                               int64_t max_ctas) {
  TORCH_CHECK(kp.is_cuda() && kp.is_contiguous() && kp.dim() == 4 && kp.size(3) == 16 && kp.element_size() == 4);
  TORCH_CHECK(ks.is_contiguous() && ks.scalar_type() == torch::kHalf);
  TORCH_CHECK(qcm.is_contiguous() && qcm.scalar_type() == torch::kHalf && qcm.numel() == kp.size(0) * kp.size(1) * NPAD * D);
  TORCH_CHECK(k0.is_contiguous() && k0.scalar_type() == torch::kFloat);
  const int B = kp.size(0), HKV = kp.size(1), S = kp.size(2);
  TORCH_CHECK(S % (splits * STAGE) == 0, "S must be a multiple of splits * 256");
  auto out_ml = torch::empty({B, HKV, splits, GQA, 2}, kp.options().dtype(torch::kFloat));
  float* sp = nullptr;
  if (scores.has_value() && scores->defined()) {
    TORCH_CHECK(scores->is_contiguous() && scores->scalar_type() == torch::kFloat && scores->numel() == (int64_t)B * HKV * GQA * S);
    sp = scores->data_ptr<float>();
  }
  const float sl2 = (float)(softmax_scale * 1.4426950408889634);
  switch (nst) {
    case 2: launch_persistent<2>(kp, ks, qcm, k0, out_ml, sp, splits, sl2, max_ctas); break;
    case 3: launch_persistent<3>(kp, ks, qcm, k0, out_ml, sp, splits, sl2, max_ctas); break;
    case 4: launch_persistent<4>(kp, ks, qcm, k0, out_ml, sp, splits, sl2, max_ctas); break;
    case 6: launch_persistent<6>(kp, ks, qcm, k0, out_ml, sp, splits, sl2, max_ctas); break;
    default: TORCH_CHECK(false, "nst must be 2, 3, 4 or 6");
  }
  return out_ml;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kside", &kside, "K-side INT4 decode prototype (wgmma RS, TMA bulk producer)", py::arg("kp"), py::arg("ks"),
        py::arg("qp"), py::arg("k0"), py::arg("softmax_scale"), py::arg("splits"), py::arg("nst"),
        py::arg("scores") = py::none(), py::arg("var") = 0, py::arg("mode") = 0);
  m.def("kside_persistent", &kside_persistent_op, "v3: v1 compute with persistent CTAs", py::arg("kp"), py::arg("ks"),
        py::arg("qcm"), py::arg("k0"), py::arg("softmax_scale"), py::arg("splits"), py::arg("nst"),
        py::arg("scores") = py::none(), py::arg("max_ctas") = 0);
}
