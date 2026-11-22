// Shared device helpers for the Hopper prototypes (11-10): mbarrier, 1D TMA bulk copy, wgmma (A from registers),
// shared-memory matrix descriptors, the fp16 magic-number nibble extraction, online-softmax merging.
#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace kv4cu {

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
// try_wait with a suspend-time hint (11-11): the waiting warp is suspended until the phase completes (or the hint
// expires) instead of spinning; NVBit showed spinning (BRA + SYNCS + YIELD) at ~0.39 instructions per value in the FP4
// kernel, ~10% of its issue slots, taken from the warps doing real work.
#ifndef KV4_WAIT_HINT_NS
#define KV4_WAIT_HINT_NS 1000000
#endif
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  uint32_t addr = smem_u32(bar), done = 0;
  while (!done) {
    asm volatile(
        "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}"
        : "=r"(done)
        : "r"(addr), "r"(parity), "n"(KV4_WAIT_HINT_NS)
        : "memory");
  }
}
// Producer-side wait (11-11): sleep between polls. The producer warp spends most of its time waiting for consumers to
// free a stage; its tight polling (~0.39 instructions per value of BRA/SYNCS/YIELD in NVBit) competes for issue slots
// with the compute warps on the same scheduler. The try_wait suspend hint did not reduce it.
#ifndef KV4_PRODUCER_SLEEP_NS
#define KV4_PRODUCER_SLEEP_NS 128
#endif
__device__ __forceinline__ void mbar_wait_sleep(uint64_t* bar, uint32_t parity) {
  uint32_t addr = smem_u32(bar), done = 0;
  while (true) {
    asm volatile(
        "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
        : "=r"(done)
        : "r"(addr), "r"(parity)
        : "memory");
    if (done) break;
    __nanosleep(KV4_PRODUCER_SLEEP_NS);
  }
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                   smem_u32(dst)),
               "l"(src), "r"(bytes), "r"(smem_u32(bar))
               : "memory");
}
__device__ __forceinline__ void named_bar(int id, int threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

// --- wgmma ----------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory"); }
__device__ __forceinline__ void fence_operand(float& x) { asm volatile("" : "+f"(x)::"memory"); }

// No-swizzle K-major descriptor: core matrices of 8 rows x 16 bytes, core matrix kc at byte offset 128 * kc along K
// (LBO = 128); SBO is unused with N = 8 (one 8-row group) and also set to 128. One k-step of 16 = +256 bytes = +16.
__device__ __forceinline__ uint64_t make_desc(uint32_t addr) {
  uint64_t d = 0;
  d |= (uint64_t)((addr & 0x3FFFF) >> 4);
  d |= (uint64_t)(128 >> 4) << 16;
  d |= (uint64_t)(128 >> 4) << 32;
  return d;
}

// D(64x8, fp32) += A(64x16, fp16, registers) * B(16x8, fp16, shared memory)
__device__ __forceinline__ void mma_rs(float (&d)[4], const uint32_t (&a)[4], uint64_t desc, int scale_d) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %9, 0;\n"
      " wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0;\n}"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));
}

// (x & mask) | magic as ONE LOP3 (one immediate per SASS LOP3, so magic must be in a register):
// lop_lo gives fp16 1024 + c of nibbles at bits 0-3 / 16-19; lop_hi gives 1024 + 16c of nibbles at bits 4-7 / 20-23.
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
__device__ __forceinline__ uint32_t opaque_magic() {
  uint32_t m;
  asm volatile("mov.b32 %0, 0x64006400;" : "=r"(m));
  return m;
}

__device__ __forceinline__ uint32_t hadd2_u32(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("add.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ void merge_ml(float& m, float& l, float m2, float l2) {
  float mn = fmaxf(m, m2);
  if (mn == -INFINITY) return;
  l = l * exp2f(m - mn) + l2 * exp2f(m2 - mn);
  m = mn;
}

}  // namespace kv4cu
