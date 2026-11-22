// Thin PyTorch binding for BitDecoding's Hopper decode kernel (OpenBitSys/BitDecoding, branch
// `hopper`, MIT), so it can be timed in the same process and with the same timer as our kernels.
// Only the build that branch ships is exposed: FP16, head dim 128, 4-bit, K and V both quantized
// per token ("k-tensor") with one asymmetric (scale, zero) pair per 128 values: 68 bytes per row,
// the same footprint as our int4a_tok.
#include <torch/extension.h>

#include "flash_api.h"

at::Tensor fwd(at::Tensor q, at::Tensor k, at::Tensor k_pack, at::Tensor k_params,
               at::Tensor v, at::Tensor v_pack, at::Tensor v_params, double softmax_scale,
               int64_t num_splits) {
    return mha_fwd_kvcache<4>(q, k, k_pack, k_params, v, v_pack, v_params,
                              static_cast<float>(softmax_scale), static_cast<int>(num_splits));
}

void pack(at::Tensor k, at::Tensor k_pack, at::Tensor k_params,
          at::Tensor v, at::Tensor v_pack, at::Tensor v_params,
          at::Tensor cu_seqlens_k, int64_t max_seqlen_k) {
    c10::optional<at::Tensor> block_table = c10::nullopt;
    kvcache_qpack<4>(k, k_pack, k_params, v, v_pack, v_params, block_table, cu_seqlens_k,
                     static_cast<int>(max_seqlen_k), "k-tensor", 128);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &fwd, "BitDecoding Hopper decode (fp16, d=128, 4-bit k-tensor, group 128)");
    m.def("pack", &pack, "BitDecoding quantize + pack of a contiguous K/V cache");
}
