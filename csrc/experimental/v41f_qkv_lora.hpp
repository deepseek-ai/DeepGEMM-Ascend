#pragma once

#include <torch/torch.h>
#include <deep_gemm/experimental/v41f_qkv_lora.hpp>
#include "../jit/jit.hpp"
#include "../utils/math.hpp"

namespace deep_gemm::experimental {

static void v41f_qkv_lora(
    const std::pair<torch::Tensor, torch::Tensor>& x,
    const std::pair<torch::Tensor, torch::Tensor>& qkv_weight,
    const std::pair<torch::Tensor, torch::Tensor>& q_b_weight,
    const torch::Tensor& q_norm_weight, const torch::Tensor& kv_norm_weight,
    const torch::Tensor& positions, const torch::Tensor& cos_sin, const torch::Tensor& slots,
    const torch::Tensor& q, const torch::Tensor& cache, double eps
) {
    // Validate device placement and problem dimensions
    constexpr uint32_t kHidden = 5120, kQLoraDim = 1280, kKVLatentDim = 512, kRopeDim = 64, kQKVDim = kQLoraDim + kKVLatentDim;
    constexpr uint32_t kKVCacheSlotBytes = kKVLatentDim + kKVLatentDim / MX_SF_DIVISOR * sizeof(uint16_t);
    DJ_HOST_ASSERT(runtime->get_num_sms() == kNumAICores);
    const int64_t m = x.first.size(0), n = q_b_weight.first.size(0);
    DJ_HOST_ASSERT(m > 0 and m < 65536);
    DJ_HOST_ASSERT(n == 32768 or n == 36864);
    DJ_HOST_ASSERT(q.sizes() == at::IntArrayRef({m, n}));
    for (const auto& t: {x.first, x.second, qkv_weight.first, qkv_weight.second, q_b_weight.first,
                         q_b_weight.second, q_norm_weight, kv_norm_weight, positions, cos_sin, slots, q, cache})
        DJ_HOST_ASSERT(t.device() == x.first.device());

    // Validate projection and normalization tensors; GM inputs may be byte-aligned
    DJ_HOST_ASSERT(x.first.sizes() == at::IntArrayRef({m, kHidden}));
    DJ_HOST_ASSERT(qkv_weight.first.sizes() == at::IntArrayRef({kQKVDim, kHidden}));
    DJ_HOST_ASSERT(q_b_weight.first.sizes() == at::IntArrayRef({n, kQLoraDim}));
    for (const auto& [data, sf]: {x, qkv_weight, q_b_weight}) {
        DJ_HOST_ASSERT(data.scalar_type() == at::kFloat8_e4m3fn and data.is_contiguous());
        DJ_HOST_ASSERT(sf.scalar_type() == at::kShort and sf.sizes() == at::IntArrayRef({data.size(0), data.size(1) / MX_SF_DIVISOR}));
        DJ_HOST_ASSERT((sf.size(0) == 1 or sf.stride(0) == 1) and sf.stride(1) >= sf.size(0));
    }
    DJ_HOST_ASSERT(qkv_weight.second.stride(1) == kQKVDim and q_b_weight.second.stride(1) == n);
    for (const auto& t: {q_norm_weight, kv_norm_weight, q})
        DJ_HOST_ASSERT(t.scalar_type() == at::kBFloat16 and t.is_contiguous());
    for (const auto& t: {positions, slots})
        DJ_HOST_ASSERT(t.scalar_type() == at::kInt and t.is_contiguous() and t.sizes() == at::IntArrayRef({m}));
    DJ_HOST_ASSERT(q_norm_weight.sizes() == at::IntArrayRef({kQLoraDim}));
    DJ_HOST_ASSERT(kv_norm_weight.sizes() == at::IntArrayRef({kKVLatentDim}));
    DJ_HOST_ASSERT(reinterpret_cast<uintptr_t>(q.data_ptr()) % 32 == 0);

    // Validate RoPE tables and KV cache layout
    DJ_HOST_ASSERT(cos_sin.dim() == 2 and cos_sin.size(0) > 0 and cos_sin.size(1) == kRopeDim);
    DJ_HOST_ASSERT(cos_sin.scalar_type() == at::kFloat and cos_sin.is_contiguous());
    DJ_HOST_ASSERT(cache.dim() == 4 and cache.size(0) > 0 and cache.size(1) > 0);
    DJ_HOST_ASSERT(cache.size(2) == 1 and cache.size(3) == kKVCacheSlotBytes);
    DJ_HOST_ASSERT(cache.scalar_type() == at::kByte and cache.stride(1) == kKVCacheSlotBytes and cache.stride(3) == 1);
    DJ_HOST_ASSERT(cache.stride(0) >= cache.size(1) * kKVCacheSlotBytes);
    const int64_t num_slots = cache.size(0) * cache.size(1);

    // Instantiate, compile and launch the fused kernel
    const uint32_t block_m = std::min<int64_t>(aligned(m, FRAC_MN), 256);
    const auto code = std::format(R"(
#include <deep_gemm/experimental/v41f_qkv_lora.hpp>

using namespace deep_gemm::experimental;
static void __instantiate() {{
    auto p = reinterpret_cast<void*>(&v41f_qkv_lora_kernel<{}, {}, {}>);
}}
)", n, block_m, m <= block_m);
    if (runtime->get_dry_run()) {
        jit->compile_without_load("v41f_qkv_lora", code);
        return;
    }
    const auto kernel = jit->compile("v41f_qkv_lora", code);
    const auto qkv = torch::empty({m, kQKVDim}, q.options());
    const auto qa = torch::empty({m, kQLoraDim}, x.first.options());
    const auto qa_sf = torch::empty({m, kQLoraDim / MX_SF_DIVISOR}, x.second.options());
    launch_replicated(kernel, {.num_blocks = kNumAICores}, QKVLoRAArgs{
        .x = x.first.data_ptr(), .x_sf = x.second.data_ptr(),
        .wa = qkv_weight.first.data_ptr(), .wa_sf = qkv_weight.second.data_ptr(),
        .qkv = qkv.data_ptr(), .wb = q_b_weight.first.data_ptr(),
        .shape_m = uint32_t(m), .x_sf_stride = uint32_t(x.second.stride(1)),
        .eps_hi = float(eps), .eps_lo = float(eps - double(float(eps))),
        .qa = qa.data_ptr(), .qa_sf = qa_sf.data_ptr(), .wb_sf = q_b_weight.second.data_ptr(),
        .q = q.data_ptr(), .qw = q_norm_weight.data_ptr(), .kw = kv_norm_weight.data_ptr(),
        .cos_sin = cos_sin.data_ptr(), .positions = positions.data_ptr(),
        .slots = slots.data_ptr(), .cache = cache.data_ptr(),
        .num_slots = uint32_t(num_slots), .cache_block_size = uint32_t(cache.size(1)),
        .cache_page_stride_bytes = uint32_t(cache.stride(0))});
}

} // namespace deep_gemm::experimental
