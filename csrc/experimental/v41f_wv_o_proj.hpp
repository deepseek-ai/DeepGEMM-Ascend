#pragma once

#include <torch/torch.h>

#include <deep_gemm/experimental/v41f_wv_o_proj.hpp>
#include "../jit/jit.hpp"
#include "../utils/math.hpp"

namespace deep_gemm::experimental {

static void v41f_wv_o_proj(const std::pair<torch::Tensor, torch::Tensor>& x,
                           const std::pair<torch::Tensor, torch::Tensor>& wv_weight,
                           const std::pair<torch::Tensor, torch::Tensor>& o_weight,
                           const torch::Tensor& y) {
    // Validate device placement and problem dimensions
    constexpr uint32_t kNumGroups = 8, kWvInputDim = 4096, kWvOutputDim = 1024;
    constexpr uint32_t kMidDim = kNumGroups * kWvOutputDim, kHidden = 5120;
    constexpr uint32_t kInputSFK = kWvInputDim / MX_SF_DIVISOR, kMidSFK = kMidDim / MX_SF_DIVISOR;
    DJ_HOST_ASSERT(runtime->get_num_sms() == kNumAICores);
    DJ_HOST_ASSERT(x.first.size(0) > 0);
    const int64_t m = x.first.size(0);
    for (const auto& t: {x.first, x.second, wv_weight.first, wv_weight.second, o_weight.first, o_weight.second, y})
        DJ_HOST_ASSERT(t.device() == x.first.device());

    // Validate operand layouts and MN-major scales; GM inputs may be byte-aligned
    for (const auto& [data, sf]: {x, wv_weight, o_weight}) {
        DJ_HOST_ASSERT(data.scalar_type() == at::kFloat8_e4m3fn and data.is_contiguous());
        DJ_HOST_ASSERT(sf.scalar_type() == at::kShort);
    }
    DJ_HOST_ASSERT(y.scalar_type() == at::kBFloat16 and y.is_contiguous());
    DJ_HOST_ASSERT(x.first.sizes() == at::IntArrayRef({m, kNumGroups, kWvInputDim}));
    DJ_HOST_ASSERT(x.second.sizes() == at::IntArrayRef({m, kNumGroups, kInputSFK}) and x.second.stride(2) >= m and
                   x.second.stride(1) == kInputSFK * x.second.stride(2) and (m == 1 or x.second.stride(0) == 1));
    DJ_HOST_ASSERT(wv_weight.first.sizes() == at::IntArrayRef({kNumGroups, kWvOutputDim, kWvInputDim}));
    DJ_HOST_ASSERT(wv_weight.second.sizes() == at::IntArrayRef({kNumGroups, kWvOutputDim, kInputSFK}) and
                   wv_weight.second.strides() == at::IntArrayRef({kInputSFK * kWvOutputDim, 1, kWvOutputDim}));
    DJ_HOST_ASSERT(o_weight.first.sizes() == at::IntArrayRef({kHidden, kMidDim}));
    DJ_HOST_ASSERT(o_weight.second.sizes() == at::IntArrayRef({kHidden, kMidSFK}) and
                   o_weight.second.strides() == at::IntArrayRef({1, kHidden}));
    DJ_HOST_ASSERT(y.sizes() == at::IntArrayRef({m, kHidden}));
    DJ_HOST_ASSERT(reinterpret_cast<uintptr_t>(y.data_ptr()) % 32 == 0);

    // Instantiate, compile and launch the fused kernel
    const uint32_t block_m = std::min<int64_t>(aligned(m, FRAC_MN), 256);
    const auto code = std::format(R"(
#include <deep_gemm/experimental/v41f_wv_o_proj.hpp>

using namespace deep_gemm::experimental;
static void __instantiate() {{
    auto p = reinterpret_cast<void*>(&v41f_wv_o_proj_kernel<{}, {}>);
}}
)", block_m, m <= block_m);
    if (runtime->get_dry_run()) {
        jit->compile_without_load("v41f_wv_o_proj", code);
        return;
    }
    const auto kernel = jit->compile("v41f_wv_o_proj", code);
    const auto mid = torch::empty({m, kNumGroups, kWvOutputDim}, x.first.options());
    const auto mid_sf = torch::empty({aligned(m, FRAC_MN), kMidSFK}, x.second.options());
    launch_replicated(kernel, {.num_blocks = kNumAICores}, WvOProjArgs{
        .x = x.first.data_ptr(), .x_sf = x.second.data_ptr(),
        .wv = wv_weight.first.data_ptr(), .wv_sf = wv_weight.second.data_ptr(),
        .o = o_weight.first.data_ptr(), .mid = mid.data_ptr(), .mid_sf = mid_sf.data_ptr(),
        .shape_m = uint32_t(m), .x_sf_stride = uint32_t(x.second.stride(2)),
        .o_sf = o_weight.second.data_ptr(), .y = y.data_ptr()});
}

} // namespace deep_gemm::experimental
