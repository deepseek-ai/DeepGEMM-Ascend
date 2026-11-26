#pragma once

#include <cstdio>
#include <cstdint>
#include <optional>
#include <c10/core/DeviceGuard.h>
#include <torch/torch.h>

#include <deep_gemm/layout/mega_mhc.hpp>

#include "../jit/jit.hpp"
#include "../utils/math.hpp"

namespace deep_gemm::mega_mhc {

// Use M8 for small batches
static int get_epilogue_block_m(int64_t shape_m) {
    return shape_m <= 96 ? 8 : 16;
}

static uint32_t get_norm_block_h(int64_t shape_m, int64_t shape_h, int epilogue_block_m) {
    using namespace layout::mega_mhc;

    const auto num_mix_tasks = ceil_div(shape_m, epilogue_block_m);
    const auto num_norm_tasks = num_mix_tasks * (shape_h / 1024);
    // Use H1024 when Norm/Quant and Mix fit on separate AIVs
    if (num_norm_tasks + num_mix_tasks <= kNumAIVCores)
        return 1024;
    return shape_h <= 8192 ? shape_h / 4 : (shape_h % 2048 == 0 ? 2048 : 1024);
}

// Use M16 only when it balances AIC work and Norm/Quant fits one wave
static uint32_t get_main_block_m(int64_t shape_m, int64_t shape_h, int epilogue_block_m, uint32_t norm_block_h) {
    using namespace layout::mega_mhc;

    const auto num_epilogue_tasks = ((shape_m + epilogue_block_m - 1) / epilogue_block_m) * (shape_h / norm_block_h);
    // Keep M32 when Norm/Quant needs more than one AIV wave
    if (num_epilogue_tasks > kNumAIVCores)
        return BLOCK_M;

    // Count valid rows on the busiest AIC, including the partial final block
    const auto max_rows_per_aic = [&](int64_t block_m) {
        const auto rows_per_chunk = kNumAICoresPerSplit * block_m;
        return shape_m / rows_per_chunk * block_m + std::min(shape_m % rows_per_chunk, block_m);
    };
    return max_rows_per_aic(BLOCK_M) - max_rows_per_aic(BLOCK_M / 2) >= BLOCK_M / 2 ? BLOCK_M / 2 : BLOCK_M;
}

// Compile and launch on the current stream
static void launch_mega_mhc(const torch::Tensor& x, const torch::Tensor& residual,
                            const torch::Tensor& post_mix, const torch::Tensor& comb_res_mix,
                            const torch::Tensor& new_residual, const torch::Tensor& fn,
                            const torch::Tensor& mix_scales, const torch::Tensor& mix_bases,
                            const torch::Tensor& new_prev_mix, const torch::Tensor& new_post_mix,
                            const torch::Tensor& new_comb_res_mix,
                            float hc_norm_eps, float hc_pre_eps, float hc_post_scale,
                            float sinkhorn_eps, int num_sinkhorn_iters,
                            const torch::Tensor& shifted_prev_mix,
                            const torch::Tensor& rmsnorm_weight, float rmsnorm_eps, float rmsnorm_scale,
                            const std::optional<torch::Tensor>& y_bf16, const std::optional<torch::Tensor>& y_fp8,
                            const std::optional<torch::Tensor>& y_sf) {
    using namespace layout::mega_mhc;

    const auto shape_m = x.size(0);
    const auto shape_h = x.size(1);
    const bool is_sf_row_major = y_sf.has_value() && y_sf->stride(1) == 1;
    const c10::OptionalDeviceGuard device_guard(x.is_meta() ? std::nullopt : std::make_optional(x.device()));
    const uint32_t num_ai_cores = runtime->get_num_sms();
    DJ_HOST_ASSERT(num_ai_cores == kNumAICores);
    const int epilogue_block_m = get_epilogue_block_m(shape_m);
    const uint32_t norm_block_h = get_norm_block_h(shape_m, shape_h, epilogue_block_m);
    // Assign Mix from the last AIV if its M blocks and the Norm/Quant H tiles fit on separate cores
    const auto tasks_per_m_block = 1 + shape_h / norm_block_h;
    const bool mix_on_tail_aivs = shape_m <= epilogue_block_m * (kNumAIVCores / tasks_per_m_block);
    const uint32_t block_m = get_main_block_m(shape_m, shape_h, epilogue_block_m, norm_block_h);
    // Keep one coefficient block per resident M block on the busiest AIC
    const uint32_t num_l1_coeff_stages = ceil_div(std::min<int64_t>(shape_m, kNumTokensPerChunk), kNumAICoresPerSplit * block_m);
    DJ_HOST_ASSERT(not mix_on_tail_aivs or shape_m <= kNumTokensPerChunk);
    // Keep early BF16/SF stores independent of the RMS workspace layout
    const bool split_norm_quant = y_bf16.has_value() and y_fp8.has_value() and mix_on_tail_aivs;
    const bool full_m_blocks = shape_m % block_m == 0;
    const auto code = std::format(R"(
#include <deep_gemm/mega_mhc.hpp>

static void __instantiate() {{
    auto p = reinterpret_cast<void*>(&deep_gemm::mega_mhc::mega_mhc_kernel<
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", shape_h * kNumRoutes, norm_block_h, num_l1_coeff_stages,
        is_sf_row_major, y_bf16.has_value(), y_fp8.has_value(), epilogue_block_m, mix_on_tail_aivs, block_m, split_norm_quant, full_m_blocks);
    if (runtime->get_dry_run()) {
        jit->compile_without_load("mega_mhc", code, {.optimize_level = "3"});
        return;
    }
    const auto kernel = jit->compile("mega_mhc", code, {.optimize_level = "3"});

    const MegaHCWorkspaceLayout<kNumHCOutputs, kNumKSplits> workspace_layout(shape_m);
    // Shifted needs intermediate BF16 Y even when only FP8 is requested
    const uint64_t num_y_bytes = !y_bf16.has_value() ? uint64_t(shape_m) * shape_h * sizeof(uint16_t) : 0;
    const auto workspace = torch::empty({static_cast<int64_t>(workspace_layout.num_bytes + num_y_bytes)}, x.options().dtype(torch::kByte));
    const uintptr_t y_bf16_addr = y_bf16.has_value() ? reinterpret_cast<uintptr_t>(y_bf16->data_ptr()) :
        reinterpret_cast<uintptr_t>(workspace.data_ptr()) + workspace_layout.num_bytes;
    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS"))
        printf("%s\n", std::format("Mega shifted mHC: M={}, H={}, cores={}, workspace={} bytes",
                                   shape_m, shape_h, num_ai_cores, workspace_layout.num_bytes + num_y_bytes).c_str());

    const MixArgs mix_args{
        .scales = gm_ptr<float>{3, reinterpret_cast<uintptr_t>(mix_scales.data_ptr())},
        .bases = gm_ptr<float>{kNumHCOutputs, reinterpret_cast<uintptr_t>(mix_bases.data_ptr())},
        .new_prev_mix = gm_ptr<float>{kNumRoutes, reinterpret_cast<uintptr_t>(new_prev_mix.data_ptr())},
        .new_post_mix = gm_ptr<float>{kNumRoutes, reinterpret_cast<uintptr_t>(new_post_mix.data_ptr())},
        .new_comb_res_mix = gm_ptr<float>{kNumRoutes, reinterpret_cast<uintptr_t>(new_comb_res_mix.data_ptr()), kNumRoutes * kNumRoutes},
        .hc_norm_eps = hc_norm_eps, .hc_pre_eps = hc_pre_eps, .hc_post_scale = hc_post_scale,
        .sinkhorn_eps = sinkhorn_eps, .num_sinkhorn_iters = static_cast<uint32_t>(num_sinkhorn_iters)};
    const NormArgs norm_args{
        .shape_m = static_cast<uint32_t>(shape_m),
        .weight = gm_ptr<uint16_t>{static_cast<uint64_t>(shape_h), reinterpret_cast<uintptr_t>(rmsnorm_weight.data_ptr())},
        .new_residual = gm_ptr<uint16_t>{static_cast<uint64_t>(kNumRoutes * shape_h), reinterpret_cast<uintptr_t>(new_residual.data_ptr())},
        .eps = rmsnorm_eps, .scale = rmsnorm_scale,
        .y_bf16 = gm_ptr<uint16_t>{static_cast<uint64_t>(shape_h), y_bf16_addr},
        .y_fp8 = gm_ptr<uint8_t>{static_cast<uint64_t>(shape_h), y_fp8.has_value() ? reinterpret_cast<uintptr_t>(y_fp8->data_ptr()) : 0},
        .y_sf = gm_ptr<uint16_t>{static_cast<uint64_t>(shape_h / MX_SF_DIVISOR), y_sf.has_value() ? reinterpret_cast<uintptr_t>(y_sf->data_ptr()) : 0},
        .y_sf_stride = y_sf.has_value() ? static_cast<uint32_t>(y_sf->stride(is_sf_row_major ? 0 : 1)) : 0u};

    launch_replicated(kernel, {.num_blocks = static_cast<int>(num_ai_cores)}, KernelArgs{
        .gm_residual = gm_ptr<uint16_t>{static_cast<uint64_t>(kNumRoutes * shape_h), reinterpret_cast<uintptr_t>(residual.data_ptr())},
        .gm_fn = gm_ptr<float>{static_cast<uint64_t>(kNumRoutes * shape_h), reinterpret_cast<uintptr_t>(fn.data_ptr())},
        .workspace = reinterpret_cast<float*>(workspace.data_ptr()),
        .norm_args = norm_args,
        .gm_layer_input = gm_ptr<uint16_t>{static_cast<uint64_t>(shape_h), reinterpret_cast<uintptr_t>(x.data_ptr())},
        .gm_shifted_prev_mix = gm_ptr<float>{kNumRoutes, reinterpret_cast<uintptr_t>(shifted_prev_mix.data_ptr())},
        .gm_post_mix = gm_ptr<float>{kNumRoutes, reinterpret_cast<uintptr_t>(post_mix.data_ptr())},
        .gm_comb_res_mix = gm_ptr<float>{kNumRoutes, reinterpret_cast<uintptr_t>(comb_res_mix.data_ptr()), kNumRoutes * kNumRoutes},
        .mix_args = mix_args});
}

} // namespace deep_gemm::mega_mhc
