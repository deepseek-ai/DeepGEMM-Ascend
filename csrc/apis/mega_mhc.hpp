#pragma once

#include <cstdint>
#include <limits>
#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>

#include "../jit_kernels/mega_mhc.hpp"

namespace deep_gemm::mega_mhc {

namespace py = pybind11;

/**
 * Execute shifted mHC, write the next residual/mixes, and apply
 * RMSNorm with optional FP8 E4M3 quantization. T is the token count; H is hidden.
 *
 * Inputs:
 *     x                    [T,H]                           BF16
 *     residual             [T,hc_mult,H]                   BF16
 *     shifted_prev_mix     [T,hc_mult,1]                   FP32
 *     post_mix             [T,hc_mult,1]                   FP32
 *     comb_res_mix         [T,hc_mult,hc_mult]             FP32
 *     fn                   [hc_mult*(hc_mult+2),hc_mult*H] FP32
 *     mix_scales           [3]                             FP32 (pre, post, comb)
 *     mix_bases            [hc_mult*(hc_mult+2)]           FP32 (pre, post, comb)
 *     rmsnorm_weight       [H]                             BF16
 *
 * Scalars:
 *     hc_mult              Number of HC routes; currently must be 4.
 *     hc_norm_eps          Epsilon for HC RMS normalization before mix logits.
 *     hc_pre_eps           Added after the pre-mix sigmoid.
 *     hc_post_scale        Multiplies the post-mix sigmoid.
 *     sinkhorn_eps         Epsilon for comb-mix Sinkhorn normalization.
 *     num_sinkhorn_iters   Number of Sinkhorn iterations; must be positive.
 *     rmsnorm_eps          Epsilon for output RMS normalization.
 *     rmsnorm_scale        Multiplies the output RMSNorm weight.
 *
 * Preallocated outputs:
 *     new_residual         [T,hc_mult,H]                   BF16
 *     new_prev_mix         [T,hc_mult,1]                   FP32
 *     new_post_mix         [T,hc_mult,1]                   FP32
 *     new_comb_res_mix     [T,hc_mult,hc_mult]             FP32
 *     y_bf16               [T,H] BF16 | None
 *     y_fp8                [T,H] float8_e4m3fn | None
 *     y_gemm_sf            [T,H/64] int16 column-major | None
 *     y_routed_sf          [T,H/64] int16 row-major | None
 *     y_shared_sf          Must be None on Ascend
 *
 * Returns None; results are written into the supplied output tensors.
 *
 * shifted_prev_mix supplies the previous sublayer's pre-mix for PreApply;
 * new_prev_mix receives the newly computed coefficients for the next sublayer.
 * y_bf16 and y_fp8 are independent optional outputs; at least one is required.
 * With y_fp8, supply exactly one of y_gemm_sf and y_routed_sf. FP8 is quantized
 * from the normalized value rounded to BF16, including FP8-only output.
 *
 * Each SF covers 32 H elements; two adjacent UE8M0 exponent bytes are packed
 * into one int16. y_routed_sf must be contiguous; y_gemm_sf must have
 * stride(0) == 1 and stride(1) >= T and divisible by 4.
 * Ascend MegaMoE shares routed SF with its shared experts; a separate
 * y_shared_sf is unsupported and shared_sf_block_m must be 0.
 * All other tensors must be contiguous. All tensors must share one NPU device,
 * or all be meta tensors in dry-run, which compiles without loading or launching.
 * H must be a positive multiple of 1024 and fit the kernel UB layout; T must fit in int32.
 * Nonempty execution requires 32 AI cores; T == 0 returns after validation.
 */
static void mega_mhc(const torch::Tensor& x,
                     const torch::Tensor& residual,
                     const torch::Tensor& shifted_prev_mix,
                     const torch::Tensor& post_mix,
                     const torch::Tensor& comb_res_mix,
                     const torch::Tensor& fn,
                     const torch::Tensor& mix_scales,
                     const torch::Tensor& mix_bases,
                     const int& hc_mult,
                     const float& hc_norm_eps,
                     const float& hc_pre_eps,
                     const float& hc_post_scale,
                     const float& sinkhorn_eps,
                     const int& num_sinkhorn_iters,
                     const torch::Tensor& rmsnorm_weight,
                     const float& rmsnorm_eps,
                     const float& rmsnorm_scale,
                     const torch::Tensor& new_residual,
                     const torch::Tensor& new_prev_mix,
                     const torch::Tensor& new_post_mix,
                     const torch::Tensor& new_comb_res_mix,
                     const std::optional<torch::Tensor>& y_bf16,
                     const std::optional<torch::Tensor>& y_fp8,
                     const std::optional<torch::Tensor>& y_gemm_sf,
                     const std::optional<torch::Tensor>& y_routed_sf,
                     const std::optional<torch::Tensor>& y_shared_sf,
                     const int& shared_sf_block_m) {
    using namespace layout::mega_mhc;
    constexpr uint32_t kHiddenPerSFWord = 64;

    DJ_HOST_ASSERT(y_bf16.has_value() or y_fp8.has_value(), "provide at least one of y_bf16 and y_fp8");
    DJ_HOST_ASSERT(not (y_gemm_sf.has_value() and y_routed_sf.has_value()), "provide only one of y_gemm_sf and y_routed_sf");
    const auto& y_sf = y_routed_sf.has_value() ? y_routed_sf : y_gemm_sf;
    DJ_HOST_ASSERT(y_fp8.has_value() == y_sf.has_value(), "y_fp8 requires exactly one of y_gemm_sf and y_routed_sf, and vice versa");
    DJ_HOST_ASSERT(not y_shared_sf.has_value(), "Ascend MegaMoE shares y_routed_sf; a separate y_shared_sf is unsupported");
    DJ_HOST_ASSERT(shared_sf_block_m == 0, "shared_sf_block_m must be 0 on Ascend");

    DJ_HOST_ASSERT(x.dim() == 2, "x must have shape [T, H]");
    const auto num_tokens = x.size(0);
    const auto hidden = x.size(1);
    DJ_HOST_ASSERT(num_tokens <= std::numeric_limits<int32_t>::max());
    DJ_HOST_ASSERT(hidden > 0 and hidden % 1024 == 0,
                   "hidden must be a positive multiple of 1024");
    DJ_HOST_ASSERT(hc_mult == kNumRoutes, "hc_mult must be 4");
    DJ_HOST_ASSERT(num_sinkhorn_iters >= 1, "num_sinkhorn_iters must be positive");

    const auto device = x.device();
    DJ_HOST_ASSERT(device.type() == c10::DeviceType::PrivateUse1 or (runtime->get_dry_run() and x.is_meta()),
                   "Mega mHC requires NPU tensors (meta tensors are supported only in dry-run)");
    const auto check_tensor = [&](const torch::Tensor& tensor, torch::ScalarType dtype,
                                  at::IntArrayRef shape, const char* name, bool dense = true) {
        DJ_HOST_ASSERT(tensor.device() == device, "{} must be on the same device as x", name);
        DJ_HOST_ASSERT(tensor.scalar_type() == dtype, "{} has an unexpected dtype", name);
        DJ_HOST_ASSERT(tensor.sizes() == shape, "{} has an unexpected shape", name);
        DJ_HOST_ASSERT(not dense or tensor.is_contiguous(), "{} must be contiguous", name);
    };
    check_tensor(x, torch::kBFloat16, {num_tokens, hidden}, "x");
    check_tensor(residual, torch::kBFloat16, {num_tokens, hc_mult, hidden}, "residual");
    check_tensor(post_mix, torch::kFloat, {num_tokens, hc_mult, 1}, "post_mix");
    check_tensor(comb_res_mix, torch::kFloat, {num_tokens, hc_mult, hc_mult}, "comb_res_mix");
    check_tensor(fn, torch::kFloat, {kNumHCOutputs, hc_mult * hidden}, "fn");
    check_tensor(mix_scales, torch::kFloat, {3}, "mix_scales");
    check_tensor(mix_bases, torch::kFloat, {kNumHCOutputs}, "mix_bases");
    check_tensor(rmsnorm_weight, torch::kBFloat16, {hidden}, "rmsnorm_weight");
    check_tensor(new_residual, torch::kBFloat16, residual.sizes(), "new_residual");
    check_tensor(new_post_mix, torch::kFloat, post_mix.sizes(), "new_post_mix");
    check_tensor(new_comb_res_mix, torch::kFloat, comb_res_mix.sizes(), "new_comb_res_mix");
    check_tensor(shifted_prev_mix, torch::kFloat, {num_tokens, hc_mult, 1}, "shifted_prev_mix");
    check_tensor(new_prev_mix, torch::kFloat, shifted_prev_mix.sizes(), "new_prev_mix");
    if (y_bf16.has_value())
        check_tensor(*y_bf16, torch::kBFloat16, x.sizes(), "y_bf16");
    if (y_fp8.has_value())
        check_tensor(*y_fp8, torch::kFloat8_e4m3fn, x.sizes(), "y_fp8");
    if (y_sf.has_value()) {
        check_tensor(*y_sf, torch::kShort, {num_tokens, hidden / kHiddenPerSFWord},
                     y_routed_sf.has_value() ? "y_routed_sf" : "y_gemm_sf", false);
        if (y_routed_sf.has_value())
            DJ_HOST_ASSERT(y_sf->is_contiguous(), "y_routed_sf must be contiguous row-major");
        else
            DJ_HOST_ASSERT(y_sf->stride(0) == 1 and y_sf->stride(1) >= num_tokens and y_sf->stride(1) % 4 == 0,
                           "y_gemm_sf must be column-major with stride(0) == 1 and a T stride divisible by 4 and at least T");
    }
    if (num_tokens == 0)
        return;

    launch_mega_mhc(x, residual, post_mix, comb_res_mix, new_residual, fn,
                    mix_scales, mix_bases, new_prev_mix, new_post_mix, new_comb_res_mix,
                    hc_norm_eps, hc_pre_eps, hc_post_scale, sinkhorn_eps, num_sinkhorn_iters, shifted_prev_mix,
                    rmsnorm_weight, rmsnorm_eps, rmsnorm_scale,
                    y_bf16, y_fp8, y_sf);
}

static void register_apis(pybind11::module_& m) {
    m.def("mega_mhc",
          &mega_mhc,
          py::arg("x"),
          py::arg("residual"),
          py::arg("shifted_prev_mix"),
          py::arg("post_mix"),
          py::arg("comb_res_mix"),
          py::arg("fn"),
          py::arg("mix_scales"),
          py::arg("mix_bases"),
          py::arg("hc_mult"),
          py::arg("hc_norm_eps"),
          py::arg("hc_pre_eps"),
          py::arg("hc_post_scale"),
          py::arg("sinkhorn_eps"),
          py::arg("num_sinkhorn_iters"),
          py::arg("rmsnorm_weight"),
          py::arg("rmsnorm_eps"),
          py::arg("rmsnorm_scale"),
          py::arg("new_residual"),
          py::arg("new_prev_mix"),
          py::arg("new_post_mix"),
          py::arg("new_comb_res_mix"),
          py::arg("y_bf16") = std::nullopt,
          py::arg("y_fp8") = std::nullopt,
          py::arg("y_gemm_sf") = std::nullopt,
          py::arg("y_routed_sf") = std::nullopt,
          py::arg("y_shared_sf") = std::nullopt,
          py::arg("shared_sf_block_m") = 0);
}

} // namespace deep_gemm::mega_mhc
