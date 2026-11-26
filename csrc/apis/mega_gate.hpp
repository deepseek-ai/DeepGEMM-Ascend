#pragma once

#include <cmath>
#include <optional>
#include <string>
#include <tuple>

#include <c10/core/DeviceGuard.h>
#include <deep_jit/cache/memory.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>

#include "../jit_kernels/mega_gate.hpp"

namespace deep_gemm::mega_gate {

namespace py = pybind11;

static std::tuple<torch::Tensor, torch::Tensor> bf16_mega_gate(
    const torch::Tensor& x, const torch::Tensor& weight, uint32_t num_topk,
    bool use_shared_as_routed, uint32_t num_shared_experts, float routed_scaling_factor, int ep_rank,
    const std::string& scoring_func, const std::optional<torch::Tensor>& mask,
    const std::optional<torch::Tensor>& bias, const std::optional<torch::Tensor>& image_bias,
    const std::optional<torch::Tensor>& image_token_mask, const std::optional<torch::Tensor>& fix_routing_mask,
    const std::optional<torch::Tensor>& to_physical_map, const std::optional<torch::Tensor>& logical_count,
    const std::optional<torch::Tensor>& unmapped_topk_idx, const std::optional<torch::Tensor>& force_random,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& out) {
    // Config checks
    DJ_HOST_ASSERT(x.dim() == 2 and weight.dim() == 2);
    DJ_HOST_ASSERT(x.device().type() == c10::DeviceType::PrivateUse1 or (runtime->get_dry_run() and x.is_meta()));
    const auto m = x.size(0), k = x.size(1), n = weight.size(0);
    DJ_HOST_ASSERT(m <= layout::kNumMaxTokens and k > 0 and k % 128 == 0 and n > 0 and n <= 512 and n % 4 == 0);
    const auto check_tensor = [&](const torch::Tensor& tensor, torch::ScalarType dtype, at::IntArrayRef shape, bool dense = true) {
        DJ_HOST_ASSERT(tensor.device() == x.device());
        DJ_HOST_ASSERT(tensor.scalar_type() == dtype and tensor.sizes() == shape);
        DJ_HOST_ASSERT(not dense or tensor.is_contiguous());
    };
    check_tensor(x, torch::kBFloat16, {m, k});
    check_tensor(weight, torch::kBFloat16, {n, k});
    DJ_HOST_ASSERT(num_topk > 0 and num_topk <= n and num_topk <= 32 and ep_rank >= 0 and std::isfinite(routed_scaling_factor));
    DJ_HOST_ASSERT(scoring_func == "sqrtsoftplus");
    if (not use_shared_as_routed) {
        num_shared_experts = 0;
    } else {
        DJ_HOST_ASSERT(num_shared_experts == 1 or num_shared_experts == 2);
        DJ_HOST_ASSERT(num_topk % num_shared_experts == 0 and n % (num_topk / num_shared_experts) == 0);
    }
    const uint32_t num_physical_topk = num_topk + num_shared_experts;
    DJ_HOST_ASSERT(num_physical_topk <= 32);

    // Optional routing inputs
    for (const auto& tensor : {mask, image_token_mask, fix_routing_mask, force_random})
        if (tensor.has_value()) check_tensor(*tensor, torch::kBool, {m});
    for (const auto& tensor : {bias, image_bias})
        if (tensor.has_value()) check_tensor(*tensor, torch::kFloat, {n});
    DJ_HOST_ASSERT(image_bias.has_value() == image_token_mask.has_value());
    DJ_HOST_ASSERT(to_physical_map.has_value() == logical_count.has_value());
    if (to_physical_map.has_value()) {
        DJ_HOST_ASSERT(to_physical_map->dim() == 2 and to_physical_map->size(1) > 0);
        check_tensor(*to_physical_map, torch::kInt, {n + num_shared_experts, to_physical_map->size(1)});
        check_tensor(*logical_count, torch::kInt, {n + num_shared_experts});
    }
    DJ_HOST_ASSERT(not fix_routing_mask.has_value() or unmapped_topk_idx.has_value());
    if (unmapped_topk_idx.has_value()) {
        check_tensor(*unmapped_topk_idx, torch::kLong, {m, num_topk}, false);
        DJ_HOST_ASSERT(unmapped_topk_idx->stride(1) == 1);
        DJ_HOST_ASSERT(m <= 1 or unmapped_topk_idx->stride(0) >= num_topk);
    }

    // Output tensors
    torch::Tensor topk_idx, topk_weights;
    if (out.has_value()) {
        std::tie(topk_idx, topk_weights) = *out;
        check_tensor(topk_idx, torch::kLong, {m, num_physical_topk});
        check_tensor(topk_weights, torch::kFloat, {m, num_physical_topk});
    } else {
        topk_idx = torch::empty({m, num_physical_topk}, x.options().dtype(torch::kInt64));
        topk_weights = torch::empty({m, num_physical_topk}, x.options().dtype(torch::kFloat32));
    }
    if (m == 0)
        return {topk_idx, topk_weights};

    const c10::OptionalDeviceGuard device_guard(x.is_meta() ? std::nullopt : std::optional(x.device()));
    // API entry holds the GIL; cached configurations contain no tensor or stream state.
    using Config = std::pair<GemmConfig, uint32_t>;
    static deep_jit::MemCache<std::string, Config> configs;
    const bool simple_routing = not (mask.has_value() or image_bias.has_value() or fix_routing_mask.has_value() or force_random.has_value());
    // Fix the lane width of each token's weight reduction, including partially filled batches.
    const uint32_t num_fixed_rows = runtime->get_deterministic_algorithms()
        ? layout::get_num_packed_rows(8, num_physical_topk, simple_routing) : 0;
    const auto key = std::format("{}/{}/{}/{}/{}/{}/{}/{}", m, n, k, num_topk, num_shared_experts,
                                 runtime->get_num_sms(), simple_routing, num_fixed_rows);
    const auto cached = configs.get_or_create(key, [&] {
        return std::make_shared<Config>(get_mega_gate_config(m, n, k, num_topk, num_shared_experts, simple_routing, num_fixed_rows));
    });
    const auto& [config, num_k_splits] = *cached;
    launch_mega_gate({
        .config = config,
        .x = x.data_ptr(),
        .weight = weight.data_ptr(),
        .gate = {
            .bias = bias.has_value() ? static_cast<float*>(bias->data_ptr()) : nullptr,
            .image_bias = image_bias.has_value() ? static_cast<float*>(image_bias->data_ptr()) : nullptr,
            .image_token_mask = image_token_mask.has_value() ? static_cast<bool*>(image_token_mask->data_ptr()) : nullptr,
            .mask = mask.has_value() ? static_cast<bool*>(mask->data_ptr()) : nullptr,
            .fix_routing_mask = fix_routing_mask.has_value() ? static_cast<bool*>(fix_routing_mask->data_ptr()) : nullptr,
            .to_physical_map = to_physical_map.has_value() ? static_cast<int32_t*>(to_physical_map->data_ptr()) : nullptr,
            .logical_count = logical_count.has_value() ? static_cast<int32_t*>(logical_count->data_ptr()) : nullptr,
            .topk_idx = static_cast<int64_t*>(topk_idx.data_ptr()),
            .unmapped_topk_idx = unmapped_topk_idx.has_value() ? static_cast<int64_t*>(unmapped_topk_idx->data_ptr()) : nullptr,
            .topk_weights = static_cast<float*>(topk_weights.data_ptr()),
            .force_random = force_random.has_value() ? static_cast<bool*>(force_random->data_ptr()) : nullptr,
            .unmapped_topk_idx_stride = unmapped_topk_idx.has_value() ? static_cast<uint64_t>(unmapped_topk_idx->stride(0)) : num_topk,
            .num_duplicate_experts = to_physical_map.has_value() ? static_cast<uint32_t>(to_physical_map->size(1)) : 0u,
            .ep_rank = static_cast<uint32_t>(ep_rank),
            .routed_scaling_factor = routed_scaling_factor,
        },
        .num_tokens = static_cast<uint32_t>(m),
        .num_experts = static_cast<uint32_t>(n),
        .hidden = static_cast<uint32_t>(k),
        .num_topk = num_topk,
        .num_shared_experts = num_shared_experts,
        .num_k_splits = num_k_splits,
        .num_fixed_rows = num_fixed_rows,
        .has_bias = bias.has_value(),
        .has_image_bias = image_bias.has_value(),
        .has_mask = mask.has_value(),
        .has_fixed_routing = fix_routing_mask.has_value(),
        .has_physical_map = to_physical_map.has_value(),
        .has_unmapped_output = unmapped_topk_idx.has_value(),
        .has_random_routing = force_random.has_value(),
    });
    return {topk_idx, topk_weights};
}

static void register_apis(py::module_& m) {
    m.def("bf16_mega_gate", &bf16_mega_gate,
          py::arg("x"), py::arg("weight"), py::arg("num_topk"), py::arg("use_shared_as_routed"),
          py::arg("num_shared_experts"), py::arg("routed_scaling_factor"), py::arg("ep_rank"),
          py::arg("scoring_func") = "sqrtsoftplus", py::arg("mask") = std::nullopt, py::arg("bias") = std::nullopt,
          py::arg("image_bias") = std::nullopt, py::arg("image_token_mask") = std::nullopt,
          py::arg("fix_routing_mask") = std::nullopt, py::arg("to_physical_map") = std::nullopt,
          py::arg("logical_count") = std::nullopt, py::arg("unmapped_topk_idx") = std::nullopt,
          py::arg("force_random") = std::nullopt, py::arg("out") = std::nullopt);
}

} // namespace deep_gemm::mega_gate
