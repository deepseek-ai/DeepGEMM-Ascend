#pragma once

#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <c10/core/DeviceGuard.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include <torch_npu/csrc/aten/common/from_blob.h>

#include <deep_gemm/layout/mega_moe.hpp>

#include "../backends/hccl.hpp"
#include "../jit/jit.hpp"
#include "../jit_kernels/mega_moe.hpp"

namespace deep_gemm::mega_moe {

namespace py = pybind11;

static layout::Workspace get_workspace(uint32_t num_ranks, uint32_t num_experts,
                                       uint32_t num_max_tokens_per_rank, uint32_t num_topk,
                                       uint32_t hidden, uint32_t intermediate_hidden, uint32_t num_shared_experts) {
    DJ_HOST_ASSERT(num_ranks > 0 and num_ranks < 256);
    DJ_HOST_ASSERT(num_experts > 0 and num_experts % num_ranks == 0);
    DJ_HOST_ASSERT(num_max_tokens_per_rank > 0);
    DJ_HOST_ASSERT(num_topk > 0 and num_topk <= 32);
    DJ_HOST_ASSERT(num_shared_experts <= 2);
    DJ_HOST_ASSERT(hidden > 0 and hidden % layout::BLOCK_N == 0);
    DJ_HOST_ASSERT(intermediate_hidden >= 512 and
                   intermediate_hidden % layout::BLOCK_N == 0);
    DJ_HOST_ASSERT(static_cast<uint64_t>(num_ranks) * num_max_tokens_per_rank * num_topk <
                   layout::kCountArrivalBit);
    return layout::Workspace{{num_ranks, num_experts, num_shared_experts, num_max_tokens_per_rank,
                              num_topk, hidden, intermediate_hidden}};
}

// The root Tensor owns the HCCL lease; data views borrow its memory
struct SymmBuffer {
    torch::Tensor buffer;
    std::vector<int64_t> buffer_ptrs;

    SymmBuffer(uint32_t rank_idx, uint32_t num_ranks, const std::string& hccl_group_name, int64_t num_bytes) {
        DJ_HOST_ASSERT(num_bytes > 0 and num_ranks > 0 and rank_idx < num_ranks);
        auto allocation = HCCLSymmetricBuffer::acquire(hccl_group_name, rank_idx, num_ranks, num_bytes);
        int32_t device_idx;
        DJ_ACL_CHECK(aclrtGetDevice(&device_idx));
        const c10::Device device(c10::DeviceType::PrivateUse1, device_idx);
        buffer = at_npu::native::from_blob(allocation->buffer, {num_bytes}, [allocation, device](void*) {
            // Wait for device work before the last owner returns the lease to the pool
            const c10::DeviceGuard guard(device);
            DJ_ACL_CHECK(aclrtSynchronizeDevice());
        }, torch::TensorOptions().dtype(torch::kInt8).device(device));
        buffer_ptrs.assign(allocation->peer_ptrs.begin(), allocation->peer_ptrs.end());
    }
};

static torch::Tensor slice_buffer(const torch::Tensor& buffer, uint64_t offset,
                                  torch::IntArrayRef shape, torch::ScalarType dtype) {
    DJ_HOST_ASSERT(buffer.dim() == 1 and buffer.scalar_type() == torch::kInt8 and buffer.is_contiguous());
    const auto num_bytes = c10::multiply_integers(shape) * c10::elementSize(dtype);
    DJ_HOST_ASSERT(offset <= buffer.nbytes() and num_bytes <= buffer.nbytes() - offset);
    return at_npu::native::from_blob(static_cast<uint8_t*>(buffer.data_ptr()) + offset, shape, buffer.options().dtype(dtype));
}

static auto get_symm_buffer_size_for_mega_moe(
    uint32_t num_ranks, uint32_t num_experts, uint32_t num_max_tokens_per_rank, uint32_t num_topk,
    uint32_t hidden, uint32_t intermediate_hidden, const std::string& mma_type,
    const std::string& activation, uint32_t num_shared_experts = 0) {
    DJ_HOST_ASSERT(mma_type == "fp8xfp4" and activation == "swiglu");
    const auto workspace = get_workspace(num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
                                         hidden, intermediate_hidden, num_shared_experts);
    auto slice_input_buffers = [workspace](const torch::Tensor& buffer) {
        DJ_HOST_ASSERT(buffer.nbytes() >= workspace.get_num_bytes());
        const auto& config = workspace.config;
        const int64_t num_tokens = config.num_max_tokens_per_rank;
        const int64_t hidden = config.hidden, intermediate_hidden = config.intermediate_hidden;
        const int64_t num_ring_tokens = workspace.num_ring_rows;
        const int64_t shared_intermediate_hidden = config.shared_intermediate_hidden;
        auto x = slice_buffer(buffer, workspace.input.acts.data.offset, {num_tokens, hidden}, torch::kFloat8_e4m3fn);
        auto x_sf = slice_buffer(buffer, workspace.input.acts.sf.offset, {num_tokens, config.num_input_sf_pairs}, torch::kInt16);
        auto topk_idx = slice_buffer(buffer, workspace.input.topk_idx.offset, {num_tokens, config.num_topk}, torch::kInt64);
        auto topk_weights = slice_buffer(buffer, workspace.input.topk_weights.offset, {num_tokens, config.num_topk}, torch::kFloat32);

        auto shared_l1_acts = x;
        auto shared_l1_acts_sf = config.num_shared_experts > 0 ? x_sf : torch::Tensor();
        auto shared_l2_acts = config.num_shared_experts > 0 ? slice_buffer(buffer, workspace.shared_l2_acts.data.offset,
            {num_tokens, shared_intermediate_hidden}, torch::kFloat8_e4m3fn) : torch::Tensor();
        // Linear2 SFs are physical storage: [M-block, K-pair, M] with fixed BLOCK_M
        auto shared_l2_acts_sf = config.num_shared_experts > 0 ? slice_buffer(buffer, workspace.shared_l2_acts.sf.offset,
            {workspace.num_padded_shared_rows * (shared_intermediate_hidden / MX_SF_DIVISOR)}, torch::kInt16) : torch::Tensor();
        auto l1_acts = slice_buffer(buffer, workspace.routed.l1_acts.data.offset, {num_ring_tokens, hidden}, torch::kFloat8_e4m3fn);
        auto l1_acts_sf = slice_buffer(buffer, workspace.routed.l1_acts.sf.offset, {num_ring_tokens, config.num_input_sf_pairs}, torch::kInt16);
        auto l2_acts = slice_buffer(buffer, workspace.routed.l2_acts.data.offset, {num_ring_tokens, intermediate_hidden}, torch::kFloat8_e4m3fn);
        auto l2_acts_sf = slice_buffer(buffer, workspace.routed.l2_acts.sf.offset,
            {num_ring_tokens * (intermediate_hidden / MX_SF_DIVISOR)}, torch::kInt16);
        auto expert_recv_count = slice_buffer(buffer, workspace.metadata.expert_recv_count.offset,
            {config.num_ranks, config.num_experts / config.num_ranks}, torch::kInt32);
        return std::make_tuple(x, x_sf, topk_idx, topk_weights,
                               shared_l1_acts, shared_l1_acts_sf, shared_l2_acts, shared_l2_acts_sf,
                               l1_acts, l1_acts_sf, l2_acts, l2_acts_sf, expert_recv_count);
    };
    return std::make_tuple(workspace.get_num_bytes(), std::function(slice_input_buffers));
}

static layout::LinearWeights check_weights(const std::tuple<torch::Tensor, torch::Tensor>& weights_tuple, uint32_t num_n_blocks,
                                           uint32_t num_k, uint32_t num_experts_per_rank = 0) {
    const auto& [weights, weights_sf] = weights_tuple;
    const bool is_shared = num_experts_per_rank == 0;
    DJ_HOST_ASSERT(weights.scalar_type() == (is_shared ? torch::kFloat8_e4m3fn : torch::kInt8) and weights.is_contiguous());
    DJ_HOST_ASSERT(weights_sf.scalar_type() == torch::kInt16 and weights_sf.is_contiguous());
    DJ_HOST_ASSERT(weights.device().type() == c10::DeviceType::PrivateUse1 and
                   weights_sf.device().type() == c10::DeviceType::PrivateUse1);
    // Shared weights omit the expert dimension and store unpacked FP8 data.
    const int64_t weight_shape[]{num_experts_per_rank, num_n_blocks, layout::BLOCK_N, num_k / (is_shared ? 1 : 2)};
    const int64_t sf_shape[]{num_experts_per_rank, num_n_blocks, num_k / MX_SF_DIVISOR, layout::BLOCK_N};
    DJ_HOST_ASSERT(weights.sizes() == torch::IntArrayRef(weight_shape + is_shared, 4 - is_shared));
    DJ_HOST_ASSERT(weights_sf.sizes() == torch::IntArrayRef(sf_shape + is_shared, 4 - is_shared));
    return {static_cast<const uint8_t*>(weights.data_ptr()), static_cast<const int16_t*>(weights_sf.data_ptr())};
}

static void fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& shared_l1_weights_tuple_opt,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& shared_l2_weights_tuple_opt,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const int& hidden, const int& intermediate_hidden, const int& num_shared_experts,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math
) {
    // Config checks
    const auto [rm, rn, rk] = recipe;
    DJ_HOST_ASSERT(rm == 1 and rn == 1 and rk == 32);
    DJ_HOST_ASSERT(activation == "swiglu" and fast_math);
    DJ_HOST_ASSERT(num_max_tokens_per_rank > 0 and num_experts > 0 and num_topk > 0);
    DJ_HOST_ASSERT(hidden > 0 and intermediate_hidden > 0 and num_shared_experts >= 0);
    const auto num_ranks = static_cast<uint32_t>(sym_buffer_ptrs.size());
    // Use the registered layout dimensions, then validate tensors against them.
    const auto workspace = get_workspace(num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
                                         hidden, intermediate_hidden, num_shared_experts);
    DJ_HOST_ASSERT(y.dim() == 2 and y.size(0) <= num_max_tokens_per_rank and y.size(1) == hidden);
    DJ_HOST_ASSERT(shared_l1_weights_tuple_opt.has_value() == (num_shared_experts > 0) and
                   shared_l2_weights_tuple_opt.has_value() == (num_shared_experts > 0));
    const auto num_tokens = static_cast<uint32_t>(y.size(0));
    DJ_HOST_ASSERT(runtime->get_num_sms() == kNumAICores);

    // Registered allocations must remain valid until completion
    DJ_HOST_ASSERT(rank_idx >= 0 and rank_idx < num_ranks);
    DJ_HOST_ASSERT(sym_buffer.dim() == 1 and sym_buffer.scalar_type() == torch::kInt8 and sym_buffer.is_contiguous());
    DJ_HOST_ASSERT(sym_buffer.nbytes() >= workspace.get_num_bytes());
    DJ_HOST_ASSERT(sym_buffer.device() == y.device());
    DJ_HOST_ASSERT(sym_buffer_ptrs[rank_idx] == reinterpret_cast<int64_t>(sym_buffer.data_ptr()));
    DJ_HOST_ASSERT(y.scalar_type() == torch::kBFloat16 and y.is_contiguous());
    DJ_HOST_ASSERT(y.device().type() == c10::DeviceType::PrivateUse1);

    // Check stats counter
    const uint32_t num_experts_per_rank = num_experts / num_ranks;
    if (cumulative_local_expert_recv_stats.has_value()) {
        DJ_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt32);
        DJ_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        DJ_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
        DJ_HOST_ASSERT(cumulative_local_expert_recv_stats->device() == y.device());
    }

    // Check transformed weights and scales
    const auto l1 = check_weights(l1_weights_tuple, intermediate_hidden / (layout::BLOCK_N / 2), hidden, num_experts_per_rank);
    const auto l2 = check_weights(l2_weights_tuple, hidden / layout::BLOCK_N, intermediate_hidden, num_experts_per_rank);
    layout::LinearWeights shared_l1{}, shared_l2{};
    if (num_shared_experts > 0) {
        shared_l1 = check_weights(shared_l1_weights_tuple_opt.value(),
                                  workspace.config.shared_intermediate_hidden / (layout::BLOCK_N / 2), hidden);
        shared_l2 = check_weights(shared_l2_weights_tuple_opt.value(),
                                  hidden / layout::BLOCK_N, workspace.config.shared_intermediate_hidden);
    }
    // NOTES: All weights/SFs must be on the current device, like y and sym_buffer
    const auto activation_clamp = activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DJ_HOST_ASSERT(activation_clamp >= 0);
    launch_mega_moe({
        .workspace = workspace,
        .buffer = sym_buffer.data_ptr(),
        .sym_buffer_ptrs = sym_buffer_ptrs,
        .y = y.data_ptr(),
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats.has_value() ?
            cumulative_local_expert_recv_stats->data_ptr<int32_t>() : nullptr,
        .weights = {l1, l2, shared_l1, shared_l2},
        .rank_idx = static_cast<uint32_t>(rank_idx),
        .num_tokens = num_tokens,
        .activation_clamp = static_cast<float>(c10::BFloat16(activation_clamp)),
    });
}

static void register_apis(py::module_& m) {
    m.attr("_mega_moe_block_n") = layout::BLOCK_N;
    py::class_<SymmBuffer>(m, "SymmBuffer")
        .def(py::init<uint32_t, uint32_t, const std::string&, int64_t>(),
             py::arg("rank_idx"), py::arg("num_ranks"), py::arg("hccl_group_name"), py::arg("num_bytes"))
        .def_readonly("buffer", &SymmBuffer::buffer)
        .def_readonly("buffer_ptrs", &SymmBuffer::buffer_ptrs);
    m.def("get_symm_buffer_size_for_mega_moe", &get_symm_buffer_size_for_mega_moe,
          py::arg("num_ranks"), py::arg("num_experts"), py::arg("num_max_tokens_per_rank"), py::arg("num_topk"),
          py::arg("hidden"), py::arg("intermediate_hidden"), py::arg("mma_type"), py::arg("activation"),
          py::arg("num_shared_experts") = 0);
    m.def("fp8_fp4_mega_moe", &fp8_fp4_mega_moe,
          py::arg("y"), py::arg("l1_weights_tuple"), py::arg("l2_weights_tuple"),
          py::arg("shared_l1_weights_tuple_opt"), py::arg("shared_l2_weights_tuple_opt"),
          py::arg("cumulative_local_expert_recv_stats"), py::arg("sym_buffer"),
          py::arg("sym_buffer_ptrs"), py::arg("rank_idx"), py::arg("num_max_tokens_per_rank"),
          py::arg("num_experts"), py::arg("num_topk"), py::arg("hidden"),
          py::arg("intermediate_hidden"), py::arg("num_shared_experts"), py::arg("recipe"),
          py::arg("activation"), py::arg("activation_clamp_opt"), py::arg("fast_math"));
}

} // namespace deep_gemm::mega_moe
