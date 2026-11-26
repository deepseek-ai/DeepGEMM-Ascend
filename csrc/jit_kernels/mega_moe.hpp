#pragma once

#include <torch/types.h>

#include <deep_gemm/layout/mega_moe.hpp>

#include "../jit/jit.hpp"

namespace deep_gemm {

struct MegaMoEArgs {
    const mega_moe::layout::Workspace& workspace;

    mega_moe::layout::SymBuffer sym_buffer;
    const torch::Tensor& y;
    int32_t* cumulative_local_expert_recv_stats;
    mega_moe::layout::Weights weights;
    uint32_t num_tokens;
    float activation_clamp;
    void* rms_weight;
    float rms_eps;
    std::optional<bool> low_latency;
};

static void launch_mega_moe(const MegaMoEArgs& args) {
    const auto& workspace = args.workspace;
    const auto& config = workspace.config;
    const auto num_local_experts = config.num_experts / config.num_ranks;
    bool is_low_latency = false;
    if (args.low_latency.has_value()) {
        is_low_latency = args.low_latency.value();
        if (is_low_latency)
            DJ_HOST_ASSERT(args.num_tokens <= mega_moe::layout::kNumFastSrcSlotsPerExpert);
    } else {
        // Derive a cross-rank-consistent default from registered capacity.
        const auto num_expected_m_blocks = num_local_experts *
            ceil_div(uint64_t{config.num_max_tokens_per_rank} * config.num_topk,
                     uint64_t{num_local_experts} * mega_moe::layout::BLOCK_M);
        is_low_latency = num_local_experts <= mega_moe::layout::kNumCompletionMBlocks and
                         num_expected_m_blocks <= mega_moe::layout::kNumCompletionMBlocks;
    }
    const auto code = std::format(R"(
#include <deep_gemm/mega_moe.hpp>

static void __instantiate() {{
    auto p = reinterpret_cast<void*>(&deep_gemm::mega_moe::mega_moe_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}

)",
        config.num_ranks, config.num_experts, config.num_shared_experts, config.num_max_tokens_per_rank,
        config.num_topk, config.hidden, config.intermediate_hidden,
        is_low_latency, workspace.num_max_progress_blocks,
        args.y.scalar_type() == torch::kFloat32 ? "float" : "bfloat16_t",
        args.rms_weight != nullptr);
    if (runtime->get_dry_run()) {
        jit->compile_without_load("mega_moe", code);
        return;
    }
    const auto kernel = jit->compile("mega_moe", code);

    const auto stream = get_current_npu_stream();
    jit->launch(kernel, {.stream = stream, .num_blocks = kNumAICores},
                args.y.data_ptr(), args.cumulative_local_expert_recv_stats, args.weights, args.num_tokens, args.activation_clamp,
                args.sym_buffer, args.rms_weight, args.rms_eps);
}

} // namespace deep_gemm
