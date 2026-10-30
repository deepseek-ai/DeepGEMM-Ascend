#pragma once

#include <vector>

#include <deep_gemm/layout/mega_moe.hpp>

#include "../jit/jit.hpp"

namespace deep_gemm {

struct MegaMoEArgs {
    const mega_moe::layout::Workspace& workspace;

    void* buffer;
    const std::vector<int64_t>& sym_buffer_ptrs;
    void* y;
    int32_t* cumulative_local_expert_recv_stats;
    mega_moe::layout::Weights weights;
    uint32_t rank_idx;
    uint32_t num_tokens;
    float activation_clamp;
};

static void launch_mega_moe(const MegaMoEArgs& args) {
    const auto& workspace = args.workspace;
    const auto& config = workspace.config;
    const auto code = std::format(R"(
#include <deep_gemm/mega_moe.hpp>

static void __instantiate() {{
    auto p = reinterpret_cast<void*>(&deep_gemm::mega_moe::mega_moe_impl<{}, {}, {}, {}, {}, {}, {}, {}>);
}}

)",
        config.num_ranks, config.num_experts, config.num_shared_experts, config.num_max_tokens_per_rank,
        config.num_topk, config.hidden, config.intermediate_hidden, config.num_ai_cores);
    if (runtime->get_dry_run()) {
        jit->compile_without_load("mega_moe", code);
        return;
    }
    const auto kernel = jit->compile("mega_moe", code);

    const auto stream = get_current_npu_stream();
    const auto progress = workspace.progress.blocks.get_ptr(args.buffer);
    DJ_ACL_CHECK(aclrtMemsetAsync(progress, workspace.num_progress_bytes, 0,
                                 workspace.num_progress_bytes, stream));

    jit->launch(kernel, {.stream = stream, .num_blocks = config.num_ai_cores},
                args.y, args.cumulative_local_expert_recv_stats, args.weights, args.num_tokens, args.activation_clamp,
                mega_moe::layout::SymBuffer<>(args.sym_buffer_ptrs, args.rank_idx));
}

} // namespace deep_gemm
