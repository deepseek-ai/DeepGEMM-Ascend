#pragma once

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/layout/mega_moe.hpp>

namespace deep_gemm::mega_moe::sched {

// kNone means no current task
enum class BlockPhase : uint32_t {kNone, kLinear1, kLinear2, kSharedLinear1, kSharedLinear2};

struct LinearTaskState {
    uint32_t shared_l1_task_base = 0;
    uint32_t l1_task_base = 0;
    uint32_t l2_task_base = 0;
    uint32_t shared_l2_task_base = 0;
};

// Map logical M-blocks to nonempty local experts
template <const auto& workspace>
struct ExpertCursor {
    uint32_t pool_block_end = 0;  // Exclusive M-block end
    uint32_t local_expert_idx = ~uint32_t{0};  // First increment wraps to expert 0
    uint32_t num_expert_tokens = 0;

    __aicore__ void next_expert(uint32_t num_tokens) {
        ++ local_expert_idx;
        num_expert_tokens = num_tokens;
        pool_block_end += ceil_div(num_tokens, layout::BLOCK_M);
    }

    // Balance FRAC_MN groups; expert M is compact, pool blocks keep the BLOCK_M stride
    // Return valid_m and write the expert-relative M start
    __aicore__ uint32_t get_m_block_range(uint32_t pool_block_idx, uint32_t& expert_m_begin) const {
        const uint32_t num_expert_m_blocks = ceil_div(num_expert_tokens, layout::BLOCK_M);
        const uint32_t expert_m_block_idx = pool_block_idx + num_expert_m_blocks - pool_block_end;
        const uint32_t num_m_groups = ceil_div(num_expert_tokens, FRAC_MN);
        const uint32_t num_groups_per_block = num_m_groups / num_expert_m_blocks;
        const uint32_t num_larger_m_blocks = num_m_groups % num_expert_m_blocks;

        expert_m_begin = (expert_m_block_idx * num_groups_per_block + min(expert_m_block_idx, num_larger_m_blocks)) * FRAC_MN;
        const uint32_t aligned_valid_m = (num_groups_per_block + (expert_m_block_idx < num_larger_m_blocks)) * FRAC_MN;
        return min(aligned_valid_m, num_expert_tokens - expert_m_begin);
    }

    // Advance the expert cursor and return valid rows
    __aicore__ uint32_t advance_to(__gm__ uint8_t* buffer, uint32_t pool_block_idx) {
        while (pool_block_idx >= pool_block_end)
            next_expert(asc_load_dev(workspace.metadata.expert_recv_count_sum.get_ptr(buffer, local_expert_idx + 1)) &
                        layout::kCountValueMask);

        uint32_t expert_m_begin;
        return get_m_block_range(pool_block_idx, expert_m_begin);
    }
};

// Replay the same routed task order on AIC and AIVs
// With kNone, metadata_ready distinguishes a pending schedule from an exhausted one
template <const auto& workspace>
__aicore__ inline uint32_t take_linear_task(uint32_t num_shared_tokens, uint32_t num_total_m_blocks,
                                            LinearTaskState& state, uint32_t core_idx,
                                            BlockPhase& block_phase, bool metadata_ready = true) {
    constexpr auto& config = workspace.config;
    constexpr uint32_t kNumAICores = config.num_ai_cores;
    constexpr uint32_t kNumLinear1NBlocks = config.intermediate_hidden / (layout::BLOCK_N / 2);
    constexpr uint32_t kNumLinear2NBlocks = config.hidden / layout::BLOCK_N;
    constexpr uint32_t kNumSharedLinear1NBlocks = config.shared_intermediate_hidden / (layout::BLOCK_N / 2);
    constexpr bool kHasSharedExperts = config.num_shared_experts > 0;

    // Schedule shared Linear1 before shared Linear2
    if constexpr (kHasSharedExperts) {
        const uint32_t num_shared_m_blocks = ceil_div(num_shared_tokens, layout::BLOCK_M);
        #pragma unroll
        for (uint32_t layer_idx = 0; layer_idx < 2; ++ layer_idx) {
            auto& task_base = layer_idx == 0 ? state.shared_l1_task_base : state.shared_l2_task_base;
            const uint32_t num_tasks = num_shared_m_blocks * (layer_idx == 0 ? kNumSharedLinear1NBlocks : kNumLinear2NBlocks);
            if (task_base >= num_tasks)
                continue;

            // Advance past waves with no task for this core
            const uint32_t task_idx = task_base + core_idx;
            task_base += kNumAICores;
            if (task_idx < num_tasks) {
                block_phase = layer_idx == 0 ? BlockPhase::kSharedLinear1 : BlockPhase::kSharedLinear2;
                return task_idx;
            }
        }
    }

    // Routed tasks require metadata
    if (not metadata_ready) {
        block_phase = BlockPhase::kNone;
        return ~uint32_t{0};
    }

    // Prepare the next Linear2 wave with one Linear1 wave of lookahead
    const uint32_t num_routed_tasks[2]{num_total_m_blocks * kNumLinear1NBlocks, num_total_m_blocks * kNumLinear2NBlocks};

    while (state.l1_task_base < num_routed_tasks[0] or state.l2_task_base < num_routed_tasks[1]) {
        bool is_linear1 = state.l2_task_base >= num_routed_tasks[1];
        if (not is_linear1 and state.l1_task_base < num_routed_tasks[0]) {
            const uint32_t required_linear1_task_end =
                ceil_div(min(state.l2_task_base + kNumAICores, num_routed_tasks[1]), kNumLinear2NBlocks) *
                kNumLinear1NBlocks;
            is_linear1 = state.l1_task_base < required_linear1_task_end + kNumAICores;
        }

        const uint32_t layer_idx = is_linear1 ? 0 : 1;
        auto& task_base = is_linear1 ? state.l1_task_base : state.l2_task_base;
        const uint32_t task_idx = task_base + core_idx;
        task_base += kNumAICores;
        if (task_idx < num_routed_tasks[layer_idx]) {
            block_phase = is_linear1 ? BlockPhase::kLinear1 : BlockPhase::kLinear2;
            return task_idx;
        }
    }

    block_phase = BlockPhase::kNone;
    return ~uint32_t{0};
}

} // namespace deep_gemm::mega_moe::sched
