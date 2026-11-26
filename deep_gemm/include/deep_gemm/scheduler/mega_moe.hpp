#pragma once

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/layout/mega_moe.hpp>

namespace deep_gemm::mega_moe::sched {

// kNone means no current task; kDone means the schedule is exhausted
enum class BlockPhase : uint32_t {kNone, kLinear1, kLinear2, kSharedLinear1, kSharedLinear2, kDone};

struct LinearTaskState {
    uint32_t task_base[static_cast<uint32_t>(BlockPhase::kDone) - 1]{};  // BlockPhase order: routed Linear1/2, then shared Linear1/2
};

// Map logical M-blocks to nonempty local experts
template <const auto& kWorkspace, bool kLowLatency>
struct ExpertCursor {
    uint32_t pool_block_end = 0;  // Exclusive M-block end
    uint32_t local_expert_idx = ~uint32_t{0};  // First increment wraps to expert 0
    uint32_t num_expert_tokens = 0;
    uint32_t num_expert_m_blocks = 0;

    // Advance the expert cursor and balance its M fracs between M-blocks
    // Return valid rows and optionally write the expert-relative M start
    __aicore__ uint32_t advance_to(__gm__ uint8_t* buffer, uint32_t pool_block_idx, uint32_t* expert_m_begin = nullptr, bool check_metadata = false) {
        constexpr auto& kConfig = kWorkspace.config;
        constexpr uint32_t kNumLocalExperts = kConfig.num_experts / kConfig.num_ranks;
        while (pool_block_idx >= pool_block_end) {
            const uint32_t next_expert = local_expert_idx + 1u;
            // Zero means pending when checking metadata; real M-blocks always have valid rows.
            if (check_metadata and (next_expert >= kNumLocalExperts or asc_load_dev(kWorkspace.progress.metadata_ready.get_ptr(buffer, next_expert % kNumAICores)) == 0))
                return 0;
            if (check_metadata)
                asc_sync_data_barrier(DSB_DDR);
            local_expert_idx = next_expert;
            const auto src_m_end_ptr = kWorkspace.expert_src_m_end.get_ptr(buffer, local_expert_idx * kConfig.src_m_end_gm_stride);
            num_expert_tokens = asc_load_dev(src_m_end_ptr + kConfig.num_ranks - 1) & layout::kCountValueMask;
            num_expert_m_blocks = kConfig.template get_routed_m_blocks<kLowLatency>(num_expert_tokens);
            pool_block_end += num_expert_m_blocks;
        }
        const uint32_t expert_m_block_idx = pool_block_idx + num_expert_m_blocks - pool_block_end;
        const uint32_t num_m_fracs = ceil_div(num_expert_tokens, FRAC_MN);
        const uint32_t num_m_fracs_per_block = num_m_fracs / num_expert_m_blocks;
        const uint32_t num_larger_m_blocks = num_m_fracs % num_expert_m_blocks;

        const uint32_t m_begin = (expert_m_block_idx * num_m_fracs_per_block + min(expert_m_block_idx, num_larger_m_blocks)) * FRAC_MN;
        if (expert_m_begin != nullptr)
            *expert_m_begin = m_begin;
        const uint32_t aligned_valid_m = (num_m_fracs_per_block + (expert_m_block_idx < num_larger_m_blocks)) * FRAC_MN;
        return min(aligned_valid_m, num_expert_tokens - m_begin);
    }
};

// Replay the same routed task order on AIC and AIVs
// Set block_phase to kNone when routed metadata is pending, or kDone when all tasks are exhausted
template <const auto& kWorkspace, bool kLowLatency>
__aicore__ inline uint32_t take_linear_task(uint32_t num_shared_tokens, uint32_t num_routed_m_blocks,
                                            LinearTaskState& state, uint32_t core_idx,
                                            BlockPhase& block_phase, bool metadata_ready = true) {
    constexpr auto& kConfig = kWorkspace.config;
    constexpr uint32_t num_local_n_blocks[]{
        kConfig.get_num_linear1_n_blocks() / kNumLocalityDomains,
        kConfig.get_num_linear2_n_blocks() / kNumLocalityDomains,
        kConfig.get_num_shared_linear1_n_blocks() / kNumLocalityDomains,
        kConfig.get_num_linear2_n_blocks() / kNumLocalityDomains};
    const uint32_t num_shared_m_blocks = kConfig.num_shared_experts > 0 ? ceil_div(num_shared_tokens, layout::BLOCK_M) : 0;
    const uint32_t num_tasks[]{
        num_routed_m_blocks * num_local_n_blocks[0],
        num_routed_m_blocks * num_local_n_blocks[1],
        num_shared_m_blocks * num_local_n_blocks[2],
        num_shared_m_blocks * num_local_n_blocks[3]};
    const auto& [num_l1_tasks, num_l2_tasks, num_shared_l1_tasks, num_shared_l2_tasks] = num_tasks;
    auto& [l1_base, l2_base, shared_l1_base, shared_l2_base] = state.task_base;
    const uint32_t domain_idx = core_idx / kNumAICoresPerDomain;
    const uint32_t local_core_idx = core_idx % kNumAICoresPerDomain;

    while (true) {
        BlockPhase phase;
        if (shared_l1_base < num_shared_l1_tasks) {
            phase = BlockPhase::kSharedLinear1;
        } else if (metadata_ready and shared_l2_base < num_shared_l2_tasks and (not kLowLatency or (l1_base >= num_l1_tasks and l2_base >= num_l2_tasks))) {
            phase = BlockPhase::kSharedLinear2;
        } else {
            block_phase = metadata_ready ? BlockPhase::kDone : BlockPhase::kNone;
            if (not metadata_ready or (l1_base >= num_l1_tasks and l2_base >= num_l2_tasks))
                return ~uint32_t{0};

            // Prepare the next Linear2 wave with Linear1 lookahead
            bool is_linear1 = l2_base >= num_l2_tasks;
            if (not is_linear1 and l1_base < num_l1_tasks) {
                const uint32_t required_linear1_task_end = ceil_div(min(l2_base + kNumAICoresPerDomain, num_l2_tasks),
                    num_local_n_blocks[1]) * num_local_n_blocks[0];
                is_linear1 = l1_base < required_linear1_task_end + (kLowLatency ? 2 * num_local_n_blocks[0] : kNumAICoresPerDomain);
            }
            phase = is_linear1 ? BlockPhase::kLinear1 : BlockPhase::kLinear2;
        }

        // Partition N-blocks across locality domains; distribute each domain's tasks cyclically.
        const uint32_t phase_offsets[]{num_shared_l1_tasks, num_shared_l1_tasks + num_l1_tasks, 0, num_shared_l1_tasks + num_l1_tasks + num_l2_tasks};
        const uint32_t phase_idx = static_cast<uint32_t>(phase) - 1;
        const uint32_t local_task_idx = state.task_base[phase_idx] +
            (local_core_idx + kNumAICoresPerDomain - phase_offsets[phase_idx] % kNumAICoresPerDomain) % kNumAICoresPerDomain;
        state.task_base[phase_idx] += kNumAICoresPerDomain;
        if (local_task_idx < num_tasks[phase_idx]) {
            block_phase = phase;
            const uint32_t local_n = num_local_n_blocks[phase_idx];
            const uint32_t pool_block_idx = local_task_idx / local_n;
            const uint32_t n_block_idx = domain_idx * local_n + local_task_idx % local_n;
            return pool_block_idx * (local_n * kNumLocalityDomains) + n_block_idx;
        }
    }
}

} // namespace deep_gemm::mega_moe::sched
