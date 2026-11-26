#pragma once

#include <deep_gemm/common.hpp>

namespace deep_gemm::mega_gate::sched {

// Multiwave launches interleave K splits so complete token rows can be consumed early.
template <uint32_t SHAPE_N, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumKSplits, bool kPipeline>
struct Scheduler {
    static constexpr uint32_t kNumNBlocks = aligned(SHAPE_N, BLOCK_N) / BLOCK_N;
    static constexpr uint32_t kGroupM = kPipeline ? 1 : 4;
    static constexpr uint32_t kNumTasksPerGroup = kGroupM * kNumNBlocks * kNumKSplits;

    uint32_t num_tokens;
    const uint32_t num_m_blocks = aligned(num_tokens, BLOCK_M) / BLOCK_M;
    const uint32_t num_tasks = num_m_blocks * kNumNBlocks * kNumKSplits;

    __aicore__ void get_task(uint32_t task_idx, uint32_t& m_block_idx, uint32_t& n_block_idx, uint32_t& k_split_idx) const {
        const uint32_t num_blocks = num_m_blocks * kNumNBlocks;
        k_split_idx = kPipeline ? task_idx % kNumKSplits : task_idx / num_blocks;
        const uint32_t block_idx = kPipeline ? task_idx / kNumKSplits : task_idx % num_blocks;

        const uint32_t tail_m_start = num_m_blocks / kGroupM * kGroupM;
        const uint32_t tail_start = tail_m_start * kNumNBlocks;
        if (block_idx >= tail_start) {
            m_block_idx = tail_m_start + (block_idx - tail_start) / kNumNBlocks;
            n_block_idx = (block_idx - tail_start) % kNumNBlocks;
            return;
        }

        // At most one N group fits: N <= 512, BLOCK_N >= 16, and a single wave has <= 32 tasks.
        const uint32_t group_idx = block_idx / (kGroupM * kNumNBlocks);
        const uint32_t local_idx = block_idx % (kGroupM * kNumNBlocks);
        const uint32_t m_idx_in_group = local_idx % kGroupM;
        m_block_idx = group_idx * kGroupM + m_idx_in_group;
        n_block_idx = local_idx / kGroupM;
        if constexpr (kNumNBlocks == 32 / kGroupM)
            n_block_idx ^= m_idx_in_group;
    }

    __aicore__ uint32_t get_num_published_tokens(uint32_t num_completed_tasks) const {
        const uint32_t num_published_tokens = num_completed_tasks / kNumTasksPerGroup * kGroupM * BLOCK_M;
        return num_published_tokens < num_tokens ? num_published_tokens : num_tokens;
    }
};

} // namespace deep_gemm::mega_gate::sched
