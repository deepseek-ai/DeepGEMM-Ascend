#pragma once

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/layout/mega_moe.hpp>

#include <c_api/asc_simd.h>
#include <simt_api/device_functions.h>
#include <utils/debug/asc_printf.h>
#include <utils/debug/asc_assert.h>

namespace deep_gemm::mega_moe::comm {

// The predicate may advance work; true means the wait is complete
template <typename pred_t, typename print_timeout_t>
__aicore__ __forceinline__ void wait_until(const pred_t& pred, const print_timeout_t& print_timeout) {
    // 60 seconds at 1 GHz on Ascend 950DT
    constexpr uint64_t kNumTimeoutCycles = 60ull * 1'000'000'000;
    const auto start_cycle = static_cast<uint64_t>(asc_get_system_cycle());
    while (not pred()) {
        if (static_cast<uint64_t>(asc_get_system_cycle()) - start_cycle >= kNumTimeoutCycles) {
            print_timeout();
            trap();
        }
    }
}

__aicore__ inline void print_local_timeout(uint32_t rank_idx) {
    __asc_aicore::printf("MegaMoE local wait timeout: rank=%u core=%u\n", rank_idx, static_cast<uint32_t>(block_idx));
}

__aicore__ inline void print_peer_timeout(uint32_t rank_idx, uint32_t aiv_idx, uint32_t peer_idx,
                                         uint64_t prefix, bool wait_done) {
    __asc_aicore::printf("MegaMoE peer wait timeout: rank=%u aiv=%u peer=%u prefix=0x%llx wait_done=%u\n",
                        rank_idx, aiv_idx, peer_idx, static_cast<unsigned long long>(prefix), static_cast<uint32_t>(wait_done));
}

template <uint32_t kGridSyncIdx>
__simt_callee__ __forceinline__ void grid_sync(__gm__ uint32_t* counter_ptr, uint32_t rank_idx,
                                               uint32_t worker_idx, uint32_t num_workers) {
    // Worker deltas sum to the phase bit; low bits return to zero
    constexpr uint32_t kFinishSumTag = 0x80000000u;
    // 60 seconds of vector cycles at 1.65 GHz on Ascend 950DT
    constexpr uint64_t kNumTimeoutCycles = 60ull * 1'650'000'000;

    asc_threadfence();
    asc_syncthreads();
    if (num_workers == 1)
        return;
    if (threadIdx.x == 0) {
        const uint32_t old_value = asc_atomic_add(
            counter_ptr, worker_idx == 0 ? kFinishSumTag - (num_workers - 1) : 1u);
        const uint32_t expected_tag = (old_value ^ kFinishSumTag) & kFinishSumTag;
        const auto start_cycle = static_cast<uint64_t>(clock64());
        // SIMT cannot use the Scalar wait_until lambda
        while ((*reinterpret_cast<volatile __gm__ uint32_t*>(counter_ptr) & kFinishSumTag) != expected_tag) {
            if (static_cast<uint64_t>(clock64()) - start_cycle >= kNumTimeoutCycles) {
                const uint32_t current = *reinterpret_cast<volatile __gm__ uint32_t*>(counter_ptr);
                __asc_simt_vf::printf("MegaMoE grid sync timeout: rank=%u worker=%u grid=%u workers=%u current=0x%x expected_tag=0x%x\n",
                                     rank_idx, worker_idx, kGridSyncIdx, num_workers, current, expected_tag);
                __trap();
            }
        }
    }
    asc_syncthreads();
}

template <const auto& workspace, uint32_t kBarrierIdx = 0>
__aicore__ __forceinline__ void rank_barrier(__gm__ uint8_t* buffer, const layout::SymBuffer<>& sym_buffer, uint32_t vec_core_idx) {
    constexpr auto& config = workspace.config;
    const uint32_t rank_idx = sym_buffer.rank_idx;
    // Native event waits still rely on the device watchdog
    constexpr uint32_t kAllAIVSyncID = 14;
    asc_sync_pipe(PIPE_ALL);
    asc_sync_inter_arrive(PIPE_MTE3, kAllAIVSyncID);
    asc_sync_inter_wait(PIPE_S, kAllAIVSyncID);
    if constexpr (config.num_ranks == 1)
        return;
    if (vec_core_idx == 0) {
        const auto barrier = workspace.sync.rank_barriers.get_ptr(buffer, kBarrierIdx);
        const uint64_t epoch = asc_load_dev(&barrier->epoch);
        const int delta = epoch & 2 ? -1 : 1;
        const int target = delta < 0 ? 0 : static_cast<int>(config.num_ranks);
        asc_store_dev(&barrier->epoch, epoch + 1);
        const auto signal_ptr = &barrier->signals[epoch & 1];
        for (uint32_t dst_rank_idx = 0; dst_rank_idx < config.num_ranks; ++ dst_rank_idx) {
            const auto remote_base = sym_buffer.map(buffer, dst_rank_idx);
            const auto remote_barrier = workspace.sync.rank_barriers.get_ptr(remote_base, kBarrierIdx);
            asc_atomic_add(&remote_barrier->signals[epoch & 1], delta);
        }
        asc_sync_data_barrier(DSB_DDR);

        wait_until([&]() __aicore__ { return asc_load_dev(signal_ptr) == target; }, [&]() __aicore__ {
            __asc_aicore::printf("MegaMoE rank barrier timeout: rank=%u barrier=%u epoch=%llu signal=%d target=%d\n",
                   rank_idx, kBarrierIdx, static_cast<unsigned long long>(epoch), asc_load_dev(signal_ptr), target);
        });
    }
    asc_sync_pipe(PIPE_ALL);
    asc_sync_inter_arrive(PIPE_MTE3, kAllAIVSyncID);
    asc_sync_inter_wait(PIPE_S, kAllAIVSyncID);
}

template <const auto& workspace>
__aicore__ inline bool is_block_ready(__gm__ uint8_t* buffer, layout::ProgressStage stage,
                                      uint32_t progress_slot_idx, uint32_t num_expected_completions) {
    // Counts fit in the low byte of the atomic uint32 counter
    const auto completion = workspace.progress.blocks.template get_ptr<uint8_t>(buffer, workspace.get_progress_idx(stage, progress_slot_idx));
    return asc_load_dev(completion) >= num_expected_completions;
}

template <const auto& workspace>
__aicore__ inline void wait_for_block(__gm__ uint8_t* buffer, uint32_t rank_idx, layout::ProgressStage stage,
                                      uint32_t progress_slot_idx, uint32_t num_expected_completions) {
    wait_until([&]() __aicore__ {
        return is_block_ready<workspace>(buffer, stage, progress_slot_idx, num_expected_completions);
    }, [&]() __aicore__ { print_local_timeout(rank_idx); });
}

// Build expert-contiguous dispatch metadata
template <const auto& workspace, bool kOverlapReduce>
__simt_vf__ __aicore__ __launch_bounds__(layout::kNumDispatchThreads)
void build_dispatch_metadata(__gm__ uint8_t* buffer, __gm__ const int64_t* topk_idx,
                             uint32_t num_tokens, uint32_t rank_idx, uint32_t worker_idx, uint32_t num_workers,
                             __ubuf__ const int64_t* offsets) {
    constexpr auto& config = workspace.config;
    constexpr uint32_t kNumExpertsPerRank = config.num_experts / config.num_ranks;

    constexpr uint32_t kNumThreadsPerRound = kNumAICores * layout::kNumDispatchThreads;

    const auto grid_sync_ptr = workspace.sync.grid_sync_count.get_ptr(buffer);
    const auto expert_count = workspace.metadata.worker_expert_count.get_ptr(buffer);
    const auto total_expert_count = workspace.metadata.total_expert_count.get_ptr(buffer);
    const auto staged_token_topk_idx = workspace.metadata.staged_token_topk_idx.get_ptr(buffer);
    // Metadata owns this UB scratch until the VF completes
    __ubuf__ uint32_t expert_token_count[config.num_experts];
    static_assert(sizeof(expert_token_count) + config.num_ranks * sizeof(int64_t) <= UBSizeBytes, "Metadata UB overflow");

    const uint32_t thread_idx = threadIdx.x;
    const uint32_t num_token_topk = num_tokens * config.num_topk;

    // Count expert selections in UB
    for (uint32_t expert_idx = thread_idx; expert_idx < config.num_experts; expert_idx += layout::kNumDispatchThreads)
        expert_token_count[expert_idx] = 0;
    asc_syncthreads();

    // With overlap reduce, assign contiguous route batches to each worker
    const uint32_t worker_stride = not kOverlapReduce ? layout::kNumDispatchThreads :
        (num_token_topk + kNumThreadsPerRound - 1) / kNumThreadsPerRound * layout::kNumDispatchThreads;
    const uint32_t worker_begin = worker_idx * worker_stride;
    const uint32_t batch_stride = (kOverlapReduce ? 1 : num_workers) * layout::kNumDispatchThreads;
    const uint32_t worker_end = worker_begin + worker_stride;
    const uint32_t token_topk_end = kOverlapReduce and worker_end < num_token_topk ? worker_end : num_token_topk;
    for (uint32_t token_topk_idx = worker_begin + thread_idx; token_topk_idx < token_topk_end; token_topk_idx += batch_stride) {
        // Clear route mappings before counts allow any peer to dispatch these inputs
        if constexpr (kOverlapReduce)
            *workspace.route_m_blocks.get_ptr(buffer, token_topk_idx) = ~uint32_t{0};
        const auto expert_idx = static_cast<uint64_t>(topk_idx[token_topk_idx]);
        if (expert_idx < config.num_experts)
            asc_atomic_add(expert_token_count + expert_idx, 1u);
    }
    asc_syncthreads();

    // Save counts, then reuse UB counters as write offsets
    for (uint32_t expert_idx = thread_idx; expert_idx < config.num_experts; expert_idx += layout::kNumDispatchThreads)
        expert_count[worker_idx * config.num_experts + expert_idx] = expert_token_count[expert_idx];
    grid_sync<0>(grid_sync_ptr, rank_idx, worker_idx, num_workers);

    for (uint32_t expert_idx = thread_idx; expert_idx < config.num_experts; expert_idx += layout::kNumDispatchThreads) {
        uint32_t slot_begin = 0;
        for (uint32_t prev_worker_idx = 0; prev_worker_idx < worker_idx; ++ prev_worker_idx)
            slot_begin += expert_count[prev_worker_idx * config.num_experts + expert_idx];
        if (worker_idx == num_workers - 1)
            total_expert_count[expert_idx] = slot_begin + expert_token_count[expert_idx];
        expert_token_count[expert_idx] = slot_begin;
    }
    asc_syncthreads();

    // Stage one route per thread in kNumDispatchThreads-route batches
    for (uint32_t batch_base = worker_begin; batch_base < token_topk_end; batch_base += batch_stride) {
        const uint32_t token_topk_idx = batch_base + thread_idx;
        if (token_topk_idx < token_topk_end) {
            const auto expert_idx = static_cast<uint64_t>(topk_idx[token_topk_idx]);
            if (expert_idx < config.num_experts) {
                const uint32_t dst_slot_idx = asc_atomic_add(expert_token_count + expert_idx, 1u);
                staged_token_topk_idx[static_cast<uint64_t>(expert_idx) * workspace.num_src_slots_per_expert + dst_slot_idx] = token_topk_idx;
            }
        }

        // Keep route batches ordered across iterations
        if (kOverlapReduce and batch_base + batch_stride < token_topk_end)
            asc_syncthreads();
    }
    grid_sync<1>(grid_sync_ptr + 1, rank_idx, worker_idx, num_workers);

    // Distribute expert ranges across warps
    constexpr uint32_t kNumThreadsPerWarp = 32;
    const uint32_t warp_idx = thread_idx / kNumThreadsPerWarp;
    const uint32_t lane_idx = thread_idx % kNumThreadsPerWarp;
    const uint32_t global_warp_idx = warp_idx * num_workers + worker_idx;

    for (uint32_t expert_idx = global_warp_idx; expert_idx < config.num_experts; expert_idx += num_workers * (layout::kNumDispatchThreads / kNumThreadsPerWarp)) {
        const uint32_t count = total_expert_count[expert_idx];
        const uint32_t dst_rank_idx = expert_idx / kNumExpertsPerRank;
        const uint32_t local_expert_idx = expert_idx % kNumExpertsPerRank;
        const auto remote_base = layout::SymBuffer<>::map(buffer, dst_rank_idx, offsets);
        const uint64_t src_idx = static_cast<uint64_t>(expert_idx) * workspace.num_src_slots_per_expert;
        const uint64_t dst_idx = workspace.get_src_metadata_idx(local_expert_idx, rank_idx, 0);
        const auto remote_metadata = workspace.metadata.src_token_metadata.get_ptr(remote_base, dst_idx);
        const auto staged_indices = workspace.metadata.staged_token_topk_idx.get_ptr(buffer, src_idx);

        // Ship route weights with indices to avoid per-route remote Scalar reads.
        for (uint32_t slot_idx = lane_idx; slot_idx < count; slot_idx += kNumThreadsPerWarp) {
            const uint32_t token_topk_idx = staged_indices[slot_idx];
            const uint32_t weight_bits = *workspace.input.topk_weights.template get_ptr<uint32_t>(buffer, token_topk_idx);
            remote_metadata[slot_idx] = (static_cast<uint64_t>(weight_bits) << 32) | token_topk_idx;
        }
    }
    grid_sync<2>(grid_sync_ptr + 2, rank_idx, worker_idx, num_workers);

    // Publish counts after the metadata copy barrier
    if (worker_idx == 0) {
        for (uint32_t expert_idx = thread_idx; expert_idx < config.num_experts; expert_idx += layout::kNumDispatchThreads) {
            const uint32_t count = total_expert_count[expert_idx];
            const uint32_t dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const uint32_t local_expert_idx = expert_idx % kNumExpertsPerRank;

            // Reset the peer prefix before publishing local expert 0's count
            if (kOverlapReduce and local_expert_idx == 0) {
                *workspace.metadata.combine_ready_prefix.get_ptr(buffer, dst_rank_idx) = 0;
                asc_threadfence();
            }

            const auto remote_base = layout::SymBuffer<>::map(buffer, dst_rank_idx, offsets);
            const uint64_t recv_count_idx = workspace.get_expert_recv_idx(rank_idx, local_expert_idx);
            *workspace.metadata.expert_recv_count.get_ptr(remote_base, recv_count_idx) = count;
            const auto remote_count = workspace.metadata.expert_recv_count_sum.get_ptr(remote_base, local_expert_idx);
            // Release the count store before publishing its arrival
            asc_threadfence();
            asc_atomic_add(remote_count, layout::kCountArrivalBit | count);
        }
        asc_threadfence();
    }
}

template <const auto& workspace>
__aicore__ __forceinline__ uint32_t wait_dispatch_metadata(__gm__ uint8_t* buffer, uint32_t rank_idx, uint32_t core_idx,
                                                           __gm__ int32_t* cumulative_local_expert_recv_stats) {
    constexpr auto& config = workspace.config;
    const auto num_m_blocks_certificate = workspace.metadata.num_m_blocks_certificate.get_ptr(buffer);

    // Wait for source counts and sum local M-blocks
    uint32_t num_contributed_m_blocks = 0;
    for (uint32_t expert_idx = core_idx; expert_idx < config.num_experts / config.num_ranks; expert_idx += kNumAICores) {
        const auto count_ptr = workspace.metadata.expert_recv_count_sum.get_ptr(buffer, expert_idx);
        uint32_t count;
        wait_until([&]() __aicore__ {
            return ((count = asc_load_dev(count_ptr)) & ~layout::kCountValueMask) == config.num_ranks * layout::kCountArrivalBit;
        }, [&]() __aicore__ {
            __asc_aicore::printf("MegaMoE metadata count timeout: rank=%u local_expert=%u arrivals=%u expected=%u\n",
                   rank_idx, expert_idx, count / layout::kCountArrivalBit, config.num_ranks);
        });
        count &= layout::kCountValueMask;
        // Accumulate received routes before padding; shared experts are excluded
        if (cumulative_local_expert_recv_stats != nullptr)
            asc_atomic_add(cumulative_local_expert_recv_stats + expert_idx, static_cast<int32_t>(count));
        num_contributed_m_blocks += ceil_div(count, layout::BLOCK_M);
    }
    asc_sync_data_barrier(DSB_DDR);

    // All AIV0 cores contribute, including idle cores
    asc_atomic_add(num_m_blocks_certificate, layout::kCountArrivalBit + num_contributed_m_blocks);
    uint32_t num_total_m_blocks;
    wait_until([&]() __aicore__ {
        return ((num_total_m_blocks = asc_load_dev(num_m_blocks_certificate)) & ~layout::kCountValueMask) == kNumAICores * layout::kCountArrivalBit;
    }, [&]() __aicore__ { print_local_timeout(rank_idx); });
    asc_sync_data_barrier(DSB_DDR);
    return num_total_m_blocks & layout::kCountValueMask;
}

} // namespace deep_gemm::mega_moe::comm
