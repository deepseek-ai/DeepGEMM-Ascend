#pragma once

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/layout/mega_moe.hpp>

#include <c_api/asc_simd.h>
#include <simt_api/device_functions.h>
#include <simt_api/common_functions.h>
#include <utils/debug/asc_assert.h>

namespace deep_gemm::mega_moe::comm {

constexpr uint32_t kNumThreadsPerWarp = 32;

// Start near the average nonempty source size; coprimality makes the row permutation bijective
__simt_callee__ __aicore__ constexpr uint32_t get_src_m_stride(uint32_t num_tokens, uint32_t num_src_ranks) {
    if (num_src_ranks <= 1)
        return 1;
    for (uint32_t stride = simt::ceil_div(num_tokens, num_src_ranks);; ++ stride) {
        uint32_t a = stride, b = num_tokens;
        while (b != 0) {
            const uint32_t remainder = a % b;
            a = b;
            b = remainder;
        }
        if (a == 1)
            return stride;
    }
}

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
    __asc_aicore::printf("MegaMoE local wait timeout: rank=%u core=%u\n", rank_idx, block_idx);
}

__aicore__ inline void print_peer_timeout(uint32_t rank_idx, uint32_t aiv_idx, uint32_t peer_idx,
                                         uint64_t prefix, bool wait_done) {
    __asc_aicore::printf("MegaMoE peer wait timeout: rank=%u aiv=%u peer=%u prefix=0x%llx wait_done=%u\n",
                         rank_idx, aiv_idx, peer_idx, prefix, wait_done);
}

template <const auto& kWorkspace>
__aicore__ inline bool is_block_ready(__gm__ uint8_t* buffer, layout::ProgressStage stage,
                                      uint32_t progress_slot_idx, uint32_t num_expected_completions) {
    constexpr auto& kConfig = kWorkspace.config;
    // Each shard increases monotonically; their sum cannot publish readiness early
    uint32_t completed = 0;
    #pragma unroll
    for (uint32_t shard_idx = 0; shard_idx < layout::kNumProgressShards; ++ shard_idx)
        completed += asc_load_dev(kWorkspace.progress.blocks.template get_ptr<uint8_t>(buffer,
            kConfig.get_progress_idx(stage, progress_slot_idx, shard_idx, kWorkspace.num_max_progress_blocks)));
    return completed >= num_expected_completions;
}

__aicore__ __forceinline__ void backoff(uint64_t cycles = 64u) {
    const auto until = static_cast<uint64_t>(asc_get_system_cycle()) + cycles;
    while (static_cast<uint64_t>(asc_get_system_cycle()) < until) {}
}

template <const auto& kWorkspace>
__aicore__ inline void wait_for_block(__gm__ uint8_t* buffer, uint32_t rank_idx, layout::ProgressStage stage,
                                      uint32_t progress_slot_idx, uint32_t num_expected_completions) {
    wait_until([&]() __aicore__ {
        if (is_block_ready<kWorkspace>(buffer, stage, progress_slot_idx, num_expected_completions))
            return true;
        backoff();
        return false;
    }, [&]() __aicore__ { print_local_timeout(rank_idx); });
}

// Work around SIMT lambda instantiation crashes in CANN 9.2 template VFs
#define DG_SIMT_WAIT_UNTIL(predicate, print_timeout) do { \
    const uint64_t wait_start_cycle = clock64(); \
    while (not (predicate)) { \
        if (static_cast<uint64_t>(clock64()) - wait_start_cycle >= 60ull * 1'650'000'000) { \
            print_timeout; \
            __trap(); \
        } \
    } \
} while (false)

// Dispatch, source prefixes, and optional reverse mapping share one VF lifetime
template <const auto& kWorkspace, bool kLowLatency>
__simt_vf__ __aicore__ __launch_bounds__(layout::kNumDispatchThreads)
void build_metadata(__gm__ uint8_t* buffer, uint32_t num_tokens, uint32_t rank_idx, uint32_t core_idx,
                     __ubuf__ const int64_t* ub_offsets, __gm__ int32_t* cumulative_local_expert_recv_stats) {
    constexpr auto& kConfig = kWorkspace.config;
    constexpr uint32_t kNumExpertsPerRank = kConfig.num_experts / kConfig.num_ranks;
    constexpr uint32_t kNumCountsPerSector = kL2SectorBytes / sizeof(uint32_t);
    // Cover destination ranks and one expert per sending thread group, within the core budget.
    const uint32_t num_metadata_cores = min(max(kConfig.num_ranks, simt::ceil_div(kConfig.num_experts, kNumCountsPerSector)), kNumAICores);
    const uint32_t thread_idx = threadIdx.x;

    if (core_idx < num_metadata_cores) {
        uint64_t next_epoch = 0;
        const uint32_t expert_begin = core_idx * kConfig.num_experts / num_metadata_cores;
        const uint32_t expert_end = (core_idx + 1) * kConfig.num_experts / num_metadata_cores;
        const auto expert_token_topk_idx = kWorkspace.expert_token_topk_idx.get_ptr(buffer);
        // Metadata owns this UB scratch until the VF completes
        __ubuf__ uint32_t ub_expert_send_count[kConfig.num_experts];
        static_assert(sizeof(ub_expert_send_count) + kConfig.num_ranks * sizeof(int64_t) <= UBSizeBytes, "Metadata UB overflow");

        // Zero this core's expert counters
        for (uint32_t expert_idx = expert_begin + thread_idx; expert_idx < expert_end; expert_idx += layout::kNumDispatchThreads)
            ub_expert_send_count[expert_idx] = 0;
        asc_syncthreads();

        const uint32_t dst_rank_begin = expert_begin / kNumExpertsPerRank;
        const uint32_t dst_rank_end = simt::ceil_div(expert_end, kNumExpertsPerRank);
        // Overlap peer retirement checks with local route staging.
        if (thread_idx < kNumThreadsPerWarp) {
            // Previous peer completion retires metadata reads; growth still needs full retirement.
            const auto previous = __builtin_bit_cast(layout::RetiredState,
                                                     asc_ldcg(reinterpret_cast<__gm__ ulong4*>(kWorkspace.retired.get_ptr(buffer))));
            next_epoch = previous.epoch + 1;
            const bool metadata_reusable = previous.num_experts == kConfig.num_experts and
                kWorkspace.src_token_metadata.offset + kConfig.num_src_indices * sizeof(uint64_t) <= previous.metadata_end;
            for (uint32_t peer_idx = dst_rank_begin + thread_idx; not metadata_reusable and peer_idx < dst_rank_end; peer_idx += kNumThreadsPerWarp) {
                const auto peer_base = layout::SymBuffer::map(buffer, peer_idx, ub_offsets);
                const auto peer_epoch_ptr = reinterpret_cast<__gm__ volatile uint64_t*>(&kWorkspace.retired.get_ptr(peer_base)->epoch);
                DG_SIMT_WAIT_UNTIL(*peer_epoch_ptr >= next_epoch - 1,
                    __asc_simt_vf::printf("MegaMoE prior retirement timeout: rank=%u core=%u peer=%u expected=%llu current=%llu\n",
                                          rank_idx, core_idx, peer_idx, next_epoch - 1, *peer_epoch_ptr));
            }
        } else {
            // Count and stage routes within this core's expert range
            for (uint32_t token_topk_idx = thread_idx - kNumThreadsPerWarp; token_topk_idx < num_tokens * kConfig.num_topk; token_topk_idx += layout::kNumDispatchThreads - kNumThreadsPerWarp) {
                const auto expert_idx = *kWorkspace.input.topk_idx.template get_ptr<uint32_t>(buffer, token_topk_idx);
                assert(expert_idx == ~uint32_t{0} or expert_idx < kConfig.num_experts);
                // Core 0 initializes every route's destination before publishing this epoch
                if (core_idx == 0)
                    *kWorkspace.token_dst_metadata.get_ptr(buffer, token_topk_idx) =
                        expert_idx == ~uint32_t{0} ? ~uint32_t{0} : (expert_idx / kNumExpertsPerRank << 24) | 0x00ffffffu;
                if (expert_idx < expert_begin or expert_idx >= expert_end)
                    continue;
                const uint32_t slot_idx = asc_atomic_add(ub_expert_send_count + expert_idx, 1u);
                expert_token_topk_idx[static_cast<uint64_t>(expert_idx) * kConfig.num_src_slots_per_expert + slot_idx] = token_topk_idx;
            }
        }
        // Retire each core's staging; core 0 also retires all route initialization
        asc_threadfence();
        asc_syncthreads();
        if (core_idx == 0 and thread_idx == 0) {
            *kWorkspace.dispatch_epoch.get_ptr(buffer) = next_epoch;
            if constexpr (not kLowLatency)
                asc_threadfence();
        }
        const auto send_counts = [&]() __aicore__ __simt_callee__ {
            for (uint32_t expert_idx = expert_begin + thread_idx; expert_idx < expert_end; expert_idx += layout::kNumDispatchThreads) {
                const auto remote_base = layout::SymBuffer::map(buffer, expert_idx / kNumExpertsPerRank, ub_offsets);
                asc_stcg(kWorkspace.expert_recv_count.get_ptr(remote_base, kConfig.get_expert_recv_idx(rank_idx, expert_idx % kNumExpertsPerRank)), layout::kCountArrivalBit | ub_expert_send_count[expert_idx]);
            }
        };
        if constexpr (kLowLatency)
            send_counts();
        // Each warp sends one expert at a time
        constexpr uint32_t kNumWarps = layout::kNumDispatchThreads / kNumThreadsPerWarp;
        const uint32_t lane_idx = thread_idx % kNumThreadsPerWarp;

        for (uint32_t expert_idx = expert_begin + thread_idx / kNumThreadsPerWarp;
             expert_idx < expert_end; expert_idx += kNumWarps) {
            const uint32_t num_send_tokens = ub_expert_send_count[expert_idx];
            if constexpr (kLowLatency)
                assert(num_send_tokens <= layout::kNumFastSrcSlotsPerExpert);
            const uint32_t dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const uint32_t local_expert_idx = expert_idx % kNumExpertsPerRank;
            const auto remote_base = layout::SymBuffer::map(buffer, dst_rank_idx, ub_offsets);
            const auto token_topk_idx_ptr = expert_token_topk_idx + static_cast<uint64_t>(expert_idx) * kConfig.num_src_slots_per_expert;

            // Ship route weights with indices to avoid per-route remote Scalar reads
            for (uint32_t slot_idx = lane_idx; slot_idx < num_send_tokens; slot_idx += kNumThreadsPerWarp) {
                const uint32_t token_topk_idx = token_topk_idx_ptr[slot_idx];
                const uint32_t weight_bits = *kWorkspace.input.topk_weights.template get_ptr<uint32_t>(buffer, token_topk_idx);
                asc_stcg((kLowLatency ? kWorkspace.fast_src_metadata : kWorkspace.src_token_metadata).get_ptr(
                    remote_base, kConfig.get_src_metadata_idx(local_expert_idx, rank_idx, slot_idx, kLowLatency ? layout::kNumFastSrcSlotsPerExpert : 0)),
                    (static_cast<uint64_t>(weight_bits) << 32) | (kLowLatency ? layout::kRecordArrivalBit : 0) | token_topk_idx);
            }
        }
        if constexpr (not kLowLatency) {
            if (core_idx != 0 and thread_idx == 0)
                DG_SIMT_WAIT_UNTIL(*reinterpret_cast<__gm__ volatile uint64_t*>(kWorkspace.dispatch_epoch.get_ptr(buffer)) >= next_epoch,
                    __asc_simt_vf::printf("MegaMoE local route initialization timeout: rank=%u core=%u\n", rank_idx, core_idx));
            asc_threadfence();
            asc_syncthreads();
            send_counts();
        }
    }

    // Retire outgoing counts while the first warp builds incoming prefixes.
    uint32_t num_contributed_m_blocks = 0;
    for (uint32_t local_expert_idx = core_idx; thread_idx < kNumThreadsPerWarp and local_expert_idx < kNumExpertsPerRank; local_expert_idx += kNumAICores) {
        const auto src_m_end_ptr = kWorkspace.expert_src_m_end.get_ptr(buffer, local_expert_idx * kConfig.src_m_end_gm_stride);
        uint32_t carry = 0, num_src_ranks = 0;
        for (uint32_t src_rank_begin = 0; src_rank_begin < kConfig.num_ranks; src_rank_begin += kNumThreadsPerWarp) {
            const uint32_t src_rank_idx = src_rank_begin + thread_idx;
            uint32_t prefix = 0;
            if (src_rank_idx < kConfig.num_ranks) {
                const auto recv_count_ptr = reinterpret_cast<__gm__ volatile uint32_t*>(
                    kWorkspace.expert_recv_count.get_ptr(buffer, kConfig.get_expert_recv_idx(src_rank_idx, local_expert_idx)));
                DG_SIMT_WAIT_UNTIL((prefix = *recv_count_ptr) != 0,
                    __asc_simt_vf::printf("MegaMoE source count timeout: core=%u expert=%u source=%u\n", core_idx, local_expert_idx, src_rank_idx));
                *recv_count_ptr = 0;
                prefix &= layout::kCountValueMask;
                *kWorkspace.expert_recv_stats.get_ptr(buffer, src_rank_idx * kNumExpertsPerRank + local_expert_idx) = prefix;
            }
            num_src_ranks += asc_reduce_add(static_cast<uint32_t>(prefix != 0));
            #pragma unroll
            for (uint32_t lane_offset = 1; lane_offset < kNumThreadsPerWarp; lane_offset *= 2) {
                const uint32_t previous = asc_shfl_up(prefix, lane_offset);
                if (thread_idx >= lane_offset)
                    prefix += previous;
            }
            prefix += carry;
            carry = asc_shfl(prefix, kNumThreadsPerWarp - 1);
            if (src_rank_idx < kConfig.num_ranks)
                src_m_end_ptr[src_rank_idx] = prefix;
        }
        if (thread_idx == 0) {
            *kWorkspace.expert_src_m_stride.get_ptr(buffer, local_expert_idx) = get_src_m_stride(carry, num_src_ranks);
            if (cumulative_local_expert_recv_stats != nullptr)
                asc_atomic_add(cumulative_local_expert_recv_stats + local_expert_idx, static_cast<int32_t>(carry));
            num_contributed_m_blocks += kConfig.template get_routed_m_blocks<kLowLatency>(carry);
        }
    }
    if (thread_idx < kNumThreadsPerWarp and core_idx < kNumExpertsPerRank) {
        asc_threadfence();
        if (thread_idx == 0) {
            asc_stcg(kWorkspace.progress.metadata_ready.get_ptr(buffer, core_idx), layout::kMetadataPrefixesReady);
            asc_atomic_add(kWorkspace.routed_m_block_count.get_ptr(buffer), layout::kCountArrivalBit + num_contributed_m_blocks);
        }
    }
    if (thread_idx == 0) {
        const auto routed_m_block_count_ptr = reinterpret_cast<__gm__ volatile uint32_t*>(kWorkspace.routed_m_block_count.get_ptr(buffer));
        uint32_t packed_count;
        DG_SIMT_WAIT_UNTIL(((packed_count = *routed_m_block_count_ptr) & ~layout::kCountValueMask) == min(kNumExpertsPerRank, kNumAICores) * layout::kCountArrivalBit,
            __asc_simt_vf::printf("MegaMoE metadata count timeout: rank=%u core=%u packed_count=0x%x\n", rank_idx, core_idx, packed_count));
    }

    // Large-batch early reduction needs each route's producer block; low-latency mode waits on the final prefix.
    if constexpr (not kLowLatency) {
        asc_syncthreads();
        alignas(32) __ubuf__ uint32_t ub_expert_src_m_end[kNumExpertsPerRank * kConfig.src_m_end_ub_stride];
        // Copy source prefixes to UB, skipping GM row padding
        for (uint32_t word_idx = threadIdx.x; word_idx < kNumExpertsPerRank * kConfig.src_m_end_ub_stride; word_idx += layout::kNumDispatchThreads)
            ub_expert_src_m_end[word_idx] = asc_ldcg(kWorkspace.expert_src_m_end.get_ptr(buffer,
                word_idx / kConfig.src_m_end_ub_stride * kConfig.src_m_end_gm_stride + word_idx % kConfig.src_m_end_ub_stride));
        asc_syncthreads();
        uint32_t pool_block_begin = 0;
        for (uint32_t local_expert_idx = 0; local_expert_idx < kNumExpertsPerRank; ++ local_expert_idx) {
            const auto ub_src_m_end = ub_expert_src_m_end + local_expert_idx * kConfig.src_m_end_ub_stride;
            const uint32_t num_expert_tokens = ub_src_m_end[kConfig.num_ranks - 1];
            if (num_expert_tokens == 0) continue;
            const uint32_t src_m_stride = asc_ldcg(kWorkspace.expert_src_m_stride.get_ptr(buffer, local_expert_idx));
            // Match ExpertCursor: distribute M fracs across M-blocks
            const uint32_t num_expert_m_blocks = simt::ceil_div(num_expert_tokens, layout::BLOCK_M);
            const uint32_t num_m_fracs = simt::ceil_div(num_expert_tokens, FRAC_MN);
            const uint32_t num_m_fracs_per_block = num_m_fracs / num_expert_m_blocks;
            const uint32_t num_larger_m_blocks = num_m_fracs % num_expert_m_blocks;
            for (uint32_t m_idx = threadIdx.x * kNumAICores + core_idx; m_idx < num_expert_tokens; m_idx += layout::kNumDispatchThreads * kNumAICores) {
                const uint32_t m_frac_idx = m_idx / FRAC_MN;
                const uint32_t expert_m_block_idx = m_frac_idx < num_larger_m_blocks * (num_m_fracs_per_block + 1) ?
                    m_frac_idx / (num_m_fracs_per_block + 1) : (m_frac_idx - num_larger_m_blocks) / num_m_fracs_per_block;
                const uint32_t src_m_idx = static_cast<uint64_t>(m_idx) * src_m_stride % num_expert_tokens;
                uint32_t src_rank_idx = 0;
                while (src_m_idx >= ub_src_m_end[src_rank_idx]) ++ src_rank_idx;
                const uint32_t src_slot_idx = src_m_idx - (src_rank_idx ? ub_src_m_end[src_rank_idx - 1] : 0);
                const uint64_t token_metadata = *kWorkspace.src_token_metadata.get_ptr(buffer,
                    kConfig.get_src_metadata_idx(local_expert_idx, src_rank_idx, src_slot_idx));
                const auto src_base = layout::SymBuffer::map(buffer, src_rank_idx, ub_offsets);
                *kWorkspace.token_dst_metadata.get_ptr(src_base, static_cast<uint32_t>(token_metadata)) = (rank_idx << 24) | (pool_block_begin + expert_m_block_idx);
            }
            pool_block_begin += num_expert_m_blocks;
        }
        asc_threadfence();
        asc_syncthreads();
    }
}

#undef DG_SIMT_WAIT_UNTIL

} // namespace deep_gemm::mega_moe::comm
