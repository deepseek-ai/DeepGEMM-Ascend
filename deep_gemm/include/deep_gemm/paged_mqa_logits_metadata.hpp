#pragma once

#include <deep_gemm/utils/simt.hpp>
#include <deep_gemm/scheduler/paged_mqa_logits.hpp>

namespace deep_gemm {

inline constexpr uint32_t kNumPagedMQAMetadataThreads = 256;
inline constexpr uint32_t kPagedMQAWarpSize = 32;
inline constexpr uint32_t kNumPagedMQAMetadataWarps = kNumPagedMQAMetadataThreads / kPagedMQAWarpSize;

__simt_callee__ __aicore__ inline uint32_t paged_mqa_thread_exclusive_sum(
    uint32_t value, uint32_t lane_idx, uint32_t warp_idx,
    __ubuf__ uint32_t* warp_sums
) {
    const uint32_t lane_prefix = warp_prefix_sum(value, lane_idx);
    if (lane_idx == kPagedMQAWarpSize - 1)
        warp_sums[warp_idx] = lane_prefix;
    asc_syncthreads();

    if (warp_idx == 0) {
        value = lane_idx < kNumPagedMQAMetadataWarps ? warp_sums[lane_idx] : 0;
        value = warp_prefix_sum(value, lane_idx);
        if (lane_idx < kNumPagedMQAMetadataWarps)
            warp_sums[lane_idx] = value;
    }
    asc_syncthreads();

    const uint32_t warp_offset = warp_idx == 0 ? 0 : warp_sums[warp_idx - 1];
    return warp_offset + (lane_idx == 0 ? 0 : asc_shfl_up(lane_prefix, 1));
}

__simt_callee__ __aicore__ inline void paged_mqa_prefix_scan(
    uint32_t num_items, __ubuf__ uint32_t* prefix, __ubuf__ uint32_t* warp_sums
) {
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t lane_idx = thread_idx % kPagedMQAWarpSize;
    const uint32_t warp_idx = thread_idx / kPagedMQAWarpSize;

    // Keep adjacent threads' chunks on different UB banks by using an odd lane stride.
    const uint32_t num_items_per_thread = ceil_div(num_items, kNumPagedMQAMetadataThreads) | 1u;
    const uint32_t item_begin_idx = min(thread_idx * num_items_per_thread, num_items);
    const uint32_t item_end_idx = min(item_begin_idx + num_items_per_thread, num_items);

    uint32_t thread_sum = 0;
    for (uint32_t item_idx = item_begin_idx; item_idx < item_end_idx; ++ item_idx) {
        thread_sum += prefix[item_idx];
        prefix[item_idx] = thread_sum;
    }

    const uint32_t thread_offset = paged_mqa_thread_exclusive_sum(thread_sum, lane_idx, warp_idx, warp_sums);
    for (uint32_t item_idx = item_begin_idx; item_idx < item_end_idx; ++ item_idx)
        prefix[item_idx] += thread_offset;
    asc_syncthreads();
}

template <uint32_t SPLIT_KV, uint32_t kNumCores>
__simt_vf__ __aicore__ __launch_bounds__(kNumPagedMQAMetadataThreads) inline void paged_mqa_logits_metadata_vf(
    uint32_t num_q_tokens,
    __gm__ uint32_t* context_lens, __gm__ uint32_t* indices,
    __ubuf__ uint32_t* metadata_ub,
    __ubuf__ uint32_t* prefix, __ubuf__ uint32_t* request_starts,
    __ubuf__ uint32_t* warp_sums
) {
    static_assert(SPLIT_KV > 0 && SPLIT_KV % 16 == 0);
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t lane_idx = thread_idx % kPagedMQAWarpSize;
    const uint32_t warp_idx = thread_idx / kPagedMQAWarpSize;
    const uint32_t metadata_idx = thread_idx * 2;
    if (thread_idx <= kNumCores) {
        metadata_ub[metadata_idx] = num_q_tokens;
        metadata_ub[metadata_idx + 1] = 0;
    }

    const uint32_t num_q_tokens_per_thread = ceil_div(num_q_tokens, kNumPagedMQAMetadataThreads);
    const uint32_t q_token_begin_idx = min(thread_idx * num_q_tokens_per_thread, num_q_tokens);
    const uint32_t q_token_end_idx = min(q_token_begin_idx + num_q_tokens_per_thread, num_q_tokens);
    uint32_t num_request_starts = 0;
    for (uint32_t q_token_idx = q_token_begin_idx; q_token_idx < q_token_end_idx; ++ q_token_idx) {
        const bool is_request_start = q_token_idx == 0 || indices[q_token_idx] != indices[q_token_idx - 1];
        prefix[q_token_idx] = is_request_start;
        num_request_starts += is_request_start;
    }
    uint32_t request_idx = paged_mqa_thread_exclusive_sum(num_request_starts, lane_idx, warp_idx, warp_sums);
    const uint32_t num_requests = warp_sums[kNumPagedMQAMetadataWarps - 1];
    for (uint32_t q_token_idx = q_token_begin_idx; q_token_idx < q_token_end_idx; ++ q_token_idx) {
        if (prefix[q_token_idx])
            request_starts[request_idx++] = q_token_idx;
    }
    asc_syncthreads();
    for (uint32_t request_idx = thread_idx; request_idx < num_requests; request_idx += kNumPagedMQAMetadataThreads) {
        const uint32_t q_token_start = request_starts[request_idx];
        const uint32_t q_token_end = request_idx + 1 < num_requests ? request_starts[request_idx + 1] : num_q_tokens;
        const uint32_t num_q_tokens_in_request = q_token_end - q_token_start;
        const uint32_t context_len = mqa_request_context_len(q_token_start, num_q_tokens_in_request, context_lens);

        // Bound short requests by KV transfer cost and scale larger requests by FIX output rows.
        prefix[request_idx] = ceil_div(context_len, 16u) * max(num_q_tokens_in_request, 3u);
    }
    asc_syncthreads();
    if (num_requests == 0)
        return;

    paged_mqa_prefix_scan(num_requests, prefix, warp_sums);
    const uint32_t total_cost = prefix[num_requests - 1];
    const uint32_t target_cost = thread_idx * (total_cost / kNumCores) + min(thread_idx, total_cost % kNumCores);
    if (thread_idx <= kNumCores && target_cost < total_cost) {
        uint32_t request_idx = 0;
        uint32_t end = num_requests;
        while (request_idx < end) {
            const uint32_t mid = (request_idx + end) / 2;
            if (prefix[mid] <= target_cost)
                request_idx = mid + 1;
            else
                end = mid;
        }

        const uint32_t q_token_start = request_starts[request_idx];
        const uint32_t q_token_end = request_idx + 1 < num_requests ? request_starts[request_idx + 1] : num_q_tokens;
        const uint32_t num_q_tokens_in_request = q_token_end - q_token_start;
        const uint32_t previous_cost = request_idx == 0 ? 0 : prefix[request_idx - 1];
        const uint32_t context_len = mqa_request_context_len(q_token_start, num_q_tokens_in_request, context_lens);
        const uint32_t split_cost = (SPLIT_KV / 16) * max(num_q_tokens_in_request, 3u);

        // Each boundary is (request Q token index, KV token offset).
        metadata_ub[metadata_idx] = q_token_start;
        metadata_ub[metadata_idx + 1] = min((target_cost - previous_cost) / split_cost,
                                            ceil_div(context_len, SPLIT_KV) - 1) * SPLIT_KV;
    }
    asc_syncthreads();
}

template <uint32_t SPLIT_KV, uint32_t kNumCores>
__global__ __vector__ void paged_mqa_logits_metadata_simt_impl(
    uint32_t num_q_tokens, __gm__ uint32_t* context_lens,
    __gm__ uint32_t* indices, __gm__ uint32_t* metadata
) {
    static_assert(kNumCores > 0 && kNumCores < kNumPagedMQAMetadataThreads);
    if (block_idx != 0)
        return;

    auto prefix = reinterpret_cast<__ubuf__ uint32_t*>(0);
    auto request_starts = prefix + num_q_tokens;
    auto warp_sums = request_starts + num_q_tokens;
    const uint32_t request_arrays_bytes = 2 * num_q_tokens * sizeof(uint32_t);
    const uint32_t metadata_offset_bytes = aligned(request_arrays_bytes + kNumPagedMQAMetadataWarps * sizeof(uint32_t), 32u);
    auto metadata_ub = reinterpret_cast<__ubuf__ uint32_t*>(metadata_offset_bytes);
    asc_sync_pipe(PIPE_ALL);
    asc_vf_call<paged_mqa_logits_metadata_vf<SPLIT_KV, kNumCores>>(
        cce::dim3(kNumPagedMQAMetadataThreads), num_q_tokens, context_lens, indices, metadata_ub, prefix,
        request_starts, warp_sums);
    asc_sync_pipe(PIPE_ALL);
    asc_copy_ub2gm_align(metadata, metadata_ub, 1, (kNumCores + 1) * 2 * sizeof(uint32_t),
                         asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, 0, 0);
    asc_sync_pipe(PIPE_ALL);
}

} // namespace deep_gemm
