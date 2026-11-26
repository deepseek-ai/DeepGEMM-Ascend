#pragma once

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/utils/simt.hpp>
#include <deep_gemm/layout/sparse_mqa_logits.hpp>

namespace deep_gemm::sparse_mqa_logits_metadata {

// The context determines the used prefix; its block indices must be sorted, unique and in range.

using namespace layout::sparse_mqa_logits;

struct QueryInfo {
    uint32_t num_blocks;
    uint32_t token_end;
    uint32_t num_tail_tokens;
};

struct alignas(256) InputHeader {
    QueryInfo queries[kBlockQ];
};

__simt_callee__ inline uint32_t sparse_mqa_merge(
    __ubuf__ volatile uint32_t** ub_query_blocks, const uint32_t* num_blocks,
    __ubuf__ volatile uint32_t* ub_query_masks, __ubuf__ volatile uint32_t* ub_warp_sums,
    __gm__ uint32_t* physical_kv_token_starts
) {
    constexpr uint32_t kWarpSize = 32;
    constexpr uint32_t kNumWarps = kNumMetadataThreads / kWarpSize;
    const uint32_t lane_idx = threadIdx.x % kWarpSize, warp_idx = threadIdx.x / kWarpSize;
    uint32_t input_begins[kBlockQ], num_input_blocks;
    uint32_t merge_begin = sorted_union_partition<kBlockQ>(
        ub_query_blocks, num_blocks, threadIdx.x, kNumMetadataThreads, input_begins, num_input_blocks);
    const uint32_t merge_end = (threadIdx.x + 1) * num_input_blocks / kNumMetadataThreads;
    uint32_t merged_base = 0, num_merged_kv_blocks = 0;
    #pragma unroll
    for (uint32_t pass_idx = 0; pass_idx < 2; ++ pass_idx) {
        uint32_t input_positions[kBlockQ];
        #pragma unroll
        for (uint32_t query_idx = 0; query_idx < kBlockQ; ++ query_idx)
            input_positions[query_idx] = input_begins[query_idx];
        uint32_t num_thread_blocks = 0, num_consumed_blocks = merge_begin;
        #pragma unroll
        for (uint32_t block_idx = 0; num_consumed_blocks < merge_end; ++ block_idx) {
            uint32_t query_positions[kBlockQ], token_start;
            const uint32_t query_mask = sorted_union_advance<kBlockQ>(
                ub_query_blocks, num_blocks, input_positions, query_positions, token_start);
            num_consumed_blocks += __popc(query_mask);
            if (pass_idx) {
                physical_kv_token_starts[merged_base + block_idx] = token_start;
                ub_query_masks[merged_base + block_idx] = query_mask;
            }
            ++ num_thread_blocks;
        }
        if (!pass_idx) {
            const uint32_t block_prefix = warp_prefix_sum(num_thread_blocks, lane_idx);
            if (lane_idx == kWarpSize - 1)
                ub_warp_sums[warp_idx] = block_prefix;
            asc_syncthreads();
            const uint32_t num_warp_blocks = lane_idx < kNumWarps ? ub_warp_sums[lane_idx] : 0;
            merged_base = asc_reduce_add(lane_idx < warp_idx ? num_warp_blocks : 0) + block_prefix - num_thread_blocks;
            num_merged_kv_blocks = asc_reduce_add(num_warp_blocks);
        }
    }
    return num_merged_kv_blocks;
}

template <uint32_t kSparseBlockKV>
__simt_callee__ inline void sparse_mqa_kv_split_outputs(
    __ubuf__ volatile uint32_t* ub_query_masks, __ubuf__ volatile uint32_t* ub_split_counts,
    uint32_t count_stride, uint32_t num_merged_kv_blocks, __ubuf__ volatile uint32_t* ub_workspace,
    __gm__ QBlockSplitKVOutputInfo* kv_split_outputs
) {
    constexpr uint32_t kWarpSize = 32;
    constexpr uint32_t kNumWarps = kNumMetadataThreads / kWarpSize;
    constexpr uint32_t kBlocksPerSplitKV = kSplitKV / kSparseBlockKV;
    const uint32_t lane_idx = threadIdx.x % kWarpSize, warp_idx = threadIdx.x / kWarpSize;
    const uint32_t num_q_block_kv_splits = (num_merged_kv_blocks + kBlocksPerSplitKV - 1) / kBlocksPerSplitKV;
    for (uint32_t kv_split_idx = warp_idx; kv_split_idx < num_q_block_kv_splits; kv_split_idx += kNumWarps) {
        const uint32_t block_idx = kv_split_idx * kBlocksPerSplitKV + lane_idx;
        const uint32_t query_mask_lo = block_idx < num_merged_kv_blocks ? ub_query_masks[block_idx] : 0;
        const uint32_t query_mask_hi = block_idx + kWarpSize < num_merged_kv_blocks && lane_idx + kWarpSize < kBlocksPerSplitKV ? ub_query_masks[block_idx + kWarpSize] : 0;
        #pragma unroll
        for (uint32_t query_idx = 0; query_idx < kBlockQ; ++ query_idx) {
            const uint32_t mask_lo = asc_ballot(query_mask_lo & (1u << query_idx));
            const uint32_t mask_hi = asc_ballot(query_mask_hi & (1u << query_idx));
            if (lane_idx == 0) {
                ub_split_counts[query_idx * count_stride + kv_split_idx] = __popc(mask_lo) + __popc(mask_hi);
                kv_split_outputs[kv_split_idx].masks[query_idx].lo = mask_lo;
                kv_split_outputs[kv_split_idx].masks[query_idx].hi = mask_hi;
            }
        }
    }
    asc_threadfence_block();
    asc_syncthreads();
    // Each query scans its KV split counts in one warp.
    if (warp_idx < kBlockQ) {
        const auto& query = reinterpret_cast<__ubuf__ volatile InputHeader*>(ub_workspace)->queries[warp_idx];
        uint32_t block_offset = 0;
        for (uint32_t kv_split_begin = 0; kv_split_begin < num_q_block_kv_splits; kv_split_begin += kWarpSize) {
            const uint32_t kv_split_idx = kv_split_begin + lane_idx;
            const uint32_t num_selected_blocks = kv_split_idx < num_q_block_kv_splits ? ub_split_counts[warp_idx * count_stride + kv_split_idx] : 0;
            const uint32_t prefix = warp_prefix_sum(num_selected_blocks, lane_idx);
            if (kv_split_idx < num_q_block_kv_splits) {
                const uint32_t block_end = block_offset + prefix;
                const uint32_t output_begin = (block_end - num_selected_blocks) * kSparseBlockKV;
                const uint32_t num_tokens = num_selected_blocks * kSparseBlockKV -
                    (num_selected_blocks && block_end == query.num_blocks ? kSparseBlockKV - query.num_tail_tokens : 0);
                kv_split_outputs[kv_split_idx].output_ranges[warp_idx] = output_begin | num_tokens << 16;
            }
            block_offset += asc_shfl(prefix, kWarpSize - 1);
        }
    }
}

template <uint32_t kSparseBlockKV>
__simd_vf__ inline void sparse_mqa_logits_stage_indices(
    __ubuf__ uint32_t* ub_workspace, uint32_t capacity, uint32_t row_stride,
    uint32_t start0, uint32_t start1, uint32_t start2, uint32_t start3,
    uint32_t end0, uint32_t end1, uint32_t end2, uint32_t end3
) {
    auto ub_header = reinterpret_cast<__ubuf__ InputHeader*>(ub_workspace);
    const auto all = asc_create_mask_b32(PAT_ALL);
    for (uint32_t query_idx = 0; query_idx < kBlockQ; ++ query_idx) {
        const uint32_t start = query_idx == 0 ? start0 : query_idx == 1 ? start1 : query_idx == 2 ? start2 : start3;
        const uint32_t end = query_idx == 0 ? end0 : query_idx == 1 ? end1 : query_idx == 2 ? end2 : end3;
        const uint32_t num_blocks = start < end ? (end - start + kSparseBlockKV - 1) / kSparseBlockKV : 0;
        const uint32_t num_selected_blocks = num_blocks < capacity ? num_blocks : capacity;
        auto ub_blocks = reinterpret_cast<__ubuf__ uint32_t*>(ub_header + 1) + query_idx * row_stride;
        vector_uint32_t value;
        asc_duplicate_scalar(value, num_selected_blocks);
        asc_storealign_1st(&ub_header->queries[query_idx].num_blocks, value);
        asc_duplicate_scalar(value, end);
        asc_storealign_1st(&ub_header->queries[query_idx].token_end, value);
        for (uint32_t block_idx = 0; block_idx < num_selected_blocks; block_idx += 64) {
            vector_uint32_t values;
            asc_loadalign(values, ub_blocks + block_idx);
            asc_shiftleft_scalar(values, values, __builtin_ctz(kSparseBlockKV), all);
            asc_add_scalar(values, values, start % kSparseBlockKV, all);
            asc_storealign(ub_blocks + block_idx, values, all);
        }
    }
}

template <uint32_t kSparseBlockKV>
__simt_vf__ __aicore__ __launch_bounds__(kNumMetadataThreads) inline void sparse_mqa_logits_metadata_vf(
    __gm__ QBlockHeader* q_block_header, __gm__ uint32_t* physical_kv_token_starts,
    __gm__ QBlockSplitKVOutputInfo* kv_split_outputs, uint32_t num_max_sparse_blocks, uint32_t row_stride,
    uint32_t num_q_block_queries, __ubuf__ volatile uint32_t* ub_workspace
) {
    constexpr uint32_t kBlocksPerSplitKV = kSplitKV / kSparseBlockKV;
    const uint32_t num_max_kv_splits_per_q_block = ceil_div(num_max_sparse_blocks * kBlockQ, kBlocksPerSplitKV);
    auto ub_header = reinterpret_cast<__ubuf__ volatile InputHeader*>(ub_workspace);
    auto ub_blocks_begin = reinterpret_cast<__ubuf__ volatile uint32_t*>(ub_header + 1);
    auto ub_query_masks = ub_blocks_begin + kBlockQ * row_stride;
    auto ub_split_counts = ub_query_masks + kBlockQ * num_max_sparse_blocks;
    auto ub_warp_sums = ub_split_counts + kBlockQ * num_max_kv_splits_per_q_block;
    __ubuf__ volatile uint32_t* ub_query_blocks[kBlockQ];
    uint32_t num_blocks[kBlockQ];
    #pragma unroll
    for (uint32_t query_idx = 0; query_idx < kBlockQ; ++ query_idx) {
        ub_query_blocks[query_idx] = ub_blocks_begin + query_idx * row_stride;
        num_blocks[query_idx] = ub_header->queries[query_idx].num_blocks;
        if (threadIdx.x == query_idx && num_blocks[query_idx])
            ub_header->queries[query_idx].num_tail_tokens = min(kSparseBlockKV, ub_header->queries[query_idx].token_end - ub_query_blocks[query_idx][num_blocks[query_idx] - 1]);
    }
    if (threadIdx.x == 0)
        q_block_header->num_q_block_queries = num_q_block_queries;
    for (uint32_t word_idx = threadIdx.x; word_idx < (num_q_block_queries - 1) * kQBlockHeaderWords; word_idx += kNumMetadataThreads)
        reinterpret_cast<__gm__ uint32_t*>(q_block_header + 1)[word_idx] = 0;
    if (!(num_blocks[0] | num_blocks[1] | num_blocks[2] | num_blocks[3])) {
        if (threadIdx.x == 0)
            q_block_header->num_merged_kv_blocks = 0;
        asc_threadfence();
        return;
    }
    asc_threadfence();
    asc_syncthreads();
    const uint32_t num_merged_kv_blocks = sparse_mqa_merge(ub_query_blocks, num_blocks, ub_query_masks, ub_warp_sums, physical_kv_token_starts);
    asc_threadfence_block();
    asc_syncthreads();
    sparse_mqa_kv_split_outputs<kSparseBlockKV>(
        ub_query_masks, ub_split_counts, num_max_kv_splits_per_q_block, num_merged_kv_blocks,
        ub_workspace, kv_split_outputs);
    if (threadIdx.x == 0)
        q_block_header->num_merged_kv_blocks = num_merged_kv_blocks;
    asc_threadfence();
    asc_syncthreads();
}

template <uint32_t kSparseBlockKV, uint32_t kNumAICores, uint32_t kNumScheduleCores>
__simt_vf__ __aicore__ __launch_bounds__(kNumMetadataThreads) inline void sparse_mqa_logits_schedule_vf(
    uint32_t core_idx, uint32_t num_q_tokens, __gm__ QBlockHeader* q_block_headers, __gm__ ScheduleBoundary* schedule,
    __ubuf__ volatile uint32_t* ub_warp_sums
) {
    // Producers may have cached a neighbor's header before its final update.
    if (threadIdx.x == 0)
        asc_dcci_entire(q_block_headers);
    asc_syncthreads();
    constexpr uint32_t kWarpSize = 32;
    constexpr uint32_t kNumWarps = kNumMetadataThreads / kWarpSize;
    const uint32_t lane_idx = threadIdx.x % kWarpSize;
    const uint32_t warp_idx = threadIdx.x / kWarpSize;
    const uint32_t q_token_begin = static_cast<uint64_t>(num_q_tokens) * core_idx / kNumScheduleCores;
    const uint32_t q_token_end = static_cast<uint64_t>(num_q_tokens) * (core_idx + 1) / kNumScheduleCores;
    uint32_t total_work = 0, work_begin = 0;
    // Q tails issue M=32 directly; metadata packing does not change the MMA/FIX packet cost.
    // Reduce immutable headers once, without a barrier per chunk.
    if constexpr (kNumScheduleCores > 1) {
        for (uint64_t query_idx = threadIdx.x; query_idx < num_q_tokens; query_idx += kNumMetadataThreads) {
            const uint32_t num_q_block_queries = q_block_headers[query_idx].num_q_block_queries;
            const uint32_t num_blocks = num_q_block_queries ? q_block_headers[query_idx].num_merged_kv_blocks : 0;
            const uint32_t work = num_q_block_queries + ceil_div(num_blocks * kSparseBlockKV, 128u) * (1 + ceil_div(num_q_block_queries, kNumQueriesPerMMA));
            total_work += work;
            work_begin += query_idx < q_token_begin ? work : 0;
        }
        total_work = asc_reduce_add(total_work);
        work_begin = asc_reduce_add(work_begin);
        if (lane_idx == 0) {
            ub_warp_sums[warp_idx] = total_work;
            ub_warp_sums[kNumWarps + warp_idx] = work_begin;
        }
        asc_threadfence_block();
        asc_syncthreads();
        total_work = asc_reduce_add(lane_idx < kNumWarps ? ub_warp_sums[lane_idx] : 0);
        work_begin = asc_reduce_add(lane_idx < kNumWarps ? ub_warp_sums[kNumWarps + lane_idx] : 0);
        asc_syncthreads();
    }
    // Keep each thread's contiguous chunk on different UB banks.
    constexpr uint32_t kNumPrefixItemsPerThread =
        ((kSparseUBBytes / sizeof(uint32_t) - 2 * kNumWarps) / kNumMetadataThreads - 1) | 1u;
    auto ub_work_prefix = ub_warp_sums + 2 * kNumWarps;
    const uint32_t num_items_per_thread = kNumScheduleCores == 1 ? 1 :
        min(ceil_div(q_token_end - q_token_begin, kNumMetadataThreads) | 1u, kNumPrefixItemsPerThread);
    uint32_t work_end = work_begin;
    for (uint32_t query_begin = q_token_begin; query_begin < q_token_end;) {
        const uint32_t num_items = min(kNumMetadataThreads * num_items_per_thread, q_token_end - query_begin);
        const uint32_t item_begin_idx = min(threadIdx.x * num_items_per_thread, num_items);
        const uint32_t item_end_idx = min(item_begin_idx + num_items_per_thread, num_items);
        uint32_t thread_work = 0;
        for (uint32_t item_idx = item_begin_idx; item_idx < item_end_idx; ++ item_idx) {
            const uint32_t num_q_block_queries = q_block_headers[query_begin + item_idx].num_q_block_queries;
            const uint32_t num_blocks = num_q_block_queries ? q_block_headers[query_begin + item_idx].num_merged_kv_blocks : 0;
            thread_work += num_q_block_queries + ceil_div(num_blocks * kSparseBlockKV, 128u) * (1 + ceil_div(num_q_block_queries, kNumQueriesPerMMA));
            if constexpr (kNumScheduleCores > 1)
                ub_work_prefix[item_idx] = thread_work;
        }
        const uint32_t thread_prefix = warp_prefix_sum(thread_work, lane_idx);
        if (lane_idx == kWarpSize - 1)
            ub_warp_sums[warp_idx] = thread_prefix;
        asc_threadfence_block();
        asc_syncthreads();
        const uint32_t warp_work = lane_idx < kNumWarps ? ub_warp_sums[lane_idx] : 0;
        const uint32_t thread_offset = asc_reduce_add(lane_idx < warp_idx ? warp_work : 0) + thread_prefix - thread_work;
        const uint32_t chunk_begin = work_end;
        work_end += asc_reduce_add(warp_work);
        for (uint32_t item_idx = item_begin_idx; item_idx < item_end_idx; ++ item_idx) {
            const uint32_t work = (kNumScheduleCores == 1 ? thread_work : ub_work_prefix[item_idx]) + chunk_begin + thread_offset;
            ub_work_prefix[item_idx] = work;
        }
        asc_threadfence_block();
        // Each thread owns boundaries inside its contiguous prefix range.
        const uint32_t thread_begin = chunk_begin + thread_offset;
        if constexpr (kNumScheduleCores == 1)
            total_work = work_end;
        const uint32_t num_work_per_core = total_work / kNumAICores;
        const uint32_t num_extra_work = total_work % kNumAICores;
        const uint32_t extra_work_end = num_extra_work * (num_work_per_core + 1);
        uint32_t schedule_idx = thread_begin < extra_work_end ? ceil_div(thread_begin, num_work_per_core + 1) :
            num_extra_work + ceil_div(thread_begin - extra_work_end, max(num_work_per_core, 1u));
        for (; schedule_idx < kNumAICores; ++ schedule_idx) {
            const uint32_t target_work = schedule_idx * num_work_per_core + min(schedule_idx, num_extra_work);
            if (target_work >= thread_begin + thread_work)
                break;
            uint32_t begin_idx = item_begin_idx, end_idx = item_end_idx;
            while (begin_idx < end_idx) {
                const uint32_t mid_idx = begin_idx + (end_idx - begin_idx) / 2;
                if (ub_work_prefix[mid_idx] <= target_work)
                    begin_idx = mid_idx + 1;
                else
                    end_idx = mid_idx;
            }
            const uint32_t num_q_block_queries = q_block_headers[query_begin + begin_idx].num_q_block_queries;
            const uint32_t previous = begin_idx > item_begin_idx ? ub_work_prefix[begin_idx - 1] : thread_begin;
            const uint32_t work_offset = target_work - previous;
            const uint32_t kv_split_work = kSplitKV / 128 * (1 + ceil_div(num_q_block_queries, kNumQueriesPerMMA));
            schedule[schedule_idx].query_begin = query_begin + begin_idx;
            schedule[schedule_idx].kv_split_begin = work_offset > num_q_block_queries ? (work_offset - num_q_block_queries) / kv_split_work : 0;
        }
        query_begin += num_items;
        if constexpr (kNumScheduleCores > 1)
            if (query_begin < q_token_end)
                asc_syncthreads();
    }
    if (core_idx == 0 && threadIdx.x <= kNumAICores) {
        const uint32_t target_work = threadIdx.x * (total_work / kNumAICores) + min(threadIdx.x, total_work % kNumAICores);
        if (target_work == total_work) {
            schedule[threadIdx.x].query_begin = num_q_tokens;
            schedule[threadIdx.x].kv_split_begin = 0;
        }
    }
}

template <uint32_t kSparseBlockKV, uint32_t kNumMetadataCores, uint32_t kNumAICores, uint32_t kNumScheduleCores>
__global__ __mix__(0, 1) void sparse_mqa_logits_metadata(
    __gm__ int32_t* sparse_kv_block_indices, __gm__ uint32_t* cu_seq_len_k_start, __gm__ uint32_t* cu_seq_len_k_end,
    uint32_t num_q_tokens, uint32_t num_kv_tokens, MetadataHeader layout, __gm__ uint32_t* metadata
) {
    asc_init();
    static_assert(kSparseBlockKV == 8 || kSparseBlockKV == 16);
    static_assert(kNumScheduleCores <= kNumMetadataCores);
    constexpr uint32_t kBlocksPerSplitKV = kSplitKV / kSparseBlockKV;
    auto* metadata_base = reinterpret_cast<__gm__ uint8_t*>(metadata);
    const auto q_block_headers = reinterpret_cast<__gm__ QBlockHeader*>(metadata_base + kMetadataAlignmentBytes);
    const auto physical_kv_token_starts = reinterpret_cast<__gm__ uint32_t*>(metadata_base + layout.physical_blocks_offset);
    const auto kv_split_outputs = reinterpret_cast<__gm__ QBlockSplitKVOutputInfo*>(metadata_base + layout.kv_split_outputs_offset);
    const auto schedule = reinterpret_cast<__gm__ ScheduleBoundary*>(metadata_base + layout.schedule_offset);
    if (get_block_idx() == 0) {
        const auto metadata_header = reinterpret_cast<__gm__ MetadataHeader*>(metadata);
        metadata_header->num_max_sparse_blocks = layout.num_max_sparse_blocks;
        metadata_header->physical_blocks_offset = layout.physical_blocks_offset;
        metadata_header->kv_split_outputs_offset = layout.kv_split_outputs_offset;
        metadata_header->schedule_offset = layout.schedule_offset;
    }
    static_assert(sizeof(InputHeader) / sizeof(uint32_t) + 4096 * kBlockQ * 2 + ceil_div(4096 * kBlockQ, kBlocksPerSplitKV) * kBlockQ + kNumMetadataThreads / 32 <=
                  kSparseUBBytes / sizeof(uint32_t), "UB overflow at the per-query API limit");
    const ub_ptr<uint32_t> ub_workspace(kSparseUBBytes / sizeof(uint32_t), 1);
    const uint32_t capacity = layout.num_max_sparse_blocks;
    const uint32_t row_stride = (capacity + 63u) / 64u * 64u;
    const uint32_t q_token_begin = static_cast<uint64_t>(num_q_tokens) * get_block_idx() / kNumMetadataCores;
    const uint32_t q_token_end = static_cast<uint64_t>(num_q_tokens) * (get_block_idx() + 1) / kNumMetadataCores;
    const uint32_t q_block_begin = (q_token_begin + kBlockQ - 1) / kBlockQ * kBlockQ;
    for (uint32_t query_begin = q_block_begin; query_begin < q_token_end; query_begin += kBlockQ) {
        const uint32_t num_remaining_queries = num_q_tokens - query_begin;
        const uint32_t num_q_block_queries = num_remaining_queries < kBlockQ ? num_remaining_queries : kBlockQ;
        uint32_t starts[kBlockQ] = {}, ends[kBlockQ] = {};
        const auto indices = sparse_kv_block_indices + static_cast<uint64_t>(query_begin) * capacity;
        bool has_input = false;
        #pragma unroll
        for (uint32_t query_idx = 0; query_idx < num_q_block_queries; ++ query_idx) {
            starts[query_idx] = cu_seq_len_k_start[query_begin + query_idx];
            const uint32_t end = cu_seq_len_k_end[query_begin + query_idx];
            ends[query_idx] = end < num_kv_tokens ? end : num_kv_tokens;
            has_input |= starts[query_idx] < ends[query_idx];
        }
        if (has_input)
            copy_gm_to_ub(ub_ptr<int32_t>(4, row_stride, reinterpret_cast<uintptr_t>(ub_workspace.ptr()) + sizeof(InputHeader)),
                          indices, num_q_block_queries, capacity * sizeof(uint32_t), capacity * sizeof(uint32_t), row_stride * sizeof(uint32_t));
        asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID0);
        asc_sync_wait(PIPE_MTE2, PIPE_V, EVENT_ID0);
        asc_vf_call<sparse_mqa_logits_stage_indices<kSparseBlockKV>>(
            ub_workspace.ptr(), has_input ? capacity : 0, row_stride,
            starts[0], starts[1], starts[2], starts[3], ends[0], ends[1], ends[2], ends[3]);
        asc_vf_call<sparse_mqa_logits_metadata_vf<kSparseBlockKV>>(
            cce::dim3(kNumMetadataThreads), q_block_headers + query_begin,
            physical_kv_token_starts + static_cast<uint64_t>(query_begin) * capacity,
            kv_split_outputs + static_cast<uint64_t>(query_begin) * ((capacity + kBlocksPerSplitKV - 1) / kBlocksPerSplitKV),
            capacity, row_stride, num_q_block_queries, ub_workspace.ptr());
        asc_sync_notify(PIPE_V, PIPE_MTE2, EVENT_ID0);
        asc_sync_wait(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }
    // All QBlock headers must be visible before any core partitions the work.
    if constexpr (kNumMetadataCores > 1) {
        asc_sync_inter_arrive(PIPE_V, 0);
        asc_sync_inter_wait(PIPE_S, 0);
    }
    if (get_block_idx() < kNumScheduleCores)
        asc_vf_call<sparse_mqa_logits_schedule_vf<kSparseBlockKV, kNumAICores, kNumScheduleCores>>(
            cce::dim3(kNumMetadataThreads),
            get_block_idx(), num_q_tokens, q_block_headers, schedule, ub_workspace.ptr());
}

} // namespace deep_gemm::sparse_mqa_logits_metadata
