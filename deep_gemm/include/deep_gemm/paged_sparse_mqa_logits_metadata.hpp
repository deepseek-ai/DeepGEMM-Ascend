#pragma once

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/utils/simt.hpp>
#include <deep_gemm/layout/sparse_mqa_logits.hpp>

namespace deep_gemm::paged_sparse_mqa_logits_metadata {

// The context determines the used prefix; its block indices must be sorted, unique and in range.

using namespace layout::paged_sparse_mqa_logits;

struct LocalHeader {
    uint32_t bucket_counts[kNumBuckets];
    uint32_t bucket_bases[kNumBuckets];
    uint32_t query_bases[kBlockQ][kNumBuckets];
    uint32_t num_padded_blocks, kv_split_begin, kv_split_end;
};

// Estimate each original prepare range in one VF, before assigning the main core intervals.
template <uint32_t kCapacity, uint32_t kSparseBlockKV>
__simt_vf__ __launch_bounds__(32) inline void plan_range(
    __gm__ uint32_t* context_lens, __gm__ uint32_t* request_indices, __gm__ QBlockStats* q_block_stats, __gm__ uint64_t* range_cost,
    uint32_t query_begin, uint32_t query_end, uint32_t num_q_tokens, __ubuf__ volatile LocalHeader* ub_header
) {
    const uint32_t lane_idx = threadIdx.x;
    if (lane_idx == 0)
        ub_header->kv_split_begin = ub_header->kv_split_end = 0;
    uint64_t cost = 0;
    uint32_t request_begin = query_begin;
    for (uint32_t query_base = query_begin; query_base < query_end; query_base += 32) {
        const uint32_t query_idx = query_base + lane_idx;
        const bool is_valid_query = query_idx < query_end;
        uint32_t request_start = is_valid_query && (query_idx == query_begin || request_indices[query_idx] != request_indices[query_idx - 1]) ? query_idx : request_begin;
        for (uint32_t delta = 1; delta < 32; delta <<= 1) {
            const uint32_t previous = asc_shfl_up(request_start, delta);
            if (lane_idx >= delta)
                request_start = max(request_start, previous);
        }
        request_begin = asc_shfl(request_start, 31);
        if (is_valid_query && (query_idx - request_start) % kBlockQ == 0) {
            uint32_t num_q_block_queries = 1;
            while (num_q_block_queries < kBlockQ && query_idx + num_q_block_queries < num_q_tokens && request_indices[query_idx + num_q_block_queries] == request_indices[query_idx])
                ++ num_q_block_queries;
            uint32_t num_blocks[kBlockQ], num_total_blocks = 0, num_max_blocks = 0;
            #pragma unroll
            for (uint32_t query_offset = 0; query_offset < kBlockQ; ++ query_offset) {
                num_blocks[query_offset] = query_offset < num_q_block_queries ? min(kCapacity, (context_lens[query_idx + query_offset] + kSparseBlockKV - 1) / kSparseBlockKV) : 0;
                num_total_blocks += num_blocks[query_offset];
                num_max_blocks = max(num_max_blocks, num_blocks[query_offset]);
            }
            auto stats = q_block_stats + query_idx;
            stats->num_q_block_queries = num_q_block_queries;
            stats->packed_block_counts[0] = num_blocks[0] | num_blocks[1] << 16;
            stats->packed_block_counts[1] = num_blocks[2] | num_blocks[3] << 16;
            // C = B * (estimated union blocks + selected query blocks) + query startup.
            const uint32_t q_block_cost = num_total_blocks ? (num_max_blocks + num_total_blocks) * kSparseBlockKV + 4096 * num_q_block_queries : 0;
            stats->cost = q_block_cost;
            cost += q_block_cost;
        }
    }
    for (uint32_t delta = 16; delta; delta >>= 1)
        cost += asc_shfl(cost, lane_idx ^ delta);
    if (lane_idx == 0)
        *range_cost = cost;
}

template <uint32_t kSparseBlockKV, uint32_t kPageKV, uint32_t kNumQBlockQueries>
__simt_callee__ __aicore__ inline void bucket_q_block_interval(
    __ubuf__ volatile uint32_t* ub_workspace, __ubuf__ volatile uint32_t* ub_merge_records, uint32_t* query_begins, uint32_t* num_blocks,
    __gm__ int32_t* block_table, __gm__ uint32_t* physical_kv_token_starts, __ubuf__ volatile LocalHeader* ub_header, __gm__ uint16_t* mapping
) {
    constexpr uint32_t kRowCapacity = kMaxOutputTokens / kSparseBlockKV, kNumThreads = 256, kNumWarps = kNumThreads / 32;
    constexpr uint32_t kBlocksPerPage = kCompactPageTokens / kSparseBlockKV;
    constexpr uint32_t kNumQBlockBuckets = 1u << kNumQBlockQueries;
    const uint32_t thread_idx = threadIdx.x, lane_idx = thread_idx % 32, warp_idx = thread_idx / 32;
    const uint32_t histogram_idx = thread_idx ^ warp_idx;  // Spread the column scan across UB subbanks.
    auto ub_histogram = ub_merge_records + kBlockQ * kRowCapacity * 3;
    auto ub_bucket_bases = ub_header->bucket_bases;
    constexpr uint32_t kNoKey = 0xffffffffu;
    // A single query needs no union or bucket scan; retain the same sorted-pair layout.
    if constexpr (kNumQBlockQueries == 1) {
        const uint32_t num_query_blocks = num_blocks[0], num_aligned_blocks = (num_query_blocks + kBlocksPerPage - 1) / kBlocksPerPage * kBlocksPerPage;
        for (uint32_t block_idx = 2 * thread_idx; block_idx < num_aligned_blocks; block_idx += 2 * kNumThreads) {
            const uint32_t first_block_idx = block_idx < num_query_blocks ? ub_workspace[query_begins[0] + block_idx] : 0;
            const uint32_t second_block_idx = block_idx + 1 < num_query_blocks ? ub_workspace[query_begins[0] + block_idx + 1] : 0;
            const uint32_t first_token = block_idx < num_query_blocks ? static_cast<uint32_t>(block_table[first_block_idx / (kPageKV / kSparseBlockKV)]) * kPageKV + first_block_idx % (kPageKV / kSparseBlockKV) * kSparseBlockKV : kNoKey;
            const uint32_t second_token = block_idx + 1 < num_query_blocks ? static_cast<uint32_t>(block_table[second_block_idx / (kPageKV / kSparseBlockKV)]) * kPageKV + second_block_idx % (kPageKV / kSparseBlockKV) * kSparseBlockKV : kNoKey;
            reinterpret_cast<__gm__ uint64_t*>(physical_kv_token_starts)[block_idx / 2] = static_cast<uint64_t>(min(first_token, second_token)) | static_cast<uint64_t>(max(first_token, second_token)) << 32;
            if (block_idx < num_query_blocks)
                mapping[query_begins[0] + block_idx] = query_begins[0] + block_idx + (first_token > second_token);
            if (block_idx + 1 < num_query_blocks)
                mapping[query_begins[0] + block_idx + 1] = query_begins[0] + block_idx + 1 - (first_token > second_token);
        }
        return;
    }
    for (uint32_t query_mask = 0; query_mask < kNumQBlockBuckets; ++ query_mask)
        ub_histogram[query_mask * kNumThreads + histogram_idx] = 0;

    __ubuf__ volatile uint32_t* ub_query_blocks[kBlockQ];
    #pragma unroll
    for (uint32_t query_idx = 0; query_idx < kBlockQ; ++ query_idx)
        ub_query_blocks[query_idx] = ub_workspace + query_idx * kRowCapacity + query_begins[query_idx];
    uint32_t input_positions[kBlockQ], num_input_blocks;
    uint32_t num_consumed_blocks = sorted_union_partition<kNumQBlockQueries>(ub_query_blocks, num_blocks, thread_idx, kNumThreads, input_positions, num_input_blocks);
    const uint32_t input_end = (thread_idx + 1) * num_input_blocks / kNumThreads;
    uint32_t num_thread_blocks = 0;
    while (num_consumed_blocks < input_end) {
        uint32_t query_positions[kBlockQ] = {}, block_idx;
        const uint32_t query_mask = sorted_union_advance<kNumQBlockQueries>(ub_query_blocks, num_blocks, input_positions, query_positions, block_idx);
        num_consumed_blocks += __popc(query_mask);
        auto ub_merge_record = ub_merge_records + num_thread_blocks * 3 * kNumThreads + thread_idx;
        ub_merge_record[0] = block_idx;
        ub_merge_record[kNumThreads] = (query_positions[0] + query_begins[0]) | query_mask << 12 | (query_positions[1] + query_begins[1]) << 16;
        ub_merge_record[2 * kNumThreads] = (query_positions[2] + query_begins[2]) | (query_positions[3] + query_begins[3]) << 16;
        ++ ub_histogram[query_mask * kNumThreads + histogram_idx];
        ++ num_thread_blocks;
    }
    asc_syncthreads();

    // Each warp scans its buckets in sorted partition order.
    for (uint32_t bucket_idx = warp_idx; bucket_idx < kNumQBlockBuckets; bucket_idx += kNumWarps) {
        uint32_t num_lane_blocks = 0, thread_counts[kNumThreads / 32];
        #pragma unroll
        for (uint32_t scan_idx = 0; scan_idx < kNumThreads / 32; ++ scan_idx) {
            const uint32_t thread_offset = lane_idx * (kNumThreads / 32) + scan_idx, num_thread_blocks = ub_histogram[bucket_idx * kNumThreads + (thread_offset ^ (thread_offset >> 5))];
            thread_counts[scan_idx] = num_thread_blocks;
            num_lane_blocks += num_thread_blocks;
        }
        const uint32_t block_prefix = warp_prefix_sum(num_lane_blocks, lane_idx);
        uint32_t block_offset = block_prefix - num_lane_blocks;
        #pragma unroll
        for (uint32_t scan_idx = 0; scan_idx < kNumThreads / 32; ++ scan_idx) {
            const uint32_t thread_offset = lane_idx * (kNumThreads / 32) + scan_idx, num_thread_blocks = thread_counts[scan_idx];
            ub_histogram[bucket_idx * kNumThreads + (thread_offset ^ (thread_offset >> 5))] = block_offset;
            block_offset += num_thread_blocks;
        }
        if (lane_idx == 31)
            ub_header->bucket_counts[bucket_idx] = block_prefix;
    }
    asc_syncthreads();
    // Place buckets in mask order; each warp scans the same 16 bucket counts.
    const uint32_t num_bucket_blocks = lane_idx < kNumQBlockBuckets ? ub_header->bucket_counts[lane_idx] : 0;
    if (warp_idx == 0) {
        const uint32_t num_aligned_blocks = (num_bucket_blocks + kBlocksPerPage - 1) / kBlocksPerPage * kBlocksPerPage;
        const uint32_t prefix = warp_prefix_sum(num_aligned_blocks, lane_idx);
        if (lane_idx < kNumBuckets) {
            ub_bucket_bases[lane_idx] = prefix - num_aligned_blocks;
            if (lane_idx >= kNumQBlockBuckets)
                ub_header->bucket_counts[lane_idx] = 0;
        }
        if (lane_idx == 31)
            ub_header->num_padded_blocks = prefix;
    }
    if (warp_idx < kNumQBlockQueries) {
        const uint32_t num_selected_blocks = (lane_idx >> warp_idx & 1) ? num_bucket_blocks : 0;
        const uint32_t prefix = warp_prefix_sum(num_selected_blocks, lane_idx);
        if (lane_idx < kNumBuckets)
            ub_header->query_bases[warp_idx][lane_idx] = prefix - num_selected_blocks;
    }
    asc_syncthreads();
    // Store physical token starts in the former input workspace, grouped by query mask.
    for (uint32_t record_idx = 0; record_idx < num_thread_blocks; ++ record_idx) {
        auto ub_merge_record = ub_merge_records + record_idx * 3 * kNumThreads + thread_idx;
        const uint32_t block_idx = ub_merge_record[0], query_mask = (ub_merge_record[kNumThreads] >> 12) & 15;
        const uint32_t token_start = static_cast<uint32_t>(block_table[block_idx / (kPageKV / kSparseBlockKV)]) * kPageKV + block_idx % (kPageKV / kSparseBlockKV) * kSparseBlockKV;
        const uint32_t offset = ub_bucket_bases[query_mask] + ub_histogram[query_mask * kNumThreads + histogram_idx]++;
        ub_workspace[offset] = token_start;
        ub_merge_record[0] = offset;
    }
    if (thread_idx < kNumQBlockBuckets)
        for (uint32_t block_idx = ub_header->bucket_counts[thread_idx]; block_idx < (ub_header->bucket_counts[thread_idx] + kBlocksPerPage - 1) / kBlocksPerPage * kBlocksPerPage; ++ block_idx)
            ub_workspace[ub_bucket_bases[thread_idx] + block_idx] = kNoKey;
    asc_syncthreads();
    // Reuse the histogram for output mapping after all bucket cursors have been consumed.
    for (uint32_t record_idx = 0; record_idx < num_thread_blocks; ++ record_idx) {
        auto ub_merge_record = ub_merge_records + record_idx * 3 * kNumThreads + thread_idx;
        const uint32_t offset = ub_merge_record[0], packed_positions01 = ub_merge_record[kNumThreads], packed_positions23 = ub_merge_record[2 * kNumThreads];
        const uint32_t query_mask = (packed_positions01 >> 12) & 15;
        uint32_t query_offsets[kNumQBlockQueries];
        #pragma unroll
        for (uint32_t query_idx = 0; query_idx < kNumQBlockQueries; ++ query_idx)
            query_offsets[query_idx] = ub_header->query_bases[query_idx][query_mask];
        const uint32_t bucket_block_idx = offset - ub_bucket_bases[query_mask];
        const uint64_t pair = reinterpret_cast<__ubuf__ volatile uint64_t*>(ub_workspace)[offset / 2];
        const bool swap_pair = static_cast<uint32_t>(pair) > static_cast<uint32_t>(pair >> 32);
        const uint32_t sorted_block_idx = bucket_block_idx ^ static_cast<uint32_t>(swap_pair);
        for (uint32_t query_idx = 0; query_idx < kNumQBlockQueries; ++ query_idx) {
            if (query_mask >> query_idx & 1) {
                const uint32_t packed_positions = query_idx < 2 ? packed_positions01 : packed_positions23;
                const uint32_t output_block_idx = (packed_positions >> ((query_idx % 2) * 16)) & (kRowCapacity - 1);
                ub_histogram[query_idx * kRowCapacity + query_offsets[query_idx] + sorted_block_idx] = static_cast<uint16_t>(output_block_idx);
            }
        }
    }
    // Publish physical blocks and the completed output mapping in consecutive lane order.
    asc_syncthreads();
    const uint32_t num_padded_blocks = ub_header->num_padded_blocks;
    for (uint32_t block_idx = 2 * thread_idx; block_idx < num_padded_blocks; block_idx += 2 * kNumThreads) {
        const uint32_t first_token = ub_workspace[block_idx], second_token = ub_workspace[block_idx + 1];
        reinterpret_cast<__gm__ uint64_t*>(physical_kv_token_starts)[block_idx / 2] = static_cast<uint64_t>(min(first_token, second_token)) | static_cast<uint64_t>(max(first_token, second_token)) << 32;
    }
    #pragma unroll
    for (uint32_t query_idx = 0; query_idx < kNumQBlockQueries; ++ query_idx)
        for (uint32_t block_idx = thread_idx; block_idx < num_blocks[query_idx]; block_idx += kNumThreads)
            mapping[query_idx * kRowCapacity + query_begins[query_idx] + block_idx] = ub_histogram[query_idx * kRowCapacity + block_idx];
}

template <uint32_t kSparseBlockKV, uint32_t kSplitKV, uint32_t kNumAICores>
__simt_callee__ __aicore__ inline void emit_bucket_kv_splits(
    __ubuf__ volatile LocalHeader* ub_header, __gm__ uint32_t* kv_splits, uint32_t kv_split_cursor,
    uint32_t query_begin, uint64_t physical_begin, uint32_t* query_begins, uint32_t* query_counts, uint32_t num_q_block_queries
) {
    constexpr uint32_t kRowCapacity = kMaxOutputTokens / kSparseBlockKV, kNumThreads = 256;
    constexpr uint32_t kBlocksPerSplitKV = kSplitKV / kSparseBlockKV, kBlocksPerPage = kCompactPageTokens / kSparseBlockKV;
    const uint32_t lane_idx = threadIdx.x % 32;
    if (num_q_block_queries == 1) {
        const uint32_t num_blocks = query_counts[0], num_kv_splits = (num_blocks + kBlocksPerSplitKV - 1) / kBlocksPerSplitKV;
        if (threadIdx.x == 0)
            ub_header->kv_split_end = kv_split_cursor + num_kv_splits;
        for (uint32_t kv_split_idx = threadIdx.x; kv_split_idx < num_kv_splits; kv_split_idx += kNumThreads) {
            const uint32_t block_begin = kv_split_idx * kBlocksPerSplitKV, num_valid_blocks = min(kBlocksPerSplitKV, num_blocks - block_begin);
            auto kv_split = reinterpret_cast<__gm__ SplitKV*>(reinterpret_cast<__gm__ SplitKVRange*>(kv_splits) + 2 * kNumAICores) + kv_split_cursor + kv_split_idx;
            kv_split->compact_page_idx = (physical_begin + block_begin) / kBlocksPerPage;
            kv_split->control = 1u | (num_valid_blocks << SplitKV::kNumBlocksShift) | (1u << SplitKV::kNumQBlockQueriesShift) |
                (kv_split_idx == 0 ? SplitKV::kQBlockBegin : 0u) | (kv_split_idx + 1 == num_kv_splits ? SplitKV::kQBlockEnd : 0u);
            kv_split->query_begin = query_begin;
            kv_split->num_q_block_kv_splits = kv_split_idx == 0 ? num_kv_splits : 0;
            kv_split->map_begins[0] = query_begins[0] + block_begin;
            kv_split->map_begins[1] = 0;
            if (kv_split_idx == 0 || kv_split_idx + 1 == num_kv_splits)
                for (uint32_t query_idx = 0; query_idx < kBlockQ; ++ query_idx)
                    kv_split->output_ranges[query_idx] = query_begins[query_idx] * kSparseBlockKV | (query_counts[query_idx] * kSparseBlockKV) << 16;
        }
        return;
    }
    const uint32_t num_blocks = lane_idx < kNumBuckets ? ub_header->bucket_counts[lane_idx] : 0;
    const uint32_t block_base = lane_idx < kNumBuckets ? ub_header->bucket_bases[lane_idx] : 0xffffffffu;
    const uint32_t num_bucket_query_pairs = ((lane_idx & 3) != 0) + ((lane_idx & 12) != 0);
    const uint32_t num_bucket_kv_splits = (num_blocks + kBlocksPerSplitKV - 1) / kBlocksPerSplitKV;
    const uint32_t aiv_idx = (lane_idx & 3) == 0;
    uint32_t query_bases[kBlockQ] = {};
    #pragma unroll
    for (uint32_t query_idx = 0; query_idx < num_q_block_queries; ++ query_idx)
        query_bases[query_idx] = lane_idx < kNumBuckets ? ub_header->query_bases[query_idx][lane_idx] : 0;
    const uint32_t shared_prefix = warp_prefix_sum(num_bucket_query_pairs == 2 ? num_bucket_kv_splits : 0, lane_idx);
    const uint32_t aiv0_prefix = warp_prefix_sum(num_bucket_query_pairs == 1 && aiv_idx == 0 ? num_bucket_kv_splits : 0, lane_idx);
    const uint32_t aiv1_prefix = warp_prefix_sum(num_bucket_query_pairs == 1 && aiv_idx == 1 ? num_bucket_kv_splits : 0, lane_idx);
    const uint32_t num_shared_kv_splits = asc_shfl(shared_prefix, 31);
    const uint32_t num_aiv0_kv_splits = asc_shfl(aiv0_prefix, 31), num_aiv1_kv_splits = asc_shfl(aiv1_prefix, 31);
    const uint32_t num_other_aiv_kv_splits = aiv_idx ? num_aiv0_kv_splits : num_aiv1_kv_splits;
    const uint32_t bucket_split_begin = (num_bucket_query_pairs == 2 ? shared_prefix : aiv_idx ? aiv1_prefix : aiv0_prefix) - num_bucket_kv_splits;
    // Spread KV splits across lanes, including heavily shared buckets.
    const uint32_t kv_split_end = warp_prefix_sum(num_bucket_kv_splits, lane_idx), num_q_block_kv_splits = asc_shfl(kv_split_end, 31);
    if (threadIdx.x == 0)
        ub_header->kv_split_end = kv_split_cursor + num_q_block_kv_splits;
    for (uint32_t split_base = threadIdx.x - lane_idx; split_base < num_q_block_kv_splits; split_base += kNumThreads) {
        const uint32_t split_idx = min(split_base + lane_idx, num_q_block_kv_splits - 1);
        uint32_t bucket_idx = 0;
        #pragma unroll
        for (uint32_t step = (1u << num_q_block_queries) / 2; step; step /= 2)
            if (asc_shfl(kv_split_end, bucket_idx + step - 1) <= split_idx)
                bucket_idx += step;
        const uint32_t kv_split_idx = split_idx - asc_shfl(kv_split_end - num_bucket_kv_splits, bucket_idx);
        const uint32_t num_bucket_blocks = asc_shfl(num_blocks, bucket_idx), bucket_base = asc_shfl(block_base, bucket_idx);
        const uint32_t num_query_pairs = ((bucket_idx & 3) != 0) + ((bucket_idx & 12) != 0), kv_split_aiv_idx = (bucket_idx & 3) == 0;
        const uint32_t kv_split_begin = asc_shfl(bucket_split_begin, bucket_idx), num_peer_kv_splits = asc_shfl(num_other_aiv_kv_splits, bucket_idx);
        uint32_t kv_split_query_bases[kBlockQ] = {};
        #pragma unroll
        for (uint32_t query_idx = 0; query_idx < num_q_block_queries; ++ query_idx)
            kv_split_query_bases[query_idx] = asc_shfl(query_bases[query_idx], bucket_idx);
        if (split_base + lane_idx >= num_q_block_kv_splits)
            continue;
        const uint32_t block_begin = kv_split_idx * kBlocksPerSplitKV, num_valid_blocks = min(kBlocksPerSplitKV, num_bucket_blocks - block_begin);
        // Shared KV splits first, then interleave AIV0-only and AIV1-only KV splits.
        const uint32_t local_kv_split_idx = kv_split_begin + kv_split_idx;  // Within the shared or single-AIV list.
        const uint32_t q_block_kv_split_idx = num_query_pairs == 2 ? local_kv_split_idx : num_shared_kv_splits + local_kv_split_idx + min(local_kv_split_idx + kv_split_aiv_idx, num_peer_kv_splits);
        auto kv_split = reinterpret_cast<__gm__ SplitKV*>(reinterpret_cast<__gm__ SplitKVRange*>(kv_splits) + 2 * kNumAICores) + kv_split_cursor + q_block_kv_split_idx;
        kv_split->compact_page_idx = physical_begin / kBlocksPerPage + (bucket_base + block_begin) / kBlocksPerPage;
        kv_split->control = bucket_idx | (num_valid_blocks << SplitKV::kNumBlocksShift) | (num_q_block_queries << SplitKV::kNumQBlockQueriesShift) |
            (q_block_kv_split_idx == 0 ? SplitKV::kQBlockBegin : 0u) | (q_block_kv_split_idx + 1 == num_q_block_kv_splits ? SplitKV::kQBlockEnd : 0u);
        kv_split->query_begin = query_begin;
        kv_split->num_q_block_kv_splits = q_block_kv_split_idx == 0 ? num_q_block_kv_splits : 0;
        if (q_block_kv_split_idx == 0 || q_block_kv_split_idx + 1 == num_q_block_kv_splits)
            for (uint32_t query_idx = 0; query_idx < kBlockQ; ++ query_idx)
                kv_split->output_ranges[query_idx] = query_begins[query_idx] * kSparseBlockKV | (query_counts[query_idx] * kSparseBlockKV) << 16;
        for (uint32_t pair_idx = 0; pair_idx < 2; ++ pair_idx) {
            const uint32_t query_mask = bucket_idx >> (pair_idx * 2) & 3;
            if (!query_mask) {
                kv_split->map_begins[pair_idx] = 0;
                continue;
            }
            const uint32_t first_query_idx = pair_idx * 2 + (query_mask == 2);
            const uint32_t first_base = query_begins[first_query_idx] + kv_split_query_bases[first_query_idx];
            const uint32_t second_base = query_begins[2 * pair_idx + 1] + kv_split_query_bases[2 * pair_idx + 1];
            const uint32_t first_map_begin = (first_query_idx % 2) * kRowCapacity + first_base + block_begin;
            const uint32_t second_map_begin = query_mask == 3 ? kRowCapacity + second_base + block_begin : 0;
            kv_split->map_begins[pair_idx] = first_map_begin | (second_map_begin << 16);
        }
    }
}

// Cut only the first and last QBlock; the complete QBlocks keep their full query ranges.
template <uint32_t kCapacity, uint32_t kSparseBlockKV, uint32_t kPageKV, uint32_t kSplitKV, uint32_t kNumAICores, bool kFullQBlock, uint32_t kNumQBlockQueries>
__simt_vf__ __launch_bounds__(256) inline void build_q_block_interval(
    __ubuf__ volatile uint32_t* ub_workspace, __ubuf__ volatile uint32_t* ub_merge_records, __ubuf__ volatile LocalHeader* ub_header, uint64_t packed_block_counts, uint32_t cost,
    __gm__ int32_t* block_table, __gm__ uint32_t* physical_kv_token_starts, __gm__ uint16_t* mapping, __gm__ uint32_t* kv_splits,
    uint32_t query_begin, uint32_t core_idx, uint64_t begin_cost, uint64_t end_cost
) {
    constexpr uint32_t kRowCapacity = kMaxOutputTokens / kSparseBlockKV, kBlocksPerSplitKV = kSplitKV / kSparseBlockKV;
    const uint32_t thread_idx = threadIdx.x, num_q_block_queries = kNumQBlockQueries;
    uint32_t kv_split_cursor = ub_header->kv_split_end;
    const bool is_first_q_block = kv_split_cursor == 0;
    uint32_t query_begins[kBlockQ], query_ends[kBlockQ], query_counts[kBlockQ], num_input_blocks = 0;
    __ubuf__ volatile uint32_t* ub_query_blocks[kBlockQ];
    #pragma unroll
    for (uint32_t query_idx = 0; query_idx < kBlockQ; ++ query_idx) {
        query_counts[query_idx] = query_idx < kNumQBlockQueries ? (packed_block_counts >> (query_idx * 16)) & 65535u : 0;
        ub_query_blocks[query_idx] = ub_workspace + query_idx * kRowCapacity;
        query_begins[query_idx] = 0;
        query_ends[query_idx] = query_counts[query_idx];
        num_input_blocks += query_counts[query_idx];
    }
    if constexpr (!kFullQBlock) {
        uint32_t num_partition_items;
        if (begin_cost)
            sorted_union_partition<kNumQBlockQueries>(ub_query_blocks, query_counts, (begin_cost * num_input_blocks + cost - 1) / cost, num_input_blocks, query_begins, num_partition_items);
        if (end_cost < cost)
            sorted_union_partition<kNumQBlockQueries>(ub_query_blocks, query_counts, (end_cost * num_input_blocks + cost - 1) / cost, num_input_blocks, query_ends, num_partition_items);
    }
    uint32_t input_begin = 0, input_end = 0;
    #pragma unroll
    for (uint32_t query_idx = 0; query_idx < kBlockQ; ++ query_idx) {
        query_counts[query_idx] = query_ends[query_idx] - query_begins[query_idx];
        input_begin += query_begins[query_idx];
        input_end += query_ends[query_idx];
    }
    if (input_begin == input_end)
        return;
    if (is_first_q_block)
        kv_split_cursor = query_begin * get_max_kv_splits_per_q_block(kCapacity, kSparseBlockKV, kSplitKV) + input_begin / kBlocksPerSplitKV + core_idx * kNumBuckets;
    if (thread_idx == 0 && is_first_q_block)
        ub_header->kv_split_begin = kv_split_cursor;
    constexpr uint32_t kBlocksPerPage = kCompactPageTokens / kSparseBlockKV;
    const uint64_t physical_begin = static_cast<uint64_t>(query_begin) * (kCapacity + kPhysicalBlockSlack) + input_begin / kBlocksPerPage * kBlocksPerPage + core_idx * get_boundary_blocks(kSparseBlockKV);
    bucket_q_block_interval<kSparseBlockKV, kPageKV, kNumQBlockQueries>(
        ub_workspace, ub_merge_records, query_begins, query_counts, block_table,
        physical_kv_token_starts + physical_begin, ub_header, mapping + static_cast<uint64_t>(query_begin) * kRowCapacity);
    emit_bucket_kv_splits<kSparseBlockKV, kSplitKV, kNumAICores>(
        ub_header, kv_splits, kv_split_cursor, query_begin, physical_begin, query_begins, query_counts, num_q_block_queries);
}

__simt_vf__ __launch_bounds__(32) inline void publish_kv_split_range(__gm__ SplitKVRange* range, __ubuf__ volatile LocalHeader* ub_header) {
    if (threadIdx.x == 0) {
        range->begin = ub_header->kv_split_begin;
        range->end = ub_header->kv_split_end;
    }
}

// Prefix-search result used to locate QBlock stats after balancing.
struct PrepareCursor {
    uint64_t total_cost, cost_prefix;
    uint32_t prepare_idx;
};

// Locate the original prepare segment containing this AIV's target cost; scan its QBlock stats next.
template <uint32_t kNumAICores>
__simt_vf__ __launch_bounds__(32) inline void find_prepare_range(__ubuf__ volatile uint64_t* ub_range_costs, uint32_t aiv_idx) {
    const uint32_t lane_idx = threadIdx.x, core_idx = aiv_idx / 2;
    uint64_t lane_costs[2], cost_prefixes[2];
    #pragma unroll
    for (uint32_t batch_idx = 0; batch_idx < 2; ++ batch_idx) {
        lane_costs[batch_idx] = lane_idx + batch_idx * 32 < 2 * kNumAICores ? ub_range_costs[lane_idx + batch_idx * 32] : 0;
        cost_prefixes[batch_idx] = warp_prefix_sum(lane_costs[batch_idx], lane_idx);
    }
    cost_prefixes[1] += asc_shfl(cost_prefixes[0], 31);
    const uint64_t total_cost = asc_shfl(cost_prefixes[1], 31);
    const uint64_t search_cost = (total_cost * core_idx + kNumAICores - 1) / kNumAICores;
    auto ub_cursor = reinterpret_cast<__ubuf__ volatile PrepareCursor*>(ub_range_costs);
    if (lane_idx == 0) {
        ub_cursor->total_cost = total_cost;
        if (search_cost >= total_cost) {
            ub_cursor->cost_prefix = total_cost;
            ub_cursor->prepare_idx = 2 * kNumAICores;
        }
    }
    #pragma unroll
    for (uint32_t batch_idx = 0; batch_idx < 2; ++ batch_idx) {
        if (cost_prefixes[batch_idx] - lane_costs[batch_idx] <= search_cost && search_cost < cost_prefixes[batch_idx]) {
            ub_cursor->cost_prefix = cost_prefixes[batch_idx] - lane_costs[batch_idx];
            ub_cursor->prepare_idx = lane_idx + batch_idx * 32;
        }
    }
}

__aicore__ inline uint32_t align_q_block_begin(__gm__ uint32_t* request_indices, uint32_t query_idx, uint32_t num_q_tokens) {
    if (query_idx >= num_q_tokens)
        return num_q_tokens;
    const uint32_t request_idx = request_indices[query_idx];
    uint32_t request_begin = query_idx;
    while (request_begin && request_indices[request_begin - 1] == request_idx)
        -- request_begin;
    while (query_idx < num_q_tokens && request_indices[query_idx] == request_idx && (query_idx - request_begin) % kBlockQ)
        ++ query_idx;
    return query_idx;
}

// Keep the main core interval; balance its two prepare AIVs by selected block count.
template <uint32_t kCapacity, uint32_t kSparseBlockKV, uint32_t kPageKV, uint32_t kSplitKV, uint32_t kNumAICores>
__aicore__ inline void build_aiv_interval(
    __ubuf__ uint32_t* ub_workspace, __ubuf__ LocalHeader* ub_header, __gm__ QBlockStats* q_block_stats, __gm__ uint64_t* range_costs,
    __gm__ uint32_t* request_indices, __gm__ int32_t* sparse_indices, __gm__ int32_t* block_table, uint64_t block_table_stride, __gm__ uint32_t* physical_kv_token_starts,
    __gm__ uint16_t* mapping, __gm__ uint32_t* kv_splits, uint32_t num_q_tokens, uint32_t aiv_idx
) {
    const uint32_t core_idx = aiv_idx / 2, aiv_idx_in_core = aiv_idx % 2;
    constexpr uint32_t kRowCapacity = kMaxOutputTokens / kSparseBlockKV;
    asc_dcci_entire_out();
    copy_gm_to_ub(ub_ptr<uint32_t>(1, 4 * kNumAICores, reinterpret_cast<uintptr_t>(ub_workspace)), reinterpret_cast<__gm__ uint32_t*>(range_costs),
                  1, 16 * kNumAICores, 16 * kNumAICores, 16 * kNumAICores);
    asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID0);
    asc_sync_wait(PIPE_MTE2, PIPE_V, EVENT_ID0);
    asc_vf_call<find_prepare_range<kNumAICores>>(cce::dim3(32), reinterpret_cast<__ubuf__ uint64_t*>(ub_workspace), aiv_idx);
    asc_sync_notify(PIPE_V, PIPE_S, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_S, EVENT_ID0);
    asc_ub_dcci_single(ub_workspace);
    auto ub_cursor = reinterpret_cast<__ubuf__ volatile PrepareCursor*>(ub_workspace);
    const uint64_t total_cost = ub_cursor->total_cost;
    const uint64_t cost_begin = (total_cost * core_idx + kNumAICores - 1) / kNumAICores;
    const uint64_t cost_end = (total_cost * (core_idx + 1) + kNumAICores - 1) / kNumAICores;
    uint64_t cost_prefix = ub_cursor->cost_prefix;
    uint32_t query_begin = num_q_tokens;
    if (ub_cursor->prepare_idx < 2 * kNumAICores) {
        const uint32_t query_start = (static_cast<uint64_t>(ub_cursor->prepare_idx) * num_q_tokens + 2 * kNumAICores - 1) / (2 * kNumAICores);
        query_begin = align_q_block_begin(request_indices, query_start, num_q_tokens);
    }
    // Estimate selected blocks in this core interval; exact main rank cuts stay unchanged.
    const auto count_selected_blocks = [&](__gm__ QBlockStats* stats, uint64_t prefix) __aicore__ {
        const uint32_t cost = stats->cost;
        if (!cost || prefix + cost <= cost_begin)
            return uint64_t{0};
        const uint64_t packed_block_counts = stats->packed_block_counts[0] | (static_cast<uint64_t>(stats->packed_block_counts[1]) << 32);
        uint64_t num_selected_blocks = 0;
        #pragma unroll
        for (uint32_t query_offset = 0; query_offset < kBlockQ; ++ query_offset)
            num_selected_blocks += (packed_block_counts >> (query_offset * 16)) & 65535u;
        if (cost_begin <= prefix && prefix + cost <= cost_end)
            return num_selected_blocks;
        const uint64_t overlap = min(prefix + cost, cost_end) - max(prefix, cost_begin);
        return (overlap * num_selected_blocks + cost - 1) / cost;
    };
    uint64_t num_core_blocks = 0;
    for (uint32_t query_idx = query_begin; query_idx < num_q_tokens && cost_prefix < cost_end;) {
        num_core_blocks += count_selected_blocks(q_block_stats + query_idx, cost_prefix);
        cost_prefix += q_block_stats[query_idx].cost;
        query_idx += q_block_stats[query_idx].num_q_block_queries;
    }
    cost_prefix = ub_cursor->cost_prefix;
    uint64_t block_prefix = 0;
    constexpr uint32_t kInputStride = kBlockQ * kRowCapacity + 128;
    auto ub_merge_records = ub_workspace + 2 * kInputStride;
    uint32_t q_block_idx = 0;
    asc_sync_notify(PIPE_V, PIPE_MTE2, EVENT_ID0);
    asc_sync_notify(PIPE_V, PIPE_MTE2, EVENT_ID1);
    for (uint32_t query_idx = query_begin; query_idx < num_q_tokens && cost_prefix < cost_end;) {
        auto stats = q_block_stats + query_idx;
        const uint32_t num_q_block_queries = stats->num_q_block_queries, cost = stats->cost;
        const uint64_t num_blocks_in_q_block = count_selected_blocks(stats, cost_prefix);
        const bool is_second_aiv = 2 * block_prefix + num_blocks_in_q_block >= num_core_blocks;
        block_prefix += num_blocks_in_q_block;
        if (!aiv_idx_in_core && is_second_aiv)
            break;
        if (cost && cost_prefix + cost > cost_begin && aiv_idx_in_core == is_second_aiv) {
            const uint64_t packed_block_counts = stats->packed_block_counts[0] | (static_cast<uint64_t>(stats->packed_block_counts[1]) << 32);
            uint32_t num_blocks = 0;
            #pragma unroll
            for (uint32_t query_offset = 0; query_offset < kBlockQ; ++ query_offset)
                num_blocks = max(num_blocks, static_cast<uint32_t>(packed_block_counts >> (query_offset * 16)) & 65535u);
            const uint32_t stage_idx = q_block_idx++ % 2;
            const auto event = static_cast<event_t>(stage_idx);
            auto ub_q_block_workspace = ub_workspace + stage_idx * kInputStride;
            asc_sync_wait(PIPE_V, PIPE_MTE2, event);
            copy_gm_to_ub(ub_ptr<int32_t>(kBlockQ, kRowCapacity, reinterpret_cast<uintptr_t>(ub_q_block_workspace)),
                          sparse_indices + static_cast<uint64_t>(query_idx) * kCapacity, num_q_block_queries, num_blocks * 4, kCapacity * 4, kRowCapacity * 4);
            asc_sync_notify(PIPE_MTE2, PIPE_V, event);
            asc_sync_wait(PIPE_MTE2, PIPE_V, event);
            const auto build_q_block = [&]<bool kFullQBlock, uint32_t kNumQBlockQueries>() __aicore__ {
                asc_vf_call<build_q_block_interval<kCapacity, kSparseBlockKV, kPageKV, kSplitKV, kNumAICores, kFullQBlock, kNumQBlockQueries>>(
                    cce::dim3(256), ub_q_block_workspace, ub_merge_records, ub_header, packed_block_counts, cost, block_table + static_cast<uint64_t>(query_idx) * block_table_stride,
                    physical_kv_token_starts, mapping, kv_splits, query_idx, core_idx,
                    cost_begin > cost_prefix ? cost_begin - cost_prefix : 0, min(static_cast<uint64_t>(cost), cost_end - cost_prefix));
            };
            const bool full_q_block = cost_begin <= cost_prefix && cost_prefix + cost <= cost_end;
            if (full_q_block) {
                if (num_q_block_queries == 1)
                    build_q_block.template operator()<true, 1>();
                else if (num_q_block_queries == 2)
                    build_q_block.template operator()<true, 2>();
                else if (num_q_block_queries == 3)
                    build_q_block.template operator()<true, 3>();
                else
                    build_q_block.template operator()<true, 4>();
            } else {
                if (num_q_block_queries == 1)
                    build_q_block.template operator()<false, 1>();
                else if (num_q_block_queries == 2)
                    build_q_block.template operator()<false, 2>();
                else if (num_q_block_queries == 3)
                    build_q_block.template operator()<false, 3>();
                else
                    build_q_block.template operator()<false, 4>();
            }
            asc_sync_notify(PIPE_V, PIPE_MTE2, event);
        }
        cost_prefix += cost;
        query_idx += num_q_block_queries;
    }
    asc_sync_wait(PIPE_V, PIPE_MTE2, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_MTE2, EVENT_ID1);
    asc_vf_call<publish_kv_split_range>(cce::dim3(32), reinterpret_cast<__gm__ SplitKVRange*>(kv_splits) + aiv_idx, ub_header);
}

template <uint32_t kNumMaxSparseBlocks, uint32_t kSparseBlockKV, uint32_t kPageKV, uint32_t kSplitKV, uint32_t kNumAIVs>
__global__ __mix__(0, 1) void paged_sparse_mqa_prepare_impl(
    __gm__ int32_t* sparse_kv_block_indices, __gm__ uint32_t* context_lens, __gm__ uint32_t* indices, __gm__ int32_t* block_table,
    __gm__ uint32_t* physical_kv_token_starts, __gm__ uint16_t* mapping, __gm__ QBlockStats* q_block_stats,
    __gm__ uint64_t* range_costs, uint64_t num_q_tokens, uint64_t block_table_stride, __gm__ uint32_t* dst_kv_splits
) {
    asc_init();
    constexpr uint32_t kRowCapacity = kMaxOutputTokens / kSparseBlockKV;
    static_assert(kNumMaxSparseBlocks <= kRowCapacity && kNumAIVs <= 64);
    static_assert(kNumAIVs * sizeof(uint64_t) <= kMetadataAlignmentBytes);
    static_assert((24 * kRowCapacity + 256) * sizeof(uint32_t) + sizeof(LocalHeader) + 1024 <= kSparseUBBytes);
    __ubuf__ uint32_t ub_workspace[24 * kRowCapacity + 256], ub_header_storage[sizeof(LocalHeader) / sizeof(uint32_t)];
    auto ub_header = reinterpret_cast<__ubuf__ LocalHeader*>(ub_header_storage);
    const uint32_t aiv_idx = asc_get_block_idx();
    const uint32_t first = (aiv_idx * num_q_tokens + kNumAIVs - 1) / kNumAIVs;
    const uint32_t end = ((aiv_idx + 1) * num_q_tokens + kNumAIVs - 1) / kNumAIVs;
    const uint32_t query_begin = first < end ? align_q_block_begin(indices, first, num_q_tokens) : first;
    asc_vf_call<plan_range<kNumMaxSparseBlocks, kSparseBlockKV>>(
        cce::dim3(32), context_lens, indices, q_block_stats,
        range_costs + aiv_idx, query_begin, end, static_cast<uint32_t>(num_q_tokens), ub_header);
    asc_sync_inter_arrive(PIPE_V, 0);
    asc_sync_inter_wait(PIPE_S, 0);
    build_aiv_interval<kNumMaxSparseBlocks, kSparseBlockKV, kPageKV, kSplitKV, kNumAIVs / 2>(
        ub_workspace, ub_header, q_block_stats, range_costs, indices, sparse_kv_block_indices, block_table, block_table_stride, physical_kv_token_starts,
        mapping, dst_kv_splits, static_cast<uint32_t>(num_q_tokens), aiv_idx);
}

} // namespace deep_gemm::paged_sparse_mqa_logits_metadata
