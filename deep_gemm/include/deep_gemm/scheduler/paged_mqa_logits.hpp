#pragma once

#include <deep_gemm/ascend.hpp>

namespace deep_gemm {

struct PagedMQARequestInfo {
    uint32_t q_token_start;
    uint32_t num_q_tokens;
    uint32_t context_len;
    uint32_t num_q_blocks;
};

__host_aicore__ constexpr uint32_t mqa_request_context_len(uint32_t q, uint32_t count, __gm__ uint32_t* context_lens) {
    return context_lens[q + count - 1];
}

template <uint32_t kNumQTokensPerBlock>
__aicore__ inline PagedMQARequestInfo make_paged_mqa_request(uint32_t q, uint32_t num_q_tokens,
                                                             __gm__ uint32_t* context_lens,
                                                             __gm__ uint32_t* indices) {
    uint32_t count = 1;
    while (q + count < num_q_tokens && indices[q + count] == indices[q])
        ++count;
    return {
        .q_token_start=q,
        .num_q_tokens=count,
        .context_len=mqa_request_context_len(q, count, context_lens),
        .num_q_blocks=ceil_div(count, kNumQTokensPerBlock),
    };
}

template <uint32_t kNumQTokensPerBlock, uint32_t SPLIT_KV, uint32_t kPageSizeValue, uint32_t kNumHeads>
struct PagedMQALogitsScheduler {
    static constexpr uint32_t kPageSize = kPageSizeValue;

    uint32_t num_q_tokens;
    __gm__ uint32_t* context_lens;
    __gm__ uint32_t* indices;
    __gm__ uint32_t* block_table;
    uint32_t block_table_row_stride;
    uint64_t kv_page_stride_bytes;
    uint32_t q_token_idx;
    uint32_t kv_token_idx;
    uint32_t end_q_token_idx;
    uint32_t end_kv_token_idx;
    PagedMQARequestInfo request;
    uint32_t current_q_block_idx = 0;

    __aicore__ PagedMQALogitsScheduler(
        uint32_t core_id, uint32_t num_q_tokens,
        __gm__ uint32_t* context_lens, __gm__ uint32_t* indices,
        __gm__ uint32_t* metadata, __gm__ uint32_t* block_table,
        uint32_t block_table_row_stride, uint64_t kv_page_stride_bytes
    ):
        num_q_tokens(num_q_tokens),
        context_lens(context_lens),
        indices(indices),
        block_table(block_table),
        block_table_row_stride(block_table_row_stride),
        kv_page_stride_bytes(kv_page_stride_bytes) {
        const uint32_t metadata_idx = core_id * 2;
        // Metadata KV offsets are in tokens, aligned to SPLIT_KV.
        q_token_idx = metadata[metadata_idx];
        kv_token_idx = metadata[metadata_idx + 1];
        end_q_token_idx = metadata[metadata_idx + 2];
        end_kv_token_idx = metadata[metadata_idx + 3];
        if (!done())
            request = make_paged_mqa_request<kNumQTokensPerBlock>(q_token_idx, num_q_tokens, context_lens, indices);
    }

    __aicore__ bool done() const {
        return q_token_idx >= num_q_tokens ||
               (q_token_idx == end_q_token_idx && kv_token_idx >= end_kv_token_idx);
    }
    __aicore__ bool next_q_block(uint32_t& q_block_idx, uint32_t& kv_token_base, uint32_t& num_kv_splits) {
        while (!done()) {
            const uint32_t kv_token_end = request.q_token_start == end_q_token_idx ? end_kv_token_idx : request.context_len;
            if (current_q_block_idx == request.num_q_blocks) {
                current_q_block_idx = 0;
                kv_token_idx = kv_token_end;
            }
            if (kv_token_idx < kv_token_end) {
                q_block_idx = current_q_block_idx++;
                kv_token_base = kv_token_idx;
                num_kv_splits = ceil_div(kv_token_end - kv_token_idx, SPLIT_KV);
                return true;
            }
            if (request.q_token_start == end_q_token_idx)
                break;
            q_token_idx = request.q_token_start + request.num_q_tokens;
            kv_token_idx = 0;
            if (!done())
                request = make_paged_mqa_request<kNumQTokensPerBlock>(q_token_idx, num_q_tokens, context_lens, indices);
        }
        return false;
    }

    __aicore__ uint32_t get_q_base(uint32_t q_block_idx) const {
        return request.q_token_start + q_block_idx * kNumQTokensPerBlock;
    }
    __aicore__ uint32_t get_num_q_tokens_in_block(uint32_t q_block_idx) const {
        return min(kNumQTokensPerBlock, request.num_q_tokens - q_block_idx * kNumQTokensPerBlock);
    }
    __aicore__ bool has_multiple_q_blocks() const {
        return request.num_q_tokens > kNumQTokensPerBlock;
    }
    // KV token bounds use global query indices.
    __aicore__ uint32_t get_kv_token_begin(uint32_t) const {
        return 0;
    }
    __aicore__ uint32_t get_kv_token_end(uint32_t q_idx) const {
        return context_lens[q_idx];
    }
    // NOTES: SF strides are in int16_t elements, not bytes.
    __aicore__ uint32_t get_q_sf_stride() const {
        return num_q_tokens * kNumHeads;
    }
    __aicore__ uint32_t get_kv_token_offset(uint32_t kv_token_base, uint32_t split_idx) const {
        return kv_token_base + split_idx * SPLIT_KV;
    }
    __aicore__ uint32_t get_num_kv_block_tokens(uint32_t kv_token_base, uint32_t split_idx) const {
        const uint32_t offset = get_kv_token_offset(kv_token_base, split_idx);
        return offset < request.context_len ? min(SPLIT_KV, request.context_len - offset) : 0;
    }
    __aicore__ uint32_t get_physical_page(uint32_t kv_token_base, uint32_t split_idx, uint32_t compact_page) const {
        const uint32_t logical_page_idx = get_kv_token_offset(kv_token_base, split_idx) / kPageSize + compact_page;
        return block_table[static_cast<uint64_t>(request.q_token_start) * block_table_row_stride +
                           logical_page_idx];
    }
    // Adjacent logical pages start at the smaller physical page; AIV restores reversed pairs.
    __aicore__ uint32_t get_physical_page_batch(uint32_t kv_token_base, uint32_t split_idx, uint32_t compact_page,
                                                uint32_t num_pages, uint32_t& first_page,
                                                uint32_t& page_gap) const {
        const uint32_t num_batch = min(2u, num_pages - compact_page);
        const uint32_t page0 = get_physical_page(kv_token_base, split_idx, compact_page);
        const uint32_t page1 = num_batch == 2 ? get_physical_page(kv_token_base, split_idx, compact_page + 1) : page0;
        first_page = min(page0, page1);
        page_gap = page0 > page1 ? page0 - page1 : page1 - page0;
        return num_batch;
    }
    __aicore__ uint64_t get_kv_page_offset_bytes(uint32_t physical_page) const {
        return physical_page * kv_page_stride_bytes;
    }
};

} // namespace deep_gemm
