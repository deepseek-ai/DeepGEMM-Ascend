#pragma once

#include <deep_gemm/ascend.hpp>

namespace deep_gemm {

template <uint32_t kNumQTokensPerBlock, uint32_t SPLIT_KV, uint32_t kNumCores>
struct MQALogitsScheduler {
    uint32_t current_q_block_idx;
    uint32_t num_q_blocks;
    uint32_t num_q_tokens;
    uint32_t num_kv_tokens;
    uint32_t block_kv_end;
    __gm__ uint32_t* cu_seq_len_k_start;
    __gm__ uint32_t* cu_seq_len_k_end;
    uint32_t num_heads;

    __aicore__ MQALogitsScheduler(
        uint32_t core_id, uint32_t num_q_tokens, uint32_t num_kv_tokens,
        __gm__ uint32_t* cu_seq_len_k_start, __gm__ uint32_t* cu_seq_len_k_end,
        uint32_t num_heads
    ):
        current_q_block_idx(core_id),
        num_q_blocks(ceil_div(num_q_tokens, kNumQTokensPerBlock)),
        num_q_tokens(num_q_tokens),
        num_kv_tokens(num_kv_tokens),
        cu_seq_len_k_start(cu_seq_len_k_start),
        cu_seq_len_k_end(cu_seq_len_k_end),
        num_heads(num_heads) {}

    __aicore__ bool next_q_block(uint32_t& q_block_idx, uint32_t& kv_token_base, uint32_t& num_kv_splits) {
        if (current_q_block_idx >= num_q_blocks)
            return false;

        // TODO: Consider metadata-based partitioning by issued MAD/FIX work for uneven CP KV ranges.
        q_block_idx = current_q_block_idx;
        current_q_block_idx += kNumCores;

        uint32_t start = 0xffffffffu;
        uint32_t end = 0;
        const uint32_t q_base = q_block_idx * kNumQTokensPerBlock;
        for (uint32_t q_idx = 0; q_idx < kNumQTokensPerBlock; ++q_idx) {
            const uint32_t row_idx = min(q_base + q_idx, num_q_tokens - 1);
            const uint32_t row_start = min(cu_seq_len_k_start[row_idx], num_kv_tokens);
            const uint32_t row_end = min(cu_seq_len_k_end[row_idx], num_kv_tokens);
            start = min(start, row_start);
            end = max(end, row_end);
        }
        block_kv_end = end;
        kv_token_base = start / 8 * 8;
        num_kv_splits = ceil_div(end - kv_token_base, SPLIT_KV);
        return true;
    }

    __aicore__ uint32_t get_q_base(uint32_t q_block_idx) const {
        return q_block_idx * kNumQTokensPerBlock;
    }
    __aicore__ uint32_t get_num_q_tokens_in_block(uint32_t q_block_idx) const {
        return min(kNumQTokensPerBlock, num_q_tokens - get_q_base(q_block_idx));
    }
    // KV token bounds use global query indices.
    __aicore__ uint32_t get_kv_token_begin(uint32_t q_idx) const {
        return cu_seq_len_k_start[q_idx];
    }
    __aicore__ uint32_t get_kv_token_end(uint32_t q_idx) const {
        return cu_seq_len_k_end[q_idx];
    }
    // NOTES: SF strides are in int16_t elements, not bytes.
    __aicore__ uint32_t get_q_sf_stride() const {
        return num_q_tokens * num_heads;
    }
    __aicore__ uint32_t get_kv_token_offset(uint32_t kv_token_base, uint32_t split_idx) const {
        return kv_token_base + split_idx * SPLIT_KV;
    }
    __aicore__ uint32_t get_num_kv_block_tokens(uint32_t kv_token_base, uint32_t split_idx) const {
        const uint32_t offset = get_kv_token_offset(kv_token_base, split_idx);
        return offset < block_kv_end ? min(SPLIT_KV, block_kv_end - offset) : 0;
    }
    __aicore__ uint32_t get_kv_sf_stride() const {
        return num_kv_tokens;
    }
};

} // namespace deep_gemm
