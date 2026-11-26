#pragma once

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/scheduler/mqa_logits.hpp>
#include <deep_gemm/scheduler/paged_mqa_logits.hpp>

namespace deep_gemm {
using namespace __cce_simd;

// Pack SF into 16-token L0B_MX fragments.
__simd_vf__ inline void vf_mqa_logits_pack_scales(__ubuf__ int16_t* scales, uint32_t num_elems) {
    vector_bool mask = asc_create_mask_b16(PAT_ALL);
    vector_int16_t lane, low, high, pair, bits;
    asc_arange(lane, static_cast<int16_t>(0));
    asc_duplicate_scalar(bits, static_cast<int16_t>(FRAC_MN - 1));
    asc_and(low, lane, bits, mask);
    asc_shiftleft_scalar(low, low, 1, mask);
    asc_duplicate_scalar(bits, static_cast<int16_t>(FRAC_MN));
    asc_and(pair, lane, bits, mask);
    asc_shiftright_scalar(pair, pair, __builtin_ctz(FRAC_MN), mask);
    asc_duplicate_scalar(bits, static_cast<int16_t>(-2 * FRAC_MN));
    asc_and(high, lane, bits, mask);
    asc_add(low, low, pair, mask);
    asc_add(low, low, high, mask);
    for (uint32_t offset = 0; offset < num_elems; offset += asc_get_vf_len() / sizeof(int16_t)) {
        vector_int16_t src, dst;
        asc_loadalign(src, scales + offset);
        asc_gather(dst, src, reinterpret_cast<vector_uint16_t&>(low));
        asc_storealign(scales + offset, dst, mask);
    }
}

// MTE3 requires a 32B-aligned UB source.
constexpr uint32_t kLogitsAlignElems = 32 / sizeof(bfloat16_t);

// Align the row head for MTE3.
__simd_vf__ inline void vf_mqa_logits_stage_head(__ubuf__ bfloat16_t* dst, __ubuf__ bfloat16_t* src) {
    vector_load_unalign desc;
    asc_loadunalign_pre(desc, src);
    vector_bfloat16_t v;
    asc_loadunalign(v, desc, src);
    asc_storealign(dst, v, asc_create_mask_b16(PAT_ALL));
}

// Restore logical page order.
// PAGE64 swaps vector halves; PAGE128 swaps vector destinations.
template <uint32_t kPageSize>
__simd_callee__ inline void vf_mqa_logits_store_bf16(
    __ubuf__ bfloat16_t* dst, vector_bfloat16_t src,
    uint32_t reversed_pairs, uint32_t vec_idx
) {
    constexpr uint32_t kNumElemsPerVecReg = asc_get_vf_len() / sizeof(bfloat16_t);
    if constexpr (kPageSize == 128) {
        const uint32_t reversed = (reversed_pairs >> (vec_idx >> 1)) & 1u;
        dst += (vec_idx ^ reversed) * kNumElemsPerVecReg;
    } else {
        dst += vec_idx * kNumElemsPerVecReg;
    }
    if constexpr (kPageSize == 64) {
        vector_bfloat16_t low, high, swapped, unused;
        asc_intlv(low, high, src, src);
        asc_deintlv(swapped, unused, high, low);
        uint32_t num_swap_elems = ((reversed_pairs >> vec_idx) & 1u) * kNumElemsPerVecReg;
        asc_select(src, swapped, src, asc_update_mask_b16(num_swap_elems));
    }
    static_assert(kPageSize == 0 or kPageSize == 64 or kPageSize == 128);
    asc_storealign(dst, src, asc_create_mask_b16(PAT_ALL));
}

template <uint32_t kPageSize, uint32_t kNumElems, uint32_t kNumHeads, uint32_t SPLIT_KV, uint32_t kNumQueries>
__simd_vf__ inline void vf_mqa_logits_reduce(
    __ubuf__ bfloat16_t* score, __ubuf__ bfloat16_t* weights,
    __ubuf__ bfloat16_t* line,
    uint32_t q_token_idx, uint32_t q_idx,
    uint32_t reversed_pairs
) {
    constexpr uint32_t kNumElemsPerVecReg = asc_get_vf_len() / sizeof(bfloat16_t);
    constexpr uint32_t kNumVecRegs = kNumElems / kNumElemsPerVecReg;
    static_assert(kNumElems % kNumElemsPerVecReg == 0 and kNumVecRegs >= 1 and kNumVecRegs <= 4);
    static_assert((kNumQueries == 1 or kNumQueries == 2) and (SPLIT_KV == 256 or SPLIT_KV == 512));
    vector_bool mask = asc_create_mask_b16(PAT_ALL);
    vector_bfloat16_t sum[kNumVecRegs][kNumQueries];
    #pragma unroll
    for (uint32_t vec_idx = 0; vec_idx < kNumVecRegs; ++ vec_idx) {
        #pragma unroll
        for (uint32_t query_idx = 0; query_idx < kNumQueries; ++ query_idx)
            asc_duplicate_scalar(sum[vec_idx][query_idx], static_cast<bfloat16_t>(0), mask);
    }

    const uint32_t qh0 = q_token_idx * kNumHeads;
    const uint32_t qh1 = qh0 + kNumHeads;
    auto weight_ptr_0 = weights + q_idx * aligned(kNumHeads, kLogitsAlignElems);
    auto weight_ptr_1 = weight_ptr_0 + aligned(kNumHeads, kLogitsAlignElems);
    auto score_ptr_0 = score + qh0 * kNumElems;
    auto score_ptr_1 = score + qh1 * kNumElems;
    #pragma unroll 2
    for (uint32_t h = 0; h < kNumHeads; ++ h) {
        vector_bfloat16_t w0, w1;
        asc_loadalign_brc_postupdate(w0, weight_ptr_0, 1);
        if constexpr (kNumQueries == 2)
            asc_loadalign_brc_postupdate(w1, weight_ptr_1, 1);
        #pragma unroll
        for (uint32_t vec_idx = 0; vec_idx < kNumVecRegs; ++ vec_idx) {
            vector_bfloat16_t score_0, score_1;
            asc_loadalign_postupdate(score_0, score_ptr_0, kNumElemsPerVecReg);
            asc_mula(sum[vec_idx][0], score_0, w0, mask);
            if constexpr (kNumQueries == 2) {
                asc_loadalign_postupdate(score_1, score_ptr_1, kNumElemsPerVecReg);
                asc_mula(sum[vec_idx][1], score_1, w1, mask);
            }
        }
    }

    #pragma unroll
    for (uint32_t vec_idx = 0; vec_idx < kNumVecRegs; ++ vec_idx) {
        if constexpr (kPageSize == 64 and kNumQueries == 2) {
            vector_bfloat16_t low, high, swapped0, swapped1;
            asc_intlv(low, high, sum[vec_idx][0], sum[vec_idx][1]);
            asc_deintlv(swapped0, swapped1, high, low);
            uint32_t num_swap_elems = ((reversed_pairs >> vec_idx) & 1u) * kNumElemsPerVecReg;
            const auto swap_mask = asc_update_mask_b16(num_swap_elems);
            asc_select(sum[vec_idx][0], swapped0, sum[vec_idx][0], swap_mask);
            asc_select(sum[vec_idx][1], swapped1, sum[vec_idx][1], swap_mask);
            asc_storealign(line + vec_idx * kNumElemsPerVecReg, sum[vec_idx][0], mask);
            asc_storealign(line + SPLIT_KV + vec_idx * kNumElemsPerVecReg, sum[vec_idx][1], mask);
        } else {
            #pragma unroll
            for (uint32_t query_idx = 0; query_idx < kNumQueries; ++ query_idx)
                vf_mqa_logits_store_bf16<kPageSize>(
                    line + query_idx * SPLIT_KV, sum[vec_idx][query_idx], reversed_pairs, vec_idx);
        }
    }
}

template <uint32_t kPageSize, uint32_t kNumQTokensPerTile, uint32_t kNumHeads, uint32_t SPLIT_KV>
__aicore__ inline uint32_t mqa_logits_reduce_queries(
    __ubuf__ bfloat16_t* score, __ubuf__ bfloat16_t* weight,
    __ubuf__ bfloat16_t* line,
    uint32_t q_token_idx,
    uint32_t num_q_tokens_in_tile, uint32_t issued_n,
    uint32_t reversed_pairs
) {
    uint32_t num_reduced_queries = 1;
    const auto reduce = [&](auto num_elems) __aicore__ {
        constexpr uint32_t kNumElems = decltype(num_elems)::value;
        if constexpr (kNumQTokensPerTile >= 2) {
            if (q_token_idx + 1 < kNumQTokensPerTile and q_token_idx + 1 < num_q_tokens_in_tile) {
                vf_mqa_logits_reduce<kPageSize, kNumElems, kNumHeads, SPLIT_KV, 2>(
                    score, weight, line, q_token_idx, q_token_idx, reversed_pairs);
                num_reduced_queries = 2;
                return;
            }
        }
        vf_mqa_logits_reduce<kPageSize, kNumElems, kNumHeads, SPLIT_KV, 1>(
            score, weight, line, q_token_idx, q_token_idx, reversed_pairs);
    };
    constexpr uint32_t kNumElemsPerVecReg = asc_get_vf_len() / sizeof(bfloat16_t);
    switch (issued_n / kNumElemsPerVecReg) {
        case 1: reduce(std::integral_constant<uint32_t, kNumElemsPerVecReg>{}); break;
        case 2: reduce(std::integral_constant<uint32_t, 2 * kNumElemsPerVecReg>{}); break;
        case 3: reduce(std::integral_constant<uint32_t, 3 * kNumElemsPerVecReg>{}); break;
        case 4: reduce(std::integral_constant<uint32_t, 4 * kNumElemsPerVecReg>{}); break;
        default: ascendc_assert(false, "Unsupported issued_n for MQA logits reduction");
    }
    return num_reduced_queries;
}

template <
    bool kIsFP4,
    uint32_t kNumHeads, uint32_t kHeadDim,
    uint32_t SPLIT_KV, uint32_t MAD_M, uint32_t kNumCDStages,
    uint32_t kNumQStages, uint32_t kNumKVStages,
    uint32_t kPageSize,
    bool kUseAscendKVLayout,
    typename MakeScheduler
>
__aicore__ inline void mqa_logits_core_impl(
    gm_ptr<uint8_t, Major::K> q,
    gm_ptr<uint8_t, Major::K> kv,
    __gm__ int16_t* sf_q,
    __gm__ int16_t* sf_kv,
    gm_ptr<bfloat16_t, Major::K> weights,
    gm_ptr<bfloat16_t, Major::K> logits,
    const MakeScheduler& make_scheduler
) {
    constexpr bool kIsPaged = kPageSize != 0;
    static_assert(kNumKVStages % 2 == 0, "SF stages alternate between two AIVs");
    static_assert(SPLIT_KV == 256 or SPLIT_KV == 512, "Unsupported MQA KV tile");
    static_assert(kNumHeads >= 4 and kNumHeads <= 64 and kNumHeads % 4 == 0,
                  "Number of heads must be a multiple of 4 in [4, 64]");
    static_assert(kHeadDim == 64 or kHeadDim == 128, "Unsupported head dimension");
    using qk_dtype_t = std::conditional_t<kIsFP4, float4_e2m1x2_t, float8_e4m3_t>;

    constexpr uint32_t kNumVecElems = asc_get_vf_len() / sizeof(bfloat16_t);
    constexpr uint32_t MAD_M_PADDED = aligned(MAD_M, FRAC_MN);

    // Bound the Q block by L0A capacity.
    constexpr uint32_t kNumQHTilesPerBlock = MAD_M_PADDED <= 64 ? 8 : 512 / MAD_M_PADDED;
    constexpr uint32_t kNumQTokensPerBlock = kNumQHTilesPerBlock * MAD_M / kNumHeads;

    // One score tile covers MAD_M query/head rows.
    constexpr uint32_t kNumQTokensPerTile = MAD_M / kNumHeads;
    constexpr uint32_t kSFPairs = kHeadDim / MX_SF_DIVISOR;
    static_assert(kNumCDStages == 2 or kNumCDStages == 4, "MQA logits currently supports 2 or 4 CD stages");
    static_assert(MAD_M % kNumHeads == 0, "MAD_M must contain full query/head groups");
    static_assert(kNumQTokensPerTile > 0, "AIV score slice must contain at least one query");
    static_assert(SPLIT_KV != 512 or kNumCDStages == 2, "SPLIT_KV=512 expects 2 CD stages to fit L0C");

    constexpr l1_ptr<qk_dtype_t, Major::K> l1q(kNumQHTilesPerBlock * MAD_M_PADDED, kHeadDim, 0);
    constexpr uint32_t kKVDataBytesPerToken = kHeadDim / (kIsFP4 ? 2 : 1);
    constexpr uint32_t kKVPageBytes = kPageSize * (kKVDataBytesPerToken + kSFPairs * sizeof(int16_t));

    // Keep whole-page DATA and SF together in L1.
    constexpr uint32_t kKVStageRows = SPLIT_KV
        + (kIsPaged and kUseAscendKVLayout ? SPLIT_KV * kSFPairs * sizeof(int16_t) / kKVDataBytesPerToken : 0);
    constexpr l1_ptr<qk_dtype_t, Major::K> l1kv(kKVStageRows, kHeadDim, l1q.offset(kNumQStages));
    constexpr l1_ptr<int16_t> l1_sf_q(l1q.shape_mn, kSFPairs, l1kv.offset(kNumKVStages));
    constexpr l1_ptr<int16_t> l1_sf_kv(SPLIT_KV, kSFPairs, l1_sf_q.offset(kNumQStages));
    constexpr l0a_ptr<qk_dtype_t> l0a(MAD_M_PADDED, kHeadDim);
    constexpr l0b_ptr<qk_dtype_t> l0b(SPLIT_KV, kHeadDim);
    constexpr uint32_t kNumL0BStages =
        ((kIsPaged and kUseAscendKVLayout) or (!kIsFP4 and SPLIT_KV == 256)) and l0b.offset(2) <= L0BSizeBytes ? 2 : 1;

    // Reuse L0A operands across KV splits.
    constexpr uint32_t kNumL0AStages = 4;
    constexpr uint32_t kNumL0ATilesPerStage = kNumQHTilesPerBlock / kNumL0AStages;
    static_assert(kNumQHTilesPerBlock % kNumL0AStages == 0);
    static_assert(kNumL0AStages + kNumL0BStages <= 8, "Operand event overflow");
    constexpr l0c_ptr<float> l0c(MAD_M_PADDED, SPLIT_KV);

    constexpr uint32_t kNumWeightStages = 2;
    constexpr uint32_t kUBBudgetBytes = 248 * 1024;
    constexpr uint32_t kWeightStride = aligned(kNumHeads, kLogitsAlignElems);
    constexpr uint32_t kNumSFUBStages = kNumKVStages / 2;
    constexpr uint32_t kSFUBRows = kIsPaged and !kUseAscendKVLayout ? SPLIT_KV : 0;
    constexpr uint32_t kSharedUBBytes = kNumWeightStages * kNumQTokensPerBlock * kWeightStride * sizeof(bfloat16_t)
        + kNumSFUBStages * kSFUBRows * kSFPairs * sizeof(int16_t) + asc_get_vf_len();

    // Reserve one score and two logits buffers per stage.
    constexpr uint32_t kScoreStageBytes = (MAD_M_PADDED + 2 * kNumQTokensPerTile) * SPLIT_KV * sizeof(bfloat16_t);
    constexpr uint32_t kScoreCapacity = (kUBBudgetBytes - kSharedUBBytes) / kScoreStageBytes;
    constexpr uint32_t kNumScoreStages = kScoreCapacity < 3 ? kScoreCapacity : 3;
    static_assert(kNumScoreStages >= 2);
    constexpr ub_ptr<bfloat16_t> ub_accum(MAD_M_PADDED, SPLIT_KV, 0);
    constexpr ub_ptr<bfloat16_t> ub_weights(kNumQTokensPerBlock, kWeightStride, ub_accum.offset(kNumScoreStages));
    constexpr ub_ptr<bfloat16_t> ub_logits(kNumQTokensPerTile, SPLIT_KV, ub_weights.offset(kNumWeightStages));
    constexpr ub_ptr<bfloat16_t> ub_logits_head(1, asc_get_vf_len() / sizeof(bfloat16_t), ub_logits.offset(2 * kNumScoreStages));
    constexpr ub_ptr<int16_t> ub_sf_kv(kSFUBRows, kSFPairs, ub_logits_head.offset(1));

    // Reserve flags 0/1 for scores/exit.
    constexpr uint32_t kSFIntraFlagBase = 3;
    static_assert(l1_sf_kv.offset(kIsPaged and kUseAscendKVLayout ? 0 : kNumKVStages) <= L1SizeBytes, "L1 overflow");
    static_assert(l0a.offset(kNumQHTilesPerBlock) <= L0ASizeBytes, "L0A overflow");
    static_assert(l0b.offset(kNumL0BStages) <= L0BSizeBytes, "L0B overflow");
    static_assert(l0c.offset(kNumCDStages) <= L0CSizeBytes, "L0C overflow");
    static_assert(ub_sf_kv.offset(kNumSFUBStages) <= kUBBudgetBytes, "Dynamic UB overflow");
    static_assert(!kIsPaged or kSFIntraFlagBase + kNumKVStages <= 16, "Scale flag overflow");

    auto scheduler = make_scheduler(block_idx);
    const gm_ptr<int16_t, Major::MN> sf_q_t(scheduler.get_q_sf_stride(), reinterpret_cast<uintptr_t>(sf_q));
    const gm_ptr<qk_dtype_t, Major::K> q_t(q.stride_outer, q.addr, q.stride_batch);
    const gm_ptr<qk_dtype_t, Major::K> kv_t(kv.stride_outer, kv.addr, kv.stride_batch);
    uint32_t q_block_idx, kv_token_base, num_kv_splits;
    if ASCEND_IS_AIC {
        asc_set_mmad_direction_n();

        constexpr uint32_t kQEventBase = 0;
        constexpr uint32_t kKVEventBase = kNumQStages;
        set_flags<PIPE_M, PIPE_MTE1, kNumL0BStages + kNumL0AStages>();
        set_flags<PIPE_MTE1, PIPE_MTE2, kNumQStages, kQEventBase>();
        set_flags<PIPE_MTE1, PIPE_MTE2, kNumKVStages, kKVEventBase>();

        // Keep SF ownership across Q blocks.
        if constexpr (kIsPaged and !kUseAscendKVLayout)
            for (uint32_t stage_idx = 0; stage_idx < kNumKVStages; ++ stage_idx)
                set_intra_block(PIPE_MTE1, (stage_idx ^ 1u) & 1u, kSFIntraFlagBase + stage_idx);

        uint32_t q_stage_idx = 0, l0a_stage_idx = 0;
        uint32_t kv_stage_idx = 0;
        uint32_t cd_stage_idx = 0, num_active_tiles = 0;
        while (scheduler.next_q_block(q_block_idx, kv_token_base, num_kv_splits)) {
            const uint32_t q_base = scheduler.get_q_base(q_block_idx);
            const uint32_t num_q_tokens_in_block = scheduler.get_num_q_tokens_in_block(q_block_idx);
            const uint32_t num_qh_rows_in_block = num_q_tokens_in_block * kNumHeads;
            const uint32_t num_qh_tiles_in_block = ceil_div(num_qh_rows_in_block, MAD_M);

            asc_sync_wait(PIPE_MTE1, PIPE_MTE2, event_t(kQEventBase + q_stage_idx));
            if constexpr (MAD_M == MAD_M_PADDED) {
                fill_mn_tail_block(l1q[q_stage_idx], num_qh_rows_in_block, aligned(num_qh_rows_in_block, FRAC_MN));
                copy_gm_to_l1(l1q[q_stage_idx], q_t.index(q_base * kNumHeads, 0), num_qh_rows_in_block, kHeadDim,
                              asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                copy_gm_to_l1_mx(l1_sf_q[q_stage_idx], sf_q_t.index(q_base * kNumHeads, 0), num_qh_rows_in_block,
                                 kSFPairs, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
            } else {
                constexpr l1_ptr<int16_t> l1_sf_q_tiles(MAD_M_PADDED, kSFPairs, l1_sf_q.addr);
                const l1_ptr<qk_dtype_t> l1q_m_tiles(MAD_M_PADDED, get_frac_k<qk_dtype_t>(), l1q[q_stage_idx].addr);
                for (uint32_t qh_tile_idx = 0; qh_tile_idx < num_qh_tiles_in_block; ++ qh_tile_idx) {
                    const uint32_t qh_base = qh_tile_idx * MAD_M;
                    const uint32_t num_m_rows = min(MAD_M, num_qh_rows_in_block - qh_base);
                    const uint32_t sf_q_tile_idx = q_stage_idx * kNumQHTilesPerBlock + qh_tile_idx;
                    const auto q_tile = l1q_m_tiles[qh_tile_idx].as_mad_aligned(l1q.shape_mn, kHeadDim);
                    const auto sf_q_tile = l1_sf_q_tiles[sf_q_tile_idx];
                    fill_mn_tail_block(q_tile, num_m_rows, MAD_M_PADDED);
                    copy_gm_to_l1(q_tile, q_t.index(q_base * kNumHeads + qh_base, 0), num_m_rows, kHeadDim,
                                  asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                    copy_gm_to_l1_mx(sf_q_tile, sf_q_t.index(q_base * kNumHeads + qh_base, 0), num_m_rows, kSFPairs,
                                     asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                }
            }
            asc_sync_notify(PIPE_MTE2, PIPE_MTE1, event_t(kQEventBase + q_stage_idx));

            asc_sync_wait(PIPE_MTE2, PIPE_MTE1, event_t(kQEventBase + q_stage_idx));
            const uint32_t num_l0a_stages = ceil_div(num_qh_tiles_in_block, kNumL0ATilesPerStage);
            const uint32_t l0a_tile_base = l0a_stage_idx * kNumL0ATilesPerStage;
            const auto q_ready_event_idx = event_t(kNumL0BStages + l0a_stage_idx);
            for (uint32_t stage_idx = 0; stage_idx < num_l0a_stages; ++ stage_idx)
                asc_sync_wait(PIPE_M, PIPE_MTE1, event_t(kNumL0BStages + (l0a_stage_idx + stage_idx) % kNumL0AStages));
            for (uint32_t qh_tile_idx = 0; qh_tile_idx < num_qh_tiles_in_block; ++ qh_tile_idx) {
                const uint32_t qh_base = qh_tile_idx * MAD_M;
                const uint32_t qh_padded_base = qh_tile_idx * MAD_M_PADDED;
                const uint32_t num_m_rows = min(MAD_M, num_qh_rows_in_block - qh_base);
                const auto l0a_tile = l0a[(l0a_tile_base + qh_tile_idx) % kNumQHTilesPerBlock].as_mad_aligned(num_m_rows, kHeadDim);
                copy_l1_to_l0a(l0a_tile, l1q[q_stage_idx], qh_padded_base, 0);
                copy_l1_to_l0a_mx(l0a_tile, l1_sf_q[q_stage_idx], qh_padded_base, 0, kHeadDim);
            }
            asc_sync_notify(PIPE_MTE1, PIPE_M, q_ready_event_idx);

            // Skip score tiles outside all query ranges.
            const auto is_tile_active = [&](uint32_t q_tile_base, uint32_t kv_idx, uint32_t kv_end) __aicore__ -> bool {
                for (uint32_t q_idx = q_tile_base; q_idx < min(q_tile_base + kNumQTokensPerTile, num_q_tokens_in_block); ++ q_idx) {
                    const uint32_t row = q_base + q_idx;
                    if constexpr (kIsPaged) {
                        if (kv_idx < scheduler.context_lens[row])
                            return true;
                    } else if (kv_idx < scheduler.cu_seq_len_k_end[row] and
                               scheduler.cu_seq_len_k_start[row] < kv_end) {
                        return true;
                    }
                }
                return false;
            };

            asc_sync_wait(PIPE_MTE1, PIPE_M, q_ready_event_idx);
            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                const uint32_t kv_idx = scheduler.get_kv_token_offset(kv_token_base, kv_split_idx);
                const uint32_t actual_kv = scheduler.get_num_kv_block_tokens(kv_token_base, kv_split_idx);
                const uint32_t issued_n = aligned(actual_kv, kNumVecElems);
                asc_sync_wait(PIPE_MTE1, PIPE_MTE2, event_t(kKVEventBase + kv_stage_idx));
                if constexpr (kIsPaged and kUseAscendKVLayout) {
                    if (actual_kv < issued_n)
                        fill_l1(l1kv[kv_stage_idx].template ptr<int16_t>(), static_cast<int16_t>(0),
                                1, ceil_div(issued_n, kPageSize) * kKVPageBytes / 32, 0);
                } else {
                    fill_mn_tail_block(l1kv[kv_stage_idx], actual_kv, issued_n);
                }
                if constexpr (kIsPaged) {
                    const asc_load_l2_cache_mode kv_load_hint = scheduler.has_multiple_q_blocks()
                        ? asc_load_l2_cache_mode::NORMAL_LAST_VICTIM : asc_load_l2_cache_mode::NOTALLOC_KEEP;
                    const uint32_t num_pages = ceil_div(actual_kv, kPageSize);
                    if constexpr (kIsPaged and kUseAscendKVLayout) {
                        asc_set_gm2l1_loop_size(1, 1);
                        asc_set_gm2l1_loop1_stride(0, 0);
                        asc_set_gm2l1_loop2_stride(0, 0);
                    }
                    for (uint32_t compact_page = 0; compact_page < num_pages; compact_page += 2) {
                        uint32_t first_page, page_gap;
                        const uint32_t num_batch = scheduler.get_physical_page_batch(
                            kv_token_base, kv_split_idx, compact_page, num_pages, first_page, page_gap);
                        const auto page_src = kv_t + scheduler.get_kv_page_offset_bytes(first_page);
                        if constexpr (kIsPaged and kUseAscendKVLayout) {
                            const auto page_dst = l1kv[kv_stage_idx].offset_bytes(compact_page * kKVPageBytes);
                            asc_copy_gm2l1_align(page_dst.template ptr<uint8_t>(), page_src.template ptr<uint8_t>(),
                                                 num_batch, kKVPageBytes, 0, 0, false, kv_load_hint,
                                                 page_gap * scheduler.kv_page_stride_bytes, kKVPageBytes);
                        } else {
                            const auto page_dst = l1kv[kv_stage_idx].offset_bytes(compact_page * kPageSize * 32u);
                            const uint32_t valid_rows = min(kPageSize, actual_kv - compact_page * kPageSize);
                            copy_gm_to_l1<kUseAscendKVLayout>(
                                page_dst, page_src, valid_rows, kHeadDim, kv_load_hint,
                                num_batch, kPageSize, page_gap * scheduler.kv_page_stride_bytes);
                        }
                    }
                } else {
                    const gm_ptr<int16_t, Major::MN> sf_kv_t(scheduler.get_kv_sf_stride(), reinterpret_cast<uintptr_t>(sf_kv));
                    copy_gm_to_l1(l1kv[kv_stage_idx], kv_t.index(kv_idx, 0), actual_kv, kHeadDim,
                                  asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                    copy_gm_to_l1_mx(l1_sf_kv[kv_stage_idx], sf_kv_t.index(kv_idx, 0), actual_kv, kSFPairs,
                                     asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                }
                asc_sync_notify(PIPE_MTE2, PIPE_MTE1, event_t(kKVEventBase + kv_stage_idx));
                asc_sync_wait(PIPE_MTE2, PIPE_MTE1, event_t(kKVEventBase + kv_stage_idx));

                asc_sync_wait(PIPE_M, PIPE_MTE1, event_t(kv_stage_idx % kNumL0BStages));
                const auto l0b_tile = l0b[kv_stage_idx % kNumL0BStages].as_mad_aligned(issued_n, kHeadDim);
                if constexpr (kIsPaged and kUseAscendKVLayout) {
                    for (uint32_t page_idx = 0; page_idx < ceil_div(issued_n, kPageSize); ++ page_idx) {
                        const auto page_src = l1kv[kv_stage_idx].offset_bytes(page_idx * kKVPageBytes);
                        const auto page_dst = l0b_tile.offset_bytes(page_idx * kPageSize * 32u);
                        asc_copy_l12l0b(page_dst.ptr(), page_src.ptr(), 0, 0,
                                        kPageSize / FRAC_MN, kHeadDim / get_frac_k<qk_dtype_t>(),
                                        kPageSize / FRAC_MN, issued_n / FRAC_MN);
                    }
                    asc_copy_l12l0b_mx(l0b_tile.addr / MX_ADDR_DIV,
                                      l1kv[kv_stage_idx].offset_bytes(kPageSize * kKVDataBytesPerToken).template ptr<fp8_e8m0_t>(),
                                      0, 0, ceil_div(issued_n, kPageSize), kPageSize * kSFPairs * sizeof(int16_t) / 32,
                                      kKVPageBytes / 32, kPageSize * kSFPairs * sizeof(int16_t) / 32);
                } else {
                    copy_l1_to_l0b(l0b_tile, l1kv[kv_stage_idx], 0, 0);
                    if constexpr (kIsPaged and !kUseAscendKVLayout)
                        wait_intra_block(PIPE_MTE1, (kv_stage_idx ^ 1u) & 1u, kSFIntraFlagBase + kv_stage_idx);
                    copy_l1_to_l0b_mx(l0b_tile, l1_sf_kv[kv_stage_idx], 0, 0, kHeadDim);
                    if constexpr (kIsPaged and !kUseAscendKVLayout)
                        set_intra_block(PIPE_MTE1, (kv_stage_idx ^ 1u) & 1u, kSFIntraFlagBase + kv_stage_idx);
                }
                asc_sync_notify(PIPE_MTE1, PIPE_M, event_t(kv_stage_idx % kNumL0BStages));
                asc_sync_wait(PIPE_MTE1, PIPE_M, event_t(kv_stage_idx % kNumL0BStages));

                for (uint32_t qh_tile_idx = 0; qh_tile_idx < num_qh_tiles_in_block; ++ qh_tile_idx) {
                    const uint32_t qh_base = qh_tile_idx * MAD_M;
                    const uint32_t q_tile_base = qh_base / kNumHeads;
                    if (!is_tile_active(q_tile_base, kv_idx, kv_idx + actual_kv))
                        continue;

                    const uint32_t num_m_rows = min(MAD_M, num_qh_rows_in_block - qh_base);
                    const auto l0c_tile = l0c[cd_stage_idx].as_mad_aligned(num_m_rows, issued_n);
                    const uint32_t accum_owner = ((num_active_tiles + kNumScoreStages - 1u) / kNumScoreStages) & 1u;
                    const uint32_t accum_stage_idx = num_active_tiles ? (num_active_tiles - 1u) % kNumScoreStages : 0;
                    mad(l0c_tile, l0a[(l0a_tile_base + qh_tile_idx) % kNumQHTilesPerBlock],
                        l0b_tile, num_m_rows, issued_n, kHeadDim, asc_unit_flag_mode::ENABLE_UPDATE, true);
                    if (!accum_stage_idx)
                        wait_intra_block(PIPE_FIX, accum_owner, 0);
                    const auto accum_tile = ub_accum[accum_stage_idx].as_mad_aligned(num_m_rows, issued_n);
                    copy_l0c_to_ub(accum_tile, l0c_tile, num_m_rows, issued_n, asc_dual_dst_mode::DUAL_DST_DISABLE,
                                   accum_owner, infer_quant_mode<bfloat16_t, float>(), true, false,
                                   asc_unit_flag_mode::ENABLE_UPDATE);
                    set_intra_block(PIPE_FIX, accum_owner, 0);
                    ++ num_active_tiles;
                    cd_stage_idx = (cd_stage_idx + 1) % kNumCDStages;
                }
                asc_sync_notify(PIPE_M, PIPE_MTE1, event_t(kv_stage_idx % kNumL0BStages));
                asc_sync_notify(PIPE_MTE1, PIPE_MTE2, event_t(kKVEventBase + kv_stage_idx));
                kv_stage_idx = (kv_stage_idx + 1) % kNumKVStages;
            }
            for (uint32_t stage_idx = 0; stage_idx < num_l0a_stages; ++ stage_idx)
                asc_sync_notify(PIPE_M, PIPE_MTE1, event_t(kNumL0BStages + (l0a_stage_idx + stage_idx) % kNumL0AStages));
            l0a_stage_idx = (l0a_stage_idx + num_l0a_stages) % kNumL0AStages;
            asc_sync_notify(PIPE_MTE1, PIPE_MTE2, event_t(kQEventBase + q_stage_idx));
            q_stage_idx = (q_stage_idx + 1) % kNumQStages;
        }

        // Drain credits and join both AIVs.
        for (uint32_t i = 0; i < kNumL0BStages + kNumL0AStages; ++ i)
            asc_sync_wait(PIPE_M, PIPE_MTE1, event_t(i));
        for (uint32_t i = 0; i < kNumQStages + kNumKVStages; ++ i)
            asc_sync_wait(PIPE_MTE1, PIPE_MTE2, event_t(i));
        for (uint32_t owner = 0; owner < 2; ++ owner)
            wait_intra_block(PIPE_FIX, owner, 0);
        for (uint32_t owner = 0; owner < 2; ++ owner)
            set_intra_block(PIPE_FIX, owner, 1);
        for (uint32_t owner = 0; owner < 2; ++ owner)
            wait_intra_block(PIPE_FIX, owner, 1);
    } else {
        const uint32_t sub_id = asc_get_sub_block_id();
        constexpr uint32_t kSFLoadEventIdx = kNumWeightStages;
        constexpr uint32_t kSFStoreEventBase = 4;
        static_assert(kSFStoreEventBase + kNumSFUBStages <= 8);
        if constexpr (kIsPaged and !kUseAscendKVLayout)
            set_flags<PIPE_MTE3, PIPE_MTE2, kNumSFUBStages, kSFLoadEventIdx>();
        uint32_t kv_stage_idx = 0;

        // Prefetch SF with an independent scheduler.
        auto sf_scheduler = scheduler;
        uint32_t sf_q_block_idx = 0, sf_kv_token_base = 0, num_sf_splits = 0, sf_split_idx = 0;
        uint32_t sf_num_pages[kNumSFUBStages] = {};
        struct QBlockInfo {
            PagedMQARequestInfo request;
            uint32_t q_block_idx, kv_token_base, num_splits;
        };
        QBlockInfo q_block_infos[kNumKVStages];
        const auto prefetch_sf_kv = [&](uint32_t kv_stage_idx) __aicore__ {
            if constexpr (kIsPaged and !kUseAscendKVLayout) {
                const uint32_t sf_stage_idx = kv_stage_idx / 2;
                if (sf_split_idx == num_sf_splits) {
                    if (!sf_scheduler.next_q_block(sf_q_block_idx, sf_kv_token_base, num_sf_splits)) {
                        // Empty slots publish no ready credit.
                        q_block_infos[kv_stage_idx].num_splits = 0;
                        if (sub_id == ((kv_stage_idx ^ 1u) & 1u))
                            sf_num_pages[sf_stage_idx] = 0;
                        return;
                    }
                    sf_split_idx = 0;
                }

                // Cache request metadata for consumption.
                q_block_infos[kv_stage_idx] = {sf_scheduler.request, sf_q_block_idx, sf_kv_token_base, num_sf_splits};

                // Both AIVs advance; only the owner loads SF.
                const uint32_t kv_split_idx = sf_split_idx++;
                const uint32_t num_pages = ceil_div(
                    sf_scheduler.get_num_kv_block_tokens(sf_kv_token_base, kv_split_idx), kPageSize);
                const auto pages = sf_scheduler.block_table +
                    static_cast<uint64_t>(sf_scheduler.request.q_token_start) * sf_scheduler.block_table_row_stride +
                    sf_scheduler.get_kv_token_offset(sf_kv_token_base, kv_split_idx) / kPageSize;
                const uintptr_t page_line = reinterpret_cast<uintptr_t>(pages) & ~uintptr_t{63};
                asc_datacache_preload(reinterpret_cast<__gm__ uint64_t*>(page_line), 0);
                if (reinterpret_cast<uintptr_t>(pages + num_pages - 1) - page_line >= 64)
                    asc_datacache_preload(reinterpret_cast<__gm__ uint64_t*>(page_line), 64);
                if (sub_id != ((kv_stage_idx ^ 1u) & 1u))
                    return;
                sf_num_pages[sf_stage_idx] = num_pages;
                const asc_load_l2_cache_mode sf_load_hint = sf_scheduler.has_multiple_q_blocks()
                    ? asc_load_l2_cache_mode::NORMAL_LAST_VICTIM : asc_load_l2_cache_mode::NOTALLOC_KEEP;
                asc_sync_wait(PIPE_MTE3, PIPE_MTE2, event_t(kSFLoadEventIdx + sf_stage_idx));
                for (uint32_t compact_page = 0; compact_page < num_pages; compact_page += 2) {
                    uint32_t first_page, page_gap;
                    const uint32_t num_batch = sf_scheduler.get_physical_page_batch(
                        sf_kv_token_base, kv_split_idx, compact_page, num_pages, first_page, page_gap);
                    const auto sf_page_src = reinterpret_cast<__gm__ int16_t*>(
                        reinterpret_cast<uintptr_t>(sf_kv) + sf_scheduler.get_kv_page_offset_bytes(first_page));
                    copy_gm_to_ub(ub_sf_kv[sf_stage_idx].ptr() + compact_page * kPageSize * kSFPairs,
                                  sf_page_src, num_batch, kPageSize * kSFPairs,
                                  page_gap * sf_scheduler.kv_page_stride_bytes, sf_load_hint);
                }
                asc_sync_notify(PIPE_MTE2, PIPE_V, event_t(kSFLoadEventIdx + sf_stage_idx));
            }
        };
        const auto publish_sf_kv = [&](uint32_t kv_stage_idx) __aicore__ {
            if constexpr (kIsPaged and !kUseAscendKVLayout) {
                if (sub_id != ((kv_stage_idx ^ 1u) & 1u))
                    return;
                const uint32_t sf_stage_idx = kv_stage_idx / 2;
                const uint32_t num_pages = sf_num_pages[sf_stage_idx];
                if (!num_pages)
                    return;
                asc_sync_wait(PIPE_MTE2, PIPE_V, event_t(kSFLoadEventIdx + sf_stage_idx));
                if constexpr (!kUseAscendKVLayout and kSFPairs == 2)
                    vf_mqa_logits_pack_scales(ub_sf_kv[sf_stage_idx].ptr(), num_pages * kPageSize * kSFPairs);
                asc_sync_notify(PIPE_V, PIPE_MTE3, event_t(kSFStoreEventBase + sf_stage_idx));
                asc_sync_wait(PIPE_V, PIPE_MTE3, event_t(kSFStoreEventBase + sf_stage_idx));
                wait_intra_block(PIPE_MTE3, sub_id, kSFIntraFlagBase + kv_stage_idx);
                asc_copy_ub2l1(l1_sf_kv[kv_stage_idx].ptr(), ub_sf_kv[sf_stage_idx].ptr(),
                               num_pages * kPageSize * kSFPairs * sizeof(int16_t));
                set_intra_block(PIPE_MTE3, sub_id, kSFIntraFlagBase + kv_stage_idx);
                asc_sync_notify(PIPE_MTE3, PIPE_MTE2, event_t(kSFLoadEventIdx + sf_stage_idx));
            }
        };
        if constexpr (kIsPaged and !kUseAscendKVLayout) {
            for (uint32_t stage_idx = 0; stage_idx < kNumKVStages; ++ stage_idx)
                prefetch_sf_kv(stage_idx);
            for (uint32_t stage_idx = 0; stage_idx < kNumKVStages; ++ stage_idx)
                publish_sf_kv(stage_idx);
        }

        auto scratch = ub_logits_head.ptr();
        struct LogitsStore {
            __gm__ bfloat16_t* dst;
            uint32_t src_offset, num_elems;
        };
        LogitsStore pending_logits_stores[kNumScoreStages * kNumQTokensPerTile];
        uint32_t num_pending_logits_stores = 0, pending_batch_idx = 0, logits_batch_idx = 0;
        const auto store_pending_logits = [&]() __aicore__ {
            auto pending_logits = ub_logits[pending_batch_idx * kNumScoreStages].ptr();
            for (uint32_t store_idx = 0; store_idx < num_pending_logits_stores; ++ store_idx) {
                const auto store = pending_logits_stores[store_idx];
                if (!store.num_elems)
                    continue;
                auto src = pending_logits + store_idx * SPLIT_KV + store.src_offset;
                const uint32_t num_prefix_elems = min(
                    store.num_elems, (kLogitsAlignElems - (store.src_offset & (kLogitsAlignElems - 1))) & (kLogitsAlignElems - 1));
                if (num_prefix_elems) {
                    asc_sync_notify(PIPE_MTE3, PIPE_V, EVENT_ID1);
                    asc_sync_wait(PIPE_MTE3, PIPE_V, EVENT_ID1);
                    vf_mqa_logits_stage_head(scratch, src);
                    asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID1);
                    asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID1);
                    asc_copy_ub2gm_align(store.dst, scratch, 1, num_prefix_elems * sizeof(bfloat16_t),
                                         asc_store_l2_cache_mode::NOTALLOC_CLEAN,
                                         num_prefix_elems * sizeof(bfloat16_t), num_prefix_elems * sizeof(bfloat16_t));
                }
                if (num_prefix_elems < store.num_elems) {
                    const uint32_t num_bytes = (store.num_elems - num_prefix_elems) * sizeof(bfloat16_t);
                    asc_copy_ub2gm_align(store.dst + num_prefix_elems, src + num_prefix_elems, 1, num_bytes,
                                         asc_store_l2_cache_mode::NOTALLOC_CLEAN, num_bytes, num_bytes);
                }
            }
        };

        // Cache weights per score stage.
        constexpr uint32_t kNumWeightTiles = kNumScoreStages;
        static_assert(kNumWeightTiles <= kNumWeightStages * kNumQHTilesPerBlock);
        uint32_t weight_queries[kNumWeightTiles];
        for (uint32_t i = 0; i < kNumWeightTiles; ++ i)
            weight_queries[i] = 0xffffffffu;

        set_intra_block(PIPE_MTE3, sub_id, 0);
        asc_sync_notify(PIPE_V, PIPE_MTE2, EVENT_ID1);

        uint32_t num_active_tiles = 0, num_score_tiles = 0;
        struct TileInfo {
            uint32_t q_idx, num_q_tokens_in_tile, weight_idx, kv_idx, actual_kv, issued_n;
        };
        TileInfo tile_desc[kNumScoreStages];
        uint32_t score_reversed_pairs[kNumScoreStages];

        const auto consume_score_tiles = [&]() __aicore__ {
            if (!num_score_tiles)
                return;

            // Protect prior VF reads before prefetching weights.
            asc_sync_wait(PIPE_V, PIPE_MTE2, EVENT_ID1);
            for (uint32_t stage_idx = 0; stage_idx < num_score_tiles; ++ stage_idx) {
                auto& desc = tile_desc[stage_idx];
                const uint32_t weight_idx = stage_idx;
                if (weight_queries[weight_idx] != desc.q_idx) {
                    auto weight = ub_weights.ptr() + weight_idx * kNumQTokensPerTile * kWeightStride;
                    asc_copy_gm2ub_align(weight, weights.index(desc.q_idx, 0).ptr(), desc.num_q_tokens_in_tile,
                                         kNumHeads * sizeof(bfloat16_t), 0, 0, false, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM,
                                         weights.stride_outer * sizeof(bfloat16_t), kWeightStride * sizeof(bfloat16_t));
                    weight_queries[weight_idx] = desc.q_idx;
                }
                desc.weight_idx = weight_idx;
            }

            // Wait for weights and all scores.
            for (uint32_t stage_idx = 0; stage_idx < num_score_tiles; ++ stage_idx)
                wait_intra_block(PIPE_MTE2, sub_id, 0);
            asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID0);
            asc_sync_wait(PIPE_MTE2, PIPE_V, EVENT_ID0);
            asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID2);

            for (uint32_t stage_idx = 0; stage_idx < num_score_tiles; ++ stage_idx) {
                const auto desc = tile_desc[stage_idx];
                uint32_t reversed_pairs = 0;
                if constexpr (kIsPaged)
                    reversed_pairs = score_reversed_pairs[stage_idx];

                auto weight = ub_weights.ptr() + desc.weight_idx * kNumQTokensPerTile * kWeightStride;
                auto lines = ub_logits[logits_batch_idx * kNumScoreStages + stage_idx].ptr();
                for (uint32_t q_token_idx = 0; q_token_idx < desc.num_q_tokens_in_tile;)
                    q_token_idx += mqa_logits_reduce_queries<kPageSize, kNumQTokensPerTile, kNumHeads, SPLIT_KV>(
                        ub_accum[stage_idx].ptr(), weight, lines + q_token_idx * SPLIT_KV,
                        q_token_idx, desc.num_q_tokens_in_tile, desc.issued_n, reversed_pairs);
            }

            // Release weights, store prior logits, and return score credits.
            asc_sync_notify(PIPE_V, PIPE_MTE2, EVENT_ID1);
            asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID2);
            store_pending_logits();
            asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
            asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
            set_intra_block(PIPE_MTE3, sub_id, 0);

            // Defer this batch's output until the next batch or exit.
            for (uint32_t stage_idx = 0; stage_idx < num_score_tiles; ++ stage_idx) {
                const auto desc = tile_desc[stage_idx];
                for (uint32_t q_token_idx = 0; q_token_idx < kNumQTokensPerTile; ++ q_token_idx) {
                    auto& store = pending_logits_stores[stage_idx * kNumQTokensPerTile + q_token_idx];
                    store.num_elems = 0;
                    const uint32_t q_idx = desc.q_idx + q_token_idx;
                    if (q_token_idx >= desc.num_q_tokens_in_tile)
                        continue;

                    const uint32_t kv_token_begin = scheduler.get_kv_token_begin(q_idx), kv_token_end = scheduler.get_kv_token_end(q_idx);
                    const uint32_t begin = max(desc.kv_idx, kv_token_begin), end = min(desc.kv_idx + desc.actual_kv, kv_token_end);
                    store.num_elems = end > begin ? end - begin : 0;
                    store.src_offset = begin - desc.kv_idx;
                    store.dst = logits.index(q_idx, begin - kv_token_begin).ptr();
                }
            }

            num_pending_logits_stores = num_score_tiles * kNumQTokensPerTile;
            pending_batch_idx = logits_batch_idx;
            logits_batch_idx ^= 1;
            num_score_tiles = 0;
        };

        bool drain = false;
        while (!drain) {
            if constexpr (kIsPaged and !kUseAscendKVLayout) {
                const auto& desc = q_block_infos[kv_stage_idx];
                drain = desc.num_splits == 0;
                if (!drain) {
                    scheduler.request = desc.request;
                    q_block_idx = desc.q_block_idx;
                    kv_token_base = desc.kv_token_base;
                    num_kv_splits = desc.num_splits;
                }
            } else {
                drain = !scheduler.next_q_block(q_block_idx, kv_token_base, num_kv_splits);
            }

            const uint32_t q_base = drain ? 0 : scheduler.get_q_base(q_block_idx);
            const uint32_t num_q_tokens = drain ? 0 : scheduler.get_num_q_tokens_in_block(q_block_idx);
            const uint32_t num_tiles = ceil_div(num_q_tokens * kNumHeads, MAD_M);

            for (uint32_t kv_split_idx = 0; kv_split_idx < (drain ? 1u : num_kv_splits); ++ kv_split_idx) {
                uint32_t kv_idx = 0, actual_kv = 0, issued_n = 0, reversed_pairs = 0;
                if (!drain) {
                    if constexpr (kIsPaged and !kUseAscendKVLayout)
                        prefetch_sf_kv(kv_stage_idx);
                    kv_idx = scheduler.get_kv_token_offset(kv_token_base, kv_split_idx);
                    actual_kv = scheduler.get_num_kv_block_tokens(kv_token_base, kv_split_idx);
                    issued_n = aligned(actual_kv, kNumVecElems);
                    if constexpr (kIsPaged) {
                        const uint32_t num_pages = ceil_div(actual_kv, kPageSize);
                        for (uint32_t compact_page = 0; compact_page + 1 < num_pages; compact_page += 2)
                            reversed_pairs |= static_cast<uint32_t>(scheduler.get_physical_page(kv_token_base, kv_split_idx, compact_page) >
                                scheduler.get_physical_page(kv_token_base, kv_split_idx, compact_page + 1)) << (compact_page / 2);
                    }
                }

                // Consume full and final partial batches here.
                for (uint32_t tile_idx = 0; tile_idx < (drain ? 1u : num_tiles); ++ tile_idx) {
                    if (!drain) {
                        bool active = false;
                        for (uint32_t q_token_idx = tile_idx * kNumQTokensPerTile; q_token_idx < min(num_q_tokens, tile_idx * kNumQTokensPerTile + kNumQTokensPerTile); ++ q_token_idx)
                            active |= kv_idx < scheduler.get_kv_token_end(q_base + q_token_idx)
                                and scheduler.get_kv_token_begin(q_base + q_token_idx) < kv_idx + actual_kv;
                        if (!active)
                            continue;

                        const uint32_t owner = ((num_active_tiles + kNumScoreStages - 1u) / kNumScoreStages) & 1u;
                        ++ num_active_tiles;
                        if (owner != sub_id)
                            continue;

                        if constexpr (kIsPaged)
                            score_reversed_pairs[num_score_tiles] = reversed_pairs;
                        tile_desc[num_score_tiles++] = {
                            q_base + tile_idx * kNumQTokensPerTile,
                            min(kNumQTokensPerTile, num_q_tokens - tile_idx * kNumQTokensPerTile),
                            tile_idx, kv_idx, actual_kv, issued_n};
                        if (num_score_tiles != kNumScoreStages and num_active_tiles != 1u)
                            continue;
                    }
                    consume_score_tiles();
                }

                if (!drain) {
                    if constexpr (kIsPaged and !kUseAscendKVLayout)
                        publish_sf_kv(kv_stage_idx);
                    kv_stage_idx = (kv_stage_idx + 1) % kNumKVStages;
                }
            }
        }

        asc_sync_wait(PIPE_V, PIPE_MTE2, EVENT_ID1);

        // Flush local logits before joining the other AIV.
        asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID2);
        asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID2);
        store_pending_logits();

        if constexpr (kIsPaged and !kUseAscendKVLayout) {
            // Drain SF credits returned by AIC.
            for (uint32_t i = 0; i < kNumKVStages; ++ i)
                if (sub_id == ((i ^ 1u) & 1u))
                    wait_intra_block(PIPE_MTE3, sub_id, kSFIntraFlagBase + i);
            for (uint32_t i = 0; i < kNumSFUBStages; ++ i)
                asc_sync_wait(PIPE_MTE3, PIPE_MTE2, event_t(kSFLoadEventIdx + i));
        }

        wait_intra_block(PIPE_V, sub_id, 1);
        asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID2);
        asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID2);
        set_intra_block(PIPE_MTE3, sub_id, 1);
        asc_sync_notify(PIPE_MTE3, PIPE_S, EVENT_ID0);
        asc_sync_wait(PIPE_MTE3, PIPE_S, EVENT_ID0);
    }
}

template <
    bool kIsFP4,
    uint32_t kNumHeads, uint32_t kHeadDim,
    uint32_t SPLIT_KV, uint32_t kMadM, uint32_t kNumCDStages,
    uint32_t kNumQStages, uint32_t kNumKVStages,
    uint32_t kNumCores
>
__global__ __mix__(1, 2) void mqa_logits_impl(
    uint32_t num_q_tokens, uint32_t num_kv_tokens,
    gm_ptr<uint8_t, Major::K> q, gm_ptr<uint8_t, Major::K> kv,
    __gm__ int16_t* sf_q, __gm__ int16_t* sf_kv,
    gm_ptr<bfloat16_t, Major::K> weights,
    gm_ptr<bfloat16_t, Major::K> logits,
    __gm__ uint32_t* cu_seq_len_k_start, __gm__ uint32_t* cu_seq_len_k_end
) {
    asc_init();
    constexpr uint32_t kNumQTokensPerBlock = 8 * kMadM / kNumHeads;
    const auto make_scheduler = [&](uint32_t core_id) __aicore__ {
        return MQALogitsScheduler<kNumQTokensPerBlock, SPLIT_KV, kNumCores>(
            core_id, num_q_tokens, num_kv_tokens, cu_seq_len_k_start, cu_seq_len_k_end, kNumHeads);
    };
    mqa_logits_core_impl<kIsFP4, kNumHeads, kHeadDim, SPLIT_KV, kMadM, kNumCDStages,
                         kNumQStages, kNumKVStages,
                         0u, false, decltype(make_scheduler)>(
        q, kv, sf_q, sf_kv, weights, logits, make_scheduler);
}

template <
    bool kIsFP4,
    uint32_t kNumHeads, uint32_t kHeadDim,
    uint32_t kPageSize,
    uint32_t SPLIT_KV, uint32_t kMadM, uint32_t kNumCDStages,
    uint32_t kNumQStages, uint32_t kNumKVStages,
    bool kUseAscendKVLayout
>
__global__ __mix__(1, 2) void paged_mqa_logits_impl(
    uint32_t num_q_tokens,
    uint32_t block_table_row_stride, uint64_t kv_page_stride_bytes,
    gm_ptr<uint8_t, Major::K> q, __gm__ int16_t* sf_q,
    gm_ptr<uint8_t, Major::K> fused_kv,
    gm_ptr<bfloat16_t, Major::K> weights,
    gm_ptr<bfloat16_t, Major::K> logits,
    __gm__ uint32_t* context_lens, __gm__ uint32_t* block_table,
    __gm__ uint32_t* indices, __gm__ uint32_t* metadata
) {
    static_assert(kPageSize == 64 or kPageSize == 128, "Paged MQA logits supports PAGE_SIZE=64/128");
    static_assert(SPLIT_KV % kPageSize == 0, "SPLIT_KV must contain complete pages");
    static_assert(SPLIT_KV / kPageSize <= 8, "Page reorder supports at most four page pairs");
    asc_init();

    const auto kv = fused_kv;
    const auto sf_kv = (fused_kv + kPageSize * fused_kv.stride_outer).template ptr<int16_t>();

    constexpr uint32_t kNumQTokensPerBlock = 8 * (kMadM <= 64 ? kMadM : 64) / kNumHeads;
    const auto make_scheduler = [&](uint32_t core_id) __aicore__ {
        auto scheduler = PagedMQALogitsScheduler<kNumQTokensPerBlock, SPLIT_KV, kPageSize, kNumHeads>(
            core_id, num_q_tokens, context_lens, indices, metadata, block_table,
            block_table_row_stride, kv_page_stride_bytes);
        return scheduler;
    };
    mqa_logits_core_impl<kIsFP4, kNumHeads, kHeadDim, SPLIT_KV, kMadM, kNumCDStages,
                         kNumQStages, kNumKVStages,
                         kPageSize, kUseAscendKVLayout, decltype(make_scheduler)>(
        q, kv, sf_q, sf_kv, weights, logits, make_scheduler);
}

} // namespace deep_gemm
