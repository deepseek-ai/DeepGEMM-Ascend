#pragma once

#include <deep_gemm/epilogue/operators.hpp>

namespace deep_gemm {

template <
    typename cd_dtype_t,
    template <typename> class epilogue_operator_t,
    bool kWithAccumulation,
    uint32_t BLOCK_M, uint32_t BLOCK_N,
    uint32_t MAD_M, uint32_t MAD_N,
    uint32_t kNumStagesPerAIV, uint32_t kNumAIVs,
    asc_store_l2_cache_mode kL2CacheMode = asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
    bool kZeroPadding = false>
struct Epilogue {
    using params_t = epilogue::Params<cd_dtype_t, BLOCK_M, BLOCK_N, MAD_M, MAD_N,
                                      kNumStagesPerAIV, kWithAccumulation, kL2CacheMode>;
    using operator_t = epilogue_operator_t<params_t>;
    using ub_dtype_t = typename operator_t::ub_dtype_t;
    static constexpr pipe_t kPipe = operator_t::kPipe;
    epilogue::Args args;

    // Assign complete MAD tiles round-robin across available AIVs.
    static constexpr uint32_t kNumStages = kNumStagesPerAIV * kNumAIVs;
    static constexpr uint32_t kOutputOffsetBytes = operator_t::kOutputOffsetBytes;
    static constexpr ub_ptr<ub_dtype_t> ub{MAD_M, MAD_N};
    static constexpr operator_t op{ub.offset(kNumStagesPerAIV)};

    static_assert(kOutputOffsetBytes + MAD_M * MAD_N * sizeof(cd_dtype_t) <= MAD_M * MAD_N * sizeof(ub_dtype_t),
                  "In-place epilogue output must fit in its input UB");
    static_assert(MAD_N % get_frac_k<ub_dtype_t>() == 0, "L0C-to-UB row stride must be 32-byte aligned");
    static_assert(MAD_M % FRAC_MN == 0, "epilogue row alignment must divide the MAD tile");
    static_assert(kNumAIVs == 1 || kNumAIVs == 2, "epilogue requires one or two AIVs");
    static_assert(kNumStagesPerAIV > 0 && kNumStagesPerAIV <= 16, "invalid intra-block stage count");
    static_assert(kNumStages >= (BLOCK_M / MAD_M) * (BLOCK_N / MAD_N), "epilogue stages must be >= CD stages");
    static_assert(op.is_supported(), "Unsupported epilogue output or accumulation");
    static_assert(op.ub_end() <= UBSizeBytes, "UB overflow");
    static_assert(!kZeroPadding || !kWithAccumulation, "Zero padding is not compatible with accumulation");

    __aicore__ void aic_wait_free(uint32_t s_epi) const {
        wait_intra_block(PIPE_FIX, 2 - kNumAIVs + s_epi % kNumAIVs, s_epi / kNumAIVs);
    }

    template <typename L0C>
    __aicore__ void aic_store(L0C l0c_tile, uint32_t s_epi, uint32_t m_elems, uint32_t n_elems) const {
        const uint32_t aiv_id = 2 - kNumAIVs + s_epi % kNumAIVs;
        const uint32_t stage_idx = s_epi / kNumAIVs;
        copy_l0c_to_ub(ub[stage_idx], l0c_tile, m_elems, n_elems, asc_dual_dst_mode::DUAL_DST_DISABLE, aiv_id);
        set_intra_block(PIPE_FIX, aiv_id, stage_idx);
    }

    __aicore__ void aic_insert_padding_stages(uint32_t actual_m, uint32_t actual_n, uint32_t num_padding_rows, uint32_t& s_epi) const {
        if constexpr (kZeroPadding) {
            if (num_padding_rows && actual_m % MAD_M == 0) {
                for (uint32_t n_mad = 0; n_mad < actual_n; n_mad += MAD_N, s_epi = s_epi == kNumStages - 1 ? 0 : s_epi + 1) {
                    aic_wait_free(s_epi);
                    set_intra_block(PIPE_FIX, 2 - kNumAIVs + s_epi % kNumAIVs, s_epi / kNumAIVs);
                }
            }
        }
    }

    static __simd_vf__ inline void zero_row(__ubuf__ uint32_t* output) {
        constexpr uint32_t kNumValuesPerVector = asc_get_vf_len() / sizeof(uint32_t);
        uint32_t num_values = MAD_N * sizeof(cd_dtype_t) / sizeof(uint32_t);
        vector_uint32_t zero;
        asc_duplicate_scalar(zero, uint32_t(0));
        for (uint32_t value_idx = 0; value_idx < MAD_N * sizeof(cd_dtype_t) / sizeof(uint32_t); value_idx += kNumValuesPerVector)
            asc_storealign(output + value_idx, zero, asc_update_mask_b32(num_values));
    }

    __aicore__ void aiv_drain(gm_ptr<cd_dtype_t, Major::K> d, uint32_t m_idx, uint32_t n_idx,
                              uint32_t actual_m, uint32_t actual_n, uint32_t& s_epi, uint32_t batch_idx = 0,
                              uint32_t num_padding_rows = 0) const {
        static_assert(kPipe == PIPE_V, "Epilogue processing must use the vector pipe");
        const uint32_t aiv_id = asc_get_sub_block_id();
        const uint32_t drain_stage_idx = s_epi / kNumAIVs;
        // Prepare the whole drain before waiting for its first MAD tile.
        op.initialize_aiv_drain(args, drain_stage_idx, n_idx, actual_n);
        // Add one stage per N tile when padding is needed but the last MAD has no spare row.
        const uint32_t drain_m = actual_m + (kZeroPadding && num_padding_rows && actual_m % MAD_M == 0 ? MAD_M : 0);
        for (uint32_t m_mad = 0; m_mad < drain_m; m_mad += MAD_M) {
            for (uint32_t n_mad = 0; n_mad < actual_n; n_mad += MAD_N, s_epi = s_epi == kNumStages - 1 ? 0 : s_epi + 1) {
                if (aiv_id != 2 - kNumAIVs + s_epi % kNumAIVs) continue;
                const uint32_t stage_idx = s_epi / kNumAIVs;
                const auto store_event = static_cast<event_t>(stage_idx % kNumPipeEvents);
                const bool is_pure_padding_stage = kZeroPadding && m_mad >= actual_m;
                const bool need_padding = kZeroPadding && num_padding_rows && m_mad + MAD_M > actual_m;
                const epilogue::operators::MadBlock block{
                    .stage_idx = stage_idx,
                    .drain_stage_idx = drain_stage_idx,
                    .m_idx = m_idx + m_mad,
                    .n_idx = n_idx + n_mad,
                    .n_mad = n_mad,
                    .num_rows = is_pure_padding_stage ? 0 : min(actual_m - m_mad, MAD_M),
                    .num_cols = min(actual_n - n_mad, MAD_N),
                    .batch_idx = batch_idx,
                };
                wait_intra_block(kPipe, aiv_id, stage_idx);

                const auto ub_stage = ub[stage_idx];
                const ub_ptr<cd_dtype_t> output_ub(MAD_M, MAD_N, ub_stage.addr + kOutputOffsetBytes);

                if (!is_pure_padding_stage)
                    op.process_mad_block(ub_stage.ptr(), output_ub.ptr(), args, block);

                // Reuse the last unused output row after the normal SIMD epilogue.
                const ub_ptr<cd_dtype_t> zero_ub(1, MAD_N, output_ub.addr + (MAD_M - 1) * MAD_N * sizeof(cd_dtype_t));
                if (need_padding)
                    zero_row(zero_ub.template ptr<uint32_t>());

                asc_sync_notify(kPipe, PIPE_MTE3, store_event);
                asc_sync_wait(kPipe, PIPE_MTE3, store_event);
                if (!is_pure_padding_stage) {
                    copy_ub_to_gm(d.index(block.m_idx, block.n_idx).ptr(), output_ub, block.num_rows, block.num_cols,
                                  d.stride_outer * sizeof(cd_dtype_t), kL2CacheMode);
                    op.postprocess_mad_block(args, block);
                }
                if (need_padding) {
                    asc_copy_ub2gm_align(d.index(m_idx + actual_m, block.n_idx).ptr(), zero_ub.ptr(), num_padding_rows,
                                        block.num_cols * sizeof(cd_dtype_t), kL2CacheMode,
                                        d.stride_outer * sizeof(cd_dtype_t), 0);
                }
                set_intra_block(PIPE_MTE3, aiv_id, stage_idx);
            }
        }
    }

    __aicore__ void aiv_initialize() const {
        set_intra_blocks<kPipe, kNumStagesPerAIV>(asc_get_sub_block_id());
        if constexpr (kWithAccumulation) {
            if constexpr (std::is_same_v<cd_dtype_t, bfloat16_t>)
                asc_set_atomic_add_bfloat();
            else
                asc_set_atomic_add_float();
        }
    }

    static __simd_vf__ inline void zero_values(__ubuf__ uint32_t* output) {
        constexpr uint32_t kNumValuesPerVector = asc_get_vf_len() / sizeof(uint32_t);
        static_assert(MAD_M * MAD_N * sizeof(cd_dtype_t) % asc_get_vf_len() == 0);
        vector_uint32_t zero;
        asc_duplicate_scalar(zero, uint32_t(0));
        for (uint32_t value_idx = 0; value_idx < MAD_M * MAD_N * sizeof(cd_dtype_t) / sizeof(uint32_t); value_idx += kNumValuesPerVector)
            asc_storealign(output + value_idx, zero, asc_create_mask_b32(PAT_ALL));
    }

    template <uint32_t kNumCores, uint32_t kAlignment>
    __aicore__ void aiv_zero_empty_groups(gm_ptr<cd_dtype_t> d, uint32_t shape_m, uint32_t shape_n,
                                          __gm__ int32_t* grouped_layout, uint32_t num_groups) const {
        static_assert(!kWithAccumulation);
        const uint32_t aiv_idx = block_idx * kNumAIVs + asc_get_sub_block_id() - (2 - kNumAIVs);
        const uint32_t num_n_tiles = ceil_div(shape_n, MAD_N);
        const uint32_t num_tiles = ceil_div(shape_m, MAD_M) * num_n_tiles;
        const ub_ptr<cd_dtype_t> output_ub(MAD_M, MAD_N, ub.addr + kOutputOffsetBytes);
        bool initialized = false;
        uint32_t k_begin = 0;
        for (uint32_t group_idx = 0; group_idx < num_groups; ++group_idx) {
            const uint32_t k_end = static_cast<uint32_t>(grouped_layout[group_idx]);
            const bool is_empty = k_begin == k_end;
            k_begin = aligned(k_end, kAlignment);
            if (!is_empty || aiv_idx >= num_tiles) continue;
            if (!initialized) {
                zero_values(output_ub.template ptr<uint32_t>());
                op.initialize_empty_group();
                asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
                asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);
                initialized = true;
            }
            for (uint32_t tile_idx = aiv_idx; tile_idx < num_tiles; tile_idx += kNumCores * kNumAIVs) {
                const uint32_t m_idx = tile_idx / num_n_tiles * MAD_M;
                const uint32_t n_idx = tile_idx % num_n_tiles * MAD_N;
                const uint32_t num_rows = min(shape_m - m_idx, MAD_M);
                const uint32_t num_cols = min(shape_n - n_idx, MAD_N);
                copy_ub_to_gm(d.group(group_idx, shape_m, shape_n).index(m_idx, n_idx).ptr(), output_ub,
                              num_rows, num_cols, d.stride_outer * sizeof(cd_dtype_t), kL2CacheMode);
                op.postprocess_mad_block(args, {.stage_idx = 0, .m_idx = m_idx, .n_idx = n_idx,
                                                .num_rows = num_rows, .num_cols = num_cols, .batch_idx = group_idx});
            }
        }
        // Release the shared UB only after all zero stores have consumed it.
        if (initialized) {
            asc_sync_notify(PIPE_MTE3, PIPE_V, EVENT_ID0);
            asc_sync_wait(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
    }
};

} // namespace deep_gemm
