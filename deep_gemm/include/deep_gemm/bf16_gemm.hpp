#pragma once

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/scheduler.hpp>
#include <deep_gemm/epilogue.hpp>

namespace deep_gemm {

template <
    Major kMajorA, Major kMajorB,
    uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t MAD_M, uint32_t MAD_N, uint32_t MAD_K,
    uint32_t kNumL1Stages, uint32_t kNumL0Stages,
    uint32_t kNumEpilogueStages,
    uint32_t kNumCores,
    asc_load_l2_cache_mode kL2CtrlA,
    asc_load_l2_cache_mode kL2CtrlB,
    asc_store_l2_cache_mode kL2CtrlStoreCd,
    bool kWithAccumulation,
    GemmType kGemmType,
    typename cd_dtype_t,
    template <typename> class epilogue_operator_t,
    uint32_t kAlignment,
    bool kDirectStore = false, bool kResidentA = false, bool kResidentB = false
>
__global__ __mix__(1, kDirectStore ? 0 : 2) void bf16_gemm_impl(
    gm_ptr<bfloat16_t, kMajorA> gm_a,
    gm_ptr<bfloat16_t, kMajorB> gm_b,
    gm_ptr<cd_dtype_t> gm_d,
    uint32_t shape_m_runtime, uint32_t shape_n_runtime, uint32_t shape_k_runtime,
    __gm__ int32_t* grouped_layout, uint32_t num_groups,
    epilogue::Args epilogue_args
) {
    asc_init();

    if constexpr (kDirectStore) {
        static_assert(!kWithAccumulation || std::is_same_v<cd_dtype_t, float>);
    }
    static_assert(!(kResidentA || kResidentB) || (kGemmType == GemmType::Normal && SHAPE_K));

    static_assert(kMajorA == Major::K || kMajorA == Major::MN, "MajorA must be K or MN");
    static_assert(kMajorB == Major::K || kMajorB == Major::MN, "MajorB must be K or MN");

    static_assert(!kResidentB || (SHAPE_K <= kNumL0Stages * MAD_K && BLOCK_N == MAD_N));

    uint32_t shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m_runtime;
    uint32_t shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n_runtime;
    uint32_t shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k_runtime;

    constexpr uint32_t num_mad_cd_stages = (BLOCK_M / MAD_M) * (BLOCK_N / MAD_N);

    const Epilogue<cd_dtype_t, epilogue_operator_t, kWithAccumulation,
                   BLOCK_M, BLOCK_N, MAD_M, MAD_N, kNumEpilogueStages, 2,
                   kL2CtrlStoreCd, kGemmType == GemmType::MGroupedContiguousWithPsumLayout> epilogue{epilogue_args};
    static_assert(!kDirectStore || std::is_same_v<typename decltype(epilogue)::operator_t,
                                                  epilogue::operators::Identity<typename decltype(epilogue)::params_t>>);

    constexpr l1_ptr<bfloat16_t, kMajorA> l1a(BLOCK_M, kResidentA ? aligned(SHAPE_K, get_frac_k<bfloat16_t>()) : BLOCK_K, 0);
    constexpr l1_ptr<bfloat16_t, kMajorB> l1b(BLOCK_N, BLOCK_K, l1a.offset(kResidentA ? 1 : kNumL1Stages));

    constexpr l0a_ptr<bfloat16_t> l0a(MAD_M, MAD_K);
    constexpr l0b_ptr<bfloat16_t> l0b(MAD_N, MAD_K);
    constexpr l0c_ptr<float> l0c(MAD_M, MAD_N);

    static_assert(l1b.offset(kNumL1Stages) <= L1SizeBytes,  "L1 overflow");
    static_assert(l0a.offset(kNumL0Stages) <= L0ASizeBytes, "L0A overflow");
    static_assert(l0b.offset(kNumL0Stages) <= L0BSizeBytes, "L0B overflow");
    static_assert(l0c.offset(num_mad_cd_stages) <= L0CSizeBytes, "L0C overflow");
    static_assert(BLOCK_K % MAD_K == 0, "BLOCK_K must be a multiple of MAD_K");

    uint32_t m_block_idx, n_block_idx;
    Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumCores, kAlignment> scheduler(block_idx, shape_m, shape_n,
                                                                            grouped_layout, num_groups);

    constexpr bool is_m_grouped = kGemmType == GemmType::MGroupedContiguousWithPsumLayout;
    constexpr bool is_k_grouped = kGemmType == GemmType::KGroupedContiguousWithPsumLayout;
    constexpr bool is_batched = kGemmType == GemmType::Batched;
    constexpr bool is_grouped = is_m_grouped || is_k_grouped;

    if ASCEND_IS_AIC {
        if constexpr (kDirectStore && kWithAccumulation) asc_set_atomic_add_float();
        set_flags<PIPE_MTE1, PIPE_MTE2, kNumL1Stages>();
        set_flags<PIPE_M, PIPE_MTE1, kNumL0Stages>();

        uint32_t epi_base = 0;
        uint32_t last_m_idx = shape_m, last_n_idx = shape_n;

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const uint32_t m_idx = scheduler.get_global_m_idx(m_block_idx);
            const uint32_t n_idx = scheduler.get_global_n_idx(n_block_idx);
            const bool reuse_a = m_idx == last_m_idx;
            const bool reuse_b = n_idx == last_n_idx;

            const uint32_t eff_shape_k = is_k_grouped ? scheduler.current_shape_k : shape_k;
            const uint32_t num_k_blocks = ceil_div(eff_shape_k, BLOCK_K);
            const uint32_t k_idx_base = scheduler.get_k_idx_base();

            const uint32_t batch_idx = is_batched ? scheduler.get_batch_idx() : 0;
            const uint32_t b_group_idx = is_m_grouped ? scheduler.get_group_idx() : 0;

            const auto a = gm_a.batch(batch_idx);
            const auto b = gm_b.batch(batch_idx).group(b_group_idx, shape_n, shape_k);

            const uint32_t actual_m = scheduler.get_actual_m(m_block_idx);
            const uint32_t actual_n = scheduler.get_actual_n(n_block_idx);
            uint32_t s_epi = epi_base;

            if (kResidentA && !reuse_a)
                copy_gm_to_l1(l1a[0], a.index(m_idx, 0), actual_m, shape_k, kL2CtrlA);

            for (uint32_t k_blk_idx = 0, s = 0; k_blk_idx < num_k_blocks; k_blk_idx++, s = s == kNumL1Stages - 1 ? 0 : s + 1) {
                const uint32_t k_idx = k_idx_base + k_blk_idx * BLOCK_K;
                const uint32_t actual_k = min(eff_shape_k - k_blk_idx * BLOCK_K, BLOCK_K);

                asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(s));
                if constexpr (!kResidentA) copy_gm_to_l1(l1a[s], a.index(m_idx, k_idx), actual_m, actual_k, kL2CtrlA);
                if (!kResidentB || !reuse_b) copy_gm_to_l1(l1b[s], b.index(n_idx, k_idx), actual_n, actual_k, kL2CtrlB);
                asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(s));

                asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(s));
                s_epi = epi_base;
                for (uint32_t m_mad_idx = 0, s_cd = 0; m_mad_idx < actual_m; m_mad_idx += MAD_M) {
                    for (uint32_t n_mad_idx = 0; n_mad_idx < actual_n; n_mad_idx += MAD_N, s_cd = s_cd == num_mad_cd_stages - 1 ? 0 : s_cd + 1) {
                        uint32_t eff_mad_m = min(actual_m - m_mad_idx, MAD_M);
                        uint32_t eff_mad_n = aligned(min(actual_n - n_mad_idx, MAD_N), get_frac_k<typename decltype(epilogue)::ub_dtype_t>());
                        auto l0c_tile = l0c[s_cd].as_mad_aligned(eff_mad_m, eff_mad_n);

                        if (k_blk_idx == 0 && !kDirectStore) {
                            epilogue.aic_wait_free(s_epi);
                        }

                        for (uint32_t k_mad_idx = 0, s_ab = 0; k_mad_idx < actual_k; k_mad_idx += MAD_K, s_ab = s_ab == kNumL0Stages - 1 ? 0 : s_ab + 1) {
                            bool is_first = k_blk_idx == 0 && k_mad_idx == 0;
                            bool is_last = k_blk_idx == num_k_blocks - 1 && k_mad_idx + MAD_K >= actual_k;

                            uint32_t eff_mad_k = min(actual_k - k_mad_idx, MAD_K);
                            auto l0a_tile = l0a[s_ab].as_mad_aligned(eff_mad_m, eff_mad_k);
                            auto l0b_tile = l0b[s_ab].as_mad_aligned(eff_mad_n, eff_mad_k);

                            asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(s_ab));
                            if (BLOCK_N == MAD_N || BLOCK_K > MAD_K * kNumL0Stages || n_mad_idx == 0)
                                copy_l1_to_l0a(l0a_tile, l1a[kResidentA ? 0 : s], m_mad_idx, (kResidentA ? k_blk_idx * BLOCK_K : 0) + k_mad_idx);
                            if ((!kResidentB || !reuse_b) && (BLOCK_M == MAD_M || BLOCK_N != MAD_N || BLOCK_K > MAD_K * kNumL0Stages || m_mad_idx == 0))
                                copy_l1_to_l0b(l0b_tile, l1b[s], n_mad_idx, k_mad_idx);
                            asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(s_ab));

                            asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(s_ab));
                            mad(l0c_tile, l0a_tile, l0b_tile, eff_mad_m, eff_mad_n, eff_mad_k,
                                is_last ? asc_unit_flag_mode::ENABLE_UPDATE : asc_unit_flag_mode::ENABLE_KEEP,
                                is_first);
                            asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(s_ab));
                        }

                        if (k_blk_idx == num_k_blocks - 1) {
                            if constexpr (kDirectStore) {
                                const auto d = gm_d.batch(batch_idx).group(is_k_grouped ? scheduler.get_group_idx() : 0, shape_m, shape_n);
                                copy_l0c_to_gm(d.index(m_idx + m_mad_idx, n_idx + n_mad_idx).ptr(),
                                               l0c_tile, eff_mad_m, eff_mad_n, d.stride_outer, kL2CtrlStoreCd);
                            } else {
                                epilogue.aic_store(l0c_tile, s_epi, eff_mad_m, eff_mad_n);
                            }
                        }
                        s_epi = s_epi == epilogue.kNumStages - 1 ? 0 : s_epi + 1;
                    }
                }
                asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(s));
            }
            if constexpr (is_m_grouped) epilogue.aic_insert_padding_stages(actual_m, actual_n, scheduler.get_num_padding_rows(m_block_idx), s_epi);
            epi_base = s_epi;
            last_m_idx = m_idx;
            last_n_idx = n_idx;
        }
    } else {
        if constexpr (kDirectStore) return;
        if constexpr (is_k_grouped && !kWithAccumulation)
            epilogue.template aiv_zero_empty_groups<kNumCores, kAlignment>(gm_d, shape_m, shape_n, grouped_layout, num_groups);
        epilogue.aiv_initialize();

        uint32_t s_epi = 0;
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const uint32_t m_idx = scheduler.get_global_m_idx(m_block_idx);
            const uint32_t n_idx = scheduler.get_global_n_idx(n_block_idx);
            const uint32_t actual_m = scheduler.get_actual_m(m_block_idx);
            const uint32_t actual_n = scheduler.get_actual_n(n_block_idx);
            const uint32_t batch_idx = is_batched ? scheduler.get_batch_idx() : 0;
            const uint32_t d_group_idx = is_k_grouped ? scheduler.get_group_idx() : 0;

            const auto d = gm_d.batch(batch_idx).group(d_group_idx, shape_m, shape_n);

            epilogue.aiv_drain(d, m_idx, n_idx, actual_m, actual_n, s_epi, batch_idx, scheduler.get_num_padding_rows(m_block_idx));
        }
    }
}

} // namespace deep_gemm
