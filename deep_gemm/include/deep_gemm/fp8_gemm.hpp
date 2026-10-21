#pragma once

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/scheduler.hpp>
#include <deep_gemm/epilogue.hpp>

namespace deep_gemm {

// FP8 MX GEMM. Computes D = A @ B^T (logical NT shape), but A and B can each be stored
// in GM as either K-major (K contiguous) or MN-major (MN contiguous). After GM→L1 nd2nz
// the L1 tile is always NZ fractal format; the difference only matters for nd2nz axis
// orientation and whether the L1→L0 load needs a transpose.
// SF is always MN-major. D is always N-major.
//
// kMajorA / kMajorB give the GM major axis:
//   K  — GM [M/N, K] row-major, nd2nz to L1 physical [M/N, K], normal L1→L0
//   MN — GM [K, M/N] row-major, nd2nz to L1 physical [K, M/N], transpose L1→L0
template <
    Major kMajorA, Major kMajorB,
    uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t MAD_M, uint32_t MAD_N, uint32_t MAD_K,
    uint32_t kNumL1Stages, uint32_t kNumL0Stages,
    uint32_t kSFKBlocks, uint32_t kNumL1SFStages,
    uint32_t kNumEpilogueStages,
    uint32_t kNumCores,
    asc_load_l2_cache_mode kL2CtrlA, asc_load_l2_cache_mode kL2CtrlB, asc_store_l2_cache_mode kL2CtrlStoreCd,
    bool kWithAccumulation,
    GemmType kGemmType,
    typename ab_dtype_t,
    typename cd_dtype_t,
    template <typename> class epilogue_operator_t,
    uint32_t kAlignment,
    bool kDirectStore = false, bool kSingleMBlock = false
>
__global__ __mix__(1, kDirectStore ? 0 : 2) void fp8_gemm_impl(
    gm_ptr<ab_dtype_t, kMajorA> gm_a,
    gm_ptr<ab_dtype_t, kMajorB> gm_b,
    __gm__ int16_t* gmem_sfa,
    __gm__ int16_t* gmem_sfb,
    gm_ptr<cd_dtype_t, Major::K> gm_d,
    uint32_t shape_m_runtime, uint32_t shape_n_runtime, uint32_t shape_k_runtime,
    __gm__ int32_t* grouped_layout, uint32_t num_groups,
    epilogue::Args epilogue_args
) {
    asc_init();
    static_assert(!kSingleMBlock || kGemmType == GemmType::Normal);

    if constexpr (kDirectStore) {
        static_assert(!kWithAccumulation || std::is_same_v<cd_dtype_t, float>);
    }

    static_assert(kMajorA == Major::K || kMajorA == Major::MN, "MajorA must be K or MN");
    static_assert(kMajorB == Major::K || kMajorB == Major::MN, "MajorB must be K or MN");
    static_assert(kMajorA == Major::K || MAD_M % 32 == 0, "MN-major A uses L1->L0 transpose and requires MAD_M % 32 == 0");
    static_assert(kMajorB == Major::K || MAD_N % 32 == 0, "MN-major B uses L1->L0 transpose and requires MAD_N % 32 == 0");
    static_assert(!is_fp4<ab_dtype_t>() || kMajorA == Major::K || MAD_M >= 64, "fp4 MN-major A requires MAD_M >= 64");
    static_assert(!is_fp4<ab_dtype_t>() || kMajorB == Major::K || MAD_N >= 64, "fp4 MN-major B requires MAD_N >= 64");

    uint32_t shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m_runtime;
    uint32_t shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n_runtime;
    uint32_t shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k_runtime;
    const auto shape_sf_k = ceil_div(shape_k, MX_SF_DIVISOR);

    const gm_ptr<int16_t, Major::MN> gm_sfa(shape_m, reinterpret_cast<uintptr_t>(gmem_sfa), static_cast<uint64_t>(shape_m) * shape_sf_k);
    const gm_ptr<int16_t, Major::MN> gm_sfb(shape_n, reinterpret_cast<uintptr_t>(gmem_sfb), static_cast<uint64_t>(shape_n) * shape_sf_k);

    constexpr uint16_t kSFPairsPerKBlock = BLOCK_K / MX_SF_DIVISOR;
    constexpr uint32_t kSFPairsPerChunk = kSFPairsPerKBlock * kSFKBlocks;
    constexpr uint32_t kFracK = get_frac_k<ab_dtype_t>();

    constexpr uint32_t kNumCDStages = (BLOCK_M / MAD_M) * (BLOCK_N / MAD_N);

    const Epilogue<cd_dtype_t, epilogue_operator_t, kWithAccumulation,
                   BLOCK_M, BLOCK_N, MAD_M, MAD_N, kNumEpilogueStages, 2,
                   kL2CtrlStoreCd> epilogue{epilogue_args};
    static_assert(!kDirectStore || std::is_same_v<typename decltype(epilogue)::operator_t,
                                                  epilogue::operators::Identity<typename decltype(epilogue)::params_t>>);

    constexpr l1_ptr<ab_dtype_t, kMajorA> l1a(BLOCK_M, BLOCK_K, 0);
    constexpr l1_ptr<ab_dtype_t, kMajorB> l1b(BLOCK_N, BLOCK_K, l1a.offset(kNumL1Stages));
    constexpr l1_ptr<int16_t> l1_sfa(BLOCK_M, kSFPairsPerChunk, l1b.offset(kNumL1Stages));
    constexpr l1_ptr<int16_t> l1_sfb(BLOCK_N, kSFPairsPerChunk, l1_sfa.offset(kNumL1SFStages));

    constexpr l0a_ptr<ab_dtype_t> l0a(MAD_M, MAD_K);
    constexpr l0b_ptr<ab_dtype_t> l0b(MAD_N, MAD_K);
    constexpr l0c_ptr<float> l0c(MAD_M, MAD_N);

    static_assert(l1_sfb.offset(kNumL1SFStages) <= L1SizeBytes, "L1 overflow");
    static_assert(l0a.offset(kNumL0Stages) <= L0ASizeBytes, "L0A overflow");
    static_assert(l0b.offset(kNumL0Stages) <= L0BSizeBytes, "L0B overflow");
    static_assert(l0c.offset(kNumCDStages) <= L0CSizeBytes, "L0C overflow");
    static_assert(BLOCK_K % MAD_K == 0, "BLOCK_K must be a multiple of MAD_K");
    static_assert(MAD_K % 64 == 0, "ascend forces MAD_K to be a multiple of 64");

    uint32_t m_block_idx, n_block_idx;
    Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumCores, kAlignment> scheduler(block_idx, kSingleMBlock ? BLOCK_M : shape_m, shape_n, grouped_layout,
                                                                            num_groups);

    constexpr bool is_m_grouped = kGemmType == GemmType::MGroupedContiguousWithPsumLayout;
    constexpr bool is_k_grouped = kGemmType == GemmType::KGroupedContiguousWithPsumLayout;
    constexpr bool is_batched = kGemmType == GemmType::Batched;

    if ASCEND_IS_AIC {
        if constexpr (kDirectStore && kWithAccumulation) asc_set_atomic_add_float();
        asc_set_mmad_direction_n();

        constexpr uint32_t kL1EventBase = 0;
        constexpr uint32_t kSFEventBase = kNumL1Stages;

        set_flags<PIPE_M, PIPE_MTE1, kNumL0Stages>();
        set_flags<PIPE_MTE1, PIPE_MTE2, kNumL1Stages, kL1EventBase>();
        set_flags<PIPE_MTE1, PIPE_MTE2, kNumL1SFStages, kSFEventBase>();

        uint32_t l1_event_idx = 0;
        uint32_t l1_sf_event_idx = 0;
        uint32_t l1a_base_stage_idx = 0, l1a_stage_idx = 0;
        uint32_t l1b_base_stage_idx = 0, l1b_stage_idx = 0;
        uint32_t l1_sfa_base_stage_idx = 0, l1_sfa_stage_idx = 0;
        uint32_t l1_sfb_base_stage_idx = 0, l1_sfb_stage_idx = 0;
        uint32_t epi_base_idx = 0;
        uintptr_t last_a = 0, last_b = 0, last_sfa = 0, last_sfb = 0;

        const bool can_reuse_a = shape_m <= BLOCK_M && ceil_div(shape_n, BLOCK_N) > kNumCores;
        const bool can_reuse_b = shape_n <= BLOCK_N && ceil_div(shape_m, BLOCK_M) > kNumCores;

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const uint32_t m_idx = scheduler.get_global_idx(BLOCK_M, m_block_idx);
            const uint32_t n_idx = scheduler.get_global_idx(BLOCK_N, n_block_idx);
            const uint32_t eff_shape_k = is_k_grouped ? scheduler.current_shape_k : shape_k;
            const uint32_t num_k_blocks = ceil_div(eff_shape_k, BLOCK_K);
            const uint32_t k_idx_base = scheduler.get_k_idx_base();
            const uint32_t sf_idx_base = scheduler.get_sf_idx_base();

            const auto batch_idx = is_batched ? scheduler.get_batch_idx() : 0;
            const auto b_group_idx = is_m_grouped ? scheduler.get_group_idx() : 0;
            const auto a = gm_a.batch(batch_idx).index(m_idx, k_idx_base);
            const auto b = gm_b.batch(batch_idx).group(b_group_idx, shape_n, shape_k).index(n_idx, k_idx_base);
            const auto sfa = gm_sfa.batch(batch_idx).index(m_idx, sf_idx_base);
            const auto sfb = gm_sfb.batch(batch_idx).group(b_group_idx, shape_n, shape_sf_k).index(n_idx, sf_idx_base);
            const bool can_reuse = num_k_blocks <= kNumL1Stages;
            const bool can_reuse_sf = can_reuse && num_k_blocks <= kSFKBlocks;
            const bool reuse_a = can_reuse_a && can_reuse && a.addr == last_a;
            const bool reuse_b = can_reuse_b && can_reuse && b.addr == last_b;
            const bool reuse_sfa = can_reuse_a && can_reuse_sf && sfa.addr == last_sfa;
            const bool reuse_sfb = can_reuse_b && can_reuse_sf && sfb.addr == last_sfb;
            // Rewind reused operands; new loads follow the previous task's last read.
            l1a_stage_idx = l1a_base_stage_idx = reuse_a ? l1a_base_stage_idx : l1a_stage_idx;
            l1b_stage_idx = l1b_base_stage_idx = reuse_b ? l1b_base_stage_idx : l1b_stage_idx;
            l1_sfa_stage_idx = l1_sfa_base_stage_idx = reuse_sfa ? l1_sfa_base_stage_idx : l1_sfa_stage_idx;
            l1_sfb_stage_idx = l1_sfb_base_stage_idx = reuse_sfb ? l1_sfb_base_stage_idx : l1_sfb_stage_idx;

            const uint32_t actual_m = kSingleMBlock ? shape_m : scheduler.get_actual_m(m_block_idx);
            const uint32_t actual_n = scheduler.get_actual_n(n_block_idx);
            uint32_t epi_stage_idx = epi_base_idx;

            for (uint32_t k_block_idx = 0, k_within_chunk = 0; k_block_idx < num_k_blocks; k_block_idx++) {
                const uint32_t k_idx = k_block_idx * BLOCK_K;
                const uint32_t actual_k = k_block_idx < eff_shape_k / BLOCK_K ? BLOCK_K : eff_shape_k % BLOCK_K;
                const uint32_t k64 = aligned(actual_k, MX_SF_DIVISOR);
                const uint32_t kc0 = aligned(actual_k, kFracK);

                // Data load: GM → L1 (helper picks ND vs DN path from the tile's compile-time layout)
                asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(kL1EventBase + l1_event_idx));
                if (!reuse_a) copy_gm_to_l1(l1a[l1a_stage_idx], a.index(0, k_idx), actual_m, actual_k, kL2CtrlA);
                if (!reuse_b) copy_gm_to_l1(l1b[l1b_stage_idx], b.index(0, k_idx), actual_n, actual_k, kL2CtrlB);
                const bool fill_a = (kMajorA == Major::K ? kc0 : actual_k) < k64;
                const bool fill_b = (kMajorB == Major::K ? kc0 : actual_k) < k64;
                if (fill_a && !reuse_a) fill_tail_block(l1a[l1a_stage_idx], actual_k);
                if (fill_b && !reuse_b) fill_tail_block(l1b[l1b_stage_idx], actual_k);
                asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(kL1EventBase + l1_event_idx));

                if (k_within_chunk == 0) {
                    uint32_t chunk_k_blocks = min(num_k_blocks - k_block_idx, kSFKBlocks);
                    uint32_t chunk_k_end = min(k_idx + chunk_k_blocks * BLOCK_K, eff_shape_k);
                    uint32_t chunk_sf_pairs = ceil_div(chunk_k_end - k_idx, MX_SF_DIVISOR);
                    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(kSFEventBase + l1_sf_event_idx));
                    if (!reuse_sfa) copy_gm_to_l1_mx(
                        l1_sfa[l1_sfa_stage_idx], sfa.index(0, k_block_idx * kSFPairsPerKBlock),
                        actual_m, chunk_sf_pairs, kL2CtrlA
                    );
                    if (!reuse_sfb) copy_gm_to_l1_mx(
                        l1_sfb[l1_sfb_stage_idx], sfb.index(0, k_block_idx * kSFPairsPerKBlock),
                        actual_n, chunk_sf_pairs, kL2CtrlB
                    );
                    asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(kSFEventBase + l1_sf_event_idx));
                }

                // Wait AB; SF is waited at its first L0 copy.
                asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(kL1EventBase + l1_event_idx));

                epi_stage_idx = epi_base_idx;
                for (uint32_t m_mad_idx = 0, s_cd = 0; m_mad_idx < actual_m; m_mad_idx += MAD_M) {
                    for (uint32_t n_mad_idx = 0; n_mad_idx < actual_n; n_mad_idx += MAD_N, s_cd = s_cd == kNumCDStages - 1 ? 0 : s_cd + 1) {
                        const auto eff_mad_m = min(actual_m - m_mad_idx, MAD_M);
                        const auto eff_mad_n = min(actual_n - n_mad_idx, MAD_N);
                        // copy_l1_to_l0x transpose requires the m/n to aligned with frac_k
                        const auto mad_m = kMajorA == Major::MN ? aligned(eff_mad_m, kFracK) : eff_mad_m;
                        const auto mad_n = aligned(eff_mad_n, kMajorB == Major::MN ? kFracK : get_frac_k<typename decltype(epilogue)::ub_dtype_t>());
                        auto l0c_tile = l0c[s_cd].as_mad_aligned(mad_m, mad_n);

                        if (k_block_idx == 0 && !kDirectStore)
                            epilogue.aic_wait_free(epi_stage_idx);

                        for (uint32_t k_mad_idx = 0, s_l0_ab = 0; k_mad_idx < k64; k_mad_idx += MAD_K, s_l0_ab = s_l0_ab == kNumL0Stages - 1 ? 0 : s_l0_ab + 1) {
                            const auto is_first = k_block_idx == 0 && k_mad_idx == 0;
                            const auto is_last = k_block_idx == num_k_blocks - 1 && k_mad_idx + MAD_K >= k64;
                            const auto eff_mad_k = min(MAD_K, k64 - k_mad_idx);
                            const auto sf_y = static_cast<uint16_t>(k_within_chunk * kSFPairsPerKBlock + k_mad_idx / MX_SF_DIVISOR);
                            auto l0a_tile = l0a[s_l0_ab].as_mad_aligned(mad_m, eff_mad_k);
                            auto l0b_tile = l0b[s_l0_ab].as_mad_aligned(mad_n, eff_mad_k);

                            asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(s_l0_ab));
                            if (BLOCK_N == MAD_N || BLOCK_K > MAD_K * kNumL0Stages || n_mad_idx == 0)
                                copy_l1_to_l0a(l0a_tile, l1a[l1a_stage_idx], m_mad_idx, k_mad_idx);
                            if (BLOCK_M == MAD_M || BLOCK_N != MAD_N || BLOCK_K > MAD_K * kNumL0Stages || m_mad_idx == 0)
                                copy_l1_to_l0b(l0b_tile, l1b[l1b_stage_idx], n_mad_idx, k_mad_idx);
                            // AB can enter L0 while the first SF request is still in flight.
                            if (k_within_chunk == 0 && m_mad_idx == 0 && n_mad_idx == 0 && k_mad_idx == 0)
                                asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(kSFEventBase + l1_sf_event_idx));
                            if (BLOCK_N == MAD_N || BLOCK_K > MAD_K * kNumL0Stages || n_mad_idx == 0)
                                copy_l1_to_l0a_mx(l0a_tile, l1_sfa[l1_sfa_stage_idx], m_mad_idx, sf_y, eff_mad_k);
                            if (BLOCK_M == MAD_M || BLOCK_N != MAD_N || BLOCK_K > MAD_K * kNumL0Stages || m_mad_idx == 0)
                                copy_l1_to_l0b_mx(l0b_tile, l1_sfb[l1_sfb_stage_idx], n_mad_idx, sf_y, eff_mad_k);
                            asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(s_l0_ab));

                            asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(s_l0_ab));
                            mad(l0c_tile, l0a_tile, l0b_tile, mad_m, mad_n, eff_mad_k,
                                is_last ? asc_unit_flag_mode::ENABLE_UPDATE : asc_unit_flag_mode::ENABLE_KEEP,
                                is_first);
                            asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(s_l0_ab));
                        }

                        if (k_block_idx == num_k_blocks - 1) {
                            if constexpr (kDirectStore) {
                                const auto d = gm_d.batch(batch_idx).group(is_k_grouped ? scheduler.get_group_idx() : 0, shape_m, shape_n);
                                copy_l0c_to_gm(d.index(m_idx + m_mad_idx, n_idx + n_mad_idx).ptr(),
                                               l0c_tile, mad_m, mad_n, d.stride_outer, kL2CtrlStoreCd);
                            } else {
                                epilogue.aic_store(l0c_tile, epi_stage_idx, mad_m, mad_n);
                            }
                        }
                        epi_stage_idx = epi_stage_idx == epilogue.kNumStages - 1 ? 0 : epi_stage_idx + 1;
                    }
                }

                asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(kL1EventBase + l1_event_idx));
                l1_event_idx = l1_event_idx == kNumL1Stages - 1 ? 0 : l1_event_idx + 1;
                l1a_stage_idx = !can_reuse_a ? l1_event_idx : l1a_stage_idx == kNumL1Stages - 1 ? 0 : l1a_stage_idx + 1;
                l1b_stage_idx = !can_reuse_b ? l1_event_idx : l1b_stage_idx == kNumL1Stages - 1 ? 0 : l1b_stage_idx + 1;

                k_within_chunk++;
                if (k_within_chunk >= kSFKBlocks || k_block_idx == num_k_blocks - 1) {
                    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(kSFEventBase + l1_sf_event_idx));
                    l1_sf_event_idx = l1_sf_event_idx == kNumL1SFStages - 1 ? 0 : l1_sf_event_idx + 1;
                    l1_sfa_stage_idx = !can_reuse_a ? l1_sf_event_idx : l1_sfa_stage_idx == kNumL1SFStages - 1 ? 0 : l1_sfa_stage_idx + 1;
                    l1_sfb_stage_idx = !can_reuse_b ? l1_sf_event_idx : l1_sfb_stage_idx == kNumL1SFStages - 1 ? 0 : l1_sfb_stage_idx + 1;
                    k_within_chunk = 0;
                }
            }
            epi_base_idx = epi_stage_idx;
            last_a = a.addr;
            last_b = b.addr;
            last_sfa = sfa.addr;
            last_sfb = sfb.addr;
        }
    } else {
        if constexpr (kDirectStore) return;
        if constexpr (is_k_grouped && !kWithAccumulation)
            epilogue.template aiv_zero_empty_groups<kNumCores, kAlignment>(gm_d, shape_m, shape_n, grouped_layout, num_groups);
        if constexpr (is_m_grouped && BLOCK_M < kAlignment)
            epilogue.template aiv_zero_group_padding<kNumCores, kAlignment>(gm_d, shape_n, grouped_layout, num_groups);
        epilogue.aiv_initialize();

        uint32_t s_epi = 0;
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const uint32_t m_idx = scheduler.get_global_idx(BLOCK_M, m_block_idx);
            const uint32_t n_idx = scheduler.get_global_idx(BLOCK_N, n_block_idx);
            const uint32_t actual_m = kSingleMBlock ? shape_m : scheduler.get_actual_m(m_block_idx);
            const uint32_t actual_n = scheduler.get_actual_n(n_block_idx);

            const auto batch_idx = is_batched ? scheduler.get_batch_idx() : 0;
            const auto d_group_idx = is_k_grouped ? scheduler.get_group_idx() : 0;
            const auto d = gm_d.batch(batch_idx).group(d_group_idx, shape_m, shape_n);

            epilogue.aiv_drain(d, m_idx, n_idx, actual_m, actual_n, s_epi, batch_idx);
        }
    }
}

} // namespace deep_gemm
