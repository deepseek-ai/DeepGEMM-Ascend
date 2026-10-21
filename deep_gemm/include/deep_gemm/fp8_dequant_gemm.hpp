#pragma once

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/scheduler.hpp>
#include <deep_gemm/epilogue.hpp>
#include <c_api/asc_simd.h>

namespace deep_gemm {

constexpr uint8_t kDequantAIVId = 0;

__simd_vf__ inline void vf_build_lut(__ubuf__ uint8_t* lut) {
    // Each packed FP4 byte holds two E2M1 values. Dequantization splits the low and high
    // nibbles and uses each as an index into this LUT, producing two raw E4M3 bytes.
    const auto mask = asc_create_mask_b8(PAT_ALL);
    vector_int8_t arange;
    vector_uint8_t values, mag_mask, sign_mask, zero, half, mag, sign;
    asc_arange(arange, static_cast<int8_t>(0));
    asc_duplicate_scalar(mag_mask, static_cast<uint8_t>(0x07));
    asc_duplicate_scalar(sign_mask, static_cast<uint8_t>(0x08));
    asc_duplicate_scalar(zero, static_cast<uint8_t>(0x00));
    asc_duplicate_scalar(half, static_cast<uint8_t>(0x30));

    // E2M1 code 0 maps to E4M3 0x00, code 1 to 0x30, and codes 2..7 to
    // 0x30 + (code << 2). FP4 sign bit 3 is moved to E4M3 sign bit 7 below.
    asc_and(mag, reinterpret_cast<vector_uint8_t&>(arange), mag_mask, mask);
    asc_shiftleft_scalar(values, mag, 2, mask);
    asc_add_scalar(values, values, static_cast<uint8_t>(0x30), mask);

    vector_bool is_zero, is_one;
    asc_eq_scalar(is_zero, mag, static_cast<uint8_t>(0), mask);
    asc_eq_scalar(is_one, mag, static_cast<uint8_t>(1), mask);
    asc_select(values, zero, values, is_zero);
    asc_select(values, half, values, is_one);

    asc_and(sign, reinterpret_cast<vector_uint8_t&>(arange), sign_mask, mask);
    asc_shiftleft_scalar(sign, sign, 4, mask);
    asc_or(values, values, sign, mask);
    asc_storealign(lut, values, mask);
}

template <int num_rows, int col_bytes, int nz_stride_32b>
__simd_vf__ inline void vf_deq_nd2nz(__ubuf__ uint8_t* src, __ubuf__ uint8_t* lut, __ubuf__ uint8_t* dst) {
    static_assert(col_bytes == 128 || col_bytes == 256, "col_bytes must be 128 or 256");
    static_assert(num_rows * col_bytes % 256 == 0, "num_rows * col_bytes must be a multiple of 256");
    // src[r][c] packs FP4 columns 2*c and 2*c+1 in its low and high nibbles.
    // dst is FP8 NZ [column group][padded row][32 FP8 values]; adjacent group bases are
    // nz_stride_32b * 32 bytes apart.

    vector_bool mask = asc_create_mask_b8(PAT_ALL);
    vector_uint8_t table;
    asc_loadalign(table, lut);
    __ubuf__ uint8_t* sp = src;
    __ubuf__ uint8_t* dp0;
    __ubuf__ uint8_t* dp1;

    if constexpr (col_bytes == 128) {
        dp0 = dst;
        dp1 = dst + 32;
    }
    if constexpr (col_bytes == 256) {
        dp0 = dst;
        dp1 = dst + nz_stride_32b * col_bytes;
    }

    for (int r = 0; r < num_rows;) {
        vector_uint8_t bytes, high, low_fp8, high_fp8, out0, out1;
        asc_loadalign_postupdate(bytes, sp, 256);
        // A 256B load holds 512 FP4 values. Low nibbles are even columns; high nibbles are odd columns.
        asc_shiftright_scalar(high, bytes, 4, mask);
        // Register LUT lookup converts each FP4 code to one E4M3 byte; interleave restores
        // [even0, odd0, even1, odd1, ...].
        // The 256-byte LUT repeats every 16 entries, so the low nibble needs no mask.
        asc_gather(low_fp8, table, bytes);
        asc_gather(high_fp8, table, high);
        asc_intlv(out0, out1, low_fp8, high_fp8);

        // With 128 packed bytes per row, out0/out1 are two complete 256B FP8 rows.
        // With 256 packed bytes per row, they are the first and second 256B halves of one 512B FP8 row.
        // Each output register holds eight 32B column groups; stores place adjacent groups
        // nz_stride_32b blocks apart and advance dp0/dp1 by 2 or 1 row blocks.
        if constexpr (col_bytes == 128) {
            asc_storealign_postupdate(dp0, out0, nz_stride_32b, 2, mask);
            asc_storealign_postupdate(dp1, out1, nz_stride_32b, 2, mask);
            r += 2;
        }
        if constexpr (col_bytes == 256) {
            asc_storealign_postupdate(dp0, out0, nz_stride_32b, 1, mask);
            asc_storealign_postupdate(dp1, out1, nz_stride_32b, 1, mask);
            r += 1;
        }
    }
}

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
    typename cd_dtype_t,
    template <typename> class epilogue_operator_t,
    uint32_t kAlignment,
    bool kDirectStore = false, bool kDualAIVDequant = false,
    bool kSingleMBlock = false
>
__global__ __mix__(1, 2) void fp8_dequant_gemm_impl(
    gm_ptr<float8_e4m3_t, kMajorA> gm_a,
    gm_ptr<float4_e2m1x2_t, kMajorB> gm_b,
    __gm__ int16_t* gmem_sfa,
    __gm__ int16_t* gmem_sfb,
    gm_ptr<cd_dtype_t, Major::K> gm_d,
    uint32_t shape_m_runtime, uint32_t shape_n_runtime, uint32_t shape_k_runtime,
    __gm__ int32_t* grouped_layout, uint32_t num_groups,
    epilogue::Args epilogue_args
) {
    asc_init();
    static_assert(!kSingleMBlock || (kGemmType == GemmType::Normal && kDualAIVDequant));

    if constexpr (kDirectStore) {
        static_assert(!kWithAccumulation || std::is_same_v<cd_dtype_t, float>);
    }
    if constexpr(kDualAIVDequant) {
        static_assert(kDirectStore);
    }

    static_assert(BLOCK_K % MX_SF_DIVISOR == 0, "BLOCK_K must be a multiple of 64");
    static_assert(BLOCK_K % MAD_K == 0, "BLOCK_K must be a multiple of MAD_K");
    static_assert(MAD_K % 64 == 0, "ascend forces MAD_K to be a multiple of 64");

    uint32_t shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m_runtime;
    uint32_t shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n_runtime;
    uint32_t shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k_runtime;
    const auto shape_sf_k = ceil_div(shape_k, MX_SF_DIVISOR);

    const gm_ptr<int16_t, Major::MN> gm_sfa(shape_m, reinterpret_cast<uintptr_t>(gmem_sfa), static_cast<uint64_t>(shape_m) * shape_sf_k);
    const gm_ptr<int16_t, Major::MN> gm_sfb(shape_n, reinterpret_cast<uintptr_t>(gmem_sfb), static_cast<uint64_t>(shape_n) * shape_sf_k);

    constexpr uint16_t kSFPairsPerKBlock = BLOCK_K / MX_SF_DIVISOR;
    constexpr uint32_t kSFPairsPerChunk = kSFPairsPerKBlock * kSFKBlocks;
    constexpr uint32_t kNumCDStages = (BLOCK_M / MAD_M) * (BLOCK_N / MAD_N);
    constexpr uint32_t kFracK = get_frac_k<float8_e4m3_t>();

    // ubuf for nd2nz always uses uint8_t for better stride/shape computation
    constexpr int C0 = get_frac_k<uint8_t>();
    constexpr int num_b_rows = kMajorB == Major::K ? BLOCK_N : BLOCK_K;
    constexpr int num_b_cols_fp4 = kMajorB == Major::K ? BLOCK_K : BLOCK_N;
    constexpr int num_b_cols = num_b_cols_fp4 / 2;
    constexpr int num_b_cols_grp = num_b_cols_fp4 / C0;
    constexpr int num_b_rows_per_stage = num_b_rows / 2;
    constexpr int num_b_rows_per_stage_pad = num_b_rows_per_stage + 1;
    constexpr int kNumUbStagesRaw = (UBSizeBytes - asc_get_vf_len()) / (num_b_rows_per_stage * num_b_cols + num_b_cols_grp * num_b_rows_per_stage_pad * C0);
    constexpr int kNumUbStages = kNumUbStagesRaw < 8 ? kNumUbStagesRaw : 8;

    constexpr ub_ptr<uint8_t> ub_lut(1, asc_get_vf_len(), 0);
    constexpr ub_ptr<uint8_t> ub_fp4(num_b_rows_per_stage, num_b_cols, ub_lut.offset(1));
    constexpr ub_ptr<uint8_t> ub_nzpad(num_b_cols_grp, num_b_rows_per_stage_pad * C0, ub_fp4.offset(kNumUbStages));

    static_assert(num_b_cols == 128 || num_b_cols == 256, "packed B row must be 128 or 256 bytes");
    static_assert(kNumUbStages > 0, "AIV0 dequant tile does not fit in UB");
    static_assert(ub_nzpad.offset(kNumUbStages) <= UBSizeBytes, "UB overflow (AIV dequant)");

    const Epilogue<cd_dtype_t, epilogue_operator_t, kWithAccumulation,
                   BLOCK_M, BLOCK_N, MAD_M, MAD_N, kNumEpilogueStages, 1,
                   kL2CtrlStoreCd> epilogue{epilogue_args};
    static_assert(!kDirectStore || std::is_same_v<typename decltype(epilogue)::operator_t,
                                                  epilogue::operators::Identity<typename decltype(epilogue)::params_t>>);

    constexpr l1_ptr<float8_e4m3_t, kMajorA> l1a(BLOCK_M, BLOCK_K, 0);
    constexpr l1_ptr<float8_e4m3_t, kMajorB> l1b(BLOCK_N, BLOCK_K, l1a.offset(kNumL1Stages));
    constexpr l1_ptr<int16_t> l1_sfa(BLOCK_M, kSFPairsPerChunk, l1b.offset(kNumL1Stages));
    constexpr l1_ptr<int16_t> l1_sfb(BLOCK_N, kSFPairsPerChunk, l1_sfa.offset(kNumL1SFStages));

    constexpr l0a_ptr<float8_e4m3_t> l0a(MAD_M, MAD_K);
    constexpr l0b_ptr<float8_e4m3_t> l0b(MAD_N, MAD_K);
    constexpr l0c_ptr<float> l0c(MAD_M, MAD_N);

    static_assert(l1_sfb.offset(kNumL1SFStages) <= L1SizeBytes, "L1 overflow");
    static_assert(l0a.offset(kNumL0Stages) <= L0ASizeBytes, "L0A overflow");
    static_assert(l0b.offset(kNumL0Stages) <= L0BSizeBytes, "L0B overflow");
    static_assert(l0c.offset(kNumCDStages) <= L0CSizeBytes, "L0C overflow");

    constexpr bool is_m_grouped = kGemmType == GemmType::MGroupedContiguousWithPsumLayout;
    constexpr bool is_k_grouped = kGemmType == GemmType::KGroupedContiguousWithPsumLayout;
    constexpr bool is_batched = kGemmType == GemmType::Batched;

    uint32_t m_block_idx, n_block_idx;
    Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumCores, kAlignment> scheduler(block_idx, kSingleMBlock ? BLOCK_M : shape_m, shape_n,
                                                                            grouped_layout, num_groups);

    const bool can_reuse_a = shape_m <= BLOCK_M && ceil_div(shape_n, BLOCK_N) > kNumCores;
    const bool can_reuse_b = shape_n <= BLOCK_N && ceil_div(shape_m, BLOCK_M) > kNumCores &&
                             (!is_m_grouped || ceil_div(shape_k, BLOCK_K) >= 2);

    if ASCEND_IS_AIC {
        if constexpr (kDirectStore && kWithAccumulation) asc_set_atomic_add_float();
        asc_set_mmad_direction_n();
        constexpr uint32_t kL1EventBase = 0;
        constexpr uint32_t kSFEventBase = kNumL1Stages;

        set_flags<PIPE_M, PIPE_MTE1, kNumL0Stages>();
        set_flags<PIPE_MTE1, PIPE_MTE2, kNumL1Stages, kL1EventBase>();
        set_flags<PIPE_MTE1, PIPE_MTE2, kNumL1SFStages, kSFEventBase>();
        set_intra_blocks<PIPE_MTE1, kNumL1Stages>(kDequantAIVId);
        if constexpr (kDualAIVDequant) set_intra_blocks<PIPE_MTE1, kNumL1Stages>(1);

        uint32_t l1_event_idx = 0;
        uint32_t l1_sf_event_idx = 0;
        uint32_t l1a_base_stage_idx = 0, l1a_stage_idx = 0;
        uint32_t l1b_base_stage_idx = 0, l1b_stage_idx = 0;
        uint32_t l1_sfa_stage_idx = 0;
        uint32_t l1_sfb_stage_idx = 0;
        uint32_t epi_base_idx = 0;
        uintptr_t last_a = 0, last_b = 0, last_sfa = 0, last_sfb = 0;

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
            // Reused SF spans one chunk, so rewind its last consumed stage.
            l1_sfa_stage_idx = reuse_sfa ? (l1_sfa_stage_idx + kNumL1SFStages - 1) % kNumL1SFStages : l1_sfa_stage_idx;
            l1_sfb_stage_idx = reuse_sfb ? (l1_sfb_stage_idx + kNumL1SFStages - 1) % kNumL1SFStages : l1_sfb_stage_idx;

            const uint32_t actual_m = kSingleMBlock ? shape_m : scheduler.get_actual_m(m_block_idx);
            const uint32_t actual_n = scheduler.get_actual_n(n_block_idx);
            uint32_t epi_stage_idx = epi_base_idx;

            for (uint32_t k_block_idx = 0, k_within_chunk = 0; k_block_idx < num_k_blocks; k_block_idx++) {
                const uint32_t k_idx = k_block_idx * BLOCK_K;
                const uint32_t actual_k = min(eff_shape_k - k_block_idx * BLOCK_K, BLOCK_K);
                const uint32_t k64 = aligned(actual_k, MX_SF_DIVISOR);

                asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(kL1EventBase + l1_event_idx));
                if (!reuse_a) copy_gm_to_l1(l1a[l1a_stage_idx], a.index(0, k_idx), actual_m, actual_k, kL2CtrlA);
                const uint32_t kc0 = aligned(actual_k, kFracK);
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
                    if (!reuse_sfa) copy_gm_to_l1_mx(l1_sfa[l1_sfa_stage_idx], sfa.index(0, k_block_idx * kSFPairsPerKBlock), actual_m,
                                     chunk_sf_pairs, kL2CtrlA);
                    if (!reuse_sfb) copy_gm_to_l1_mx(l1_sfb[l1_sfb_stage_idx], sfb.index(0, k_block_idx * kSFPairsPerKBlock), actual_n,
                                     chunk_sf_pairs, kL2CtrlB);
                    asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(kSFEventBase + l1_sf_event_idx));
                }

                // Wait A and decoded B; SF is waited at its first L0 copy.
                asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(kL1EventBase + l1_event_idx));
                wait_intra_block(PIPE_MTE1, kDequantAIVId, static_cast<uint8_t>(l1_event_idx));
                if constexpr (kDualAIVDequant) wait_intra_block(PIPE_MTE1, 1, l1_event_idx);

                epi_stage_idx = epi_base_idx;
                for (uint32_t m_mad_idx = 0, s_cd = 0; m_mad_idx < actual_m; m_mad_idx += MAD_M) {
                    for (uint32_t n_mad_idx = 0; n_mad_idx < actual_n; n_mad_idx += MAD_N, s_cd = s_cd == kNumCDStages - 1 ? 0 : s_cd + 1) {
                        const auto eff_mad_m = min(actual_m - m_mad_idx, MAD_M);
                        const auto eff_mad_n = min(actual_n - n_mad_idx, MAD_N);
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
                set_intra_block(PIPE_MTE1, kDequantAIVId, static_cast<uint8_t>(l1_event_idx));
                if constexpr (kDualAIVDequant) set_intra_block(PIPE_MTE1, 1, l1_event_idx);
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
        if constexpr (is_m_grouped && kDirectStore && BLOCK_M < kAlignment) {
            if (asc_get_sub_block_id() == 1)
                epilogue.template aiv_zero_group_padding<kNumCores, kAlignment>(gm_d, shape_n, grouped_layout, num_groups);
        }
        if (kDualAIVDequant || asc_get_sub_block_id() == kDequantAIVId) {
            const uint32_t dequant_aiv_idx = kDualAIVDequant ? asc_get_sub_block_id() : kDequantAIVId;
            vf_build_lut(ub_lut.ptr());
            set_flags<PIPE_MTE3, PIPE_MTE2, kNumUbStages>();

            uint32_t l1_event_idx = 0;
            uint32_t l1b_base_stage_idx = 0, l1b_stage_idx = 0;
            uint32_t ub_stage_idx = 0;
            uintptr_t last_b = 0;
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                const uint32_t n_idx = scheduler.get_global_idx(BLOCK_N, n_block_idx);
                const uint32_t eff_shape_k = is_k_grouped ? scheduler.current_shape_k : shape_k;
                const uint32_t num_k_blocks = ceil_div(eff_shape_k, BLOCK_K);
                const uint32_t k_idx_base = scheduler.get_k_idx_base();
                const uint32_t actual_n = scheduler.get_actual_n(n_block_idx);
                const auto batch_idx = is_batched ? scheduler.get_batch_idx() : 0;
                const auto group_idx = is_m_grouped ? scheduler.get_group_idx() : 0;
                const auto b = gm_b.batch(batch_idx).group(group_idx, shape_n, shape_k).index(n_idx, k_idx_base);
                const bool can_reuse = num_k_blocks <= kNumL1Stages;
                const bool reuse_b = can_reuse_b && can_reuse && b.addr == last_b;
                l1b_stage_idx = l1b_base_stage_idx = reuse_b ? l1b_base_stage_idx : l1b_stage_idx;

                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; k_block_idx++) {
                    const uint32_t k_idx = k_block_idx * BLOCK_K;
                    const uint32_t actual_k = min(eff_shape_k - k_block_idx * BLOCK_K, BLOCK_K);
                    const uint32_t b_valid_rows = kMajorB == Major::K ? actual_n : actual_k;
                    const uint32_t b_valid_cols = ceil_div(kMajorB == Major::K ? actual_k : actual_n, 2u);

                    if (kDualAIVDequant || reuse_b) wait_intra_block(PIPE_MTE3, dequant_aiv_idx, l1_event_idx);
                    const uint32_t row_end = reuse_b ? 0 : kDualAIVDequant ? min(b_valid_rows, (dequant_aiv_idx + 1) * (num_b_rows / 2)) : b_valid_rows;
                    for (uint32_t row_idx = dequant_aiv_idx * (num_b_rows / 2); row_idx < row_end; row_idx += num_b_rows_per_stage) {
                        const uint32_t actual_rows = min(b_valid_rows - row_idx, static_cast<uint32_t>(num_b_rows_per_stage));
                        const uint32_t b_n_off = kMajorB == Major::K ? row_idx : 0u;
                        const uint32_t b_k_off = k_idx + (kMajorB == Major::K ? 0u : row_idx);
                        asc_sync_wait(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(ub_stage_idx));
                        copy_gm_to_ub(ub_fp4[ub_stage_idx], b.index(b_n_off, b_k_off).template ptr<uint8_t>(), actual_rows,
                                      b_valid_cols, b.stride_outer, static_cast<uint32_t>(num_b_cols), kL2CtrlB);
                        asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(ub_stage_idx));
                        asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(ub_stage_idx));
                        vf_deq_nd2nz<num_b_rows_per_stage, num_b_cols, num_b_rows_per_stage_pad>(ub_fp4[ub_stage_idx].ptr(),
                                                                                                 ub_lut.ptr(),
                                                                                                 ub_nzpad[ub_stage_idx].ptr());
                        asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(ub_stage_idx));
                        asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(ub_stage_idx));
                        if (!kDualAIVDequant && row_idx == 0)
                            wait_intra_block(PIPE_MTE3, kDequantAIVId, l1_event_idx);
                        asc_copy_ub2l1(/* dst = */ l1b[l1b_stage_idx].template ptr<uint8_t>() + row_idx * C0,
                                       /* src = */ ub_nzpad[ub_stage_idx].ptr(),
                                       /* block_count    = */ static_cast<uint16_t>(num_b_cols_grp),
                                       /* block_size_32b = */ static_cast<uint16_t>(actual_rows),
                                       /* src_pad_32b    = */ static_cast<uint16_t>(num_b_rows_per_stage_pad - actual_rows),
                                       /* dst_pad_32b    = */ static_cast<uint16_t>(num_b_rows - actual_rows));
                        asc_sync_notify(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(ub_stage_idx));
                        ub_stage_idx = ub_stage_idx == kNumUbStages - 1 ? 0 : ub_stage_idx + 1;
                    }
                    set_intra_block(PIPE_MTE3, dequant_aiv_idx, l1_event_idx);
                    l1_event_idx = l1_event_idx == kNumL1Stages - 1 ? 0 : l1_event_idx + 1;
                    l1b_stage_idx = !can_reuse_b ? l1_event_idx : l1b_stage_idx == kNumL1Stages - 1 ? 0 : l1b_stage_idx + 1;
                }
                last_b = b.addr;
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
                const uint32_t batch_idx = is_batched ? scheduler.get_batch_idx() : 0;
                const uint32_t d_group_idx = is_k_grouped ? scheduler.get_group_idx() : 0;
                const auto d = gm_d.batch(batch_idx).group(d_group_idx, shape_m, shape_n);

                epilogue.aiv_drain(d, m_idx, n_idx, actual_m, actual_n, s_epi);
            }
        }
    }
}

} // namespace deep_gemm
