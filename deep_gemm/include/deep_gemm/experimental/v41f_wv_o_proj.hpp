#pragma once

#include <deep_gemm/common.hpp>

namespace deep_gemm::experimental {

// Pack all AIV fields and AIC Wv startup inputs into Line 0 (0..63B), late AIC fields (o_sf, y) into Line 1.
struct WvOProjArgs {
    DG_GM_PTR(float8_e4m3_t) x; DG_GM_PTR(int16_t) x_sf; DG_GM_PTR(float8_e4m3_t) wv; DG_GM_PTR(int16_t) wv_sf;
    DG_GM_PTR(float8_e4m3_t) o; DG_GM_PTR(float8_e4m3_t) mid; DG_GM_PTR(int16_t) mid_sf;
    uint32_t shape_m, x_sf_stride;
    DG_GM_PTR(int16_t) o_sf; DG_GM_PTR(bfloat16_t) y;
};

} // namespace deep_gemm::experimental

#ifdef DG_DEVICE_CODE
#include <deep_gemm/ascend.hpp>
#include <deep_gemm/epilogue/operators.hpp>

namespace deep_gemm::experimental {

inline constexpr uint32_t kNumGroups = 8, kWvInputDim = 4096, kWvOutputDim = 1024;
inline constexpr uint32_t kMidDim = kNumGroups * kWvOutputDim, kHidden = 5120;

inline constexpr uint32_t kWvL1BSlotBase = 0, kOL1BSlotBase = 4, kOPrefetchStartSlot = 8;
inline constexpr uint32_t kWvEpilogueSlot = 9, kSelfMidReadySlot = 11, kMidStoredSlot = 13, kODrainSlot = 15;

// Quantize BF16 Wv activations into padded NZ FP8 columns and contiguous MX-NZ scales
__simd_vf__ inline void vf_quantize_mid(__ubuf__ bfloat16_t* values, __ubuf__ uint8_t* output_nz,
                                        __ubuf__ uint16_t* output_sf, uint32_t num_rows) {
    constexpr uint32_t kVectorLenB16 = sizeof(vector_bfloat16_t) / sizeof(bfloat16_t);
    constexpr uint32_t kQuantBlockN = MX_SF_DIVISOR / 2;
    static_assert(kVectorLenB16 / kQuantBlockN == 4);
    const auto mask_b8 = asc_create_mask_b8(PAT_ALL);
    const auto mask_half = asc_create_mask_b8(PAT_VL128);
    const auto mask_b16 = asc_create_mask_b16(PAT_ALL);
    const auto sf_mask = asc_create_mask_b16(PAT_VL4);
    auto output_nd = reinterpret_cast<__ubuf__ uint8_t*>(values);

    // Map lane l to MX32 group l / 16 and MX-NZ scale offset (l % 2) * 16 + l / 2
    vector_uint16_t lane_idx, group_idx, pair_bit, row_in_pair, one, sf_offsets, sign_mask;
    asc_arange(reinterpret_cast<vector_int16_t&>(lane_idx), 0);
    asc_shiftright_scalar(group_idx, lane_idx, __builtin_ctz(kQuantBlockN / 2), mask_b16);
    asc_duplicate_scalar(one, uint16_t(1), mask_b16);
    asc_and(pair_bit, lane_idx, one, mask_b16);
    asc_shiftright_scalar(row_in_pair, lane_idx, 1, mask_b16);
    asc_mul_scalar(sf_offsets, pair_bit, uint16_t(FRAC_MN), mask_b16);
    asc_add(sf_offsets, sf_offsets, row_in_pair, mask_b16);
    asc_duplicate_scalar(sign_mask, uint16_t(0x7fff), mask_b16);

    // Quantize 256-element BF16 pairs in-place to contiguous FP8 rows and [M / 16, 2, 16] MX-NZ scales
    for (uint32_t pair_idx = 0; pair_idx < ceil_div(num_rows, 2u); ++ pair_idx) {
        const uint32_t row0 = 2 * pair_idx, offset = row0 * kVectorLenB16;
        vector_bfloat16_t even, odd;
        asc_loadalign_deintlv(even, odd, values + offset);
        vector_uint16_t abs_even, abs_odd, maxima;
        asc_and(abs_even, reinterpret_cast<vector_uint16_t&>(even), sign_mask, mask_b16);
        asc_and(abs_odd, reinterpret_cast<vector_uint16_t&>(odd), sign_mask, mask_b16);

        asc_max(maxima, abs_even, abs_odd, mask_b16);
        asc_reduce_max_datablock(maxima, maxima, mask_b16);
        math::get_ue8m0_sf_exp<float8_e4m3_t>(maxima);
        vector_uint8_t sf_bytes;
        asc_pack_to_low(sf_bytes, maxima);
        vector_uint16_t sf_store_indices;
        const auto sf_base = uint16_t(row0 + (row0 & ~uint32_t(FRAC_MN - 1)));
        asc_add_scalar(sf_store_indices, sf_offsets, sf_base, mask_b16);
        asc_scatter(output_sf, reinterpret_cast<vector_uint16_t&>(sf_bytes), sf_store_indices, sf_mask);

        vector_bfloat16_t inverse_table, inverse;
        math::get_ue8m0_sf_inv(inverse_table, maxima);
        asc_gather(inverse, inverse_table, group_idx);
        asc_mul(even, even, inverse, mask_b16);
        asc_mul(odd, odd, inverse, mask_b16);

        vector_float y[4];
        asc_bfloat162float(y[0], even, mask_b16);
        asc_bfloat162float(y[1], odd, mask_b16);
        asc_bfloat162float_v2(y[2], even, mask_b16);
        asc_bfloat162float_v2(y[3], odd, mask_b16);

        vector_fp8_e4m3fn_t fp8[4];
        auto& fp8_u8 = reinterpret_cast<vector_uint8_t(&)[4]>(fp8);
        vector_uint8_t quantized_fp8;
        asc_float2e4m3_rn_sat(fp8[0], y[0], mask_b8);
        asc_float2e4m3_rn_sat_v2(fp8[1], y[1], mask_b8);
        asc_float2e4m3_rn_sat_v3(fp8[2], y[2], mask_b8);
        asc_float2e4m3_rn_sat_v4(fp8[3], y[3], mask_b8);
        asc_or(quantized_fp8, fp8_u8[0], fp8_u8[1], mask_b8);
        asc_or(quantized_fp8, quantized_fp8, fp8_u8[2], mask_b8);
        asc_or(quantized_fp8, quantized_fp8, fp8_u8[3], mask_b8);
        asc_storealign(output_nd + offset, quantized_fp8, mask_b8);
    }

    // Transpose 128-byte quantized rows into 4 padded NZ column fractals
    const uint16_t pitch_rows = uint16_t(num_rows | 1);
    asc_mem_bar(VST_VLD);
    for (uint32_t row_idx = 0; row_idx < num_rows; ++ row_idx) {
        vector_uint8_t row_values;
        asc_loadalign(row_values, output_nd + row_idx * kVectorLenB16);
        asc_storealign(output_nz + row_idx * kDataBlockBytes, row_values, pitch_rows, 0, mask_half);
    }
}

// Repack ND tiles into padded NZ columns without writing the padded row
template <uint32_t kNumRows, uint32_t kCols>
__simd_vf__ inline void vf_nd_to_nz(__ubuf__ uint8_t* src, __ubuf__ uint8_t* dst) {
    constexpr uint32_t kPitchRows = kNumRows + 1;
    const auto mask = asc_create_mask_b8(PAT_ALL);

    // Transpose 32-byte row segments into padded NZ columns
    for (uint32_t row_idx = 0; row_idx < kNumRows; ++ row_idx) {
        for (uint32_t col_idx = 0; col_idx < kCols; col_idx += asc_get_vf_len()) {
            vector_uint8_t values;
            asc_loadalign(values, src + row_idx * kCols + col_idx);
            asc_storealign(dst + col_idx * kPitchRows + row_idx * kDataBlockBytes, values, kPitchRows, 0, mask);
        }
    }
}

template <uint32_t BLOCK_M, bool kSingleMBlock>
__global__ __mix__(1, 2) void v41f_wv_o_proj_kernel(WvOProjArgs) {
    asc_init();
    const auto& [x, x_sf, wv, wv_sf, o, mid, mid_sf, shape_m, x_sf_stride, o_sf, y] = get_replicated_args<WvOProjArgs>();

    constexpr uint32_t BLOCK_N_WV = 256, MAD_N_WV = 128, BLOCK_N_O = 160;
    constexpr uint32_t BLOCK_K = 256, MAD_K = 128, FRAC_K = get_frac_k<uint8_t>();
    constexpr uint32_t kNumWvKBlocks = kWvInputDim / BLOCK_K, kNumOKBlocks = kMidDim / BLOCK_K;
    constexpr bool kAIVStreamsWv = BLOCK_M <= 128;
    constexpr uint32_t kNumL0Stages = 2, kNumWvSFStages = 2, kNumWvSFKBlocks = 4;
    constexpr uint32_t kNumWvNBlocks = kWvOutputDim / BLOCK_N_WV;
    constexpr uint32_t kBlockSFK = BLOCK_K / MX_SF_DIVISOR, kMadSFK = MAD_K / MX_SF_DIVISOR;
    constexpr uint32_t kChunkWvSFK = kNumWvSFKBlocks * kBlockSFK;
    constexpr uint32_t kWvShapeSFK = kWvInputDim / MX_SF_DIVISOR, kOShapeSFK = kMidDim / MX_SF_DIVISOR;
    constexpr uint32_t kNumL1StagesWv = kAIVStreamsWv ? 4 : 3, kNumL1StagesO = 4;
    constexpr uint32_t kNumAIVWvL1Slots = kAIVStreamsWv ? kNumL1StagesWv : 1;
    constexpr auto kL2CtrlA = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM;
    constexpr auto kL2CtrlB = kSingleMBlock ? asc_load_l2_cache_mode::NOTALLOC_KEEP : asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM;
    constexpr auto kL2CtrlStore = asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM;
    static_assert(kSingleMBlock or not kAIVStreamsWv);
    static_assert(kSingleMBlock or kNumOKBlocks % kNumL1StagesO == 0);
    static_assert(kNumWvKBlocks % (kNumWvSFKBlocks * kNumWvSFStages) == 0);

    const uint32_t core_idx = block_idx, num_m_tiles = kSingleMBlock ? 1 : ceil_div(shape_m, BLOCK_M);
    const uint32_t group_idx = core_idx / kNumWvNBlocks, wv_n_idx = core_idx % kNumWvNBlocks * BLOCK_N_WV;
    const uint32_t o_n_idx = core_idx * BLOCK_N_O;

    const gm_ptr<float8_e4m3_t, Major::K> gm_a_wv(uint64_t(kNumGroups) * kWvInputDim, reinterpret_cast<uintptr_t>(x), kWvInputDim);
    const gm_ptr<float8_e4m3_t, Major::K> gm_b_wv(kWvInputDim, reinterpret_cast<uintptr_t>(wv), uint64_t(kWvOutputDim) * kWvInputDim);
    const gm_ptr<int16_t, Major::MN> gm_sfa_wv(x_sf_stride, reinterpret_cast<uintptr_t>(x_sf), uint64_t(x_sf_stride) * kWvShapeSFK);
    const gm_ptr<int16_t, Major::MN> gm_sfb_wv(kWvOutputDim, reinterpret_cast<uintptr_t>(wv_sf), uint64_t(kWvOutputDim) * kWvShapeSFK);
    const gm_ptr<float8_e4m3_t, Major::K> gm_b_o(kMidDim, reinterpret_cast<uintptr_t>(o));
    const gm_ptr<int16_t, Major::MN> gm_sfb_o(kHidden, reinterpret_cast<uintptr_t>(o_sf));
    const gm_ptr<bfloat16_t, Major::K> gm_y(kHidden, reinterpret_cast<uintptr_t>(y));

    // Pin O-proj B scales at the base of L1 across Wv and all O-proj M-tiles
    constexpr l1_ptr<int16_t> l1_sfb_o(BLOCK_N_O, kOShapeSFK, 0);
    constexpr l1_ptr<float8_e4m3_t, Major::K> l1a_wv(BLOCK_M, BLOCK_K, l1_sfb_o.offset(1));
    constexpr l1_ptr<float8_e4m3_t, Major::K> l1b_wv(BLOCK_N_WV, BLOCK_K, l1a_wv.offset(kNumL1StagesWv));
    constexpr l1_ptr<int16_t> l1_sfa_wv(BLOCK_M, kChunkWvSFK, l1b_wv.offset(kNumL1StagesWv));
    constexpr l1_ptr<int16_t> l1_sfb_wv(BLOCK_N_WV, kChunkWvSFK, l1_sfa_wv.offset(kNumWvSFStages));

    constexpr l1_ptr<int16_t> l1_sfa_o(BLOCK_M, kBlockSFK, l1_sfb_o.offset(1));
    constexpr l1_ptr<float8_e4m3_t, Major::K> l1a_o(BLOCK_M, BLOCK_K, l1_sfa_o.offset(kNumL1StagesO));
    constexpr l1_ptr<float8_e4m3_t, Major::K> l1b_o(BLOCK_N_O, BLOCK_K, l1a_o.offset(kNumL1StagesO));
    static_assert(l1_sfb_wv.offset(kNumWvSFStages) <= L1SizeBytes and l1b_o.offset(kNumL1StagesO) <= L1SizeBytes, "L1 overflow");

    constexpr l0a_ptr<float8_e4m3_t> l0a(BLOCK_M, MAD_K);
    constexpr l0b_ptr<float8_e4m3_t> l0b_wv(MAD_N_WV, MAD_K);
    constexpr l0b_ptr<float8_e4m3_t> l0b_o(BLOCK_N_O, MAD_K);
    constexpr l0c_ptr<float> l0c_wv(BLOCK_M, MAD_N_WV);
    constexpr l0c_ptr<float> l0c_o(BLOCK_M, BLOCK_N_O);
    static_assert(l0a.offset(kNumL0Stages) <= L0ASizeBytes and l0b_wv.offset(kNumL0Stages) <= L0BSizeBytes and l0b_o.offset(kNumL0Stages) <= L0BSizeBytes);
    static_assert(l0c_wv.offset(BLOCK_N_WV / MAD_N_WV) <= L0CSizeBytes and l0c_o.offset(1) <= L0CSizeBytes);

    // Partition UB into Wv/epilogue [0, kORegionBase) and disjoint O-weight buffers above kORegionBase
    // Host selects BLOCK_M = aligned(M, FRAC_MN) for single-tile M, guaranteeing actual_m > kEpiM under kSplitMidM
    constexpr bool kSplitMidM = kSingleMBlock and not kAIVStreamsWv;
    constexpr uint32_t kHalfNWv = BLOCK_N_WV / 2, kHalfNO = BLOCK_N_O / 2;
    constexpr uint32_t kEpiM = kSplitMidM ? aligned(BLOCK_M / 2, FRAC_MN) : BLOCK_M;
    constexpr uint32_t kNumEpiStages = kSplitMidM ? 2 : 1, kNumWvStagesUB = kAIVStreamsWv ? 2 : 1;
    constexpr uint32_t kNumORawStages = 2, kNumONZStages = 3, kNumOPrestageNZ = BLOCK_M >= 96 ? 2 : 1;
    constexpr ub_ptr<bfloat16_t> ub_epi_bf16(kEpiM, MAD_N_WV, 0);
    constexpr ub_ptr<uint8_t> ub_epi_nz(MAD_N_WV / FRAC_K, (kEpiM + 1) * FRAC_K, ub_epi_bf16.offset(kNumEpiStages));
    constexpr ub_ptr<uint16_t> ub_epi_sf(kEpiM / FRAC_MN, 2 * FRAC_MN, ub_epi_nz.offset(kNumEpiStages));
    constexpr ub_ptr<uint8_t> ub_wv_raw(kHalfNWv, BLOCK_K, 0);
    constexpr ub_ptr<uint8_t> ub_wv_nz(BLOCK_K / FRAC_K, (kHalfNWv + 1) * FRAC_K, ub_wv_raw.offset(kNumWvStagesUB));
    constexpr uint32_t kEpiUBEnd = ub_epi_sf.offset(kNumEpiStages), kWvUBEnd = ub_wv_nz.offset(kNumWvStagesUB);
    constexpr uint32_t kORegionBase = aligned(kEpiUBEnd > kWvUBEnd ? kEpiUBEnd : kWvUBEnd, uint32_t(sizeof(vector_uint8_t)));
    constexpr ub_ptr<uint8_t> ub_o_raw(kHalfNO, BLOCK_K, kORegionBase);
    constexpr ub_ptr<uint8_t> ub_o_nz(BLOCK_K / FRAC_K, (kHalfNO + 1) * FRAC_K, ub_o_raw.offset(kNumORawStages));
    static_assert(ub_o_nz.offset(kNumONZStages) <= UBSizeBytes, "UB overflow");
    static_assert(not kAIVStreamsWv or kNumWvStagesUB == kNumORawStages);
    static_assert(kODrainSlot < 16);

    if ASCEND_IS_AIC {
        asc_set_mmad_direction_n();

        constexpr uint32_t kOSFBEvent = kNumL1StagesWv + kNumWvSFStages;
        static_assert(kOSFBEvent > kNumL1StagesO and kOSFBEvent < kNumPipeEvents);
        constexpr PipeEvent<PIPE_MTE2, PIPE_MTE1> kWvL1Full{0}, kWvSFFull{kNumL1StagesWv}, kOSFBFull{kOSFBEvent};
        constexpr PipeEvent<PIPE_MTE1, PIPE_MTE2> kWvL1Empty{0}, kWvSFEmpty{kNumL1StagesWv};
        constexpr PipeEvent<PIPE_MTE2, PIPE_MTE1> kOL1Full{0}, kOL1SecondHalfFull{kNumL1StagesO};
        constexpr PipeEvent<PIPE_MTE1, PIPE_MTE2> kOL1Empty{0};
        constexpr PipeEvent<PIPE_MTE1, PIPE_M> kL0Full{0};
        constexpr PipeEvent<PIPE_M, PIPE_MTE1> kL0Empty{0};

        const auto wait_mid_stage = [&]<pipe_t kPipe>(uint32_t slot_base, uint32_t half_idx) __aicore__ {
            if constexpr (kSplitMidM) {
                wait_intra_block(kPipe, 0, slot_base + half_idx);
                wait_intra_block(kPipe, 1, slot_base + half_idx);
            } else {
                wait_intra_block(kPipe, half_idx, slot_base);
            }
        };
        const auto sync_mid_stored = [&](uint32_t half_idx) __aicore__ {
            wait_mid_stage.template operator()<PIPE_MTE2>(kMidStoredSlot, half_idx);
            asc_sync_inter_arrive(PIPE_MTE2, half_idx);
            asc_sync_inter_wait(PIPE_MTE2, half_idx);
        };

        notify<PIPE_M, PIPE_MTE1, kNumL0Stages>(kL0Empty);
        notify<PIPE_MTE1, PIPE_MTE2, kNumL1StagesWv>(kWvL1Empty);
        notify<PIPE_MTE1, PIPE_MTE2, kNumWvSFStages>(kWvSFEmpty);
        set_intra_blocks<PIPE_MTE1, kNumAIVWvL1Slots, kWvL1BSlotBase>(0);
        set_intra_blocks<PIPE_MTE1, kNumAIVWvL1Slots, kWvL1BSlotBase>(1);

        const auto b_wv = gm_b_wv.batch(group_idx).index(wv_n_idx, 0);
        const auto sfb_wv = gm_sfb_wv.batch(group_idx).index(wv_n_idx, 0);

        // Offload Wv weights to AIVs for MTE2-bound M <= 128; stream uncontended on AIC for Cube-bound M > 128
        for (uint32_t m_tile_idx = 0, m_idx = 0; m_tile_idx < num_m_tiles; ++ m_tile_idx, m_idx += BLOCK_M) {
            const uint32_t actual_m = kSingleMBlock ? shape_m : min(shape_m - m_idx, BLOCK_M);
            const auto a_wv = gm_a_wv.batch(group_idx).index(m_idx, 0);
            const auto sfa_wv = gm_sfa_wv.batch(group_idx).index(m_idx, 0);

            for (uint32_t k_block_idx = 0; k_block_idx < kNumWvKBlocks; ++ k_block_idx) {
                const uint32_t k_idx = k_block_idx * BLOCK_K;
                const uint32_t l1_stage_idx = (m_tile_idx * kNumWvKBlocks + k_block_idx) % kNumL1StagesWv;
                const uint32_t k_within_chunk = k_block_idx % kNumWvSFKBlocks;
                const uint32_t wv_sf_stage_idx = (k_block_idx / kNumWvSFKBlocks) % kNumWvSFStages;
                const bool aiv_b = kAIVStreamsWv or (m_tile_idx == 0 and k_block_idx == 0);

                wait<PIPE_MTE1, PIPE_MTE2>(kWvL1Empty, l1_stage_idx);
                copy_gm_to_l1(l1a_wv[l1_stage_idx], a_wv.index(0, k_idx), actual_m, BLOCK_K, kL2CtrlA);
                if (not aiv_b)
                    copy_gm_to_l1(l1b_wv[l1_stage_idx], b_wv.index(0, k_idx), BLOCK_N_WV, BLOCK_K, kL2CtrlB);
                notify<PIPE_MTE2, PIPE_MTE1>(kWvL1Full, l1_stage_idx);

                if (k_within_chunk == 0) {
                    wait<PIPE_MTE1, PIPE_MTE2>(kWvSFEmpty, wv_sf_stage_idx);
                    copy_gm_to_l1_mx(l1_sfa_wv[wv_sf_stage_idx], sfa_wv.index(0, k_block_idx * kBlockSFK), actual_m, kChunkWvSFK, kL2CtrlA);
                    copy_gm_to_l1_mx(l1_sfb_wv[wv_sf_stage_idx], sfb_wv.index(0, k_block_idx * kBlockSFK), BLOCK_N_WV, kChunkWvSFK, kL2CtrlB);
                    notify<PIPE_MTE2, PIPE_MTE1>(kWvSFFull, wv_sf_stage_idx);
                }
                if (m_tile_idx + 1 == num_m_tiles and k_block_idx == kNumWvKBlocks - 1) {
                    if constexpr (not kAIVStreamsWv) {
                        set_intra_block(PIPE_MTE2, 0, kOPrefetchStartSlot);
                        set_intra_block(PIPE_MTE2, 1, kOPrefetchStartSlot);
                    }
                    copy_gm_to_l1_mx(l1_sfb_o, gm_sfb_o.index(o_n_idx, 0), BLOCK_N_O, kOShapeSFK, kL2CtrlB);
                    notify<PIPE_MTE2, PIPE_MTE1>(kOSFBFull);
                }

                wait<PIPE_MTE2, PIPE_MTE1>(kWvL1Full, l1_stage_idx);

                for (uint32_t n_mad_idx = 0, cd_stage_idx = 0; n_mad_idx < BLOCK_N_WV; n_mad_idx += MAD_N_WV, ++ cd_stage_idx) {
                    auto l0c_tile = l0c_wv[cd_stage_idx].as_mad_aligned(actual_m, MAD_N_WV);
                    if (aiv_b)
                        wait_intra_block(PIPE_MTE1, cd_stage_idx, kWvL1BSlotBase + l1_stage_idx);
                    if (k_block_idx == kNumWvKBlocks - 1)
                        wait_mid_stage.template operator()<PIPE_FIX>(kWvEpilogueSlot, cd_stage_idx);

                    for (uint32_t k_mad_idx = 0, l0_stage_idx = 0; k_mad_idx < BLOCK_K; k_mad_idx += MAD_K, ++ l0_stage_idx) {
                        const auto sf_pair_idx = uint16_t(k_within_chunk * kBlockSFK + k_mad_idx / MX_SF_DIVISOR);
                        auto l0a_tile = l0a[l0_stage_idx].as_mad_aligned(actual_m, MAD_K);
                        auto l0b_tile = l0b_wv[l0_stage_idx].as_mad_aligned(MAD_N_WV, MAD_K);

                        wait<PIPE_M, PIPE_MTE1>(kL0Empty, l0_stage_idx);
                        if (cd_stage_idx == 0)
                            copy_l1_to_l0a(l0a_tile, l1a_wv[l1_stage_idx], 0, k_mad_idx);
                        copy_l1_to_l0b(l0b_tile, l1b_wv[l1_stage_idx], n_mad_idx, k_mad_idx);
                        if (k_within_chunk == 0 and cd_stage_idx == 0 and k_mad_idx == 0)
                            wait<PIPE_MTE2, PIPE_MTE1>(kWvSFFull, wv_sf_stage_idx);
                        if (cd_stage_idx == 0)
                            copy_l1_to_l0a_mx(l0a_tile, l1_sfa_wv[wv_sf_stage_idx], 0, sf_pair_idx, MAD_K);
                        copy_l1_to_l0b_mx(l0b_tile, l1_sfb_wv[wv_sf_stage_idx], n_mad_idx, sf_pair_idx, MAD_K);
                        notify<PIPE_MTE1, PIPE_M>(kL0Full, l0_stage_idx);

                        wait<PIPE_MTE1, PIPE_M>(kL0Full, l0_stage_idx);
                        mad(l0c_tile, l0a_tile, l0b_tile, actual_m, MAD_N_WV, MAD_K,
                            (k_block_idx == kNumWvKBlocks - 1 and k_mad_idx + MAD_K >= BLOCK_K) ? asc_unit_flag_mode::ENABLE_UPDATE : asc_unit_flag_mode::ENABLE_KEEP,
                            k_block_idx == 0 and k_mad_idx == 0);
                        notify<PIPE_M, PIPE_MTE1>(kL0Empty, l0_stage_idx);
                    }
                    if constexpr (kAIVStreamsWv)
                        set_intra_block(PIPE_MTE1, cd_stage_idx, kWvL1BSlotBase + l1_stage_idx);

                    if (k_block_idx == kNumWvKBlocks - 1) {
                        if constexpr (kSplitMidM) {
                            copy_l0c_to_ub(ub_epi_bf16[cd_stage_idx], l0c_tile, kEpiM, MAD_N_WV, asc_dual_dst_mode::DUAL_DST_DISABLE, 0);
                            set_intra_block(PIPE_FIX, 0, kWvEpilogueSlot + cd_stage_idx);
                            copy_l0c_to_ub(ub_epi_bf16[cd_stage_idx], l0c_tile.offset_nz_m(kEpiM), actual_m - kEpiM, MAD_N_WV, asc_dual_dst_mode::DUAL_DST_DISABLE, 1);
                            set_intra_block(PIPE_FIX, 1, kWvEpilogueSlot + cd_stage_idx);
                        } else {
                            copy_l0c_to_ub(ub_epi_bf16, l0c_tile, actual_m, MAD_N_WV, asc_dual_dst_mode::DUAL_DST_DISABLE, cd_stage_idx);
                            set_intra_block(PIPE_FIX, cd_stage_idx, kWvEpilogueSlot);
                        }
                    }
                }

                notify<PIPE_MTE1, PIPE_MTE2>(kWvL1Empty, l1_stage_idx);
                if (k_within_chunk + 1 == kNumWvSFKBlocks)
                    notify<PIPE_MTE1, PIPE_MTE2>(kWvSFEmpty, wv_sf_stage_idx);
            }
        }
        wait<PIPE_MTE1, PIPE_MTE2, kNumL1StagesWv>(kWvL1Empty);
        wait<PIPE_MTE1, PIPE_MTE2, kNumWvSFStages>(kWvSFEmpty);

        // Hand L1 over to O-proj; single-tile M <= 256 forwards step 0 on-chip and splits step 1 inter-core sync
        set_intra_blocks<PIPE_MTE1, kNumL1StagesO, kOL1BSlotBase>(0);
        set_intra_blocks<PIPE_MTE1, kNumL1StagesO, kOL1BSlotBase>(1);
        if constexpr (kSingleMBlock) {
            notify<PIPE_MTE1, PIPE_MTE2, kNumL1StagesO - 1>(kOL1Empty, 1);
        } else {
            wait_mid_stage.template operator()<PIPE_MTE2>(kMidStoredSlot, 1);
            sync_mid_stored(0);
            notify<PIPE_MTE1, PIPE_MTE2, kNumL1StagesO>(kOL1Empty);
        }

        for (uint32_t m_tile_idx = 0, m_idx = 0; m_tile_idx < num_m_tiles; ++ m_tile_idx, m_idx += BLOCK_M) {
            const uint32_t actual_m = kSingleMBlock ? shape_m : min(shape_m - m_idx, BLOCK_M);
            const uint32_t sf_half_bytes = ceil_div(actual_m, FRAC_MN) * kMadSFK * kDataBlockBytes;
            const gm_ptr<float8_e4m3_t, Major::K> gm_mid(actual_m, reinterpret_cast<uintptr_t>(mid + uint64_t(m_idx) * kMidDim));
            const auto sf_tile = reinterpret_cast<uintptr_t>(mid_sf + uint64_t(m_idx) * kOShapeSFK);
            auto l0c_o_tile = l0c_o[0].as_mad_aligned(actual_m, BLOCK_N_O);
            const auto load_o_a = [&](uint32_t l1_stage_idx, uint32_t k_block_idx, uint32_t half_idx, uint32_t width_k) __aicore__ {
                copy_gm_to_l1<true>(l1a_o[l1_stage_idx].offset_bytes(half_idx * BLOCK_M * MAD_K),
                                    gm_mid.index(k_block_idx * BLOCK_K + half_idx * MAD_K, 0), actual_m, width_k, kL2CtrlA, 1, actual_m);
                const gm_ptr<int16_t> sfa_src(0, sf_tile + uint64_t(k_block_idx * 2 + half_idx) * sf_half_bytes);
                copy_gm_to_l1_mx<true>(l1_sfa_o[l1_stage_idx].offset_bytes(half_idx * sf_half_bytes),
                                       sfa_src, actual_m, width_k / MX_SF_DIVISOR, kL2CtrlA);
            };

            for (uint32_t step_idx = 0; step_idx < kNumOKBlocks; ++ step_idx) {
                const uint32_t k_block_idx = (core_idx + step_idx) % kNumOKBlocks;
                const uint32_t l1_stage_idx = step_idx % kNumL1StagesO;

                if (kSingleMBlock and step_idx == 1) {
                    sync_mid_stored(1);
                    load_o_a(1, k_block_idx, 1, MAD_K);
                    notify<PIPE_MTE2, PIPE_MTE1>(kOL1SecondHalfFull);
                } else if (not kSingleMBlock or step_idx >= 2) {
                    wait<PIPE_MTE1, PIPE_MTE2>(kOL1Empty, l1_stage_idx);
                    load_o_a(l1_stage_idx, k_block_idx, 0, BLOCK_K);
                    notify<PIPE_MTE2, PIPE_MTE1>(kOL1Full, l1_stage_idx);
                }

                wait_intra_block(PIPE_MTE1, 0, kOL1BSlotBase + l1_stage_idx);
                wait_intra_block(PIPE_MTE1, 1, kOL1BSlotBase + l1_stage_idx);
                if (not kSingleMBlock or step_idx >= 1)
                    wait<PIPE_MTE2, PIPE_MTE1>(kOL1Full, l1_stage_idx);
                if (m_tile_idx == 0 and step_idx == 0)
                    wait<PIPE_MTE2, PIPE_MTE1>(kOSFBFull);

                for (uint32_t k_mad_idx = 0, l0_stage_idx = 0; k_mad_idx < BLOCK_K; k_mad_idx += MAD_K, ++ l0_stage_idx) {
                    auto l0a_tile = l0a[l0_stage_idx].as_mad_aligned(actual_m, MAD_K);
                    auto l0b_tile = l0b_o[l0_stage_idx].as_mad_aligned(BLOCK_N_O, MAD_K);
                    const l1_ptr<int16_t> sfa_l1_half(BLOCK_M, kMadSFK, l1_sfa_o[l1_stage_idx].addr + l0_stage_idx * sf_half_bytes);

                    if constexpr (kSingleMBlock) {
                        if (step_idx == 0)
                            wait_mid_stage.template operator()<PIPE_MTE1>(kSelfMidReadySlot, l0_stage_idx);
                        if (step_idx == 1 and l0_stage_idx == 1)
                            wait<PIPE_MTE2, PIPE_MTE1>(kOL1SecondHalfFull);
                    }

                    wait<PIPE_M, PIPE_MTE1>(kL0Empty, l0_stage_idx);
                    copy_l1_to_l0a(l0a_tile, l1a_o[l1_stage_idx], 0, k_mad_idx);
                    copy_l1_to_l0b(l0b_tile, l1b_o[l1_stage_idx], 0, k_mad_idx);
                    copy_l1_to_l0a_mx(l0a_tile, sfa_l1_half, 0, 0, MAD_K);
                    copy_l1_to_l0b_mx(l0b_tile, l1_sfb_o, 0, uint16_t(k_block_idx * kBlockSFK + k_mad_idx / MX_SF_DIVISOR), MAD_K);
                    notify<PIPE_MTE1, PIPE_M>(kL0Full, l0_stage_idx);

                    wait<PIPE_MTE1, PIPE_M>(kL0Full, l0_stage_idx);
                    mad(l0c_o_tile, l0a_tile, l0b_tile, actual_m, BLOCK_N_O, MAD_K,
                        (step_idx == kNumOKBlocks - 1 and k_mad_idx + MAD_K >= BLOCK_K) ? asc_unit_flag_mode::ENABLE_UPDATE : asc_unit_flag_mode::ENABLE_KEEP,
                        step_idx == 0 and k_mad_idx == 0);
                    notify<PIPE_M, PIPE_MTE1>(kL0Empty, l0_stage_idx);

                    // Prefetch the first half of step 1 right after issuing step 0 half 0, overlapping second-half quantization
                    if (kSingleMBlock and step_idx == 0 and l0_stage_idx == 0) {
                        sync_mid_stored(0);
                        wait<PIPE_MTE1, PIPE_MTE2>(kOL1Empty, 1);
                        load_o_a(1, (core_idx + 1) % kNumOKBlocks, 0, MAD_K);
                        notify<PIPE_MTE2, PIPE_MTE1>(kOL1Full, 1);
                    }
                }

                if (step_idx == kNumOKBlocks - 1) {
                    auto dst = gm_y.index(m_idx, o_n_idx).ptr();
                    auto store_tile = l0c_o_tile;

                    // Split disjoint N fractals; put the 32-column edge first at a 64-byte offset
                    const uint32_t num_first_cols = reinterpret_cast<uintptr_t>(dst) % kL2SectorBytes == kDCacheLineBytes ? FRAC_K : BLOCK_N_O - FRAC_K;
                    copy_l0c_to_gm(dst, store_tile, actual_m, num_first_cols, gm_y.stride_outer, kL2CtrlStore);
                    store_tile.addr += num_first_cols * store_tile.shape_m * sizeof(float);
                    copy_l0c_to_gm(dst + num_first_cols, store_tile, actual_m, BLOCK_N_O - num_first_cols, gm_y.stride_outer, kL2CtrlStore);
                }

                set_intra_block(PIPE_MTE1, 0, kOL1BSlotBase + l1_stage_idx);
                set_intra_block(PIPE_MTE1, 1, kOL1BSlotBase + l1_stage_idx);
                notify<PIPE_MTE1, PIPE_MTE2>(kOL1Empty, l1_stage_idx);
            }
        }

        wait<PIPE_M, PIPE_MTE1, kNumL0Stages>(kL0Empty);
        wait<PIPE_MTE1, PIPE_MTE2, kNumL1StagesO>(kOL1Empty);
        wait_intra_block(PIPE_MTE1, 0, kODrainSlot);
        wait_intra_block(PIPE_MTE1, 1, kODrainSlot);
    } else {
        static_assert(kNumORawStages + kNumWvStagesUB <= kNumPipeEvents and kNumONZStages + kNumWvStagesUB + kNumEpiStages <= kNumPipeEvents);
        constexpr PipeEvent<PIPE_MTE2, PIPE_V> kORawFull{0}, kWvRawFull{kNumORawStages};
        constexpr PipeEvent<PIPE_V, PIPE_MTE2> kORawEmpty{0}, kWvRawEmpty{kNumORawStages};
        constexpr PipeEvent<PIPE_V, PIPE_MTE3> kONZFull{0}, kWvNZFull{kNumONZStages}, kMidFull{kNumONZStages + kNumWvStagesUB};
        constexpr PipeEvent<PIPE_MTE3, PIPE_V> kONZEmpty{0}, kWvNZEmpty{kNumONZStages}, kMidEmpty{kNumONZStages + kNumWvStagesUB};

        const uint32_t subblock_idx = asc_get_sub_block_id();
        const uint32_t num_total_o_steps = num_m_tiles * kNumOKBlocks;
        const auto b_wv = gm_b_wv.batch(group_idx).index(wv_n_idx + subblock_idx * kHalfNWv, 0);
        const auto b_o = gm_b_o.index(o_n_idx + subblock_idx * kHalfNO, 0);

        const auto load_wv_raw = [&](uint32_t k_block_idx) __aicore__ {
            const uint32_t stage_idx = k_block_idx % kNumWvStagesUB;
            copy_gm_to_ub(ub_wv_raw[stage_idx].ptr(), b_wv.index(0, k_block_idx * BLOCK_K).template ptr<uint8_t>(),
                          kHalfNWv, BLOCK_K, b_wv.stride_outer, kL2CtrlB);
            notify<PIPE_MTE2, PIPE_V>(kWvRawFull, stage_idx);
        };
        const auto load_o_raw = [&](uint32_t step_idx) __aicore__ {
            const uint32_t k_block_idx = (core_idx + step_idx) % kNumOKBlocks;
            const uint32_t stage_idx = step_idx % kNumORawStages;
            copy_gm_to_ub(ub_o_raw[stage_idx].ptr(), b_o.index(0, k_block_idx * BLOCK_K).template ptr<uint8_t>(),
                          kHalfNO, BLOCK_K, b_o.stride_outer, kL2CtrlB);
            notify<PIPE_MTE2, PIPE_V>(kORawFull, stage_idx);
        };
        const auto stage_o_step = [&](uint32_t total_step_idx) __aicore__ {
            const uint32_t raw_stage_idx = total_step_idx % kNumORawStages;
            const uint32_t nz_stage_idx = total_step_idx % kNumONZStages;
            const uint32_t l1_stage_idx = total_step_idx % kNumL1StagesO;

            wait<PIPE_MTE2, PIPE_V>(kORawFull, raw_stage_idx);
            wait<PIPE_MTE3, PIPE_V>(kONZEmpty, nz_stage_idx);
            vf_nd_to_nz<kHalfNO, BLOCK_K>(ub_o_raw[raw_stage_idx].ptr(), ub_o_nz[nz_stage_idx].ptr());
            notify<PIPE_V, PIPE_MTE2>(kORawEmpty, raw_stage_idx);
            notify<PIPE_V, PIPE_MTE3>(kONZFull, nz_stage_idx);
            if (total_step_idx + kNumORawStages < num_total_o_steps) {
                wait<PIPE_V, PIPE_MTE2>(kORawEmpty, raw_stage_idx);
                load_o_raw(total_step_idx + kNumORawStages);
            }

            wait<PIPE_V, PIPE_MTE3>(kONZFull, nz_stage_idx);
            wait_intra_block(PIPE_MTE3, subblock_idx, kOL1BSlotBase + l1_stage_idx);
            asc_copy_ub2l1(l1b_o[l1_stage_idx].template ptr<uint8_t>() + subblock_idx * kHalfNO * FRAC_K,
                           ub_o_nz[nz_stage_idx].ptr(), BLOCK_K / FRAC_K, kHalfNO, 1, kHalfNO);
            set_intra_block(PIPE_MTE3, subblock_idx, kOL1BSlotBase + l1_stage_idx);
            notify<PIPE_MTE3, PIPE_V>(kONZEmpty, nz_stage_idx);
        };
        notify<PIPE_MTE3, PIPE_V, kNumONZStages>(kONZEmpty);

        // Stream all Wv weights on AIVs for M <= 128, and stage only block 0 for M > 128
        notify<PIPE_MTE3, PIPE_V, kNumWvStagesUB>(kWvNZEmpty);
        for (uint32_t k_block_idx = 0; k_block_idx < kNumWvStagesUB; ++ k_block_idx)
            load_wv_raw(k_block_idx);
        for (uint32_t k_block_idx = 0; k_block_idx < (kAIVStreamsWv ? kNumWvKBlocks : 1u); ++ k_block_idx) {
            const uint32_t ub_stage_idx = k_block_idx % kNumWvStagesUB, l1_stage_idx = k_block_idx % kNumL1StagesWv;
            wait<PIPE_MTE2, PIPE_V>(kWvRawFull, ub_stage_idx);
            wait<PIPE_MTE3, PIPE_V>(kWvNZEmpty, ub_stage_idx);
            vf_nd_to_nz<kHalfNWv, BLOCK_K>(ub_wv_raw[ub_stage_idx].ptr(), ub_wv_nz[ub_stage_idx].ptr());
            notify<PIPE_V, PIPE_MTE2>(kWvRawEmpty, ub_stage_idx);
            notify<PIPE_V, PIPE_MTE3>(kWvNZFull, ub_stage_idx);
            if constexpr (kAIVStreamsWv) {
                if (k_block_idx + kNumWvStagesUB < kNumWvKBlocks) {
                    wait<PIPE_V, PIPE_MTE2>(kWvRawEmpty, ub_stage_idx);
                    load_wv_raw(k_block_idx + kNumWvStagesUB);
                } else {
                    load_o_raw(k_block_idx + kNumWvStagesUB - kNumWvKBlocks);
                }
            }

            wait<PIPE_V, PIPE_MTE3>(kWvNZFull, ub_stage_idx);
            wait_intra_block(PIPE_MTE3, subblock_idx, kWvL1BSlotBase + l1_stage_idx);
            asc_copy_ub2l1(l1b_wv[l1_stage_idx].template ptr<uint8_t>() + subblock_idx * kHalfNWv * FRAC_K,
                           ub_wv_nz[ub_stage_idx].ptr(), BLOCK_K / FRAC_K, kHalfNWv, 1, kHalfNWv);
            set_intra_block(PIPE_MTE3, subblock_idx, kWvL1BSlotBase + l1_stage_idx);
            notify<PIPE_MTE3, PIPE_V>(kWvNZEmpty, ub_stage_idx);
        }
        wait<PIPE_V, PIPE_MTE2, kNumWvStagesUB>(kWvRawEmpty);
        wait<PIPE_MTE3, PIPE_V, kNumWvStagesUB>(kWvNZEmpty);
        set_intra_blocks<PIPE_V, kNumEpiStages, kWvEpilogueSlot>(subblock_idx);

        // Quantize Wv outputs into mid and prestage initial O weights on the last M-tile
        for (uint32_t m_tile_idx = 0, m_idx = 0; m_tile_idx < num_m_tiles; ++ m_tile_idx, m_idx += BLOCK_M) {
            const uint32_t actual_m = kSingleMBlock ? shape_m : min(shape_m - m_idx, BLOCK_M);
            const uint32_t sf_half_bytes = ceil_div(actual_m, FRAC_MN) * kMadSFK * kDataBlockBytes;
            auto mid_tile = reinterpret_cast<__gm__ uint8_t*>(mid + uint64_t(m_idx) * kMidDim);
            auto sf_tile = reinterpret_cast<__gm__ uint8_t*>(mid_sf + uint64_t(m_idx) * kOShapeSFK);
            const bool last_m_tile = m_tile_idx + 1 == num_m_tiles;

            if (last_m_tile) {
                if constexpr (not kAIVStreamsWv) {
                    wait_intra_block(PIPE_MTE2, subblock_idx, kOPrefetchStartSlot);
                    for (uint32_t step_idx = 0; step_idx < kNumORawStages; ++ step_idx)
                        load_o_raw(step_idx);
                }
                for (uint32_t step_idx = 0; step_idx < kNumOPrestageNZ; ++ step_idx)
                    stage_o_step(step_idx);
            }

            const uint32_t row_offset = kSplitMidM ? subblock_idx * kEpiM : 0;
            const uint32_t num_rows = actual_m > row_offset ? min(actual_m - row_offset, kEpiM) : 0;
            const uint32_t pitch_rows = num_rows | 1;
            const uint32_t sf_bytes = ceil_div(num_rows, FRAC_MN) * kMadSFK * kDataBlockBytes;
            const uint32_t sf_row_offset_bytes = (row_offset / FRAC_MN) * kMadSFK * kDataBlockBytes;

            for (uint32_t pass_idx = 0; pass_idx < kNumEpiStages; ++ pass_idx) {
                const uint32_t col_half_idx = kSplitMidM ? pass_idx : subblock_idx;
                wait_intra_block(PIPE_V, subblock_idx, kWvEpilogueSlot + pass_idx);
                // Keep a single num_rows guard so Bisheng does not hoist epilogue descriptors across stage_o_step
                if (num_rows) {
                    vf_quantize_mid(ub_epi_bf16[pass_idx].ptr(), ub_epi_nz[pass_idx].ptr(), ub_epi_sf[pass_idx].ptr(), num_rows);
                    notify<PIPE_V, PIPE_MTE3>(kMidFull, pass_idx);
                    wait<PIPE_V, PIPE_MTE3>(kMidFull, pass_idx);

                    if constexpr (kSingleMBlock) {
                        asc_copy_ub2l1(l1a_o[0].template ptr<uint8_t>() + col_half_idx * BLOCK_M * MAD_N_WV + row_offset * FRAC_K,
                                       ub_epi_nz[pass_idx].ptr(), MAD_N_WV / FRAC_K, num_rows, pitch_rows - num_rows, BLOCK_M - num_rows);
                        asc_copy_ub2l1(l1_sfa_o[0].template ptr<uint8_t>() + col_half_idx * sf_half_bytes + sf_row_offset_bytes,
                                       ub_epi_sf[pass_idx].template ptr<uint8_t>(), sf_bytes);
                        set_intra_block(PIPE_MTE3, subblock_idx, kSelfMidReadySlot + pass_idx);
                    }

                    const uint64_t mid_col_idx = core_idx * BLOCK_K + col_half_idx * MAD_N_WV;
                    asc_copy_ub2gm_align(mid_tile + mid_col_idx * actual_m + row_offset * kDataBlockBytes,
                                         ub_epi_nz[pass_idx].ptr(), MAD_N_WV / FRAC_K, num_rows * kDataBlockBytes, kL2CtrlStore,
                                         uint64_t(actual_m) * kDataBlockBytes, pitch_rows * kDataBlockBytes);
                    asc_copy_ub2gm_align(sf_tile + (core_idx * 2 + col_half_idx) * sf_half_bytes + sf_row_offset_bytes,
                                         ub_epi_sf[pass_idx].template ptr<uint8_t>(), 1, sf_bytes, kL2CtrlStore, 0, 0);
                } else if constexpr (kSingleMBlock) {
                    set_intra_block(PIPE_MTE3, subblock_idx, kSelfMidReadySlot + pass_idx);
                }

                if (not last_m_tile) {
                    notify<PIPE_MTE3, PIPE_V>(kMidEmpty, pass_idx);
                    wait<PIPE_MTE3, PIPE_V>(kMidEmpty, pass_idx);
                    set_intra_block(PIPE_V, subblock_idx, kWvEpilogueSlot + pass_idx);
                } else {
                    set_intra_block(PIPE_MTE3, subblock_idx, kMidStoredSlot + pass_idx);
                }
            }
        }

        // Stream remaining O-proj weight blocks continuously across all M-tiles
        for (uint32_t total_step_idx = kNumOPrestageNZ; total_step_idx < num_total_o_steps; ++ total_step_idx)
            stage_o_step(total_step_idx);

        if constexpr (kAIVStreamsWv)
            wait_intra_blocks<PIPE_MTE3, kNumAIVWvL1Slots, kWvL1BSlotBase>(subblock_idx);
        wait<PIPE_V, PIPE_MTE2, kNumORawStages>(kORawEmpty);
        wait<PIPE_MTE3, PIPE_V, kNumONZStages>(kONZEmpty);
        wait_intra_blocks<PIPE_MTE3, kNumL1StagesO, kOL1BSlotBase>(subblock_idx);
        set_intra_block(PIPE_MTE3, subblock_idx, kODrainSlot);
    }
}

} // namespace deep_gemm::experimental
#endif
