#pragma once

#include <deep_gemm/ascend.hpp>

namespace deep_gemm::mega_mhc {

// Pack X into padded NZ; zero inactive token rows
template <uint32_t BLOCK_M, uint32_t BLOCK_H>
__simd_vf__ inline void vf_pack_x_nz(__ubuf__ bfloat16_t* x, __ubuf__ bfloat16_t* x_nz, uint32_t actual_m) {
    constexpr uint32_t VL_b16 = sizeof(vector_bfloat16_t) / sizeof(bfloat16_t);

    static_assert(BLOCK_H == VL_b16);
    const auto mask_b16 = asc_create_mask_b16(PAT_ALL);
    vector_bfloat16_t zero;
    asc_duplicate_scalar(zero, 0);

    for (uint32_t m_idx = 0; m_idx < actual_m; ++ m_idx) {
        vector_bfloat16_t value;
        asc_loadalign_postupdate(value, x, BLOCK_H);
        asc_storealign_postupdate(x_nz, value, BLOCK_M + 1, 1, mask_b16);
    }

    // Zero inactive token rows
    for (uint32_t m_idx = actual_m; m_idx < BLOCK_M; ++ m_idx)
        asc_storealign_postupdate(x_nz, zero, BLOCK_M + 1, 1, mask_b16);
}

// Clear coefficient slots before masked Comb/Post stores
template <uint32_t kNumElements>
__simd_vf__ inline void vf_init_post_coeff(__ubuf__ bfloat16_t* post_coeff) {
    constexpr uint32_t VL_b16 = sizeof(vector_bfloat16_t) / sizeof(bfloat16_t);

    vector_bfloat16_t zero;
    asc_duplicate_scalar(zero, 0);
    const auto mask_b16 = asc_create_mask_b16(PAT_ALL);

    for (uint32_t offset = 0; offset < kNumElements; offset += VL_b16)
        asc_storealign_postupdate(post_coeff, zero, VL_b16, mask_b16);
}

// Pack comb/post coefficients as three BF16 matrices for Post MAD
template <
    uint32_t BLOCK_M, uint32_t kNumRoutes, uint32_t kPostMadM, uint32_t kPostMadK,
    uint32_t kPostMadResidualK, uint32_t kNumFP32toBF16Splits
>
__simd_vf__ inline void vf_prepare_post_coeff(__ubuf__ float* comb, __ubuf__ float* post, __ubuf__ bfloat16_t* post_coeff) {
    constexpr uint32_t VL_b16 = sizeof(vector_bfloat16_t) / sizeof(bfloat16_t);
    constexpr uint32_t VL_b32 = sizeof(vector_float) / sizeof(float);

    static_assert(kNumFP32toBF16Splits == 3);
    constexpr uint32_t kNumTokensPerPostMad = kPostMadM / kNumRoutes;
    constexpr uint32_t kNumPostMadsPerMBlock = BLOCK_M / kNumTokensPerPostMad;
    constexpr uint32_t kNumCoeffElementsPerSplit = kPostMadM * kPostMadK;
    constexpr uint32_t kNumCoeffVectors = kPostMadM * kPostMadResidualK / VL_b16;

    // Coefficient packing uses four routes and BF16 M16/K16 fractals
    static_assert(kNumRoutes == 4 && kPostMadM == FRAC_MN && kPostMadResidualK == get_frac_k<bfloat16_t>());
    // Post K32 holds this MAD's four-token residuals in K16, then this AIV's X rows in K16
    // Each AIV uses the first BLOCK_M rows of its own X fractal
    static_assert(kPostMadK == 2 * kPostMadResidualK && BLOCK_M <= kPostMadResidualK);

    const auto mask_b16 = asc_create_mask_b16(PAT_ALL);
    const auto mask_b32 = asc_create_mask_b32(PAT_ALL);

    vector_int16_t lane_idx_b16;
    asc_arange(lane_idx_b16, int16_t(0));
    auto lane_idx = reinterpret_cast<vector_uint16_t&>(lane_idx_b16);

    vector_uint16_t row_idx, col_idx;
    vector_uint16_t token_idx, col_group_idx;
    vector_uint16_t route_mask, residual_k_mask;
    vector_uint16_t input_route_idx, output_route_idx;
    vector_uint16_t comb_src_idx, comb_row_offset;
    vector_uint16_t comb_cast_idx, post_cast_idx, post_col_offset;

    // First index: Comb or Post; second index: vector within the K16 half
    vector_uint16_t coeff_indices[2][kNumCoeffVectors];
    vector_bool coeff_masks_b16[2][kNumCoeffVectors];

    // First 8 rows: row_idx = lane // 16 = 4 * token_idx + output_route_idx, col_idx = lane % 16
    asc_duplicate_scalar(route_mask, uint16_t(kNumRoutes - 1), mask_b16);
    asc_duplicate_scalar(residual_k_mask, uint16_t(kPostMadResidualK - 1), mask_b16);
    asc_shiftright_scalar(row_idx, lane_idx, uint16_t(4), mask_b16);
    asc_and(col_idx, lane_idx, residual_k_mask, mask_b16);
    asc_shiftright_scalar(token_idx, row_idx, uint16_t(2), mask_b16);
    asc_shiftright_scalar(col_group_idx, col_idx, uint16_t(2), mask_b16);
    asc_and(input_route_idx, col_idx, route_mask, mask_b16);
    asc_and(output_route_idx, row_idx, route_mask, mask_b16);

    // Comb source layout: [token, input route, output route]
    asc_shiftleft_scalar(comb_src_idx, token_idx, uint16_t(4), mask_b16);
    asc_shiftleft_scalar(comb_row_offset, input_route_idx, uint16_t(2), mask_b16);
    asc_add(comb_src_idx, comb_src_idx, comb_row_offset, mask_b16);
    asc_add(comb_src_idx, comb_src_idx, output_route_idx, mask_b16);

    // FP32-to-BF16 cast: [c0, c1, ...] -> [c0, _, c1, _, ...]; gather indices count BF16 lanes
    asc_shiftleft_scalar(comb_cast_idx, comb_src_idx, uint16_t(1), mask_b16);
    asc_shiftleft_scalar(post_cast_idx, row_idx, uint16_t(1), mask_b16);

    // Post [token, output route] selects the X column belonging to this token
    asc_sub(post_col_offset, col_idx, token_idx, mask_b16);

    // For vector v: row = 8 * v + lane // 16, col = lane % 16, token = row // 4
    // Comb index = 2 * (16 * token + 4 * (col % 4) + row % 4), active when col // 4 == token
    // Post index = 2 * row, active when col == 4 * post_mad_idx + token in the X half
    #pragma unroll
    for (uint32_t vector_idx = 0; vector_idx < kNumCoeffVectors; ++ vector_idx) {
        vector_uint16_t vector_token_idx;
        asc_add_scalar(vector_token_idx, token_idx, uint16_t(vector_idx * kNumTokensPerPostMad / kNumCoeffVectors), mask_b16);
        asc_eq(coeff_masks_b16[0][vector_idx], col_group_idx, vector_token_idx, mask_b16);
        asc_add_scalar(coeff_indices[0][vector_idx], comb_cast_idx, uint16_t(vector_idx * VL_b32), mask_b16);
        asc_add_scalar(coeff_indices[1][vector_idx], post_cast_idx, uint16_t(vector_idx * kPostMadM), mask_b16);
    }

    __ubuf__ float* coeff_src[2] = {comb, post};
    #pragma unroll 1
    for (uint32_t post_mad_idx = 0; post_mad_idx < kNumPostMadsPerMBlock; ++ post_mad_idx) {
        #pragma unroll
        for (uint32_t vector_idx = 0; vector_idx < kNumCoeffVectors; ++ vector_idx)
            asc_eq_scalar(coeff_masks_b16[1][vector_idx], post_col_offset,
                          uint16_t(post_mad_idx * kNumTokensPerPostMad + vector_idx * kNumTokensPerPostMad / kNumCoeffVectors), mask_b16);

        #pragma unroll
        for (uint32_t coeff_idx = 0; coeff_idx < 2; ++ coeff_idx) {
            vector_float remainder;
            vector_bfloat16_t coeff_bf16[kNumFP32toBF16Splits];
            asc_loadalign_postupdate(remainder, coeff_src[coeff_idx],
                                     kNumTokensPerPostMad * kNumRoutes * (coeff_idx == 0 ? kNumRoutes : 1));

            // Each split rounds the remainder left by the preceding split
            #pragma unroll
            for (uint32_t split_idx = 0; split_idx < kNumFP32toBF16Splits; ++ split_idx) {
                asc_float2bfloat16_rn_sat(coeff_bf16[split_idx], remainder, mask_b32);
                if (split_idx + 1 < kNumFP32toBF16Splits) {
                    vector_float coeff_fp32;
                    asc_bfloat162float(coeff_fp32, coeff_bf16[split_idx], mask_b16);
                    asc_sub(remainder, remainder, coeff_fp32, mask_b32);
                }
            }

            // Gather each split into the masked Comb or Post half
            #pragma unroll
            for (uint32_t split_idx = 0; split_idx < kNumFP32toBF16Splits; ++ split_idx) {
                auto* dst = post_coeff + (post_mad_idx * kNumFP32toBF16Splits + split_idx) * kNumCoeffElementsPerSplit +
                            coeff_idx * kPostMadM * kPostMadResidualK;
                #pragma unroll
                for (uint32_t vector_idx = 0; vector_idx < kNumCoeffVectors; ++ vector_idx) {
                    vector_bfloat16_t coeff;
                    asc_gather(coeff, coeff_bf16[split_idx], coeff_indices[coeff_idx][vector_idx]);
                    asc_storealign_postupdate(dst, coeff, VL_b16, coeff_masks_b16[coeff_idx][vector_idx]);
                }
            }
        }
    }
}

// BF16 Post output -> FP32 NZ, with one padding row per fractal column
template <uint32_t BLOCK_M, uint32_t BLOCK_H, uint32_t kNumRoutes>
__simd_vf__ inline void vf_cast_new_residual_nz(__ubuf__ bfloat16_t* new_residual, __ubuf__ float* new_residual_nz,
                                                uint32_t actual_m) {
    constexpr uint32_t VL_b32 = sizeof(vector_float) / sizeof(float);

    static_assert(kNumRoutes == 4);
    static_assert(BLOCK_H == 2 * VL_b32);

    constexpr uint32_t FRAC_K = get_frac_k<float>();
    constexpr uint32_t kNumHVectors = BLOCK_H / VL_b32;
    constexpr uint32_t kNZMStride = BLOCK_M + 1;

    const auto mask_b16 = asc_create_mask_b16(PAT_ALL);
    const auto mask_b32 = asc_create_mask_b32(PAT_ALL);

    #pragma unroll 1
    for (uint32_t m_idx = 0; m_idx < actual_m; ++ m_idx) {
        vector_bfloat16_t residual_bf16[kNumRoutes][kNumHVectors];
        vector_float residual_fp32[kNumRoutes][kNumHVectors];

        // Load and cast each route to FP32
        #pragma unroll
        for (uint32_t route_idx = 0; route_idx < kNumRoutes; ++ route_idx) {
            #pragma unroll
            for (uint32_t vector_idx = 0; vector_idx < kNumHVectors; ++ vector_idx)
                asc_loadalign_unpack_postupdate(residual_bf16[route_idx][vector_idx], new_residual, VL_b32);

            #pragma unroll
            for (uint32_t vector_idx = 0; vector_idx < kNumHVectors; ++ vector_idx)
                asc_bfloat162float(residual_fp32[route_idx][vector_idx], residual_bf16[route_idx][vector_idx], mask_b16);
        }

        // [token, route, H] -> [route, H8, token (padded), h8]
        auto* dst = new_residual_nz + m_idx * FRAC_K;
        #pragma unroll
        for (uint32_t route_idx = 0; route_idx < kNumRoutes; ++ route_idx) {
            #pragma unroll
            for (uint32_t vector_idx = 0; vector_idx < kNumHVectors; ++ vector_idx)
                asc_storealign(dst + (route_idx * kNumHVectors + vector_idx) * VL_b32 * kNZMStride,
                               residual_fp32[route_idx][vector_idx], kNZMStride, 0, mask_b32);
        }
    }
}

// PreApply from FP32 NZ: write BF16 Y and accumulate HC/RMS square sums for valid rows
template <uint32_t BLOCK_M, uint32_t BLOCK_H, uint32_t kNumRoutes>
__simd_vf__ inline void vf_preapply_sqr_sum_from_nz(__ubuf__ float* new_residual_nz, __ubuf__ float* hc_sqr_sum,
                                                    __ubuf__ float* shifted_prev_mix, __ubuf__ bfloat16_t* y_bf16,
                                                    __ubuf__ float* rms_sqr_sum, uint32_t actual_m, const bool is_first_h_block) {
    constexpr uint32_t VL_b32 = sizeof(vector_float) / sizeof(float);

    constexpr uint32_t FRAC_K = get_frac_k<float>();
    constexpr uint32_t kNumRowsPerVector = VL_b32 / FRAC_K;
    constexpr uint32_t kNZKStride = (BLOCK_M + 1) * FRAC_K;
    constexpr uint32_t kNZRouteStride = BLOCK_H / FRAC_K * kNZKStride;
    static_assert(kNumRoutes == 4 && FRAC_K == 8 && (BLOCK_M == kNumRowsPerVector || BLOCK_M == 2 * kNumRowsPerVector));

    const auto mask_b32 = asc_create_mask_b32(PAT_ALL);
    const auto mask_b16 = asc_create_mask_b16(PAT_ALL);

    // Broadcast [token, route] -> [token, h8]: [t0_r x8, t1_r x8, ...], with t0 at m_idx
    // pre_coeff_idx = (m_idx + lane // 8) * 4 + route_idx
    vector_int32_t lane_idx_b32, pre_base_idx;
    asc_arange(lane_idx_b32, 0);
    asc_shiftright_scalar(pre_base_idx, lane_idx_b32, __builtin_ctz(FRAC_K), mask_b32);
    asc_shiftleft_scalar(pre_base_idx, pre_base_idx, __builtin_ctz(kNumRoutes), mask_b32);
    vector_float pre_coeff_all;
    asc_loadalign(pre_coeff_all, shifted_prev_mix);

    // Gather [H8 half, token, h8] into [token, h16] after packing the even BF16 cast lanes
    // y_nd_idx = (lane // 16) * 8 + lane % 8 + ((lane % 16) // 8) * 64
    vector_int16_t lane_idx_b16;
    vector_int16_t h_lane_idx, h_half_offset;
    vector_int16_t m_offset, y_nd_idx;
    vector_int16_t frac_k_mask, h_half_mask;
    asc_arange(lane_idx_b16, 0);
    asc_duplicate_scalar(frac_k_mask, int16_t(FRAC_K - 1), mask_b16);
    asc_duplicate_scalar(h_half_mask, int16_t(FRAC_K), mask_b16);
    asc_and(h_lane_idx, lane_idx_b16, frac_k_mask, mask_b16);
    asc_shiftright_scalar(m_offset, lane_idx_b16, __builtin_ctz(2 * FRAC_K), mask_b16);
    asc_shiftleft_scalar(m_offset, m_offset, __builtin_ctz(FRAC_K), mask_b16);
    asc_and(h_half_offset, lane_idx_b16, h_half_mask, mask_b16);
    asc_shiftleft_scalar(h_half_offset, h_half_offset, __builtin_ctz(kNumRowsPerVector), mask_b16);
    asc_add(y_nd_idx, m_offset, h_lane_idx, mask_b16);
    asc_add(y_nd_idx, y_nd_idx, h_half_offset, mask_b16);

    #pragma unroll
    for (uint32_t m_idx = 0; m_idx < BLOCK_M; m_idx += kNumRowsPerVector) {
        if (m_idx >= actual_m)
            continue;
        uint32_t num_rows = min(kNumRowsPerVector, actual_m - m_idx);
        uint32_t num_y_values = num_rows * 2 * FRAC_K;
        const auto row_mask_b32 = asc_update_mask_b32(num_rows);
        const auto y_mask_b16 = asc_update_mask_b16(num_y_values);

        vector_float pre_coeff[kNumRoutes];
        #pragma unroll
        for (uint32_t route_idx = 0; route_idx < kNumRoutes; ++ route_idx) {
            vector_int32_t pre_coeff_idx;
            asc_add_scalar(pre_coeff_idx, pre_base_idx, int32_t(m_idx * kNumRoutes + route_idx), mask_b32);
            asc_gather(pre_coeff[route_idx], pre_coeff_all, reinterpret_cast<vector_uint32_t&>(pre_coeff_idx));
        }

        vector_float hc_sqr_sum0, hc_sqr_sum1;
        vector_float rms_sqr_sum0, rms_sqr_sum1;
        asc_duplicate_scalar(hc_sqr_sum0, 0.0f, mask_b32);
        asc_duplicate_scalar(hc_sqr_sum1, 0.0f, mask_b32);
        asc_duplicate_scalar(rms_sqr_sum0, 0.0f, mask_b32);
        asc_duplicate_scalar(rms_sqr_sum1, 0.0f, mask_b32);

        auto* y_bf16_ptr = y_bf16 + m_idx * BLOCK_H;
        #pragma unroll 1
        for (uint32_t h_idx = 0; h_idx < BLOCK_H; h_idx += 2 * FRAC_K) {
            // Load two H8 fragments for every route
            vector_float new_residual[kNumRoutes][2];
            #pragma unroll
            for (uint32_t route_idx = 0; route_idx < kNumRoutes; ++ route_idx) {
                auto* new_residual_ptr = new_residual_nz + route_idx * kNZRouteStride + (h_idx / FRAC_K) * kNZKStride + m_idx * FRAC_K;
                #pragma unroll
                for (uint32_t h_frac_idx = 0; h_frac_idx < 2; ++ h_frac_idx)
                    asc_loadalign(new_residual[route_idx][h_frac_idx], new_residual_ptr + h_frac_idx * kNZKStride);
            }

            // Independent route-parity chains preserve the reduction order
            #pragma unroll
            for (uint32_t h_frac_idx = 0; h_frac_idx < 2; ++ h_frac_idx) {
                #pragma unroll
                for (uint32_t route_idx = 0; route_idx < kNumRoutes; ++ route_idx) {
                    if (route_idx % 2 == 0)
                        asc_mula(hc_sqr_sum0, new_residual[route_idx][h_frac_idx], new_residual[route_idx][h_frac_idx], mask_b32);
                    else
                        asc_mula(hc_sqr_sum1, new_residual[route_idx][h_frac_idx], new_residual[route_idx][h_frac_idx], mask_b32);
                }
            }

            // Explicit named accumulators: nested unrolled updates to y[h_frac_idx]
            // lost the middle two route FMAs in this toolchain's generated VF
            vector_float y0, y1;
            asc_mul(y0, new_residual[0][0], pre_coeff[0], mask_b32);
            asc_mul(y1, new_residual[0][1], pre_coeff[0], mask_b32);
            #pragma unroll
            for (uint32_t route_idx = 1; route_idx < kNumRoutes; ++ route_idx) {
                asc_mula(y0, new_residual[route_idx][0], pre_coeff[route_idx], mask_b32);
                asc_mula(y1, new_residual[route_idx][1], pre_coeff[route_idx], mask_b32);
            }

            vector_bfloat16_t y_bf16_0, y_bf16_1;
            asc_float2bfloat16_rn_sat(y_bf16_0, y0, mask_b16);
            asc_float2bfloat16_rn_sat(y_bf16_1, y1, mask_b16);

            // RMS must observe the same BF16-rounded y that is written to GM
            asc_bfloat162float(y0, y_bf16_0, mask_b16);
            asc_bfloat162float(y1, y_bf16_1, mask_b16);
            asc_mula(rms_sqr_sum0, y0, y0, mask_b32);
            asc_mula(rms_sqr_sum1, y1, y1, mask_b32);

            // BF16 casts are [t0_h0, _, ...] and [t0_h8, _, ...], with t0/h0 at m_idx/h_idx
            // Deinterleave: y_packed = [t0_h0..7, ..., t7_h0..7, t0_h8..15, ..., t7_h8..15]
            vector_bfloat16_t y_packed, unused;
            asc_deintlv(y_packed, unused, y_bf16_0, y_bf16_1);

            // y_nd = [t0_h0..15, t1_h0..15, ...]; store each H16 fragment in its token row
            vector_bfloat16_t y_nd;
            asc_gather(y_nd, y_packed, reinterpret_cast<vector_uint16_t&>(y_nd_idx));
            asc_storealign_postupdate(y_bf16_ptr, y_nd, BLOCK_H / (2 * FRAC_K), 1, y_mask_b16);
        }

        // Reduce H8 lanes to one square sum per token
        vector_float hc_sqr_sum_reduced, rms_sqr_sum_reduced;
        asc_add(hc_sqr_sum0, hc_sqr_sum0, hc_sqr_sum1, mask_b32);
        asc_add(rms_sqr_sum0, rms_sqr_sum0, rms_sqr_sum1, mask_b32);
        asc_reduce_sum_datablock(hc_sqr_sum_reduced, hc_sqr_sum0, mask_b32);
        asc_reduce_sum_datablock(rms_sqr_sum_reduced, rms_sqr_sum0, mask_b32);

        // Accumulate previous H-block square sums
        if (!is_first_h_block) {
            vector_float previous_hc_sqr_sum, previous_rms_sqr_sum;
            asc_loadalign(previous_hc_sqr_sum, hc_sqr_sum + m_idx);
            asc_loadalign(previous_rms_sqr_sum, rms_sqr_sum + m_idx);
            asc_add(hc_sqr_sum_reduced, hc_sqr_sum_reduced, previous_hc_sqr_sum, row_mask_b32);
            asc_add(rms_sqr_sum_reduced, rms_sqr_sum_reduced, previous_rms_sqr_sum, row_mask_b32);
        }

        asc_storealign(hc_sqr_sum + m_idx, hc_sqr_sum_reduced, row_mask_b32);
        asc_storealign(rms_sqr_sum + m_idx, rms_sqr_sum_reduced, row_mask_b32);
    }
}

} // namespace deep_gemm::mega_mhc
