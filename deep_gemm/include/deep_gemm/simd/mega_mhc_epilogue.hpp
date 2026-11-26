#pragma once

#include <deep_gemm/ascend.hpp>

namespace deep_gemm::mega_mhc {

// Reduce K-split partials in ascending split order into split 0
template <uint32_t kNumKSplits, uint32_t kNumElements>
__simd_vf__ inline void vf_reduce_splits(__ubuf__ float* partials) {
    constexpr uint32_t VL_b32 = sizeof(vector_float) / sizeof(float);

    static_assert(kNumElements < VL_b32 || kNumElements % VL_b32 == 0);
    vector_bool mask_b32;
    if constexpr (kNumElements < VL_b32) {
        uint32_t num_elements = kNumElements;
        mask_b32 = asc_update_mask_b32(num_elements);
    } else {
        mask_b32 = asc_create_mask_b32(PAT_ALL);
    }

    for (uint32_t offset = 0; offset < kNumElements; offset += VL_b32) {
        vector_float sum, partial;
        asc_loadalign(sum, partials + offset);

        for (uint32_t split_idx = 1; split_idx < kNumKSplits; ++ split_idx) {
            asc_loadalign(partial, partials + split_idx * kNumElements + offset);
            asc_add(sum, sum, partial, mask_b32);
        }

        asc_storealign(partials + offset, sum, mask_b32);
    }
}

// Broadcast each four-lane row or column sum with two butterfly gathers
__simd_callee__ inline void mix_axis_sum(vector_float& dst, vector_float src,
                                         vector_uint32_t first_swap_idx, vector_uint32_t second_swap_idx, vector_bool mask_b32) {
    vector_float neighbor, pair_sum;
    asc_gather(neighbor, src, first_swap_idx);
    asc_add(pair_sum, src, neighbor, mask_b32);
    asc_gather(neighbor, pair_sum, second_swap_idx);
    asc_add(dst, pair_sum, neighbor, mask_b32);
}

template <uint32_t BLOCK_M, uint32_t kNumRoutes>
__simd_callee__ inline void mix_sinkhorn(__ubuf__ float* ub_comb_logits, __ubuf__ float* ub_comb,
                                         float sinkhorn_eps, uint32_t num_sinkhorn_iters) {
    constexpr uint32_t VL_b32 = sizeof(vector_float) / sizeof(float);

    static_assert(kNumRoutes == 4 and BLOCK_M * kNumRoutes * kNumRoutes % VL_b32 == 0);
    constexpr int32_t kNumDataBlockElements = kDataBlockBytes / sizeof(float);
    const vector_bool mask_b32 = asc_create_mask_b32(PAT_ALL);

    vector_int32_t lane_idx, index_mask;
    vector_int32_t datablock_lane_idx, m_datablock_offset;
    vector_int32_t comb_lane_idx, row_max_idx;
    asc_arange(lane_idx, 0);

    // Lane 16*t + 4*r + c holds token t, row r, column c of a row-major 4x4 matrix
    // Each eight-lane datablock spans two rows; low/high masks select even/odd rows
    asc_duplicate_scalar(index_mask, kNumDataBlockElements - 1, mask_b32);
    asc_and(datablock_lane_idx, lane_idx, index_mask, mask_b32);
    vector_bool row_mask_b32_low, row_mask_b32_high;
    asc_lt_scalar(row_mask_b32_low, datablock_lane_idx, int32_t(kNumRoutes), mask_b32);
    asc_ge_scalar(row_mask_b32_high, datablock_lane_idx, int32_t(kNumRoutes), mask_b32);

    // row_max_idx = (lane / 16) * 2 + (lane % 16) / 8: broadcast one packed result per datablock
    asc_shiftright_scalar(m_datablock_offset, lane_idx, 4, mask_b32);
    asc_shiftleft_scalar(m_datablock_offset, m_datablock_offset, 1, mask_b32);
    asc_duplicate_scalar(index_mask, int32_t(kNumRoutes * kNumRoutes - 1), mask_b32);
    asc_and(comb_lane_idx, lane_idx, index_mask, mask_b32);
    asc_shiftright_scalar(row_max_idx, comb_lane_idx, 3, mask_b32);
    asc_add(row_max_idx, m_datablock_offset, row_max_idx, mask_b32);

    // Row gathers use lane ^ {1, 2} to swap columns; column gathers use lane ^ {4, 8} to swap rows
    vector_int32_t row_swap1;
    asc_duplicate_scalar(index_mask, 1, mask_b32);
    asc_xor(row_swap1, lane_idx, index_mask, mask_b32);
    auto row_swap1_idx = reinterpret_cast<vector_uint32_t&>(row_swap1);
    vector_int32_t row_swap2;
    asc_duplicate_scalar(index_mask, 2, mask_b32);
    asc_xor(row_swap2, lane_idx, index_mask, mask_b32);
    auto row_swap2_idx = reinterpret_cast<vector_uint32_t&>(row_swap2);

    vector_int32_t col_swap1;
    asc_duplicate_scalar(index_mask, int32_t(kNumRoutes), mask_b32);
    asc_xor(col_swap1, lane_idx, index_mask, mask_b32);
    auto col_swap1_idx = reinterpret_cast<vector_uint32_t&>(col_swap1);
    vector_int32_t col_swap2;
    asc_duplicate_scalar(index_mask, int32_t(2 * kNumRoutes), mask_b32);
    asc_xor(col_swap2, lane_idx, index_mask, mask_b32);
    auto col_swap2_idx = reinterpret_cast<vector_uint32_t&>(col_swap2);
    auto row_max_broadcast_idx = reinterpret_cast<vector_uint32_t&>(row_max_idx);

    // Load combination logits
    constexpr uint32_t kNumVectors = BLOCK_M * kNumRoutes * kNumRoutes / VL_b32;
    vector_float comb_values[kNumVectors];
    vector_float reduced_values[kNumVectors];
    #pragma unroll
    for (uint32_t vector_idx = 0; vector_idx < kNumVectors; ++ vector_idx)
        asc_loadalign(comb_values[vector_idx], ub_comb_logits + vector_idx * VL_b32);

    // row_max_low[0..7] = [t0_r0, t0_r2, t1_r0, t1_r2, ...]
    // row_max_high[0..7] = [t0_r1, t0_r3, t1_r1, t1_r3, ...]
    // Gather broadcasts both maxima over eight lanes; select assigns each row's four lanes its maximum
    #pragma unroll
    for (uint32_t vector_idx = 0; vector_idx < kNumVectors; ++ vector_idx) {
        vector_float row_max_low, row_max_high;
        vector_uint32_t row_max_broadcast_low, row_max_broadcast_high;
        asc_reduce_max_datablock(row_max_low, comb_values[vector_idx], row_mask_b32_low);
        asc_gather(row_max_broadcast_low, reinterpret_cast<vector_uint32_t&>(row_max_low), row_max_broadcast_idx);
        asc_reduce_max_datablock(row_max_high, comb_values[vector_idx], row_mask_b32_high);
        asc_gather(row_max_broadcast_high, reinterpret_cast<vector_uint32_t&>(row_max_high), row_max_broadcast_idx);
        asc_select(reduced_values[vector_idx], reinterpret_cast<vector_float&>(row_max_broadcast_low),
                   reinterpret_cast<vector_float&>(row_max_broadcast_high), row_mask_b32_low);
    }

    // Exponentiate logits after subtracting each row maximum
    #pragma unroll
    for (uint32_t vector_idx = 0; vector_idx < kNumVectors; ++ vector_idx)
        asc_exp_sub(comb_values[vector_idx], comb_values[vector_idx], reduced_values[vector_idx], mask_b32);

    // First row normalization adds epsilon after division
    #pragma unroll
    for (uint32_t vector_idx = 0; vector_idx < kNumVectors; ++ vector_idx)
        mix_axis_sum(reduced_values[vector_idx], comb_values[vector_idx], row_swap1_idx, row_swap2_idx, mask_b32);

    #pragma unroll
    for (uint32_t vector_idx = 0; vector_idx < kNumVectors; ++ vector_idx) {
        asc_div(comb_values[vector_idx], comb_values[vector_idx], reduced_values[vector_idx], mask_b32);
        asc_add_scalar(comb_values[vector_idx], comb_values[vector_idx], sinkhorn_eps, mask_b32);
    }

    // First column normalization adds epsilon to the denominator
    #pragma unroll
    for (uint32_t vector_idx = 0; vector_idx < kNumVectors; ++ vector_idx)
        mix_axis_sum(reduced_values[vector_idx], comb_values[vector_idx], col_swap1_idx, col_swap2_idx, mask_b32);

    #pragma unroll
    for (uint32_t vector_idx = 0; vector_idx < kNumVectors; ++ vector_idx) {
        asc_add_scalar(reduced_values[vector_idx], reduced_values[vector_idx], sinkhorn_eps, mask_b32);
        asc_div(comb_values[vector_idx], comb_values[vector_idx], reduced_values[vector_idx], mask_b32);
    }

    // Repeat row and column normalization in the original FP32 accumulation order
    for (uint32_t iteration_idx = 1; iteration_idx < num_sinkhorn_iters; ++ iteration_idx) {
        // Normalize rows
        #pragma unroll
        for (uint32_t vector_idx = 0; vector_idx < kNumVectors; ++ vector_idx)
            mix_axis_sum(reduced_values[vector_idx], comb_values[vector_idx], row_swap1_idx, row_swap2_idx, mask_b32);

        #pragma unroll
        for (uint32_t vector_idx = 0; vector_idx < kNumVectors; ++ vector_idx) {
            asc_add_scalar(reduced_values[vector_idx], reduced_values[vector_idx], sinkhorn_eps, mask_b32);
            asc_div(comb_values[vector_idx], comb_values[vector_idx], reduced_values[vector_idx], mask_b32);
        }

        // Normalize columns
        #pragma unroll
        for (uint32_t vector_idx = 0; vector_idx < kNumVectors; ++ vector_idx)
            mix_axis_sum(reduced_values[vector_idx], comb_values[vector_idx], col_swap1_idx, col_swap2_idx, mask_b32);

        #pragma unroll
        for (uint32_t vector_idx = 0; vector_idx < kNumVectors; ++ vector_idx) {
            asc_add_scalar(reduced_values[vector_idx], reduced_values[vector_idx], sinkhorn_eps, mask_b32);
            asc_div(comb_values[vector_idx], comb_values[vector_idx], reduced_values[vector_idx], mask_b32);
        }
    }

    // Store combination coefficients
    #pragma unroll
    for (uint32_t vector_idx = 0; vector_idx < kNumVectors; ++ vector_idx)
        asc_storealign(ub_comb + vector_idx * VL_b32, comb_values[vector_idx], mask_b32);
}

// Apply HC normalization and the learned affine transform, preserving separate FP32 rounding
__simd_callee__ inline void mix_logits(vector_float& value, vector_float rms_scale,
                                       vector_float scale, vector_float base, vector_bool mask_b32) {
    asc_mul(value, value, rms_scale, mask_b32);
    asc_mul(value, value, scale, mask_b32);
    asc_add(value, value, base, mask_b32);
}

// Sigmoid: 1 / (1 + exp(-value))
__simd_callee__ inline void sigmoid(vector_float& value, vector_float zero,
                                    vector_float one, vector_bool mask_b32) {
    asc_exp_sub(value, zero, value, mask_b32);
    asc_add_scalar(value, value, 1.0f, mask_b32);
    asc_div(value, one, value, mask_b32);
}

// Normalize Prenorm sums, produce pre/post coefficients, and leave comb logits for Sinkhorn
template <uint32_t SHAPE_H, uint32_t SHAPE_N, uint32_t BLOCK_M, uint32_t kNumRoutes>
__simd_callee__ inline void pre_post_and_comb_logits(__ubuf__ float* ub_prenorm, __ubuf__ float* ub_hc_sqr_sum,
                                                     __ubuf__ float* ub_scales, __ubuf__ float* ub_bases,
                                                     __ubuf__ float* ub_pre, __ubuf__ float* ub_post,
                                                     __ubuf__ float* ub_comb_logits,
                                                     float hc_norm_eps, float hc_pre_eps, float hc_post_scale) {
    constexpr uint32_t VL_b32 = sizeof(vector_float) / sizeof(float);

    static_assert(kNumRoutes == 4 and BLOCK_M * kNumRoutes <= VL_b32);
    static_assert(SHAPE_N == 2 * kNumRoutes + kNumRoutes * kNumRoutes);

    const auto mask_b32 = asc_create_mask_b32(PAT_ALL);
    uint32_t actual_m = BLOCK_M;
    uint32_t num_coeffs = BLOCK_M * kNumRoutes;
    const auto m_mask_b32 = asc_update_mask_b32(actual_m);
    const auto coeff_mask_b32 = asc_update_mask_b32(num_coeffs);

    vector_float one, zero;
    vector_float inverse_hc_width;
    vector_float hc_rms_scale_all, bases_all;
    vector_int32_t lane_idx;
    vector_int32_t route_index_mask, comb_index_mask;
    vector_int32_t route_idx, comb_idx;
    asc_duplicate_scalar(one, 1.0f, mask_b32);
    asc_duplicate_scalar(zero, 0.0f, mask_b32);
    asc_duplicate_scalar(inverse_hc_width, 1.0f / float(kNumRoutes * SHAPE_H), mask_b32);

    // Compute one HC RMS scale per token
    asc_loadalign(hc_rms_scale_all, ub_hc_sqr_sum);
    asc_mul(hc_rms_scale_all, hc_rms_scale_all, inverse_hc_width, m_mask_b32);
    asc_add_scalar(hc_rms_scale_all, hc_rms_scale_all, hc_norm_eps, m_mask_b32);
    asc_sqrt(hc_rms_scale_all, hc_rms_scale_all, m_mask_b32);
    asc_div(hc_rms_scale_all, one, hc_rms_scale_all, m_mask_b32);

    // Prenorm rows and bases use [pre(4), post(4), row-major comb(16)]
    // route_idx = lane % 4; comb_idx = lane % 16
    asc_loadalign(bases_all, ub_bases);
    asc_arange(lane_idx, 0);
    asc_duplicate_scalar(route_index_mask, kNumRoutes - 1, mask_b32);
    asc_duplicate_scalar(comb_index_mask, kNumRoutes * kNumRoutes - 1, mask_b32);
    asc_and(route_idx, lane_idx, route_index_mask, mask_b32);
    asc_and(comb_idx, lane_idx, comb_index_mask, mask_b32);

    // Pre/post output lane = 4*t + r: pre_idx = 24*t + r, post_idx = pre_idx + 4
    // Gather RMS[t] for all four routes; bases repeat [0..3] and [4..7] per token
    vector_int32_t m_idx, prenorm_m_offset;
    vector_int32_t pre_idx, post_idx, post_base_idx;
    vector_float hc_rms_scale;
    vector_float pre_scale, post_scale;
    vector_float pre_base, post_base;
    vector_float pre_value, post_value;
    asc_shiftright_scalar(m_idx, lane_idx, __builtin_ctz(kNumRoutes), mask_b32);
    asc_mul_scalar(prenorm_m_offset, m_idx, int32_t(SHAPE_N), mask_b32);
    asc_add(pre_idx, prenorm_m_offset, route_idx, mask_b32);
    asc_add_scalar(post_idx, pre_idx, kNumRoutes, mask_b32);

    asc_gather(hc_rms_scale, hc_rms_scale_all, reinterpret_cast<vector_uint32_t&>(m_idx));
    asc_gather(pre_value, ub_prenorm, reinterpret_cast<vector_uint32_t&>(pre_idx), coeff_mask_b32);
    asc_gather(post_value, ub_prenorm, reinterpret_cast<vector_uint32_t&>(post_idx), coeff_mask_b32);
    asc_loadalign_brc_elem(pre_scale, ub_scales);
    asc_loadalign_brc_elem(post_scale, ub_scales + 1);
    asc_gather(pre_base, bases_all, reinterpret_cast<vector_uint32_t&>(route_idx));
    asc_add_scalar(post_base_idx, route_idx, kNumRoutes, mask_b32);
    asc_gather(post_base, bases_all, reinterpret_cast<vector_uint32_t&>(post_base_idx));

    mix_logits(pre_value, hc_rms_scale, pre_scale, pre_base, mask_b32);
    mix_logits(post_value, hc_rms_scale, post_scale, post_base, mask_b32);
    sigmoid(pre_value, zero, one, mask_b32);
    sigmoid(post_value, zero, one, mask_b32);
    asc_add_scalar(pre_value, pre_value, hc_pre_eps, mask_b32);
    asc_mul_scalar(post_value, post_value, hc_post_scale, mask_b32);

    asc_storealign(ub_pre, pre_value, coeff_mask_b32);
    asc_storealign(ub_post, post_value, coeff_mask_b32);

    // Compute combination logits for Sinkhorn
    vector_int32_t comb_base_idx;
    vector_float comb_base, comb_scale;
    asc_add_scalar(comb_base_idx, comb_idx, 2 * kNumRoutes, mask_b32);
    asc_gather(comb_base, bases_all, reinterpret_cast<vector_uint32_t&>(comb_base_idx));
    asc_loadalign_brc_elem(comb_scale, ub_scales + 2);

    // Each vector holds four row-major 4x4 matrices: m = 4*vector_idx + lane/16
    // Gather prenorm[24*m + 8 + lane%16]; RMS[m] broadcasts across the token's 16 lanes
    #pragma unroll
    for (int32_t vector_idx = 0; vector_idx < BLOCK_M * kNumRoutes * kNumRoutes / VL_b32; ++ vector_idx) {
        vector_int32_t m_idx, prenorm_m_offset;
        vector_int32_t prenorm_idx;
        vector_float comb_value, hc_rms_scale;
        asc_shiftright_scalar(m_idx, lane_idx, __builtin_ctz(kNumRoutes * kNumRoutes), mask_b32);
        asc_add_scalar(m_idx, m_idx, vector_idx * (VL_b32 / (kNumRoutes * kNumRoutes)), mask_b32);
        asc_mul_scalar(prenorm_m_offset, m_idx, int32_t(SHAPE_N), mask_b32);
        asc_add_scalar(prenorm_idx, prenorm_m_offset, 2 * kNumRoutes, mask_b32);
        asc_add(prenorm_idx, prenorm_idx, comb_idx, mask_b32);

        asc_gather(comb_value, ub_prenorm, reinterpret_cast<vector_uint32_t&>(prenorm_idx), mask_b32);
        asc_gather(hc_rms_scale, hc_rms_scale_all, reinterpret_cast<vector_uint32_t&>(m_idx));

        mix_logits(comb_value, hc_rms_scale, comb_scale, comb_base, mask_b32);

        asc_storealign(ub_comb_logits + vector_idx * VL_b32, comb_value, mask_b32);
    }
}

template <uint32_t SHAPE_H, uint32_t SHAPE_N, uint32_t BLOCK_M, uint32_t kNumRoutes>
__simd_vf__ inline void vf_process_mix_coeff(__ubuf__ float* ub_prenorm, __ubuf__ float* ub_hc_sqr_sum,
                                             __ubuf__ float* ub_scales, __ubuf__ float* ub_bases,
                                             __ubuf__ float* ub_pre, __ubuf__ float* ub_post,
                                             __ubuf__ float* ub_comb_logits, __ubuf__ float* ub_comb,
                                             float hc_norm_eps, float hc_pre_eps, float hc_post_scale,
                                             float sinkhorn_eps, uint32_t num_sinkhorn_iters) {
    pre_post_and_comb_logits<SHAPE_H, SHAPE_N, BLOCK_M, kNumRoutes>(
        ub_prenorm, ub_hc_sqr_sum, ub_scales, ub_bases, ub_pre, ub_post, ub_comb_logits,
        hc_norm_eps, hc_pre_eps, hc_post_scale);

    // Make combination-logit stores visible to Sinkhorn loads
    asc_mem_bar(VST_VLD);
    mix_sinkhorn<BLOCK_M, kNumRoutes>(
        ub_comb_logits, ub_comb, sinkhorn_eps, num_sinkhorn_iters);
}

// Convert square sums to scale / sqrt(mean square + eps) in registers
template <uint32_t SHAPE_H>
__simd_callee__ inline void get_rmsnorm_scale(vector_float& rms_scale, float rmsnorm_eps,
                                              float rmsnorm_scale, vector_bool mask_b32) {
    vector_float one;
    asc_duplicate_scalar(one, 1.0f, mask_b32);
    asc_mul_scalar(rms_scale, rms_scale, 1.0f / SHAPE_H, mask_b32);
    asc_add_scalar(rms_scale, rms_scale, rmsnorm_eps, mask_b32);
    asc_sqrt(rms_scale, rms_scale, mask_b32);
    asc_div(rms_scale, one, rms_scale, mask_b32);
    asc_mul_scalar(rms_scale, rms_scale, rmsnorm_scale, mask_b32);
}

template <uint32_t SHAPE_H, uint32_t BLOCK_M, uint32_t kNumKSplits, uint32_t kSqrSumStride>
__simd_vf__ inline void vf_reduce_rmsnorm_scale(__ubuf__ float* ub_rms_sqr_sum,
                                                float rmsnorm_eps, float rmsnorm_scale) {
    constexpr uint32_t VL_b32 = sizeof(vector_float) / sizeof(float);

    static_assert(BLOCK_M > 0 and BLOCK_M <= VL_b32);
    uint32_t actual_m = BLOCK_M;
    const auto m_mask_b32 = asc_update_mask_b32(actual_m);

    // Accumulate K splits in order
    vector_float rms_scale, partial;
    asc_loadalign(rms_scale, ub_rms_sqr_sum);
    for (uint32_t split_idx = 1; split_idx < kNumKSplits; ++ split_idx) {
        asc_loadalign(partial, ub_rms_sqr_sum + split_idx * kSqrSumStride);
        asc_add(rms_scale, rms_scale, partial, m_mask_b32);
    }

    // Compute and store RMS scales in split 0
    get_rmsnorm_scale<SHAPE_H>(
        rms_scale, rmsnorm_eps, rmsnorm_scale, m_mask_b32);

    asc_storealign(ub_rms_sqr_sum, rms_scale, m_mask_b32);
}

// Normalize to BF16, then quantize the rounded values
template <uint32_t BLOCK_H, uint32_t SUB_BLOCK_H, uint32_t kSFStride = 1, bool kStoreFP8 = true>
__simd_vf__ inline void vf_norm_quant_row(__ubuf__ bfloat16_t* ub_y_bf16, __ubuf__ float* ub_rms_scale,
                                          __ubuf__ bfloat16_t* ub_rmsnorm_weight, __ubuf__ float8_e4m3_t* ub_y_fp8,
                                          __ubuf__ uint16_t* ub_y_sf, uint16_t sf_offset = 0) {
    constexpr uint32_t VL_b16 = sizeof(vector_bfloat16_t) / sizeof(bfloat16_t);

    constexpr uint32_t kQuantBlockH = MX_SF_DIVISOR / 2;  // Two H32 scales per packed SF
    constexpr uint32_t kNumQuantBlocksPerPair = 2 * SUB_BLOCK_H / kQuantBlockH;
    static_assert(SUB_BLOCK_H == VL_b16 and BLOCK_H % (2 * SUB_BLOCK_H) == 0 and BLOCK_H / kQuantBlockH <= VL_b16);

    const auto mask_b8 = asc_create_mask_b8(PAT_ALL);
    const auto mask_b16 = asc_create_mask_b16(PAT_ALL);
    const auto mask_b32 = asc_create_mask_b32(PAT_ALL);

    vector_float rms_scale;
    asc_loadalign_brc(rms_scale, ub_rms_scale);

    // Lane l collects H32 group l: pair = l / 8, group within pair = l % 8
    vector_int16_t lane_idx;
    vector_uint16_t quant_block_idx_in_pair, h_pair_idx, index_mask, abs_bits_mask, quant_abs_max;
    if constexpr (kStoreFP8) {
        asc_arange(lane_idx, int16_t(0));
        asc_duplicate_scalar(index_mask, uint16_t(kNumQuantBlocksPerPair - 1), mask_b16);
        asc_and(quant_block_idx_in_pair, reinterpret_cast<vector_uint16_t&>(lane_idx), index_mask, mask_b16);
        asc_shiftright_scalar(h_pair_idx, reinterpret_cast<vector_uint16_t&>(lane_idx), uint16_t(__builtin_ctz(kNumQuantBlocksPerPair)), mask_b16);
        asc_duplicate_scalar(abs_bits_mask, uint16_t(0x7fff), mask_b16);
        asc_duplicate_scalar(quant_abs_max, uint16_t(0), mask_b16);
    }

    // Keep BF16 Y deinterleaved: [0] = H even, [1] = H odd
    vector_bfloat16_t norm_y_bf16[BLOCK_H / (2 * SUB_BLOCK_H)][2];
    #pragma unroll
    for (uint32_t h_idx = 0; h_idx < BLOCK_H; h_idx += 2 * SUB_BLOCK_H) {
        vector_bfloat16_t input_y[2], input_weight[2];
        asc_loadalign_deintlv(input_y[0], input_y[1], ub_y_bf16 + h_idx);
        asc_loadalign_deintlv(input_weight[0], input_weight[1], ub_rmsnorm_weight + h_idx);
        #pragma unroll
        for (uint32_t parity_idx = 0; parity_idx < 2; ++ parity_idx) {
            vector_float y_low, y_high, weight_low, weight_high;
            vector_bfloat16_t result_low, result_high;
            asc_bfloat162float(y_low, input_y[parity_idx], mask_b16);
            asc_bfloat162float_v2(y_high, input_y[parity_idx], mask_b16);
            asc_bfloat162float(weight_low, input_weight[parity_idx], mask_b16);
            asc_bfloat162float_v2(weight_high, input_weight[parity_idx], mask_b16);
            asc_mul(y_low, y_low, rms_scale, mask_b32);
            asc_mul(y_low, y_low, weight_low, mask_b32);
            asc_mul(y_high, y_high, rms_scale, mask_b32);
            asc_mul(y_high, y_high, weight_high, mask_b32);
            asc_float2bfloat16_rn_sat(result_low, y_low, mask_b16);
            asc_float2bfloat16_rn_sat_v2(result_high, y_high, mask_b16);
            asc_or(reinterpret_cast<vector_uint16_t&>(norm_y_bf16[h_idx / (2 * SUB_BLOCK_H)][parity_idx]),
                   reinterpret_cast<vector_uint16_t&>(result_low), reinterpret_cast<vector_uint16_t&>(result_high), mask_b16);
        }
        asc_storealign_intlv(ub_y_bf16 + h_idx, norm_y_bf16[h_idx / (2 * SUB_BLOCK_H)][0], norm_y_bf16[h_idx / (2 * SUB_BLOCK_H)][1]);

        if constexpr (kStoreFP8) {
            // One datablock spans H32 after pairing even and odd elements
            vector_uint16_t abs_even, abs_odd, pair_abs_max, pair_abs_max_broadcast;
            asc_and(abs_even, reinterpret_cast<vector_uint16_t&>(norm_y_bf16[h_idx / (2 * SUB_BLOCK_H)][0]), abs_bits_mask, mask_b16);
            asc_and(abs_odd, reinterpret_cast<vector_uint16_t&>(norm_y_bf16[h_idx / (2 * SUB_BLOCK_H)][1]), abs_bits_mask, mask_b16);
            asc_max(pair_abs_max, abs_even, abs_odd, mask_b16);
            asc_reduce_max_datablock(pair_abs_max, pair_abs_max, mask_b16);
            asc_gather(pair_abs_max_broadcast, pair_abs_max, quant_block_idx_in_pair);
            vector_bool h_pair_mask_b16;
            asc_eq_scalar(h_pair_mask_b16, h_pair_idx, uint16_t(h_idx / (2 * SUB_BLOCK_H)), mask_b16);
            asc_select(quant_abs_max, pair_abs_max_broadcast, quant_abs_max, h_pair_mask_b16);
        }
    }

    if constexpr (kStoreFP8) {
        // Widen dense BF16 amax lanes to two contiguous FP32 vectors
        vector_uint16_t zero;
        vector_float amax[2];
        asc_duplicate_scalar(zero, uint16_t(0), mask_b16);
        asc_intlv(reinterpret_cast<vector_uint16_t&>(amax[0]), reinterpret_cast<vector_uint16_t&>(amax[1]), zero, quant_abs_max);

        // Match TileKernels: clamp amax, round scale upward, then cast inverse scale to BF16
        vector_uint32_t scale_exponents[2], exponent_bias_sum;
        vector_bfloat16_t inverse_scales[2];
        asc_duplicate_scalar(exponent_bias_sum, uint32_t(254), mask_b32);
        #pragma unroll
        for (uint32_t vector_idx = 0; vector_idx < 2; ++ vector_idx) {
            asc_max_scalar(amax[vector_idx], amax[vector_idx], 1e-4f, mask_b32);
            asc_mul_scalar(amax[vector_idx], amax[vector_idx], 1.0f / 448.0f, mask_b32);
            asc_add_scalar(scale_exponents[vector_idx], reinterpret_cast<vector_uint32_t&>(amax[vector_idx]), uint32_t(-1), mask_b32);
            asc_shiftright_scalar(scale_exponents[vector_idx], scale_exponents[vector_idx], uint32_t(23), mask_b32);
            asc_add_scalar(scale_exponents[vector_idx], scale_exponents[vector_idx], uint32_t(1), mask_b32);

            vector_uint32_t inverse_bits;
            asc_sub(inverse_bits, exponent_bias_sum, scale_exponents[vector_idx], mask_b32);
            asc_shiftleft_scalar(inverse_bits, inverse_bits, uint32_t(23), mask_b32);
            asc_float2bfloat16_rn_sat(inverse_scales[vector_idx], reinterpret_cast<vector_float&>(inverse_bits), mask_b16);
        }

        // Compact the low halfwords back into H32 group order
        vector_uint16_t scale_exponent, inverse_scale_bits, unused;
        asc_deintlv(scale_exponent, unused,
                    reinterpret_cast<vector_uint16_t&>(scale_exponents[0]), reinterpret_cast<vector_uint16_t&>(scale_exponents[1]));
        asc_deintlv(inverse_scale_bits, unused,
                    reinterpret_cast<vector_uint16_t&>(inverse_scales[0]), reinterpret_cast<vector_uint16_t&>(inverse_scales[1]));
        // Pack two exponents per uint16 and scatter to the requested SF stride
        vector_uint8_t packed_sf;
        asc_pack_to_low(packed_sf, scale_exponent);
        vector_int16_t sf_idx;
        asc_arange(sf_idx, int16_t(0));
        asc_mul_scalar(sf_idx, sf_idx, int16_t(kSFStride), mask_b16);
        asc_add_scalar(sf_idx, sf_idx, int16_t(sf_offset), mask_b16);
        uint32_t num_sf_pairs = BLOCK_H / MX_SF_DIVISOR;
        const auto sf_pair_mask_b16 = asc_update_mask_b16(num_sf_pairs);
        asc_scatter(ub_y_sf, reinterpret_cast<vector_uint16_t&>(packed_sf), reinterpret_cast<vector_uint16_t&>(sf_idx), sf_pair_mask_b16);

        // inverse_scale_idx = h_idx/32 + lane/16 for both even and odd H vectors
        vector_uint16_t inverse_scale_base_idx;
        asc_shiftright_scalar(inverse_scale_base_idx, reinterpret_cast<vector_uint16_t&>(lane_idx), uint16_t(__builtin_ctz(kQuantBlockH / 2)), mask_b16);

        #pragma unroll
        for (uint32_t h_idx = 0; h_idx < BLOCK_H; h_idx += 2 * SUB_BLOCK_H) {
            vector_bfloat16_t inverse_scale;
            vector_uint16_t inverse_scale_idx;
            asc_add_scalar(inverse_scale_idx, inverse_scale_base_idx, uint16_t(h_idx / kQuantBlockH), mask_b16);
            asc_gather(inverse_scale, reinterpret_cast<vector_bfloat16_t&>(inverse_scale_bits), inverse_scale_idx);

            vector_bfloat16_t scaled_y_even, scaled_y_odd;
            vector_float y[4];
            asc_mul(scaled_y_even, norm_y_bf16[h_idx / (2 * SUB_BLOCK_H)][0], inverse_scale, mask_b16);
            asc_mul(scaled_y_odd, norm_y_bf16[h_idx / (2 * SUB_BLOCK_H)][1], inverse_scale, mask_b16);
            asc_bfloat162float(y[0], scaled_y_even, mask_b16);
            asc_bfloat162float(y[1], scaled_y_odd, mask_b16);
            asc_bfloat162float_v2(y[2], scaled_y_even, mask_b16);
            asc_bfloat162float_v2(y[3], scaled_y_odd, mask_b16);

            vector_fp8_e4m3fn_t fp8[4];
            vector_uint8_t packed;
            asc_float2e4m3_rn_sat(fp8[0], y[0], mask_b8);
            asc_float2e4m3_rn_sat_v2(fp8[1], y[1], mask_b8);
            asc_float2e4m3_rn_sat_v3(fp8[2], y[2], mask_b8);
            asc_float2e4m3_rn_sat_v4(fp8[3], y[3], mask_b8);
            asc_or(packed, reinterpret_cast<vector_uint8_t&>(fp8[0]), reinterpret_cast<vector_uint8_t&>(fp8[1]), mask_b8);
            asc_or(packed, packed, reinterpret_cast<vector_uint8_t&>(fp8[2]), mask_b8);
            asc_or(packed, packed, reinterpret_cast<vector_uint8_t&>(fp8[3]), mask_b8);
            asc_storealign(reinterpret_cast<__ubuf__ uint8_t*>(ub_y_fp8 + h_idx), packed, mask_b8);
        }
    }
}

template <
    uint32_t BLOCK_M, uint32_t BLOCK_H, uint32_t SUB_BLOCK_H,
    uint32_t kSFStride, bool kStoreFP8, uint32_t kSFRowStride,
    bool kNormalize = true, bool kPrepareQuant = true, bool kApplyQuant = true
>
__simd_vf__ inline void vf_norm_quant_block(__ubuf__ bfloat16_t* y, __ubuf__ float* scale,
                                          __ubuf__ bfloat16_t* weight, __ubuf__ float8_e4m3_t* fp8,
                                          __ubuf__ uint16_t* sf, __ubuf__ uint16_t* amax) {
    constexpr uint32_t kQuantBlockH = MX_SF_DIVISOR / 2;
    constexpr uint32_t kNumGroups = 2 * SUB_BLOCK_H / kQuantBlockH;
    static_assert(BLOCK_M * kNumGroups <= sizeof(vector_uint16_t) / sizeof(uint16_t));
    const auto mask_b8 = asc_create_mask_b8(PAT_ALL);
    const auto mask_b16 = asc_create_mask_b16(PAT_ALL);
    const auto mask_b32 = asc_create_mask_b32(PAT_ALL);
    vector_int16_t lane_idx;
    asc_arange(lane_idx, int16_t(0));
    auto unsigned_lane_idx = reinterpret_cast<vector_uint16_t&>(lane_idx);
    vector_uint16_t group_idx, row_group_idx, group_mask, abs_mask, inverse_base_idx, sf_base_idx;
    asc_duplicate_scalar(group_mask, uint16_t(kNumGroups - 1), mask_b16);
    asc_and(group_idx, unsigned_lane_idx, group_mask, mask_b16);
    asc_shiftright_scalar(row_group_idx, unsigned_lane_idx, uint16_t(__builtin_ctz(kNumGroups)), mask_b16);
    asc_duplicate_scalar(abs_mask, uint16_t(0x7fff), mask_b16);
    asc_shiftright_scalar(inverse_base_idx, unsigned_lane_idx, uint16_t(__builtin_ctz(kQuantBlockH / 2)), mask_b16);
    vector_uint16_t sf_row_idx, sf_col_idx;
    asc_shiftright_scalar(sf_row_idx, unsigned_lane_idx, uint16_t(__builtin_ctz(kNumGroups / 2)), mask_b16);
    asc_duplicate_scalar(group_mask, uint16_t(kNumGroups / 2 - 1), mask_b16);
    asc_and(sf_col_idx, unsigned_lane_idx, group_mask, mask_b16);
    asc_mul_scalar(sf_row_idx, sf_row_idx, uint16_t(kSFRowStride), mask_b16);
    asc_mul_scalar(sf_col_idx, sf_col_idx, uint16_t(kSFStride), mask_b16);
    asc_add(sf_base_idx, sf_row_idx, sf_col_idx, mask_b16);
    uint32_t num_sf_words = BLOCK_M * kNumGroups / 2;
    const auto sf_mask_b16 = asc_update_mask_b16(num_sf_words);

    #pragma unroll 1
    for (uint32_t h_idx = 0; h_idx < BLOCK_H; h_idx += 2 * SUB_BLOCK_H) {
        // Fused passes reuse one slice; split passes retain every H slice between VF calls.
        auto amax_h = amax + ((kNormalize and kApplyQuant) ? 0 : h_idx / (2 * SUB_BLOCK_H) * BLOCK_M * kDataBlockBytes / sizeof(uint16_t));
        if constexpr (kNormalize) {
            vector_bfloat16_t weight_bf16[2];
            vector_float weight_fp32[2][2];
            asc_loadalign_deintlv(weight_bf16[0], weight_bf16[1], weight + h_idx);
            #pragma unroll
            for (uint32_t parity_idx = 0; parity_idx < 2; ++ parity_idx) {
                asc_bfloat162float(weight_fp32[parity_idx][0], weight_bf16[parity_idx], mask_b16);
                asc_bfloat162float_v2(weight_fp32[parity_idx][1], weight_bf16[parity_idx], mask_b16);
            }
            uint32_t num_groups = kNumGroups;
            const auto group_mask_b16 = asc_update_mask_b16(num_groups);
            #pragma unroll 1
            for (uint32_t row_idx = 0; row_idx < BLOCK_M; ++ row_idx) {
                vector_bfloat16_t input_y[2], norm_y[2];
                vector_float rms_scale;
                asc_loadalign_brc(rms_scale, scale + row_idx);
                asc_loadalign_deintlv(input_y[0], input_y[1], y + row_idx * BLOCK_H + h_idx);
                #pragma unroll
                for (uint32_t parity_idx = 0; parity_idx < 2; ++ parity_idx) {
                    vector_float lo, hi;
                    vector_bfloat16_t bf16_lo, bf16_hi;
                    asc_bfloat162float(lo, input_y[parity_idx], mask_b16);
                    asc_bfloat162float_v2(hi, input_y[parity_idx], mask_b16);
                    asc_mul(lo, lo, rms_scale, mask_b32);
                    asc_mul(hi, hi, rms_scale, mask_b32);
                    asc_mul(lo, lo, weight_fp32[parity_idx][0], mask_b32);
                    asc_mul(hi, hi, weight_fp32[parity_idx][1], mask_b32);
                    asc_float2bfloat16_rn_sat(bf16_lo, lo, mask_b16);
                    asc_float2bfloat16_rn_sat_v2(bf16_hi, hi, mask_b16);
                    asc_or(reinterpret_cast<vector_uint16_t&>(norm_y[parity_idx]), reinterpret_cast<vector_uint16_t&>(bf16_lo), reinterpret_cast<vector_uint16_t&>(bf16_hi), mask_b16);
                }
                asc_storealign_intlv(y + row_idx * BLOCK_H + h_idx, norm_y[0], norm_y[1]);
                if constexpr (kStoreFP8) {
                    vector_uint16_t even_abs, odd_abs, group_max;
                    asc_and(even_abs, reinterpret_cast<vector_uint16_t&>(norm_y[0]), abs_mask, mask_b16);
                    asc_and(odd_abs, reinterpret_cast<vector_uint16_t&>(norm_y[1]), abs_mask, mask_b16);
                    asc_max(group_max, even_abs, odd_abs, mask_b16);
                    asc_reduce_max_datablock(group_max, group_max, mask_b16);
                    asc_storealign(amax_h + row_idx * (kDataBlockBytes / sizeof(uint16_t)), group_max, group_mask_b16);
                }
            }
        }
        if constexpr (kStoreFP8 and (kPrepareQuant or kApplyQuant)) {
            vector_uint16_t inverse_bits;
            if constexpr (kPrepareQuant) {
                vector_uint16_t quant_abs_max;
                asc_mem_bar(VST_VLD);
                vector_uint16_t amax_idx;
                asc_mul_scalar(amax_idx, row_group_idx, uint16_t(kDataBlockBytes / sizeof(uint16_t)), mask_b16);
                asc_add(amax_idx, amax_idx, group_idx, mask_b16);
                uint32_t num_amax = BLOCK_M * kNumGroups;
                asc_gather(quant_abs_max, amax_h, amax_idx, asc_update_mask_b16(num_amax));
                constexpr uint16_t kMaxBits = __builtin_bit_cast(uint32_t, 448.0f) >> 16;
                constexpr uint16_t kOffset = (kMaxBits >> 7) - 127;
                constexpr uint16_t kBias = 127 - (kMaxBits & 127);
                constexpr uint16_t kMinExponent = ((__builtin_bit_cast(uint32_t, 1e-4f * (1.0f / 448.0f)) - 1) >> 23) + 1;
                constexpr uint16_t kMinBits = ((kMinExponent + kOffset) << 7) | (kMaxBits & 127);
                vector_uint16_t exponent, bias, exceptional_bias;
                vector_bool inf_nan;
                asc_ge_scalar(inf_nan, quant_abs_max, uint16_t(0x7f80), mask_b16);
                asc_duplicate_scalar(bias, uint16_t(kBias - (kOffset << 7)), mask_b16);
                asc_duplicate_scalar(exceptional_bias, uint16_t(127), mask_b16);
                asc_select(bias, exceptional_bias, bias, inf_nan);
                asc_max_scalar(exponent, quant_abs_max, kMinBits, mask_b16);
                asc_add(exponent, exponent, bias, mask_b16);
                asc_shiftright_scalar(exponent, exponent, uint16_t(7), mask_b16);
                asc_duplicate_scalar(bias, uint16_t(254), mask_b16);
                asc_sub(inverse_bits, bias, exponent, mask_b16);
                asc_shiftleft_scalar(inverse_bits, inverse_bits, uint16_t(7), mask_b16);
                vector_uint8_t packed_sf;
                vector_uint16_t sf_idx;
                asc_pack_to_low(packed_sf, exponent);
                asc_add_scalar(sf_idx, sf_base_idx, uint16_t(h_idx / MX_SF_DIVISOR * kSFStride), mask_b16);
                asc_scatter(sf, reinterpret_cast<vector_uint16_t&>(packed_sf), sf_idx, sf_mask_b16);
                // Reuse each consumed amax slice for its inverse scales
                if constexpr (not kApplyQuant) {
                    static_assert(sizeof(vector_uint16_t) <= BLOCK_M * kDataBlockBytes);
                    asc_storealign(amax_h, inverse_bits, mask_b16);
                }
            } else {
                asc_mem_bar(VST_VLD);
                asc_loadalign(inverse_bits, amax_h);
            }
            if constexpr (kApplyQuant) {
                #pragma unroll 1
                for (uint32_t row_idx = 0; row_idx < BLOCK_M; ++ row_idx) {
                    vector_bfloat16_t norm_y[2], inverse_scale, scaled[2];
                    vector_uint16_t inverse_idx;
                    asc_loadalign_deintlv(norm_y[0], norm_y[1], y + row_idx * BLOCK_H + h_idx);
                    asc_add_scalar(inverse_idx, inverse_base_idx, uint16_t(row_idx * kNumGroups), mask_b16);
                    asc_gather(inverse_scale, reinterpret_cast<vector_bfloat16_t&>(inverse_bits), inverse_idx);
                    asc_mul(scaled[0], norm_y[0], inverse_scale, mask_b16);
                    asc_mul(scaled[1], norm_y[1], inverse_scale, mask_b16);
                    vector_float value[4];
                    asc_bfloat162float(value[0], scaled[0], mask_b16);
                    asc_bfloat162float(value[1], scaled[1], mask_b16);
                    asc_bfloat162float_v2(value[2], scaled[0], mask_b16);
                    asc_bfloat162float_v2(value[3], scaled[1], mask_b16);
                    vector_fp8_e4m3fn_t quantized[4];
                    vector_uint8_t packed;
                    asc_float2e4m3_rn_sat(quantized[0], value[0], mask_b8);
                    asc_float2e4m3_rn_sat_v2(quantized[1], value[1], mask_b8);
                    asc_float2e4m3_rn_sat_v3(quantized[2], value[2], mask_b8);
                    asc_float2e4m3_rn_sat_v4(quantized[3], value[3], mask_b8);
                    asc_or(packed, reinterpret_cast<vector_uint8_t&>(quantized[0]), reinterpret_cast<vector_uint8_t&>(quantized[1]), mask_b8);
                    asc_or(packed, packed, reinterpret_cast<vector_uint8_t&>(quantized[2]), mask_b8);
                    asc_or(packed, packed, reinterpret_cast<vector_uint8_t&>(quantized[3]), mask_b8);
                    asc_storealign(reinterpret_cast<__ubuf__ uint8_t*>(fp8 + row_idx * BLOCK_H + h_idx), packed, mask_b8);
                }
            }
        }
    }
}

} // namespace deep_gemm::mega_mhc
