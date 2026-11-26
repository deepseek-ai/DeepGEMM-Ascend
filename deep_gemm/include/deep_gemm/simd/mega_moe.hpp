#pragma once

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/layout/mega_moe.hpp>

#include <c_api/asc_simd.h>

namespace deep_gemm::mega_moe {

// Expand FP4 to FP8 divided by 64, preserving element order; weight SFs compensate with +6
template <uint32_t kNumElements>
__simd_vf__ inline void vf_cast_fp4_to_fp8_scaled(__ubuf__ uint8_t* src, __ubuf__ uint8_t* dst) {
    constexpr uint32_t kNumPackedBytesPerVector = sizeof(vector_int8_t) / 2;
    constexpr uint32_t kNumVectors = 8;
    static_assert(kNumElements / 2 % (kNumVectors * kNumPackedBytesPerVector) == 0);
    const auto mask = asc_create_mask_b8(PAT_ALL);
    vector_int8_t shift_right, shift_left, fp8_mask;
    asc_duplicate_scalar(shift_right, static_cast<int8_t>(2));
    asc_duplicate_scalar(reinterpret_cast<vector_uint16_t&>(shift_left), uint16_t{4});  // Alternating byte shifts: 4, 0
    asc_duplicate_scalar(fp8_mask, static_cast<int8_t>(0x9c));  // Sign and shifted FP4 payload

    auto src_ptr = reinterpret_cast<__ubuf__ int8_t*>(src);
    auto dst_ptr = dst;

    for (uint32_t packed_byte_idx = 0; packed_byte_idx < kNumElements / 2; packed_byte_idx += kNumVectors * kNumPackedBytesPerVector) {
        #pragma unroll
        for (uint32_t vector_idx = 0; vector_idx < kNumVectors; ++ vector_idx) {
            vector_int8_t value;
            asc_loadalign_upsample_postupdate(value, src_ptr, kNumPackedBytesPerVector);
            asc_shiftleft(value, value, shift_left, mask);
            asc_shiftright(value, value, shift_right, mask);
            asc_and(value, value, fp8_mask, mask);
            asc_storealign_postupdate(dst_ptr, reinterpret_cast<vector_uint8_t&>(value), sizeof(vector_uint8_t), mask);
        }
    }
}

// Linear1 epilogue: SwiGLU and FP8 quantization
template <bool kApplyRouteWeight, uint32_t STORE_BLOCK_M>
__simd_vf__ inline void vf_swiglu_quant(__ubuf__ bfloat16_t* input, __ubuf__ float8_e4m3_t* output,
                                        __ubuf__ int16_t* output_sf, __ubuf__ float* route_weights,
                                        uint32_t valid_m, float activation_clamp) {
    constexpr uint32_t kNumRowsPerIteration = 2;
    constexpr uint32_t kNumPairsPerRow = (layout::BLOCK_N / 2) / MX_SF_DIVISOR;
    constexpr uint32_t kNumPairs = kNumRowsPerIteration * kNumPairsPerRow;
    static_assert(layout::BLOCK_N == 256);

    const vector_bool group_mask_b32 = asc_create_mask_b32(PAT_VL32);
    const vector_bool pair_mask_b32 = asc_create_mask_b32(PAT_VL64);
    const auto high_group_mask_b32 = asc_not(group_mask_b32, pair_mask_b32);
    vector_bfloat16_t zero_bf16;
    asc_duplicate_scalar(zero_bf16, 0);
    vector_int32_t arange_b32;
    vector_uint32_t high_group_idx;
    asc_arange(arange_b32, 0);
    asc_add_scalar(high_group_idx, reinterpret_cast<vector_uint32_t&>(arange_b32), MX_SF_DIVISOR / 2, group_mask_b32);
    vector_uint32_t exponent_mask, inv_scale_bias;
    asc_duplicate_scalar(exponent_mask, 0x7f800000u);
    asc_duplicate_scalar(inv_scale_bias, 0x7f000000u);  // (2 * 127) << 23

    auto gate_ptr = input;
    auto up_ptr = input + layout::BLOCK_N / 2;
    auto output_ptr = reinterpret_cast<__ubuf__ uint8_t*>(output);

    for (uint32_t m_idx = 0; m_idx < valid_m; m_idx += kNumRowsPerIteration) {
        // Load gate/up as FP32
        vector_float gate[kNumPairs], up[kNumPairs], value[kNumPairs];
        #pragma unroll
        for (uint32_t m_offset = 0; m_offset < kNumRowsPerIteration; ++ m_offset) {
            const uint32_t pair_idx = m_offset * kNumPairsPerRow;
            vector_bfloat16_t gate_bf16, up_bf16;
            asc_loadalign_postupdate(gate_bf16, gate_ptr, layout::BLOCK_N);
            asc_loadalign_postupdate(up_bf16, up_ptr, layout::BLOCK_N);
            asc_intlv(reinterpret_cast<vector_bfloat16_t&>(gate[pair_idx]),
                      reinterpret_cast<vector_bfloat16_t&>(gate[pair_idx + 1]), zero_bf16, gate_bf16);
            asc_intlv(reinterpret_cast<vector_bfloat16_t&>(up[pair_idx]),
                      reinterpret_cast<vector_bfloat16_t&>(up[pair_idx + 1]), zero_bf16, up_bf16);
        }

        // Load route weights; odd tails use the allocated padding row
        vector_float route_weight[kNumRowsPerIteration];
        if constexpr (kApplyRouteWeight) {
            asc_loadalign_brc_postupdate(route_weight[0], route_weights, 1);
            asc_loadalign_brc_postupdate(route_weight[1], route_weights, 1);
        }

        // Clamp gate above and up on both sides
        #pragma unroll
        for (uint32_t pair_idx = 0; pair_idx < kNumPairs; ++ pair_idx) {
            asc_min_scalar(gate[pair_idx], gate[pair_idx], activation_clamp, pair_mask_b32);
            asc_min_scalar(up[pair_idx], up[pair_idx], activation_clamp, pair_mask_b32);
            asc_max_scalar(up[pair_idx], up[pair_idx], -activation_clamp, pair_mask_b32);
            asc_exp_sub(value[pair_idx], reinterpret_cast<vector_float&>(zero_bf16), gate[pair_idx], pair_mask_b32);
        }

        // SwiGLU: gate * up / (1 + exp(-gate))
        #pragma unroll
        for (uint32_t pair_idx = 0; pair_idx < kNumPairs; ++ pair_idx) {
            asc_add_scalar(value[pair_idx], value[pair_idx], 1.0f, pair_mask_b32);
            asc_div(value[pair_idx], gate[pair_idx], value[pair_idx], pair_mask_b32);
            asc_mul(value[pair_idx], value[pair_idx], up[pair_idx], pair_mask_b32);
            if constexpr (kApplyRouteWeight)
                asc_mul(value[pair_idx], value[pair_idx], route_weight[pair_idx / kNumPairsPerRow], pair_mask_b32);
        }

        // Reduce amax for both 32-channel groups in each SF pair
        #pragma unroll
        for (uint32_t pair_idx = 0; pair_idx < kNumPairs; ++ pair_idx) {
            const uint32_t m_offset = pair_idx / kNumPairsPerRow;
            const uint32_t sf_pair_idx = pair_idx % kNumPairsPerRow;
            vector_uint32_t bits[2];
            asc_abs(reinterpret_cast<vector_float&>(bits[0]), value[pair_idx], pair_mask_b32);
            asc_reduce_max(bits[1], bits[0], high_group_mask_b32);
            asc_reduce_max(bits[0], bits[0], group_mask_b32);
            asc_duplicate(bits[0], bits[0], pair_mask_b32);
            asc_duplicate(bits[1], bits[1], pair_mask_b32);

            // Round max(amax, 1e-4) / 448 up to a power of two
            asc_select(bits[0], bits[0], bits[1], group_mask_b32);
            asc_max_scalar(bits[0], bits[0], 0x38d1b717u, pair_mask_b32);  // FP32 bits of 1e-4
            asc_add_scalar(bits[0], bits[0], 0x001fffffu, pair_mask_b32);  // Carry when significand > 1.75
            asc_and(bits[0], bits[0], exponent_mask, pair_mask_b32);
            asc_add_scalar(reinterpret_cast<vector_int32_t&>(bits[0]),
                           reinterpret_cast<vector_int32_t&>(bits[0]), -0x04000000, pair_mask_b32);  // Subtract 8 from exponent
            asc_min_scalar(bits[0], bits[0], 0x7f000000u, pair_mask_b32);
            vector_uint32_t inv_scale_bits;
            asc_sub(inv_scale_bits, inv_scale_bias, bits[0], pair_mask_b32);
            asc_mul(value[pair_idx], value[pair_idx], reinterpret_cast<vector_float&>(inv_scale_bits), pair_mask_b32);

            // Store FP8 and UE8M0 scales
            vector_fp8_e4m3fn_t value_fp8;
            asc_float2e4m3_rn_sat(value_fp8, value[pair_idx], pair_mask_b32, ASC_DISPERSE_FIRST_QUARTER);
            asc_storealign_pack_quarter(reinterpret_cast<__ubuf__ uint32_t*>(output_ptr),
                                        reinterpret_cast<vector_uint32_t&>(value_fp8), pair_mask_b32);
            output_ptr += MX_SF_DIVISOR;

            // Pack both group scales into int16 entries indexed by SF pair and row
            asc_shiftright_scalar(bits[0], bits[0], 23, pair_mask_b32);
            asc_gather(bits[1], bits[0], high_group_idx);
            vector_uint8_t packed_sf_half, unused_sf_half;
            asc_intlv(packed_sf_half, unused_sf_half, reinterpret_cast<vector_uint8_t&>(bits[0]),
                      reinterpret_cast<vector_uint8_t&>(bits[1]));
            asc_storealign_1st(output_sf + sf_pair_idx * STORE_BLOCK_M,
                               reinterpret_cast<vector_int16_t&>(packed_sf_half), m_idx + m_offset);
        }
    }
}

template <typename rmsnorm_type>
__simd_callee__ inline void load_as_float(vector_float& value, __ubuf__ rmsnorm_type* ptr) {
    static_assert(std::is_same_v<rmsnorm_type, float> or std::is_same_v<rmsnorm_type, bfloat16_t>);
    if constexpr (std::is_same_v<rmsnorm_type, float>)
        asc_loadalign(value, ptr);
    else {
        asc_loadalign_unpack(reinterpret_cast<vector_bfloat16_t&>(value), ptr);
        asc_bfloat162float(value, reinterpret_cast<vector_bfloat16_t&>(value), asc_create_mask_b16(PAT_ALL), ASC_POSITION_EVEN);
    }
}

// Combine BF16 inputs in their fixed route order; input stride is num_channels
template <bool kRMSNorm, typename rmsnorm_type>
__simd_vf__ inline void vf_combine_reduce(__ubuf__ bfloat16_t* inputs, __ubuf__ rmsnorm_type* output,
                                          uint32_t num_inputs, uint32_t num_channels, __ubuf__ float* row_square_sum, bool is_first_chunk) {
    constexpr uint32_t kNumChannelsPerVector = sizeof(vector_bfloat16_t) / sizeof(bfloat16_t);
    const auto mask_b16 = asc_create_mask_b16(PAT_ALL);
    const auto mask_b32 = asc_create_mask_b32(PAT_ALL);
    vector_bfloat16_t zero_bf16;
    asc_duplicate_scalar(zero_bf16, 0);
    auto output_ptr = output;

    vector_float square_sum;
    if constexpr (kRMSNorm) {
        if (is_first_chunk)
            asc_duplicate_scalar(square_sum, 0.0f);
        else
            asc_loadalign(square_sum, row_square_sum);
    }

    for (uint32_t channel_idx = 0; channel_idx < num_channels; channel_idx += kNumChannelsPerVector) {
        vector_float sum0, sum1;
        asc_duplicate_scalar(sum0, 0.0f);
        asc_duplicate_scalar(sum1, 0.0f);
        auto input_ptr = inputs + channel_idx;
        for (uint32_t input_idx = 0; input_idx < num_inputs; ++ input_idx) {
            vector_bfloat16_t value;
            vector_float value0, value1;
            asc_loadalign_postupdate(value, input_ptr, num_channels);

            // Widen BF16 exactly with zero low bits
            asc_intlv(reinterpret_cast<vector_bfloat16_t&>(value0), reinterpret_cast<vector_bfloat16_t&>(value1), zero_bf16, value);
            asc_add(sum0, sum0, value0, mask_b32);
            asc_add(sum1, sum1, value1, mask_b32);
        }

        // Store output; RMS uses the rounded BF16 values
        if constexpr (std::is_same_v<rmsnorm_type, bfloat16_t>) {
            vector_bfloat16_t output0, output1;
            asc_float2bfloat16_rn(output0, sum0, mask_b32, ASC_POSITION_EVEN);
            asc_storealign_pack(reinterpret_cast<__ubuf__ float*>(output_ptr), reinterpret_cast<vector_float&>(output0), mask_b32);
            output_ptr += kNumChannelsPerVector / 2;
            asc_float2bfloat16_rn(output1, sum1, mask_b32, ASC_POSITION_EVEN);
            asc_storealign_pack(reinterpret_cast<__ubuf__ float*>(output_ptr), reinterpret_cast<vector_float&>(output1), mask_b32);
            output_ptr += kNumChannelsPerVector / 2;
            if constexpr (kRMSNorm) {
                asc_bfloat162float(sum0, output0, mask_b16, ASC_POSITION_EVEN);
                asc_bfloat162float(sum1, output1, mask_b16, ASC_POSITION_EVEN);
            }
        } else {
            asc_storealign_postupdate(output_ptr, sum0, kNumChannelsPerVector / 2, mask_b32);
            asc_storealign_postupdate(output_ptr, sum1, kNumChannelsPerVector / 2, mask_b32);
        }

        // Accumulate square sums in ascending 64-channel order
        if constexpr (kRMSNorm) {
            asc_mul(sum0, sum0, sum0, mask_b32);
            asc_add(square_sum, square_sum, sum0, mask_b32);
            asc_mul(sum1, sum1, sum1, mask_b32);
            asc_add(square_sum, square_sum, sum1, mask_b32);
        }
    }
    if constexpr (kRMSNorm)
        asc_storealign(row_square_sum, square_sum, mask_b32);
}

// Normalize the combined output in FP32.
template <uint32_t kHidden, typename rmsnorm_type>
__simd_vf__ inline void vf_rms_norm(__ubuf__ rmsnorm_type* row, __ubuf__ rmsnorm_type* weight,
                                    float rms_eps, __ubuf__ float* row_square_sum) {
    constexpr uint32_t kNumChannelsPerVector = sizeof(vector_float) / sizeof(float);
    const auto mask_b32 = asc_create_mask_b32(PAT_ALL);
    vector_float inv_rms, value, weight_value, one;
    asc_loadalign(inv_rms, row_square_sum);
    asc_reduce_sum(inv_rms, inv_rms, mask_b32);
    asc_duplicate(inv_rms, inv_rms, mask_b32);
    asc_duplicate_scalar(one, 1.0f, mask_b32);
    asc_mul_scalar(inv_rms, inv_rms, 1.0f / kHidden, mask_b32);
    asc_add_scalar(inv_rms, inv_rms, rms_eps, mask_b32);
    asc_sqrt(inv_rms, inv_rms, mask_b32);
    asc_div(inv_rms, one, inv_rms, mask_b32);
    for (uint32_t channel_idx = 0; channel_idx < kHidden; channel_idx += kNumChannelsPerVector) {
        load_as_float(value, row + channel_idx);
        load_as_float(weight_value, weight + channel_idx);
        asc_mul(value, value, inv_rms, mask_b32);
        asc_mul(value, value, weight_value, mask_b32);
        if constexpr (std::is_same_v<rmsnorm_type, float>) {
            asc_storealign(row + channel_idx, value, mask_b32);
        } else {
            vector_bfloat16_t packed;
            asc_float2bfloat16_rn(packed, value, mask_b32, ASC_POSITION_EVEN);
            asc_storealign_pack(reinterpret_cast<__ubuf__ float*>(row + channel_idx), reinterpret_cast<vector_float&>(packed), mask_b32);
        }
    }
}

} // namespace deep_gemm::mega_moe
