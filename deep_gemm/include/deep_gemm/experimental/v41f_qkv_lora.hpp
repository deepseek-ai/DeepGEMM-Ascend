#pragma once

#include <deep_gemm/common.hpp>

namespace deep_gemm::experimental {

// Pack Phase 1 (AIC + AIV) and shared scalars into Line 0 (0..63B), Phase 2 AIC + AIV0 into Line 1, then AIV1.
struct QKVLoRAArgs {
    DG_GM_PTR(float8_e4m3_t) x; DG_GM_PTR(int16_t) x_sf; DG_GM_PTR(float8_e4m3_t) wa; DG_GM_PTR(int16_t) wa_sf;
    DG_GM_PTR(bfloat16_t) qkv; DG_GM_PTR(float8_e4m3_t) wb;
    uint32_t shape_m, x_sf_stride; float eps_hi, eps_lo;
    DG_GM_PTR(float8_e4m3_t) qa; DG_GM_PTR(int16_t) qa_sf; DG_GM_PTR(int16_t) wb_sf;
    DG_GM_PTR(bfloat16_t) q; DG_GM_PTR(bfloat16_t) qw; DG_GM_PTR(bfloat16_t) kw;
    DG_GM_PTR(float) cos_sin; DG_GM_PTR(int32_t) positions; DG_GM_PTR(int32_t) slots; DG_GM_PTR(uint8_t) cache;
    uint32_t num_slots, cache_block_size, cache_page_stride_bytes;
};

} // namespace deep_gemm::experimental

#ifdef DG_DEVICE_CODE
#include <deep_gemm/ascend.hpp>
#include <deep_gemm/epilogue/operators.hpp>
#include <deep_gemm/scheduler.hpp>

namespace deep_gemm::experimental {

inline constexpr uint32_t kHidden = 5120, kQLoraDim = 1280, kKVLatentDim = 512, kRopeDim = 64, kQKVDim = kQLoraDim + kKVLatentDim;
inline constexpr uint32_t kKVCacheSlotBytes = kKVLatentDim + kKVLatentDim / MX_SF_DIVISOR * sizeof(uint16_t);

inline constexpr uint32_t kWeightFreeEventBase = 0, kWeightReadyEventBase = 4;
inline constexpr uint32_t kWeightDrainEvent = 8, kQFirstReadyEvent = 13, kQReadyEvent = 14, kQKVReadyEvent = 15;

inline constexpr PipeEvent<PIPE_MTE2, PIPE_V> kRawFull{0}, kTransformFull{6};
inline constexpr PipeEvent<PIPE_V, PIPE_MTE2> kRawEmpty{0};
inline constexpr PipeEvent<PIPE_V, PIPE_MTE3> kNZFull{0}, kTransformDone{7};
inline constexpr PipeEvent<PIPE_MTE3, PIPE_V> kNZEmpty{0};
inline constexpr PipeEvent<PIPE_MTE2, PIPE_S> kMetadataFull{0};
inline constexpr PipeEvent<PIPE_MTE3, PIPE_MTE2> kStoresDone{0};

// Pack per-core Q/KV normalization, RoPE and quantization buffers in UB
template <bool kIsQ>
struct TransformLayout {
    static constexpr uint32_t kNumRows = 16, kLatentDim = kIsQ ? kQLoraDim : kKVLatentDim;
    static constexpr uint32_t kVectorLenB32 = sizeof(vector_float) / sizeof(float);
    static constexpr uint32_t kOutputStride = kIsQ ? kQLoraDim : aligned(kKVCacheSlotBytes, kDataBlockBytes);
    static constexpr ub_ptr<bfloat16_t> kInput{kNumRows, kLatentDim};
    static constexpr ub_ptr<bfloat16_t> kNormWeight{1, kLatentDim, kInput.offset(1)};
    static constexpr ub_ptr<float> kRMSScale{kNumRows, kVectorLenB32, kNormWeight.offset(1)};
    static constexpr ub_ptr<float8_e4m3_t> kOutput{kIsQ ? kNumRows + 1 : kNumRows, kOutputStride, kRMSScale.offset(1)};
    static constexpr ub_ptr<uint16_t> kSF{kLatentDim / MX_SF_DIVISOR, kNumRows, kOutput.offset(1)};
    static constexpr ub_ptr<float> kCosSin{kNumRows, kRopeDim, kOutput.offset(1)};
    static constexpr ub_ptr<int32_t> kMetadata{2, kNumRows, kCosSin.offset(1)};
    static constexpr uint32_t kEnd = kIsQ ? kSF.offset(1) : kMetadata.offset(1);
};

// Apply RoPE to the last 64 KV channels [448..511] within the 128-element tail vector [384..511]
__simd_callee__ inline void rope(__ubuf__ bfloat16_t* input, __ubuf__ float* cos_sin) {
    constexpr uint32_t kVectorLenB16 = sizeof(vector_bfloat16_t) / sizeof(bfloat16_t);
    constexpr uint32_t kTailOffset = kKVLatentDim - kVectorLenB16, kNumRopePairs = kRopeDim / 2;
    const auto mask = asc_create_mask_b32(PAT_ALL);
    const auto mask_b16 = asc_create_mask_b16(PAT_ALL);

    // Unpack 64 pairs: even[i] = x_{2i}, odd[i] = x_{2i+1} for i in [0, 64)
    vector_bfloat16_t source;
    vector_uint16_t result_low, result_high;
    vector_float even, odd, angles, cosine, sine, negative_sine, rotated_even, rotated_odd;
    asc_loadalign(source, input + kTailOffset);
    asc_bfloat162float(even, source, mask_b16);
    asc_bfloat162float_v2(odd, source, mask_b16);

    // Rotate pair i (p = i % 32): (x_{2i} * cos_p - x_{2i+1} * sin_p, x_{2i+1} * cos_p + x_{2i} * sin_p)
    asc_loadalign(angles, cos_sin);
    vector_int32_t lane;
    vector_uint32_t index, index_mask;
    asc_arange(lane, int32_t(0));
    asc_duplicate_scalar(index_mask, uint32_t(kNumRopePairs - 1), mask);
    asc_and(index, reinterpret_cast<vector_uint32_t&>(lane), index_mask, mask);
    asc_gather(cosine, angles, index);
    asc_add_scalar(index, index, uint32_t(kNumRopePairs), mask);
    asc_gather(sine, angles, index);
    asc_mul_scalar(negative_sine, sine, -1.0f, mask);
    asc_mul(rotated_even, odd, negative_sine, mask);
    asc_mula(rotated_even, even, cosine, mask);
    asc_mul(rotated_odd, odd, cosine, mask);
    asc_mula(rotated_odd, even, sine, mask);

    // Keep unrotated prefix pairs i < 32 (channels [384..447]) and pack back to BF16
    const auto prefix_mask = asc_create_mask_b32(PAT_VL32);
    asc_select(rotated_even, even, rotated_even, prefix_mask);
    asc_select(rotated_odd, odd, rotated_odd, prefix_mask);
    asc_float2bfloat16_rn_sat(reinterpret_cast<vector_bfloat16_t&>(result_low), rotated_even, mask_b16);
    asc_float2bfloat16_rn_sat_v2(reinterpret_cast<vector_bfloat16_t&>(result_high), rotated_odd, mask_b16);
    asc_or(reinterpret_cast<vector_uint16_t&>(source), result_low, result_high, mask_b16);
    asc_storealign(input + kTailOffset, source, mask_b16);
}

// Normalize a BF16 row slice, apply KV RoPE and quantize to FP8
template <uint32_t BLOCK_H>
__simd_callee__ inline void norm_quant_row(
    __ubuf__ bfloat16_t* input, __ubuf__ float* rms_scale_ptr, __ubuf__ bfloat16_t* weight,
    __ubuf__ float8_e4m3_t* output, __ubuf__ uint16_t* output_sf, uint16_t sf_offset,
    __ubuf__ float* cos_sin, uint16_t pitch_rows = 1
) {
    constexpr uint32_t kVectorLenB16 = sizeof(vector_bfloat16_t) / sizeof(bfloat16_t);
    constexpr uint32_t kQuantBlockH = MX_SF_DIVISOR / 2;
    constexpr uint32_t kNumQuantBlocksPerPair = 2 * kVectorLenB16 / kQuantBlockH;

    const auto mask_b8 = asc_create_mask_b8(PAT_ALL);
    const auto mask_b16 = asc_create_mask_b16(PAT_ALL);
    const auto mask_b32 = asc_create_mask_b32(PAT_ALL);

    vector_float rms_scale;
    asc_loadalign_brc(rms_scale, rms_scale_ptr);

    // Map lane g in [0, BLOCK_H / 32) to MX32 group g: h_pair_idx = g / 8, quant_block_idx_in_pair = g % 8
    vector_uint16_t lane_idx, quant_block_idx_in_pair, h_pair_idx, index_mask, abs_bits_mask, quant_abs_max;
    asc_arange(reinterpret_cast<vector_int16_t&>(lane_idx), int16_t(0));
    asc_duplicate_scalar(index_mask, uint16_t(kNumQuantBlocksPerPair - 1), mask_b16);
    asc_and(quant_block_idx_in_pair, lane_idx, index_mask, mask_b16);
    asc_shiftright_scalar(h_pair_idx, lane_idx, uint16_t(__builtin_ctz(kNumQuantBlocksPerPair)), mask_b16);
    asc_duplicate_scalar(abs_bits_mask, uint16_t(0x7fff), mask_b16);
    asc_duplicate_scalar(quant_abs_max, uint16_t(0), mask_b16);

    // Deinterleave [0] = x_{2k}, [1] = x_{2k+1} and compute normalized = bf16(fp32(x) * rms_scale * fp32(w))
    vector_bfloat16_t normalized_bf16[BLOCK_H / (2 * kVectorLenB16)][2];
    #pragma unroll
    for (uint32_t h_idx = 0; h_idx < BLOCK_H; h_idx += 2 * kVectorLenB16) {
        auto& normalized = normalized_bf16[h_idx / (2 * kVectorLenB16)];
        auto& normalized_u16 = reinterpret_cast<vector_uint16_t(&)[2]>(normalized);
        vector_bfloat16_t input_pair[2], weight_pair[2];
        asc_loadalign_deintlv(input_pair[0], input_pair[1], input + h_idx);
        asc_loadalign_deintlv(weight_pair[0], weight_pair[1], weight + h_idx);
        #pragma unroll
        for (uint32_t parity_idx = 0; parity_idx < 2; ++ parity_idx) {
            vector_float val_low, val_high, weight_low, weight_high;
            vector_uint16_t result_low, result_high;
            asc_bfloat162float(val_low, input_pair[parity_idx], mask_b16);
            asc_bfloat162float_v2(val_high, input_pair[parity_idx], mask_b16);
            asc_bfloat162float(weight_low, weight_pair[parity_idx], mask_b16);
            asc_bfloat162float_v2(weight_high, weight_pair[parity_idx], mask_b16);

            asc_mul(val_low, val_low, rms_scale, mask_b32);
            asc_mul(val_low, val_low, weight_low, mask_b32);
            asc_mul(val_high, val_high, rms_scale, mask_b32);
            asc_mul(val_high, val_high, weight_high, mask_b32);
            asc_float2bfloat16_rn_sat(reinterpret_cast<vector_bfloat16_t&>(result_low), val_low, mask_b16);
            asc_float2bfloat16_rn_sat_v2(reinterpret_cast<vector_bfloat16_t&>(result_high), val_high, mask_b16);
            asc_or(normalized_u16[parity_idx], result_low, result_high, mask_b16);
        }

        // Round to BF16 before RoPE, preserving the original tail layout and arithmetic
        if constexpr (BLOCK_H == kKVLatentDim) {
            if (h_idx == kKVLatentDim - 2 * kVectorLenB16) {
                asc_storealign_intlv(input + h_idx, normalized[0], normalized[1]);
                asc_mem_bar(VST_VLD);
                rope(input, cos_sin);
                asc_mem_bar(VST_VLD);
                asc_loadalign_deintlv(normalized[0], normalized[1], input + h_idx);
            }
        }

        // Reduce datablock b in [0, 8) to max_{j < 32} |normalized_{32g+j}| at lane g = h_idx / 32 + b
        vector_uint16_t abs_even, abs_odd, pair_abs_max, pair_abs_max_broadcast;
        asc_and(abs_even, normalized_u16[0], abs_bits_mask, mask_b16);
        asc_and(abs_odd, normalized_u16[1], abs_bits_mask, mask_b16);
        asc_max(pair_abs_max, abs_even, abs_odd, mask_b16);
        asc_reduce_max_datablock(pair_abs_max, pair_abs_max, mask_b16);
        asc_gather(pair_abs_max_broadcast, pair_abs_max, quant_block_idx_in_pair);
        vector_bool h_pair_mask_b16;
        asc_eq_scalar(h_pair_mask_b16, h_pair_idx, uint16_t(h_idx / (2 * kVectorLenB16)), mask_b16);
        asc_select(quant_abs_max, pair_abs_max_broadcast, quant_abs_max, h_pair_mask_b16);
    }

    // Derive UE8M0 exponent E_g = ceil(log2(max_g / 448)) + 127 and inverse scale 2^(127 - E_g)
    vector_uint16_t scale_exponent = quant_abs_max;
    vector_bfloat16_t inverse_scale_table;
    math::get_ue8m0_sf_exp<float8_e4m3_t>(scale_exponent);
    math::get_ue8m0_sf_inv(inverse_scale_table, scale_exponent);

    // Pack adjacent group exponents into uint16: packed_sf[p] = E_{2p} | (E_{2p+1} << 8)
    vector_uint8_t packed_sf;
    asc_pack_to_low(packed_sf, scale_exponent);
    vector_int16_t sf_idx;
    asc_arange(sf_idx, int16_t(0));
    asc_add_scalar(sf_idx, sf_idx, int16_t(sf_offset), mask_b16);
    uint32_t num_sf_pairs = BLOCK_H / MX_SF_DIVISOR;
    const auto sf_pair_mask_b16 = asc_update_mask_b16(num_sf_pairs);
    asc_scatter(output_sf, reinterpret_cast<vector_uint16_t&>(packed_sf), reinterpret_cast<vector_uint16_t&>(sf_idx), sf_pair_mask_b16);

    // Map lane l to its MX32 group: g = h_idx / 32 + l / 16
    vector_uint16_t inverse_scale_base_idx;
    asc_shiftright_scalar(inverse_scale_base_idx, lane_idx, uint16_t(__builtin_ctz(kQuantBlockH / 2)), mask_b16);

    // Quantize out = fp8_e4m3(bf16(normalized * 2^(127 - E_g))) and pack 4-way (4m..4m+3) into uint32 bytes
    #pragma unroll
    for (uint32_t h_idx = 0; h_idx < BLOCK_H; h_idx += 2 * kVectorLenB16) {
        auto& normalized = normalized_bf16[h_idx / (2 * kVectorLenB16)];
        vector_bfloat16_t inverse_scale, scaled_even, scaled_odd;
        vector_uint16_t inverse_scale_idx;
        asc_add_scalar(inverse_scale_idx, inverse_scale_base_idx, uint16_t(h_idx / kQuantBlockH), mask_b16);
        asc_gather(inverse_scale, inverse_scale_table, inverse_scale_idx);
        vector_float y[4];
        asc_mul(scaled_even, normalized[0], inverse_scale, mask_b16);
        asc_mul(scaled_odd, normalized[1], inverse_scale, mask_b16);
        asc_bfloat162float(y[0], scaled_even, mask_b16);
        asc_bfloat162float(y[1], scaled_odd, mask_b16);
        asc_bfloat162float_v2(y[2], scaled_even, mask_b16);
        asc_bfloat162float_v2(y[3], scaled_odd, mask_b16);

        vector_fp8_e4m3fn_t fp8[4];
        auto& fp8_u8 = reinterpret_cast<vector_uint8_t(&)[4]>(fp8);
        vector_uint8_t packed;
        asc_float2e4m3_rn_sat(fp8[0], y[0], mask_b8);
        asc_float2e4m3_rn_sat_v2(fp8[1], y[1], mask_b8);
        asc_float2e4m3_rn_sat_v3(fp8[2], y[2], mask_b8);
        asc_float2e4m3_rn_sat_v4(fp8[3], y[3], mask_b8);
        asc_or(packed, fp8_u8[0], fp8_u8[1], mask_b8);
        asc_or(packed, packed, fp8_u8[2], mask_b8);
        asc_or(packed, packed, fp8_u8[3], mask_b8);
        if constexpr (BLOCK_H != kKVLatentDim)
            asc_storealign(reinterpret_cast<__ubuf__ uint8_t*>(output + h_idx * pitch_rows), packed, pitch_rows, 0, mask_b8);
        else
            asc_storealign(reinterpret_cast<__ubuf__ uint8_t*>(output + h_idx), packed, mask_b8);
    }
}

template <uint32_t kHidden>
inline constexpr float kReciprocalHi = float(1.0 / kHidden);
template <uint32_t kHidden>
inline constexpr float kReciprocalLo = float(1.0 / kHidden - double(kReciprocalHi<kHidden>));

// Reduce one BF16 row to its FP32 square sum
template <uint32_t kHidden>
__simd_callee__ inline vector_float square_sum(__ubuf__ bfloat16_t* input) {
    constexpr uint32_t kVectorLenB16 = sizeof(vector_bfloat16_t) / sizeof(bfloat16_t);
    const auto mask = asc_create_mask_b32(PAT_ALL);
    const auto mask_b16 = asc_create_mask_b16(PAT_ALL);

    // Accumulate four FP32 partial square sums across the row
    vector_float sum0, sum1, sum2, sum3;
    asc_duplicate_scalar(sum0, 0.0f, mask);
    asc_duplicate_scalar(sum1, 0.0f, mask);
    asc_duplicate_scalar(sum2, 0.0f, mask);
    asc_duplicate_scalar(sum3, 0.0f, mask);
    for (uint32_t col_idx = 0; col_idx < kHidden; col_idx += 2 * kVectorLenB16) {
        vector_bfloat16_t input0, input1;
        vector_float value0, value1, value2, value3;
        asc_loadalign(input0, input + col_idx);
        asc_loadalign(input1, input + col_idx + kVectorLenB16);
        asc_bfloat162float(value0, input0, mask_b16);
        asc_bfloat162float_v2(value1, input0, mask_b16);
        asc_bfloat162float(value2, input1, mask_b16);
        asc_bfloat162float_v2(value3, input1, mask_b16);
        asc_mula(sum0, value0, value0, mask);
        asc_mula(sum1, value1, value1, mask);
        asc_mula(sum2, value2, value2, mask);
        asc_mula(sum3, value3, value3, mask);
    }

    // Reduce the four accumulators to a single scalar sum
    asc_add(sum0, sum0, sum1, mask);
    asc_add(sum0, sum0, sum2, mask);
    asc_add(sum0, sum0, sum3, mask);
    vector_float sum;
    asc_reduce_sum(sum, sum0, mask);
    return sum;
}

// Compute rstd = 1 / sqrt(sum / kHidden + eps) in double-FP32 (value + error) precision
template <uint32_t kHidden>
__simd_callee__ inline vector_float rstd_from_sum(vector_float sum, float eps_hi, float eps_lo) {
    const auto mask = asc_create_mask_b32(PAT_ALL);

    // TwoSum on product = rn(sum * R_hi) and eps_hi: value + error = product + eps_hi
    vector_float product, value, virtual_eps, error, temporary, reciprocal;
    asc_mul_scalar(product, sum, kReciprocalHi<kHidden>, mask);
    asc_add_scalar(value, product, eps_hi, mask);
    asc_sub(virtual_eps, value, product, mask);
    asc_sub(temporary, value, virtual_eps, mask);
    asc_sub(error, product, temporary, mask);
    asc_mul_scalar(temporary, virtual_eps, -1.0f, mask);
    asc_add_scalar(temporary, temporary, eps_hi, mask);
    asc_add(error, error, temporary, mask);

    // Add low-order terms: error += fma(sum, R_hi, -product) + sum * R_lo + eps_lo
    asc_mul_scalar(temporary, product, -1.0f, mask);
    asc_duplicate_scalar(reciprocal, kReciprocalHi<kHidden>, mask);
    asc_mula(temporary, sum, reciprocal, mask);
    asc_add(error, temporary, error, mask);
    asc_duplicate_scalar(reciprocal, kReciprocalLo<kHidden>, mask);
    asc_mula(error, sum, reciprocal, mask);
    asc_add_scalar(error, error, eps_lo, mask);

    // Compensated Newton step from y0 = 1 / sqrt(value) with y0^2 = squared + square_error:
    //   rstd = y0 + 0.5 * y0 * (1 - (value + error) * (squared + square_error))
    vector_float estimate, one, squared, square_error, residual, negative_value, negative_error;
    asc_duplicate_scalar(one, 1.0f, mask);
    asc_sqrt(estimate, value, mask);
    asc_div(estimate, one, estimate, mask);
    asc_mul(squared, estimate, estimate, mask);
    asc_mul_scalar(square_error, squared, -1.0f, mask);
    asc_mula(square_error, estimate, estimate, mask);
    residual = one;
    asc_mul_scalar(negative_value, value, -1.0f, mask);
    asc_mul_scalar(negative_error, error, -1.0f, mask);
    asc_mula(residual, negative_value, squared, mask);
    asc_mula(residual, negative_value, square_error, mask);
    asc_mula(residual, negative_error, squared, mask);
    asc_mula(residual, negative_error, square_error, mask);
    asc_mul_scalar(temporary, estimate, 0.5f, mask);
    asc_mula(estimate, temporary, residual, mask);
    return estimate;
}

// Normalize and quantize Q/KV rows, supporting sliced Q K-block handoff
template <bool kIsQ, uint32_t H_BEGIN = 0, uint32_t BLOCK_H = TransformLayout<kIsQ>::kLatentDim - H_BEGIN>
__simd_vf__ inline void vf_transform(uint16_t num_rows, float eps_hi, float eps_lo) {
    using Layout = TransformLayout<kIsQ>;
    auto input = reinterpret_cast<__ubuf__ bfloat16_t*>(Layout::kInput.addr);
    auto weight = reinterpret_cast<__ubuf__ bfloat16_t*>(Layout::kNormWeight.addr);
    auto rms_scale = reinterpret_cast<__ubuf__ float*>(Layout::kRMSScale.addr);
    auto output = reinterpret_cast<__ubuf__ float8_e4m3_t*>(Layout::kOutput.addr);
    constexpr uint32_t kVectorLenB32 = sizeof(vector_float) / sizeof(float);

    // Batch square-sum reductions across rows on the initial pass
    if constexpr (H_BEGIN == 0) {
        const auto mask = asc_create_mask_b32(PAT_ALL);
        uint32_t num_lanes = 1;
        const auto one_lane = asc_update_mask_b32(num_lanes);
        for (uint16_t row_idx = 0; row_idx < num_rows; ++ row_idx) {
            const auto sum = square_sum<Layout::kLatentDim>(input + row_idx * Layout::kLatentDim);
            vector_uint32_t index;
            asc_duplicate_scalar(index, uint32_t(row_idx), mask);
            asc_scatter(rms_scale, sum, index, one_lane);
        }
        asc_mem_bar(VST_VLD);
        vector_float sums;
        asc_loadalign(sums, rms_scale);
        auto scales = rstd_from_sum<Layout::kLatentDim>(sums, eps_hi, eps_lo);
        if constexpr (kIsQ)
            asc_mul_scalar(scales, scales, 2.0f, mask);
        vector_int32_t lane;
        vector_uint32_t index;
        asc_arange(lane, int32_t(0));
        asc_mul_scalar(index, reinterpret_cast<vector_uint32_t&>(lane), uint32_t(kVectorLenB32), mask);
        num_lanes = num_rows;
        asc_scatter(rms_scale, scales, index, asc_update_mask_b32(num_lanes));
        asc_mem_bar(VST_VLD);
    }

    // Normalize and quantize each row; KV applies RoPE before quantization
    const uint16_t pitch_rows = num_rows | 1;
    for (uint16_t row_idx = 0; row_idx < num_rows; ++ row_idx) {
        auto row_input = input + row_idx * Layout::kLatentDim + H_BEGIN;
        auto row_scale = rms_scale + row_idx * kVectorLenB32;
        if constexpr (kIsQ) {
            auto sf = reinterpret_cast<__ubuf__ uint16_t*>(Layout::kSF.addr);
            constexpr uint32_t kSFBase = Layout::kNumRows * (H_BEGIN / MX_SF_DIVISOR);
            norm_quant_row<BLOCK_H>(
                row_input, row_scale, weight + H_BEGIN, output + H_BEGIN * pitch_rows + row_idx * kDataBlockBytes,
                sf, kSFBase + row_idx * (BLOCK_H / MX_SF_DIVISOR), nullptr, pitch_rows);
        } else {
            auto cos_sin = reinterpret_cast<__ubuf__ float*>(Layout::kCosSin.addr);
            auto row_output = output + row_idx * Layout::kOutputStride;
            auto row_sf = reinterpret_cast<__ubuf__ uint16_t*>(row_output + kKVLatentDim);
            norm_quant_row<kKVLatentDim>(row_input, row_scale, weight, row_output, row_sf, 0, cos_sin + row_idx * kRopeDim);
        }
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

// Stage the first projection's weight tiles before Q/KV transformation
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumL1Stages, bool kSingleMBlock, asc_load_l2_cache_mode kL2CtrlB>
__aicore__ inline void stage_qkv_weight(__gm__ float8_e4m3_t* weights, uint32_t shape_m) {
    constexpr uint32_t SHAPE_N = kQKVDim, SHAPE_K = kHidden, BLOCK_K = 256;
    constexpr uint32_t FRAC_K = get_frac_k<uint8_t>();
    constexpr uint32_t kNumRawStages = 2, kHalfN = BLOCK_N / 2, kNumKBlocks = SHAPE_K / BLOCK_K;
    constexpr uint32_t kNumMaxNZStages = UBSizeBytes / ((2 * kHalfN + 1) * BLOCK_K);
    constexpr uint32_t kNumNZStages = kNumMaxNZStages < 8 ? kNumMaxNZStages : 8;
    constexpr ub_ptr<uint8_t> ub_raw(kHalfN, BLOCK_K);
    constexpr ub_ptr<uint8_t> ub_nz(BLOCK_K / FRAC_K, (kHalfN + 1) * FRAC_K, ub_raw.offset(kNumRawStages));
    constexpr l1_ptr<uint8_t> l1(BLOCK_N, BLOCK_K, BLOCK_M * BLOCK_K * kNumL1Stages);
    static_assert(ub_nz.offset(kNumNZStages) <= UBSizeBytes);

    // Split the weight tile into two equal row halves across the subblocks
    const uint32_t core_idx = block_idx, subblock_idx = asc_get_sub_block_id();
    uint32_t m_block_idx = 0, n_block_idx = 0;
    Scheduler<GemmType::Normal, BLOCK_M, BLOCK_N, kNumAICores> scheduler(core_idx, shape_m, SHAPE_N);
    const uint32_t num_tasks = kSingleMBlock ? uint32_t(scheduler.get_next_block(m_block_idx, n_block_idx)) :
                              ceil_div(scheduler.num_blocks > core_idx ? scheduler.num_blocks - core_idx : 0, kNumAICores);
    const uint32_t num_blocks = num_tasks * kNumKBlocks;
    uint32_t n_idx = n_block_idx * BLOCK_N + subblock_idx * kHalfN;
    notify<PIPE_MTE3, PIPE_V, kNumNZStages>(kNZEmpty);
    const auto k_block_idx = [&](uint32_t step_idx) __aicore__ { return kSingleMBlock ? step_idx : step_idx % kNumKBlocks; };
    const auto load = [&](uint32_t step_idx) __aicore__ {
        if constexpr (not kSingleMBlock) {
            if (step_idx % kNumKBlocks == 0) {
                scheduler.get_next_block(m_block_idx, n_block_idx);
                n_idx = n_block_idx * BLOCK_N + subblock_idx * kHalfN;
            }
        }
        copy_gm_to_ub(ub_raw[step_idx % kNumRawStages], reinterpret_cast<__gm__ uint8_t*>(weights) + n_idx * SHAPE_K + k_block_idx(step_idx) * BLOCK_K,
                      kHalfN, BLOCK_K, SHAPE_K, BLOCK_K, kL2CtrlB);
        notify<PIPE_MTE2, PIPE_V>(kRawFull, step_idx % kNumRawStages);
    };

    // Issue initial weight loads
    for (uint32_t step_idx = 0; step_idx < min(num_blocks, kNumRawStages); ++ step_idx)
        load(step_idx);

    // Refill raw input independently of the returned L1 credit
    for (uint32_t step_idx = 0; step_idx < num_blocks; ++ step_idx) {
        const uint32_t raw_stage_idx = step_idx % kNumRawStages, nz_stage_idx = step_idx % kNumNZStages, l1_stage_idx = k_block_idx(step_idx) % kNumL1Stages;
        wait<PIPE_MTE2, PIPE_V>(kRawFull, raw_stage_idx);
        wait<PIPE_MTE3, PIPE_V>(kNZEmpty, nz_stage_idx);
        vf_nd_to_nz<kHalfN, BLOCK_K>(ub_raw[raw_stage_idx].ptr(), ub_nz[nz_stage_idx].ptr());
        notify<PIPE_V, PIPE_MTE2>(kRawEmpty, raw_stage_idx);
        notify<PIPE_V, PIPE_MTE3>(kNZFull, nz_stage_idx);
        if (step_idx + kNumRawStages < num_blocks) {
            wait<PIPE_V, PIPE_MTE2>(kRawEmpty, raw_stage_idx);
            load(step_idx + kNumRawStages);
        }
        wait<PIPE_V, PIPE_MTE3>(kNZFull, nz_stage_idx);
        wait_intra_block(PIPE_MTE3, subblock_idx, kWeightFreeEventBase + l1_stage_idx);
        asc_copy_ub2l1(l1[l1_stage_idx].ptr() + subblock_idx * kHalfN * FRAC_K, ub_nz[nz_stage_idx].ptr(), BLOCK_K / FRAC_K,
                       kHalfN, 1, kHalfN);
        set_intra_block(PIPE_MTE3, subblock_idx, kWeightReadyEventBase + l1_stage_idx);
        notify<PIPE_MTE3, PIPE_V>(kNZEmpty, nz_stage_idx);
    }

    // Drain UB rings and returned L1 credits
    for (uint32_t step_idx = 0; step_idx < min(num_blocks, kNumRawStages); ++ step_idx)
        wait<PIPE_V, PIPE_MTE2>(kRawEmpty, step_idx);
    wait<PIPE_MTE3, PIPE_V, kNumNZStages>(kNZEmpty);
    wait_intra_blocks<PIPE_MTE3, kNumL1Stages, kWeightFreeEventBase>(subblock_idx);
    set_intra_block(PIPE_MTE3, subblock_idx, kWeightDrainEvent);
    notify<PIPE_MTE3, PIPE_MTE2>(kStoresDone);
    wait<PIPE_MTE3, PIPE_MTE2>(kStoresDone);
}

// Reserve weight rings with the prefetched raw slot right above the Q transform
template <uint32_t BLOCK_N>
struct QWeightLayout {
    static constexpr uint32_t BLOCK_K = 256, FRAC_K = get_frac_k<uint8_t>();
    static constexpr uint32_t kHalfN = BLOCK_N / 2, kNumRawStages = 2;
    static constexpr uint32_t kRawSizeBytes = kHalfN * BLOCK_K, kNZSizeBytes = (kHalfN + 1) * BLOCK_K;
    static constexpr uint32_t kTransformEnd = TransformLayout<true>::kEnd;
    static constexpr uint32_t kRawBase = aligned(kTransformEnd, 256u) - kRawSizeBytes;

    static constexpr ub_ptr<uint8_t> kRaw{kHalfN, BLOCK_K, kRawBase};
    static constexpr ub_ptr<uint8_t> kNZ{BLOCK_K / FRAC_K, (kHalfN + 1) * FRAC_K, kRaw.offset(kNumRawStages)};
    static constexpr uint32_t kNumNZStages = (UBSizeBytes - kRaw.offset(kNumRawStages)) / kNZSizeBytes;
    static constexpr uint32_t kEnd = kNZ.offset(kNumNZStages);
};

// Prestage the first Q weight block into L1 before QKV arrives
template <uint32_t BLOCK_M, uint32_t BLOCK_N, asc_load_l2_cache_mode kL2CtrlB>
__aicore__ inline void prestage_q_weight(__gm__ float8_e4m3_t* weights, uint32_t core_idx, uint32_t subblock_idx) {
    using Layout = QWeightLayout<BLOCK_N>;
    constexpr uint32_t BLOCK_K = Layout::BLOCK_K, FRAC_K = Layout::FRAC_K, kHalfN = Layout::kHalfN;
    static_assert(Layout::kRaw[1].addr >= Layout::kTransformEnd and Layout::kNumNZStages <= kTransformDone.base and Layout::kEnd <= UBSizeBytes);
    constexpr l1_ptr<uint8_t> l1(BLOCK_N, BLOCK_K, BLOCK_M * kQLoraDim);
    const uint32_t n_idx = core_idx * BLOCK_N + subblock_idx * kHalfN;
    copy_gm_to_ub(Layout::kRaw[1], reinterpret_cast<__gm__ uint8_t*>(weights) + n_idx * kQLoraDim,
                  kHalfN, BLOCK_K, kQLoraDim, BLOCK_K, kL2CtrlB);
    notify<PIPE_MTE2, PIPE_V>(kRawFull, 1);
    wait<PIPE_MTE2, PIPE_V>(kRawFull, 1);
    vf_nd_to_nz<kHalfN, BLOCK_K>(Layout::kRaw[1].ptr(), Layout::kNZ[0].ptr());
    notify<PIPE_V, PIPE_MTE2>(kRawEmpty, 1);
    notify<PIPE_V, PIPE_MTE3>(kNZFull, 0);
    wait<PIPE_V, PIPE_MTE3>(kNZFull, 0);
    asc_copy_ub2l1(l1[0].ptr() + subblock_idx * kHalfN * FRAC_K, Layout::kNZ[0].ptr(), BLOCK_K / FRAC_K, kHalfN, 1, kHalfN);
    set_intra_block(PIPE_MTE3, subblock_idx, kWeightReadyEventBase);
    notify<PIPE_MTE3, PIPE_V>(kNZEmpty, 0);
    wait<PIPE_MTE3, PIPE_V>(kNZEmpty, 0);
    wait<PIPE_V, PIPE_MTE2>(kRawEmpty, 1);
}

// Prefetch the second Q weight block without touching active transform buffers
template <uint32_t BLOCK_N, asc_load_l2_cache_mode kL2CtrlB>
__aicore__ inline void prefetch_q_weight(__gm__ float8_e4m3_t* weights, uint32_t core_idx, uint32_t subblock_idx) {
    using Layout = QWeightLayout<BLOCK_N>;
    const uint32_t n_idx = core_idx * BLOCK_N + subblock_idx * Layout::kHalfN;
    copy_gm_to_ub(Layout::kRaw[1], reinterpret_cast<__gm__ uint8_t*>(weights) + n_idx * kQLoraDim + Layout::BLOCK_K,
                  Layout::kHalfN, Layout::BLOCK_K, kQLoraDim, Layout::BLOCK_K, kL2CtrlB);
    notify<PIPE_MTE2, PIPE_V>(kRawFull, 1);
}

// Stage ND Q weights through UB into both halves of each L1 tile
template <uint32_t SHAPE_N, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumL1Stages, bool kSingleMBlock, asc_load_l2_cache_mode kL2CtrlB>
__aicore__ inline void stage_q_weight(__gm__ float8_e4m3_t* weights, uint32_t shape_m) {
    using Layout = QWeightLayout<BLOCK_N>;
    constexpr uint32_t BLOCK_K = Layout::BLOCK_K, FRAC_K = Layout::FRAC_K;
    constexpr uint32_t kHalfN = Layout::kHalfN, kNumKBlocks = kQLoraDim / BLOCK_K;
    constexpr uint32_t kNumRawStages = Layout::kNumRawStages, kNumNZStages = Layout::kNumNZStages;
    constexpr uint32_t kNumNWaves = ceil_div(SHAPE_N, kNumAICores * BLOCK_N);
    constexpr uint32_t kTailHalfN = SHAPE_N % (kNumAICores * BLOCK_N) / (2 * kNumAICores);
    constexpr bool kHasTailWave = kTailHalfN != 0;
    static_assert(not kHasTailWave or kSingleMBlock);
    constexpr auto ub_raw = Layout::kRaw, ub_nz = Layout::kNZ;
    constexpr l1_ptr<uint8_t> l1(BLOCK_N, BLOCK_K, BLOCK_M * kQLoraDim);
    const uint32_t core_idx = block_idx, subblock_idx = asc_get_sub_block_id();
    const uint32_t num_blocks = (kSingleMBlock ? 1 : ceil_div(shape_m, BLOCK_M)) * kNumNWaves * kNumKBlocks;

    const auto get_half_n = [&](uint32_t step_idx) __aicore__ {
        return kHasTailWave and step_idx >= (kNumNWaves - 1) * kNumKBlocks ? kTailHalfN : kHalfN;
    };
    const auto load = [&](uint32_t step_idx) __aicore__ {
        const uint32_t raw_stage_idx = step_idx % kNumRawStages, half_n = get_half_n(step_idx);
        const uint32_t wave_idx = kSingleMBlock ? step_idx / kNumKBlocks : step_idx / kNumKBlocks % kNumNWaves;
        const uint32_t n_idx = wave_idx * kNumAICores * BLOCK_N + (2 * core_idx + subblock_idx) * half_n;
        copy_gm_to_ub(ub_raw[raw_stage_idx], reinterpret_cast<__gm__ uint8_t*>(weights) + n_idx * kQLoraDim + step_idx % kNumKBlocks * BLOCK_K,
                      half_n, BLOCK_K, kQLoraDim, BLOCK_K, kL2CtrlB);
        notify<PIPE_MTE2, PIPE_V>(kRawFull, raw_stage_idx);
    };

    // Fill the raw ring while retaining the prefetched second block
    notify<PIPE_MTE3, PIPE_V, kNumNZStages>(kNZEmpty);
    load(2);

    // Repack weights and refill raw slots independently of L1 credits
    for (uint32_t step_idx = 1; step_idx < num_blocks; ++ step_idx) {
        const uint32_t raw_stage_idx = step_idx % kNumRawStages, half_n = get_half_n(step_idx);
        const uint32_t nz_stage_idx = step_idx % kNumNZStages, l1_stage_idx = step_idx % kNumL1Stages;
        wait<PIPE_MTE2, PIPE_V>(kRawFull, raw_stage_idx);
        wait<PIPE_MTE3, PIPE_V>(kNZEmpty, nz_stage_idx);
        if (not kHasTailWave or half_n == kHalfN)
            vf_nd_to_nz<kHalfN, BLOCK_K>(ub_raw[raw_stage_idx].ptr(), ub_nz[nz_stage_idx].ptr());
        else
            vf_nd_to_nz<kTailHalfN, BLOCK_K>(ub_raw[raw_stage_idx].ptr(), ub_nz[nz_stage_idx].ptr());
        notify<PIPE_V, PIPE_MTE2>(kRawEmpty, raw_stage_idx);
        notify<PIPE_V, PIPE_MTE3>(kNZFull, nz_stage_idx);
        if (step_idx + kNumRawStages < num_blocks) {
            wait<PIPE_V, PIPE_MTE2>(kRawEmpty, raw_stage_idx);
            load(step_idx + kNumRawStages);
        }
        wait<PIPE_V, PIPE_MTE3>(kNZFull, nz_stage_idx);
        wait_intra_block(PIPE_MTE3, subblock_idx, kWeightFreeEventBase + l1_stage_idx);
        asc_copy_ub2l1(l1[l1_stage_idx].ptr() + subblock_idx * half_n * FRAC_K, ub_nz[nz_stage_idx].ptr(), BLOCK_K / FRAC_K,
                       half_n, 1, BLOCK_N - half_n);
        set_intra_block(PIPE_MTE3, subblock_idx, kWeightReadyEventBase + l1_stage_idx);
        notify<PIPE_MTE3, PIPE_V>(kNZEmpty, nz_stage_idx);
    }

    // Drain UB rings and returned L1 credits
    wait<PIPE_V, PIPE_MTE2, kNumRawStages>(kRawEmpty);
    wait<PIPE_MTE3, PIPE_V, kNumNZStages>(kNZEmpty);
    wait_intra_blocks<PIPE_MTE3, kNumL1Stages, kWeightFreeEventBase>(subblock_idx);
    set_intra_block(PIPE_MTE3, subblock_idx, kWeightDrainEvent);
}

// Project X into the Q and KV latent representations
template <uint32_t kTileM, bool kSingleMBlock>
__aicore__ __forceinline__ void qkv_projection(
    __gm__ float8_e4m3_t* x, __gm__ float8_e4m3_t* wa,
    __gm__ int16_t* x_sf, __gm__ int16_t* wa_sf,
    __gm__ bfloat16_t* qkv, uint32_t shape_m, uint32_t x_sf_stride
) {
    const gm_ptr<float8_e4m3_t, Major::K> gm_a(kHidden, reinterpret_cast<uintptr_t>(x));
    const gm_ptr<float8_e4m3_t, Major::K> gm_b(kHidden, reinterpret_cast<uintptr_t>(wa));
    const gm_ptr<bfloat16_t, Major::K> gm_d(kQKVDim, reinterpret_cast<uintptr_t>(qkv));
    constexpr uint32_t BLOCK_M = kTileM <= 128 ? kTileM : aligned(kTileM / 2, 16u);
    constexpr uint32_t BLOCK_N = kTileM <= 128 ? 64 : 112, BLOCK_K = 256;
    constexpr uint32_t SHAPE_N = kQKVDim, SHAPE_K = kHidden, kNumKBlocks = SHAPE_K / BLOCK_K;
    constexpr uint32_t kNumL0Stages = 2, kNumL1SFStages = 2, kShapeSFK = SHAPE_K / MX_SF_DIVISOR;
    constexpr uint32_t kNumL1Stages = 4, kSFPairsPerKBlock = BLOCK_K / MX_SF_DIVISOR;
    constexpr uint32_t kChunkSFK = kNumL1Stages * kSFPairsPerKBlock;
    static_assert(kNumKBlocks % kNumL1Stages == 0);
    constexpr auto kL2CtrlA = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM;
    constexpr auto kL2CtrlB = kSingleMBlock ? asc_load_l2_cache_mode::NOTALLOC_KEEP : asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM;
    constexpr auto kL2CtrlStoreCd = asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM;
    const gm_ptr<int16_t, Major::MN> gm_sfa(x_sf_stride, reinterpret_cast<uintptr_t>(x_sf), uint64_t(x_sf_stride) * kShapeSFK);
    const gm_ptr<int16_t, Major::MN> gm_sfb(SHAPE_N, reinterpret_cast<uintptr_t>(wa_sf), uint64_t(SHAPE_N) * kShapeSFK);

    // Partition L1 and L0 buffers for data, scales and accumulators
    constexpr l1_ptr<float8_e4m3_t, Major::K> l1a(BLOCK_M, BLOCK_K, 0);
    constexpr l1_ptr<float8_e4m3_t, Major::K> l1b(BLOCK_N, BLOCK_K, l1a.offset(kNumL1Stages));
    constexpr l1_ptr<int16_t> l1_sfa(BLOCK_M, kChunkSFK, l1b.offset(kNumL1Stages));
    constexpr l1_ptr<int16_t> l1_sfb(BLOCK_N, kChunkSFK, l1_sfa.offset(kNumL1SFStages));
    constexpr l0a_ptr<float8_e4m3_t> l0a(BLOCK_M, BLOCK_K);
    constexpr l0b_ptr<float8_e4m3_t> l0b(BLOCK_N, BLOCK_K);
    constexpr l0c_ptr<float> l0c(BLOCK_M, BLOCK_N);
    static_assert(l1_sfb.offset(kNumL1SFStages) <= L1SizeBytes, "L1 overflow");
    static_assert(l0a.offset(kNumL0Stages) <= L0ASizeBytes and l0b.offset(kNumL0Stages) <= L0BSizeBytes and l0c.offset(1) <= L0CSizeBytes);

    const uint32_t core_idx = block_idx;

    if ASCEND_IS_AIC {
        asc_set_mmad_direction_n();

        static_assert(kNumL1Stages + kNumL1SFStages <= kNumPipeEvents);
        constexpr PipeEvent<PIPE_MTE2, PIPE_MTE1> kL1Full{0}, kSFFull{kNumL1Stages};
        constexpr PipeEvent<PIPE_MTE1, PIPE_MTE2> kL1Empty{0}, kSFEmpty{kNumL1Stages};
        constexpr PipeEvent<PIPE_MTE1, PIPE_M> kL0Full{0};
        constexpr PipeEvent<PIPE_M, PIPE_MTE1> kL0Empty{0};
        notify<PIPE_M, PIPE_MTE1, kNumL0Stages>(kL0Empty);
        notify<PIPE_MTE1, PIPE_MTE2, kNumL1Stages>(kL1Empty);
        notify<PIPE_MTE1, PIPE_MTE2, kNumL1SFStages>(kSFEmpty);

        set_intra_blocks<PIPE_MTE1, kNumL1Stages, kWeightFreeEventBase>(0);
        set_intra_blocks<PIPE_MTE1, kNumL1Stages, kWeightFreeEventBase>(1);

        Scheduler<GemmType::Normal, BLOCK_M, BLOCK_N, kNumAICores> scheduler(core_idx, shape_m, SHAPE_N);
        uint32_t m_block_idx, n_block_idx, sf_stage_idx = 0, l0_stage_idx = 0;

        for (bool has_task = scheduler.get_next_block(m_block_idx, n_block_idx); has_task;
             has_task = not kSingleMBlock and scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const uint32_t m_idx = scheduler.get_global_m_idx(m_block_idx);
            const uint32_t n_idx = scheduler.get_global_n_idx(n_block_idx);
            const auto a = gm_a.index(m_idx, 0);
            const auto sfa = gm_sfa.index(m_idx, 0);
            const auto sfb = gm_sfb.index(n_idx, 0);
            const uint32_t actual_m = scheduler.get_actual_m(m_block_idx);
            auto l0c_tile = l0c[0].as_mad_aligned(actual_m, BLOCK_N);
            for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocks; ++ k_block_idx, l0_stage_idx = (l0_stage_idx + 1) % kNumL0Stages) {
                const uint32_t k_idx = k_block_idx * BLOCK_K, l1_stage_idx = k_block_idx % kNumL1Stages;
                const auto sf_pair_idx = uint16_t(l1_stage_idx * kSFPairsPerKBlock);

                // Load data and 4-block scale chunks from GM to L1
                wait<PIPE_MTE1, PIPE_MTE2>(kL1Empty, l1_stage_idx);
                copy_gm_to_l1(l1a[l1_stage_idx], a.index(0, k_idx), actual_m, BLOCK_K, kL2CtrlA);
                notify<PIPE_MTE2, PIPE_MTE1>(kL1Full, l1_stage_idx);
                if (l1_stage_idx == 0) {
                    wait<PIPE_MTE1, PIPE_MTE2>(kSFEmpty, sf_stage_idx);
                    copy_gm_to_l1_mx(l1_sfa[sf_stage_idx], sfa.index(0, k_block_idx * kSFPairsPerKBlock), actual_m, kChunkSFK, kL2CtrlA);
                    copy_gm_to_l1_mx(l1_sfb[sf_stage_idx], sfb.index(0, k_block_idx * kSFPairsPerKBlock), BLOCK_N, kChunkSFK, kL2CtrlB);
                    notify<PIPE_MTE2, PIPE_MTE1>(kSFFull, sf_stage_idx);
                }

                wait_intra_block(PIPE_MTE1, 0, kWeightReadyEventBase + l1_stage_idx);
                wait_intra_block(PIPE_MTE1, 1, kWeightReadyEventBase + l1_stage_idx);
                wait<PIPE_MTE2, PIPE_MTE1>(kL1Full, l1_stage_idx);

                auto l0a_tile = l0a[l0_stage_idx].as_mad_aligned(actual_m, BLOCK_K);
                auto l0b_tile = l0b[l0_stage_idx].as_mad_aligned(BLOCK_N, BLOCK_K);

                wait<PIPE_M, PIPE_MTE1>(kL0Empty, l0_stage_idx);
                copy_l1_to_l0a(l0a_tile, l1a[l1_stage_idx], 0, 0);
                copy_l1_to_l0b(l0b_tile, l1b[l1_stage_idx], 0, 0);
                if (l1_stage_idx == 0)
                    wait<PIPE_MTE2, PIPE_MTE1>(kSFFull, sf_stage_idx);
                copy_l1_to_l0a_mx(l0a_tile, l1_sfa[sf_stage_idx], 0, sf_pair_idx, BLOCK_K);
                copy_l1_to_l0b_mx(l0b_tile, l1_sfb[sf_stage_idx], 0, sf_pair_idx, BLOCK_K);
                notify<PIPE_MTE1, PIPE_M>(kL0Full, l0_stage_idx);

                wait<PIPE_MTE1, PIPE_M>(kL0Full, l0_stage_idx);
                mad(l0c_tile, l0a_tile, l0b_tile, actual_m, BLOCK_N, BLOCK_K,
                    k_block_idx == kNumKBlocks - 1 ? asc_unit_flag_mode::ENABLE_UPDATE : asc_unit_flag_mode::ENABLE_KEEP,
                    k_block_idx == 0);
                notify<PIPE_M, PIPE_MTE1>(kL0Empty, l0_stage_idx);

                // Store the final accumulated tile directly from L0C to GM
                if (k_block_idx == kNumKBlocks - 1)
                    copy_l0c_to_gm(gm_d.index(m_idx, n_idx).ptr(),
                                   l0c_tile, actual_m, BLOCK_N, gm_d.stride_outer, kL2CtrlStoreCd);

                // Release L1 data and completed scale chunks
                set_intra_block(PIPE_MTE1, 0, kWeightFreeEventBase + l1_stage_idx);
                set_intra_block(PIPE_MTE1, 1, kWeightFreeEventBase + l1_stage_idx);
                notify<PIPE_MTE1, PIPE_MTE2>(kL1Empty, l1_stage_idx);
                if (l1_stage_idx + 1 == kNumL1Stages) {
                    notify<PIPE_MTE1, PIPE_MTE2>(kSFEmpty, sf_stage_idx);
                    sf_stage_idx = (sf_stage_idx + 1) % kNumL1SFStages;
                }
            }
        }
        wait_intra_block(PIPE_MTE1, 0, kWeightDrainEvent);
        wait_intra_block(PIPE_MTE1, 1, kWeightDrainEvent);
        wait<PIPE_M, PIPE_MTE1, kNumL0Stages>(kL0Empty);
        wait<PIPE_MTE1, PIPE_MTE2, kNumL1Stages>(kL1Empty);
        wait<PIPE_MTE1, PIPE_MTE2, kNumL1SFStages>(kSFEmpty);

        // Publish QKV activations after every AIC finishes its stores
        asc_sync_inter_arrive(PIPE_FIX, 0);
        asc_sync_inter_wait(PIPE_FIX, 0);
        set_intra_block(PIPE_FIX, 0, kQKVReadyEvent);
        set_intra_block(PIPE_FIX, 1, kQKVReadyEvent);
    } else {
        stage_qkv_weight<BLOCK_M, BLOCK_N, kNumL1Stages, kSingleMBlock, kL2CtrlB>(gm_b.ptr(), shape_m);
    }
}

// Project the normalized and quantized Q latent representation
template <uint32_t SHAPE_N, uint32_t BLOCK_M, bool kSingleMBlock>
__aicore__ __forceinline__ void q_projection(
    __gm__ float8_e4m3_t* qa, __gm__ float8_e4m3_t* wb,
    __gm__ int16_t* qa_sf, __gm__ int16_t* wb_sf,
    __gm__ bfloat16_t* q, uint32_t shape_m,
    __gm__ bfloat16_t* qkv, __gm__ bfloat16_t* qw, __gm__ bfloat16_t* kw,
    __gm__ float* cos_sin, __gm__ int32_t* positions, __gm__ int32_t* slots, __gm__ uint8_t* cache,
    uint32_t num_slots, uint32_t cache_block_size, uint32_t cache_page_stride_bytes, float eps_hi, float eps_lo
) {
    static_assert(kSingleMBlock or BLOCK_M == 256);
    const gm_ptr<float8_e4m3_t, Major::K> gm_a(shape_m, reinterpret_cast<uintptr_t>(qa));
    const gm_ptr<float8_e4m3_t, Major::K> gm_b(kQLoraDim, reinterpret_cast<uintptr_t>(wb));
    const gm_ptr<bfloat16_t, Major::K> gm_d(SHAPE_N, reinterpret_cast<uintptr_t>(q));
    constexpr uint32_t BLOCK_N = BLOCK_M <= 128 ? 512 : (SHAPE_N == 32768 or kSingleMBlock ? 256 : 192);
    constexpr uint32_t kTailN = SHAPE_N % (kNumAICores * BLOCK_N) / kNumAICores;
    constexpr bool kHasTailWave = kTailN != 0;
    constexpr uint32_t BLOCK_K = 256, SHAPE_K = kQLoraDim, MAD_K = BLOCK_K / 2, kNumKBlocks = SHAPE_K / BLOCK_K;
    constexpr uint32_t MAD_M = BLOCK_N == 192 ? BLOCK_M / 2 : BLOCK_M;
    constexpr uint32_t MAD_N = BLOCK_N == 192 ? BLOCK_N : BLOCK_N / 2;
    constexpr uint32_t kNumL0Stages = 2, kNumL1SFStages = 2;
    constexpr uint32_t kNumSFAStages = kSingleMBlock ? 1 : kNumL1SFStages;
    constexpr uint32_t kShapeSFK = SHAPE_K / MX_SF_DIVISOR;

    // Fit resident A and B tiles in L1
    constexpr uint32_t kNumMaxL1Stages = (L1SizeBytes - BLOCK_M * SHAPE_K - (BLOCK_M * kNumSFAStages + BLOCK_N * kNumL1SFStages) * kShapeSFK * sizeof(int16_t)) / (BLOCK_N * BLOCK_K);
    constexpr uint32_t kNumL1Stages = kNumMaxL1Stages < 4 ? kNumMaxL1Stages : 4;
    constexpr auto kL2CtrlA = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM;
    constexpr auto kL2CtrlB = kSingleMBlock ? asc_load_l2_cache_mode::NOTALLOC_KEEP : asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM;
    constexpr auto kL2CtrlStoreCd = asc_store_l2_cache_mode::NOTALLOC_CLEAN;

    constexpr uint32_t kBlockSFK = BLOCK_K / MX_SF_DIVISOR, kRemainingSFK = kShapeSFK - kBlockSFK;
    const gm_ptr<int16_t, Major::K> gm_sfa_first(kBlockSFK, reinterpret_cast<uintptr_t>(qa_sf));
    const gm_ptr<int16_t, Major::K> gm_sfa_remaining(kRemainingSFK, reinterpret_cast<uintptr_t>(qa_sf + uint64_t(shape_m) * kBlockSFK));
    const gm_ptr<int16_t, Major::MN> gm_sfb(SHAPE_N, reinterpret_cast<uintptr_t>(wb_sf), uint64_t(SHAPE_N) * kShapeSFK);

    constexpr uint32_t kNumCDStages = ceil_div(BLOCK_M, MAD_M) * (BLOCK_N / MAD_N);

    // Partition L1 and L0 buffers for data, scales and accumulators
    constexpr l1_ptr<float8_e4m3_t, Major::K> l1a(BLOCK_M, BLOCK_K, 0);
    constexpr l1_ptr<float8_e4m3_t, Major::K> l1b(BLOCK_N, BLOCK_K, l1a.offset(kNumKBlocks));
    constexpr l1_ptr<int16_t> l1_sfa(BLOCK_M, kShapeSFK, l1b.offset(kNumL1Stages));
    constexpr l1_ptr<int16_t> l1_sfb(BLOCK_N, kShapeSFK, l1_sfa.offset(kNumSFAStages));
    constexpr l0a_ptr<float8_e4m3_t> l0a(MAD_M, MAD_K);
    constexpr l0b_ptr<float8_e4m3_t> l0b(MAD_N, MAD_K);
    constexpr l0c_ptr<float> l0c(MAD_M, MAD_N);
    static_assert(l1_sfb.offset(kNumL1SFStages) <= L1SizeBytes, "L1 overflow");
    static_assert(l0a.offset(kNumL0Stages) <= L0ASizeBytes and l0b.offset(kNumL0Stages) <= L0BSizeBytes and l0c.offset(kNumCDStages) <= L0CSizeBytes);

    const uint32_t core_idx = block_idx;

    if ASCEND_IS_AIC {
        asc_set_mmad_direction_n();

        static_assert(kNumL1Stages + kNumL1SFStages < kNumPipeEvents);
        constexpr PipeEvent<PIPE_MTE2, PIPE_MTE1> kL1Full{0}, kSFFull{kNumL1Stages}, kRemainingSFFull{kNumL1Stages + kNumL1SFStages};
        constexpr PipeEvent<PIPE_MTE1, PIPE_MTE2> kL1Empty{0}, kSFEmpty{kNumL1Stages};
        constexpr PipeEvent<PIPE_MTE1, PIPE_M> kL0Full{0};
        constexpr PipeEvent<PIPE_M, PIPE_MTE1> kL0Empty{0};
        notify<PIPE_M, PIPE_MTE1, kNumL0Stages>(kL0Empty);
        notify<PIPE_MTE1, PIPE_MTE2, kNumL1Stages>(kL1Empty);
        notify<PIPE_MTE1, PIPE_MTE2, kNumL1SFStages>(kSFEmpty);

        set_intra_blocks<PIPE_MTE1, kNumL1Stages - 1, kWeightFreeEventBase + 1>(0);
        set_intra_blocks<PIPE_MTE1, kNumL1Stages - 1, kWeightFreeEventBase + 1>(1);

        // Preload the first B scales while arriving on idle PIPE_FIX as each Q slice finishes
        copy_gm_to_l1_mx(l1_sfb[0], gm_sfb.index(core_idx * BLOCK_N, 0), BLOCK_N, kShapeSFK, kL2CtrlB);
        wait_intra_block(PIPE_FIX, 0, kQFirstReadyEvent);
        asc_sync_inter_arrive(PIPE_FIX, 1);
        wait_intra_block(PIPE_FIX, 0, kQReadyEvent);
        asc_sync_inter_arrive(PIPE_FIX, 2);
        asc_sync_inter_wait(PIPE_MTE2, 1);

        uint32_t l1_stage_idx = 0, sf_stage_idx = 0;

        for (uint32_t m_idx = 0; m_idx < (kSingleMBlock ? BLOCK_M : shape_m); m_idx += BLOCK_M) {
            const uint32_t actual_m = kSingleMBlock ? shape_m : min(shape_m - m_idx, BLOCK_M), m_sf_stage_idx = m_idx / BLOCK_M % kNumSFAStages;
            for (uint32_t n_base = 0; n_base < SHAPE_N; n_base += kNumAICores * BLOCK_N, sf_stage_idx = (sf_stage_idx + 1) % kNumL1SFStages) {
                const uint32_t tile_n = kHasTailWave and n_base + kNumAICores * BLOCK_N > SHAPE_N ? kTailN : BLOCK_N;
                const uint32_t n_idx = n_base + core_idx * tile_n;
                const bool first_n = n_base == 0;

                for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocks; ++ k_block_idx) {
                    const uint32_t k_idx = k_block_idx * BLOCK_K;
                    if (m_idx == 0 and first_n and k_block_idx == 1)
                        asc_sync_inter_wait(PIPE_MTE2, 2);

                    // Load data from GM to L1
                    wait<PIPE_MTE1, PIPE_MTE2>(kL1Empty, l1_stage_idx);
                    if (first_n)
                        copy_gm_to_l1<true>(l1a[k_block_idx], gm_a.index(k_idx, uint64_t(m_idx) * kDataBlockBytes), actual_m, BLOCK_K, kL2CtrlA, 1, shape_m);
                    notify<PIPE_MTE2, PIPE_MTE1>(kL1Full, l1_stage_idx);

                    // Load first-block and remaining Q scales separately on the initial N tile
                    if (k_block_idx == 0) {
                        wait<PIPE_MTE1, PIPE_MTE2>(kSFEmpty, sf_stage_idx);
                        if (first_n)
                            copy_gm_to_l1_mx(l1_sfa[m_sf_stage_idx], gm_sfa_first.index(m_idx, 0), actual_m, kBlockSFK, kL2CtrlA);
                        if (m_idx > 0 or not first_n)
                            copy_gm_to_l1_mx(l1_sfb[sf_stage_idx], gm_sfb.index(n_idx, 0), tile_n, kShapeSFK, kL2CtrlB);
                        notify<PIPE_MTE2, PIPE_MTE1>(kSFFull, sf_stage_idx);
                    }
                    if (first_n and k_block_idx == 1) {
                        copy_gm_to_l1_mx(l1_sfa[m_sf_stage_idx].offset_bytes(kBlockSFK * kDataBlockBytes), gm_sfa_remaining.index(m_idx, 0), actual_m, kRemainingSFK, kL2CtrlA);
                        notify<PIPE_MTE2, PIPE_MTE1>(kRemainingSFFull);
                    }

                    if (MAD_N == BLOCK_N or tile_n < BLOCK_N) {
                        wait_intra_block(PIPE_MTE1, 0, kWeightReadyEventBase + l1_stage_idx);
                        wait_intra_block(PIPE_MTE1, 1, kWeightReadyEventBase + l1_stage_idx);
                    }
                    wait<PIPE_MTE2, PIPE_MTE1>(kL1Full, l1_stage_idx);

                    for (uint32_t m_mad_idx = 0, cd_stage_idx = 0; m_mad_idx < actual_m; m_mad_idx += MAD_M) {
                        for (uint32_t n_mad_idx = 0; n_mad_idx < tile_n; n_mad_idx += MAD_N, ++ cd_stage_idx) {
                            // Wait for each subblock's weight half before its MAD
                            if (MAD_N < BLOCK_N and tile_n == BLOCK_N)
                                wait_intra_block(PIPE_MTE1, n_mad_idx / MAD_N, kWeightReadyEventBase + l1_stage_idx);
                            const auto mad_m = min(actual_m - m_mad_idx, MAD_M), mad_n = min(tile_n, MAD_N);
                            auto l0c_tile = l0c[cd_stage_idx].as_mad_aligned(mad_m, mad_n);

                            for (uint32_t k_mad_idx = 0, l0_stage_idx = 0; k_mad_idx < BLOCK_K; k_mad_idx += MAD_K, ++ l0_stage_idx) {
                                const auto sf_pair_idx = uint16_t((k_idx + k_mad_idx) / MX_SF_DIVISOR);
                                auto l0a_tile = l0a[l0_stage_idx].as_mad_aligned(mad_m, MAD_K);
                                auto l0b_tile = l0b[l0_stage_idx].as_mad_aligned(mad_n, MAD_K);

                                wait<PIPE_M, PIPE_MTE1>(kL0Empty, l0_stage_idx);
                                if (n_mad_idx == 0)
                                    copy_l1_to_l0a(l0a_tile, l1a[k_block_idx], m_mad_idx, k_mad_idx);
                                if (m_mad_idx == 0)
                                    copy_l1_to_l0b(l0b_tile, l1b[l1_stage_idx], n_mad_idx, k_mad_idx);

                                // Wait for scales only after issuing L0 data copies
                                if (k_block_idx == 0 and m_mad_idx == 0 and n_mad_idx == 0 and k_mad_idx == 0)
                                    wait<PIPE_MTE2, PIPE_MTE1>(kSFFull, sf_stage_idx);
                                if (first_n and k_block_idx == 1 and m_mad_idx == 0 and n_mad_idx == 0 and k_mad_idx == 0)
                                    wait<PIPE_MTE2, PIPE_MTE1>(kRemainingSFFull);
                                if (n_mad_idx == 0)
                                    copy_l1_to_l0a_mx(l0a_tile, l1_sfa[m_sf_stage_idx], m_mad_idx, sf_pair_idx, MAD_K);
                                if (m_mad_idx == 0)
                                    copy_l1_to_l0b_mx(l0b_tile, l1_sfb[sf_stage_idx], n_mad_idx, sf_pair_idx, MAD_K);
                                notify<PIPE_MTE1, PIPE_M>(kL0Full, l0_stage_idx);

                                wait<PIPE_MTE1, PIPE_M>(kL0Full, l0_stage_idx);
                                mad(l0c_tile, l0a_tile, l0b_tile, mad_m, mad_n, MAD_K,
                                    (k_block_idx == kNumKBlocks - 1 and k_mad_idx + MAD_K >= BLOCK_K) ? asc_unit_flag_mode::ENABLE_UPDATE : asc_unit_flag_mode::ENABLE_KEEP,
                                    k_block_idx == 0 and k_mad_idx == 0);
                                notify<PIPE_M, PIPE_MTE1>(kL0Empty, l0_stage_idx);
                            }

                            // Store the final accumulated tile directly from L0C to GM
                            if (k_block_idx == kNumKBlocks - 1)
                                copy_l0c_to_gm(gm_d.index(m_idx + m_mad_idx, n_idx + n_mad_idx).ptr(),
                                               l0c_tile, mad_m, mad_n, gm_d.stride_outer, kL2CtrlStoreCd);
                        }
                    }

                    // Advance the L1 data ring cursor
                    set_intra_block(PIPE_MTE1, 0, kWeightFreeEventBase + l1_stage_idx);
                    set_intra_block(PIPE_MTE1, 1, kWeightFreeEventBase + l1_stage_idx);
                    notify<PIPE_MTE1, PIPE_MTE2>(kL1Empty, l1_stage_idx);
                    l1_stage_idx = (l1_stage_idx + 1) % kNumL1Stages;
                }
                notify<PIPE_MTE1, PIPE_MTE2>(kSFEmpty, sf_stage_idx);
            }
        }
        wait_intra_block(PIPE_MTE1, 0, kWeightDrainEvent);
        wait_intra_block(PIPE_MTE1, 1, kWeightDrainEvent);
        wait<PIPE_M, PIPE_MTE1, kNumL0Stages>(kL0Empty);
        wait<PIPE_MTE1, PIPE_MTE2, kNumL1Stages>(kL1Empty);
        wait<PIPE_MTE1, PIPE_MTE2, kNumL1SFStages>(kSFEmpty);
    } else {
        // Split Q and KV normalization across the two AIV subblocks
        const uint32_t subblock_idx = asc_get_sub_block_id();
        const auto run_transform = [&]<bool kIsQ>() __aicore__ {
            using Layout = TransformLayout<kIsQ>;
            constexpr uint32_t kNumRows = Layout::kNumRows, kLatentDim = Layout::kLatentDim, kOutputStride = Layout::kOutputStride;
            static_assert(Layout::kEnd <= UBSizeBytes);

            const auto metadata = Layout::kMetadata.ptr();
            const auto load_rope = [&](uint32_t row_base, uint32_t num_rows) __aicore__ {
                if constexpr (not kIsQ) {
                    asc_copy_gm2ub_align(metadata, positions + row_base, num_rows * sizeof(int32_t));
                    asc_copy_gm2ub_align(metadata + kNumRows, slots + row_base, num_rows * sizeof(int32_t));
                    notify<PIPE_MTE2, PIPE_S>(kMetadataFull);
                    wait<PIPE_MTE2, PIPE_S>(kMetadataFull);
                    for (uint32_t row_idx = 0; row_idx < num_rows; ++ row_idx) {
                        const int32_t position = uint32_t(metadata[kNumRows + row_idx]) < num_slots ? metadata[row_idx] : 0;
                        asc_copy_gm2ub_align(Layout::kCosSin.ptr() + row_idx * kRopeDim, cos_sin + uint64_t(position) * kRopeDim, kRopeDim * sizeof(float));
                    }
                }
            };

            // Preload normalization weights, RoPE tables and first Q weight block before QKV arrives
            const uint32_t num_rows_per_core = ceil_div(kSingleMBlock ? shape_m : BLOCK_M, kNumAICores);
            const uint32_t first_row = core_idx * num_rows_per_core;
            if (first_row < shape_m) {
                asc_copy_gm2ub_align(Layout::kNormWeight.ptr(), kIsQ ? qw : kw, kLatentDim * sizeof(bfloat16_t));
                load_rope(first_row, min(num_rows_per_core, shape_m - first_row));
            }
            prestage_q_weight<BLOCK_M, BLOCK_N, kL2CtrlB>(gm_b.ptr(), core_idx, subblock_idx);

            // Transform QKV activations once every AIC finishes its stores
            wait_intra_block(PIPE_MTE2, subblock_idx, kQKVReadyEvent);
            if (first_row >= shape_m) {
                if constexpr (kIsQ)
                    set_intra_block(PIPE_MTE3, subblock_idx, kQFirstReadyEvent);
                prefetch_q_weight<BLOCK_N, kL2CtrlB>(gm_b.ptr(), core_idx, subblock_idx);
            }

            for (uint32_t row_base = first_row; row_base < shape_m; row_base += kNumAICores * num_rows_per_core) {
                const uint32_t num_rows = min(num_rows_per_core, shape_m - row_base);
                if (not kSingleMBlock and row_base > first_row)
                    load_rope(row_base, num_rows);

                // Load QKV activations and overlap second-block weight prefetch
                copy_gm_to_ub(Layout::kInput, qkv + uint64_t(row_base) * kQKVDim + (kIsQ ? 0 : kQLoraDim), num_rows,
                              kLatentDim * sizeof(bfloat16_t), kQKVDim * sizeof(bfloat16_t), kLatentDim * sizeof(bfloat16_t));
                notify<PIPE_MTE2, PIPE_V>(kTransformFull);
                if (row_base == first_row)
                    prefetch_q_weight<BLOCK_N, kL2CtrlB>(gm_b.ptr(), core_idx, subblock_idx);
                wait<PIPE_MTE2, PIPE_V>(kTransformFull);

                // Publish the first Q K-block immediately so AIC overlaps L1A startup with remaining quantization
                if constexpr (kIsQ) {
                    const auto store_q_slice = [&](uint32_t h_begin, uint32_t block_h) __aicore__ {
                        const uint32_t slice_sf_k = block_h / MX_SF_DIVISOR, sf_begin = h_begin / MX_SF_DIVISOR;
                        notify<PIPE_V, PIPE_MTE3>(kTransformDone);
                        wait<PIPE_V, PIPE_MTE3>(kTransformDone);
                        asc_copy_ub2gm_align(qa + uint64_t(h_begin) * shape_m + uint64_t(row_base) * kDataBlockBytes,
                                             Layout::kOutput.ptr() + h_begin * (num_rows | 1),
                                             block_h / kDataBlockBytes, num_rows * kDataBlockBytes,
                                             asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                                             uint64_t(shape_m) * kDataBlockBytes, (num_rows | 1) * kDataBlockBytes);
                        asc_copy_ub2gm_align(qa_sf + uint64_t(sf_begin) * shape_m + uint64_t(row_base) * slice_sf_k,
                                             reinterpret_cast<__ubuf__ int16_t*>(Layout::kSF.ptr() + kNumRows * sf_begin), 1,
                                             num_rows * slice_sf_k * sizeof(int16_t), asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, 0, 0);
                    };
                    vf_transform<true, 0, BLOCK_K>(num_rows, eps_hi, eps_lo);
                    store_q_slice(0, BLOCK_K);
                    if (row_base == first_row)
                        set_intra_block(PIPE_MTE3, subblock_idx, kQFirstReadyEvent);
                    vf_transform<true, BLOCK_K>(num_rows, eps_hi, eps_lo);
                    store_q_slice(BLOCK_K, kQLoraDim - BLOCK_K);
                } else {
                    vf_transform<false>(num_rows, eps_hi, eps_lo);
                    notify<PIPE_V, PIPE_MTE3>(kTransformDone);
                    wait<PIPE_V, PIPE_MTE3>(kTransformDone);
                    for (uint32_t row_idx = 0; row_idx < num_rows; ++ row_idx) {
                        const uint32_t slot_idx = metadata[kNumRows + row_idx];
                        if (slot_idx < num_slots) {
                            const uint64_t offset_bytes = uint64_t(slot_idx / cache_block_size) * cache_page_stride_bytes +
                                                          uint64_t(slot_idx % cache_block_size) * kKVCacheSlotBytes;
                            copy_ub_to_gm(cache + offset_bytes, reinterpret_cast<__ubuf__ uint8_t*>(Layout::kOutput.ptr() + row_idx * kOutputStride),
                                          1, kKVCacheSlotBytes, kKVCacheSlotBytes);
                        }
                    }
                }
                notify<PIPE_MTE3, PIPE_MTE2>(kStoresDone);
                wait<PIPE_MTE3, PIPE_MTE2>(kStoresDone);
            }

            if constexpr (kIsQ)
                set_intra_block(PIPE_MTE3, subblock_idx, kQReadyEvent);
            stage_q_weight<SHAPE_N, BLOCK_M, BLOCK_N, kNumL1Stages, kSingleMBlock, kL2CtrlB>(gm_b.ptr(), shape_m);
        };
        if (subblock_idx == 0)
            run_transform.template operator()<true>();
        else
            run_transform.template operator()<false>();
    }
}

template <uint32_t SHAPE_N, uint32_t BLOCK_M, bool kSingleMBlock>
__global__ __mix__(1, 2) void v41f_qkv_lora_kernel(QKVLoRAArgs) {
    asc_init();
    const auto& [x, x_sf, wa, wa_sf, qkv, wb, shape_m, x_sf_stride, eps_hi, eps_lo,
                 qa, qa_sf, wb_sf, q, qw, kw, cos_sin, positions, slots, cache,
                 num_slots, cache_block_size, cache_page_stride_bytes] = get_replicated_args<QKVLoRAArgs>();

    // Run QKV down projection and publish intermediate activations
    qkv_projection<BLOCK_M, kSingleMBlock>(x, wa, x_sf, wa_sf, qkv, shape_m, x_sf_stride);

    // Normalize Q/KV, update the cache and run Q up projection
    q_projection<SHAPE_N, BLOCK_M, kSingleMBlock>(
        qa, wb, qa_sf, wb_sf, q, shape_m, qkv, qw, kw, cos_sin, positions, slots, cache,
        num_slots, cache_block_size, cache_page_stride_bytes, eps_hi, eps_lo);
}

} // namespace deep_gemm::experimental
#endif
