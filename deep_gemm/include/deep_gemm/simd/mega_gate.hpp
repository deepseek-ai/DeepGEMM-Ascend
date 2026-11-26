#pragma once

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/layout/mega_gate.hpp>

#include <c_api/asc_simd.h>

namespace deep_gemm::mega_gate {

template <uint32_t SHAPE_N, uint32_t kStride, uint32_t kNumKSplits, uint32_t kUnroll, bool kMaskRows = false>
__simd_callee__ inline void score(__ubuf__ float* values, uint32_t num_rows, __ubuf__ uint32_t* status, uint64_t active_rows = UINT64_MAX) {
    constexpr uint32_t kNumVectorElements = asc_get_vf_len() / sizeof(float);
    const auto mask = asc_create_mask_b32(PAT_ALL);
    vector_float one;
    asc_duplicate_scalar(one, 1.0f);
    for (uint32_t row_idx = 0; row_idx < num_rows; ++ row_idx) {
        if constexpr (kMaskRows)
            if (not ((active_rows >> row_idx) & 1)) continue;
        vector_uint32_t errors, zeros, ones;
        asc_duplicate_scalar(zeros, 0u);
        asc_duplicate_scalar(ones, 1u);
        errors = zeros;
        // Bisheng miscompiles the masked-row loop when unrolling an incomplete vector group.
        #pragma clang loop unroll_count(SHAPE_N % (kNumVectorElements * kUnroll) == 0 ? kUnroll : 1)
        for (uint32_t expert_idx = 0; expert_idx < SHAPE_N; expert_idx += kNumVectorElements) {
            vector_float value, invalid;
            // Read token-major partials in a fixed order, then compact scores in place.
            auto partial_ptr = values + row_idx * kNumKSplits * kStride + expert_idx;
            asc_loadalign_postupdate(value, partial_ptr, kStride);
            #pragma clang loop unroll(disable)
            for (uint32_t split_idx = 1; split_idx < kNumKSplits; ++ split_idx) {
                vector_float partial;
                asc_loadalign_postupdate(partial, partial_ptr, kStride);
                asc_add(value, value, partial, mask);
            }
            asc_sub(invalid, value, value, mask);
            vector_uint32_t bad;
            vector_bool finite;
            asc_eq(finite, invalid, invalid, mask);
            asc_select(bad, zeros, ones, finite);
            uint32_t count = SHAPE_N - expert_idx;
            asc_select(bad, bad, zeros, asc_update_mask_b32(count));
            asc_add(errors, errors, bad, mask);
            // Same sqrt(log1p(q)/q) polynomial as tile-kernels' Ascend gate.
            vector_float a, t, p, negative, positive;
            vector_bool is_positive;
            asc_gt_scalar(is_positive, value, 0.0f, mask);
            asc_abs(a, value, mask);
            asc_mul_scalar(a, a, -0.5f, mask);
            asc_exp(a, a, mask);
            asc_mul(t, a, a, mask);
            asc_sub(t, one, t, mask);

            // Horner evaluation.
            constexpr float kCoefficients[]{
                -0.0004226923920214176f, -8.536428504157811e-05f, 0.002192076062783599f,
                0.000292900949716568f, 0.004986039828509092f, 0.011144610121846199f,
                0.03287079185247421f, 0.11599518358707428f, 0.8325546383857727f};
            asc_duplicate_scalar(p, 0.000471802573883906f);
            #pragma clang loop unroll(full)
            for (const float c : kCoefficients) {
                vector_float coefficient;
                asc_duplicate_scalar(coefficient, c);
                asc_madd(p, t, coefficient, mask);
            }

            asc_mul(negative, a, p, mask);
            asc_relu(positive, value, mask);
            asc_mula(positive, negative, negative, mask);
            asc_sqrt(positive, positive, mask);
            asc_select(value, positive, negative, is_positive);
            asc_add(value, value, invalid, mask);
            uint32_t remaining = SHAPE_N - expert_idx;
            asc_storealign(values + row_idx * kStride + expert_idx, value, asc_update_mask_b32(remaining));
        }
        asc_reduce_sum(errors, errors, mask);
        asc_storealign_1st(status + row_idx, errors);
    }
}

template <uint32_t SHAPE_N, uint32_t kNumTopk, bool kFixed>
__simd_callee__ inline vector_int32_t select_topk(
    __ubuf__ float* scores, __ubuf__ float* bias, bool has_bias, __ubuf__ float* ranking,
    __ubuf__ int64_t* fixed, vector_uint32_t& errors) {
    constexpr uint32_t kNumElements = asc_get_vf_len() / sizeof(float);
    const auto mask = asc_create_mask_b32(PAT_ALL);
    uint32_t count = 1;
    const auto first = asc_update_mask_b32(count);
    count = ceil_div(SHAPE_N, kNumElements);
    const auto column_mask = asc_update_mask_b32(count);
    vector_int32_t lanes, column_offsets, lane_mask, invalid, result, candidates;
    vector_uint32_t zeros, ones;
    vector_float negative_inf, maxima;
    asc_arange(lanes, 0);
    asc_duplicate_scalar(zeros, 0u);
    asc_duplicate_scalar(ones, 1u);
    asc_duplicate_scalar(invalid, INT32_MAX);
    asc_duplicate_scalar(result, -1);
    asc_duplicate_scalar(negative_inf, -__builtin_inff());
    asc_duplicate_scalar(lane_mask, int32_t(kNumElements - 1));
    asc_shiftleft_scalar(column_offsets, lanes, __builtin_ctz(kNumElements), mask);
    maxima = negative_inf;
    candidates = lanes;
    if constexpr (not kFixed) {
        #pragma clang loop unroll(disable)
        for (uint32_t offset = 0; offset < SHAPE_N; offset += kNumElements) {
            vector_float value, ranking_bias;
            vector_bool valid_number;
            uint32_t remaining = SHAPE_N - offset;
            const auto valid = asc_update_mask_b32(remaining);
            asc_loadalign(value, scores + offset);
            if (has_bias) {
                asc_loadalign(ranking_bias, bias + offset);
                asc_add(value, value, ranking_bias, mask);
            }
            asc_eq(valid_number, value, value, valid);
            asc_select(value, value, negative_inf, valid_number);
            asc_storealign(ranking + offset, value, mask);
            vector_int32_t indices;
            vector_bool better;
            asc_add_scalar(indices, lanes, int32_t(offset), mask);
            asc_gt(better, value, maxima, mask);
            asc_select(maxima, value, maxima, better);
            asc_select(candidates, indices, candidates, better);
        }
    }
    if constexpr (kFixed) {
        vector_int32_t packed, high;
        vector_uint32_t bad;
        vector_bool valid;
        count = kNumTopk;
        const auto selected_mask = asc_update_mask_b32(count);
        asc_loadalign(packed, reinterpret_cast<__ubuf__ int32_t*>(fixed));
        asc_deintlv(result, high, packed, reinterpret_cast<vector_int32_t&>(zeros));
        asc_eq_scalar(valid, high, 0, selected_mask);
        asc_lt_scalar(valid, reinterpret_cast<vector_uint32_t&>(result), SHAPE_N, valid);
        asc_select(bad, zeros, ones, valid);
        asc_select(bad, bad, zeros, selected_mask);
        asc_add(errors, errors, bad, mask);
        asc_min_scalar(reinterpret_cast<vector_uint32_t&>(result), reinterpret_cast<vector_uint32_t&>(result), SHAPE_N - 1, mask);
    } else {
        asc_mem_bar(VST_VLD);
        #pragma clang loop unroll(disable)
        for (uint32_t topk_idx = 0; topk_idx < kNumTopk; ++ topk_idx) {
            vector_float maximum;
            vector_int32_t best_idx;
            vector_bool equal, output_slot;
            asc_reduce_max(maximum, maxima, mask);
            asc_duplicate(maximum, maximum, mask);
            asc_eq(equal, maxima, maximum, mask);
            asc_select(best_idx, candidates, invalid, equal);
            asc_reduce_min(best_idx, best_idx, mask);
            asc_duplicate(best_idx, best_idx, mask);
            asc_eq_scalar(output_slot, lanes, int32_t(topk_idx), mask);
            asc_select(result, best_idx, result, output_slot);
            if (topk_idx + 1 < kNumTopk) {
                vector_int32_t winner_lane, indices, next_idx;
                vector_float values, next_max;
                vector_bool winner;
                asc_scatter(ranking, negative_inf, reinterpret_cast<vector_uint32_t&>(best_idx), first);
                asc_mem_bar(VST_VLD);
                asc_and(winner_lane, best_idx, lane_mask, mask);
                asc_add(indices, column_offsets, winner_lane, mask);
                asc_gather(values, ranking, reinterpret_cast<vector_uint32_t&>(indices), column_mask);
                asc_reduce_max(next_max, values, column_mask);
                asc_duplicate(next_max, next_max, mask);
                asc_eq(equal, values, next_max, column_mask);
                asc_select(next_idx, indices, invalid, equal);
                asc_reduce_min(next_idx, next_idx, mask);
                asc_duplicate(next_idx, next_idx, mask);
                asc_eq(winner, lanes, winner_lane, mask);
                asc_select(maxima, next_max, maxima, winner);
                asc_select(candidates, next_idx, candidates, winner);
            }
        }
    }
    return result;
}

// Store the first 32 signed indices as int64.
__simd_callee__ inline void store_indices(__ubuf__ int32_t* output, vector_int32_t low, vector_int32_t high, vector_bool mask) {
    vector_int32_t packed, unused;
    asc_intlv(packed, unused, low, high);
    asc_storealign(output, packed, mask);
}

template <uint32_t kNumCached>
__simd_callee__ inline void map_experts(
    __ubuf__ int32_t* map, __ubuf__ uint32_t* counts, __ubuf__ layout::GateArgs* args,
    vector_int32_t& logical, vector_int32_t& high, vector_uint32_t& errors,
    vector_uint32_t& missing, vector_uint32_t token, vector_bool physical_mask) {
    const auto mask = asc_create_mask_b32(PAT_ALL);
    vector_uint32_t zeros, ones;
    asc_duplicate_scalar(zeros, 0u);
    asc_duplicate_scalar(ones, 1u);
    vector_uint32_t duplicates, max_duplicates, rank, quotient, replica, offset, bad;
    vector_bool valid, hit;
    asc_gather(duplicates, counts, reinterpret_cast<vector_uint32_t&>(logical), physical_mask);
    asc_loadalign_brc(max_duplicates, &args->num_duplicate_experts);
    asc_gt_scalar(valid, duplicates, 0u, physical_mask);
    asc_le(valid, duplicates, max_duplicates, valid);
    asc_select(bad, zeros, ones, valid);
    asc_select(bad, bad, zeros, physical_mask);
    asc_add(errors, errors, bad, mask);
    asc_max_scalar(duplicates, duplicates, 1u, mask);
    replica = zeros;
    if constexpr (kNumCached != 0) {
        // Reduce before multiplying; cached counts fit UB.
        static_assert(uint64_t(kNumCached) * 23333u + INT32_MAX <= UINT32_MAX);
        asc_div(quotient, token, duplicates, physical_mask);
        asc_mul(quotient, quotient, duplicates, physical_mask);
        asc_sub(token, token, quotient, physical_mask);
        asc_mul_scalar(token, token, 23333u, physical_mask);
        asc_loadalign_brc(rank, &args->ep_rank);
        asc_add(token, token, rank, physical_mask);
        asc_div(quotient, token, duplicates, physical_mask);
        asc_mul(quotient, quotient, duplicates, physical_mask);
        asc_sub(replica, token, quotient, physical_mask);
    }
    asc_lt_scalar(hit, replica, kNumCached, physical_mask);
    vector_int32_t mapped, shared;
    mapped = reinterpret_cast<vector_int32_t&>(zeros);
    if constexpr (kNumCached != 0) {
        asc_mul_scalar(offset, reinterpret_cast<vector_uint32_t&>(logical), kNumCached, physical_mask);
        asc_add(offset, offset, replica, physical_mask);
        asc_select(offset, offset, zeros, hit);
        asc_gather(mapped, map, offset, physical_mask);
    }
    asc_shiftright_scalar(high, mapped, 31, mask);
    asc_add_scalar(shared, logical, 1, mask);
    asc_select(high, high, shared, hit);
    asc_select(logical, mapped, reinterpret_cast<vector_int32_t&>(replica), hit);
    asc_select(missing, zeros, ones, hit);
    asc_select(missing, missing, zeros, physical_mask);
}

template <uint32_t SHAPE_N, uint32_t kNumTopk, uint32_t kNumShared, uint32_t kNumCached,
          bool kHasPhysicalMap, bool kHasUnmappedOutput, bool kFixed,
          uint32_t kScoreStride = 0, uint32_t kNumKSplits = 1>
__simd_vf__ inline void vf_gate(
    __ubuf__ float* scores, __ubuf__ float* bias, bool has_bias, __ubuf__ int32_t* output,
    __ubuf__ float* ranking, __ubuf__ int32_t* map, __ubuf__ uint32_t* counts,
    __ubuf__ layout::GateArgs* args, __ubuf__ int64_t* fixed, __ubuf__ uint32_t* status, uint32_t token_idx) {
    if constexpr (kScoreStride != 0) {
        score<SHAPE_N, kScoreStride, kNumKSplits, 1>(scores, 1, status);
        asc_mem_bar(VST_VLD);
    }
    constexpr uint32_t kNumPhysicalTopk = kNumTopk + kNumShared;
    constexpr uint32_t kWeightsOffset = aligned(kNumPhysicalTopk * sizeof(int64_t), 32u) / sizeof(float);
    constexpr uint32_t kLogicalOffset = kWeightsOffset + aligned(kNumPhysicalTopk * sizeof(float), 32u) / sizeof(float);
    const auto mask = asc_create_mask_b32(PAT_ALL);
    uint32_t count = kNumTopk;
    const auto top_mask = asc_update_mask_b32(count);
    count = kNumPhysicalTopk;
    const auto physical_mask = asc_update_mask_b32(count);
    count = 2 * kNumPhysicalTopk;
    const auto index_mask = asc_update_mask_b32(count);
    vector_uint32_t zeros, errors, missing;
    asc_duplicate_scalar(zeros, 0u);
    errors = missing = zeros;
    vector_uint32_t recorded;
    asc_loadalign_brc(recorded, status);
    count = 1;
    asc_select(errors, recorded, zeros, asc_update_mask_b32(count));
    vector_int32_t logical = select_topk<SHAPE_N, kNumTopk, kFixed>(scores, bias, has_bias, ranking, fixed, errors);

    // Unbiased normalization and shared slots.
    vector_float weights, denominator, scale, one;
    asc_gather(weights, scores, reinterpret_cast<vector_uint32_t&>(logical), top_mask);
    asc_reduce_sum(denominator, weights, top_mask);
    asc_duplicate(denominator, denominator, mask);
    asc_add_scalar(denominator, denominator, 1e-20f, mask);
    asc_loadalign_brc(scale, &args->routed_scaling_factor);
    asc_div(weights, weights, denominator, top_mask);
    asc_mul(weights, weights, scale, top_mask);
    asc_duplicate_scalar(one, 1.0f);
    asc_select(weights, weights, one, top_mask);
    asc_storealign(reinterpret_cast<__ubuf__ float*>(output + kWeightsOffset), weights, physical_mask);
    if constexpr (kHasUnmappedOutput) {
        count = 2 * kNumTopk;
        store_indices(output + kLogicalOffset, logical, reinterpret_cast<vector_int32_t&>(zeros), asc_update_mask_b32(count));
    }
    vector_int32_t shared, high;
    asc_arange(shared, int32_t(SHAPE_N - kNumTopk));
    asc_select(logical, logical, shared, top_mask);
    high = reinterpret_cast<vector_int32_t&>(zeros);

    // Gather the selected physical replicas.
    if constexpr (kHasPhysicalMap) {
        vector_uint32_t token;
        asc_duplicate_scalar(token, token_idx);
        map_experts<kNumCached>(map, counts, args, logical, high, errors, missing, token, physical_mask);
    }
    store_indices(output, logical, high, index_mask);
    asc_reduce_sum(errors, errors, mask);
    asc_mul_scalar(errors, errors, 2u, mask);
    asc_reduce_max(missing, missing, mask);
    asc_add(errors, errors, missing, mask);
    asc_storealign_1st(status, errors);
}

template <uint32_t kNumTopk, uint32_t kNumShared, bool kHasUnmappedOutput>
__simd_vf__ inline void vf_masked(__ubuf__ int32_t* output, __ubuf__ uint32_t* status) {
    constexpr uint32_t kNumPhysicalTopk = kNumTopk + kNumShared;
    constexpr uint32_t kWeightsOffset = aligned(kNumPhysicalTopk * sizeof(int64_t), 32u) / sizeof(float);
    constexpr uint32_t kLogicalOffset = kWeightsOffset + aligned(kNumPhysicalTopk * sizeof(float), 32u) / sizeof(float);
    vector_int32_t invalid, zeros;
    asc_duplicate_scalar(invalid, -1);
    asc_duplicate_scalar(zeros, 0);
    uint32_t count = 2 * kNumPhysicalTopk;
    asc_storealign(output, invalid, asc_update_mask_b32(count));
    count = kNumPhysicalTopk;
    asc_storealign(output + kWeightsOffset, zeros, asc_update_mask_b32(count));
    if constexpr (kHasUnmappedOutput) {
        count = 2 * kNumTopk;
        asc_storealign(output + kLogicalOffset, invalid, asc_update_mask_b32(count));
    }
    asc_storealign_1st(status, reinterpret_cast<vector_uint32_t&>(zeros));
}

template <uint32_t SHAPE_N, uint32_t kNumTopk, uint32_t kNumShared, bool kHasPhysicalMap, bool kHasUnmappedOutput>
__simd_vf__ inline void vf_random(__ubuf__ int32_t* output, __ubuf__ uint32_t* counts,
                                __ubuf__ uint32_t* randoms, __ubuf__ uint32_t* status) {
    constexpr uint32_t kNumPhysicalTopk = kNumTopk + kNumShared;
    constexpr uint32_t kNumElements = asc_get_vf_len() / sizeof(float);
    constexpr uint32_t kWeightsOffset = aligned(kNumPhysicalTopk * sizeof(int64_t), 32u) / sizeof(float);
    constexpr uint32_t kLogicalOffset = kWeightsOffset + aligned(kNumPhysicalTopk * sizeof(float), 32u) / sizeof(float);
    const auto mask = asc_create_mask_b32(PAT_ALL);
    uint32_t count = kNumPhysicalTopk;
    const auto top_mask = asc_update_mask_b32(count);
    vector_uint32_t lanes, zeros, invalid, total, errors;
    asc_arange(reinterpret_cast<vector_int32_t&>(lanes), 0);
    asc_duplicate_scalar(zeros, 0u);
    asc_duplicate_scalar(invalid, uint32_t(INT32_MAX));
    if constexpr (kHasPhysicalMap) {
        total = zeros;
        #pragma clang loop unroll(disable)
        for (uint32_t offset = 0; offset < SHAPE_N + kNumShared; offset += kNumElements) {
            vector_uint32_t value;
            uint32_t remaining = SHAPE_N + kNumShared - offset;
            asc_loadalign(value, counts + offset);
            asc_select(value, value, zeros, asc_update_mask_b32(remaining));
            asc_add(total, total, value, mask);
        }
        asc_reduce_sum(total, total, mask);
        asc_duplicate(total, total, mask);
    } else {
        asc_duplicate_scalar(total, SHAPE_N + kNumShared);
    }
    vector_bool valid;
    asc_ge_scalar(valid, total, kNumPhysicalTopk, mask);
    asc_duplicate_scalar(errors, 2u);
    asc_select(errors, zeros, errors, valid);
    asc_storealign_1st(status, errors);
    asc_max_scalar(total, total, kNumPhysicalTopk, mask);

    vector_uint32_t hash, divisor, quotient, candidate;
    asc_loadalign(hash, randoms);
    asc_shiftright_scalar(hash, hash, 9, mask);
    vector_float weights;
    asc_int322float_rn(weights, reinterpret_cast<vector_int32_t&>(hash), mask);
    asc_add_scalar(weights, weights, 0.5f, mask);
    asc_mul_scalar(weights, weights, 0x1p-23f, mask);
    asc_storealign(reinterpret_cast<__ubuf__ float*>(output + kWeightsOffset), weights, top_mask);
    asc_loadalign(hash, randoms + aligned(kNumPhysicalTopk, 8u));
    asc_sub(divisor, total, lanes, top_mask);
    asc_div(quotient, hash, divisor, top_mask);
    asc_mul(quotient, quotient, divisor, top_mask);
    asc_sub(candidate, hash, quotient, top_mask);
    asc_select(candidate, candidate, invalid, top_mask);
    vector_int32_t selected;
    asc_duplicate_scalar(selected, -1);

    #pragma clang loop unroll(disable)
    for (uint32_t topk_idx = 0; topk_idx < kNumPhysicalTopk; ++ topk_idx) {
        vector_uint32_t minimum, min_lane, next;
        vector_bool equal, active, increment, winner;
        asc_reduce_min(minimum, candidate, top_mask);
        asc_duplicate(minimum, minimum, mask);
        asc_eq(equal, candidate, minimum, top_mask);
        asc_select(min_lane, lanes, invalid, equal);
        asc_reduce_min(min_lane, min_lane, top_mask);
        asc_duplicate(min_lane, min_lane, mask);
        asc_eq_scalar(active, selected, -1, top_mask);
        asc_ge(increment, candidate, minimum, active);
        asc_lt(increment, min_lane, lanes, increment);
        asc_add_scalar(next, candidate, 1u, mask);
        asc_select(candidate, next, candidate, increment);
        asc_eq(winner, lanes, min_lane, active);
        asc_select(selected, reinterpret_cast<vector_int32_t&>(candidate), selected, winner);
        asc_select(candidate, invalid, candidate, winner);
    }
    vector_int32_t high;
    asc_shiftright_scalar(high, selected, 31, mask);
    count = 2 * kNumPhysicalTopk;
    store_indices(output, selected, high, asc_update_mask_b32(count));
    if constexpr (kHasUnmappedOutput) {
        asc_duplicate_scalar(selected, -1);
        count = 2 * kNumTopk;
        asc_storealign(output + kLogicalOffset, selected, asc_update_mask_b32(count));
    }
}

template <uint32_t kWidth, typename T>
__simd_callee__ inline void row_max(T& value, vector_uint32_t lanes) {
    vector_uint32_t rows;
    asc_shiftright_scalar(rows, lanes, 3, asc_create_mask_b32(PAT_ALL));
    asc_reduce_max_datablock(value, value, asc_create_mask_b32(PAT_ALL));
    asc_gather(value, value, rows);
    #pragma clang loop unroll(full)
    for (uint32_t shift = 8; shift < kWidth; shift <<= 1) {
        vector_uint32_t index, bits;
        asc_duplicate_scalar(bits, shift);
        asc_xor(index, lanes, bits, asc_create_mask_b32(PAT_ALL));
        T other;
        asc_gather(other, value, index);
        asc_max(value, value, other, asc_create_mask_b32(PAT_ALL));
    }
}

template <uint32_t kWidth>
__simd_callee__ inline void row_min(vector_uint32_t& value, vector_uint32_t lanes) {
    vector_uint32_t rows;
    asc_shiftright_scalar(rows, lanes, 3, asc_create_mask_b32(PAT_ALL));
    asc_reduce_min_datablock(value, value, asc_create_mask_b32(PAT_ALL));
    asc_gather(value, value, rows);
    #pragma clang loop unroll(full)
    for (uint32_t shift = 8; shift < kWidth; shift <<= 1) {
        vector_uint32_t index, bits;
        asc_duplicate_scalar(bits, shift);
        asc_xor(index, lanes, bits, asc_create_mask_b32(PAT_ALL));
        vector_uint32_t other;
        asc_gather(other, value, index);
        asc_min(value, value, other, asc_create_mask_b32(PAT_ALL));
    }
}

template <uint32_t kWidth, typename T>
__simd_callee__ inline void row_sum(T& value, vector_uint32_t lanes) {
    vector_uint32_t rows;
    asc_shiftright_scalar(rows, lanes, 3, asc_create_mask_b32(PAT_ALL));
    asc_reduce_sum_datablock(value, value, asc_create_mask_b32(PAT_ALL));
    asc_gather(value, value, rows);
    #pragma clang loop unroll(full)
    for (uint32_t shift = 8; shift < kWidth; shift <<= 1) {
        vector_uint32_t index, bits;
        asc_duplicate_scalar(bits, shift);
        asc_xor(index, lanes, bits, asc_create_mask_b32(PAT_ALL));
        T other;
        asc_gather(other, value, index);
        asc_add(value, value, other, asc_create_mask_b32(PAT_ALL));
    }
}

template <uint32_t SHAPE_N, uint32_t kNumTopk, uint32_t kNumShared,
          bool kHasPhysicalMap, bool kHasUnmappedOutput, uint32_t kWidth>
__simd_callee__ inline void random_packed(__ubuf__ int32_t* output, __ubuf__ uint32_t* counts,
                                __ubuf__ uint32_t* randoms, __ubuf__ uint32_t* status, uint32_t num_rows, __ubuf__ uint32_t* row_list) {
    constexpr uint32_t kNumPhysicalTopk = kNumTopk + kNumShared;
    constexpr uint32_t kNumElements = asc_get_vf_len() / sizeof(float);
    constexpr uint32_t kWeightsOffset = aligned(kNumPhysicalTopk * sizeof(int64_t), 32u) / sizeof(float);
    constexpr uint32_t kLogicalOffset = kWeightsOffset + aligned(kNumPhysicalTopk * sizeof(float), 32u) / sizeof(float);
    constexpr uint32_t kOutputStride = kLogicalOffset + (kHasUnmappedOutput ? aligned(kNumTopk * sizeof(int64_t), 32u) / sizeof(float) : 0);
    uint32_t count = num_rows * kWidth;
    const auto mask = asc_update_mask_b32(count);
    vector_uint32_t rows, columns, base, address, random_address;
    vector_bool top_mask, first;
    vector_uint32_t lanes, zeros, invalid, total, errors;
    asc_arange(reinterpret_cast<vector_int32_t&>(lanes), 0);
    asc_shiftright_scalar(rows, lanes, __builtin_ctz(kWidth), mask);
    asc_gather(rows, row_list, rows, mask);
    asc_duplicate_scalar(columns, kWidth - 1);
    asc_and(columns, lanes, columns, mask);
    asc_mul_scalar(base, rows, kOutputStride, mask);
    asc_mul_scalar(random_address, rows, 2 * aligned(kNumPhysicalTopk, 8u), mask);
    asc_add(random_address, random_address, columns, mask);
    asc_lt_scalar(top_mask, columns, kNumPhysicalTopk, mask);
    asc_eq_scalar(first, columns, 0u, mask);
    asc_duplicate_scalar(zeros, 0u);
    asc_duplicate_scalar(invalid, uint32_t(INT32_MAX));
    if constexpr (kHasPhysicalMap) {
        total = zeros;
        #pragma clang loop unroll(disable)
        for (uint32_t offset = 0; offset < SHAPE_N + kNumShared; offset += kNumElements) {
            vector_uint32_t value;
            uint32_t remaining = SHAPE_N + kNumShared - offset;
            asc_loadalign(value, counts + offset);
            asc_select(value, value, zeros, asc_update_mask_b32(remaining));
            asc_add(total, total, value, asc_create_mask_b32(PAT_ALL));
        }
        asc_reduce_sum(total, total, asc_create_mask_b32(PAT_ALL));
        asc_duplicate(total, total, mask);
    } else {
        asc_duplicate_scalar(total, SHAPE_N + kNumShared);
    }
    vector_bool valid;
    asc_ge_scalar(valid, total, kNumPhysicalTopk, mask);
    asc_duplicate_scalar(errors, 2u);
    asc_select(errors, zeros, errors, valid);
    asc_scatter(status, errors, rows, first);
    asc_max_scalar(total, total, kNumPhysicalTopk, mask);

    vector_uint32_t hash, divisor, quotient, candidate;
    asc_gather(hash, randoms, random_address, top_mask);
    asc_shiftright_scalar(hash, hash, 9, mask);
    vector_float weights;
    asc_int322float_rn(weights, reinterpret_cast<vector_int32_t&>(hash), mask);
    asc_add_scalar(weights, weights, 0.5f, mask);
    asc_mul_scalar(weights, weights, 0x1p-23f, mask);
    asc_add(address, base, columns, mask);
    asc_add_scalar(address, address, kWeightsOffset, mask);
    asc_scatter(reinterpret_cast<__ubuf__ float*>(output), weights, address, top_mask);
    asc_add_scalar(random_address, random_address, aligned(kNumPhysicalTopk, 8u), mask);
    asc_gather(hash, randoms, random_address, top_mask);
    asc_sub(divisor, total, columns, top_mask);
    asc_div(quotient, hash, divisor, top_mask);
    asc_mul(quotient, quotient, divisor, top_mask);
    asc_sub(candidate, hash, quotient, top_mask);
    asc_select(candidate, candidate, invalid, top_mask);
    vector_int32_t selected;
    asc_duplicate_scalar(selected, -1);

    #pragma clang loop unroll(disable)
    for (uint32_t topk_idx = 0; topk_idx < kNumPhysicalTopk; ++ topk_idx) {
        vector_uint32_t minimum, min_lane, next;
        vector_bool equal, active, increment, winner;
        minimum = candidate;
        row_min<kWidth>(minimum, lanes);
        asc_eq(equal, candidate, minimum, top_mask);
        asc_select(min_lane, columns, invalid, equal);
        row_min<kWidth>(min_lane, lanes);
        asc_eq_scalar(active, selected, -1, top_mask);
        asc_ge(increment, candidate, minimum, active);
        asc_lt(increment, min_lane, columns, increment);
        asc_add_scalar(next, candidate, 1u, mask);
        asc_select(candidate, next, candidate, increment);
        asc_eq(winner, columns, min_lane, active);
        asc_select(selected, reinterpret_cast<vector_int32_t&>(candidate), selected, winner);
        asc_select(candidate, invalid, candidate, winner);
    }
    asc_mul_scalar(address, columns, 2u, mask);
    asc_add(address, address, base, mask);
    asc_scatter(output, selected, address, top_mask);
    asc_add_scalar(address, address, 1u, mask);
    asc_scatter(reinterpret_cast<__ubuf__ uint32_t*>(output), zeros, address, top_mask);
    if constexpr (kHasUnmappedOutput) {
        asc_lt_scalar(top_mask, columns, kNumTopk, mask);
        asc_add_scalar(address, address, kLogicalOffset - 1, mask);
        asc_duplicate_scalar(selected, -1);
        asc_scatter(output, selected, address, top_mask);
        asc_add_scalar(address, address, 1u, mask);
        asc_scatter(output, selected, address, top_mask);
    }
}

template <uint32_t kNumTopk, uint32_t kNumShared, bool kHasUnmappedOutput, uint32_t kWidth>
__simd_callee__ inline void masked_packed(__ubuf__ int32_t* output, __ubuf__ uint32_t* status,
                                       __ubuf__ uint32_t* row_list, uint32_t num_rows) {
    constexpr uint32_t kNumPhysical = kNumTopk + kNumShared;
    constexpr uint32_t kWeightsOffset = aligned(kNumPhysical * sizeof(int64_t), 32u) / sizeof(float);
    constexpr uint32_t kLogicalOffset = kWeightsOffset + aligned(kNumPhysical * sizeof(float), 32u) / sizeof(float);
    constexpr uint32_t kOutputStride = kLogicalOffset + (kHasUnmappedOutput ? aligned(kNumTopk * sizeof(int64_t), 32u) / sizeof(float) : 0);
    uint32_t count = num_rows * kWidth;
    const auto mask = asc_update_mask_b32(count);
    vector_uint32_t lanes, rows, columns, base, address, zeros;
    vector_int32_t invalid;
    vector_bool selected, first;
    asc_arange(reinterpret_cast<vector_int32_t&>(lanes), 0);
    asc_shiftright_scalar(rows, lanes, __builtin_ctz(kWidth), mask);
    asc_gather(rows, row_list, rows, mask);
    asc_duplicate_scalar(columns, kWidth - 1);
    asc_and(columns, lanes, columns, mask);
    asc_duplicate_scalar(zeros, 0u);
    asc_duplicate_scalar(invalid, -1);
    asc_mul_scalar(base, rows, kOutputStride, mask);
    asc_lt_scalar(selected, columns, kNumPhysical, mask);
    asc_add(address, base, columns, mask);
    asc_add_scalar(address, address, kWeightsOffset, mask);
    asc_scatter(reinterpret_cast<__ubuf__ uint32_t*>(output), zeros, address, selected);
    asc_mul_scalar(address, columns, 2u, mask);
    asc_add(address, address, base, mask);
    asc_scatter(output, invalid, address, selected);
    asc_add_scalar(address, address, 1u, mask);
    asc_scatter(output, invalid, address, selected);
    if constexpr (kHasUnmappedOutput) {
        asc_lt_scalar(selected, columns, kNumTopk, mask);
        asc_add_scalar(address, address, kLogicalOffset - 1, mask);
        asc_scatter(output, invalid, address, selected);
        asc_add_scalar(address, address, 1u, mask);
        asc_scatter(output, invalid, address, selected);
    }
    asc_eq_scalar(first, columns, 0u, mask);
    asc_scatter(status, zeros, rows, first);
}

template <uint32_t SHAPE_N, uint32_t kWidth>
__simd_callee__ inline void scan_experts(
    __ubuf__ float* scores, __ubuf__ float* bias, bool has_bias, __ubuf__ float* ranking,
    vector_uint32_t& expert, vector_uint32_t& score_address, vector_uint32_t rank_base,
    vector_float& maximum, vector_uint32_t& candidate, vector_float negative_inf, vector_bool mask) {
    vector_uint32_t address;
    vector_float value;
    vector_bool valid, finite, better;
    asc_lt_scalar(valid, expert, SHAPE_N, mask);
    asc_gather_datablock(value, scores, score_address, mask);
    if (has_bias) {
        vector_float b;
        asc_gather(b, bias, expert, valid);
        asc_add(value, value, b, mask);
    }
    asc_eq(finite, value, value, valid);
    asc_select(value, value, negative_inf, finite);
    asc_add(address, rank_base, expert, mask);
    asc_scatter(ranking, value, address, mask);
    asc_gt(better, value, maximum, mask);
    asc_select(maximum, value, maximum, better);
    asc_select(candidate, expert, candidate, better);
    asc_add_scalar(expert, expert, kWidth, mask);
    asc_add_scalar(score_address, score_address, uint32_t(kWidth * sizeof(float)), mask);
}

template <uint32_t SHAPE_N, uint32_t kNumTopk, uint32_t kNumShared, uint32_t kNumCached,
          bool kHasPhysicalMap, bool kHasUnmappedOutput>
__simd_callee__ inline void store_gate_part(
    __ubuf__ int32_t* output, __ubuf__ int32_t* map, __ubuf__ uint32_t* counts, __ubuf__ layout::GateArgs* args,
    vector_uint32_t logical, vector_float weights, vector_float denominator, vector_uint32_t columns,
    vector_uint32_t rows, vector_uint32_t& errors, vector_uint32_t& missing, uint32_t token_idx, vector_bool mask, uint32_t offset) {
    constexpr uint32_t kNumPhysical = kNumTopk + kNumShared;
    constexpr uint32_t kWeightsOffset = aligned(kNumPhysical * sizeof(int64_t), 32u) / sizeof(float);
    constexpr uint32_t kLogicalOffset = kWeightsOffset + aligned(kNumPhysical * sizeof(float), 32u) / sizeof(float);
    constexpr uint32_t kOutputStride = kLogicalOffset + (kHasUnmappedOutput ? aligned(kNumTopk * sizeof(int64_t), 32u) / sizeof(float) : 0);
    vector_uint32_t slots, zeros, base, address, shared;
    vector_bool top_mask, physical_mask;
    asc_add_scalar(slots, columns, offset, mask);
    asc_lt_scalar(top_mask, slots, kNumTopk, mask);
    asc_lt_scalar(physical_mask, slots, kNumPhysical, mask);
    asc_duplicate_scalar(zeros, 0u);
    vector_float scale, one;
    asc_loadalign_brc(scale, &args->routed_scaling_factor);
    asc_div(weights, weights, denominator, top_mask);
    asc_mul(weights, weights, scale, top_mask);
    asc_duplicate_scalar(one, 1.0f);
    asc_select(weights, weights, one, top_mask);
    asc_mul_scalar(base, rows, kOutputStride, mask);
    asc_add(address, base, slots, mask);
    asc_add_scalar(address, address, kWeightsOffset, mask);
    asc_scatter(reinterpret_cast<__ubuf__ float*>(output), weights, address, physical_mask);
    asc_mul_scalar(address, slots, 2u, mask);
    asc_add(address, address, base, mask);
    if constexpr (kHasUnmappedOutput) {
        vector_uint32_t logical_address;
        asc_add_scalar(logical_address, address, kLogicalOffset, mask);
        asc_scatter(reinterpret_cast<__ubuf__ uint32_t*>(output), logical, logical_address, top_mask);
        asc_add_scalar(logical_address, logical_address, 1u, mask);
        asc_scatter(reinterpret_cast<__ubuf__ uint32_t*>(output), zeros, logical_address, top_mask);
    }
    asc_add_scalar(shared, slots, SHAPE_N - kNumTopk, mask);
    asc_select(logical, logical, shared, top_mask);
    vector_uint32_t high = zeros, part_missing = zeros;
    if constexpr (kHasPhysicalMap) {
        vector_uint32_t token;
        asc_add_scalar(token, rows, token_idx, mask);
        map_experts<kNumCached>(map, counts, args, reinterpret_cast<vector_int32_t&>(logical),
                               reinterpret_cast<vector_int32_t&>(high), errors, part_missing, token, physical_mask);
    }
    asc_scatter(reinterpret_cast<__ubuf__ uint32_t*>(output), logical, address, physical_mask);
    asc_add_scalar(address, address, 1u, mask);
    asc_scatter(reinterpret_cast<__ubuf__ uint32_t*>(output), high, address, physical_mask);
    asc_max(missing, missing, part_missing, mask);
}

template <uint32_t SHAPE_N, uint32_t kNumTopk, uint32_t kNumShared, uint32_t kStride, uint32_t kNumCached,
          bool kHasPhysicalMap, bool kHasUnmappedOutput, uint32_t kWidth, bool kIndirect = false, bool kFixed = false>
__simd_callee__ inline void gate_packed(
    __ubuf__ float* scores, __ubuf__ float* bias, bool has_bias, __ubuf__ int32_t* output,
    __ubuf__ float* ranking, __ubuf__ int32_t* map, __ubuf__ uint32_t* counts,
    __ubuf__ layout::GateArgs* args, __ubuf__ uint32_t* status, uint32_t token_idx, uint32_t num_rows,
    __ubuf__ uint32_t* row_list, __ubuf__ int32_t* fixed = nullptr) {
    constexpr uint32_t kRankStride = aligned(SHAPE_N, kWidth) + kWidth;
    constexpr uint32_t kNumPhysical = kNumTopk + kNumShared;
    uint32_t count = num_rows * kWidth;
    const auto mask = asc_update_mask_b32(count);
    vector_uint32_t lanes, rows, columns, zeros, ones, invalid, errors, score_base, rank_base;
    vector_float negative_inf, maxima;
    vector_uint32_t candidates, result, result_hi;
    asc_arange(reinterpret_cast<vector_int32_t&>(lanes), 0);
    asc_shiftright_scalar(rows, lanes, __builtin_ctz(kWidth), mask);
    asc_duplicate_scalar(columns, kWidth - 1);
    asc_and(columns, lanes, columns, mask);
    asc_duplicate_scalar(zeros, 0u);
    asc_duplicate_scalar(ones, 1u);
    asc_duplicate_scalar(invalid, uint32_t(INT32_MAX));
    asc_duplicate_scalar(negative_inf, -__builtin_inff());
    asc_mul_scalar(rank_base, rows, kRankStride, mask);
    if constexpr (kIndirect) asc_gather(rows, row_list, rows, mask);
    asc_mul_scalar(score_base, rows, kStride, mask);
    maxima = negative_inf;
    candidates = columns;
    result = result_hi = errors = zeros;
    // Retain both scan partitions; refresh only the winning partition.
    vector_float maxima0 = negative_inf, maxima1 = negative_inf;
    vector_uint32_t candidates0 = columns, candidates1 = columns;
    vector_uint32_t expert = columns, score_address, block_col;
    asc_shiftright_scalar(score_address, lanes, __builtin_ctz(kWidth / 8), mask);
    if constexpr (kIndirect) {
        uint32_t num_blocks = num_rows * (kWidth / 8);
        asc_gather(score_address, row_list, score_address, asc_update_mask_b32(num_blocks));
    }
    asc_mul_scalar(score_address, score_address, kStride, mask);
    asc_duplicate_scalar(block_col, kWidth / 8 - 1);
    asc_and(block_col, block_col, lanes, mask);
    asc_mul_scalar(block_col, block_col, 8u, mask);
    asc_add(score_address, score_address, block_col, mask);
    asc_mul_scalar(score_address, score_address, uint32_t(sizeof(float)), mask);
    if constexpr (not kFixed) {
        #pragma clang loop unroll(disable)
        for (uint32_t offset = 0; offset < SHAPE_N; offset += 2 * kWidth) {
            scan_experts<SHAPE_N, kWidth>(scores, bias, has_bias, ranking, expert, score_address, rank_base, maxima0,
                                          candidates0, negative_inf, mask);
            scan_experts<SHAPE_N, kWidth>(scores, bias, has_bias, ranking, expert, score_address, rank_base, maxima1,
                                          candidates1, negative_inf, mask);
        }
    }
    vector_bool first;
    asc_eq_scalar(first, columns, 0u, mask);
    asc_gather(errors, status, rows, first);
    asc_select(errors, errors, zeros, first);
    if constexpr (kFixed) {
        vector_uint32_t address, high, bad;
        vector_bool selected, valid;
        asc_mul_scalar(address, rows, aligned(kNumTopk * sizeof(int64_t), 32u) / sizeof(int32_t), mask);
        asc_mul_scalar(high, columns, 2u, mask);
        asc_add(address, address, high, mask);
        asc_lt_scalar(selected, columns, kNumTopk, mask);
        asc_gather(result, reinterpret_cast<__ubuf__ uint32_t*>(fixed), address, selected);
        asc_add_scalar(address, address, 1u, mask);
        asc_gather(high, reinterpret_cast<__ubuf__ uint32_t*>(fixed), address, selected);
        asc_eq_scalar(valid, high, 0u, selected);
        asc_lt_scalar(valid, result, SHAPE_N, valid);
        asc_select(bad, zeros, ones, valid);
        asc_select(bad, bad, zeros, selected);
        asc_add(errors, errors, bad, mask);
        asc_min_scalar(result, result, SHAPE_N - 1, mask);
    } else {
        asc_mem_bar(VST_VLD);
        #pragma clang loop unroll(disable)
        for (uint32_t topk_idx = 0; topk_idx < kNumTopk; ++ topk_idx) {
            vector_bool better, equal;
            asc_gt(better, maxima1, maxima0, mask);
            asc_select(maxima, maxima1, maxima0, better);
            asc_select(candidates, candidates1, candidates0, better);
            asc_eq(equal, maxima0, maxima1, mask);
            vector_uint32_t tie;
            asc_min(tie, candidates0, candidates1, mask);
            asc_select(candidates, tie, candidates, equal);
            vector_float maximum = maxima;
            vector_uint32_t best, address, winner_column, winner_offset;
            vector_bool slot, winner;
            row_max<kWidth>(maximum, lanes);
            asc_eq(equal, maxima, maximum, mask);
            asc_select(best, candidates, invalid, equal);
            row_min<kWidth>(best, lanes);
            asc_eq_scalar(slot, columns, topk_idx, mask);
            asc_select(result, best, result, slot);
            if constexpr (kNumTopk > kWidth) {
                vector_uint32_t columns_hi;
                asc_add_scalar(columns_hi, columns, kWidth, mask);
                asc_eq_scalar(slot, columns_hi, topk_idx, mask);
                asc_select(result_hi, best, result_hi, slot);
            }
            if (topk_idx + 1 < kNumTopk) {
                asc_duplicate_scalar(winner_offset, 2 * kWidth - 1);
                asc_and(winner_offset, best, winner_offset, mask);
                asc_duplicate_scalar(winner_column, kWidth - 1);
                asc_and(winner_column, best, winner_column, mask);
                vector_float next_max = negative_inf;
                vector_uint32_t next_idx = invalid;
                vector_uint32_t expert;
                asc_mul_scalar(expert, columns, 2 * kWidth, mask);
                asc_add(expert, expert, winner_offset, mask);
                #pragma clang loop unroll(disable)
                for (uint32_t offset = 0; offset < ceil_div(SHAPE_N, 2 * kWidth); offset += kWidth) {
                    vector_float value;
                    vector_bool valid, better;
                    asc_lt_scalar(valid, expert, SHAPE_N, mask);
                    asc_add(address, rank_base, expert, mask);
                    asc_gather(value, ranking, address, valid);
                    vector_bool selected;
                    asc_ne(selected, expert, best, valid);
                    asc_select(value, value, negative_inf, selected);
                    asc_gt(better, value, next_max, mask);
                    asc_select(next_max, value, next_max, better);
                    asc_select(next_idx, expert, next_idx, better);
                    asc_add_scalar(expert, expert, 2 * kWidth * kWidth, mask);
                }
                // Mask the winner in registers while its UB update overlaps the refresh.
                asc_eq_scalar(slot, columns, 0u, mask);
                asc_add(address, rank_base, best, mask);
                asc_scatter(ranking, negative_inf, address, slot);
                asc_mem_bar(VST_VLD);
                vector_float candidate_value = next_max;
                row_max<kWidth>(next_max, lanes);
                asc_eq(equal, candidate_value, next_max, mask);
                asc_select(next_idx, next_idx, invalid, equal);
                row_min<kWidth>(next_idx, lanes);
                asc_eq(winner, columns, winner_column, mask);
                vector_bool first_half, second_half;
                asc_lt_scalar(first_half, winner_offset, kWidth, winner);
                asc_ge_scalar(second_half, winner_offset, kWidth, winner);
                asc_select(maxima0, next_max, maxima0, first_half);
                asc_select(candidates0, next_idx, candidates0, first_half);
                asc_select(maxima1, next_max, maxima1, second_half);
                asc_select(candidates1, next_idx, candidates1, second_half);
            }
        }
    }
    vector_bool top_mask;
    asc_lt_scalar(top_mask, columns, kNumTopk, mask);
    asc_eq_scalar(first, columns, 0u, mask);
    vector_uint32_t address, missing = zeros;
    vector_float weights, weights_hi, denominator, zero;
    asc_duplicate_scalar(zero, 0.0f);
    asc_add(address, score_base, result, mask);
    asc_gather(weights, scores, address, top_mask);
    asc_select(weights, weights, zero, top_mask);
    weights_hi = zero;
    if constexpr (kNumTopk > kWidth) {
        asc_lt_scalar(top_mask, columns, kNumTopk - kWidth, mask);
        asc_add(address, score_base, result_hi, mask);
        asc_gather(weights_hi, scores, address, top_mask);
        asc_select(weights_hi, weights_hi, zero, top_mask);
    }
    asc_add(denominator, weights, weights_hi, mask);
    row_sum<kWidth>(denominator, lanes);
    asc_add_scalar(denominator, denominator, 1e-20f, mask);
    #pragma clang loop unroll(disable)
    for (uint32_t offset = 0; offset < kNumPhysical; offset += kWidth) {
        const auto logical = offset == 0 ? result : result_hi;
        const auto selected_weights = offset == 0 ? weights : weights_hi;
        store_gate_part<SHAPE_N, kNumTopk, kNumShared, kNumCached, kHasPhysicalMap, kHasUnmappedOutput>(
            output, map, counts, args, logical, selected_weights, denominator, columns, rows,
            errors, missing, token_idx, mask, offset);
    }
    row_sum<kWidth>(errors, lanes);
    row_max<kWidth>(missing, lanes);
    asc_mul_scalar(errors, errors, 2u, mask);
    asc_add(errors, errors, missing, mask);
    asc_scatter(status, errors, rows, first);
}

__simd_vf__ inline void vf_check_status(__ubuf__ uint32_t* status, uint32_t num_rows) {
    asc_mem_bar(VST_VLD);
    vector_uint32_t value;
    asc_loadalign(value, status);
    asc_reduce_max(value, value, asc_update_mask_b32(num_rows));
    asc_storealign_1st(status, value);
}

template <uint32_t SHAPE_N, uint32_t kNumTopk, uint32_t kNumSharedExperts, uint32_t kNumCachedDuplicates,
          bool kHasBias, bool kHasImageBias, bool kHasMask, bool kHasFixedRouting,
          bool kHasPhysicalMap, bool kHasUnmappedOutput, bool kHasRandomRouting,
          uint32_t kNumRows, uint32_t kNumKSplits>
__simd_vf__ inline void vf_route_packed(uint32_t token_idx, uint32_t num_rows, uint64_t active_rows) {
    constexpr layout::UBLayout ub_layout{
        SHAPE_N, kNumSharedExperts, kNumRows, kNumKSplits, kNumCachedDuplicates,
        kHasImageBias, kHasMask, kHasFixedRouting,
        kHasPhysicalMap, kHasUnmappedOutput, kHasRandomRouting, asc_get_vf_len(), kNumTopk};
    constexpr uint32_t kControlWidth = kNumTopk + kNumSharedExperts <= 8 and ub_layout.num_packed_rows == 8 ? 8 : 16;
    constexpr uint32_t kNumControlRows = 64 / kControlWidth;
    const auto rows = reinterpret_cast<__ubuf__ uint32_t*>(ub_layout.row_list);
    const auto num_routed_rows = rows + 5 * aligned(kNumRows, 8u);
    const auto ub_selected = reinterpret_cast<__ubuf__ int32_t*>(ub_layout.selected);
    const auto ub_status = reinterpret_cast<__ubuf__ uint32_t*>(ub_layout.status);
    const auto ub_counts = reinterpret_cast<__ubuf__ int32_t*>(ub_layout.counts);
    const auto ub_ranking = reinterpret_cast<__ubuf__ float*>(ub_layout.ranking);
    const auto ub_map = reinterpret_cast<__ubuf__ int32_t*>(ub_layout.map);
    const auto ub_args = reinterpret_cast<__ubuf__ layout::GateArgs*>(ub_layout.args);
    const auto ub_bias = reinterpret_cast<__ubuf__ float*>(ub_layout.bias);
    score<SHAPE_N, ub_layout.stride, kNumKSplits, (kNumRows >= 4 and kNumTopk <= 10) ? 2 : 1, true>(
        reinterpret_cast<__ubuf__ float*>(0), num_rows, ub_status, active_rows);
    asc_mem_bar(VST_VLD);
    #pragma clang loop unroll(full)
    for (uint32_t kind = 0; kind < 5; ++ kind) {
        if ((kind == 1 and not kHasImageBias) or
            (kind == 2 and not kHasFixedRouting) or
            (kind == 3 and not kHasRandomRouting) or
            (kind == 4 and not kHasMask)) continue;
        const uint32_t num_group_rows = kind < 2 ? ub_layout.num_packed_rows : kNumControlRows;
        for (uint32_t row_idx = 0; row_idx < num_routed_rows[kind]; row_idx += num_group_rows) {
            const auto row_list = rows + kind * aligned(kNumRows, 8u) + row_idx;
            const uint32_t num_grouped_rows = min(num_group_rows, num_routed_rows[kind] - row_idx);
            if (kind == 4) {
                masked_packed<kNumTopk, kNumSharedExperts, kHasUnmappedOutput, kControlWidth>(
                    ub_selected, ub_status, row_list, num_grouped_rows);
            } else if (kind == 3) {
                random_packed<SHAPE_N, kNumTopk, kNumSharedExperts, kHasPhysicalMap, kHasUnmappedOutput, kControlWidth>(
                    ub_selected, reinterpret_cast<__ubuf__ uint32_t*>(ub_counts),
                    reinterpret_cast<__ubuf__ uint32_t*>(ub_layout.random), ub_status,
                    num_grouped_rows, row_list);
            } else if (kind == 2) {
                gate_packed<SHAPE_N, kNumTopk, kNumSharedExperts, ub_layout.stride,
                            kNumCachedDuplicates, kHasPhysicalMap, kHasUnmappedOutput, kControlWidth, true, true>(
                    reinterpret_cast<__ubuf__ float*>(0), nullptr, false, ub_selected, ub_ranking, ub_map,
                    reinterpret_cast<__ubuf__ uint32_t*>(ub_counts), ub_args, ub_status, token_idx,
                    num_grouped_rows, row_list, reinterpret_cast<__ubuf__ int32_t*>(ub_layout.fixed));
            } else {
                const auto bias = kind == 1 ? ub_bias + ub_layout.stride : ub_bias;
                gate_packed<SHAPE_N, kNumTopk, kNumSharedExperts, ub_layout.stride,
                            kNumCachedDuplicates, kHasPhysicalMap, kHasUnmappedOutput, ub_layout.packed_width, true>(
                    reinterpret_cast<__ubuf__ float*>(0), bias, kHasBias or kind == 1, ub_selected, ub_ranking, ub_map,
                    reinterpret_cast<__ubuf__ uint32_t*>(ub_counts), ub_args, ub_status, token_idx,
                    num_grouped_rows, row_list);
            }
        }
    }
}

template <bool kHasImageBias, bool kHasMask, bool kHasFixedRouting, bool kHasRandomRouting,
          uint32_t kNumRows, uint64_t kFlags, uint64_t kRowList>
__simd_vf__ inline void vf_classify(uint32_t num_rows) {
    const auto rows = reinterpret_cast<__ubuf__ uint32_t*>(kRowList);
    const auto num_routed_rows = rows + 5 * aligned(kNumRows, 8u);
    const auto flags = reinterpret_cast<__ubuf__ uint8_t*>(kFlags);
    uint32_t count = num_rows;
    const auto mask = asc_update_mask_b32(count);
    vector_uint32_t kinds, lanes, zeros, ones;
    asc_duplicate_scalar(zeros, 0u);
    asc_duplicate_scalar(ones, 1u);
    asc_arange(reinterpret_cast<vector_int32_t&>(lanes), 0);
    kinds = zeros;
    // Priority: masked, random, fixed, image bias, ordinary.
    #pragma clang loop unroll(full)
    for (uint32_t kind = 1; kind < 5; ++ kind) {
        constexpr bool kFeatureEnabled[]{false, kHasImageBias, kHasFixedRouting, kHasRandomRouting, kHasMask};
        if (kFeatureEnabled[kind]) {
            vector_uint8_t packed;
            vector_uint32_t flag, value;
            vector_bool selected;
            asc_loadalign_unpack4(packed, flags + (4 - kind) * aligned(kNumRows, 32u));
            asc_uint82uint32(flag, packed, mask,
                             std::integral_constant<asc_position_quarter_mode,
                                                    asc_position_quarter_mode::DISPERSE_FIRST_QUARTER>{});
            if (kind == 4) asc_eq_scalar(selected, flag, 0u, mask);
            else asc_ne_scalar(selected, flag, 0u, mask);
            asc_duplicate_scalar(value, kind);
            asc_select(kinds, value, kinds, selected);
        }
    }
    vector_uint32_t shifts, bits, low_bits, high_bits;
    vector_bool active, low, high;
    asc_duplicate_scalar(shifts, 31u);
    asc_and(shifts, lanes, shifts, mask);
    asc_shiftleft(bits, ones, reinterpret_cast<vector_int32_t&>(shifts), mask);
    asc_lt_scalar(active, kinds, 3u, mask);
    asc_lt_scalar(low, lanes, 32u, active);
    asc_ge_scalar(high, lanes, 32u, active);
    asc_select(low_bits, bits, zeros, low);
    asc_select(high_bits, bits, zeros, high);
    asc_reduce_sum(low_bits, low_bits, mask);
    asc_reduce_sum(high_bits, high_bits, mask);
    asc_storealign_1st(num_routed_rows + 5, low_bits);
    asc_storealign_1st(num_routed_rows + 6, high_bits);
    #pragma clang loop unroll(full)
    for (uint32_t kind = 0; kind < 5; ++ kind) {
        vector_bool selected;
        vector_uint32_t indices, count;
        asc_eq_scalar(selected, kinds, kind, mask);
        asc_squeeze(indices, lanes, selected);
        asc_storealign(rows + kind * aligned(kNumRows, 8u), indices, mask);
        asc_select(count, ones, zeros, selected);
        asc_reduce_sum(count, count, mask);
        asc_storealign_1st(num_routed_rows + kind, count);
    }
}

template <uint32_t SHAPE_N, uint32_t kStride, uint32_t kNumKSplits, uint32_t kUnroll>
__simd_vf__ inline void vf_score(__ubuf__ float* values, uint32_t num_rows, __ubuf__ uint32_t* status) {
    score<SHAPE_N, kStride, kNumKSplits, kUnroll>(values, num_rows, status);
}

template <uint32_t SHAPE_N, uint32_t kNumTopk, uint32_t kNumShared, uint32_t kStride, uint32_t kNumCached,
          bool kHasPhysicalMap, bool kHasUnmappedOutput, uint32_t kWidth>
__simd_vf__ inline void vf_gate_packed(
    __ubuf__ float* scores, __ubuf__ float* bias, bool has_bias, __ubuf__ int32_t* output,
    __ubuf__ float* ranking, __ubuf__ int32_t* map, __ubuf__ uint32_t* counts,
    __ubuf__ layout::GateArgs* args, __ubuf__ uint32_t* status, uint32_t token_idx, uint32_t num_rows) {
    gate_packed<SHAPE_N, kNumTopk, kNumShared, kStride, kNumCached, kHasPhysicalMap, kHasUnmappedOutput, kWidth>(
        scores, bias, has_bias, output, ranking, map, counts, args, status, token_idx, num_rows, nullptr);
}

} // namespace deep_gemm::mega_gate
