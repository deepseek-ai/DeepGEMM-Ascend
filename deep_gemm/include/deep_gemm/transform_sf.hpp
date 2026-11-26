#pragma once

#include <cstdint>
#include <type_traits>

#include <deep_gemm/ascend.hpp>
#include <c_api/asc_simd.h>
#include <utils/debug/asc_assert.h>  // ascendc_assert: exists in CANN 9.2 but is not pulled in via kernel_operator.h

using namespace __cce_simd;

namespace deep_gemm {

template <uint32_t kValidationElems>
__aicore__ inline void check_validation(__ubuf__ uint32_t* validation) {
    uint32_t combined = 0;
    #pragma unroll
    for (uint32_t i = 0; i < kValidationElems; i += 1)
        combined |= validation[i];
    ascendc_assert((combined & 0x80000000u) == 0 and (combined & 0x007fffffu) == 0,
                   "Float32 SF must be non-negative and can't have non-zero mantissa, got %x", combined);
}

template <uint32_t kSrcRows, uint32_t kSrcCols, uint32_t kSrcStride, uint32_t kDstStride>
__simd_callee__ inline void transpose_16rows_simd_impl(
    __ubuf__ uint16_t* src,
    __ubuf__ uint16_t* dst
) {
    static_assert(kSrcRows % 16 == 0 and kDstStride % 16 == 0, "kSrcRows and kDstStride must be multiples of 16");
    static_assert(kSrcCols == 128 or kSrcStride == kSrcCols, "kSrcStride must equal kSrcCols when kSrcCols != 128");
    static_assert(kSrcCols == 32 or kSrcCols == 64 or kSrcCols == 128,
                  "kSrcCols must be 32, 64, or 128 when transpose");

    constexpr uint32_t VL = 256 / sizeof(uint16_t);
    constexpr uint32_t kNumVectors = 16 * kSrcCols / VL;
    constexpr uint32_t kRowsPerVector = VL / kSrcCols;
    constexpr uint32_t load_offset = kRowsPerVector * kSrcStride;
    constexpr uint32_t store_offset = 8 * kDstStride;

    vector_uint16_t sf[kNumVectors];
    vector_bool mask_all_b16 = asc_create_mask_b16(PAT_ALL);
    #pragma unroll
    for (uint32_t i = 0; i < kNumVectors; i += 1)
        asc_loadalign_postupdate(sf[i], src, load_offset);

    #pragma unroll
    for (uint32_t offset = kNumVectors / 2; offset > 0; offset >>= 1) {
        #pragma unroll
        for (uint32_t sf_idx = 0; sf_idx < kNumVectors; sf_idx += offset * 2) {
            #pragma unroll
            for (uint32_t i = 0; i < offset; i += 1)
                asc_intlv(sf[sf_idx + i], sf[sf_idx + offset + i], sf[sf_idx + i], sf[sf_idx + offset + i]);
        }
    }
    if constexpr (kSrcCols == 64) {
        asc_intlv(sf[0], sf[4], sf[0], sf[4]);
        asc_intlv(sf[1], sf[5], sf[1], sf[5]);
        asc_intlv(sf[2], sf[6], sf[2], sf[6]);
        asc_intlv(sf[3], sf[7], sf[3], sf[7]);
    } else if constexpr (kSrcCols == 32) {
        asc_intlv(sf[0], sf[2], sf[0], sf[2]);
        asc_intlv(sf[1], sf[3], sf[1], sf[3]);
        asc_intlv(sf[0], sf[1], sf[0], sf[1]);
        asc_intlv(sf[2], sf[3], sf[2], sf[3]);
    }

    #pragma unroll
    for (uint32_t i = 0; i < kNumVectors; i += 1) {
        const uint32_t sf_idx = kSrcCols == 64 ? (i >> 1) + (i & 1) * 4 : i;
        asc_storealign(dst + i * store_offset, sf[sf_idx], kDstStride / 16, 0, mask_all_b16);
    }
}

template <uint32_t kSrcRows, uint32_t kSrcCols, uint32_t kSrcStride,
          uint32_t kDstStride>
__simd_vf__ inline void transpose_simd(
    __ubuf__ uint16_t* src,
    __ubuf__ uint16_t* dst,
    uint32_t num_rows
) {
    vector_bool mask_all_b16 = asc_create_mask_b16(PAT_ALL);
    #pragma unroll 1
    for (uint32_t row = 0; row < num_rows; row += 16) {
        auto* src_row = src + row * kSrcStride;
        auto* dst_row = dst + row;
        transpose_16rows_simd_impl<kSrcRows, kSrcCols, kSrcStride, kDstStride>(src_row, dst_row);
    }
}

template <uint32_t kGranMN>
__simd_callee__ inline void repeat_to_256_simd_impl(
    __ubuf__ uint16_t* src,
    __ubuf__ uint16_t* dst
) {
    static_assert(kGranMN >= 2 and kGranMN <= 128 and (kGranMN & (kGranMN - 1)) == 0,
                  "kGranMN must be a power of two between 2 and 128");

    constexpr int32_t VL = 256 / sizeof(uint16_t);
    constexpr int32_t load_offset = VL;
    constexpr int32_t store_offset = VL;
    constexpr int32_t load_e2b_offset = 8;  // 8 datablock per vector

    vector_bool mask_all_b16 = asc_create_mask_b16(PAT_ALL);
    vector_uint16_t sf0, sf1;
    if constexpr (kGranMN == 16) {
        asc_loadalign_brc_elem2datablock_postupdate(sf0, src, load_e2b_offset);
        asc_loadalign_brc_elem2datablock_postupdate(sf1, src, load_e2b_offset);
        asc_storealign_postupdate(dst, sf0, store_offset, mask_all_b16);
        asc_storealign_postupdate(dst, sf1, store_offset, mask_all_b16);
    } else {
        constexpr uint32_t repeat = kGranMN > 16 ? 16 : 1;
        if constexpr (kGranMN < 16)
            asc_loadalign_postupdate(sf0, src, load_offset);
        else
            asc_loadalign_brc_elem2datablock_postupdate(sf0, src, load_e2b_offset);
        #pragma unroll
        for (uint32_t r = repeat; r < kGranMN; r <<= 1)
            asc_intlv(sf0, sf1, sf0, sf0);
        asc_storealign_postupdate(dst, sf0, store_offset, mask_all_b16);
        asc_storealign_postupdate(dst, sf1, store_offset, mask_all_b16);
    }
}

template <uint32_t kDstCols, uint32_t kGranMN>
__simd_callee__ inline void repeat_to_1024x_simd_impl(
    __ubuf__ uint16_t* src,
    __ubuf__ uint16_t* dst
) {
    static_assert(kGranMN >= 2 and kGranMN <= 128 and (kGranMN & (kGranMN - 1)) == 0,
                  "kGranMN must be a power of two between 2 and 128");

    static_assert(kDstCols % 1024 == 0, "large repeat destination columns must be 1024-aligned");

    constexpr uint32_t num_dst_elems = kDstCols;
    constexpr uint32_t VL = 256 / sizeof(uint16_t);
    constexpr int32_t load_offset = kGranMN < 16 ? VL : 8;
    constexpr int32_t base_repeat = kGranMN < 16 ? 1 : 16;
    constexpr int32_t num_repeats = kGranMN / base_repeat;
    constexpr int32_t store_offset = VL;
    constexpr int32_t num_groups = (num_dst_elems / kGranMN) / load_offset;
    src = src + (num_groups - 1) * load_offset;
    dst = dst + num_dst_elems - store_offset;

    vector_bool mask_all_b16 = asc_create_mask_b16(PAT_ALL);
    vector_uint16_t sf0, sf1, sf2, sf3, sf4, sf5, sf6, sf7;
    #pragma unroll
    for (int32_t group_idx = num_groups - 1; group_idx >= 0; --group_idx) {
        if constexpr (kGranMN < 16)
            asc_loadalign_postupdate(sf0, src, -load_offset);
        else
            asc_loadalign_brc_elem2datablock_postupdate(sf0, src, -load_offset);

        if constexpr (num_repeats >= 2)
            asc_intlv(sf0, sf1, sf0, sf0);
        if constexpr (num_repeats >= 4) {
            asc_intlv(sf2, sf3, sf1, sf1);
            asc_intlv(sf0, sf1, sf0, sf0);
        }
        if constexpr (num_repeats >= 8) {
            asc_intlv(sf4, sf5, sf2, sf2);
            asc_intlv(sf6, sf7, sf3, sf3);
            asc_intlv(sf2, sf3, sf1, sf1);
            asc_intlv(sf0, sf1, sf0, sf0);
        }

        if constexpr (num_repeats >= 8) {
            asc_storealign_postupdate(dst, sf7, -store_offset, mask_all_b16);
            asc_storealign_postupdate(dst, sf6, -store_offset, mask_all_b16);
            asc_storealign_postupdate(dst, sf5, -store_offset, mask_all_b16);
            asc_storealign_postupdate(dst, sf4, -store_offset, mask_all_b16);
        }
        if constexpr (num_repeats >= 4) {
            asc_storealign_postupdate(dst, sf3, -store_offset, mask_all_b16);
            asc_storealign_postupdate(dst, sf2, -store_offset, mask_all_b16);
        }
        if constexpr (num_repeats >= 2)
            asc_storealign_postupdate(dst, sf1, -store_offset, mask_all_b16);
        asc_storealign_postupdate(dst, sf0, -store_offset, mask_all_b16);
    }
}

template <uint32_t kSrcCols, uint32_t kDstCols,
          uint32_t kSrcStride, uint32_t kDstStride,
          uint32_t kGranMN>
__simd_vf__ inline void repeat_simd(
    __ubuf__ uint16_t* src,
    __ubuf__ uint16_t* dst,
    int32_t num_rows
) {
    static_assert(kDstCols == kSrcCols * kGranMN, "repeat destination columns must equal source columns times gran_mn");
    static_assert(kDstCols == 256 or kDstCols % 1024 == 0, "repeat destination columns must be 256 or 1024 aligned");

    #pragma unroll 1
    for (int32_t row = num_rows - 1; row >= 0; --row) {
        auto* src_row = src + row * kSrcStride;
        auto* dst_row = dst + row * kDstStride;
        if constexpr (kDstCols == 256)
            repeat_to_256_simd_impl<kGranMN>(src_row, dst_row);
        else
            repeat_to_1024x_simd_impl<kDstCols, kGranMN>(src_row, dst_row);
    }
}

__simd_callee__ inline void merge_validation_simd_impl(
    __ubuf__ uint32_t* validation,
    vector_uint32_t& block_val0,
    vector_uint32_t& block_val1
) {
    vector_uint32_t all_validation;
    vector_bool mask_all_b32 = asc_create_mask_b32(PAT_ALL);
    asc_loadalign(all_validation, validation);
    asc_or(all_validation, all_validation, block_val0, mask_all_b32);
    asc_or(all_validation, all_validation, block_val1, mask_all_b32);
    asc_storealign(validation, all_validation, mask_all_b32);
}

template <bool kHasGroupedFlag, uint32_t kSrcCols, uint32_t kSrcStride, uint32_t kDstStride>
__simd_callee__ inline void pack_validate_k_major_2row_simd_impl(
    __ubuf__ uint32_t*& src,
    __ubuf__ uint32_t*& dst,  // pack_quarter requires u32
    __ubuf__ uint32_t* grouped_flag,
    vector_uint32_t& block_val0,
    vector_uint32_t& block_val1
) {
    constexpr uint32_t VL = 256 / sizeof(uint32_t);
    constexpr int32_t load_offset = VL;
    constexpr int32_t store_offset = VL / 4;  // 64 * ue8m0x2 in unit of u32

    vector_bool mask_all_b32 = asc_create_mask_b32(PAT_ALL);
    #pragma unroll
    for (uint32_t col = 0; col < kSrcCols; col += load_offset) {
        vector_uint32_t sf0, sf1, grouped_val0, grouped_val1;
        asc_loadalign_postupdate(sf0, src, load_offset);
        asc_loadalign_postupdate(sf1, src, load_offset);
        if constexpr (kHasGroupedFlag) {
            auto* grouped_flag_row = grouped_flag;
            asc_loadalign_postupdate(grouped_val0, grouped_flag_row, load_offset);
            asc_loadalign_postupdate(grouped_val1, grouped_flag_row, load_offset);
            asc_and(sf0, sf0, grouped_val0, mask_all_b32);
            asc_and(sf1, sf1, grouped_val1, mask_all_b32);
        }
        asc_or(block_val0, block_val0, sf0, mask_all_b32);
        asc_or(block_val1, block_val1, sf1, mask_all_b32);
        asc_shiftright_scalar(sf0, sf0, 23, mask_all_b32);
        asc_shiftright_scalar(sf1, sf1, 23, mask_all_b32);
        asc_storealign_pack_quarter_postupdate(dst, sf0, store_offset, mask_all_b32);
        asc_storealign_pack_quarter_postupdate(dst, sf1, store_offset, mask_all_b32);
    }
}

template <bool kHasGroupedFlag, uint32_t kSrcStride>
__simd_callee__ inline void pack_validate_mn_major_simd_impl(
    __ubuf__ uint32_t* src0,
    __ubuf__ uint32_t* dst,
    __ubuf__ uint32_t* grouped_flag,
    uint32_t num_cols,
    vector_uint32_t& block_val0,
    vector_uint32_t& block_val1
) {
    constexpr uint32_t VL = 256 / sizeof(uint32_t);
    constexpr int32_t load_offset = VL;
    constexpr int32_t store_offset = VL / 2;  // 64 * ue8m0x2 in unit of u16
    vector_bool mask_all_b32 = asc_create_mask_b32(PAT_ALL);
    auto* src1 = src0 + kSrcStride;
    #pragma unroll
    for (uint32_t col = 0; col < num_cols; col += load_offset) {
        vector_uint32_t sf0, sf1, sf_packed, grouped_val;
        asc_loadalign_postupdate(sf0, src0, load_offset);
        asc_loadalign_postupdate(sf1, src1, load_offset);
        if constexpr (kHasGroupedFlag) {
            asc_loadalign_postupdate(grouped_val, grouped_flag, load_offset);
            asc_and(sf0, sf0, grouped_val, mask_all_b32);
            asc_and(sf1, sf1, grouped_val, mask_all_b32);
        }
        asc_or(block_val0, block_val0, sf0, mask_all_b32);
        asc_or(block_val1, block_val1, sf1, mask_all_b32);
        asc_shiftright_scalar(sf0, sf0, 23, mask_all_b32);
        asc_shiftright_scalar(sf1, sf1, 15, mask_all_b32);
        asc_or(sf_packed, sf0, sf1, mask_all_b32);
        asc_storealign_pack_postupdate(dst, sf_packed, store_offset, mask_all_b32);
    }
}

template <uint32_t kBytes>
__simd_callee__ inline void fill_bytes_simd_impl(
    __ubuf__ uint8_t* src,
    uint8_t value
) {
    static_assert(kBytes % 256 == 0, "kBytes must be a multiple of 256");

    vector_bool mask_all_b8 = asc_create_mask_b8(PAT_ALL);
    vector_uint8_t value_vec;
    asc_duplicate_scalar(value_vec, value, mask_all_b8);
    constexpr uint32_t store_offset = 256;
    for (uint32_t offset = 0; offset < kBytes; offset += store_offset) {
        asc_storealign_postupdate(src, value_vec, store_offset, mask_all_b8);
    }
}

template <Major kMajor, bool kHasGroupedFlag,
          uint32_t kSrcCols, uint32_t kSrcStride,
          uint32_t kDstCols, uint32_t kDstStride>
__simd_vf__ inline void pack_validate_simd(
    __ubuf__ uint32_t* src,
    __ubuf__ uint16_t* dst,
    __ubuf__ uint32_t* validation,
    __ubuf__ uint32_t* grouped_flag,
    uint32_t num_rows,
    uint32_t num_cols
) {
    static_assert(kSrcStride % 64 == 0, "kSrcStride must be a multiple of 64 for mn-major packing");
    static_assert(kDstStride % 16 == 0, "kDstStride must be a multiple of 16 for mn-major packing");

    vector_bool mask_all_b32 = asc_create_mask_b32(PAT_ALL);
    vector_uint32_t block_acc0, block_acc1;
    asc_duplicate_scalar(block_acc0, 0, mask_all_b32);
    asc_duplicate_scalar(block_acc1, 0, mask_all_b32);

    if (num_rows % 2 != 0) {
        constexpr uint32_t kBytesPerRow = kSrcStride * sizeof(uint32_t);
        fill_bytes_simd_impl<kBytesPerRow>(reinterpret_cast<__ubuf__ uint8_t*>(src + num_rows * kSrcStride), 0u);
        asc_mem_bar(VST_VLD);
    }

    if constexpr (kMajor == Major::K) {
        auto* src_row = src;
        auto* dst_row = reinterpret_cast<__ubuf__ uint32_t*>(dst);
        constexpr uint32_t kDstStride_u32 = kDstStride * sizeof(uint16_t) / sizeof(uint32_t);
        #pragma unroll 1
        for (uint32_t row = 0; row < num_rows; row += 2) {
            pack_validate_k_major_2row_simd_impl<kHasGroupedFlag, kSrcCols, kSrcStride, kDstStride_u32>(src_row,
                                                                                                        dst_row,
                                                                                                        grouped_flag,
                                                                                                        block_acc0,
                                                                                                        block_acc1);
        }
    } else {
        #pragma unroll 1
        for (uint32_t row = 0; row < num_rows; row += 2) {
            auto* src_row = src + row * kSrcStride;
            auto* dst_row = reinterpret_cast<__ubuf__ uint32_t*>(dst + row / 2 * kDstStride);
            pack_validate_mn_major_simd_impl<kHasGroupedFlag, kSrcStride>(src_row, dst_row, grouped_flag, num_cols,
                                                                          block_acc0, block_acc1);
        }
    }

    merge_validation_simd_impl(validation, block_acc0, block_acc1);
    asc_mem_bar(VST_VLD);
}

template <bool kFloat, uint32_t kStride>
__simd_vf__ inline void clear_group_tail_simd(
    __ubuf__ uint8_t* src,
    uint32_t row_idx,
    uint32_t num_rows
) {
    constexpr uint32_t kBytesPerRow = kStride * (kFloat ? 4 : 2);
    uint32_t row_end = row_idx + num_rows;
    #pragma unroll
    for (uint32_t row = row_idx; row < row_end; row += 1) {
        auto* src_row = src + row * kBytesPerRow;
        fill_bytes_simd_impl<kBytesPerRow>(src_row, 0u);
    }
    asc_mem_bar(VST_VLD);
}

template <uint32_t kBytes>
__simd_vf__ inline void clear_ub_simd(
    __ubuf__ uint8_t* src
) {
    fill_bytes_simd_impl<kBytes>(src, 0u);
    asc_mem_bar(VST_VLD);
}

template <bool kHasGroupedFlag, bool kFloat, uint32_t kSrcStride,
          uint32_t kGroupedFlagElems, uint32_t kGranGroup, uint32_t kGroupAlignment>
__aicore__ inline void prepare_grouped_input(
    __ubuf__ uint8_t* src,
    __gm__ int32_t* grouped_layout,
    __ubuf__ uint32_t* grouped_flag,
    __ubuf__ uint32_t* grouped_tail,
    uint32_t num_groups,
    uint32_t current_src_idx,
    uint32_t src_actual,
    uint32_t& last_src_idx,
    uint32_t& current_group_idx,
    uint32_t& num_group_tails
) {
    const uint32_t current_src_end = current_src_idx + src_actual;
    const auto apply_group_tails = [&]() __aicore__ {
        for (uint32_t i = 0; i < num_group_tails; i += 1) {
            const uint32_t clear_begin = max(current_src_idx, grouped_tail[i]);
            const uint32_t clear_end = min(current_src_end, aligned(grouped_tail[i], kGroupAlignment));
            if (clear_begin >= clear_end)
                continue;

            const uint32_t clear_offset = clear_begin - current_src_idx;
            const uint32_t clear_count = clear_end - clear_begin;
            if constexpr (kHasGroupedFlag) {
                for (uint32_t j = 0; j < clear_count; j += 1)
                    grouped_flag[clear_offset + j] = 0u;
            } else {
                clear_group_tail_simd<kFloat, kSrcStride>(src, clear_offset, clear_count);
            }
        }
    };

    if (last_src_idx != current_src_idx) {
        if constexpr (kHasGroupedFlag) {
            // TODO: multiple scalar buffers if scalar-bound
            asc_sync_notify(PIPE_V, PIPE_S, EVENT_ID0);
            asc_sync_wait(PIPE_V, PIPE_S, EVENT_ID0);
        }
        while (current_group_idx < num_groups and
               aligned(ceil_div(grouped_layout[current_group_idx], kGranGroup), kGroupAlignment) <= current_src_idx)
            current_group_idx += 1;
        if (num_group_tails > 0 and
            aligned(grouped_tail[num_group_tails - 1], kGroupAlignment) > current_src_idx) {
            grouped_tail[0] = grouped_tail[num_group_tails - 1];
            num_group_tails = 1;
        } else {
            num_group_tails = 0;
        }

        while (current_group_idx < num_groups) {
            const uint32_t group_tail = ceil_div(grouped_layout[current_group_idx], kGranGroup);
            if (group_tail >= current_src_end)
                break;
            current_group_idx += 1;
            if (group_tail < aligned(group_tail, kGroupAlignment))
                grouped_tail[num_group_tails++] = group_tail;
        }

        if constexpr (kHasGroupedFlag) {
            for (uint32_t i = 0; i < kGroupedFlagElems; i += 1)
                grouped_flag[i] = i < src_actual ? 0xffffffffu : 0u;
            apply_group_tails();
            asc_sync_notify(PIPE_S, PIPE_V, EVENT_ID0);
            asc_sync_wait(PIPE_S, PIPE_V, EVENT_ID0);
        }
        last_src_idx = current_src_idx;
    }

    if constexpr (not kHasGroupedFlag)
        apply_group_tails();
}

template <bool kFloat, Major kMajor,
          uint32_t kSrcBlockM, uint32_t kSrcBlockK,
          GemmType kGemmType = GemmType::Normal,
          uint32_t kNumCores = 64,
          uint32_t kAlignment = 256,
          uint32_t kGranMN = 1,
          asc_load_l2_cache_mode kL2CtrlLoad = asc_load_l2_cache_mode::NOTALLOC_KEEP,
          asc_store_l2_cache_mode kL2CtrlStore = asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM>
__global__ __vector__ void transform_sf(
    gm_ptr<std::conditional_t<kFloat, uint32_t, uint16_t>, kMajor> src_gm,
    __gm__ uint16_t* dst_gm,
    __gm__ int32_t* grouped_layout,
    uint32_t src_shape_m,
    uint32_t src_shape_k,
    uint32_t dst_shape_m,
    uint32_t dst_shape_k,
    uint32_t num_groups,
    uint32_t num_batches
) {
    asc_init();

    static_assert(kGemmType != GemmType::MGroupedContiguousWithPsumLayout or kAlignment % kGranMN == 0,
                  "M-grouped transform requires Alignment % GranMN == 0");

    using InputT = std::conditional_t<kFloat, uint32_t, uint16_t>;
    asc_set_copy_pad_val(static_cast<InputT>(0));

    constexpr uint32_t kNumElemsPerPackedSF = kFloat ? 2 : 1;
    constexpr uint32_t kNumElemsPerVector = 256 / sizeof(InputT);

    constexpr uint32_t kDstBlockM = kSrcBlockM * kGranMN;
    constexpr uint32_t kDstBlockK = kSrcBlockK / kNumElemsPerPackedSF;

    constexpr uint32_t kSrcUBRows = kMajor == Major::K ? (kSrcBlockM < 16 ? 16 : kSrcBlockM) : kSrcBlockK;
    constexpr uint32_t kSrcUBCols = kMajor == Major::K ? kSrcBlockK : (kSrcBlockM < kNumElemsPerVector ? kNumElemsPerVector : kSrcBlockM);
    constexpr uint32_t kPackedRows = kMajor == Major::K ? kSrcUBRows : kSrcUBRows / kNumElemsPerPackedSF;
    constexpr uint32_t kPackedCols = kMajor == Major::K ? kSrcUBCols / kNumElemsPerPackedSF : kSrcUBCols;
    constexpr uint32_t kDstUBRows = kDstBlockK;
    constexpr uint32_t kDstUBCols = kDstBlockM;

    constexpr uint32_t kSrcUBStride = kSrcUBCols;
    constexpr uint32_t kPackedStride = kPackedCols;
    constexpr uint32_t kDstUBStride = kDstUBCols + (kMajor == Major::K ? 16 : 0); // pad to avoid bank conflict when transpose

    constexpr bool is_m_grouped = kGemmType == GemmType::MGroupedContiguousWithPsumLayout;
    constexpr bool is_k_grouped = kGemmType == GemmType::KGroupedContiguousWithPsumLayout;
    constexpr bool is_grouped = is_m_grouped or is_k_grouped;
    constexpr uint32_t kGranGroup = is_m_grouped ? kGranMN : (kFloat ? 32 : MX_SF_DIVISOR);
    static_assert(not is_grouped or kAlignment % kGranGroup == 0,
                  "group alignment must be divisible by group granularity");

    // stages are derived from UB capacity
    constexpr bool kHasPackedBuffer = kFloat and kMajor == Major::K;
    constexpr bool kHasGroupedFlag = (is_m_grouped and kMajor == Major::MN) or (is_k_grouped and kMajor == Major::K);
    constexpr uint32_t kGroupedFlagAlign = 32 / sizeof(uint32_t);
    constexpr uint32_t kGroupedFlagElems = not is_grouped ? 0 : kHasGroupedFlag ? kSrcUBCols : aligned(is_m_grouped ? kSrcBlockM : kSrcBlockK, kGroupedFlagAlign);
    constexpr uint32_t kGroupedFlagBytes = kGroupedFlagElems * sizeof(uint32_t);

    constexpr uint32_t kValidationElems = kFloat ? 64 : 0;
    constexpr uint32_t kReservedUBBytes = 8 * 1024;  // predicate-register spill area
    constexpr uint32_t kUsableUBBytes = UBSizeBytes - kReservedUBBytes;
    constexpr uint32_t kValidUBBytes = kUsableUBBytes - kValidationElems * sizeof(uint32_t) - 2 * kGroupedFlagBytes;
    constexpr uint32_t kBytesPerStage = kSrcUBRows * kSrcUBStride * sizeof(InputT) + kDstUBRows * kDstUBStride * sizeof(uint16_t) + (kHasPackedBuffer ? kPackedRows * kPackedStride * sizeof(uint16_t) : 0);
    constexpr uint32_t kNumStages = kValidUBBytes / kBytesPerStage > 8 ? 8 : kValidUBBytes / kBytesPerStage;
    static_assert(kNumStages >= 2, "UB too small for a 2-stage pipeline");
    static_assert(kNumStages <= 8, "NumStages exceeds event capacity");

    // kPackedRows/Cols == kSrcUBRows/Cols when dont need pack buffer
    static_assert(kFloat or (kPackedRows == kSrcUBRows and kPackedCols == kSrcUBCols),
                  "kPackedRows/Cols must match kSrcUBRows/Cols when short input");
    constexpr ub_ptr<InputT> src_ub(kSrcUBRows, kSrcUBStride);
    constexpr ub_ptr<uint16_t> packed_ub(kPackedRows, kPackedStride, kHasPackedBuffer ? src_ub.offset(kNumStages) : src_ub.offset(0));
    constexpr ub_ptr<uint16_t> dst_ub(kDstUBRows, kDstUBStride, kHasPackedBuffer ? packed_ub.offset(kNumStages) : src_ub.offset(kNumStages));
    constexpr ub_ptr<uint32_t> validation_ub(1, kValidationElems, dst_ub.offset(kNumStages));
    constexpr ub_ptr<uint32_t> grouped_flag_ub(1, kGroupedFlagElems, validation_ub.offset(1));
    constexpr ub_ptr<uint32_t> grouped_tail_ub(1, kGroupedFlagElems, grouped_flag_ub.offset(1));
    static_assert(grouped_tail_ub.offset(1) <= kUsableUBBytes, "UB overflow");

    if constexpr (kFloat) {
        clear_ub_simd<UBSizeBytes>(0);
        asc_sync_pipe(PIPE_ALL);
    }

    const uint32_t num_m_blocks = ceil_div(src_shape_m, kSrcBlockM);
    const uint32_t num_k_blocks = ceil_div(src_shape_k, kSrcBlockK);
    const uint32_t num_batch_blocks = num_m_blocks * num_k_blocks;
    const uint32_t num_blocks = num_batch_blocks * num_batches;

    const uint32_t num_blocks_per_core = ceil_div(num_blocks, kNumCores);
    const uint32_t block_begin = block_idx * num_blocks_per_core;
    const uint32_t block_end = min(block_begin + num_blocks_per_core, num_blocks);

    uint32_t last_src_idx = UINT32_MAX;
    uint32_t current_group_idx = 0;
    uint32_t num_group_tails = 0;

    set_flags<PIPE_V, PIPE_MTE2, kNumStages>();
    set_flags<PIPE_MTE3, PIPE_V, kNumStages>();
    uint32_t stage = 0;
    for (uint32_t block_idx = block_begin; block_idx < block_end; block_idx += 1) {
        const uint32_t batch_idx = block_idx / num_batch_blocks;
        const uint32_t in_batch = block_idx % num_batch_blocks;

        const uint32_t m_block_idx = is_k_grouped ? in_batch % num_m_blocks : in_batch / num_k_blocks;
        const uint32_t k_block_idx = is_k_grouped ? in_batch / num_m_blocks : in_batch % num_k_blocks;

        const uint32_t src_m_idx = m_block_idx * kSrcBlockM;
        const uint32_t src_k_idx = k_block_idx * kSrcBlockK;
        const uint32_t dst_m_idx = src_m_idx * kGranMN;
        const uint32_t dst_k_idx = src_k_idx / kNumElemsPerPackedSF;

        uint32_t src_actual_m = min(kSrcBlockM, src_shape_m - src_m_idx);
        uint32_t src_actual_k = min(kSrcBlockK, src_shape_k - src_k_idx);
        uint32_t dst_actual_m = min(kDstBlockM, dst_shape_m - dst_m_idx);
        uint32_t dst_actual_k = min(kDstBlockK, dst_shape_k - dst_k_idx);

        const uint32_t num_src_rows = kMajor == Major::K ? src_actual_m : src_actual_k;
        const uint32_t num_src_cols = kMajor == Major::K ? src_actual_k : src_actual_m;
        // dst is mn major
        const uint32_t num_dst_rows = dst_actual_k;
        const uint32_t num_dst_cols = dst_actual_m;

        const auto src_ub_stage = src_ub[stage];
        const auto packed_ub_stage = packed_ub[stage];
        const auto dst_ub_stage = dst_ub[stage];
        auto* grouped_flag = grouped_flag_ub[0].ptr();
        auto* grouped_tail = grouped_tail_ub[0].ptr();

        const auto src_gm_ptr = src_gm.batch(batch_idx).index(src_m_idx, src_k_idx);

        asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(stage));

        if (src_actual_m > 0 and src_actual_k > 0)
            copy_gm_to_ub(src_ub_stage, src_gm_ptr, src_actual_m, src_actual_k, kL2CtrlLoad, true);

        asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(stage));
        asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(stage));
        asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(stage));

        if constexpr (is_grouped) {
            const uint32_t current_src_idx = is_m_grouped ? src_m_idx : src_k_idx;
            const uint32_t src_actual = is_m_grouped ? src_actual_m : src_actual_k;
            constexpr uint32_t kGroupAlignment = kAlignment / kGranGroup;
            prepare_grouped_input<kHasGroupedFlag, kFloat, kSrcUBStride, kGroupedFlagElems, kGranGroup, kGroupAlignment>(src_ub_stage.template ptr<uint8_t>(), grouped_layout, grouped_flag,
                                                                                                                         grouped_tail, num_groups, current_src_idx, src_actual, last_src_idx,
                                                                                                                         current_group_idx, num_group_tails);
        }

        if constexpr (kFloat and kMajor == Major::K) {
            pack_validate_simd<kMajor, kHasGroupedFlag, kSrcUBCols, kSrcUBStride, kPackedCols, kPackedStride>(src_ub_stage.ptr(), packed_ub_stage.ptr(), validation_ub.ptr(),
                                                                                                              grouped_flag, num_src_rows, num_src_cols);
            transpose_simd<kPackedRows, kPackedCols, kPackedStride, kDstUBStride>(packed_ub_stage.ptr(),
                                                                                  dst_ub_stage.ptr(), num_src_rows);
        } else if constexpr (kFloat and kMajor == Major::MN) {
            pack_validate_simd<kMajor, kHasGroupedFlag, kSrcUBCols, kSrcUBStride, kDstUBCols, kDstUBStride>(src_ub_stage.ptr(), dst_ub_stage.ptr(), validation_ub.ptr(), grouped_flag,
                                                                                                            num_src_rows, num_src_cols);
        } else if constexpr (not kFloat and kMajor == Major::K) {
            transpose_simd<kSrcUBRows, kSrcUBCols, kSrcUBStride, kDstUBStride>(src_ub_stage.ptr(), dst_ub_stage.ptr(),
                                                                               num_src_rows);
        } else if constexpr (not kFloat and kMajor == Major::MN) {
            constexpr uint32_t kElemsPerDataBlock = 32 / sizeof(uint16_t);
            const uint32_t burst_len = ceil_div(num_src_cols, kElemsPerDataBlock);
            asc_copy_ub2ub(dst_ub_stage.ptr(), src_ub_stage.ptr(), num_src_rows, burst_len,
                           ceil_div(kSrcUBStride, kElemsPerDataBlock) - burst_len,
                           ceil_div(kDstUBStride, kElemsPerDataBlock) - burst_len);
        }

        asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(stage));

        if constexpr (kGranMN > 1) {
            repeat_simd<kSrcBlockM, kDstBlockM, kDstUBStride, kDstUBStride, kGranMN>(dst_ub_stage.ptr(),
                                                                                     dst_ub_stage.ptr(), dst_actual_k);
        }

        asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(stage));
        asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(stage));

        const uint64_t dst_offset = (static_cast<uint64_t>(batch_idx) * dst_shape_k + dst_k_idx) * dst_shape_m + dst_m_idx;
        copy_ub_to_gm(dst_gm + dst_offset, dst_ub_stage, dst_actual_k, dst_actual_m, dst_shape_m * sizeof(uint16_t),
                      kL2CtrlStore);

        asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(stage));
        if (++stage == kNumStages)
            stage = 0;
    }
    wait_flags<PIPE_V, PIPE_MTE2, kNumStages>();
    wait_flags<PIPE_MTE3, PIPE_V, kNumStages>();

    if constexpr (kFloat) {
        asc_sync_pipe(PIPE_ALL);
        check_validation<kValidationElems>(validation_ub.ptr());
    }
}

} // namespace deep_gemm
