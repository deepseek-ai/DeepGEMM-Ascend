#pragma once

#include <c_api/asc_simd.h>
#include <deep_gemm/common.hpp>
#include <deep_gemm/ascend/tile_ptr.hpp>

namespace deep_gemm {

__aicore__ inline void set_l0c_nz2dn(uint64_t src_matrix_elems, uint64_t dst_matrix_elems) {
    asc_set_l0c2gm_nz2nd(1, src_matrix_elems / FRAC_MN, dst_matrix_elems);
    asc_set_l0c2gm_channel_para(static_cast<uint64_t>(1) << 48);
}

__aicore__ inline void set_mte2_nz(
    uint16_t num_batch, uint16_t dst_nz_n_stride, uint16_t dst_nz_c0_stride,
    uint16_t dst_batch_stride_blocks = 0
) {
    asc_set_gm2l1_nz_para(static_cast<uint64_t>(num_batch) | (static_cast<uint64_t>(dst_nz_n_stride) << 16) |
                          (static_cast<uint64_t>(dst_nz_c0_stride) << 32) |
                          (static_cast<uint64_t>(dst_batch_stride_blocks) << 48));
}

template <uint32_t kNumBatch = 1, typename dtype_t>
__aicore__ inline void copy_ub_to_gm(
    __gm__ dtype_t* dst_ptr, __ubuf__ dtype_t* src_ptr,
    uint64_t shape_m, uint64_t shape_n,
    uint64_t dst_stride_bytes,
    asc_store_l2_cache_mode l2_ctrl = asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
    uint64_t dst_batch_stride_bytes = 0, uint64_t src_batch_stride_bytes = 0
) {
    if constexpr (kNumBatch >= 2) {
        asc_set_ub2gm_loop_size(kNumBatch, 1);
        asc_set_ub2gm_loop1_stride(src_batch_stride_bytes, dst_batch_stride_bytes);
    }
    asc_copy_ub2gm_align(dst_ptr, src_ptr,
                         /* burst_num */ shape_m,
                         /* burst_len */ shape_n * sizeof(dtype_t),
                         l2_ctrl,
                         /* burst_dst_stride */ dst_stride_bytes,
                         /* burst_src_stride */ shape_n * sizeof(dtype_t));
    if constexpr (kNumBatch >= 2)
        asc_set_ub2gm_loop_size(1, 1);
}

template <uint32_t kNumBatch = 1, typename T>
__aicore__ inline void copy_gm_to_ub(
    ub_ptr<T> dst, __gm__ T* src,
    uint32_t n_rows, uint32_t n_cols_bytes,
    uint64_t src_stride_bytes, uint32_t dst_stride_bytes,
    asc_load_l2_cache_mode l2_ctrl = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
    uint8_t right_padding = 0,
    bool enable_constant_pad = false,
    uint64_t dst_batch_stride_bytes = 0, uint64_t src_batch_stride_bytes = 0
) {
    if constexpr (kNumBatch >= 2) {
        asc_set_gm2ub_loop_size(kNumBatch, 1);
        asc_set_gm2ub_loop1_stride(src_batch_stride_bytes, dst_batch_stride_bytes);
    }
    asc_copy_gm2ub_align(dst.ptr(), src, n_rows, n_cols_bytes, 0, right_padding, enable_constant_pad, l2_ctrl,
                         src_stride_bytes, dst_stride_bytes);
    if constexpr (kNumBatch >= 2)
        asc_set_gm2ub_loop_size(1, 1);
}

template <uint32_t kNumBatch = 1, typename T, Major M>
__aicore__ inline void copy_gm_to_ub(
    ub_ptr<T> dst, gm_ptr<T, M> src,
    uint32_t valid_m, uint32_t valid_k,
    asc_load_l2_cache_mode l2_ctrl = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
    bool enable_constant_pad = false,
    uint64_t dst_batch_stride_bytes = 0, uint64_t src_batch_stride_bytes = 0
) {
    const uint32_t rows = M == Major::K ? valid_m : valid_k;
    const uint32_t cols = M == Major::K ? valid_k : valid_m;
    constexpr uint32_t kBlockElems = 32 / sizeof(T);
    const uint8_t right_padding = static_cast<uint8_t>((kBlockElems - cols % kBlockElems) % kBlockElems);
    copy_gm_to_ub<kNumBatch>(dst, src.ptr(), rows, cols * sizeof(T), src.stride_outer * sizeof(T), dst.shape_n * sizeof(T),
                            l2_ctrl, right_padding, enable_constant_pad, dst_batch_stride_bytes, src_batch_stride_bytes);
}

template <uint32_t kNumBatch = 1, typename dtype_t>
__aicore__ inline void copy_gm_to_ub(
    __ubuf__ dtype_t* dst_ptr, __gm__ dtype_t* src_ptr,
    uint64_t shape_m, uint64_t shape_n,
    uint64_t src_stride_bytes,
    asc_load_l2_cache_mode l2_ctrl = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
    uint64_t dst_batch_stride_bytes = 0, uint64_t src_batch_stride_bytes = 0
) {
    copy_gm_to_ub<kNumBatch>(ub_ptr<dtype_t>(shape_m, shape_n, reinterpret_cast<uintptr_t>(dst_ptr)), src_ptr,
                            shape_m, shape_n * sizeof(dtype_t), src_stride_bytes, shape_n * sizeof(dtype_t),
                            l2_ctrl, 0, false, dst_batch_stride_bytes, src_batch_stride_bytes);
}

template <uint32_t kNumBatch = 1, typename T>
__aicore__ inline void copy_gm_to_ub(
    ub_ptr<T> dst, __gm__ T* src,
    uint64_t src_stride_bytes,
    asc_load_l2_cache_mode l2_ctrl = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
    uint64_t dst_batch_stride_bytes = 0, uint64_t src_batch_stride_bytes = 0
) {
    copy_gm_to_ub<kNumBatch>(dst, src, dst.shape_m, dst.shape_n * sizeof(T), src_stride_bytes, dst.shape_n * sizeof(T),
                            l2_ctrl, 0, false, dst_batch_stride_bytes, src_batch_stride_bytes);
}

template <bool kSrcIsNZ = false, typename T, Major M>
__aicore__ inline void copy_gm_to_l1(
    l1_ptr<T, M> dst, gm_ptr<T, M> src,
    uint32_t actual_mn, uint32_t actual_k,
    asc_load_l2_cache_mode l2_ctrl = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
    uint16_t num_batch = 1, uint16_t dst_batch_stride_blocks = 0,
    uint64_t src_batch_stride_bytes = 0
) {
    constexpr uint32_t pack_factor = is_fp4<T>() ? 2 : 1;
    using copy_dtype_t = std::conditional_t<is_fp4<T>(), int8_t, T>;
    if constexpr (kSrcIsNZ) {
        static_assert(M == Major::K, "Direct NZ copy requires K-major GM storage");
        const uint32_t page_size = dst_batch_stride_blocks;
        const uint32_t num_fragments = actual_k / (32 * pack_factor);
        asc_set_gm2l1_loop_size(num_batch, 1);
        asc_set_gm2l1_loop1_stride(src_batch_stride_bytes, page_size * 32);
        asc_set_gm2l1_loop2_stride(0, 0);
        asc_copy_gm2l1_align(dst.template ptr<uint8_t>(), src.template ptr<uint8_t>(), num_fragments, actual_mn * 32, 0,
                             0, false, l2_ctrl, page_size * 32, dst.shape_mn * 32);
    } else if constexpr (M == Major::K) {
        set_mte2_nz(num_batch, 1, dst.shape_mn, dst_batch_stride_blocks);
        asc_copy_gm2l1_nd2nz(dst.template ptr<copy_dtype_t>(), src.template ptr<copy_dtype_t>(),
                             src.stride_outer * sizeof(T), l2_ctrl, actual_mn, actual_k / pack_factor,
                             src_batch_stride_bytes, false);
    } else {
        set_mte2_nz(num_batch, 1, dst.shape_k, dst_batch_stride_blocks);
        asc_copy_gm2l1_nd2nz(dst.template ptr<copy_dtype_t>(), src.template ptr<copy_dtype_t>(),
                             src.stride_outer * sizeof(T), l2_ctrl, actual_k, actual_mn / pack_factor,
                             src_batch_stride_bytes, false);
    }
}

template <typename T, Major M>
__aicore__ inline void copy_l1_to_l0a(l0a_ptr<T> dst, l1_ptr<T, M> src, uint64_t mn_idx, uint64_t k_idx) {
    constexpr uint32_t FRAC_K = get_frac_k<T>();
    if constexpr (M == Major::K) {
        asc_copy_l12l0a(dst.ptr(), src.ptr(), mn_idx / FRAC_MN, k_idx / FRAC_K, dst.shape_mn / FRAC_MN,
                        dst.shape_k / FRAC_K, src.shape_mn / FRAC_MN, dst.shape_mn / FRAC_MN);
    } else {
        uint32_t m_step = dst.shape_k / FRAC_MN;
        if constexpr (sizeof(T) == 1) m_step = (m_step + 1) / 2 * 2;
        asc_copy_l12l0a_transpose(dst.ptr(), src.ptr(), k_idx / FRAC_MN, mn_idx / FRAC_K, m_step,
                                  ceil_div(dst.shape_mn, FRAC_K), src.shape_k / FRAC_MN, dst.shape_mn / FRAC_MN);
    }
}

template <typename T, Major M>
__aicore__ inline void copy_l1_to_l0b(l0b_ptr<T> dst, l1_ptr<T, M> src, uint64_t mn_idx, uint64_t k_idx) {
    constexpr uint32_t FRAC_K = get_frac_k<T>();
    if constexpr (M == Major::K) {
        asc_copy_l12l0b(dst.ptr(), src.ptr(), mn_idx / FRAC_MN, k_idx / FRAC_K, dst.shape_mn / FRAC_MN,
                        dst.shape_k / FRAC_K, src.shape_mn / FRAC_MN, dst.shape_mn / FRAC_MN);
    } else {
        uint32_t m_step = dst.shape_k / FRAC_MN;
        if constexpr (sizeof(T) == 1) m_step = (m_step + 1) / 2 * 2;
        asc_copy_l12l0b_transpose(dst.ptr(), src.ptr(), k_idx / FRAC_MN, mn_idx / FRAC_K, m_step,
                                  ceil_div(dst.shape_mn, FRAC_K), src.shape_k / FRAC_MN, dst.shape_mn / FRAC_MN);
    }
}

template <bool kSrcIsNZ = false, Major M>
__aicore__ inline void copy_gm_to_l1_mx(
    l1_ptr<int16_t> dst, gm_ptr<int16_t, M> src,
    uint32_t mn_elems, uint32_t sf_pairs,
    asc_load_l2_cache_mode l2_ctrl = asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
    uint16_t num_batch = 1, uint16_t dst_batch_stride_blocks = 0,
    uint64_t src_batch_stride_bytes = 0
) {
    if constexpr (kSrcIsNZ) {
        const uint32_t page_bytes = dst_batch_stride_blocks * 32;
        const uint32_t valid_bytes = ceil_div(mn_elems, FRAC_MN) * sf_pairs * 32;
        asc_set_gm2l1_loop_size(1, 1);
        asc_set_gm2l1_loop1_stride(0, 0);
        asc_set_gm2l1_loop2_stride(0, 0);
        asc_copy_gm2l1_align(dst.ptr(), src.ptr(), num_batch, valid_bytes, 0, 0, false, l2_ctrl, src_batch_stride_bytes,
                             page_bytes);
    } else {
        set_mte2_nz(num_batch, 1, dst.shape_k, dst_batch_stride_blocks);
        if constexpr (M == Major::K) {
            asc_copy_gm2l1_dn2nz(dst.ptr(), src.ptr(),
                                 /* stride */ src.stride_outer * sizeof(int16_t),
                                 /* l2ctrl */ l2_ctrl,
                                 /* n (rows) */ sf_pairs,
                                 /* d (cols) */ mn_elems,
                                 /* batch_stride */ src_batch_stride_bytes,
                                 /* small */ false);
        } else {
            asc_copy_gm2l1_nd2nz(dst.ptr(), src.ptr(),
                                 /* stride */ src.stride_outer * sizeof(int16_t),
                                 /* l2ctrl */ l2_ctrl,
                                 /* n (rows) */ sf_pairs,
                                 /* d (cols) */ mn_elems,
                                 /* batch_stride */ src_batch_stride_bytes,
                                 /* small */ false);
        }
    }
}

template <typename T>
__aicore__ inline void copy_l1_to_l0a_mx(
    l0a_ptr<T> l0a, l1_ptr<int16_t> sf_l1,
    uint64_t m_idx, uint16_t k_idx,
    uint32_t eff_k = 0
) {
    uint16_t sf_pairs = (eff_k ? eff_k : l0a.shape_k) / MX_SF_DIVISOR;
    asc_copy_l12l0a_mx(l0a.addr / MX_ADDR_DIV, sf_l1.ptr<fp8_e8m0_t>(),
                       /* x_start */ m_idx / FRAC_MN,
                       /* y_start */ k_idx,
                       /* x_step */ l0a.shape_mn / FRAC_MN,
                       /* y_step */ sf_pairs,
                       /* src_stride */ sf_l1.shape_k,
                       /* dst_stride */ sf_pairs);
}

template <typename T>
__aicore__ inline void copy_l1_to_l0b_mx(
    l0b_ptr<T> l0b, l1_ptr<int16_t> sf_l1,
    uint64_t m_idx, uint16_t k_idx,
    uint32_t eff_k = 0
) {
    uint16_t sf_pairs = (eff_k ? eff_k : l0b.shape_k) / MX_SF_DIVISOR;
    asc_copy_l12l0b_mx(l0b.addr / MX_ADDR_DIV, sf_l1.ptr<fp8_e8m0_t>(),
                       /* x_start */ m_idx / FRAC_MN,
                       /* y_start */ k_idx,
                       /* x_step */ l0b.shape_mn / FRAC_MN,
                       /* y_step */ sf_pairs,
                       /* src_stride */ sf_l1.shape_k,
                       /* dst_stride */ sf_pairs);
}

template <typename T, typename ST = float>
constexpr __aicore__ asc_quant_mode infer_quant_mode(bool scalar = false) {
    if constexpr (std::is_same_v<ST, float>) {
        if constexpr (std::is_same_v<T, float>) {
            return asc_quant_mode::NoQuant;
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            return asc_quant_mode::F322BF16;
        } else if constexpr (std::is_same_v<T, half>) {
            return asc_quant_mode::F322F16;
        } else if constexpr (std::is_same_v<T, float8_e4m3_t>) {
            return scalar ? asc_quant_mode::QF322FP8_PRE : asc_quant_mode::VQF322FP8_PRE;
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            return scalar ? asc_quant_mode::QF322B8_PRE : asc_quant_mode::VQF322B8_PRE;
        } else if constexpr (std::is_same_v<T, int4x2_t>) {
            return scalar ? asc_quant_mode::QF322S4_PRE : asc_quant_mode::VQF322S4_PRE;
        } else if constexpr (std::is_same_v<T, hifloat8_t>) {
            return scalar ? asc_quant_mode::QF322HIF8_PRE : asc_quant_mode::VQF322HIF8_PRE;
        } else {
            static_assert(!std::is_same_v<T, T>, "Unsupported quantization type");
        }
    } else if constexpr (std::is_same_v<ST, int32_t>) {
        if constexpr (std::is_same_v<T, int32_t>) {
            return asc_quant_mode::NoQuant;
        } else if constexpr (std::is_same_v<T, int8_t>) {
            return scalar ? asc_quant_mode::REQ8 : asc_quant_mode::VREQ8;
        } else if constexpr (std::is_same_v<T, int4x2_t>) {
            return scalar ? asc_quant_mode::REQ4 : asc_quant_mode::VREQ4;
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            return scalar ? asc_quant_mode::DEQF16 : asc_quant_mode::VDEQF16;
        } else {
            static_assert(!std::is_same_v<T, T>, "Unsupported quantization type");
        }
    }
}

template <typename T, typename ST>
__aicore__ inline void copy_l0c_to_ub(
    ub_ptr<T> dst, l0c_ptr<ST> src,
    uint32_t m_elems, uint32_t n_elems,
    asc_dual_dst_mode dual_dst_ctrl,
    uint32_t dst_subblock_id = 0,
    asc_quant_mode quant_mode = infer_quant_mode<T, ST>(),
    bool relu = false,
    bool transpose_out = false,  // false: write row-major [M, N]; true: col-major [N, M]
    asc_unit_flag_mode unit_flag = asc_unit_flag_mode::ENABLE_UPDATE
) {
    if (transpose_out) {
        set_l0c_nz2dn(src.shape_m * src.shape_n, n_elems * dst.shape_n);
    } else {
        asc_set_l0c2gm_nz2nd(1, 0, 0);
    }
    asc_copy_l0c2ub(dst.ptr(), src.ptr(), n_elems, m_elems, dst.shape_n, src.shape_m, dst_subblock_id, dual_dst_ctrl,
                    unit_flag, quant_mode, relu ? asc_relu_pre_mode::NORMAL : asc_relu_pre_mode::NONE, false,
                    !transpose_out, transpose_out, false);
}

template <typename T, typename ST>
__aicore__ inline void copy_l0c_to_gm(
    __gm__ T* dst, l0c_ptr<ST> src,
    uint32_t m_elems, uint32_t n_elems,
    uint64_t dst_stride_elems,
    asc_store_l2_cache_mode l2_ctrl = asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
    asc_quant_mode quant_mode = infer_quant_mode<T, ST>(),
    bool relu = false,
    bool transpose_out = false,  // false: write row-major [M, N]; true: col-major [N, M]
    asc_unit_flag_mode unit_flag = asc_unit_flag_mode::ENABLE_UPDATE
) {
    if (transpose_out) {
        set_l0c_nz2dn(src.shape_m * src.shape_n, n_elems * dst_stride_elems);
    } else {
        asc_set_l0c2gm_nz2nd(1, 0, 0);
    }
    asc_copy_l0c2gm(dst, src.ptr(), n_elems, m_elems, dst_stride_elems, src.shape_m, l2_ctrl, unit_flag, quant_mode,
                    relu ? asc_relu_pre_mode::NORMAL : asc_relu_pre_mode::NONE, false, !transpose_out, transpose_out,
                    false);
}

template <uint32_t kNumBatch = 1, typename T>
__aicore__ inline void copy_ub_to_gm(
    __gm__ T* dst, ub_ptr<T> src,
    uint32_t n_rows, uint32_t n_cols,
    uint64_t dst_stride_bytes,
    asc_store_l2_cache_mode l2_ctrl = asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
    uint64_t dst_batch_stride_bytes = 0, uint64_t src_batch_stride_bytes = 0
) {
    if constexpr (kNumBatch >= 2) {
        asc_set_ub2gm_loop_size(kNumBatch, 1);
        asc_set_ub2gm_loop1_stride(src_batch_stride_bytes, dst_batch_stride_bytes);
    }
    asc_copy_ub2gm_align(dst, src.ptr(), n_rows, n_cols * sizeof(T), l2_ctrl, dst_stride_bytes,
                         src.shape_n * sizeof(T));
    if constexpr (kNumBatch >= 2)
        asc_set_ub2gm_loop_size(1, 1);
}

template <typename T>
__aicore__ inline void fill_l1(
    __cbuf__ T* dst, T value,
    uint32_t num_rows, uint32_t col_blks_32b,
    uint32_t dst_gap_32b
) {
    asc_fill_value_config config;
    config.repeat = num_rows;
    config.blk_num = col_blks_32b;
    config.dst_gap = dst_gap_32b;
    using value_t = std::conditional_t<std::is_integral_v<T>, uint32_t, T>;
    asc_fill_l1(dst, static_cast<value_t>(value), config);
}

template <typename T, Major M>
__aicore__ inline void fill_tail_block(l1_ptr<T, M> ptr, uint32_t actual_k) {
    static_assert(sizeof(T) == 1, "Expect 8-bit (fp8) or packed-4bit (fp4) data type");
    constexpr uint32_t FRAC_K = get_frac_k<T>();
    const uint32_t k64 = aligned(actual_k, 64);
    if constexpr (M == Major::K) {
        const uint32_t nz32 = aligned(actual_k, 32);
        if (nz32 < k64) {
            fill_l1(reinterpret_cast<__cbuf__ int16_t*>(ptr.ptr() + (nz32 / 32) * ptr.shape_mn * 32),
                    static_cast<int16_t>(0),
                    /* num_rows */ 1,
                    /* col_blks_32b */ ptr.shape_mn,
                    /* dst_gap_32b */ 0);
        }
    } else {
        if (actual_k < k64) {
            fill_l1(reinterpret_cast<__cbuf__ int16_t*>(ptr.ptr() + actual_k * 32),
                    static_cast<int16_t>(0),
                    /* num_rows */ ptr.shape_mn / FRAC_K,
                    /* col_blks_32b */ k64 - actual_k,
                    /* dst_gap_32b */ ptr.shape_k - (k64 - actual_k));
        }
    }
}

template <typename T>
__aicore__ inline void fill_mn_tail_block(l1_ptr<T, Major::K> ptr, uint32_t actual_mn, uint32_t padded_mn) {
    static_assert(sizeof(T) == 1, "Expect 8-bit (fp8) or packed-4bit (fp4) data type");
    // ptr.shape_mn remains the physical NZ stride; padded_mn only bounds the
    // region consumed by the following L1->L0 transfer.
    if (actual_mn >= padded_mn) return;
    const uint32_t k_blocks = ptr.shape_k / get_frac_k<T>();  // one 32B row per C0
    fill_l1(reinterpret_cast<__cbuf__ int16_t*>(ptr.ptr() + actual_mn * 32), static_cast<int16_t>(0),
            /* num_rows */ k_blocks, /* col_blks_32b */ padded_mn - actual_mn,
            /* dst_gap_32b */ ptr.shape_mn - (padded_mn - actual_mn));
}

template <typename TA, typename TB, typename TC>
__aicore__ inline void mad(l0c_ptr<TC> l0c, l0a_ptr<TA> l0a, l0b_ptr<TB> l0b,
                           uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                           asc_unit_flag_mode unit_flag, bool clear_C = false) {
    if constexpr (sizeof(TA) == 1 || sizeof(TB) == 1) {
        asc_mmad_mx(l0c.ptr(), l0a.ptr(), l0b.ptr(), shape_m, shape_k, shape_n, unit_flag, true,
                    false, clear_C);
    } else {
        asc_mmad(l0c.ptr(), l0a.ptr(), l0b.ptr(), shape_m, shape_k, shape_n, unit_flag, true,
                 false, clear_C);
    }
}

} // namespace deep_gemm
