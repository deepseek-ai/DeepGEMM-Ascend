#pragma once

#include <deep_gemm/layout/mega_mhc.hpp>
#include <deep_gemm/simd/mega_mhc_epilogue.hpp>
#include <c_api/cube_compute/cube_compute.h>

namespace deep_gemm::mega_mhc {

using namespace layout::mega_mhc;

template <
    uint32_t SHAPE_H, uint32_t BLOCK_M, uint32_t BLOCK_H, uint32_t SUB_BLOCK_H,
    uint32_t kNumKSplits, uint32_t kMainUBEndOffset, uint32_t kNumAIVCores,
    uint32_t kBufOffset, bool kIsSFRowMajor,
    bool kStoreBF16, bool kStoreFP8, uint32_t kRMSSqrSumBlockM, bool kSplitNormQuant
>
struct NormQuantEpilogue {
    // Computation and task partitioning
    static_assert(SHAPE_H % BLOCK_H == 0 and BLOCK_H % SUB_BLOCK_H == 0);
    static constexpr uint32_t kNumHBlocks = SHAPE_H / BLOCK_H;
    static constexpr uint32_t kNumTasksPerMBlock = kNumHBlocks;
    static constexpr uint32_t kNumStepsPerTask = BLOCK_M;

    // Block Y and FP8 reuse the main UB region after its final stores complete
    static constexpr ub_ptr<bfloat16_t> ub_block_y{BLOCK_M, BLOCK_H};
    static constexpr ub_ptr<float8_e4m3_t> ub_block_fp8{BLOCK_M, BLOCK_H, ub_block_y.offset(1)};
    static constexpr uint32_t kUBOffset = aligned(kMainUBEndOffset > ub_block_fp8.offset(1) ? kMainUBEndOffset : ub_block_fp8.offset(1), sizeof(vector_float));

    // Y buffers and stages
    static constexpr uint32_t kNumYBF16Stages = 4;
    static constexpr ub_ptr<bfloat16_t> ub_y_bf16{1, BLOCK_H, kUBOffset};

    // RMSNorm buffers and stages
    static constexpr uint32_t kNumRMSNormWeightStages = 1;
    static constexpr uint32_t kNumRMSSqrSumStages = 2;
    static constexpr ub_ptr<bfloat16_t> ub_rmsnorm_weight{1, SHAPE_H, ub_y_bf16.offset(kNumYBF16Stages)};
    static constexpr ub_ptr<float> ub_rms_sqr_sum{kNumKSplits, aligned(BLOCK_M, kDataBlockBytes / sizeof(float)), ub_rmsnorm_weight.offset(kNumRMSNormWeightStages)};

    // Quant output buffers and stages
    static constexpr uint32_t kNumSFStages = 2;
    static constexpr uint32_t kNumSFPairsPerRow = BLOCK_H / MX_SF_DIVISOR;
    static constexpr uint32_t kNumYFP8Stages = kNumYBF16Stages;
    static constexpr ub_ptr<float8_e4m3_t> ub_y_fp8{1, BLOCK_H, ub_rms_sqr_sum.offset(kNumRMSSqrSumStages)};
    static constexpr ub_ptr<uint16_t> ub_y_sf{kIsSFRowMajor ? BLOCK_M : kNumSFPairsPerRow,
        aligned(kIsSFRowMajor ? kNumSFPairsPerRow : BLOCK_M, kDataBlockBytes / sizeof(uint16_t)), ub_y_fp8.offset(kNumYFP8Stages)};
    static constexpr uint32_t kSFStride = kIsSFRowMajor ? 1 : ub_y_sf.shape_n;
    static constexpr uint32_t kSFRowStride = kIsSFRowMajor ? ub_y_sf.shape_n : 1;

    // Buffer IDs and capacity constraints
    // Y and FP8 share one ID through the load, compute and store sequence
    static constexpr uint32_t kYFP8BufOffset = kBufOffset;
    static constexpr uint32_t kYBF16BufOffset = kYFP8BufOffset;
    static constexpr uint32_t kRMSSqrSumBufOffset = kYFP8BufOffset + kNumYFP8Stages;
    static constexpr uint32_t kSFBufOffset = kRMSSqrSumBufOffset + kNumRMSSqrSumStages;
    static constexpr uint32_t kRMSNormWeightBufOffset = kSFBufOffset + kNumSFStages;

    // Block mode borrows row slot 1 only for the early BF16 store
    // Row slot 0 remains held through block Norm/Quant and the final MTE3 store
    static constexpr uint32_t kBlockYBF16StoreBufId = kYBF16BufOffset + 1;

    // Resource boundaries
    static constexpr uint32_t kUBEndOffset = ub_y_sf.offset(kNumSFStages);
    static constexpr uint32_t kBufEndOffset = kRMSNormWeightBufOffset + kNumRMSNormWeightStages;

    // Storage and synchronization constraints
    static_assert(kUBEndOffset <= UBSizeBytes);
    static_assert(kBufEndOffset <= 32);
    // The prefetch window must not reuse an RMS slot before its last row is consumed
    static_assert(kNumYBF16Stages <= (kNumRMSSqrSumStages - 1) * kNumStepsPerTask + 1);

    __aicore__ static constexpr uint32_t get_ub_end_offset() { return kUBEndOffset; }

    // Runtime inputs
    uint32_t shape_m;
    uint32_t aiv_idx;
    gm_ptr<bfloat16_t> gm_y_bf16;
    gm_ptr<bfloat16_t> gm_rmsnorm_weight;
    gm_ptr<float8_e4m3_t> gm_y_fp8;
    gm_ptr<uint16_t, kIsSFRowMajor ? Major::K : Major::MN> gm_y_sf;
    gm_ptr<float> gm_rms_sqr_sum;
    float rmsnorm_eps;
    float rmsnorm_scale;

    // Task admission and step cursors
    const uint32_t num_tasks = ceil_div(shape_m, BLOCK_M) * kNumTasksPerMBlock;
    uint32_t num_ready_tasks = 0;
    uint32_t prefetch_step_idx = 0;
    uint32_t norm_quant_step_idx = 0;

    __aicore__ NormQuantEpilogue(const NormArgs& args, uint32_t shape_m, uint32_t aiv_idx, gm_ptr<float> gm_rms_sqr_sum)
        : shape_m(shape_m), aiv_idx(aiv_idx), gm_y_bf16(args.y_bf16), gm_rmsnorm_weight(args.weight),
          gm_y_fp8(args.y_fp8), gm_y_sf(args.y_sf_stride, args.y_sf.addr), gm_rms_sqr_sum(gm_rms_sqr_sum),
          rmsnorm_eps(args.eps), rmsnorm_scale(args.scale) {}

    __aicore__ inline bool has_ready_steps() const {
        return aiv_idx + norm_quant_step_idx / kNumStepsPerTask * kNumAIVCores < num_ready_tasks;
    }

    // Admit the chunk's global M range after the caller joins its completion notification
    __aicore__ inline void prepare(uint32_t num_m_blocks) {
        if (num_ready_tasks == 0 and aiv_idx < num_tasks) {
            // A single task or whole H waves keep this AIV on one weight tile.
            const bool fixed_h = num_tasks <= kNumAIVCores or kNumAIVCores % kNumHBlocks == 0;
            const uint32_t num_weight_elems = fixed_h ? BLOCK_H : SHAPE_H;
            const uint32_t h_idx = fixed_h ? aiv_idx % kNumHBlocks * BLOCK_H : 0;
            asc_lock(PIPE_MTE2, kRMSNormWeightBufOffset);
            copy_gm_to_ub(ub_rmsnorm_weight.ptr() + h_idx, gm_rmsnorm_weight.ptr() + h_idx, 1, num_weight_elems,
                          num_weight_elems * sizeof(bfloat16_t), asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM);
            asc_unlock(PIPE_MTE2, kRMSNormWeightBufOffset);
        }
        num_ready_tasks = min(num_ready_tasks + num_m_blocks * kNumTasksPerMBlock, num_tasks);
    }

    __aicore__ inline void load_rms_sqr_sum_gm2ub(uint32_t m_idx, uint32_t stage_idx) {
        const uint32_t actual_m = min(shape_m - m_idx, BLOCK_M);
        const uint64_t workspace_m_idx = m_idx / kRMSSqrSumBlockM * kRMSSqrSumBlockM * kNumKSplits + m_idx % kRMSSqrSumBlockM;
        asc_lock(PIPE_MTE2, kRMSSqrSumBufOffset + stage_idx);
        asc_copy_gm2ub_align(ub_rms_sqr_sum[stage_idx].ptr(), gm_rms_sqr_sum.index(workspace_m_idx, 0).ptr(),
                             kNumKSplits, actual_m * sizeof(float), 0, 0, false, asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM,
                             kRMSSqrSumBlockM * sizeof(float), ub_rms_sqr_sum.shape_n * sizeof(float));
        asc_unlock(PIPE_MTE2, kRMSSqrSumBufOffset + stage_idx);
    }

    __aicore__ inline void load_y_bf16_gm2ub(uint32_t m_idx, uint32_t h_idx, uint32_t stage_idx) {
        asc_lock(PIPE_MTE2, kYBF16BufOffset + stage_idx);
        copy_gm_to_ub(ub_y_bf16[stage_idx].ptr(), gm_y_bf16.index(m_idx, h_idx).ptr(),
                      1, BLOCK_H, SHAPE_H * sizeof(bfloat16_t), asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM);
        asc_unlock(PIPE_MTE2, kYBF16BufOffset + stage_idx);
    }

    // Issue bounded lookahead without changing the consumer cursor
    __aicore__ inline void prefetch(uint32_t num_steps = 1) {
        for (uint32_t step_idx = 0; step_idx < num_steps; ++ step_idx) {
            const uint32_t task_idx_in_core = prefetch_step_idx / kNumStepsPerTask;
            const uint32_t task_idx = aiv_idx + task_idx_in_core * kNumAIVCores;
            if (task_idx >= num_ready_tasks)
                return;
            if (prefetch_step_idx >= norm_quant_step_idx + kNumYBF16Stages)
                return;
            const uint32_t m_idx = task_idx / kNumHBlocks * BLOCK_M + prefetch_step_idx % BLOCK_M;
            const uint32_t h_idx = task_idx % kNumHBlocks * BLOCK_H;
            if (m_idx >= shape_m) {
                prefetch_step_idx = (task_idx_in_core + 1) * kNumStepsPerTask;
                return;
            }
            if (prefetch_step_idx % kNumStepsPerTask == 0)
                load_rms_sqr_sum_gm2ub(m_idx, task_idx_in_core % kNumRMSSqrSumStages);
            load_y_bf16_gm2ub(m_idx, h_idx, prefetch_step_idx % kNumYBF16Stages);
            ++ prefetch_step_idx;
        }
    }

    __aicore__ inline void prefetch_all_stages() {
        prefetch(kNumYBF16Stages);
    }

    __aicore__ inline void store_sf_ub2gm(uint32_t m_idx, uint32_t h_idx, uint32_t stage_idx) {
        const uint32_t sub_m_idx = m_idx % BLOCK_M;
        asc_lock(PIPE_MTE3, kSFBufOffset + stage_idx);
        asc_copy_ub2gm_align(gm_y_sf.index(m_idx - sub_m_idx, h_idx / MX_SF_DIVISOR).ptr(), ub_y_sf[stage_idx].ptr(),
                             kIsSFRowMajor ? sub_m_idx + 1 : kNumSFPairsPerRow, (kIsSFRowMajor ? kNumSFPairsPerRow : sub_m_idx + 1) * sizeof(uint16_t),
                             asc_store_l2_cache_mode::NORMAL_LAST_VICTIM, gm_y_sf.stride_outer * sizeof(uint16_t), ub_y_sf.shape_n * sizeof(uint16_t));
        asc_unlock(PIPE_MTE3, kSFBufOffset + stage_idx);
    }

    // Block mode may reuse the main UB region only after its final chunk completes
    // use_block requires a full M block at its first row, with no pending row prefetches
    __aicore__ inline void step(bool use_block = false) {
        const uint32_t task_idx_in_core = norm_quant_step_idx / kNumStepsPerTask;
        const uint32_t task_idx = aiv_idx + task_idx_in_core * kNumAIVCores;
        if (task_idx >= num_ready_tasks)
            return;

        const uint32_t sub_m_idx = norm_quant_step_idx % BLOCK_M;
        const uint32_t m_idx = task_idx / kNumHBlocks * BLOCK_M + sub_m_idx;
        if (m_idx >= shape_m) {
            // Match prefetch tail skipping to keep Y/FP8 stages aligned
            norm_quant_step_idx = (task_idx_in_core + 1) * kNumStepsPerTask;
            return;
        }

        // Select storage and row count once; the rest shares one compute/store sequence
        static_assert(BLOCK_M * kDataBlockBytes <= BLOCK_H * sizeof(bfloat16_t));
        const uint32_t num_rows = use_block ? BLOCK_M : 1;
        const uint32_t stage_idx = use_block ? 0 : norm_quant_step_idx % kNumYFP8Stages;
        const uint32_t h_idx = task_idx % kNumHBlocks * BLOCK_H;
        const uint32_t rms_stage_idx = task_idx_in_core % kNumRMSSqrSumStages;
        const uint32_t sf_stage_idx = task_idx_in_core % kNumSFStages;
        const auto ub_y_bf16_ptr = use_block ? ub_block_y.ptr() : ub_y_bf16[stage_idx].ptr();
        const auto ub_y_fp8_ptr = use_block ? ub_block_fp8.ptr() : ub_y_fp8[stage_idx].ptr();
        const auto ub_rms_scale_ptr = ub_rms_sqr_sum[rms_stage_idx].ptr() + sub_m_idx;
        const uint32_t rms_sqr_sum_buf_id = kRMSSqrSumBufOffset + rms_stage_idx;

        // Blocks load together; row inputs are prefetched by the caller
        if (use_block) {
            load_rms_sqr_sum_gm2ub(m_idx, rms_stage_idx);
            asc_lock(PIPE_MTE2, kYBF16BufOffset + stage_idx);
            copy_gm_to_ub(ub_y_bf16_ptr, gm_y_bf16.index(m_idx, h_idx).ptr(), num_rows, BLOCK_H,
                          SHAPE_H * sizeof(bfloat16_t), asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM);
            asc_unlock(PIPE_MTE2, kYBF16BufOffset + stage_idx);
        }

        // Reduce RMS square sums once per task
        if (sub_m_idx == 0) {
            asc_lock(PIPE_V, rms_sqr_sum_buf_id);
            vf_reduce_rmsnorm_scale<SHAPE_H, BLOCK_M, kNumKSplits, ub_rms_sqr_sum.shape_n>(
                ub_rms_sqr_sum[rms_stage_idx].ptr(), rmsnorm_eps, rmsnorm_scale);
            asc_unlock(PIPE_V, rms_sqr_sum_buf_id);
        }

        // Normalize and quantize Y; input Y and FP8 share one buffer ID
        asc_lock(PIPE_V, kYFP8BufOffset + stage_idx);
        asc_lock(PIPE_V, kRMSNormWeightBufOffset);
        asc_lock(PIPE_V, rms_sqr_sum_buf_id);
        kStoreFP8 ? asc_lock(PIPE_V, kSFBufOffset + sf_stage_idx) : void();
        if (use_block) {
            // amax/inverse scales reuse row Y stage 0, protected by the outer slot-0 lock
            // The separate slot-1 ID below publishes BF16 readiness; it does not own amax
            if constexpr (kSplitNormQuant) {
                static_assert(BLOCK_H / (2 * SUB_BLOCK_H) * BLOCK_M * kDataBlockBytes <= BLOCK_H * sizeof(bfloat16_t));
                asc_lock(PIPE_V, kBlockYBF16StoreBufId);
                vf_norm_quant_block<
                    BLOCK_M, BLOCK_H, SUB_BLOCK_H, kSFStride, kStoreFP8, kSFRowStride,
                    /*kNormalize=*/true, /*kPrepareQuant=*/false, /*kApplyQuant=*/false>(
                    ub_y_bf16_ptr, ub_rms_scale_ptr, ub_rmsnorm_weight.ptr() + h_idx,
                    ub_y_fp8_ptr, ub_y_sf[sf_stage_idx].ptr(), reinterpret_cast<__ubuf__ uint16_t*>(ub_y_bf16.ptr()));
                asc_unlock(PIPE_V, kBlockYBF16StoreBufId);
                asc_lock(PIPE_MTE3, kBlockYBF16StoreBufId);
                copy_ub_to_gm(gm_y_bf16.index(m_idx, h_idx).ptr(), ub_y_bf16_ptr, num_rows, BLOCK_H,
                              SHAPE_H * sizeof(bfloat16_t), asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM);
                asc_unlock(PIPE_MTE3, kBlockYBF16StoreBufId);
                vf_norm_quant_block<
                    BLOCK_M, BLOCK_H, SUB_BLOCK_H, kSFStride, kStoreFP8, kSFRowStride,
                    /*kNormalize=*/false, /*kPrepareQuant=*/true, /*kApplyQuant=*/false>(
                    ub_y_bf16_ptr, ub_rms_scale_ptr, ub_rmsnorm_weight.ptr() + h_idx,
                    ub_y_fp8_ptr, ub_y_sf[sf_stage_idx].ptr(), reinterpret_cast<__ubuf__ uint16_t*>(ub_y_bf16.ptr()));
                asc_unlock(PIPE_V, kSFBufOffset + sf_stage_idx);
                store_sf_ub2gm(m_idx + num_rows - 1, h_idx, sf_stage_idx);
                vf_norm_quant_block<
                    BLOCK_M, BLOCK_H, SUB_BLOCK_H, kSFStride, kStoreFP8, kSFRowStride,
                    /*kNormalize=*/false, /*kPrepareQuant=*/false, /*kApplyQuant=*/true>(
                    ub_y_bf16_ptr, ub_rms_scale_ptr, ub_rmsnorm_weight.ptr() + h_idx,
                    ub_y_fp8_ptr, ub_y_sf[sf_stage_idx].ptr(), reinterpret_cast<__ubuf__ uint16_t*>(ub_y_bf16.ptr()));
            } else {
                vf_norm_quant_block<BLOCK_M, BLOCK_H, SUB_BLOCK_H, kSFStride, kStoreFP8, kSFRowStride>(
                    ub_y_bf16_ptr, ub_rms_scale_ptr, ub_rmsnorm_weight.ptr() + h_idx,
                    ub_y_fp8_ptr, ub_y_sf[sf_stage_idx].ptr(), reinterpret_cast<__ubuf__ uint16_t*>(ub_y_bf16.ptr()));
            }
        } else {
            vf_norm_quant_row<BLOCK_H, SUB_BLOCK_H, kSFStride, kStoreFP8>(
                ub_y_bf16_ptr, ub_rms_scale_ptr, ub_rmsnorm_weight.ptr() + h_idx, ub_y_fp8_ptr,
                ub_y_sf[sf_stage_idx].ptr(), sub_m_idx * kSFRowStride);
        }
        (kStoreFP8 and not (kSplitNormQuant and use_block)) ? asc_unlock(PIPE_V, kSFBufOffset + sf_stage_idx) : void();
        asc_unlock(PIPE_V, rms_sqr_sum_buf_id);
        asc_unlock(PIPE_V, kRMSNormWeightBufOffset);
        asc_unlock(PIPE_V, kYFP8BufOffset + stage_idx);

        // Store the requested BF16/FP8 outputs
        asc_lock(PIPE_MTE3, kYFP8BufOffset + stage_idx);
        kStoreFP8 ? copy_ub_to_gm(gm_y_fp8.index(m_idx, h_idx).ptr(), ub_y_fp8_ptr, num_rows, BLOCK_H,
                                  SHAPE_H, asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM) : void();
        (kStoreBF16 and not (kSplitNormQuant and use_block)) ? copy_ub_to_gm(gm_y_bf16.index(m_idx, h_idx).ptr(), ub_y_bf16_ptr, num_rows, BLOCK_H,
                                  SHAPE_H * sizeof(bfloat16_t), asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM) : void();
        // MTE3 FIFO also completes the early block BF16 store before slot 0 is reused
        asc_unlock(PIPE_MTE3, kYFP8BufOffset + stage_idx);

        // Store accumulated SF after the block's final valid row
        if (kStoreFP8 and not (kSplitNormQuant and use_block) and (sub_m_idx + num_rows == BLOCK_M or m_idx + num_rows == shape_m))
            store_sf_ub2gm(m_idx + num_rows - 1, h_idx, sf_stage_idx);

        norm_quant_step_idx += num_rows;
        if (use_block)
            prefetch_step_idx = norm_quant_step_idx;
    }

    // Drain after the main UB is free; finish prefetched and partial tasks row by row
    __aicore__ inline void drain() {
        while (has_ready_steps()) {
            const uint32_t task_idx_in_core = norm_quant_step_idx / kNumStepsPerTask;
            const uint32_t task_idx = aiv_idx + task_idx_in_core * kNumAIVCores;
            const uint32_t m_idx = task_idx / kNumHBlocks * BLOCK_M;
            const uint32_t end_step = (task_idx_in_core + 1) * kNumStepsPerTask;
            const bool use_block = norm_quant_step_idx >= prefetch_step_idx and norm_quant_step_idx % BLOCK_M == 0 and shape_m - m_idx >= BLOCK_M;

            if (not use_block and prefetch_step_idx < end_step)
                prefetch(min(kNumYBF16Stages, end_step - prefetch_step_idx));
            step(use_block);
            if (not use_block and prefetch_step_idx < end_step)
                prefetch();
        }
    }
};

template <
    uint32_t SHAPE_H, uint32_t SHAPE_N, uint32_t BLOCK_M, uint32_t kNumRoutes,
    uint32_t kNumKSplits, uint32_t kNumAIVCores, uint32_t kUBOffset = 0
>
struct MixEpilogue {
    static constexpr uint32_t VL_b32 = sizeof(vector_float) / sizeof(float);

    // Task partitioning
    static constexpr uint32_t kMStride = kNumAIVCores * BLOCK_M;

    // Resident Mix parameters
    static constexpr ub_ptr<float> ub_mix_scales{1, kDataBlockBytes / sizeof(float), kUBOffset};
    static constexpr ub_ptr<float> ub_mix_bases{1, aligned(SHAPE_N, FRAC_MN), ub_mix_scales.offset(1)};

    // Prenorm and HC square-sum input stages; each stage contains all K splits
    static constexpr uint32_t kNumMixInputStages = 2;
    static constexpr ub_ptr<float> ub_prenorm{kNumKSplits, aligned(BLOCK_M * SHAPE_N, VL_b32), ub_mix_bases.offset(1)};
    static constexpr ub_ptr<float> ub_hc_sqr_sum{kNumKSplits, VL_b32, ub_prenorm.offset(kNumMixInputStages)};

    // Mix coefficient output stages follow the same block-to-stage mapping as inputs
    static constexpr uint32_t kNumMixOutputStages = kNumMixInputStages;
    static constexpr ub_ptr<float> ub_new_prev_mix{BLOCK_M, kNumRoutes, ub_hc_sqr_sum.offset(kNumMixInputStages)};
    static constexpr ub_ptr<float> ub_new_post_mix{BLOCK_M, kNumRoutes, ub_new_prev_mix.offset(kNumMixOutputStages)};
    static constexpr ub_ptr<float> ub_new_comb_res_mix{BLOCK_M, kNumRoutes * kNumRoutes, ub_new_post_mix.offset(kNumMixOutputStages)};

    // VF-local logits need no stage ring; all accesses stay on PIPE_V
    static constexpr ub_ptr<float> ub_comb_logits{BLOCK_M, kNumRoutes * kNumRoutes, ub_new_comb_res_mix.offset(kNumMixOutputStages)};

    // Event IDs are independent of mutex IDs; input and output use different pipe pairs
    static constexpr uint32_t kMixInputFlagOffset = 0;
    static constexpr uint32_t kMixOutputFlagOffset = 0;
    static constexpr uint32_t kMixParamsFlag = kMixInputFlagOffset + kNumMixInputStages;

private:
    // Resource boundaries
    static constexpr uint32_t kUBEndOffset = ub_comb_logits.offset(1);

    // Storage and synchronization constraints
    static_assert(kUBEndOffset <= UBSizeBytes);
    static_assert(kMixParamsFlag < 8 and kMixOutputFlagOffset + kNumMixOutputStages <= 8);

public:
    __aicore__ static constexpr uint32_t get_ub_end_offset() { return kUBEndOffset; }

    // Runtime inputs
    uint32_t shape_m;
    uint32_t aiv_idx;
    gm_ptr<float> gm_prenorm;
    gm_ptr<float> gm_hc_sqr_sum;
    gm_ptr<float> gm_mix_scales;
    gm_ptr<float> gm_mix_bases;
    gm_ptr<float> gm_new_prev_mix;
    gm_ptr<float> gm_new_post_mix;
    gm_ptr<float> gm_new_comb_res_mix;
    float hc_norm_eps;
    float hc_pre_eps;
    float hc_post_scale;
    float sinkhorn_eps;
    uint32_t num_sinkhorn_iters;
    const uint32_t m_begin_idx = aiv_idx * BLOCK_M;
    const uint32_t num_m_blocks_in_core = m_begin_idx < shape_m ? ceil_div(shape_m - m_begin_idx, kMStride) : 0;

    // Work admission and per-core cursors
    uint32_t num_ready_steps = 0;
    uint32_t step_idx = 0;
    uint32_t prefetch_step_idx = 0;

    __aicore__ MixEpilogue(const MixArgs& args, uint32_t shape_m, uint32_t aiv_idx,
                           gm_ptr<float> gm_prenorm, gm_ptr<float> gm_hc_sqr_sum)
        : shape_m(shape_m), aiv_idx(aiv_idx), gm_prenorm(gm_prenorm), gm_hc_sqr_sum(gm_hc_sqr_sum),
          gm_mix_scales(args.scales), gm_mix_bases(args.bases), gm_new_prev_mix(args.new_prev_mix),
          gm_new_post_mix(args.new_post_mix), gm_new_comb_res_mix(args.new_comb_res_mix),
          hc_norm_eps(args.hc_norm_eps), hc_pre_eps(args.hc_pre_eps), hc_post_scale(args.hc_post_scale),
          sinkhorn_eps(args.sinkhorn_eps), num_sinkhorn_iters(args.num_sinkhorn_iters) {}

    __aicore__ inline void load_mix_input_gm2ub(uint32_t m_idx, uint32_t stage_idx) {
        const uint32_t actual_m = min(shape_m - m_idx, BLOCK_M);
        asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(kMixInputFlagOffset + stage_idx));
        const uint64_t workspace_m_idx = get_workspace_m_idx<kNumKSplits, BLOCK_M>(m_idx);
        // Keep the per-split UB stride fixed for partial M blocks
        // One descriptor traverses every GEMM split; the other traverses HC statistics
        asc_copy_gm2ub_align(ub_prenorm[stage_idx].ptr(), gm_prenorm.index(workspace_m_idx, 0).ptr(), kNumKSplits,
                             actual_m * SHAPE_N * sizeof(float), 0, 0, false, asc_load_l2_cache_mode::NOTALLOC_KEEP, get_prenorm_split_stride_bytes<SHAPE_N>(),
                             ub_prenorm.shape_n * sizeof(float));
        asc_copy_gm2ub_align(ub_hc_sqr_sum[stage_idx].ptr(), gm_hc_sqr_sum.index(workspace_m_idx, 0).ptr(), kNumKSplits, actual_m * sizeof(float),
                             0, 0, false, asc_load_l2_cache_mode::NOTALLOC_KEEP, get_sqr_sum_split_stride_bytes(), ub_hc_sqr_sum.shape_n * sizeof(float));
        asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(kMixInputFlagOffset + stage_idx));
    }

    __aicore__ inline void prefetch() {
        if (prefetch_step_idx < num_ready_steps and prefetch_step_idx < step_idx + kNumMixInputStages) {
            const uint32_t m_idx = m_begin_idx + prefetch_step_idx * kMStride;
            const uint32_t stage_idx = prefetch_step_idx % kNumMixInputStages;
            load_mix_input_gm2ub(m_idx, stage_idx);
            ++ prefetch_step_idx;
        }
    }

    __aicore__ inline void initialize() {
        if (num_m_blocks_in_core == 0)
            return;
        // Seed the free slots; drain_flags consumes their final notifications
        set_flags<PIPE_V, PIPE_MTE2, kNumMixInputStages, kMixInputFlagOffset>();
        set_flags<PIPE_MTE3, PIPE_V, kNumMixOutputStages, kMixOutputFlagOffset>();
        asc_copy_gm2ub_align(ub_mix_scales.ptr(), gm_mix_scales.ptr(), 1, 3 * sizeof(float), 0, 0, false,
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, 3 * sizeof(float), 3 * sizeof(float));
        asc_copy_gm2ub_align(ub_mix_bases.ptr(), gm_mix_bases.ptr(), 1, SHAPE_N * sizeof(float), 0, 0, false,
                             asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM, SHAPE_N * sizeof(float), SHAPE_N * sizeof(float));
        asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(kMixParamsFlag));
    }

    __aicore__ inline bool has_ready_steps() const {
        return step_idx < num_ready_steps;
    }

    // Admit global M blocks; the caller guarantees whole chunks divisible by the AIV core count
    __aicore__ inline void prepare(uint32_t num_m_blocks) {
        num_ready_steps = min(num_ready_steps + num_m_blocks / kNumAIVCores, num_m_blocks_in_core);
    }

    __aicore__ inline void step() {
        if (step_idx >= num_ready_steps)
            return;

        const uint32_t stage_idx = step_idx % kNumMixInputStages;
        const uint32_t m_idx = m_begin_idx + step_idx * kMStride;
        const uint32_t actual_m = min(shape_m - m_idx, BLOCK_M);
        auto *ub_prenorm_stage = ub_prenorm[stage_idx].ptr();
        auto *ub_hc_sqr_sum_stage = ub_hc_sqr_sum[stage_idx].ptr();

        // Reduce Prenorm and HC square sums across K splits
        asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(kMixInputFlagOffset + stage_idx));
        vf_reduce_splits<kNumKSplits, ub_prenorm.shape_n>(ub_prenorm_stage);
        vf_reduce_splits<kNumKSplits, ub_hc_sqr_sum.shape_n>(ub_hc_sqr_sum_stage);

        // Compute pre/post coefficients and Sinkhorn combination weights
        asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(kMixOutputFlagOffset + stage_idx));
        step_idx == 0 ? asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(kMixParamsFlag)) : void();
        vf_process_mix_coeff<SHAPE_H, SHAPE_N, BLOCK_M, kNumRoutes>(
            ub_prenorm_stage, ub_hc_sqr_sum_stage, ub_mix_scales.ptr(), ub_mix_bases.ptr(),
            ub_new_prev_mix[stage_idx].ptr(), ub_new_post_mix[stage_idx].ptr(), ub_comb_logits.ptr(), ub_new_comb_res_mix[stage_idx].ptr(),
            hc_norm_eps, hc_pre_eps, hc_post_scale, sinkhorn_eps, num_sinkhorn_iters);
        asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(kMixInputFlagOffset + stage_idx));
        asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(kMixOutputFlagOffset + stage_idx));

        // Store all three coefficient arrays
        asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(kMixOutputFlagOffset + stage_idx));
        asc_copy_ub2gm_align(gm_new_prev_mix.index(m_idx, 0).ptr(), ub_new_prev_mix[stage_idx].ptr(), 1,
                             actual_m * kNumRoutes * sizeof(float), asc_store_l2_cache_mode::NOTALLOC_CLEAN,
                             actual_m * kNumRoutes * sizeof(float), actual_m * kNumRoutes * sizeof(float));
        asc_copy_ub2gm_align(gm_new_post_mix.index(m_idx, 0).ptr(), ub_new_post_mix[stage_idx].ptr(), 1,
                             actual_m * kNumRoutes * sizeof(float), asc_store_l2_cache_mode::NOTALLOC_CLEAN,
                             actual_m * kNumRoutes * sizeof(float), actual_m * kNumRoutes * sizeof(float));
        asc_copy_ub2gm_align(gm_new_comb_res_mix.batch(m_idx).ptr(), ub_new_comb_res_mix[stage_idx].ptr(), 1,
                             actual_m * kNumRoutes * kNumRoutes * sizeof(float), asc_store_l2_cache_mode::NOTALLOC_CLEAN,
                             actual_m * kNumRoutes * kNumRoutes * sizeof(float), actual_m * kNumRoutes * kNumRoutes * sizeof(float));
        asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(kMixOutputFlagOffset + stage_idx));

        ++ step_idx;
    }

    __aicore__ inline void drain() {
        prefetch();
        while (has_ready_steps()) {
            step();
            prefetch();
        }
    }

    __aicore__ inline void drain_flags() {
        if (num_m_blocks_in_core == 0)
            return;
        wait_flags<PIPE_V, PIPE_MTE2, kNumMixInputStages, kMixInputFlagOffset>();
        wait_flags<PIPE_MTE3, PIPE_V, kNumMixOutputStages, kMixOutputFlagOffset>();
    }
};

// Chunk synchronization and insertion order belong to the epilogue
template <
    typename NormQuant, typename Mix, uint32_t BLOCK_H,
    uint32_t kNumEpilogueMBlocksPerChunk, uint32_t kFFTSChunkFlagOffset, uint32_t kNumChunkStages, uint32_t kNormQuantBudgetPerMainStep
>
struct MegaHCEpilogue {
    NormQuant norm_quant;
    Mix mix;
    uint32_t chunk_stage_idx = 0;
    uint32_t epilogue_budget = 0;

    __aicore__ inline void initialize() {
        mix.initialize();
    }

    __aicore__ inline void prepare() {
        asc_sync_inter_wait(PIPE_MTE2, kFFTSChunkFlagOffset + chunk_stage_idx);
        chunk_stage_idx = (chunk_stage_idx + 1) % kNumChunkStages;

        norm_quant.prepare(kNumEpilogueMBlocksPerChunk);
        norm_quant.prefetch_all_stages();

        mix.prepare(kNumEpilogueMBlocksPerChunk);
        mix.prefetch();
    }

    __aicore__ inline void step() {
        mix.step();
        mix.prefetch();

        // Carry unused elements into the next Norm/Quant step
        epilogue_budget += kNormQuantBudgetPerMainStep;
        for (; epilogue_budget >= BLOCK_H; epilogue_budget -= BLOCK_H) {
            norm_quant.step();
            norm_quant.prefetch();
        }
    }

    __aicore__ inline void drain_steps() {
        // Y and statistics are ready before the final Prenorm FIX completes
        asc_sync_inter_wait(PIPE_MTE2, kFFTSChunkFlagOffset + kNumChunkStages);
        norm_quant.prepare(kNumEpilogueMBlocksPerChunk);
        norm_quant.drain();

        // Mix also needs the final Prenorm partials
        asc_sync_inter_wait(PIPE_MTE2, kFFTSChunkFlagOffset + chunk_stage_idx);
        chunk_stage_idx = (chunk_stage_idx + 1) % kNumChunkStages;
        mix.prepare(kNumEpilogueMBlocksPerChunk);
        mix.drain();
    }

    __aicore__ inline void drain_flags() {
        mix.drain_flags();
    }
};

} // namespace deep_gemm::mega_mhc
