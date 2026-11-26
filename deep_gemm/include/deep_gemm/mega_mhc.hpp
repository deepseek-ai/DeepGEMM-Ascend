#pragma once

#include <deep_gemm/layout/mega_mhc.hpp>
#include <deep_gemm/mega_mhc_epilogue.hpp>
#include <deep_gemm/simd/mega_mhc.hpp>

namespace deep_gemm::mega_mhc {

using namespace layout::mega_mhc;

template <
    uint32_t SHAPE_K, uint32_t NORM_BLOCK_H, uint32_t kNumL1CoeffStages,
    bool kIsSFRowMajor = false, bool kStoreBF16 = true, bool kStoreFP8 = true,
    uint32_t kEpilogueBlockM = 16, bool kMixOnTailAIVs = false,
    uint32_t BLOCK_M = layout::mega_mhc::BLOCK_M, bool kSplitNormQuant = false, bool kFullMBlocks = false
>
__global__ __mix__(1, 2) void mega_mhc_kernel(KernelArgs) {
    static_assert(BLOCK_M == 16 or BLOCK_M == 32);
    static_assert(kEpilogueBlockM == 8 or kEpilogueBlockM == 16);
    constexpr uint32_t SUB_BLOCK_M = BLOCK_M / kNumSubAIVs;
    static_assert(kStoreBF16 or kStoreFP8);
    asc_init();
    const auto& args = get_replicated_args<KernelArgs>();
    const auto &gm_residual = args.gm_residual, &gm_layer_input = args.gm_layer_input;
    const auto &gm_fn = args.gm_fn, &gm_shifted_prev_mix = args.gm_shifted_prev_mix,
               &gm_post_mix = args.gm_post_mix, &gm_comb_res_mix = args.gm_comb_res_mix;
    const auto& workspace = args.workspace;
    const auto& norm_args = args.norm_args;
    const auto& mix_args = args.mix_args;
    const uint32_t shape_m = norm_args.shape_m;

    constexpr uint32_t SHAPE_H = SHAPE_K / kNumRoutes;
    static_assert(NORM_BLOCK_H > 0 and SHAPE_H % NORM_BLOCK_H == 0 and NORM_BLOCK_H % BLOCK_H == 0);
    constexpr uint32_t kHiddenPerSplit = SHAPE_H / kNumKSplits;
    static_assert(SHAPE_K % (kNumKSplits * BLOCK_K) == 0);

    // Post MAD geometry
    //                           [ residual (16x128) ]
    // [comb | post] (16x32)  *  [     X    (16x128) ]  =  new_residual (16x128)
    // Residual: 4 tokens x 4 routes; post selects 4 of 16 X tokens
    constexpr uint32_t kPostFracK = get_frac_k<bfloat16_t>();
    constexpr uint32_t kNumTokensPerPostMad = FRAC_MN / kNumRoutes;
    constexpr uint32_t kNumPostMadsPerBlock = BLOCK_M / kNumTokensPerPostMad;
    constexpr uint32_t kPostMadM = kNumTokensPerPostMad * kNumRoutes;
    constexpr uint32_t kPostMadN = BLOCK_H;
    constexpr uint32_t kPostMadResidualK = aligned(kNumTokensPerPostMad * kNumRoutes, kPostFracK);
    constexpr uint32_t kPostMadXK = aligned(SUB_BLOCK_M, kPostFracK);
    constexpr uint32_t kPostMadK = kPostMadResidualK + kPostMadXK;

    // Prenorm MAD geometry
    constexpr uint32_t kPrenormMadM = BLOCK_M;
    constexpr uint32_t kPrenormMadN = kNumHCOutputs;
    constexpr uint32_t kPrenormMadK = MAD_K;
    constexpr uint32_t kNumPrenormMadsPerBlock = BLOCK_K / kPrenormMadK;

    // Post L0C -> UB -> VF; each AIV casts two FIXPIPE outputs to FP32 NZ at a time
    constexpr uint32_t kNumPostFixesPerVF = 2;
    constexpr uint32_t kPostVFBlockM = kNumPostFixesPerVF * kNumTokensPerPostMad;
    constexpr uint32_t kNumPostVFBlocks = SUB_BLOCK_M / kPostVFBlockM;
    static_assert(SUB_BLOCK_M % kPostVFBlockM == 0);

    // split-k cross-core storage stages
    constexpr uint32_t kNumMBlocksPerChunk = kNumTokensPerChunk / BLOCK_M;
    constexpr uint32_t kNumPrenormL0CStages = kNumMBlocksPerChunk / kNumAICoresPerSplit;

    // post buffers and stages
    constexpr uint32_t kNumL1ResidualStages = 5;
    static_assert(kNumL1CoeffStages > 0 and kNumL1CoeffStages <= kNumPrenormL0CStages);
    constexpr uint32_t kNumL1XStages = 4;
    constexpr l1_ptr<bfloat16_t, Major::K>  l1_coeff{kNumPostMadsPerBlock * kPostMadM, kNumFP32toBF16Splits * kPostMadK};
    constexpr l1_ptr<bfloat16_t, Major::MN> l1_residual{BLOCK_H, BLOCK_M * kNumRoutes, l1_coeff.offset(kNumL1CoeffStages)};
    constexpr l1_ptr<bfloat16_t, Major::MN> l1_x{BLOCK_H, kNumSubAIVs * kPostMadXK, l1_residual.offset(kNumL1ResidualStages)};
    constexpr ub_ptr<bfloat16_t> ub_new_residual{SUB_BLOCK_M, BLOCK_K};

    // prenorm buffers and stages
    constexpr uint32_t kNumL1NewResidualStages = 1;
    constexpr uint32_t kNumUBNewResidualStages = 2;
    constexpr uint32_t kNumMaxL1FNStages = BLOCK_M == 32 and kMixOnTailAIVs ? 3 : (BLOCK_M == 16 ? 2 : 1);
    // Separate Post reads and NewResidual writes across L1 banks when Post fits in the first bank
    constexpr l1_ptr<float> l1_new_residual{BLOCK_M, BLOCK_K, (l1_x.offset(kNumL1XStages) < L1SizeBytes / 2 ? L1SizeBytes / 2 : l1_x.offset(kNumL1XStages))};
    constexpr l1_ptr<float> l1_fn{BLOCK_N, BLOCK_K, l1_new_residual.offset(kNumL1NewResidualStages)};

    static_assert(l1_fn.offset(1) <= L1SizeBytes);
    // Cap FN prefetch depth by the L1 space left after Post and NewResidual
    constexpr uint32_t kNumL1FNStagesByCapacity = (L1SizeBytes - l1_fn.addr) / l1_fn.size_per_stage();
    constexpr uint32_t kNumL1FNStages = kNumMaxL1FNStages < kNumL1FNStagesByCapacity ? kNumMaxL1FNStages : kNumL1FNStagesByCapacity;
    static_assert(l1_fn.offset(kNumL1FNStages) <= L1SizeBytes);

    // Shared buffer notifications
    constexpr uint32_t kNumChunkStages = 2;
    constexpr uint32_t kIntraL1NewResidualFlagOffset = 0;
    constexpr uint32_t kIntraL1XFlagOffset = kIntraL1NewResidualFlagOffset + kNumL1NewResidualStages;
    constexpr uint32_t kIntraUBNewResidualFlagOffset = kIntraL1XFlagOffset + kNumL1XStages;
    constexpr uint32_t kIntraPrenormFlagOffset = kIntraUBNewResidualFlagOffset + kNumPostVFBlocks * kNumUBNewResidualStages;
    static_assert(kIntraPrenormFlagOffset + kNumChunkStages <= 16);

    // FFTS notifications
    constexpr uint32_t kFFTSChunkFlagOffset = 0;

    // Workspace for Prenorm partials and HC/RMS square sums
    const MegaHCWorkspaceLayout<kNumHCOutputs, kNumKSplits> workspace_layout(shape_m);
    const gm_ptr<float> gm_prenorm(kNumHCOutputs, reinterpret_cast<uintptr_t>(workspace_layout.prenorm.get_ptr(workspace)));
    const gm_ptr<float> gm_sqr_sum(1, reinterpret_cast<uintptr_t>(workspace_layout.hc_sqr_sum.get_ptr(workspace)));
    const gm_ptr<float> gm_rms_sqr_sum(1, reinterpret_cast<uintptr_t>(workspace_layout.rms_sqr_sum.get_ptr(workspace)));

    // Keep the same token capacity per chunk for either M tile
    const uint32_t k_split_idx = asc_get_block_idx() / kNumAICoresPerSplit;
    const uint32_t core_idx_in_k_split = asc_get_block_idx() % kNumAICoresPerSplit;
    const uint32_t h_begin_idx = k_split_idx * kHiddenPerSplit;
    const uint32_t h_end_idx = h_begin_idx + kHiddenPerSplit;
    const uint32_t num_m_blocks = ceil_div(shape_m, BLOCK_M);

    // AIC: seed free events, produce Post results and accumulate resident Prenorm blocks
    if DG_IS_AIC {
        // Stage counts and storage shapes
        constexpr uint32_t kNumPostL0AStages = 4;
        constexpr uint32_t kNumPostL0BStages = 4;
        constexpr uint32_t kNumPostL0CStages = 2;
        constexpr uint32_t kNumPrenormL0AStages = 4;
        constexpr uint32_t kNumFNMadSlicesPerPostL0BStage = ceil_div(kPostMadN * kPostMadK * sizeof(bfloat16_t), BLOCK_N * kPrenormMadK * sizeof(float));

        // Post aliases the upper FN L0B slices; L0A and L0C regions are disjoint
        constexpr l0a_ptr<float> l0a_new_residual{kPrenormMadM, kPrenormMadK};
        constexpr l0a_ptr<bfloat16_t> l0a_coeff{kPostMadM, kNumFP32toBF16Splits * kPostMadK, l0a_new_residual.offset(kNumPrenormL0AStages)};
        constexpr l0b_ptr<float> l0b_fn{BLOCK_N, kPrenormMadK};
        constexpr l0b_ptr<bfloat16_t> l0b_residual_x{kPostMadN, kPostMadK, l0b_fn.offset(kNumPrenormMadsPerBlock - kNumPostL0BStages * kNumFNMadSlicesPerPostL0BStage)};
        constexpr l0c_ptr<float> l0c_prenorm{kPrenormMadM, BLOCK_N};
        constexpr l0c_ptr<float> l0c_new_residual{kPostMadM, kPostMadN, l0c_prenorm.offset(kNumPrenormL0CStages)};

        // Local buffer IDs
        constexpr uint32_t kL1FNBufOffset = 0;
        constexpr uint32_t kL1ResidualBufOffset = kL1FNBufOffset + kNumL1FNStages;
        constexpr uint32_t kL0APrenormBufOffset = kL1ResidualBufOffset + kNumL1ResidualStages;
        constexpr uint32_t kL0BPostBufOffset = kL0APrenormBufOffset + kNumPrenormL0AStages;
        // Post L0A and L0B advance together and share each buffer handoff
        static_assert(kNumPostL0AStages == kNumPostL0BStages);
        constexpr uint32_t kL0APostCoeffBufOffset = kL0BPostBufOffset;

        // Resource boundaries
        constexpr uint32_t kBufEndOffset = kL0BPostBufOffset + kNumPostL0BStages;

        // Storage and synchronization constraints
        static_assert(kBufEndOffset <= 32);
        static_assert(l0a_coeff.offset(kNumPostL0AStages) <= L0ASizeBytes and l0b_residual_x.offset(kNumPostL0BStages) <= L0BSizeBytes);
        static_assert(l0c_new_residual.offset(kNumPostL0CStages) <= L0CSizeBytes);

        // AIC consumer stages: L1, Post L0, Prenorm L0 and the AIV output handoff
        uint32_t l1_fn_stage_idx = 0;
        uint32_t l1_residual_stage_idx = 0;
        uint32_t l1_x_stage_idx = 0;
        uint32_t l1_new_residual_stage_idx = 0;
        uint32_t post_l0a_stage_idx = 0;
        uint32_t post_l0b_stage_idx = 0;
        uint32_t post_l0c_stage_idx = 0;
        uint32_t prenorm_l0a_stage_idx = 0;
        uint32_t ub_new_residual_stage_idx = 0;

        // AIC prefetch stages and M/H positions
        uint32_t l1_fn_prefetch_stage_idx = 0;
        uint32_t l1_residual_prefetch_stage_idx = 0;
        uint32_t prefetch_fn_m_block_idx = core_idx_in_k_split;
        uint32_t prefetch_fn_h_idx = h_begin_idx;
        uint32_t prefetch_residual_m_block_idx = core_idx_in_k_split;
        uint32_t prefetch_residual_h_idx = h_begin_idx;

        asc_set_mmad_direction_n();
        asc_enable_hf32();
        asc_enable_hf32_trans(0);

        fill_l1(l1_residual.ptr<int16_t>(), int16_t(0), 1, l1_residual.size_per_stage() * kNumL1ResidualStages / kDataBlockBytes, 0);
        fill_l1(l1_x.ptr<int16_t>(), int16_t(0), 1, l1_x.size_per_stage() * kNumL1XStages / kDataBlockBytes, 0);

        // Publish initial free slots after MTE2 finishes clearing L1
        set_intra_blocks<PIPE_MTE2, kNumL1NewResidualStages, kIntraL1NewResidualFlagOffset>(0);
        set_intra_blocks<PIPE_MTE2, kNumL1NewResidualStages, kIntraL1NewResidualFlagOffset>(1);
        set_intra_blocks<PIPE_MTE2, kNumL1XStages, kIntraL1XFlagOffset>(0);
        set_intra_blocks<PIPE_MTE2, kNumL1XStages, kIntraL1XFlagOffset>(1);
        asc_set_atomic_none();

        // MTE1 releases each FN stage before MTE2 overwrites it
        const auto load_fn_gm2l1 = [&](uint32_t h_idx, uint32_t stage_idx) __aicore__ {
            constexpr uint32_t kL1BRouteStride = BLOCK_N * BLOCK_H;
            asc_lock(PIPE_MTE2, kL1FNBufOffset + stage_idx);
            set_mte2_nz(kNumRoutes, 1, BLOCK_N, kL1BRouteStride * sizeof(float) / kDataBlockBytes);
            asc_copy_gm2l1_nd2nz(l1_fn[stage_idx].ptr(), gm_fn.index(0, h_idx).ptr(), gm_fn.stride_outer * sizeof(float),
                                 asc_load_l2_cache_mode::NORMAL_LAST_VICTIM, kNumHCOutputs, BLOCK_H, SHAPE_H * sizeof(float), false);
            asc_unlock(PIPE_MTE2, kL1FNBufOffset + stage_idx);
        };

        // Load this chunk's FN, then advance the producer to the next H or chunk
        const auto prefetch_fn_gm2l1 = [&](uint32_t& m_block_idx, uint32_t& h_idx) __aicore__ {
            if (m_block_idx >= num_m_blocks)
                return;
            load_fn_gm2l1(h_idx, l1_fn_prefetch_stage_idx);
            h_idx += BLOCK_H;
            if (h_idx == h_end_idx) {
                h_idx = h_begin_idx;
                m_block_idx += kNumMBlocksPerChunk;
            }
            l1_fn_prefetch_stage_idx = (l1_fn_prefetch_stage_idx + 1) % kNumL1FNStages;
        };

        // Refill a residual L1 stage only after its Post MTE1 reads complete
        const auto load_residual_gm2l1 = [&](uint32_t m_block_idx, uint32_t h_idx, uint32_t stage_idx) __aicore__ {
            const uint32_t m_idx = m_block_idx * BLOCK_M;
            const uint32_t actual_m = kFullMBlocks ? BLOCK_M : min(shape_m - m_idx, BLOCK_M);
            asc_lock(PIPE_MTE2, kL1ResidualBufOffset + stage_idx);
            if (actual_m < BLOCK_M) {
                fill_l1(l1_residual[stage_idx].template ptr<int16_t>(), int16_t(0),
                        1, l1_residual.size_per_stage() / kDataBlockBytes, 0);
                asc_sync_pipe(PIPE_MTE2);
            }
            set_mte2_nz(1, 1, BLOCK_M * kNumRoutes, 0);
            asc_copy_gm2l1_nd2nz(l1_residual[stage_idx].ptr(), gm_residual.index(m_idx, h_idx).ptr(), SHAPE_H * sizeof(bfloat16_t),
                                 asc_load_l2_cache_mode::NOTALLOC_KEEP, actual_m * kNumRoutes, BLOCK_H, 0, false);
            asc_unlock(PIPE_MTE2, kL1ResidualBufOffset + stage_idx);
        };

        // Load the producer position, then advance it across M, H and chunk boundaries
        const auto prefetch_residual_gm2l1 = [&](uint32_t& m_block_idx, uint32_t& h_idx) __aicore__ {
            if (m_block_idx >= num_m_blocks)
                return;
            load_residual_gm2l1(m_block_idx, h_idx, l1_residual_prefetch_stage_idx);
            const uint32_t m_block_base_idx = m_block_idx / kNumMBlocksPerChunk * kNumMBlocksPerChunk;
            m_block_idx += kNumAICoresPerSplit;
            if (m_block_idx >= min(num_m_blocks, m_block_base_idx + kNumMBlocksPerChunk)) {
                m_block_idx = m_block_base_idx + core_idx_in_k_split;
                h_idx += BLOCK_H;
                if (h_idx == h_end_idx) {
                    h_idx = h_begin_idx;
                    m_block_idx += kNumMBlocksPerChunk;
                }
            }
            l1_residual_prefetch_stage_idx = (l1_residual_prefetch_stage_idx + 1) % kNumL1ResidualStages;
        };

        const auto run_post_comb_mix_gemm = [&](uint32_t m_block_base_idx, uint32_t m_block_idx) __aicore__ {
            const uint32_t m_idx = m_block_idx * BLOCK_M;
            const uint32_t actual_m = kFullMBlocks ? BLOCK_M : min(shape_m - m_idx, BLOCK_M);
            const bool is_aiv_1_alive = actual_m > SUB_BLOCK_M;
            const uint32_t m_block_idx_in_core = (m_block_idx - m_block_base_idx - core_idx_in_k_split) / kNumAICoresPerSplit;

            wait_intra_block(PIPE_MTE1, 0, kIntraL1XFlagOffset + l1_x_stage_idx);
            is_aiv_1_alive ? wait_intra_block(PIPE_MTE1, 1, kIntraL1XFlagOffset + l1_x_stage_idx) : void();
            wait_intra_block(PIPE_FIX, 0, kIntraUBNewResidualFlagOffset + ub_new_residual_stage_idx);
            is_aiv_1_alive ? wait_intra_block(PIPE_FIX, 1, kIntraUBNewResidualFlagOffset + ub_new_residual_stage_idx) : void();

            asc_lock(PIPE_MTE1, kL1ResidualBufOffset + l1_residual_stage_idx);

            const l1_ptr<bfloat16_t, Major::MN> l1_x_sub_blocks(BLOCK_H, kPostMadXK, l1_x[l1_x_stage_idx].addr);
            const l1_ptr<bfloat16_t, Major::K> l1_coeff_mads(kPostMadM, kNumFP32toBF16Splits * kPostMadK, l1_coeff[m_block_idx_in_core].addr);

            // Finish one AIV's Post stores before switching destinations
            constexpr uint32_t kNumPostMadsPerAIV = kNumPostMadsPerBlock / kNumSubAIVs;
            for (uint32_t post_mad_idx = 0; post_mad_idx < kNumPostMadsPerBlock; ++ post_mad_idx) {
                const uint32_t sub_idx = post_mad_idx / kNumPostMadsPerAIV;
                const uint32_t sub_post_mad_idx = post_mad_idx % kNumPostMadsPerAIV;
                const uint32_t sub_m_idx = m_idx + sub_idx * SUB_BLOCK_M + sub_post_mad_idx * kNumTokensPerPostMad;
                if (kFullMBlocks or sub_m_idx < shape_m) {
                    const auto l0c_new_residual_stage = l0c_new_residual[post_l0c_stage_idx];
                    const auto l0b_residual_x_stage = l0b_residual_x[post_l0b_stage_idx];
                    const auto l0a_coeff_stage = l0a_coeff[post_l0a_stage_idx];

                    const l0b_ptr<bfloat16_t> l0b_residual(kPostMadN, kPostMadResidualK, l0b_residual_x_stage.addr);
                    const l0b_ptr<bfloat16_t> l0b_x(kPostMadN, kPostMadXK, l0b_residual_x_stage.addr + kPostMadN * kPostMadResidualK * sizeof(bfloat16_t));
                    const l0a_ptr<bfloat16_t> l0a_coeff_bf16_split0(kPostMadM, kPostMadK, l0a_coeff_stage.addr);
                    const l0a_ptr<bfloat16_t> l0a_coeff_bf16_split1(kPostMadM, kPostMadK, l0a_coeff_bf16_split0.addr + kPostMadM * kPostMadK * sizeof(bfloat16_t));
                    const l0a_ptr<bfloat16_t> l0a_coeff_bf16_split2(kPostMadM, kPostMadK, l0a_coeff_bf16_split1.addr + kPostMadM * kPostMadK * sizeof(bfloat16_t));

                    asc_lock(PIPE_MTE1, kL0APostCoeffBufOffset + post_l0a_stage_idx);
                    copy_l1_to_l0a(l0a_coeff_stage, l1_coeff_mads[sub_post_mad_idx * kNumSubAIVs + sub_idx], 0, 0);
                    copy_l1_to_l0b(l0b_residual, l1_residual[l1_residual_stage_idx], 0,
                                   sub_idx * SUB_BLOCK_M * kNumRoutes + sub_post_mad_idx * kPostMadResidualK);
                    // TODO: Reuse X across Post groups when the L0B stage maps to the same AIV, including tails
                    copy_l1_to_l0b(l0b_x, l1_x_sub_blocks[sub_idx], 0, 0);
                    asc_unlock(PIPE_MTE1, kL0BPostBufOffset + post_l0b_stage_idx);
                    asc_lock(PIPE_M, kL0BPostBufOffset + post_l0b_stage_idx);
                    mad(l0c_new_residual_stage, l0a_coeff_bf16_split0, l0b_residual_x_stage, kPostMadM, kPostMadN, kPostMadK, asc_unit_flag_mode::ENABLE_KEEP, true);
                    mad(l0c_new_residual_stage, l0a_coeff_bf16_split1, l0b_residual_x_stage, kPostMadM, kPostMadN, kPostMadK, asc_unit_flag_mode::ENABLE_KEEP, false);
                    mad(l0c_new_residual_stage, l0a_coeff_bf16_split2, l0b_residual_x_stage, kPostMadM, kPostMadN, kPostMadK, asc_unit_flag_mode::ENABLE_UPDATE, false);
                    asc_unlock(PIPE_M, kL0BPostBufOffset + post_l0b_stage_idx);
                    const ub_ptr<bfloat16_t> ub_new_residual_post_mad(kPostMadM, kPostMadN, ub_new_residual[ub_new_residual_stage_idx].addr + sub_post_mad_idx * kPostMadM * kPostMadN * sizeof(bfloat16_t));
                    copy_l0c_to_ub(ub_new_residual_post_mad, l0c_new_residual_stage, kPostMadM, kPostMadN, asc_dual_dst_mode::DUAL_DST_DISABLE, sub_idx, asc_quant_mode::F322BF16, false, false, asc_unit_flag_mode::ENABLE_UPDATE);
                    post_l0a_stage_idx = (post_l0a_stage_idx + 1) % kNumPostL0AStages;
                    post_l0b_stage_idx = (post_l0b_stage_idx + 1) % kNumPostL0BStages;
                    post_l0c_stage_idx = (post_l0c_stage_idx + 1) % kNumPostL0CStages;
                }
                // Each live AIV consumes both flags, including for its padded tail
                if ((sub_post_mad_idx + 1) % kNumPostFixesPerVF == 0 and (sub_idx == 0 or is_aiv_1_alive)) {
                    const uint32_t flag_idx = kIntraUBNewResidualFlagOffset +
                        sub_post_mad_idx / kNumPostFixesPerVF * kNumUBNewResidualStages + ub_new_residual_stage_idx;
                    set_intra_block(PIPE_FIX, sub_idx, flag_idx);
                }
            }
            asc_unlock(PIPE_MTE1, kL1ResidualBufOffset + l1_residual_stage_idx);
            l1_residual_stage_idx = (l1_residual_stage_idx + 1) % kNumL1ResidualStages;

            set_intra_block(PIPE_MTE1, 0, kIntraL1XFlagOffset + l1_x_stage_idx);
            is_aiv_1_alive ? set_intra_block(PIPE_MTE1, 1, kIntraL1XFlagOffset + l1_x_stage_idx) : void();

            l1_x_stage_idx = (l1_x_stage_idx + 1) % kNumL1XStages;
            ub_new_residual_stage_idx = (ub_new_residual_stage_idx + 1) % kNumUBNewResidualStages;
        };

        // Consume AIV FP32 NZ input and retain this M32 accumulator across H blocks
        const auto run_prenorm_gemm = [&](uint32_t m_block_base_idx, uint32_t m_block_idx,
                                          uint32_t h_idx, bool restore_fn, bool release_post_l0b, bool advance_post_across_h) __aicore__ {
            // L0B (8 KiB/slice, total 64KiB): FN MAD order 0 -> 7
            // FN:    [ 0  1  2  3 | 4  5  6  7 ]
            // Usage: [resident FN | Post  0..3 ]
            // Restore FN 4..7 from L1 after Post; the final M block reuses them
            constexpr uint32_t kFNRestoreK = kNumPostL0BStages * kNumFNMadSlicesPerPostL0BStage * kPrenormMadK;
            constexpr uint32_t kFNRestoreKBegin = BLOCK_K - kFNRestoreK;

            const uint32_t m_block_end_idx = min(num_m_blocks, m_block_base_idx + kNumMBlocksPerChunk);
            const uint32_t m_idx = m_block_idx * BLOCK_M;
            const uint32_t prenorm_l0c_stage_idx = (m_block_idx - m_block_base_idx) / kNumAICoresPerSplit;
            const uint32_t actual_m = kFullMBlocks ? kPrenormMadM : min(shape_m - m_idx, kPrenormMadM);
            const bool is_aiv_1_alive = actual_m > SUB_BLOCK_M;
            const auto l0c_prenorm_stage = l0c_prenorm[prenorm_l0c_stage_idx].as_mad_aligned(actual_m, kPrenormMadN);

            if (m_block_idx == m_block_base_idx + core_idx_in_k_split) {
                asc_lock(PIPE_MTE1, kL1FNBufOffset + l1_fn_stage_idx);
                copy_l1_to_l0b(l0b_fn.as_mad_aligned(BLOCK_N, kFNRestoreKBegin), l1_fn[l1_fn_stage_idx], 0, 0);
            }
            wait_intra_block(PIPE_MTE1, 0, kIntraL1NewResidualFlagOffset + l1_new_residual_stage_idx);
            is_aiv_1_alive ? wait_intra_block(PIPE_MTE1, 1, kIntraL1NewResidualFlagOffset + l1_new_residual_stage_idx) : void();
            for (uint32_t k_idx = 0; k_idx < BLOCK_K; k_idx += kPrenormMadK) {
                // Post aliases the last four FN slices; consume the first four before restoring them
                if (restore_fn and k_idx == kFNRestoreKBegin) {
                    for (uint32_t stage_idx = 0; stage_idx < kNumPostL0BStages; ++ stage_idx)
                        asc_lock(PIPE_MTE1, kL0BPostBufOffset + stage_idx);
                    copy_l1_to_l0b(l0b_fn[kFNRestoreKBegin / kPrenormMadK].as_mad_aligned(BLOCK_N, kFNRestoreK),
                                   l1_fn[l1_fn_stage_idx], 0, kFNRestoreKBegin);
                    for (uint32_t stage_idx = 0; stage_idx < kNumPostL0BStages; ++ stage_idx)
                        asc_unlock(PIPE_MTE1, kL0BPostBufOffset + stage_idx);
                    // Keep FN in L1 until the last M block that restores its upper slices
                    if (m_block_idx + (advance_post_across_h ? 1 : 2) * kNumAICoresPerSplit >= m_block_end_idx) {
                        asc_unlock(PIPE_MTE1, kL1FNBufOffset + l1_fn_stage_idx);
                        l1_fn_stage_idx = (l1_fn_stage_idx + 1) % kNumL1FNStages;
                        prefetch_fn_gm2l1(prefetch_fn_m_block_idx, prefetch_fn_h_idx);
                    }
                }
                if (restore_fn and k_idx >= kFNRestoreKBegin and (k_idx - kFNRestoreKBegin) % (kNumFNMadSlicesPerPostL0BStage * kPrenormMadK) == 0)
                    asc_lock(PIPE_M, kL0BPostBufOffset + (k_idx - kFNRestoreKBegin) / (kNumFNMadSlicesPerPostL0BStage * kPrenormMadK));
                const auto l0a_new_residual_stage = l0a_new_residual[prenorm_l0a_stage_idx].as_mad_aligned(actual_m, kPrenormMadK);
                asc_lock(PIPE_MTE1, kL0APrenormBufOffset + prenorm_l0a_stage_idx);

                copy_l1_to_l0a(l0a_new_residual_stage, l1_new_residual[l1_new_residual_stage_idx], 0, k_idx);
                asc_unlock(PIPE_MTE1, kL0APrenormBufOffset + prenorm_l0a_stage_idx);
                asc_lock(PIPE_M, kL0APrenormBufOffset + prenorm_l0a_stage_idx);
                mad(l0c_prenorm_stage, l0a_new_residual_stage, l0b_fn[k_idx / kPrenormMadK], actual_m, kPrenormMadN, kPrenormMadK,
                    h_idx + BLOCK_H == h_end_idx and k_idx + kPrenormMadK == BLOCK_K ? asc_unit_flag_mode::ENABLE_UPDATE : asc_unit_flag_mode::ENABLE_KEEP,
                    h_idx == h_begin_idx and k_idx == 0);
                asc_unlock(PIPE_M, kL0APrenormBufOffset + prenorm_l0a_stage_idx);
                // Release a Post L0B stage after Prenorm consumes all of its FN slices
                if (release_post_l0b and k_idx >= kFNRestoreKBegin and (k_idx - kFNRestoreKBegin + kPrenormMadK) % (kNumFNMadSlicesPerPostL0BStage * kPrenormMadK) == 0)
                    asc_unlock(PIPE_M, kL0BPostBufOffset + (k_idx - kFNRestoreKBegin) / (kNumFNMadSlicesPerPostL0BStage * kPrenormMadK));
                prenorm_l0a_stage_idx = (prenorm_l0a_stage_idx + 1) % kNumPrenormL0AStages;
            }
            set_intra_block(PIPE_MTE1, 0, kIntraL1NewResidualFlagOffset + l1_new_residual_stage_idx);
            is_aiv_1_alive ? set_intra_block(PIPE_MTE1, 1, kIntraL1NewResidualFlagOffset + l1_new_residual_stage_idx) : void();
            l1_new_residual_stage_idx = (l1_new_residual_stage_idx + 1) % kNumL1NewResidualStages;
        };

        static_assert(kNumL1FNStages <= kNumL1ResidualStages);
        // Seed the first Post inputs before filling the remaining FN slots
        for (uint32_t stage_idx = 0; stage_idx < kNumL1ResidualStages; ++ stage_idx) {
            if (stage_idx < kNumL1FNStages)
                prefetch_fn_gm2l1(prefetch_fn_m_block_idx, prefetch_fn_h_idx);
            prefetch_residual_gm2l1(prefetch_residual_m_block_idx, prefetch_residual_h_idx);
        }

        // Walk resident M chunks; visit every H128 before storing the Prenorm partials
        for (uint32_t m_block_base_idx = 0; m_block_base_idx < num_m_blocks; m_block_base_idx += kNumMBlocksPerChunk) {
            const uint32_t m_block_end_idx = min(num_m_blocks, m_block_base_idx + kNumMBlocksPerChunk);
            const uint32_t m_block_begin_idx = m_block_base_idx + core_idx_in_k_split;
            if (m_block_begin_idx < m_block_end_idx) {
                const uint32_t m_block_last_idx = m_block_begin_idx +
                    (ceil_div(m_block_end_idx - m_block_begin_idx, kNumAICoresPerSplit) - 1) * kNumAICoresPerSplit;
                const bool advance_post_across_h = BLOCK_M == 16 or m_block_last_idx == m_block_begin_idx;
                for (uint32_t h_idx = h_begin_idx; h_idx < h_end_idx; h_idx += BLOCK_H) {
                    for (uint32_t m_block_idx = m_block_begin_idx; m_block_idx < m_block_end_idx; m_block_idx += kNumAICoresPerSplit) {
                        run_post_comb_mix_gemm(m_block_base_idx, m_block_idx);
                        prefetch_residual_gm2l1(prefetch_residual_m_block_idx, prefetch_residual_h_idx);
                        // Keep Post ahead of Prenorm, crossing H when the final FN slice is restored.
                        if (m_block_idx != m_block_begin_idx)
                            run_prenorm_gemm(m_block_base_idx, m_block_idx - kNumAICoresPerSplit,
                                             h_idx, /*restore_fn=*/true,
                                             /*release_post_l0b=*/advance_post_across_h or m_block_idx + kNumAICoresPerSplit < m_block_end_idx, advance_post_across_h);
                        else if (advance_post_across_h and h_idx > h_begin_idx)
                            run_prenorm_gemm(m_block_base_idx, m_block_last_idx,
                                             h_idx - BLOCK_H, /*restore_fn=*/true, /*release_post_l0b=*/true, advance_post_across_h);
                    }
                    if (not advance_post_across_h)
                        run_prenorm_gemm(m_block_base_idx, m_block_last_idx, h_idx,
                                         /*restore_fn=*/false, /*release_post_l0b=*/true, advance_post_across_h);
                }
                if (advance_post_across_h)
                    run_prenorm_gemm(m_block_base_idx, m_block_last_idx, h_end_idx - BLOCK_H,
                                     /*restore_fn=*/true, /*release_post_l0b=*/true, advance_post_across_h);
                // Each split publishes its own partial; consumers reduce in split order
                for (uint32_t m_block_idx = m_block_begin_idx; m_block_idx < m_block_end_idx; m_block_idx += kNumAICoresPerSplit) {
                    const uint32_t m_idx = m_block_idx * BLOCK_M;
                    const uint32_t prenorm_l0c_stage_idx = (m_block_idx - m_block_base_idx) / kNumAICoresPerSplit;
                    const uint32_t actual_m = kFullMBlocks ? BLOCK_M : min(shape_m - m_idx, BLOCK_M);
                    const auto l0c_prenorm_stage = l0c_prenorm[prenorm_l0c_stage_idx].as_mad_aligned(actual_m, kNumHCOutputs);
                    const uint64_t workspace_m_idx = get_workspace_m_idx<kNumKSplits, BLOCK_M>(m_idx, k_split_idx);
                    copy_l0c_to_gm(gm_prenorm.index(workspace_m_idx, 0).ptr(), l0c_prenorm_stage, actual_m,
                                   kNumHCOutputs, kNumHCOutputs, asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM);
                }
            }
            // Two notification slots: reclaim only when this slot is reused
            const uint32_t flag = kIntraPrenormFlagOffset + m_block_base_idx / kNumMBlocksPerChunk % kNumChunkStages;
            if (m_block_base_idx >= kNumChunkStages * kNumMBlocksPerChunk) {
                wait_intra_block(PIPE_FIX, 0, flag);
                wait_intra_block(PIPE_FIX, 1, flag);
            }
            set_intra_block(PIPE_FIX, 0, flag);
            set_intra_block(PIPE_FIX, 1, flag);
        }
        const uint32_t num_chunks = ceil_div(num_m_blocks, kNumMBlocksPerChunk);
        for (uint32_t stage_idx = 0; stage_idx < min(num_chunks, kNumChunkStages); ++ stage_idx) {
            const uint32_t flag = kIntraPrenormFlagOffset + stage_idx;
            wait_intra_block(PIPE_FIX, 0, flag);
            wait_intra_block(PIPE_FIX, 1, flag);
        }
    }

    // AIV: feed Post/Prenorm, compute PreApply/statistics and overlap older-chunk Norm/Quant
    if DG_IS_AIV {
        // Post buffers and stages
        constexpr uint32_t kNumUBXStages = 4;
        constexpr uint32_t kNumUBPostCoeffStages = 2;
        constexpr uint32_t kNumUBXNZStages = kNumUBPostCoeffStages;
        constexpr ub_ptr<bfloat16_t> ub_x{SUB_BLOCK_M, BLOCK_H, ub_new_residual.offset(kNumUBNewResidualStages)};
        constexpr ub_ptr<float> ub_post_mix{SUB_BLOCK_M, kNumRoutes, ub_x.offset(kNumUBXStages)};
        constexpr ub_ptr<float> ub_comb_res_mix{SUB_BLOCK_M, kNumRoutes * kNumRoutes, ub_post_mix.offset(kNumUBXStages)};
        constexpr ub_ptr<float> ub_shifted_prev_mix{SUB_BLOCK_M, kNumRoutes, ub_comb_res_mix.offset(kNumUBXStages)};
        constexpr ub_ptr<bfloat16_t> ub_post_coeff_matrix{kNumPostMadsPerBlock / kNumSubAIVs * kPostMadM, kNumFP32toBF16Splits * kPostMadK, ub_shifted_prev_mix.offset(kNumUBXStages)};
        constexpr ub_ptr<bfloat16_t> ub_x_nz{BLOCK_H / kPostFracK, (SUB_BLOCK_M + 1) * kPostFracK, ub_post_coeff_matrix.offset(kNumUBPostCoeffStages)};

        // PreApply buffers and stages
        constexpr uint32_t kNumUBYBF16Stages = 2;
        constexpr ub_ptr<bfloat16_t> ub_y_bf16{SUB_BLOCK_M, BLOCK_H, ub_x_nz.offset(kNumUBXNZStages)};

        // Prenorm buffers
        constexpr uint32_t FRAC_K = get_frac_k<float>();
        constexpr uint32_t VL_b32 = sizeof(vector_float) / sizeof(float);
        constexpr uint32_t kNumUBNewResidualNZStages = 2;
        constexpr ub_ptr<float> ub_new_residual_nz{BLOCK_K / FRAC_K, (SUB_BLOCK_M + 1) * FRAC_K, ub_y_bf16.offset(kNumUBYBF16Stages)};

        // Chunk square sums for Mix and Norm; one ID protects all resident M blocks
        constexpr uint32_t kNumUBSqrSumStages = 1;
        constexpr ub_ptr<float> ub_hc_sqr_sum{1, VL_b32, ub_new_residual_nz.offset(kNumUBNewResidualNZStages)};
        constexpr ub_ptr<float> ub_rms_sqr_sum{1, VL_b32, ub_hc_sqr_sum.offset(kNumPrenormL0CStages)};

        // Local buffer IDs
        constexpr uint32_t kUBXBufOffset = 0;
        constexpr uint32_t kUBPostCoeffBufOffset = kUBXBufOffset + kNumUBXStages;
        constexpr uint32_t kUBXNZBufOffset = kUBPostCoeffBufOffset + kNumUBPostCoeffStages;
        constexpr uint32_t kUBNewResidualBufOffset = kUBXNZBufOffset + kNumUBXNZStages;
        constexpr uint32_t kUBNewResidualNZBufOffset = kUBNewResidualBufOffset + kNumUBNewResidualStages;
        constexpr uint32_t kUBYBF16BufOffset = kUBNewResidualNZBufOffset + kNumUBNewResidualNZStages;
        constexpr uint32_t kUBSqrSumBufOffset = kUBYBF16BufOffset + kNumUBYBF16Stages;

        constexpr uint32_t kUBEndOffset = ub_rms_sqr_sum.offset(kNumPrenormL0CStages);
        constexpr uint32_t kBufEndOffset = kUBSqrSumBufOffset + kNumUBSqrSumStages;

        // Storage and synchronization constraints
        // Keep the derived coefficient stages rotating evenly across L1 X wraparound
        static_assert(kNumL1XStages % kNumUBPostCoeffStages == 0);
        static_assert(kUBEndOffset <= UBSizeBytes and kBufEndOffset <= 32);

        constexpr uint32_t kNumMainStepsBeforeNorm = 4;
        constexpr uint32_t kSqrSumMStride = kNumAICoresPerSplit * BLOCK_M;
        constexpr uint32_t kSqrSumGMStride = get_sqr_sum_m_stride_bytes<kNumKSplits, kSqrSumMStride>();

        // RMS groups contain whole producer stores and whole epilogue tasks
        constexpr uint32_t kRMSSqrSumBlockM = kEpilogueBlockM > SUB_BLOCK_M ? kEpilogueBlockM : SUB_BLOCK_M;
        static_assert(kRMSSqrSumBlockM % SUB_BLOCK_M == 0 and kRMSSqrSumBlockM % kEpilogueBlockM == 0 and kSqrSumMStride % kRMSSqrSumBlockM == 0);
        static_assert(not kSplitNormQuant or (kStoreBF16 and kStoreFP8));
        // Place epilogue storage and buffer IDs after the main AIV workspace
        constexpr uint32_t kNormQuantBudgetPerMainStep = SUB_BLOCK_M * BLOCK_H;
        using NormQuant = NormQuantEpilogue<
            SHAPE_H, kEpilogueBlockM, NORM_BLOCK_H, BLOCK_H, kNumKSplits, kUBEndOffset,
            kNumAIVCores, kBufEndOffset, kIsSFRowMajor, kStoreBF16, kStoreFP8, kRMSSqrSumBlockM, kSplitNormQuant>;
        using Mix = MixEpilogue<
            SHAPE_H, kNumHCOutputs, kEpilogueBlockM, kNumRoutes, kNumKSplits,
            kNumAIVCores, aligned(NormQuant::get_ub_end_offset(), sizeof(vector_float))>;
        static_assert(kNumMBlocksPerChunk * BLOCK_M % (kNumAIVCores * kEpilogueBlockM) == 0);
        constexpr uint32_t kNumEpilogueMBlocksPerChunk = kNumMBlocksPerChunk * BLOCK_M / kEpilogueBlockM;

        // AIV identity and sub-block position
        const uint32_t sub_idx = asc_get_sub_block_id();
        const uint32_t aiv_idx = asc_get_block_idx() * kNumSubAIVs + sub_idx;
        const uint32_t sub_m_offset = sub_idx * SUB_BLOCK_M;

        // AIV consumer stages: X, BF16/NZ new residual and the L1 handoffs
        uint32_t ub_x_stage_idx = 0;
        uint32_t ub_new_residual_stage_idx = 0;
        uint32_t ub_new_residual_nz_stage_idx = 0;
        uint32_t ub_y_bf16_stage_idx = 0;
        uint32_t l1_x_stage_idx = 0;
        uint32_t l1_new_residual_stage_idx = 0;

        // AIV prefetch stage and M/H position
        uint32_t ub_x_prefetch_stage_idx = 0;
        uint32_t prefetch_x_m_block_idx = core_idx_in_k_split;
        uint32_t prefetch_x_h_idx = h_begin_idx;

        // Epilogue completion notifications and private Mix/NQ state
        uint32_t chunk_stage_idx = 0;
        using Epilogue = MegaHCEpilogue<
            NormQuant, Mix, NORM_BLOCK_H, kNumEpilogueMBlocksPerChunk,
            kFFTSChunkFlagOffset, kNumChunkStages, kNormQuantBudgetPerMainStep>;
        const uint32_t mix_aiv_idx = kMixOnTailAIVs ? kNumAIVCores - 1 - aiv_idx : aiv_idx;
        Epilogue epilogue{NormQuant{norm_args, shape_m, aiv_idx, gm_rms_sqr_sum},
                          Mix{mix_args, shape_m, mix_aiv_idx, gm_prenorm, gm_sqr_sum}};

        set_intra_blocks<PIPE_MTE3, kNumUBNewResidualStages, kIntraUBNewResidualFlagOffset>(sub_idx);
        asc_set_atomic_none();
        for (uint32_t stage_idx = 0; stage_idx < kNumUBPostCoeffStages; ++ stage_idx)
            vf_init_post_coeff<ub_post_coeff_matrix.size_per_stage() / sizeof(bfloat16_t)>(ub_post_coeff_matrix[stage_idx].ptr());
        // Join initialization before idle AIVs reuse main UB for final Norm/Quant
        asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(kNumPipeEvents - 1));
        asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(kNumPipeEvents - 1));

        const auto load_x_coeff_gm2ub = [&](uint32_t m_block_idx, uint32_t h_idx, uint32_t stage_idx) __aicore__ {
            const uint32_t m_idx = m_block_idx * BLOCK_M;
            const uint32_t actual_m = kFullMBlocks ? BLOCK_M : min(shape_m - m_idx, BLOCK_M);
            const uint32_t sub_m_idx = m_idx + sub_m_offset;
            const uint32_t sub_actual_m = actual_m > sub_m_offset ? min(actual_m - sub_m_offset, SUB_BLOCK_M) : 0;
            if (sub_actual_m == 0)
                return;
            asc_lock(PIPE_MTE2, kUBXBufOffset + stage_idx);
            copy_gm_to_ub(ub_x[stage_idx].ptr(), gm_layer_input.index(sub_m_idx, h_idx).ptr(), sub_actual_m,
                          BLOCK_H, SHAPE_H * sizeof(bfloat16_t), asc_load_l2_cache_mode::NOTALLOC_KEEP);
            if (h_idx == h_begin_idx) {
                copy_gm_to_ub(ub_post_mix[stage_idx].ptr(), gm_post_mix.index(sub_m_idx, 0).ptr(), 1,
                              sub_actual_m * kNumRoutes, sub_actual_m * kNumRoutes * sizeof(float),
                              asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM);
                copy_gm_to_ub(ub_comb_res_mix[stage_idx].ptr(), gm_comb_res_mix.batch(sub_m_idx).ptr(), 1,
                              sub_actual_m * kNumRoutes * kNumRoutes,
                              sub_actual_m * kNumRoutes * kNumRoutes * sizeof(float), asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM);
            }
            copy_gm_to_ub(ub_shifted_prev_mix[stage_idx].ptr(), gm_shifted_prev_mix.index(sub_m_idx, 0).ptr(), 1,
                          sub_actual_m * kNumRoutes, sub_actual_m * kNumRoutes * sizeof(float),
                          asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM);

            asc_unlock(PIPE_MTE2, kUBXBufOffset + stage_idx);
        };

        const auto prefetch_x_coeff_gm2ub = [&](uint32_t& m_block_idx, uint32_t& h_idx) __aicore__ {
            if (m_block_idx >= num_m_blocks)
                return;
            load_x_coeff_gm2ub(m_block_idx, h_idx, ub_x_prefetch_stage_idx);
            const uint32_t m_block_base_idx = m_block_idx / kNumMBlocksPerChunk * kNumMBlocksPerChunk;
            m_block_idx += kNumAICoresPerSplit;
            if (m_block_idx >= min(num_m_blocks, m_block_base_idx + kNumMBlocksPerChunk)) {
                m_block_idx = m_block_base_idx + core_idx_in_k_split;
                h_idx += BLOCK_H;
                if (h_idx == h_end_idx) {
                    h_idx = h_begin_idx;
                    m_block_idx += kNumMBlocksPerChunk;
                }
            }
            ub_x_prefetch_stage_idx = (ub_x_prefetch_stage_idx + 1) % kNumUBXStages;
        };

        const auto prepare_x_coeff_ub2l1 = [&](uint32_t m_block_base_idx, uint32_t m_block_idx, uint32_t h_idx, uint32_t ub_x_prepare_stage_idx) __aicore__ {
            if (m_block_idx >= min(num_m_blocks, m_block_base_idx + kNumMBlocksPerChunk))
                return;

            const auto ub_x_stage = ub_x[ub_x_prepare_stage_idx];
            const uint32_t sub_m_idx = m_block_idx * BLOCK_M + sub_m_offset;
            const uint32_t sub_actual_m = kFullMBlocks ? SUB_BLOCK_M : (sub_m_idx < shape_m ? min(shape_m - sub_m_idx, SUB_BLOCK_M) : 0);
            if (sub_actual_m == 0) {
                l1_x_stage_idx = (l1_x_stage_idx + 1) % kNumL1XStages;
                return;
            }

            // Keep X/coeff locked until PreApply consumes shifted_prev_mix
            asc_lock(PIPE_V, kUBXBufOffset + ub_x_prepare_stage_idx);
            const uint32_t ub_x_coeff_stage_idx = l1_x_stage_idx % kNumUBPostCoeffStages;
            wait_intra_block(PIPE_MTE3, sub_idx, kIntraL1XFlagOffset + l1_x_stage_idx);

            // Pack coefficients once for this M block
            if (h_idx == h_begin_idx) {
                asc_lock(PIPE_V, kUBPostCoeffBufOffset + ub_x_coeff_stage_idx);
                vf_prepare_post_coeff<SUB_BLOCK_M, kNumRoutes, kPostMadM, kPostMadK, kPostMadResidualK, kNumFP32toBF16Splits>(
                    ub_comb_res_mix[ub_x_prepare_stage_idx].ptr(), ub_post_mix[ub_x_prepare_stage_idx].ptr(),
                    ub_post_coeff_matrix[ub_x_coeff_stage_idx].ptr());
                asc_unlock(PIPE_V, kUBPostCoeffBufOffset + ub_x_coeff_stage_idx);

                // Copy the coefficient matrices to L1
                const uint32_t m_block_idx_in_core = (m_block_idx - m_block_base_idx - core_idx_in_k_split) / kNumAICoresPerSplit;
                const l1_ptr<bfloat16_t, Major::K> l1_coeff_mads(kPostMadM, kNumFP32toBF16Splits * kPostMadK, l1_coeff[m_block_idx_in_core].addr);
                asc_lock(PIPE_MTE3, kUBPostCoeffBufOffset + ub_x_coeff_stage_idx);
                asc_copy_ub2l1(l1_coeff_mads[sub_idx].ptr(),
                               ub_post_coeff_matrix[ub_x_coeff_stage_idx].ptr(), kNumPostMadsPerBlock / kNumSubAIVs,
                               kPostMadM * kNumFP32toBF16Splits * kPostMadK * sizeof(bfloat16_t) / kDataBlockBytes, 0,
                               (kNumSubAIVs - 1) * kPostMadM * kNumFP32toBF16Splits * kPostMadK * sizeof(bfloat16_t) / kDataBlockBytes);
                asc_unlock(PIPE_MTE3, kUBPostCoeffBufOffset + ub_x_coeff_stage_idx);
            }

            // Each AIV uses its own X fractal; rows beyond SUB_BLOCK_M stay zero
            asc_lock(PIPE_V, kUBXNZBufOffset + ub_x_coeff_stage_idx);
            vf_pack_x_nz<SUB_BLOCK_M, BLOCK_H>(
                ub_x_stage.ptr(), ub_x_nz[ub_x_coeff_stage_idx].ptr(), sub_actual_m);
            asc_unlock(PIPE_V, kUBXNZBufOffset + ub_x_coeff_stage_idx);

            // Copy X to L1 and notify Post
            asc_lock(PIPE_MTE3, kUBXNZBufOffset + ub_x_coeff_stage_idx);
            const l1_ptr<bfloat16_t, Major::MN> l1_x_sub_blocks(BLOCK_H, kPostMadXK, l1_x[l1_x_stage_idx].addr);
            asc_copy_ub2l1(l1_x_sub_blocks[sub_idx].ptr(), ub_x_nz[ub_x_coeff_stage_idx].ptr(),
                           BLOCK_H / kPostFracK, SUB_BLOCK_M, 1,
                           kPostMadXK - SUB_BLOCK_M);
            set_intra_block(PIPE_MTE3, sub_idx, kIntraL1XFlagOffset + l1_x_stage_idx);
            asc_unlock(PIPE_MTE3, kUBXNZBufOffset + ub_x_coeff_stage_idx);

            l1_x_stage_idx = (l1_x_stage_idx + 1) % kNumL1XStages;
        };

        const auto prefetch_prepare_x_coeff = [&](uint32_t m_block_base_idx, uint32_t m_block_idx, uint32_t h_idx) __aicore__ {
            // Seed the first input; later H blocks are prepared by their predecessor
            if (m_block_base_idx == 0 and m_block_idx == core_idx_in_k_split and h_idx == h_begin_idx)
                prepare_x_coeff_ub2l1(m_block_base_idx, m_block_idx, h_idx, ub_x_stage_idx);

            // Prepare one input ahead across M, H and chunk boundaries
            const bool is_last_m = m_block_idx + kNumAICoresPerSplit >= min(num_m_blocks, m_block_base_idx + kNumMBlocksPerChunk);
            const bool is_next_chunk = is_last_m and h_idx + BLOCK_H == h_end_idx;
            const uint32_t next_m_block_base_idx = is_next_chunk ? m_block_base_idx + kNumMBlocksPerChunk : m_block_base_idx;
            prepare_x_coeff_ub2l1(next_m_block_base_idx,
                                  is_last_m ? next_m_block_base_idx + core_idx_in_k_split : m_block_idx + kNumAICoresPerSplit,
                                  is_last_m ? (is_next_chunk ? h_begin_idx : h_idx + BLOCK_H) : h_idx,
                                  (ub_x_stage_idx + 1) % kNumUBXStages);
        };

        const auto process_new_residual_ub2gm_ub2l1 = [&](uint32_t m_block_base_idx, uint32_t m_block_idx, uint32_t h_idx) __aicore__ {
            const uint32_t m_block_idx_in_core = (m_block_idx - m_block_base_idx - core_idx_in_k_split) / kNumAICoresPerSplit;
            const uint32_t m_idx = m_block_idx * BLOCK_M;
            const uint32_t actual_m = kFullMBlocks ? BLOCK_M : min(shape_m - m_idx, BLOCK_M);
            const uint32_t sub_m_idx = m_idx + sub_m_offset;
            const uint32_t sub_actual_m = actual_m > sub_m_offset ? min(actual_m - sub_m_offset, SUB_BLOCK_M) : 0;

            if (sub_actual_m > 0) {
                const auto ub_new_residual_stage = ub_new_residual[ub_new_residual_stage_idx];
                const auto ub_new_residual_nz_stage = ub_new_residual_nz[ub_new_residual_nz_stage_idx];
                const auto ub_y_bf16_stage = ub_y_bf16[ub_y_bf16_stage_idx];
                const auto ub_shifted_prev_mix_stage = ub_shifted_prev_mix[ub_x_stage_idx];

                // Receive Post outputs and cast them to FP32 NZ
                asc_lock(PIPE_V, kUBNewResidualBufOffset + ub_new_residual_stage_idx);

                #pragma unroll
                for (uint32_t new_residual_block_idx = 0; new_residual_block_idx < kNumPostVFBlocks; ++ new_residual_block_idx) {
                    const uint32_t flag_idx = kIntraUBNewResidualFlagOffset +
                        new_residual_block_idx * kNumUBNewResidualStages + ub_new_residual_stage_idx;
                    wait_intra_block(PIPE_V, sub_idx, flag_idx);
                    if (new_residual_block_idx == 0)
                        asc_lock(PIPE_V, kUBNewResidualNZBufOffset + ub_new_residual_nz_stage_idx);
                    const uint32_t sub_m_begin = new_residual_block_idx * kPostVFBlockM;
                    if (sub_m_begin < sub_actual_m)
                        vf_cast_new_residual_nz<SUB_BLOCK_M, BLOCK_H, kNumRoutes>(
                            ub_new_residual_stage.ptr() + sub_m_begin * BLOCK_K,
                            ub_new_residual_nz_stage.ptr() + sub_m_begin * FRAC_K,
                            min(aligned(sub_actual_m - sub_m_begin, kNumTokensPerPostMad), kPostVFBlockM));
                }
                asc_unlock(PIPE_V, kUBNewResidualNZBufOffset + ub_new_residual_nz_stage_idx);

                // Copy NZ to L1 and notify Prenorm
                asc_lock(PIPE_MTE3, kUBNewResidualNZBufOffset + ub_new_residual_nz_stage_idx);
                wait_intra_block(PIPE_MTE3, sub_idx, kIntraL1NewResidualFlagOffset + l1_new_residual_stage_idx);
                asc_copy_ub2l1(l1_new_residual[l1_new_residual_stage_idx].ptr() + sub_m_offset * FRAC_K,
                               ub_new_residual_nz_stage.ptr(), BLOCK_K / FRAC_K, SUB_BLOCK_M, 1, SUB_BLOCK_M);
                set_intra_block(PIPE_MTE3, sub_idx, kIntraL1NewResidualFlagOffset + l1_new_residual_stage_idx);
                asc_unlock(PIPE_MTE3, kUBNewResidualNZBufOffset + ub_new_residual_nz_stage_idx);

                // Acquire square sums at the chunk's first V write
                if (h_idx == h_begin_idx and m_block_idx_in_core == 0)
                    asc_lock(PIPE_V, kUBSqrSumBufOffset);

                // PreApply and square sums; V ordering protects NZ reads from the next V overwrite
                asc_lock(PIPE_V, kUBYBF16BufOffset + ub_y_bf16_stage_idx);
                vf_preapply_sqr_sum_from_nz<SUB_BLOCK_M, BLOCK_H, kNumRoutes>(
                    ub_new_residual_nz_stage.ptr(), ub_hc_sqr_sum[m_block_idx_in_core].ptr(),
                    ub_shifted_prev_mix_stage.ptr(), ub_y_bf16_stage.ptr(),
                    ub_rms_sqr_sum[m_block_idx_in_core].ptr(), sub_actual_m, h_idx == h_begin_idx);
                asc_unlock(PIPE_V, kUBYBF16BufOffset + ub_y_bf16_stage_idx);
                // Release the X/coeff stage held by prepare_x_coeff_ub2l1
                asc_unlock(PIPE_V, kUBXBufOffset + ub_x_stage_idx);
                asc_unlock(PIPE_V, kUBNewResidualBufOffset + ub_new_residual_stage_idx);

                // Store new residual and intermediate BF16 Y
                asc_lock(PIPE_MTE3, kUBNewResidualBufOffset + ub_new_residual_stage_idx);
                copy_ub_to_gm(norm_args.new_residual.index(sub_m_idx, h_idx).ptr(),
                              ub_new_residual_stage.ptr(), sub_actual_m * kNumRoutes, BLOCK_H,
                              SHAPE_H * sizeof(bfloat16_t), asc_store_l2_cache_mode::NOTALLOC_CLEAN);
                asc_lock(PIPE_MTE3, kUBYBF16BufOffset + ub_y_bf16_stage_idx);
                copy_ub_to_gm(norm_args.y_bf16.index(sub_m_idx, h_idx).ptr(), ub_y_bf16_stage.ptr(),
                              sub_actual_m, BLOCK_H, SHAPE_H * sizeof(bfloat16_t), asc_store_l2_cache_mode::NORMAL_LAST_VICTIM);
                asc_unlock(PIPE_MTE3, kUBYBF16BufOffset + ub_y_bf16_stage_idx);
                asc_unlock(PIPE_MTE3, kUBNewResidualBufOffset + ub_new_residual_stage_idx);
                set_intra_block(PIPE_MTE3, sub_idx, kIntraUBNewResidualFlagOffset + ub_new_residual_stage_idx);
            }

            // Advance stages for every block, including empty AIV subblocks
            ub_x_stage_idx = (ub_x_stage_idx + 1) % kNumUBXStages;
            ub_new_residual_stage_idx = (ub_new_residual_stage_idx + 1) % kNumUBNewResidualStages;
            l1_new_residual_stage_idx = (l1_new_residual_stage_idx + 1) % kNumL1NewResidualStages;
            ub_new_residual_nz_stage_idx = (ub_new_residual_nz_stage_idx + 1) % kNumUBNewResidualNZStages;
            ub_y_bf16_stage_idx = (ub_y_bf16_stage_idx + 1) % kNumUBYBF16Stages;
        };

        for (uint32_t stage_idx = 0; stage_idx < kNumUBXStages; ++ stage_idx)
            prefetch_x_coeff_gm2ub(prefetch_x_m_block_idx, prefetch_x_h_idx);
        epilogue.initialize();

        for (uint32_t m_block_base_idx = 0; m_block_base_idx < num_m_blocks; m_block_base_idx += kNumMBlocksPerChunk) {
            const uint32_t m_block_end_idx = min(num_m_blocks, m_block_base_idx + kNumMBlocksPerChunk);
            const uint32_t m_block_begin_idx = m_block_base_idx + core_idx_in_k_split;
            const uint32_t num_m_blocks_in_core = m_block_begin_idx < m_block_end_idx
                ? ceil_div(m_block_end_idx - m_block_begin_idx, kNumAICoresPerSplit) : 0;
            uint32_t num_completed_main_steps = 0;
            const uint32_t num_main_steps = num_m_blocks_in_core * ((h_end_idx - h_begin_idx) / BLOCK_H);
            if (m_block_begin_idx < m_block_end_idx) {
                for (uint32_t h_idx = h_begin_idx; h_idx < h_end_idx; h_idx += BLOCK_H) {
                    // Trailing-AIV batches have one chunk; other batches interleave completed-chunk NQ
                    for (uint32_t m_block_idx = m_block_begin_idx; m_block_idx < m_block_end_idx; m_block_idx += kNumAICoresPerSplit) {
                        prefetch_prepare_x_coeff(m_block_base_idx, m_block_idx, h_idx);
                        // Keep one main step between NQ admission and computation
                        if (not kMixOnTailAIVs and num_completed_main_steps != kNumMainStepsBeforeNorm)
                            epilogue.step();
                        process_new_residual_ub2gm_ub2l1(m_block_base_idx, m_block_idx, h_idx);
                        prefetch_x_coeff_gm2ub(prefetch_x_m_block_idx, prefetch_x_h_idx);

                        // Delay the previous-chunk NQ handoff until Post/Prenorm has started
                        ++ num_completed_main_steps;
                        if (not kMixOnTailAIVs and m_block_base_idx > 0 and num_completed_main_steps == kNumMainStepsBeforeNorm)
                            epilogue.prepare();
                    }
                }
            }

            // Short or idle chunks still consume the previous-chunk notification exactly once
            if (not kMixOnTailAIVs and m_block_base_idx > 0 and num_main_steps < kNumMainStepsBeforeNorm) {
                epilogue.prepare();
            }

            // Store HC/RMS statistics per AIV subblock, including the final block's workspace padding
            const uint32_t sub_m_idx = m_block_begin_idx * BLOCK_M + sub_m_offset;
            const uint32_t num_sqr_sum_m_blocks = sub_m_idx < shape_m ?
                min(num_m_blocks_in_core, ceil_div(shape_m - sub_m_idx, kSqrSumMStride)) : 0;
            if (num_sqr_sum_m_blocks > 0) {
                asc_unlock(PIPE_V, kUBSqrSumBufOffset);
                asc_lock(PIPE_MTE3, kUBSqrSumBufOffset);
                const uint64_t workspace_m_idx = get_workspace_m_idx<kNumKSplits, SUB_BLOCK_M>(sub_m_idx, k_split_idx);
                asc_copy_ub2gm_align(gm_sqr_sum.index(workspace_m_idx, 0).ptr(), ub_hc_sqr_sum.ptr(), num_sqr_sum_m_blocks,
                                     SUB_BLOCK_M * sizeof(float), asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                                     kSqrSumGMStride, ub_hc_sqr_sum.size_per_stage());
                const uint64_t rms_workspace_m_idx =
                    sub_m_idx / kRMSSqrSumBlockM * kRMSSqrSumBlockM * kNumKSplits + k_split_idx * kRMSSqrSumBlockM + sub_m_idx % kRMSSqrSumBlockM;
                asc_copy_ub2gm_align(gm_rms_sqr_sum.index(rms_workspace_m_idx, 0).ptr(), ub_rms_sqr_sum.ptr(), num_sqr_sum_m_blocks,
                                     SUB_BLOCK_M * sizeof(float), asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM,
                                     kSqrSumGMStride, ub_rms_sqr_sum.size_per_stage());
            }

            // Norm needs only AIV Y/statistics; final Prenorm FIX can overlap its drain
            if (m_block_base_idx + kNumMBlocksPerChunk >= num_m_blocks)
                asc_sync_inter_arrive(PIPE_MTE3, kFFTSChunkFlagOffset + kNumChunkStages);

            // Join local FIX completion before publishing this chunk's GEMM and statistics
            asc_sync_intra_wait(PIPE_MTE3, kIntraPrenormFlagOffset + chunk_stage_idx);
            set_intra_block(PIPE_MTE3, sub_idx, kIntraPrenormFlagOffset + chunk_stage_idx);
            asc_sync_inter_arrive(PIPE_MTE3, kFFTSChunkFlagOffset + chunk_stage_idx);
            chunk_stage_idx = (chunk_stage_idx + 1) % kNumChunkStages;
            if (num_sqr_sum_m_blocks > 0)
                asc_unlock(PIPE_MTE3, kUBSqrSumBufOffset);

            if constexpr (not kMixOnTailAIVs)
                epilogue.step();
        }

        // Admit the final chunk and finish all remaining epilogue work
        epilogue.drain_steps();
        epilogue.drain_flags();
    }
    asc_set_atomic_none();
}

} // namespace deep_gemm::mega_mhc
