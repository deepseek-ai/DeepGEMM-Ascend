#pragma once

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/layout/mega_gate.hpp>
#include <deep_gemm/scheduler/mega_gate.hpp>
#include <deep_gemm/simd/mega_gate.hpp>
#include <adv_api/math/philox.h>

namespace deep_gemm::mega_gate {

constexpr PipeEvent<PIPE_MTE2, PIPE_MTE1> kL1Full{0};
constexpr PipeEvent<PIPE_MTE1, PIPE_MTE2> kL1Empty{0};
constexpr PipeEvent<PIPE_MTE1, PIPE_M> kL0Full{0};
constexpr PipeEvent<PIPE_M, PIPE_MTE1> kL0Empty{0};
constexpr PipeEvent<PIPE_S, PIPE_V> kArgsFull{7};
constexpr PipeEvent<PIPE_MTE2, PIPE_V> kScoresFull{7};
constexpr PipeEvent<PIPE_V, PIPE_S> kResultReady{6};
constexpr PipeEvent<PIPE_MTE2, PIPE_S> kFlagsFull{6};
constexpr PipeEvent<PIPE_S, PIPE_MTE3> kResultFull{6};
constexpr PipeEvent<PIPE_MTE3, PIPE_MTE2> kScoresEmpty{7};

// AIC publishes complete GEMM waves; both AIVs reduce and route their token ranges.
template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t MAD_M, uint32_t MAD_N, uint32_t MAD_K,
          uint32_t kNumL1Stages, uint32_t kNumL0Stages, uint32_t kNumAICores,
          asc_load_l2_cache_mode kL2CtrlA, asc_load_l2_cache_mode kL2CtrlB,
          uint32_t kNumTopk, uint32_t kNumSharedExperts, uint32_t kNumCachedDuplicates,
          bool kHasBias, bool kHasImageBias, bool kHasMask, bool kHasFixedRouting,
          bool kHasPhysicalMap, bool kHasUnmappedOutput, bool kHasRandomRouting,
          uint32_t kNumRows, uint32_t kNumKSplits, bool kPipeline>
__global__ __mix__(1, 2) void mega_gate_impl(layout::KernelArgs<bfloat16_t>) {
    asc_init();

    const auto& kernel_args = get_replicated_args<layout::KernelArgs<bfloat16_t>>();
    const auto& gm_a = kernel_args.gm_a;
    const auto& gm_b = kernel_args.gm_b;
    const auto& gm_d = kernel_args.gm_d;
    const uint32_t shape_m = kernel_args.shape_m;
    const auto& args = kernel_args.gate;

    static_assert(kNumL1Stages > 0 and kNumL1Stages <= 8 and kNumL0Stages > 0 and kNumL0Stages <= 8);
    static_assert(BLOCK_M % MAD_M == 0 and BLOCK_N % MAD_N == 0 and BLOCK_K % MAD_K == 0);
    static_assert(kNumRows > 0 and kNumKSplits > 0);

    // The launcher specializes whether the task count exceeds one wave.
    const sched::Scheduler<SHAPE_N, BLOCK_M, BLOCK_N, kNumKSplits, kPipeline> scheduler{shape_m};
    const uint32_t num_waves = kPipeline ? ceil_div(scheduler.num_tasks, kNumAICores) : 1;
    if DG_IS_AIC {
        constexpr uint32_t kNumL0CStages = (BLOCK_M / MAD_M) * (BLOCK_N / MAD_N);
        constexpr l1_ptr<bfloat16_t> l1a(BLOCK_M, BLOCK_K);
        constexpr l1_ptr<bfloat16_t> l1b(BLOCK_N, BLOCK_K, l1a.offset(kNumL1Stages));
        constexpr l0a_ptr<bfloat16_t> l0a(MAD_M, MAD_K);
        constexpr l0b_ptr<bfloat16_t> l0b(MAD_N, MAD_K);
        constexpr l0c_ptr<float> l0c(MAD_M, MAD_N);
        constexpr layout::UBLayout ub_layout{
            SHAPE_N, kNumSharedExperts, kNumRows, kNumKSplits, kNumCachedDuplicates,
            kHasImageBias, kHasMask, kHasFixedRouting,
            kHasPhysicalMap, kHasUnmappedOutput, kHasRandomRouting, asc_get_vf_len(), kNumTopk};
        static_assert(ub_layout.num_bytes <= UBSizeBytes);
        // Single-wave staging overlaps FIX to UB with the paired AIV stores to GM.
        constexpr bool kUseFixStaging = layout::use_fix_staging(BLOCK_M, BLOCK_N, MAD_M, MAD_N, ub_layout.num_bytes, kPipeline);
        constexpr uint64_t kFixStageOffset = layout::fix_staging_offset(ub_layout.num_bytes);
        constexpr uint64_t kFixStageBufferBytes = layout::fix_staging_buffer_bytes(MAD_N);
        static_assert(not kUseFixStaging or kFixStageOffset + 2 * kFixStageBufferBytes <= UBSizeBytes);
        static_assert(l1b.offset(kNumL1Stages) <= L1SizeBytes);
        static_assert(l0a.offset(kNumL0Stages) <= L0ASizeBytes and l0b.offset(kNumL0Stages) <= L0BSizeBytes);
        static_assert(l0c.offset(kNumL0CStages) <= L0CSizeBytes);

        notify<PIPE_MTE1, PIPE_MTE2, kNumL1Stages>(kL1Empty);
        notify<PIPE_M, PIPE_MTE1, kNumL0Stages>(kL0Empty);

        for (uint32_t wave_idx = 0; wave_idx < num_waves; ++ wave_idx) {
            const uint32_t task_idx = wave_idx * kNumAICores + block_idx;
            if (task_idx < scheduler.num_tasks) {
                uint32_t m_block_idx, n_block_idx, k_split_idx;
                scheduler.get_task(task_idx, m_block_idx, n_block_idx, k_split_idx);
                const uint32_t m_idx = m_block_idx * BLOCK_M;
                const uint32_t n_idx = n_block_idx * BLOCK_N;
                const uint32_t actual_m = min(shape_m - m_idx, BLOCK_M);
                const uint32_t actual_n = min(SHAPE_N - n_idx, BLOCK_N);
                const auto a = gm_a.batch(k_split_idx).index(m_idx, 0);
                const auto b = gm_b.batch(k_split_idx).index(n_idx, 0);
                const auto d = gm_d.batch(k_split_idx).index(m_idx, n_idx);

                // GEMM: GM -> L1 -> L0 -> FP32 partials.
                // Continue the L1 ring across waves so the next DMA uses the oldest free slot.
                for (uint32_t k_idx = 0, l1_stage_idx = wave_idx * (ceil_div(SHAPE_K, BLOCK_K) % kNumL1Stages) % kNumL1Stages;
                     k_idx < SHAPE_K; k_idx += BLOCK_K, l1_stage_idx = (l1_stage_idx + 1) % kNumL1Stages) {
                    const uint32_t actual_k = min(SHAPE_K - k_idx, BLOCK_K);
                    wait<PIPE_MTE1, PIPE_MTE2>(kL1Empty, l1_stage_idx);
                    copy_gm_to_l1(l1a[l1_stage_idx], a.index(0, k_idx), actual_m, actual_k, kL2CtrlA);
                    copy_gm_to_l1(l1b[l1_stage_idx], b.index(0, k_idx), actual_n, actual_k, kL2CtrlB);
                    notify<PIPE_MTE2, PIPE_MTE1>(kL1Full, l1_stage_idx);
                    wait<PIPE_MTE2, PIPE_MTE1>(kL1Full, l1_stage_idx);

                    for (uint32_t m_mad_idx = 0, l0c_stage_idx = 0; m_mad_idx < actual_m; m_mad_idx += MAD_M) {
                        for (uint32_t n_mad_idx = 0; n_mad_idx < actual_n; n_mad_idx += MAD_N, ++ l0c_stage_idx) {
                            const uint32_t mad_m = min(actual_m - m_mad_idx, MAD_M);
                            const uint32_t mad_n = min(actual_n - n_mad_idx, MAD_N);
                            const auto l0c_tile = l0c[l0c_stage_idx].as_mad_aligned(mad_m, mad_n);
                            for (uint32_t k_mad_idx = 0, l0_stage_idx = 0; k_mad_idx < actual_k;
                                 k_mad_idx += MAD_K, l0_stage_idx = (l0_stage_idx + 1) % kNumL0Stages) {
                                const uint32_t mad_k = min(actual_k - k_mad_idx, MAD_K);
                                const auto l0a_tile = l0a[l0_stage_idx].as_mad_aligned(mad_m, mad_k);
                                const auto l0b_tile = l0b[l0_stage_idx].as_mad_aligned(mad_n, mad_k);
                                wait<PIPE_M, PIPE_MTE1>(kL0Empty, l0_stage_idx);
                                if (BLOCK_N == MAD_N or BLOCK_K > MAD_K * kNumL0Stages or n_mad_idx == 0)
                                    copy_l1_to_l0a(l0a_tile, l1a[l1_stage_idx], m_mad_idx, k_mad_idx);
                                if (BLOCK_M == MAD_M or BLOCK_N != MAD_N or BLOCK_K > MAD_K * kNumL0Stages or m_mad_idx == 0)
                                    copy_l1_to_l0b(l0b_tile, l1b[l1_stage_idx], n_mad_idx, k_mad_idx);
                                notify<PIPE_MTE1, PIPE_M>(kL0Full, l0_stage_idx);
                                wait<PIPE_MTE1, PIPE_M>(kL0Full, l0_stage_idx);
                                mad(l0c_tile, l0a_tile, l0b_tile, mad_m, mad_n, mad_k,
                                    k_idx + BLOCK_K >= SHAPE_K and k_mad_idx + MAD_K >= actual_k ?
                                        asc_unit_flag_mode::ENABLE_UPDATE : asc_unit_flag_mode::ENABLE_KEEP,
                                    k_idx == 0 and k_mad_idx == 0);
                                notify<PIPE_M, PIPE_MTE1>(kL0Empty, l0_stage_idx);
                            }
                            if (k_idx + BLOCK_K >= SHAPE_K) {
                                const auto dst = d.index(m_mad_idx, n_mad_idx).ptr();
                                if constexpr (kUseFixStaging) {
                                    if (mad_m % layout::kFixChunkRows == 0) {
                                        const uint32_t num_chunks = mad_m / layout::kFixChunkRows;
                                        for (uint32_t chunk = 0; chunk < num_chunks; ++ chunk) {
                                            const uint32_t parity = chunk & 1u;
                                            if (chunk >= 2) {
                                                wait_intra_block(PIPE_FIX, 0, layout::kChunkAckFlag + parity);
                                                wait_intra_block(PIPE_FIX, 1, layout::kChunkAckFlag + parity);
                                            }
                                            const ub_ptr<float> ub_stage(
                                                layout::kFixChunkHalfRows, mad_n,
                                                kFixStageOffset + parity * kFixStageBufferBytes);
                                            copy_l0c_to_ub(
                                                ub_stage, l0c_tile.offset_nz_m(chunk * layout::kFixChunkRows),
                                                layout::kFixChunkRows, mad_n, asc_dual_dst_mode::DUAL_DST_SPLIT_M, 0,
                                                infer_quant_mode<float, float>(), false, false,
                                                chunk == 0 ? asc_unit_flag_mode::ENABLE_UPDATE
                                                           : asc_unit_flag_mode::DISABLE);
                                            set_intra_block(PIPE_FIX, 0, layout::kChunkReadyFlag + parity);
                                            set_intra_block(PIPE_FIX, 1, layout::kChunkReadyFlag + parity);
                                        }
                                        continue;
                                    }
                                }
                                copy_l0c_to_gm(dst, l0c_tile, mad_m, mad_n, d.stride_outer);
                            }
                        }
                    }
                    notify<PIPE_MTE1, PIPE_MTE2>(kL1Empty, l1_stage_idx);
                }
            }

            // Wait for both AIVs to drain this wave's staged partials into GM.
            if constexpr (kUseFixStaging) {
                wait_intra_block(PIPE_FIX, 0, layout::kStoreDoneFlag);
                wait_intra_block(PIPE_FIX, 1, layout::kStoreDoneFlag);
            }
            // Idle cores join the final wave; FIX completion precedes AIV reads.
            const uint32_t stage_idx = wave_idx % 2;
            wait_intra_block(PIPE_FIX, 0, stage_idx);
            wait_intra_block(PIPE_FIX, 1, stage_idx);
            asc_sync_inter_arrive(PIPE_FIX, stage_idx);
            asc_sync_inter_wait(PIPE_FIX, stage_idx);
            set_intra_block(PIPE_FIX, 0, stage_idx);
            set_intra_block(PIPE_FIX, 1, stage_idx);
        }
    } else {
        constexpr layout::UBLayout ub_layout{
            SHAPE_N, kNumSharedExperts, kNumRows, kNumKSplits, kNumCachedDuplicates,
            kHasImageBias, kHasMask, kHasFixedRouting,
            kHasPhysicalMap, kHasUnmappedOutput, kHasRandomRouting, asc_get_vf_len(), kNumTopk};
        static_assert(ub_layout.num_bytes <= UBSizeBytes);
        constexpr ub_ptr<float> scores(kNumRows, ub_layout.stride);
        const auto ub_selected = reinterpret_cast<__ubuf__ int32_t*>(ub_layout.selected);
        const auto ub_ranking = reinterpret_cast<__ubuf__ float*>(ub_layout.ranking);
        const auto ub_status = reinterpret_cast<__ubuf__ uint32_t*>(ub_layout.status);
        const auto ub_flags = reinterpret_cast<__ubuf__ uint8_t*>(ub_layout.flags);
        constexpr ub_ptr<int32_t> fixed(kNumRows, ub_layout.fixed_stride / sizeof(int32_t), ub_layout.fixed);
        const auto ub_map = reinterpret_cast<__ubuf__ int32_t*>(ub_layout.map);
        const auto ub_counts = reinterpret_cast<__ubuf__ int32_t*>(ub_layout.counts);
        const auto ub_bias = reinterpret_cast<__ubuf__ float*>(ub_layout.bias);
        const auto ub_args = reinterpret_cast<__ubuf__ layout::GateArgs*>(ub_layout.args);
        constexpr bool kUseFixStaging = layout::use_fix_staging(BLOCK_M, BLOCK_N, MAD_M, MAD_N, ub_layout.num_bytes, kPipeline);
        constexpr uint64_t kFixStageOffset = layout::fix_staging_offset(ub_layout.num_bytes);
        constexpr uint64_t kFixStageBufferBytes = layout::fix_staging_buffer_bytes(MAD_N);

        const uint32_t aiv_id = asc_get_sub_block_id();
        set_intra_blocks<PIPE_V, 2>(aiv_id);
        const uint32_t aiv_idx = block_idx * 2 + aiv_id;
        // Single-wave, single-row routing leaves the remaining AIVs idle.
        if (kPipeline or kNumRows >= 2 or aiv_idx < shape_m) {
            ub_args->num_duplicate_experts = args.num_duplicate_experts;
            ub_args->ep_rank = args.ep_rank;
            ub_args->routed_scaling_factor = args.routed_scaling_factor;
            notify<PIPE_S, PIPE_V>(kArgsFull);
            wait<PIPE_S, PIPE_V>(kArgsFull);
        }

        if (aiv_idx < shape_m) {
            if constexpr (kNumCachedDuplicates != 0) {
                copy_gm_to_ub(ub_map, args.to_physical_map, 1, (SHAPE_N + kNumSharedExperts) * kNumCachedDuplicates,
                              (SHAPE_N + kNumSharedExperts) * kNumCachedDuplicates * sizeof(int32_t));
            }
            if constexpr (kHasPhysicalMap)
                copy_gm_to_ub(ub_counts, args.logical_count, 1, SHAPE_N + kNumSharedExperts,
                              (SHAPE_N + kNumSharedExperts) * sizeof(int32_t));
            if (args.bias != nullptr) copy_gm_to_ub(ub_bias, args.bias, 1, SHAPE_N, SHAPE_N * sizeof(float));
            if (args.image_bias != nullptr) copy_gm_to_ub(ub_bias + ub_layout.stride, args.image_bias, 1, SHAPE_N, SHAPE_N * sizeof(float));
        }
        for (uint32_t wave_idx = 0; wave_idx < num_waves; ++ wave_idx) {
            // Drain the local AIC's staged partial tile: UB -> GM (our half of the M-split tile).
            if constexpr (kUseFixStaging) {
                const uint32_t task_idx = wave_idx * kNumAICores + block_idx;
                if (task_idx < scheduler.num_tasks) {
                    uint32_t m_block_idx, n_block_idx, k_split_idx;
                    scheduler.get_task(task_idx, m_block_idx, n_block_idx, k_split_idx);
                    const uint32_t m_idx = m_block_idx * BLOCK_M;
                    const uint32_t n_idx = n_block_idx * BLOCK_N;
                    const uint32_t actual_m = min(shape_m - m_idx, BLOCK_M);
                    const uint32_t actual_n = min(SHAPE_N - n_idx, BLOCK_N);
                    if (actual_m % layout::kFixChunkRows == 0) {
                        const auto d = gm_d.batch(k_split_idx).index(m_idx, n_idx);
                        const uint32_t num_chunks = actual_m / layout::kFixChunkRows;
                        for (uint32_t chunk = 0; chunk < num_chunks; ++ chunk) {
                            const uint32_t parity = chunk & 1u;
                            wait_intra_block(PIPE_MTE3, aiv_id, layout::kChunkReadyFlag + parity);
                            const ub_ptr<float> ub_stage(
                                layout::kFixChunkHalfRows, actual_n,
                                kFixStageOffset + parity * kFixStageBufferBytes);
                            const uint64_t row_offset =
                                uint64_t(chunk) * layout::kFixChunkRows + aiv_id * layout::kFixChunkHalfRows;
                            copy_ub_to_gm(d.ptr() + row_offset * d.stride_outer, ub_stage,
                                          layout::kFixChunkHalfRows, actual_n,
                                          uint64_t(d.stride_outer) * sizeof(float));
                            set_intra_block(PIPE_MTE3, 0, layout::kChunkAckFlag + parity);
                        }
                    }
                }
                set_intra_block(PIPE_MTE3, 0, layout::kStoreDoneFlag);
            }
            // Only complete swizzle groups are readable before the final wave.
            const uint32_t begin_token_idx = scheduler.get_num_published_tokens(wave_idx * kNumAICores);
            const uint32_t end_token_idx =
                wave_idx + 1 == num_waves ? shape_m : scheduler.get_num_published_tokens((wave_idx + 1) * kNumAICores);
            const uint32_t num_local_tokens = ceil_div(end_token_idx - begin_token_idx, 2 * kNumAICores);
            const uint32_t local_end_token_idx = min(end_token_idx, begin_token_idx + (aiv_idx + 1) * num_local_tokens);
            for (uint32_t token_idx = begin_token_idx + aiv_idx * num_local_tokens; token_idx < local_end_token_idx;
                 token_idx += kNumRows) {
                const uint32_t num_rows = min(kNumRows, local_end_token_idx - token_idx);
                if constexpr (ub_layout.num_row_list_bytes != 0 and kHasRandomRouting) {
                    const uint32_t num_randoms = num_rows * ub_layout.num_randoms_per_row;
                    AscendC::LocalTensor<uint32_t> randoms(AscendC::TPosition::VECCALC, ub_layout.random, num_randoms);
                    const AscendC::PhiloxKey key{args.ep_rank, 0};
                    const AscendC::PhiloxCounter counter{token_idx * (ub_layout.num_randoms_per_row / 4), 0, 0, 0};
                    AscendC::PhiloxRandom(randoms, key, counter, num_randoms);
                }
                if constexpr (kHasMask)
                    copy_gm_to_ub(ub_flags, reinterpret_cast<__gm__ uint8_t*>(args.mask + token_idx), 1, num_rows, num_rows);
                if constexpr (kHasRandomRouting)
                    copy_gm_to_ub(ub_flags + ub_layout.flag_stride,
                                  reinterpret_cast<__gm__ uint8_t*>(args.force_random + token_idx), 1, num_rows, num_rows);
                if constexpr (kHasFixedRouting) {
                    copy_gm_to_ub(ub_flags + 2 * ub_layout.flag_stride,
                                  reinterpret_cast<__gm__ uint8_t*>(args.fix_routing_mask + token_idx), 1, num_rows, num_rows);
                    copy_gm_to_ub(fixed,
                                  reinterpret_cast<__gm__ int32_t*>(
                                      args.unmapped_topk_idx + uint64_t(token_idx) * args.unmapped_topk_idx_stride),
                                  num_rows, kNumTopk * sizeof(int64_t), args.unmapped_topk_idx_stride * sizeof(int64_t),
                                  ub_layout.fixed_stride);
                }
                if constexpr (kHasImageBias)
                    copy_gm_to_ub(ub_flags + 3 * ub_layout.flag_stride,
                                  reinterpret_cast<__gm__ uint8_t*>(args.image_token_mask + token_idx), 1, num_rows, num_rows);
                if (token_idx == begin_token_idx + aiv_idx * num_local_tokens)
                    wait_intra_block(PIPE_MTE2, aiv_id, wave_idx % 2);
                copy_gm_to_ub(scores, gm_d.index(token_idx, 0).ptr(), num_rows * kNumKSplits, SHAPE_N * sizeof(float),
                              ub_layout.stride * sizeof(float), ub_layout.stride * sizeof(float));
                if constexpr (ub_layout.num_row_list_bytes == 0 and (kHasMask or kHasRandomRouting or kHasFixedRouting or kHasImageBias)) {
                    notify<PIPE_MTE2, PIPE_S>(kFlagsFull);
                    wait<PIPE_MTE2, PIPE_S>(kFlagsFull);
                }
                notify<PIPE_MTE2, PIPE_V>(kScoresFull);
                wait<PIPE_MTE2, PIPE_V>(kScoresFull);
                uint64_t active_rows = UINT64_MAX;
                bool need_scores = true, need_random = false;
                if constexpr (ub_layout.num_row_list_bytes != 0) {
                    vf_classify<kHasImageBias, kHasMask, kHasFixedRouting, kHasRandomRouting,
                                kNumRows, ub_layout.flags, ub_layout.row_list>(num_rows);
                    notify<PIPE_V, PIPE_S>(kResultReady);
                    wait<PIPE_V, PIPE_S>(kResultReady);
                    const auto num_routed_rows = reinterpret_cast<__ubuf__ uint32_t*>(ub_layout.row_list) + 5 * aligned(kNumRows, 8u);
                    active_rows = uint64_t(num_routed_rows[6]) << 32 | num_routed_rows[5];
                } else if constexpr (kHasMask or kHasRandomRouting) {
                    need_scores = false;
                    for (uint32_t row_idx = 0; row_idx < num_rows; ++ row_idx) {
                        const bool active = not kHasMask or ub_flags[row_idx];
                        const bool random = kHasRandomRouting and ub_flags[ub_layout.flag_stride + row_idx];
                        need_scores |= active and not random;
                        need_random |= active and random;
                    }
                }
                // Preserve batched scoring; fuse a single row with its top-k consumer.
                constexpr bool kFuseScore = kNumRows == 1 and not (kHasRandomRouting or kHasFixedRouting);
                if constexpr (ub_layout.num_row_list_bytes == 0 and not kFuseScore) {
                    if (need_scores) {
                        vf_score<SHAPE_N, ub_layout.stride, kNumKSplits,
                                 (kNumRows >= 4 and kNumTopk <= 10) ? 2 : 1>(scores.ptr(), num_rows, ub_status);
                        asc_sync_pipe(PIPE_V);
                    }
                }
                if constexpr (ub_layout.num_row_list_bytes == 0 and kHasRandomRouting) {
                    if (need_random) {
                        const uint32_t num_randoms = num_rows * ub_layout.num_randoms_per_row;
                        AscendC::LocalTensor<uint32_t> randoms(AscendC::TPosition::VECCALC, ub_layout.random, num_randoms);
                        const AscendC::PhiloxKey key{args.ep_rank, 0};
                        const AscendC::PhiloxCounter counter{token_idx * (ub_layout.num_randoms_per_row / 4), 0, 0, 0};
                        AscendC::PhiloxRandom(randoms, key, counter, num_randoms);
                        asc_sync_pipe(PIPE_V);
                    }
                }
                if constexpr (ub_layout.num_row_list_bytes != 0) {
                    vf_route_packed<SHAPE_N, kNumTopk, kNumSharedExperts, kNumCachedDuplicates,
                                    kHasBias, kHasImageBias, kHasMask, kHasFixedRouting,
                                    kHasPhysicalMap, kHasUnmappedOutput, kHasRandomRouting,
                                    kNumRows, kNumKSplits>(token_idx, num_rows, active_rows);
                } else {
                    for (uint32_t row_idx = 0; row_idx < num_rows; row_idx += ub_layout.num_packed_rows) {
                        const auto output = ub_selected + row_idx * ub_layout.output_stride / sizeof(int32_t);
                        if constexpr (ub_layout.num_packed_rows > 1) {
                            vf_gate_packed<SHAPE_N, kNumTopk, kNumSharedExperts, ub_layout.stride,
                                           kNumCachedDuplicates, kHasPhysicalMap, kHasUnmappedOutput, ub_layout.packed_width>(
                                scores.ptr() + row_idx * ub_layout.stride, ub_bias, kHasBias, output,
                                ub_ranking, ub_map, reinterpret_cast<__ubuf__ uint32_t*>(ub_counts), ub_args,
                                ub_status + row_idx, token_idx + row_idx,
                                min(ub_layout.num_packed_rows, num_rows - row_idx));
                        } else if (kHasMask and not ub_flags[row_idx]) {
                            vf_masked<kNumTopk, kNumSharedExperts, kHasUnmappedOutput>(output, ub_status + row_idx);
                        } else if (kHasRandomRouting and ub_flags[ub_layout.flag_stride + row_idx]) {
                            vf_random<SHAPE_N, kNumTopk, kNumSharedExperts, kHasPhysicalMap, kHasUnmappedOutput>(
                                output, reinterpret_cast<__ubuf__ uint32_t*>(ub_counts),
                                reinterpret_cast<__ubuf__ uint32_t*>(ub_layout.random) +
                                    row_idx * ub_layout.num_randoms_per_row,
                                ub_status + row_idx);
                        } else {
                            const bool image = kHasImageBias and ub_flags[3 * ub_layout.flag_stride + row_idx];
                            const auto row_bias = image ? ub_bias + ub_layout.stride : ub_bias;
                            if (kHasFixedRouting and ub_flags[2 * ub_layout.flag_stride + row_idx])
                                vf_gate<SHAPE_N, kNumTopk, kNumSharedExperts, kNumCachedDuplicates,
                                        kHasPhysicalMap, kHasUnmappedOutput, true>(
                                    scores.ptr() + row_idx * ub_layout.stride, nullptr, false, output, ub_ranking, ub_map,
                                    reinterpret_cast<__ubuf__ uint32_t*>(ub_counts), ub_args,
                                    reinterpret_cast<__ubuf__ int64_t*>(
                                        fixed.ptr() + row_idx * ub_layout.fixed_stride / sizeof(int32_t)),
                                    ub_status + row_idx, token_idx + row_idx);
                            else
                                vf_gate<SHAPE_N, kNumTopk, kNumSharedExperts, kNumCachedDuplicates,
                                        kHasPhysicalMap, kHasUnmappedOutput, false,
                                        kFuseScore ? ub_layout.stride : 0, kNumKSplits>(
                                    scores.ptr() + row_idx * ub_layout.stride, row_bias, kHasBias or image, output, ub_ranking, ub_map,
                                    reinterpret_cast<__ubuf__ uint32_t*>(ub_counts), ub_args, nullptr,
                                    ub_status + row_idx, token_idx + row_idx);
                        }
                    }
                }
                if constexpr (kNumRows >= 4 and (kNumCachedDuplicates != 0 or not kHasPhysicalMap))
                    vf_check_status(ub_status, num_rows);
                notify<PIPE_V, PIPE_S>(kResultReady);
                wait<PIPE_V, PIPE_S>(kResultReady);
                if constexpr (kNumRows >= 4 and (kNumCachedDuplicates != 0 or not kHasPhysicalMap)) {
                    if (ub_status[0] >= 2) trap();
                } else for (uint32_t row_idx = 0; row_idx < num_rows; ++ row_idx) {
                    if (ub_status[row_idx] >= 2) trap();
                    if (ub_status[row_idx] & 1) {
                        const auto indices = reinterpret_cast<__ubuf__ int64_t*>(ub_selected) +
                                             row_idx * ub_layout.output_stride / sizeof(int64_t);
                        // Resolve uncached replicas in 64 bits.
                        if constexpr (kNumCachedDuplicates == 0) {
                            const uint64_t token = uint64_t(token_idx + row_idx) * 23333u + args.ep_rank;
                            for (uint32_t topk_idx = 0; topk_idx < kNumTopk + kNumSharedExperts; ++ topk_idx) {
                                const uint64_t entry = indices[topk_idx];
                                const uint32_t logical_idx = entry >> 32;
                                if (logical_idx != 0 and logical_idx != UINT32_MAX) {
                                    const uint32_t replica = token % uint32_t(ub_counts[logical_idx - 1]);
                                    indices[topk_idx] = (uint64_t(logical_idx) << 32) | replica;
                                    if constexpr (kNumTopk + kNumSharedExperts >= 2)
                                        asc_datacache_preload(
                                            reinterpret_cast<__gm__ uint64_t*>(
                                                args.to_physical_map + uint64_t(logical_idx - 1) * args.num_duplicate_experts +
                                                    replica),
                                            0);
                                }
                            }
                        }
                        for (uint32_t topk_idx = 0; topk_idx < kNumTopk + kNumSharedExperts; ++ topk_idx) {
                            const uint64_t entry = indices[topk_idx];
                            const uint32_t logical_idx = entry >> 32;
                            if (logical_idx != 0 and logical_idx != UINT32_MAX)
                                indices[topk_idx] = args.to_physical_map[
                                    uint64_t(logical_idx - 1) * args.num_duplicate_experts + uint32_t(entry)];
                        }
                    }
                }
                notify<PIPE_S, PIPE_MTE3>(kResultFull);
                wait<PIPE_S, PIPE_MTE3>(kResultFull);
                constexpr ub_ptr<int32_t> indices(kNumRows, ub_layout.output_stride / sizeof(int32_t), ub_layout.selected);
                constexpr ub_ptr<float> weights(
                    kNumRows, ub_layout.output_stride / sizeof(float), ub_layout.selected + ub_layout.weights_offset);
                copy_ub_to_gm(
                    reinterpret_cast<__gm__ int32_t*>(args.topk_idx + uint64_t(token_idx) * (kNumTopk + kNumSharedExperts)),
                    indices, num_rows, 2 * (kNumTopk + kNumSharedExperts),
                    (kNumTopk + kNumSharedExperts) * sizeof(int64_t));
                copy_ub_to_gm(args.topk_weights + uint64_t(token_idx) * (kNumTopk + kNumSharedExperts), weights,
                              num_rows, kNumTopk + kNumSharedExperts, (kNumTopk + kNumSharedExperts) * sizeof(float));
                if constexpr (kHasUnmappedOutput) {
                    constexpr ub_ptr<int32_t> logical(
                        kNumRows, ub_layout.output_stride / sizeof(int32_t), ub_layout.selected + ub_layout.logical_offset);
                    copy_ub_to_gm(
                        reinterpret_cast<__gm__ int32_t*>(args.unmapped_topk_idx +
                                                          uint64_t(token_idx) * args.unmapped_topk_idx_stride),
                        logical, num_rows, 2 * kNumTopk, args.unmapped_topk_idx_stride * sizeof(int64_t));
                }
                notify<PIPE_MTE3, PIPE_MTE2>(kScoresEmpty);
                wait<PIPE_MTE3, PIPE_MTE2>(kScoresEmpty);
            }
            if (begin_token_idx + aiv_idx * num_local_tokens >= local_end_token_idx)
                wait_intra_block(PIPE_MTE2, aiv_id, wave_idx % 2);
            set_intra_block(PIPE_V, aiv_id, wave_idx % 2);
        }
    }
}

} // namespace deep_gemm::mega_gate
