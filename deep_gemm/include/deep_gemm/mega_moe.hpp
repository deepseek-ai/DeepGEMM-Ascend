#pragma once

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/comm/mega_moe.hpp>
#include <deep_gemm/layout/mega_moe.hpp>
#include <deep_gemm/scheduler/mega_moe.hpp>
#include <deep_gemm/simd/mega_moe.hpp>

#include <c_api/asc_simd.h>
#include <c_api/cache_ctrl/cache_ctrl.h>
#include <simt_api/device_functions.h>
#include <simt_api/common_functions.h>

namespace deep_gemm::mega_moe {

using layout::BLOCK_M;
using layout::BLOCK_N;
using layout::kNumCompletionMBlocks;

// Fixed L1 blocks; choose the MAD shape from each task's actual rows.
constexpr uint32_t kNumL0Stages = 2;
constexpr uint32_t BLOCK_K = L0ASizeBytes / BLOCK_M;

template <uint32_t kMadM>
struct MMAConfig {
    static constexpr uint32_t MAD_M = kMadM;
    static constexpr uint32_t MAD_K = L0ASizeBytes / kNumL0Stages / MAD_M;
    static constexpr uint32_t MAD_N = (L0BSizeBytes / kNumL0Stages / MAD_K) < 128u ? (L0BSizeBytes / kNumL0Stages / MAD_K) : 128u;
    static_assert(MAD_M == 128 or MAD_M == BLOCK_M);
    static_assert(BLOCK_K % MAD_K == 0 and BLOCK_N % MAD_N == 0);
    static_assert(MAD_M * BLOCK_N * sizeof(float) <= L0CSizeBytes);
};

// Both shapes use two 32-KiB slots per operand, so their reuse credits remain valid.
constexpr PipeEvent<PIPE_MTE1, PIPE_M> kL0Full{0};
constexpr PipeEvent<PIPE_M, PIPE_MTE1> kL0Empty{0};

// AIC layout (KiB); shared and routed B occupy the same L1 region
// Memory  Region       Offset  Stages x size
// L1      A                 0      3 x 64
//         routed B        192      3 x 64
//         shared B        192      3 x 64
//         A SF            384      4 x 16
//         B SF            448      4 x 16
// L0A     A                 0      2 x 32
// L0B     B                 0      2 x 32
// L0C     FP32 accum        0      2 x 64 (M128), 1 x 256 (M256)

// Shared B uses A's ready/free credits on the same pipes.
constexpr uint32_t kNumL1AStages = 3, kNumL1BStages = kNumL1AStages, kNumSFStages = 4;
constexpr PipeEvent<PIPE_MTE2, PIPE_MTE1> kL1AFull{0}, kSFFull{kNumL1AStages};
constexpr PipeEvent<PIPE_MTE1, PIPE_MTE2> kL1AEmpty{0}, kSFEmpty{kNumL1AStages};
static_assert(kNumL1AStages + kNumSFStages <= 8);
// AIC/AIV0 L1B: MTE3 -> MTE1 full, MTE1 -> MTE3 empty, flags 0–2
constexpr IntraBlockEvent<0> kL1BFull{0}, kL1BEmpty{0};

// AIV0: FP4 0–3, full-tile NZ 0–1
constexpr uint32_t kNumFP4Stages = 4;
constexpr PipeEvent<PIPE_MTE2, PIPE_V> kFP4Full{0};
constexpr PipeEvent<PIPE_V, PIPE_MTE2> kFP4Empty{0};
constexpr PipeEvent<PIPE_V, PIPE_MTE3> kNZFull{0};
constexpr PipeEvent<PIPE_MTE3, PIPE_V> kNZEmpty{0};

// AIV0 metadata: V -> S completion, then S -> AIC S flag 2 (reused by routed L1B)
constexpr uint32_t kCombinePushRetiredID = 13, kPayloadRetiredID = 14;
constexpr PipeEvent<PIPE_V, PIPE_S> kMetadataDone{0};
constexpr PipeEvent<PIPE_MTE2, PIPE_V> kOffsetsFull{0};
constexpr IntraBlockEvent<0> kMetadataFull{2};

// AIV1: two 128-row epilogue buffers per Linear1 M-block
constexpr uint32_t kNumEpilogueStages = 2;
constexpr uint32_t STORE_BLOCK_M = BLOCK_M / kNumEpilogueStages;
constexpr PipeEvent<PIPE_MTE2, PIPE_V> kTopkFull{1};
constexpr PipeEvent<PIPE_V, PIPE_MTE2> kTopkEmpty{0};
constexpr PipeEvent<PIPE_V, PIPE_MTE3> kQuantFull{0};
// AIC/AIV1 Linear1: FIX -> V full, MTE3 -> FIX empty
constexpr IntraBlockEvent<1> kLinear1Full{0}, kLinear1Empty{0};
// AIC/AIV1 Linear2: FIX -> MTE3 full, MTE3 -> FIX empty
constexpr IntraBlockEvent<1> kLinear2Full{kNumEpilogueStages}, kLinear2Empty{kNumEpilogueStages};

// AIV1: pull events 0–2; push is limited by UB capacity and 32 BufferIDs, or 3 with early reduction
constexpr uint32_t kNumPullStages = 3;
constexpr uint32_t kNumCommRowsPerStep = 2;
constexpr uint32_t kNumPushStages = 32, kNumPushStagesWithEarlyReduce = 3;
constexpr PipeEvent<PIPE_MTE2, PIPE_MTE3> kPullFull{0};
constexpr PipeEvent<PIPE_MTE3, PIPE_MTE2> kPullEmpty{0};
// Peer-offset loads consumed by Scalar
constexpr PipeEvent<PIPE_MTE2, PIPE_S> kPeerOffsetsFull{0};

// AIV0/AIV1: reduce events 5–7, UB reuse, and output completion
constexpr uint32_t kNumReduceStages = 2, kMaxReduceStages = 3;
static_assert(kNumPullStages + kMaxReduceStages <= kNumPipeEvents);
constexpr PipeEvent<PIPE_MTE2, PIPE_V> kReduceFull{5};
constexpr PipeEvent<PIPE_V, PIPE_MTE3> kReduceOutputFull{5};
constexpr PipeEvent<PIPE_MTE3, PIPE_MTE2> kReduceEmpty{5};
constexpr PipeEvent<PIPE_V, PIPE_MTE2> kReduceInputsEmpty{5};
constexpr PipeEvent<PIPE_MTE3, PIPE_V> kOutputRowEmpty{5};
constexpr PipeEvent<PIPE_V, PIPE_MTE2> kEarlyReduceReadsDone{4};
constexpr PipeEvent<PIPE_MTE3, PIPE_S> kCommDone{3};
// Both AIVs notify AIC when their local workspace reads have retired
constexpr uint32_t kCleanupReadyID = 4;

// CANN 9.2 forwards mode 12 but does not name write-through-share in its C API enum
constexpr auto kWriteThroughShare = static_cast<asc_store_l2_cache_mode>(12);

template <const auto& kWorkspace>
struct CombineReduceState {
    static constexpr uint32_t kNumBitsPerWord = sizeof(uint32_t) * 8;

    // Resume at a channel boundary
    uint32_t token_idx, channel_idx = 0, stage_idx = 0;
    uint32_t num_remaining_tokens;

    bool shared_output_ready = false;
    uint32_t blocking_rank_idx = 0;
    uint64_t combine_ready_prefix[kWorkspace.config.num_ranks]{};

    // One bit per owned token, strided by the AIV count
    uint32_t reduced_token_mask[ceil_div(kWorkspace.config.num_max_tokens_per_rank, kNumAIVs * kNumBitsPerWord)]{};
};

// Use the remaining UB for reduce stages, rounded down to whole vectors
template <const auto& kWorkspace, bool kRMSNorm, typename rmsnorm_type, bool kFinalCombine = false>
__aicore__ constexpr auto make_reduce_buffer(uint32_t addr = 0, uint32_t end_addr = UBSizeBytes) {
    constexpr auto& kConfig = kWorkspace.config;
    constexpr uint32_t kNumInputs = kConfig.num_topk + (kConfig.num_shared_experts > 0);
    // Extra FP32 OFF output storage in BF16 row units
    constexpr uint32_t kNumOutputBF16Rows = not kRMSNorm and std::is_same_v<rmsnorm_type, float> ? sizeof(rmsnorm_type) / sizeof(bfloat16_t) : 0;
    constexpr uint32_t kNumBF16Rows = kNumInputs + kNumOutputBF16Rows;
    constexpr uint32_t kNumChannelsPerVector = sizeof(vector_bfloat16_t) / sizeof(bfloat16_t);
    constexpr uint32_t kReservedBytes = kRMSNorm ? (1 + kFinalCombine) * kConfig.hidden * sizeof(rmsnorm_type) + sizeof(vector_float) : 0;
    const uint32_t num_available_bytes = addr + kReservedBytes <= end_addr ? end_addr - addr - kReservedBytes : 0;
    const uint32_t num_channels = num_available_bytes / (kNumReduceStages * kNumBF16Rows * sizeof(bfloat16_t)) /
        kNumChannelsPerVector * kNumChannelsPerVector;
    return ub_ptr<bfloat16_t>(kNumBF16Rows, kConfig.hidden < num_channels ? kConfig.hidden : num_channels, addr);
}

template <const auto& kWorkspace, typename rmsnorm_type>
__aicore__ constexpr uint32_t final_reduce_stages() {
    constexpr uint32_t kEnd = UBSizeBytes - kDataBlockBytes;
    constexpr uint32_t kFit = kEnd / make_reduce_buffer<kWorkspace, false, rmsnorm_type, true>(0, kEnd).size_per_stage();
    return kFit < kMaxReduceStages ? kFit : kMaxReduceStages;
}

// True once all output chunks are issued; when kPreloadStages > 0, preloads ready routes across UB stages
template <const auto& kWorkspace, bool kRMSNorm, bool kFinalCombine = false, uint32_t kPreloadStages = 0,
          typename rmsnorm_type = bfloat16_t, typename slot_t = void>
__aicore__ inline bool advance_combine_reduce(__gm__ uint8_t* buffer, __gm__ rmsnorm_type* output,
                                              uint32_t num_tokens, CombineReduceState<kWorkspace>& state,
                                              const ub_ptr<bfloat16_t>& ub_reduce_inputs, float rms_eps,
                                              __gm__ rmsnorm_type* rms_weight = nullptr, uint32_t num_steps = ~0u,
                                              slot_t* slots = nullptr) {
    constexpr auto& kConfig = kWorkspace.config;
    constexpr bool kHasSharedExperts = kConfig.num_shared_experts > 0;
    constexpr uint32_t kNumInputs = kConfig.num_topk + kHasSharedExperts;

    constexpr ub_ptr<rmsnorm_type> ub_output_row(1, kConfig.hidden, UBSizeBytes - kConfig.hidden * sizeof(rmsnorm_type));
    // One FP32 square-sum partial per vector lane (64 lanes)
    constexpr ub_ptr<float> ub_square_sum(1, sizeof(vector_float) / sizeof(float), ub_output_row.addr - sizeof(vector_float));
    const uint32_t num_chunk_channels = ub_reduce_inputs.shape_n;
    const auto combine_input = kWorkspace.combine_input.get_ptr(buffer);
    auto& stage_idx = state.stage_idx;
    const auto emit_chunk = [&](uint32_t reduce_stage_idx, uint32_t token_idx, uint32_t channel_base,
                                uint32_t num_inputs, uint32_t num_channels) __aicore__ {
        notify<PIPE_MTE2, PIPE_V>(kReduceFull, reduce_stage_idx);
        // Accumulate in FP32; keep widening stores separate from unread inputs
        const auto ub_inputs = ub_reduce_inputs[reduce_stage_idx].ptr();
        __ubuf__ rmsnorm_type* ub_output;
        if constexpr (kRMSNorm)
            ub_output = ub_output_row.ptr() + channel_base;
        else if constexpr (std::is_same_v<rmsnorm_type, float>)
            ub_output = reinterpret_cast<__ubuf__ rmsnorm_type*>(ub_inputs + kNumInputs * num_chunk_channels);
        else
            ub_output = reinterpret_cast<__ubuf__ rmsnorm_type*>(ub_inputs);
        wait<PIPE_MTE2, PIPE_V>(kReduceFull, reduce_stage_idx);
        if constexpr (kRMSNorm) {
            if (channel_base == 0)
                wait<PIPE_MTE3, PIPE_V>(kOutputRowEmpty);
        }
        vf_combine_reduce<kRMSNorm>(ub_inputs, ub_output, num_inputs, num_channels, ub_square_sum.ptr(), channel_base == 0);
        const auto output_row = output + static_cast<uint64_t>(token_idx) * kConfig.hidden;
        if constexpr (kRMSNorm) {
            notify<PIPE_V, PIPE_MTE2>(kReduceInputsEmpty, reduce_stage_idx);
            if (channel_base + num_channels != kConfig.hidden)
                return;
            const uint32_t weight_addr = kFinalCombine ? ub_square_sum.addr - kConfig.hidden * sizeof(rmsnorm_type) : ub_reduce_inputs.addr;
            const ub_ptr<rmsnorm_type> ub_weight(1, kConfig.hidden, weight_addr);
            if constexpr (not kFinalCombine) {
                // Retire both input readers before their storage becomes the weight row.
                wait<PIPE_V, PIPE_MTE2, kNumReduceStages>(kReduceInputsEmpty);
                copy_gm_to_ub(ub_weight, rms_weight, kConfig.hidden * sizeof(rmsnorm_type));
                notify<PIPE_MTE2, PIPE_V>(kReduceFull, reduce_stage_idx);
                wait<PIPE_MTE2, PIPE_V>(kReduceFull, reduce_stage_idx);
            }
            vf_rms_norm<kConfig.hidden>(ub_output_row.ptr(), ub_weight.ptr(), rms_eps, ub_square_sum.ptr());
            if constexpr (not kFinalCombine)
                notify<PIPE_V, PIPE_MTE2, kNumReduceStages>(kReduceInputsEmpty);
        }
        notify<PIPE_V, PIPE_MTE3>(kReduceOutputFull, reduce_stage_idx);
        wait<PIPE_V, PIPE_MTE3>(kReduceOutputFull, reduce_stage_idx);
        copy_ub_to_gm(output_row + (kRMSNorm ? 0 : channel_base),
                      kRMSNorm ? ub_output_row.ptr() : ub_output, 1,
                      kRMSNorm ? kConfig.hidden : num_channels,
                      (kRMSNorm ? kConfig.hidden : num_channels) * sizeof(rmsnorm_type), asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM);
        if constexpr (kRMSNorm)
            notify<PIPE_MTE3, PIPE_V>(kOutputRowEmpty);
        else
            notify<PIPE_MTE3, PIPE_MTE2>(kReduceEmpty, reduce_stage_idx);
    };
    const auto is_route_ready = [&](uint32_t dst_metadata) __aicore__ {
        const uint32_t dst_rank_idx = dst_metadata >> 24, dst_m_block_idx = dst_metadata & 0x00ffffffu;
        auto& ready_prefix = state.combine_ready_prefix[dst_rank_idx];
        if (dst_m_block_idx >= ready_prefix)
            ready_prefix = asc_load_dev(kWorkspace.combine_ready_prefix.get_ptr(buffer, dst_rank_idx));
        if (dst_m_block_idx < ready_prefix)
            return true;
        state.blocking_rank_idx = dst_rank_idx;
        return false;
    };

    // Check shared Linear2 completion before reducing tokens
    if (kHasSharedExperts and state.num_remaining_tokens > 0 and not state.shared_output_ready)
        state.shared_output_ready = comm::is_block_ready<kWorkspace>(buffer, layout::kLinear2Ready, 0,
            min(ceil_div(num_tokens, BLOCK_M) * kConfig.get_num_linear2_n_blocks(), kNumAICores));

    if constexpr (kPreloadStages > 0) {
        static_assert(kNumInputs < 32, "Preload bitmask requires fewer than 32 inputs");
        for (bool emitted = true; emitted and state.num_remaining_tokens > 0;) {
            emitted = false;
            #pragma unroll
            for (uint32_t s = 0; s < kPreloadStages; ++ s) {
                auto& slot = slots[s];
                if (slot.tok == ~0u and state.token_idx < num_tokens) {
                    wait<PIPE_MTE3, PIPE_MTE2>(kReduceEmpty, s);
                    uint32_t valid = uint32_t{kHasSharedExperts} << kConfig.num_topk;
                    for (uint32_t i = 0; i < kConfig.num_topk; ++ i)
                        if ((slot.meta[i] = asc_load_dev(kWorkspace.token_dst_metadata.get_ptr(buffer, state.token_idx * kConfig.num_topk + i))) != ~uint32_t{0})
                            valid |= 1u << i;
                    slot.tok = state.token_idx, slot.valid = valid, slot.pending = valid;
                    state.token_idx += kNumAIVs;
                }
                for (uint32_t i = kNumInputs; slot.pending and i -- > 0;) {
                    if ((slot.pending & (1u << i)) and (i == kConfig.num_topk ? state.shared_output_ready : is_route_ready(slot.meta[i]))) {
                        copy_gm_to_ub(ub_reduce_inputs[s].ptr() + static_cast<uint32_t>(asc_popc(slot.valid & ((1u << i) - 1u))) * kConfig.hidden,
                                      combine_input + kConfig.get_combine_m_idx(i, slot.tok) * kConfig.hidden,
                                      1, kConfig.hidden, kConfig.hidden * sizeof(bfloat16_t), asc_load_l2_cache_mode::NOTALLOC_DROP);
                        slot.pending &= ~(1u << i);
                    }
                }
                if (slot.tok != ~0u and slot.pending == 0) {
                    emit_chunk(s, slot.tok, 0, static_cast<uint32_t>(asc_popc(slot.valid)), kConfig.hidden);
                    slot.tok = ~0u, -- state.num_remaining_tokens, emitted = true;
                }
            }
        }
        return state.num_remaining_tokens == 0;
    }

    if (kHasSharedExperts and state.num_remaining_tokens > 0 and not state.shared_output_ready)
        return false;

    while (state.num_remaining_tokens > 0) {
        const uint32_t token_idx = state.token_idx;
        const uint32_t next_token_idx = token_idx + kNumAIVs < num_tokens ? token_idx + kNumAIVs : token_idx % kNumAIVs;
        const uint32_t word_idx = token_idx / (kNumAIVs * state.kNumBitsPerWord);
        const uint32_t token_bit = 1u << (token_idx / kNumAIVs % state.kNumBitsPerWord);

        // Skip completed bits; a full uint32_t word has its first zero at bit 32
        const uint32_t completed_mask = state.reduced_token_mask[word_idx];
        if (completed_mask & token_bit) {
            const uint32_t next_bit_idx = asc_ffz(static_cast<uint64_t>(completed_mask | (token_bit - 1u)));
            const uint32_t next_pending_token_idx = (word_idx * state.kNumBitsPerWord + next_bit_idx) * kNumAIVs + token_idx % kNumAIVs;
            state.token_idx = next_pending_token_idx < num_tokens ? next_pending_token_idx : token_idx % kNumAIVs;
            continue;
        }

        // Reuse first-chunk validity checks when loading the inputs.
        bool all_routes_valid = state.channel_idx == 0;
        if (state.channel_idx == 0) {
            uint64_t metadata = 0;
            for (uint32_t topk_slot_idx = 0; topk_slot_idx < kConfig.num_topk; ++ topk_slot_idx) {
                const uint32_t token_topk_idx = token_idx * kConfig.num_topk + topk_slot_idx;
                if (topk_slot_idx == 0 or token_topk_idx % 2 == 0)
                    metadata = asc_load_dev(kWorkspace.token_dst_metadata.template get_ptr<uint64_t>(buffer, token_topk_idx & ~1u));
                const uint32_t dst_metadata = static_cast<uint32_t>(metadata >> (32 * (token_topk_idx % 2)));
                if (dst_metadata == ~uint32_t{0}) {
                    all_routes_valid = false;
                } else if (not is_route_ready(dst_metadata)) {
                    state.token_idx = next_token_idx;
                    return false;
                }
            }
        }

        for (uint32_t channel_base = state.channel_idx; channel_base < kConfig.hidden; channel_base += num_chunk_channels) {
            const uint32_t num_channels = min(kConfig.hidden - channel_base, num_chunk_channels);
            if constexpr (kRMSNorm)
                wait<PIPE_V, PIPE_MTE2>(kReduceInputsEmpty, stage_idx);
            else
                wait<PIPE_MTE3, PIPE_MTE2>(kReduceEmpty, stage_idx);

            // Pack valid routed inputs, then shared output, preserving accumulation order.
            uint32_t num_inputs = 0;
            for (uint32_t topk_slot_idx = 0; topk_slot_idx < kNumInputs; ++ topk_slot_idx) {
                if (not all_routes_valid and topk_slot_idx < kConfig.num_topk and asc_load_dev(kWorkspace.token_dst_metadata.get_ptr(
                        buffer, token_idx * kConfig.num_topk + topk_slot_idx)) == ~uint32_t{0})
                    continue;
                const auto src = combine_input + kConfig.get_combine_m_idx(topk_slot_idx, token_idx) * kConfig.hidden + channel_base;
                copy_gm_to_ub(ub_reduce_inputs[stage_idx].ptr() + num_inputs * num_channels,
                              src, 1, num_channels, num_channels * sizeof(bfloat16_t), asc_load_l2_cache_mode::NOTALLOC_KEEP);
                ++ num_inputs;
            }
            emit_chunk(stage_idx, token_idx, channel_base, num_inputs, num_channels);
            stage_idx = (stage_idx + 1) % kNumReduceStages;

            // Save the resume position
            state.channel_idx = channel_base + num_channels;
            if (state.channel_idx == kConfig.hidden) {
                state.channel_idx = 0;
                state.token_idx = next_token_idx;
                state.reduced_token_mask[word_idx] |= token_bit;
                -- state.num_remaining_tokens;
            }

            if (-- num_steps == 0)
                return state.num_remaining_tokens == 0;
        }
    }
    return true;
}

template <const auto& kWorkspace, bool kLowLatency, bool kRMSNorm, typename rmsnorm_type,
          typename epilogue_fn_t = decltype([]() __aicore__ { return true; })>
__aicore__ __forceinline__ void combine_reduce(__gm__ uint8_t* buffer, __gm__ rmsnorm_type* output,
                                               uint32_t num_tokens, uint32_t rank_idx, uint32_t aiv_idx,
                                               CombineReduceState<kWorkspace>& reduce, __gm__ rmsnorm_type* rms_weight,
                                               float rms_eps, const epilogue_fn_t& poll_epilogue = {}) {
    constexpr auto& kConfig = kWorkspace.config;
    if constexpr (not kLowLatency or kRMSNorm) {
        notify<PIPE_MTE3, PIPE_S>(kCommDone);
        wait<PIPE_MTE3, PIPE_S>(kCommDone);
    }
    if constexpr (kRMSNorm) {
        notify<PIPE_V, PIPE_MTE2>(kEarlyReduceReadsDone);
        wait<PIPE_V, PIPE_MTE2>(kEarlyReduceReadsDone);
    }

    constexpr uint32_t kEnd = UBSizeBytes - (kLowLatency and not kRMSNorm ? kDataBlockBytes : 0);
    constexpr auto ub_reduce_inputs = make_reduce_buffer<kWorkspace, kRMSNorm, rmsnorm_type, true>(0, kEnd);
    static_assert(ub_reduce_inputs.shape_n > 0, "Combine UB overflow");
    if constexpr (kRMSNorm) {
        constexpr uint32_t kWeightBytes = kConfig.hidden * sizeof(rmsnorm_type);
        constexpr ub_ptr<rmsnorm_type> ub_weight(1, kConfig.hidden, UBSizeBytes - 2 * kWeightBytes - sizeof(vector_float));
        copy_gm_to_ub(ub_weight, rms_weight, kWeightBytes);
        notify<PIPE_MTE2, PIPE_V>(kReduceFull);
        wait<PIPE_MTE2, PIPE_V>(kReduceFull);
    }
    constexpr uint32_t kPreloadStages = not kLowLatency or kRMSNorm or
        kConfig.num_topk + (kConfig.num_shared_experts > 0) >= 32 or ub_reduce_inputs.shape_n < kConfig.hidden ?
        0 : final_reduce_stages<kWorkspace, rmsnorm_type>();
    struct { uint32_t tok = ~0u, valid = 0, pending = 0, meta[kConfig.num_topk]{}; } slots[kPreloadStages ? kPreloadStages : 1]{};
    comm::wait_until([&]() __aicore__ {
        const uint32_t prev_rem = reduce.num_remaining_tokens;
        const bool reduce_done = advance_combine_reduce<kWorkspace, kRMSNorm, true, kPreloadStages>(
            buffer, output, num_tokens, reduce, ub_reduce_inputs, rms_eps, rms_weight, ~0u, slots);
        if (poll_epilogue() and reduce_done)
            return true;
        if constexpr (kLowLatency) {
            if (reduce.num_remaining_tokens == prev_rem) {
                comm::backoff();
            }
        }
        return false;
    }, [&]() __aicore__ {
        if (kConfig.num_shared_experts > 0 and not reduce.shared_output_ready) {
            comm::print_local_timeout(rank_idx);
            return;
        }
        // This is the last observed blocker; token_idx already points to the next probe
        const uint32_t peer_idx = reduce.blocking_rank_idx;
        comm::print_peer_timeout(rank_idx, aiv_idx, peer_idx,
            asc_load_dev(kWorkspace.combine_ready_prefix.get_ptr(buffer, peer_idx)), false);
    });

    // Empty ranks also wait for peer writes before the next invocation reuses scratch
    for (uint32_t peer_idx = aiv_idx; peer_idx < kConfig.num_ranks; peer_idx += kNumAIVs) {
        const auto ready_prefix_ptr = kWorkspace.combine_ready_prefix.get_ptr(buffer, peer_idx);
        comm::wait_until([&]() __aicore__ {
            if (asc_load_dev(ready_prefix_ptr) == ~uint64_t{0})
                return true;
            comm::backoff();
            return false;
        }, [&]() __aicore__ {
            comm::print_peer_timeout(rank_idx, aiv_idx, peer_idx, asc_load_dev(ready_prefix_ptr), true);
        });
    }
}

// AIV0: FP4 weights -> FP8 NZ -> L1B
template <const auto& kWorkspace, bool kLowLatency, bool kRMSNorm, typename rmsnorm_type>
__aicore__ inline void aiv0_weight_pipeline(__gm__ uint8_t* buffer, __gm__ rmsnorm_type* output,
                                            uint32_t num_tokens, CombineReduceState<kWorkspace>& reduce, uint32_t num_routed_m_blocks,
                                            const layout::Weights& weights, uint32_t core_idx, __gm__ rmsnorm_type* rms_weight, float rms_eps) {
    constexpr auto& kConfig = kWorkspace.config;

    constexpr uint32_t FRAC_K = get_frac_k<uint8_t>();
    constexpr l1_ptr<float8_e4m3_t, Major::K> l1b(BLOCK_N, BLOCK_K, kNumL1AStages * L0ASizeBytes);

    // Weight staging; final reduction reuses UB after MTE3 drains
    constexpr ub_ptr<uint8_t> ub_fp4(1, L0BSizeBytes / 2);
    constexpr ub_ptr<uint8_t> ub_nz(1, L0BSizeBytes, ub_fp4.offset(kNumFP4Stages));
    // Large-batch streaming mode (!kLowLatency) keeps 1 NZ slot to reserve UB for early combine reduction
    constexpr uint32_t kNumNZStages = kLowLatency ? 2 : 1;
    constexpr auto ub_reduce_inputs = make_reduce_buffer<kWorkspace, kRMSNorm, rmsnorm_type>(ub_nz.offset(1));
    static_assert(ub_nz.offset(kNumNZStages) <= UBSizeBytes and ub_reduce_inputs.shape_n > 0, "AIV0 UB overflow");

    if constexpr (not kLowLatency and kRMSNorm)
        static_assert(ub_reduce_inputs.addr + kConfig.hidden * (2 * sizeof(rmsnorm_type)) + sizeof(vector_float) <= UBSizeBytes);

    // Initialize UB reuse credits; AIC releases the initial L1B credits
    notify<PIPE_V, PIPE_MTE2, kNumFP4Stages>(kFP4Empty);
    notify<PIPE_MTE3, PIPE_V, kNumNZStages>(kNZEmpty);

    sched::LinearTaskState schedule{{0, 0, ~0u, ~0u}};
    sched::ExpertCursor<kWorkspace, kLowLatency> expert_cursors[2]{};
    uint32_t fp4_stage_idx = 0, nz_stage_idx = 0;
    uint32_t l1b_stage_idx = 0;

    while (true) {
        sched::BlockPhase block_phase;
        const auto task_idx = sched::take_linear_task<kWorkspace, kLowLatency>(num_tokens, num_routed_m_blocks, schedule, core_idx, block_phase);
        if (block_phase == sched::BlockPhase::kDone)
            break;

        // Select task weights
        const bool is_linear1 = block_phase == sched::BlockPhase::kLinear1;
        const uint32_t num_n_blocks = is_linear1 ? kConfig.get_num_linear1_n_blocks() : kConfig.get_num_linear2_n_blocks();
        const uint32_t num_k = is_linear1 ? kConfig.hidden : kConfig.intermediate_hidden;
        const uint32_t pool_block_idx = task_idx / num_n_blocks;
        const uint32_t n_block_idx = task_idx % num_n_blocks;
        const uint32_t layer_idx = is_linear1 ? 0 : 1;
        expert_cursors[layer_idx].advance_to(buffer, pool_block_idx);
        const auto weight_l2_hint = pool_block_idx + 1 < expert_cursors[layer_idx].pool_block_end ?
            asc_load_l2_cache_mode::NORMAL_LAST_VICTIM : (expert_cursors[layer_idx].num_expert_m_blocks > 1 ?
                asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM : asc_load_l2_cache_mode::NOTALLOC_KEEP);
        const uint64_t weight_block_idx =
            static_cast<uint64_t>(expert_cursors[layer_idx].local_expert_idx) * num_n_blocks + n_block_idx;
        const auto weight_data = (is_linear1 ? weights.l1.data : weights.l2.data) +
                                 weight_block_idx * BLOCK_N * (num_k / 2);

        for (uint32_t k_idx = 0; k_idx < num_k; k_idx += BLOCK_K) {
            // valid_k remains a multiple of MAD_K, including the final K block
            const uint32_t valid_k = min(BLOCK_K, num_k - k_idx);

            // Load packed FP4
            wait<PIPE_V, PIPE_MTE2>(kFP4Empty, fp4_stage_idx);
            copy_gm_to_ub(ub_fp4[fp4_stage_idx],
                          const_cast<__gm__ uint8_t*>(weight_data + static_cast<uint64_t>(k_idx) * BLOCK_N / 2),
                          valid_k * BLOCK_N / 2, weight_l2_hint);
            notify<PIPE_MTE2, PIPE_V>(kFP4Full, fp4_stage_idx);

            // Expand FP4 to FP8 NZ at 1/64 scale; weight SFs compensate
            wait<PIPE_MTE2, PIPE_V>(kFP4Full, fp4_stage_idx);
            wait<PIPE_MTE3, PIPE_V>(kNZEmpty, nz_stage_idx);
            vf_cast_fp4_to_fp8_scaled<BLOCK_N * BLOCK_K>(ub_fp4[fp4_stage_idx].ptr(), ub_nz[nz_stage_idx].ptr());
            notify<PIPE_V, PIPE_MTE2>(kFP4Empty, fp4_stage_idx);
            notify<PIPE_V, PIPE_MTE3>(kNZFull, nz_stage_idx);

            // Publish L1B to AIC
            wait<PIPE_V, PIPE_MTE3>(kNZFull, nz_stage_idx);
            wait_intra_block<PIPE_MTE3, 0>(kL1BEmpty, l1b_stage_idx);
            asc_copy_ub2l1(l1b[l1b_stage_idx].template ptr<uint8_t>(), ub_nz[nz_stage_idx].ptr(), BLOCK_K / FRAC_K, BLOCK_N, 0, 0);
            notify_intra_block<PIPE_MTE3, 0>(kL1BFull, l1b_stage_idx);
            notify<PIPE_MTE3, PIPE_V>(kNZEmpty, nz_stage_idx);
            fp4_stage_idx = (fp4_stage_idx + 1) % kNumFP4Stages;
            nz_stage_idx = (nz_stage_idx + 1) % kNumNZStages;
            l1b_stage_idx = (l1b_stage_idx + 1) % kNumL1BStages;
            if constexpr (not kLowLatency)
                advance_combine_reduce<kWorkspace, kRMSNorm>(buffer, output, num_tokens, reduce, ub_reduce_inputs, rms_eps, rms_weight, 1);
        }
    }
}

// AIC: shared and routed GEMMs
template <const auto& kWorkspace, bool kLowLatency>
__aicore__ inline void aic_gemm_pipeline(__gm__ uint8_t* buffer, uint32_t num_tokens, uint32_t rank_idx,
                                         const layout::Weights& weights, uint32_t core_idx) {
    constexpr auto& kConfig = kWorkspace.config;
    constexpr bool kHasSharedExperts = kConfig.num_shared_experts > 0;
    constexpr l1_ptr<float8_e4m3_t, Major::K> l1a(BLOCK_M, BLOCK_K, 0);
    constexpr l1_ptr<float8_e4m3_t, Major::K> l1b(BLOCK_N, BLOCK_K, l1a.offset(kNumL1AStages));
    constexpr uint32_t kNumSFPairsPerChunk = (L1SizeBytes - l1b.offset(kNumL1BStages)) / (kNumSFStages * (BLOCK_M + BLOCK_N) * sizeof(int16_t));
    constexpr l1_ptr<int16_t> l1_sfa(BLOCK_M, kNumSFPairsPerChunk, l1b.offset(kNumL1BStages));
    constexpr l1_ptr<int16_t> l1_sfb(BLOCK_N, kNumSFPairsPerChunk, l1_sfa.offset(kNumSFStages));
    // FIX destination in AIV1 UB, matching aiv1_epilogue_comm
    constexpr ub_ptr<bfloat16_t> ub_accum(STORE_BLOCK_M, BLOCK_N);
    static_assert(kNumSFPairsPerChunk > 0 and kNumSFPairsPerChunk % (BLOCK_K / MX_SF_DIVISOR) == 0);
    static_assert(l1_sfb.offset(kNumSFStages) <= L1SizeBytes, "L1 overflow");

    const auto num_issued_gemms_ptr = kWorkspace.progress.num_issued_gemms.get_ptr(buffer, core_idx);
    const uint32_t num_shared_m_blocks = kHasSharedExperts ? ceil_div(num_tokens, BLOCK_M) : 0;
    const bool reuse_shared_weights = num_shared_m_blocks > 1;
    const auto routed_output = kWorkspace.routed.output.get_ptr(buffer);
    const auto combine_input = kWorkspace.combine_input.get_ptr(buffer);

    // Initialize reuse credits
    asc_set_mmad_direction_n();
    notify<PIPE_M, PIPE_MTE1, kNumL0Stages>(kL0Empty);
    notify<PIPE_MTE1, PIPE_MTE2, kNumL1AStages>(kL1AEmpty);
    notify<PIPE_MTE1, PIPE_MTE2, kNumSFStages>(kSFEmpty);

    sched::LinearTaskState schedule;
    sched::ExpertCursor<kWorkspace, kLowLatency> expert_cursors[2]{};
    uint32_t l1_stage_idx = 0;
    uint32_t sf_stage_idx = 0;
    uint32_t num_issued_gemms = 0;
    bool routed_l1b_initialized = false;
    uint32_t l0a_stage_idx = 0, l0b_stage_idx = 0;
    bool wait_before_l0a_load = true;
    uint32_t num_routed_m_blocks = 0;  // Routed pool only; shared blocks are separate
    bool metadata_ready = false;
    const auto routed_m_block_count_ptr = kWorkspace.routed_m_block_count.get_ptr(buffer);

    while (true) {
        sched::BlockPhase block_phase;
        const auto task_idx = sched::take_linear_task<kWorkspace, kLowLatency>(
            num_tokens, num_routed_m_blocks, schedule, core_idx, block_phase, metadata_ready);
        if (block_phase == sched::BlockPhase::kDone)
            break;
        if (block_phase == sched::BlockPhase::kNone) {
            // Consume metadata before routed L1B reuses the same intra-block event
            wait_intra_block<PIPE_S, 0>(kMetadataFull);
            num_routed_m_blocks = asc_load_dev(routed_m_block_count_ptr) & layout::kCountValueMask;
            metadata_ready = true;
            continue;
        }

        // Decode the GEMM task
        const bool is_shared = block_phase == sched::BlockPhase::kSharedLinear1 or block_phase == sched::BlockPhase::kSharedLinear2;
        const bool is_linear1 = block_phase == sched::BlockPhase::kLinear1 or block_phase == sched::BlockPhase::kSharedLinear1;

        // Shared experts widen Linear1 N and Linear2 K
        const uint32_t intermediate_hidden = is_shared ? kConfig.shared_intermediate_hidden : kConfig.intermediate_hidden;
        const uint32_t num_linear1_n_blocks = intermediate_hidden / (BLOCK_N / 2);
        const uint32_t num_n_blocks = is_linear1 ? num_linear1_n_blocks : kConfig.hidden / BLOCK_N;
        const uint32_t num_k = is_linear1 ? kConfig.hidden : intermediate_hidden;
        const uint32_t num_sf_pairs = num_k / MX_SF_DIVISOR;
        const uint32_t pool_block_idx = task_idx / num_n_blocks;
        const uint32_t n_block_idx = task_idx % num_n_blocks;
        const uint32_t layer_idx = is_linear1 ? 0 : 1;
        const uint32_t valid_m = is_shared ? min(num_tokens - pool_block_idx * BLOCK_M, BLOCK_M) :
            expert_cursors[layer_idx].advance_to(buffer, pool_block_idx);
        const auto weight_sf_l2_hint = (is_shared ? reuse_shared_weights : pool_block_idx + 1 < expert_cursors[layer_idx].pool_block_end) ?
            asc_load_l2_cache_mode::NORMAL_LAST_VICTIM : asc_load_l2_cache_mode::NOTALLOC_KEEP;
        const uint32_t aligned_valid_m = aligned(valid_m, FRAC_MN);
        const uint64_t weight_block_idx = is_shared ? n_block_idx :
            static_cast<uint64_t>(expert_cursors[layer_idx].local_expert_idx) * num_n_blocks + n_block_idx;
        const uint64_t sf_offset = weight_block_idx * num_sf_pairs * BLOCK_N;
        const auto task_weights = is_linear1 ? (is_shared ? weights.shared_l1 : weights.l1) : (is_shared ? weights.shared_l2 : weights.l2);

        // Hand L1B over to AIV0 after shared tasks
        if (not is_shared and not routed_l1b_initialized) {
            l1_stage_idx = 0;
            notify_intra_block<PIPE_MTE1, 0, kNumL1BStages>(kL1BEmpty);
            routed_l1b_initialized = true;
        }

        // Select input data and scales
        const uint32_t input_block_idx = is_shared ? pool_block_idx : kConfig.get_ring_block_idx(pool_block_idx);
        const auto input_layout = is_linear1 ?
            (is_shared ? kWorkspace.input.acts : kWorkspace.routed.l1_acts) :
            (is_shared ? kWorkspace.shared_l2_acts : kWorkspace.routed.l2_acts);
        const uint64_t input_m_idx = static_cast<uint64_t>(input_block_idx) * BLOCK_M;
        const auto input = input_layout.data.get_ptr(buffer, input_m_idx * num_k);
        const auto input_sf = input_layout.sf.get_ptr(buffer, input_m_idx * num_sf_pairs);

        const auto compute = [&]<uint32_t MAD_M>() __aicore__ {
            constexpr uint32_t MAD_N = MMAConfig<MAD_M>::MAD_N, MAD_K = MMAConfig<MAD_M>::MAD_K;
            constexpr l0a_ptr<float8_e4m3_t> l0a(MAD_M, MAD_K);
            constexpr l0b_ptr<float8_e4m3_t> l0b(MAD_N, MAD_K);
            constexpr l0c_ptr<float> l0c(MAD_M, MAD_N);
            for (uint32_t k_idx = 0; k_idx < num_k; k_idx += BLOCK_K) {
                // valid_k remains a multiple of MAD_K, including the final K block
                const uint32_t valid_k = min(BLOCK_K, num_k - k_idx);
                const uint32_t sf_pair_idx = k_idx / MX_SF_DIVISOR;
                const uint32_t num_chunk_pairs = min(kNumSFPairsPerChunk, num_sf_pairs - sf_pair_idx);

                // Load shared FP8 weights
                if (is_shared) {
                    wait<PIPE_MTE1, PIPE_MTE2>(kL1AEmpty, l1_stage_idx);
                    const uint64_t weight_offset = weight_block_idx * BLOCK_N * num_k;
                    asc_set_gm2l1_loop_size(1, 1);
                    asc_copy_gm2l1_align(l1b[l1_stage_idx].template ptr<uint8_t>(),
                                         const_cast<__gm__ uint8_t*>(task_weights.data + weight_offset + k_idx * BLOCK_N),
                                         1, BLOCK_N * valid_k, 0, 0, false,
                                         reuse_shared_weights ? asc_load_l2_cache_mode::NORMAL_LAST_VICTIM : asc_load_l2_cache_mode::NOTALLOC_KEEP,
                                         BLOCK_N * BLOCK_K, BLOCK_N * BLOCK_K);
                }

                // Load weight scales once per SF chunk
                if (sf_pair_idx % kNumSFPairsPerChunk == 0) {
                    wait<PIPE_MTE1, PIPE_MTE2>(kSFEmpty, sf_stage_idx);
                    copy_gm_to_l1_mx(l1_sfb[sf_stage_idx],
                                     gm_ptr<int16_t, Major::MN>(BLOCK_N, reinterpret_cast<uintptr_t>(task_weights.sf + sf_offset + sf_pair_idx * BLOCK_N)),
                                     BLOCK_N, num_chunk_pairs, weight_sf_l2_hint);
                }

                // Prefetch independent weights before waiting for activations
                if (k_idx == 0 and (not is_shared or not is_linear1))
                    comm::wait_for_block<kWorkspace>(buffer, rank_idx,
                                                     is_linear1 ? layout::kDispatched : layout::kLinear1Ready,
                                                     (is_shared ? 0 : num_shared_m_blocks) + pool_block_idx,
                                                     is_linear1 ? kNumAICores : num_linear1_n_blocks);

                // Load activations and scales
                if (not is_shared)
                    wait<PIPE_MTE1, PIPE_MTE2>(kL1AEmpty, l1_stage_idx);
                copy_gm_to_l1(l1a[l1_stage_idx],
                              gm_ptr<float8_e4m3_t, Major::K>(num_k, reinterpret_cast<uintptr_t>(input + k_idx)),
                              valid_m, valid_k, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                notify<PIPE_MTE2, PIPE_MTE1>(kL1AFull, l1_stage_idx);

                if (sf_pair_idx % kNumSFPairsPerChunk == 0) {
                    if (is_linear1) {
                        copy_gm_to_l1_mx(l1_sfa[sf_stage_idx],
                                         gm_ptr<int16_t, Major::K>(num_sf_pairs, reinterpret_cast<uintptr_t>(input_sf + sf_pair_idx)),
                                         valid_m, num_chunk_pairs, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                    } else {
                        copy_gm_to_l1_mx(l1_sfa[sf_stage_idx],
                                         gm_ptr<int16_t, Major::MN>(BLOCK_M, reinterpret_cast<uintptr_t>(input_sf + sf_pair_idx * BLOCK_M)),
                                         valid_m, num_chunk_pairs, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                    }

                    // One completion covers both scale loads on the same MTE2 queue
                    notify<PIPE_MTE2, PIPE_MTE1>(kSFFull, sf_stage_idx);
                }

                // Wait for L1 inputs
                wait<PIPE_MTE2, PIPE_MTE1>(kL1AFull, l1_stage_idx);
                if (not is_shared)
                    wait_intra_block<PIPE_MTE1, 0>(kL1BFull, l1_stage_idx);

                // Double-buffer A and B across K
                for (uint32_t k_mad_idx = 0; k_mad_idx < valid_k; k_mad_idx += MAD_K) {
                    const uint32_t sf_pair_idx_in_chunk = (sf_pair_idx + k_mad_idx / MX_SF_DIVISOR) % kNumSFPairsPerChunk;
                    const auto l0a_mad = l0a[l0a_stage_idx].as_mad_aligned(aligned_valid_m, MAD_K);
                    // A reuse is covered by the preceding split-N B wait, or this B wait.
                    if (wait_before_l0a_load)
                        wait<PIPE_M, PIPE_MTE1>(kL0Empty, l0b_stage_idx);
                    copy_l1_to_l0a(l0a_mad, l1a[l1_stage_idx], 0, k_mad_idx);
                    if (sf_pair_idx % kNumSFPairsPerChunk == 0 and k_mad_idx == 0)
                        wait<PIPE_MTE2, PIPE_MTE1>(kSFFull, sf_stage_idx);
                    copy_l1_to_l0a_mx(l0a_mad, l1_sfa[sf_stage_idx], 0, sf_pair_idx_in_chunk, MAD_K);
                    const bool is_last_k = k_idx + k_mad_idx + MAD_K == num_k;

                    for (uint32_t n_idx = 0; n_idx < BLOCK_N; n_idx += MAD_N) {
                        const auto l0c_mad = l0c[n_idx / MAD_N].as_mad_aligned(aligned_valid_m, MAD_N);
                        if (n_idx != 0 or not wait_before_l0a_load)
                            wait<PIPE_M, PIPE_MTE1>(kL0Empty, l0b_stage_idx);
                        const auto l0b_stage = l0b_ptr<float8_e4m3_t>(MAD_N, MAD_K, l0b_stage_idx * (L0BSizeBytes / kNumL0Stages));
                        copy_l1_to_l0b(l0b_stage, l1b[l1_stage_idx], n_idx, k_mad_idx);
                        copy_l1_to_l0b_mx(l0b_stage, l1_sfb[sf_stage_idx], n_idx, sf_pair_idx_in_chunk, MAD_K);
                        // B readiness also covers the earlier A/scales loads on the same MTE1 queue
                        notify<PIPE_MTE1, PIPE_M>(kL0Full, l0b_stage_idx);
                        wait<PIPE_MTE1, PIPE_M>(kL0Full, l0b_stage_idx);
                        mad(l0c_mad, l0a_mad, l0b_stage, aligned_valid_m, MAD_N, MAD_K,
                            is_last_k ? asc_unit_flag_mode::ENABLE_UPDATE : asc_unit_flag_mode::ENABLE_KEEP,
                            k_idx + k_mad_idx == 0);
                        notify<PIPE_M, PIPE_MTE1>(kL0Empty, l0b_stage_idx);
                        l0b_stage_idx = (l0b_stage_idx + 1) % kNumL0Stages;
                        if (not is_last_k)
                            continue;

                        if (n_idx == 0 and block_phase == sched::BlockPhase::kLinear2 and pool_block_idx >= kConfig.get_num_ring_blocks()) {
                            const uint32_t block_end = ceil_div(pool_block_idx - kConfig.get_num_ring_blocks() + 1, kNumCompletionMBlocks) * kNumCompletionMBlocks;
                            comm::wait_for_block<kWorkspace>(buffer, rank_idx, layout::kCombineSent,
                                                             num_shared_m_blocks + block_end - 1, kNumAICores);
                        }
                        // One epilogue credit covers BLOCK_N columns
                        const uint32_t fix_step_m = is_linear1 ? STORE_BLOCK_M : valid_m;
                        for (uint32_t m_idx_in_block = 0; m_idx_in_block < valid_m; m_idx_in_block += fix_step_m) {
                            const uint32_t fix_stage_idx = is_linear1 ? m_idx_in_block / STORE_BLOCK_M : num_issued_gemms % kNumEpilogueStages;
                            if (n_idx == 0)
                                wait_intra_block<PIPE_FIX, 1>(is_linear1 ? kLinear1Empty : kLinear2Empty, fix_stage_idx);
                            if (is_linear1) {
                                copy_l0c_to_ub(ub_ptr<bfloat16_t>(STORE_BLOCK_M, BLOCK_N, ub_accum[fix_stage_idx].addr + n_idx * sizeof(bfloat16_t)), l0c_mad.offset_nz_m(m_idx_in_block),
                                               min(STORE_BLOCK_M, valid_m - m_idx_in_block), MAD_N, asc_dual_dst_mode::DUAL_DST_DISABLE, 1);
                            } else {
                                // Store shared output to combine; routed output to the ring
                                const auto output_base = is_shared ? combine_input : routed_output;
                                const uint64_t output_m_idx = is_shared ?
                                    kConfig.get_combine_m_idx(kConfig.num_topk, static_cast<uint64_t>(pool_block_idx) * BLOCK_M) :
                                    kConfig.get_ring_m_idx(pool_block_idx);
                                copy_l0c_to_gm(output_base + output_m_idx * kConfig.hidden + n_block_idx * BLOCK_N + n_idx, l0c_mad, valid_m, MAD_N, kConfig.hidden, asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM);
                            }
                            if (n_idx + MAD_N == BLOCK_N)
                                notify_intra_block<PIPE_FIX, 1>(is_linear1 ? kLinear1Full : kLinear2Full, fix_stage_idx);
                        }
                    }
                    wait_before_l0a_load = BLOCK_N == MAD_N;
                    l0a_stage_idx = (l0a_stage_idx + 1) % kNumL0Stages;
                }

                // Release L1 after the last L0 load
                if (not is_shared)
                    notify_intra_block<PIPE_MTE1, 0>(kL1BEmpty, l1_stage_idx);
                notify<PIPE_MTE1, PIPE_MTE2>(kL1AEmpty, l1_stage_idx);
                l1_stage_idx = (l1_stage_idx + 1) % kNumL1AStages;
                if ((sf_pair_idx + BLOCK_K / MX_SF_DIVISOR) % kNumSFPairsPerChunk == 0 or k_idx + valid_k == num_k) {
                    notify<PIPE_MTE1, PIPE_MTE2>(kSFEmpty, sf_stage_idx);
                    sf_stage_idx = (sf_stage_idx + 1) % kNumSFStages;
                }
            }
        };
        if (valid_m <= 128)
            compute.template operator()<128>();
        else
            compute.template operator()<BLOCK_M>();

        asc_store_dev(num_issued_gemms_ptr, ++ num_issued_gemms);
    }
    if constexpr (kHasSharedExperts)
        asc_sync_data_barrier(DSB_DDR);
    asc_sync_block_wait(PIPE_S, kCleanupReadyID);
    // Each AIC has observed both local readers; join before clearing shared controls.
    asc_sync_inter_arrive(PIPE_MTE2, kPayloadRetiredID);
    asc_sync_inter_wait(PIPE_S, kPayloadRetiredID);
    for (uint32_t stage_idx = 0; stage_idx < layout::kNumProgressStages; ++ stage_idx)
        for (uint32_t counter_idx = core_idx; counter_idx < (num_shared_m_blocks + num_routed_m_blocks) * layout::kNumProgressShards; counter_idx += kNumAICores)
            asc_store_dev(kWorkspace.progress.blocks.get_ptr(buffer, stage_idx * kWorkspace.num_max_progress_blocks * layout::kNumProgressShards + counter_idx), uint32_t{0});
    asc_store_dev(kWorkspace.progress.num_issued_gemms.get_ptr(buffer, core_idx), uint32_t{0});
    asc_store_dev(kWorkspace.progress.metadata_ready.get_ptr(buffer, core_idx), uint8_t{0});
    for (uint32_t peer_idx = core_idx; peer_idx < kConfig.num_ranks; peer_idx += kNumAICores)
        asc_store_dev(kWorkspace.combine_ready_prefix.get_ptr(buffer, peer_idx), uint64_t{0});
    if (core_idx == 0) {
        asc_store_dev(routed_m_block_count_ptr, uint32_t{0});
        const uint64_t epoch = asc_load_dev(kWorkspace.dispatch_epoch.get_ptr(buffer));
        const auto retired = kWorkspace.retired.get_ptr(buffer);
        // Publish the epoch after recording the preceding metadata extent.
        asc_store_dev(&retired->num_experts, uint64_t{kConfig.num_experts});
        asc_store_dev(&retired->metadata_end, kWorkspace.src_token_metadata.offset + kConfig.num_src_indices * sizeof(uint64_t));
        asc_store_dev(&retired->epoch, epoch);
    }
    asc_sync_data_barrier(DSB_DDR);
}

// AIV1: epilogue, dispatch, and combine
template <const auto& kWorkspace, bool kLowLatency, bool kRMSNorm, typename rmsnorm_type>
__aicore__ inline void aiv1_epilogue_comm(__gm__ uint8_t* buffer, __gm__ rmsnorm_type* output,
                                          uint32_t num_tokens, uint32_t rank_idx,
                                          CombineReduceState<kWorkspace>& reduce,
                                          float activation_clamp, uint32_t core_idx, const layout::SymBuffer& sym_buffer, __gm__ rmsnorm_type* rms_weight, float rms_eps) {
    constexpr auto& kConfig = kWorkspace.config;
    constexpr bool kHasSharedExperts = kConfig.num_shared_experts > 0;
    constexpr uint32_t num_first_wave_blocks = ceil_div(kNumAICores, kConfig.get_num_linear1_n_blocks());

    constexpr uint32_t kNumOutputSFPairs = 2;
    constexpr uint32_t kNumUBAlignmentBytes = 32;

    // AIV1 UB with early reduction (bytes); H=hidden
    // Region        Buffers  Bytes/buffer
    // FIX accum           2  STORE_BLOCK_M * BLOCK_N * 2
    // Route weights       1  STORE_BLOCK_M * 4
    // Output SF           2  STORE_BLOCK_M * 4
    // Pull tokens         3  H
    // Pull SF             3  align(H/32, 32)
    // Peer offsets        1  num_ranks * 8
    // Progress delta      1  32: [1, 0, ..., 0] for atomic publication
    // Push tokens         3  H * 2
    // Early reduce        2  Remaining UB, rounded down to whole vectors
    // Without early reduction, push expands into available UB up to kNumPushStages
    // Peer offsets and push tokens start at 32-byte boundaries
    // Final reduce reuses UB from offset 0 after MTE3 stores complete
    constexpr ub_ptr<bfloat16_t> ub_accum(STORE_BLOCK_M, BLOCK_N);
    constexpr ub_ptr<float> ub_route_weights(1, STORE_BLOCK_M, ub_accum.offset(kNumEpilogueStages));
    constexpr ub_ptr<int16_t> ub_output_sf(kNumOutputSFPairs, STORE_BLOCK_M, ub_route_weights.offset(1));

    constexpr ub_ptr<float8_e4m3_t> ub_pull_token(1, kConfig.hidden, ub_output_sf.offset(kNumEpilogueStages));
    constexpr ub_ptr<int16_t> ub_pull_sf(1, aligned(kConfig.num_input_sf_pairs * sizeof(int16_t), kNumUBAlignmentBytes) / sizeof(int16_t),
                                         ub_pull_token.offset(kNumPullStages));

    constexpr ub_ptr<int64_t> ub_offsets(1, kConfig.num_ranks, aligned(ub_pull_sf.offset(kNumPullStages), kNumUBAlignmentBytes));

    // Progress stays separate; push and early reduction share only the optional stages
    constexpr ub_ptr<uint32_t> ub_progress_delta(1, kNumUBAlignmentBytes / sizeof(uint32_t), aligned(ub_offsets.offset(1), kNumUBAlignmentBytes));
    constexpr ub_ptr<bfloat16_t> ub_push_token(1, kConfig.hidden, ub_progress_delta.offset(1));
    constexpr ub_ptr<uint64_t> ub_ready_sentinel(1, kDataBlockBytes / sizeof(uint64_t), UBSizeBytes - (kLowLatency ? kDataBlockBytes : 0));
    static_assert(ub_push_token.offset(kNumPushStagesWithEarlyReduce) <= ub_ready_sentinel.addr, "AIV1 push UB overflow");
    constexpr auto ub_reduce_inputs = make_reduce_buffer<kWorkspace, kRMSNorm, rmsnorm_type>(ub_push_token.offset(kNumPushStagesWithEarlyReduce));
    if constexpr (not kLowLatency and kRMSNorm)
        static_assert(ub_reduce_inputs.addr + kConfig.hidden * (2 * sizeof(rmsnorm_type)) + sizeof(vector_float) <= UBSizeBytes);
    constexpr uint32_t num_push_stages_in_ub = (ub_ready_sentinel.addr - ub_push_token.addr) / ub_push_token.size_per_stage();
    constexpr uint32_t num_push_stages = kLowLatency ? (num_push_stages_in_ub < kNumPushStages ? num_push_stages_in_ub : kNumPushStages) : kNumPushStagesWithEarlyReduce;

    static_assert(kLowLatency or (ub_reduce_inputs.offset(kNumReduceStages) <= UBSizeBytes and ub_reduce_inputs.shape_n > 0), "AIV1 UB overflow");

    const uint32_t num_shared_m_blocks = kHasSharedExperts ? ceil_div(num_tokens, BLOCK_M) : 0;

    notify_intra_block<PIPE_MTE3, 1, kNumEpilogueStages>(kLinear1Empty);
    notify_intra_block<PIPE_S, 1, kNumEpilogueStages>(kLinear2Empty);
    notify<PIPE_V, PIPE_MTE2>(kTopkEmpty);
    notify<PIPE_MTE3, PIPE_MTE2, kNumPullStages>(kPullEmpty);

    const auto routed_output = kWorkspace.routed.output.get_ptr(buffer);

    uint32_t num_routed_m_blocks = 0;  // Routed pool only; shared blocks are separate
    bool metadata_ready = false;
    const auto routed_m_block_count_ptr = kWorkspace.routed_m_block_count.get_ptr(buffer);

    for (uint32_t i = 0; i < ub_progress_delta.shape_n; ++ i)
        ub_progress_delta.ptr()[i] = i == 0;
    if constexpr (kLowLatency)
        ub_ready_sentinel.ptr()[0] = ~uint64_t{0};
    asc_sync_data_barrier(DSB_UB);
    copy_gm_to_ub(ub_offsets.template ptr<uint8_t>(), reinterpret_cast<__gm__ uint8_t*>(sym_buffer.offsets), 1, kConfig.num_ranks * sizeof(int64_t), kConfig.num_ranks * sizeof(int64_t));
    notify<PIPE_MTE2, PIPE_S>(kPeerOffsetsFull);
    wait<PIPE_MTE2, PIPE_S>(kPeerOffsetsFull);
    const auto publish_block_progress = [&](layout::ProgressStage stage, uint32_t slot_idx) __aicore__ {
        const auto progress = kWorkspace.progress.blocks.get_ptr(buffer, kConfig.get_progress_idx(stage, slot_idx, core_idx % layout::kNumProgressShards, kWorkspace.num_max_progress_blocks));
        // FIX has retired; only the other stages need to follow MTE3 stores.
        if (stage == layout::kLinear2Ready) {
            asc_atomic_add(progress, uint32_t{1});
            return;
        }
        asc_sync_data_barrier(DSB_DDR);
        // MTE3 orders payload stores before the progress atomic
        asc_sync_pipe(PIPE_MTE3);
        asc_set_atomic_add_int32();
        copy_ub_to_gm(progress, ub_progress_delta.ptr(), 1, ub_progress_delta.shape_n, kNumUBAlignmentBytes, asc_store_l2_cache_mode::NORMAL_LAST_VICTIM);
        asc_set_atomic_none();
    };

    const auto num_issued_gemms_ptr = kWorkspace.progress.num_issued_gemms.get_ptr(buffer, core_idx);

    struct DispatchPullState {
        __gm__ uint32_t* src_m_end = nullptr;
        uint32_t pool_block_idx = 0;
        uint32_t stage_idx = 0;
        uint32_t m_idx_in_block;
        uint32_t valid_m;
        sched::ExpertCursor<kWorkspace, kLowLatency> expert;

        // Permute the concatenated source rows without changing their counts
        uint32_t src_m_stride = 1;
        uint32_t src_m_step = 0;
        uint32_t src_m_idx = 0;
        uint32_t src_rank_idx = 0;
    };
    DispatchPullState pull{.m_idx_in_block = core_idx};

    // Burst the initial routed pull in low-latency mode
    const auto advance_dispatch_pull = [&]() __aicore__ {
        if (pull.m_idx_in_block == core_idx) {
            // Reuse input rows after all Linear1 N-blocks finish
            if (pull.pool_block_idx >= kConfig.get_num_ring_blocks() and
                    not comm::is_block_ready<kWorkspace>(buffer, layout::kLinear1Ready,
                                                        num_shared_m_blocks + pull.pool_block_idx - kConfig.get_num_ring_blocks(),
                                                        kConfig.get_num_linear1_n_blocks()))
                return;

            uint32_t expert_m_begin;
            pull.valid_m = pull.expert.advance_to(buffer, pull.pool_block_idx, &expert_m_begin, not metadata_ready);
            if (pull.valid_m == 0)
                return;
            if (expert_m_begin == 0) {
                pull.src_m_end = kWorkspace.expert_src_m_end.get_ptr(buffer, pull.expert.local_expert_idx * kConfig.src_m_end_gm_stride);
                // Invalidate only this ready prefix row.
                for (uint32_t src_rank_idx = 0; src_rank_idx < kConfig.num_ranks; src_rank_idx += kDCacheLineBytes / sizeof(uint32_t))
                    __asc_aicore::asc_dcci_single(pull.src_m_end + src_rank_idx);
                pull.src_m_stride = asc_load_dev(kWorkspace.expert_src_m_stride.get_ptr(buffer, pull.expert.local_expert_idx));
                pull.src_m_step = kNumAICores * pull.src_m_stride % pull.expert.num_expert_tokens;
            }
            pull.src_m_idx = static_cast<uint64_t>(expert_m_begin + core_idx) * pull.src_m_stride % pull.expert.num_expert_tokens;
            if (pull.src_rank_idx != 0 and pull.src_m_idx < pull.src_m_end[pull.src_rank_idx - 1])
                pull.src_rank_idx = 0;
        }

        for (uint32_t step_idx = 0; step_idx < (kLowLatency ? ceil_div(BLOCK_M, kNumAICores) : kNumCommRowsPerStep) and pull.m_idx_in_block < pull.valid_m; ++ step_idx) {
            const uint32_t m_idx_in_block = pull.m_idx_in_block;
            pull.m_idx_in_block += kNumAICores;
            const uint32_t stage_idx = pull.stage_idx;
            pull.stage_idx = (pull.stage_idx + 1) % kNumPullStages;
            const uint64_t pool_m_idx = static_cast<uint64_t>(pull.pool_block_idx) * BLOCK_M + m_idx_in_block;
            const uint64_t ring_m_idx = kConfig.get_ring_m_idx(pull.pool_block_idx, m_idx_in_block);
            while (pull.src_m_idx >= pull.src_m_end[pull.src_rank_idx])
                ++ pull.src_rank_idx;
            const uint32_t src_rank_idx = pull.src_rank_idx;
            const uint32_t src_slot_idx = pull.src_m_idx - (src_rank_idx == 0 ? 0 : pull.src_m_end[src_rank_idx - 1]);

            pull.src_m_idx += pull.src_m_step;
            if (pull.src_m_idx >= pull.expert.num_expert_tokens) {
                pull.src_m_idx -= pull.expert.num_expert_tokens;
                pull.src_rank_idx = 0;
            }
            const auto remote_base = layout::SymBuffer::map(buffer, src_rank_idx, ub_offsets.ptr());

            // Resolve the source token
            wait<PIPE_MTE3, PIPE_MTE2>(kPullEmpty, stage_idx);
            uint64_t token_metadata = 0;
            if constexpr (kLowLatency) {
                const auto metadata_ptr = kWorkspace.fast_src_metadata.get_ptr(buffer,
                    kConfig.get_src_metadata_idx(pull.expert.local_expert_idx, src_rank_idx, src_slot_idx, layout::kNumFastSrcSlotsPerExpert));
                comm::wait_until([&]() __aicore__ {
                    if (((token_metadata = asc_load_dev(metadata_ptr)) & layout::kRecordArrivalBit) != 0)
                        return true;
                    comm::backoff();
                    return false;
                }, [&]() __aicore__ { comm::print_local_timeout(rank_idx); });
                asc_store_dev(metadata_ptr, uint64_t{0});
                token_metadata &= ~uint64_t{layout::kRecordArrivalBit};
            } else {
                token_metadata = asc_load_dev(kWorkspace.src_token_metadata.get_ptr(buffer,
                    kConfig.get_src_metadata_idx(pull.expert.local_expert_idx, src_rank_idx, src_slot_idx)));
            }
            const uint32_t token_topk_idx = static_cast<uint32_t>(token_metadata);
            const uint32_t src_token_idx = token_topk_idx / kConfig.num_topk;

            // Pull token data and scales
            copy_gm_to_ub(ub_pull_token[stage_idx], kWorkspace.input.acts.data.get_ptr(remote_base, static_cast<uint64_t>(src_token_idx) * kConfig.hidden),
                          kConfig.hidden, kLowLatency ? asc_load_l2_cache_mode::NOTALLOC_KEEP : asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
            copy_gm_to_ub(ub_pull_sf[stage_idx].ptr(),
                          kWorkspace.input.acts.sf.get_ptr(remote_base, static_cast<uint64_t>(src_token_idx) * kConfig.num_input_sf_pairs),
                          1, kConfig.num_input_sf_pairs, kConfig.num_input_sf_pairs * sizeof(int16_t),
                          kLowLatency ? asc_load_l2_cache_mode::NOTALLOC_KEEP : asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM);
            notify<PIPE_MTE2, PIPE_MTE3>(kPullFull, stage_idx);

            // Stage routed inputs
            wait<PIPE_MTE2, PIPE_MTE3>(kPullFull, stage_idx);
            copy_ub_to_gm(kWorkspace.routed.l1_acts.data.get_ptr(buffer, ring_m_idx * kConfig.hidden),
                          ub_pull_token[stage_idx].ptr(), 1, kConfig.hidden,
                          kConfig.hidden, kWriteThroughShare);
            copy_ub_to_gm(kWorkspace.routed.l1_acts.sf.get_ptr(buffer, ring_m_idx * kConfig.num_input_sf_pairs),
                          ub_pull_sf[stage_idx].ptr(), 1, kConfig.num_input_sf_pairs,
                          kConfig.num_input_sf_pairs * sizeof(int16_t), kWriteThroughShare);
            notify<PIPE_MTE3, PIPE_MTE2>(kPullEmpty, stage_idx);

            const uint64_t combine_m_idx = kConfig.get_combine_m_idx(token_topk_idx % kConfig.num_topk, src_token_idx);

            // Copy route weights as FP32 bits while DMA runs
            asc_store_dev(kWorkspace.routed.topk_weights.template get_ptr<uint32_t>(buffer, pool_m_idx), static_cast<uint32_t>(token_metadata >> 32));
            // Source metadata: rank [31:24], combine M [23:0]
            asc_store_dev(kWorkspace.routed.src_metadata.get_ptr(buffer, pool_m_idx), static_cast<uint32_t>(combine_m_idx) | (src_rank_idx << 24));
        }

        if (pull.m_idx_in_block >= pull.valid_m) {
            publish_block_progress(layout::kDispatched, num_shared_m_blocks + pull.pool_block_idx);
            ++ pull.pool_block_idx;
            pull.m_idx_in_block = core_idx;
        }
    };

    struct EpilogueState {
        sched::LinearTaskState schedule;
        sched::ExpertCursor<kWorkspace, kLowLatency> expert;
        uint32_t task_idx;
        uint32_t valid_m;
        sched::BlockPhase phase = sched::BlockPhase::kNone;
        uint32_t m_idx_in_block = 0;
        uint32_t num_completed_gemms = 0;  // Advanced by epilogue, paired with num_issued_gemms
    };
    EpilogueState epi{};

    // Process Linear1 rows or wait for Linear2 stores
    const auto advance_epilogue = [&](bool routed_push_submitted) __aicore__ -> void {
        if (epi.phase == sched::BlockPhase::kNone)
            epi.task_idx = sched::take_linear_task<kWorkspace, kLowLatency>(
                num_tokens, num_routed_m_blocks, epi.schedule, core_idx, epi.phase, metadata_ready);

        // A longer shared phase finishes before consuming routed metadata.
        if (not metadata_ready and (kLowLatency or epi.phase != sched::BlockPhase::kSharedLinear1) and
            asc_load_dev(kWorkspace.progress.metadata_ready.get_ptr(buffer, core_idx)) == layout::kMetadataComplete) {
            asc_sync_data_barrier(DSB_DDR);
            num_routed_m_blocks = asc_load_dev(routed_m_block_count_ptr) & layout::kCountValueMask;
            metadata_ready = true;
        }
        if (epi.phase == sched::BlockPhase::kNone or epi.phase == sched::BlockPhase::kDone)
            return;
        // Prioritize initial pull and routed pushes over shared epilogue in low-latency mode.
        if (kLowLatency and ((epi.phase == sched::BlockPhase::kSharedLinear1 and pull.pool_block_idx < (metadata_ready ? min(num_first_wave_blocks, num_routed_m_blocks) : num_first_wave_blocks)) or (epi.phase == sched::BlockPhase::kSharedLinear2 and not routed_push_submitted)))
            return;

        // Issued tasks still need their FIX completion event
        if (epi.m_idx_in_block == 0 and asc_load_dev(num_issued_gemms_ptr) == epi.num_completed_gemms)
            return;

        const bool is_shared = epi.phase == sched::BlockPhase::kSharedLinear1 or epi.phase == sched::BlockPhase::kSharedLinear2;
        const bool is_linear1 = epi.phase == sched::BlockPhase::kLinear1 or epi.phase == sched::BlockPhase::kSharedLinear1;
        const uint32_t num_n_blocks = is_linear1 ? (is_shared ? kConfig.get_num_shared_linear1_n_blocks() : kConfig.get_num_linear1_n_blocks()) : kConfig.get_num_linear2_n_blocks();
        const uint32_t pool_block_idx = epi.task_idx / num_n_blocks;

        if (not is_linear1) {
            // Wait for Linear2 stores and return FIX credit
            wait_intra_block<PIPE_S, 1>(kLinear2Full, epi.num_completed_gemms % kNumEpilogueStages);
            notify_intra_block<PIPE_S, 1>(kLinear2Empty, epi.num_completed_gemms % kNumEpilogueStages);
        } else {
            // Reuse Linear2 inputs after all output N-blocks finish
            if (not is_shared and epi.m_idx_in_block == 0 and pool_block_idx >= kConfig.get_num_ring_blocks() and
                    not comm::is_block_ready<kWorkspace>(buffer, layout::kLinear2Ready,
                                                        num_shared_m_blocks + pool_block_idx - kConfig.get_num_ring_blocks(),
                                                        kConfig.get_num_linear2_n_blocks()))
                return;

            if (epi.m_idx_in_block == 0)
                epi.valid_m = is_shared ? min(num_tokens - pool_block_idx * BLOCK_M, BLOCK_M) :
                    epi.expert.advance_to(buffer, pool_block_idx);
            const uint32_t stage_idx = epi.m_idx_in_block / STORE_BLOCK_M;
            const uint32_t stage_m_begin = stage_idx * STORE_BLOCK_M;
            const uint32_t stage_valid_m = min(STORE_BLOCK_M, epi.valid_m - stage_m_begin);
            // Load route weights and wait for FIX
            if (not is_shared) {
                const auto route_weights = kWorkspace.routed.topk_weights.get_ptr(
                    buffer, static_cast<uint64_t>(pool_block_idx) * BLOCK_M + stage_m_begin);
                wait<PIPE_V, PIPE_MTE2>(kTopkEmpty);
                copy_gm_to_ub(ub_route_weights.ptr(), route_weights, 1, stage_valid_m,
                              stage_valid_m * sizeof(float), asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM);
                notify<PIPE_MTE2, PIPE_V>(kTopkFull);
            }
            wait_intra_block<PIPE_V, 1>(kLinear1Full, stage_idx);
            if (not is_shared)
                wait<PIPE_MTE2, PIPE_V>(kTopkFull);

            // Apply SwiGLU and quantize one epilogue stage in place
            const auto ub_input = ub_accum[stage_idx].ptr();
            const auto ub_output = reinterpret_cast<__ubuf__ float8_e4m3_t*>(ub_input);
            const auto ub_output_sf_ptr = ub_output_sf[stage_idx].ptr();
            if (is_shared) {
                vf_swiglu_quant<false, STORE_BLOCK_M>(ub_input, ub_output, ub_output_sf_ptr, nullptr, stage_valid_m, activation_clamp);
            } else {
                vf_swiglu_quant<true, STORE_BLOCK_M>(ub_input, ub_output, ub_output_sf_ptr, ub_route_weights.ptr(), stage_valid_m, activation_clamp);
            }
            epi.m_idx_in_block += stage_valid_m;

            // Finish GM stores before FIX reuses this UB stage
            if (not is_shared)
                notify<PIPE_V, PIPE_MTE2>(kTopkEmpty);
            notify<PIPE_V, PIPE_MTE3>(kQuantFull);
            wait<PIPE_V, PIPE_MTE3>(kQuantFull);

            const uint32_t n_block_idx = epi.task_idx % num_n_blocks;
            const uint64_t output_m_idx = static_cast<uint64_t>(
                is_shared ? pool_block_idx : kConfig.get_ring_block_idx(pool_block_idx)) * BLOCK_M;
            const auto dst_layout = is_shared ? kWorkspace.shared_l2_acts : kWorkspace.routed.l2_acts;
            const uint32_t num_cols = num_n_blocks * (BLOCK_N / 2);
            const uint32_t num_sf_pairs = num_cols / MX_SF_DIVISOR;
            const auto dst = dst_layout.data.get_ptr(buffer,
                (output_m_idx + stage_m_begin) * num_cols + n_block_idx * (BLOCK_N / 2));
            const auto dst_sf = dst_layout.sf.get_ptr(buffer,
                output_m_idx * num_sf_pairs + n_block_idx * kNumOutputSFPairs * BLOCK_M + stage_m_begin);

            copy_ub_to_gm(dst, ub_ptr<float8_e4m3_t>(STORE_BLOCK_M, BLOCK_N / 2, ub_accum[stage_idx].addr),
                          stage_valid_m, BLOCK_N / 2, num_cols, kWriteThroughShare);
            copy_ub_to_gm(dst_sf, ub_output_sf[stage_idx],
                          kNumOutputSFPairs, stage_valid_m, BLOCK_M * sizeof(int16_t), kWriteThroughShare);
            notify_intra_block<PIPE_MTE3, 1>(kLinear1Empty, stage_idx);

            if (epi.m_idx_in_block != epi.valid_m)
                return;
            epi.m_idx_in_block = 0;
        }

        ++ epi.num_completed_gemms;

        // Shared Linear2 counts each active core once in slot 0
        if (is_linear1) {
            publish_block_progress(layout::kLinear1Ready, (is_shared ? 0 : num_shared_m_blocks) + pool_block_idx);
        } else if (not is_shared) {
            publish_block_progress(layout::kLinear2Ready, num_shared_m_blocks + pool_block_idx);
        } else {
            const uint32_t local_n = num_n_blocks / kNumLocalityDomains;
            const uint32_t local_task_idx = pool_block_idx * local_n + (epi.task_idx % num_n_blocks) % local_n;
            if (local_task_idx + kNumAICoresPerDomain >= num_shared_m_blocks * local_n)
                publish_block_progress(layout::kLinear2Ready, 0);
        }
        epi.phase = sched::BlockPhase::kNone;
    };

    struct CombinePushState {
        sched::ExpertCursor<kWorkspace, kLowLatency> expert;
        uint32_t stage_idx = 0;
        uint32_t pool_block_idx = 0;
        uint32_t m_idx_in_block;
        uint32_t valid_m;
    };
    CombinePushState push{.m_idx_in_block = core_idx};

    // Send BF16 rows
    const auto advance_combine_push = [&]() __aicore__ {
        if (push.pool_block_idx == num_routed_m_blocks or (push.m_idx_in_block == core_idx and
                not comm::is_block_ready<kWorkspace>(buffer, layout::kLinear2Ready,
                                                    num_shared_m_blocks + push.pool_block_idx,
                                                    kConfig.get_num_linear2_n_blocks())))
            return;

        const uint32_t pool_block_idx = push.pool_block_idx;
        if (push.m_idx_in_block == core_idx)
            push.valid_m = push.expert.advance_to(buffer, pool_block_idx);
        const auto src_metadata_base = kWorkspace.routed.src_metadata.get_ptr(buffer, static_cast<uint64_t>(pool_block_idx) * BLOCK_M);
        const auto src_block = routed_output + kConfig.get_ring_m_idx(pool_block_idx) * kConfig.hidden;

        for (uint32_t step_idx = 0; step_idx < (kLowLatency ? 1u : kNumCommRowsPerStep) and push.m_idx_in_block < push.valid_m; ++ step_idx) {
            const uint32_t m_idx_in_block = push.m_idx_in_block;
            push.m_idx_in_block += kNumAICores;
            const uint32_t stage_idx = push.stage_idx;
            if (++ push.stage_idx == num_push_stages)
                push.stage_idx = 0;
            const uint32_t combine_metadata = asc_load_dev(src_metadata_base + m_idx_in_block);
            const uint32_t combine_m_idx = combine_metadata & 0x00ffffffu;
            const auto src = src_block + m_idx_in_block * kConfig.hidden;

            // Load a BF16 output row
            asc_lock(PIPE_MTE2, stage_idx);
            copy_gm_to_ub(ub_push_token[stage_idx], src,
                          kConfig.hidden * sizeof(bfloat16_t), asc_load_l2_cache_mode::NOTALLOC_DROP);
            asc_unlock(PIPE_MTE2, stage_idx);

            const uint32_t dst_rank_idx = combine_metadata >> 24;
            const auto dst = layout::SymBuffer::map(
                kWorkspace.combine_input.get_ptr(buffer, static_cast<uint64_t>(combine_m_idx) * kConfig.hidden),
                dst_rank_idx, ub_offsets.ptr());

            // Push the BF16 row to its source rank through the mapped UBMEM address
            asc_lock(PIPE_MTE3, stage_idx);
            copy_ub_to_gm(dst, ub_push_token[stage_idx].ptr(), 1, kConfig.hidden,
                          kConfig.hidden * sizeof(bfloat16_t), asc_store_l2_cache_mode::NOTALLOC_CLEAN);
            asc_unlock(PIPE_MTE3, stage_idx);
        }

        if (push.m_idx_in_block >= push.valid_m) {
            push.m_idx_in_block = core_idx;
            ++ push.pool_block_idx;

            // Enqueue completion publication after this batch's stores, including empty row owners
            if ((not kLowLatency or num_routed_m_blocks > kConfig.get_num_ring_blocks()) and
                (push.pool_block_idx % kNumCompletionMBlocks == 0 or (not kLowLatency and push.pool_block_idx == num_routed_m_blocks)))
                publish_block_progress(layout::kCombineSent, num_shared_m_blocks + push.pool_block_idx - 1);
        }
    };

    // All ones marks final routed combine completion; cores without peers start finished
    uint64_t published_prefix = core_idx >= kConfig.num_ranks ? ~uint64_t{0} : 0;
    const auto advance_combine_completion = [&]() __aicore__ {
        if (published_prefix == ~uint64_t{0})
            return;
        const uint32_t pool_block_end = min(static_cast<uint32_t>(published_prefix) + kNumCompletionMBlocks, num_routed_m_blocks);
        if (push.pool_block_idx < pool_block_end)
            return;
        if (pool_block_end != 0 and not comm::is_block_ready<kWorkspace>(
                buffer, layout::kCombineSent, num_shared_m_blocks + pool_block_end - 1, kNumAICores))
            return;

        published_prefix = pool_block_end == num_routed_m_blocks ? ~uint64_t{0} : pool_block_end;
        asc_sync_data_barrier(DSB_DDR);
        for (uint32_t step_idx = core_idx; step_idx < kConfig.num_ranks; step_idx += kNumAICores)
            asc_store_dev(layout::SymBuffer::map(kWorkspace.combine_ready_prefix.get_ptr(buffer, rank_idx), (rank_idx + step_idx) % kConfig.num_ranks, ub_offsets.ptr()), published_prefix);
    };

    comm::wait_until([&]() __aicore__ {
        const uint32_t prev_pull_blk = pull.pool_block_idx, prev_pull_m = pull.m_idx_in_block;
        const uint32_t prev_epi_done = epi.num_completed_gemms, prev_epi_m = epi.m_idx_in_block;
        const uint32_t prev_push_blk = push.pool_block_idx, prev_push_m = push.m_idx_in_block;
        if (not kLowLatency and metadata_ready)
            advance_combine_completion();

        advance_epilogue(not kLowLatency);

        const bool pull_finished = metadata_ready and pull.pool_block_idx == num_routed_m_blocks;
        if (pull_finished and (epi.phase == sched::BlockPhase::kDone or (kLowLatency and epi.phase == sched::BlockPhase::kSharedLinear2)) and
            (kLowLatency or published_prefix == ~uint64_t{0}) and push.pool_block_idx == num_routed_m_blocks)
            return true;

        // Advance pull and push independently of epilogue readiness
        // Prefix-ready experts can fill the first compute wave before metadata fully retires.
        if (not pull_finished and (metadata_ready or (kLowLatency and pull.pool_block_idx < num_first_wave_blocks)))
            advance_dispatch_pull();
        if (metadata_ready)
            advance_combine_push();
        // Return epilogue credits and submit sends before spending the remaining turn on reduction
        if (not kLowLatency and pull_finished and epi.m_idx_in_block == 0)
            advance_combine_reduce<kWorkspace, kRMSNorm>(buffer, output, num_tokens, reduce, ub_reduce_inputs, rms_eps, rms_weight, 2);
        if constexpr (kLowLatency) {
            const bool progressed = (pull.pool_block_idx != prev_pull_blk) or (pull.m_idx_in_block != prev_pull_m) or
                (epi.num_completed_gemms != prev_epi_done) or (epi.m_idx_in_block != prev_epi_m) or
                (push.pool_block_idx != prev_push_blk) or (push.m_idx_in_block != prev_push_m);
            if (not progressed) {
                comm::backoff();
            }
        }
        return false;
    }, [&]() __aicore__ { comm::print_local_timeout(rank_idx); });
    if constexpr (kLowLatency) {
        if constexpr (not kRMSNorm)
            notify<PIPE_MTE3, PIPE_MTE2, final_reduce_stages<kWorkspace, rmsnorm_type>()>(kReduceEmpty);
        asc_sync_inter_arrive(PIPE_MTE3, kCombinePushRetiredID);
        asc_sync_inter_wait(PIPE_MTE3, kCombinePushRetiredID);
        for (uint32_t step_idx = core_idx; step_idx < kConfig.num_ranks; step_idx += kNumAICores)
            copy_ub_to_gm(layout::SymBuffer::map(kWorkspace.combine_ready_prefix.template get_ptr<uint32_t>(buffer, rank_idx),
                                                 (rank_idx + step_idx) % kConfig.num_ranks, ub_offsets.ptr()),
                          ub_ready_sentinel.template ptr<uint32_t>(), 1, 8, kDataBlockBytes, asc_store_l2_cache_mode::NOTALLOC_CLEAN);
    }
    combine_reduce<kWorkspace, kLowLatency, kRMSNorm>(buffer, output, num_tokens, rank_idx, 2 * core_idx + 1, reduce, rms_weight, rms_eps, [&]() __aicore__ {
        if (epi.phase != sched::BlockPhase::kDone)
            advance_epilogue(true);
        return epi.phase == sched::BlockPhase::kDone;
    });
}

// kLowLatency trades early combine reduction for faster small-batch dispatch, scheduling, and tail preload
template <uint32_t kNumRanks, uint32_t kNumExperts, uint32_t kNumSharedExperts,
          uint32_t kNumMaxTokensPerRank, uint32_t kNumTopk, uint32_t kHidden, uint32_t kIntermediateHidden,
          bool kLowLatency, uint32_t kNumMaxProgressBlocks = 0, typename rmsnorm_type = bfloat16_t, bool kRMSNorm = false>
__global__ __mix__(1, 2) void mega_moe_impl(__gm__ rmsnorm_type* output, __gm__ int32_t* cumulative_local_expert_recv_stats,
                                            const layout::Weights weights, uint32_t num_tokens, float activation_clamp,
                                            layout::SymBuffer sym_buffer,
                                            __gm__ rmsnorm_type* rms_weight, float rms_eps) {
    static constexpr layout::Workspace kWorkspace{
        {kNumRanks, kNumExperts, kNumSharedExperts, kNumMaxTokensPerRank, kNumTopk, kHidden, kIntermediateHidden}, kNumMaxProgressBlocks};
    constexpr auto& kConfig = kWorkspace.config;
    static_assert(kWorkspace.num_max_progress_blocks >= kConfig.get_num_max_progress_blocks());

    static_assert(kConfig.get_num_max_m_blocks() < 0x00ffffffu);
    static_assert(kConfig.get_num_ring_blocks() >= kNumCompletionMBlocks or
                  kConfig.get_num_ring_blocks() == kConfig.get_num_max_m_blocks());

    // Block readiness reads the low byte of each completion counter
    static_assert(kConfig.hidden / BLOCK_N <= UINT8_MAX and
                  kConfig.intermediate_hidden / (BLOCK_N / 2) <= UINT8_MAX and
                  kConfig.shared_intermediate_hidden / (BLOCK_N / 2) <= UINT8_MAX);
    static_assert((kConfig.hidden / BLOCK_N) % kNumLocalityDomains == 0 and
                  (kConfig.intermediate_hidden / (BLOCK_N / 2)) % kNumLocalityDomains == 0);

    static_assert(kNumExperts <= layout::kNumMaxExperts and kNumExperts / kNumRanks <= layout::kNumMaxLocalExperts);
    asc_init();
    const uint32_t core_idx = static_cast<uint32_t>(block_idx);

    const auto buffer = __builtin_bit_cast(__gm__ uint8_t*, static_cast<uint64_t>(sym_buffer.base));
    const uint32_t rank_idx = sym_buffer.rank_idx;
    const auto routed_m_block_count_ptr = kWorkspace.routed_m_block_count.get_ptr(buffer);

    // AIC clears controls while AIV finishes vector computation and output stores.
    if ASCEND_IS_AIC {
        aic_gemm_pipeline<kWorkspace, kLowLatency>(buffer, num_tokens, rank_idx, weights, core_idx);
        return;
    }

    // Partition tokens across AIVs
    const uint32_t subblock_idx = static_cast<uint32_t>(asc_get_sub_block_id());
    const uint32_t aiv_idx = 2 * core_idx + subblock_idx;

    // Launch metadata before initializing the later reduction state.
    alignas(32) __ubuf__ int64_t ub_offsets[kNumRanks];
    if (subblock_idx == 0) {
        copy_gm_to_ub(reinterpret_cast<__ubuf__ uint8_t*>(ub_offsets), reinterpret_cast<__gm__ uint8_t*>(sym_buffer.offsets), 1, sizeof(ub_offsets), sizeof(ub_offsets));
        notify<PIPE_MTE2, PIPE_V>(kOffsetsFull);
        wait<PIPE_MTE2, PIPE_V>(kOffsetsFull);
        asc_vf_call<comm::build_metadata<kWorkspace, kLowLatency>>(
            dim3(layout::kNumDispatchThreads, 1, 1), buffer, num_tokens, rank_idx,
            core_idx, ub_offsets, cumulative_local_expert_recv_stats);
    }

    // Both AIVs own round-robin tokens and reduce as their producers finish
    const uint32_t num_reduce_tokens = num_tokens > aiv_idx ? ceil_div(num_tokens - aiv_idx, kNumAIVs) : 0;
    CombineReduceState<kWorkspace> reduce{.token_idx = aiv_idx, .num_remaining_tokens = num_reduce_tokens};
    if constexpr (kRMSNorm) {
        notify<PIPE_V, PIPE_MTE2, kNumReduceStages>(kReduceInputsEmpty);
        notify<PIPE_MTE3, PIPE_V>(kOutputRowEmpty);
    } else if constexpr (not kLowLatency) {
        notify<PIPE_MTE3, PIPE_MTE2, kNumReduceStages>(kReduceEmpty);
    }

    // AIV0: metadata and routed weights
    if (subblock_idx == 0) {
        notify<PIPE_V, PIPE_S>(kMetadataDone);
        wait<PIPE_V, PIPE_S>(kMetadataDone);
        const uint32_t num_routed_m_blocks = asc_load_dev(routed_m_block_count_ptr) & layout::kCountValueMask;

        // Publish this core's metadata completion to AIV1 via GM and AIC via event
        asc_store_dev(kWorkspace.progress.metadata_ready.get_ptr(buffer, core_idx), layout::kMetadataComplete);
        notify_intra_block<PIPE_S, 0>(kMetadataFull);

        aiv0_weight_pipeline<kWorkspace, kLowLatency, kRMSNorm>(buffer, output, num_tokens, reduce, num_routed_m_blocks, weights, core_idx, rms_weight, rms_eps);
        if constexpr (kLowLatency) {
            if constexpr (not kRMSNorm)
                notify<PIPE_MTE3, PIPE_MTE2, final_reduce_stages<kWorkspace, rmsnorm_type>()>(kReduceEmpty);
            asc_sync_inter_arrive(PIPE_MTE3, kCombinePushRetiredID);
        }
        combine_reduce<kWorkspace, kLowLatency, kRMSNorm>(buffer, output, num_tokens, rank_idx, aiv_idx, reduce, rms_weight, rms_eps);
        if constexpr (kLowLatency)
            asc_sync_inter_wait(PIPE_V, kCombinePushRetiredID);
    } else {
        // AIV1: epilogue, communication, and early reduction
        aiv1_epilogue_comm<kWorkspace, kLowLatency, kRMSNorm>(buffer, output, num_tokens, rank_idx, reduce, activation_clamp, core_idx, sym_buffer, rms_weight, rms_eps);
    }
    // Handoff read retirement without waiting for the final output stores.
    asc_sync_block_arrive(PIPE_MTE2, kCleanupReadyID);

    // Drain this AIV's output stores independently of AIC cleanup.
    notify<PIPE_MTE3, PIPE_S>(kCommDone);
    wait<PIPE_MTE3, PIPE_S>(kCommDone);
}

} // namespace deep_gemm::mega_moe
