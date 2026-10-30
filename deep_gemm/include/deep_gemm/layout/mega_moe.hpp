#pragma once

#include <cstdint>
#include <type_traits>

#include <deep_gemm/common.hpp>
#include <deep_gemm/layout/sym_buffer.hpp>

namespace deep_gemm::mega_moe::layout {

// Logical blocks keep the GM stride and task count independent of the MAD shape
constexpr uint32_t BLOCK_M = 256, BLOCK_N = 256;

using deep_gemm::layout::SymBuffer;

// GM buffer alignment/count stride, not the 128-byte L2 cache sector
constexpr uint64_t kNumBufferAlignmentBytes = 512;

// SIMT threads per metadata worker and routes per batch
constexpr uint32_t kNumDispatchThreads = 512;

// Packed count: arrivals [31:24], value [23:0]
constexpr uint32_t kCountArrivalBit = 1u << 24;
constexpr uint32_t kCountValueMask = kCountArrivalBit - 1;

// Routed progress counts workers or GEMM N blocks
enum ProgressStage : uint32_t {
    kDispatched,    // Linear1 inputs ready
    kLinear1Ready,  // Quantized Linear2 inputs ready
    kLinear2Ready,  // BF16 expert outputs ready
    kCombineSent,   // Sends complete; output ring reusable
    kNumProgressStages
};

// Fixed shape and registered token capacity
struct Config {
    uint32_t num_ranks;
    uint32_t num_experts;
    uint32_t num_shared_experts;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_topk;
    uint32_t hidden;
    uint32_t intermediate_hidden;
    uint32_t num_ai_cores;

    const uint32_t num_aivs = 2 * num_ai_cores;
    const uint32_t shared_intermediate_hidden = num_shared_experts * intermediate_hidden;
    const uint32_t num_input_sf_pairs = hidden / MX_SF_DIVISOR;
};

// Typed GM range with a fixed byte stride
template <typename dtype_t, uint64_t kStrideBytes = sizeof(dtype_t)>
struct Buffer {
    static_assert(kStrideBytes >= sizeof(dtype_t) and kStrideBytes % alignof(dtype_t) == 0);
    uint64_t offset;

    constexpr __aicore__ Buffer(uint64_t& next_offset, uint64_t num_elements, uint64_t alignment = kStrideBytes):
        offset(aligned(next_offset, alignment)) { next_offset = offset + num_elements * kStrideBytes; }

    // entry_idx uses kStrideBytes even when view_t changes
    // Buffer<uint32_t>::get_ptr<uint64_t>(base, i) advances by i * 4 bytes
    template <typename view_t = dtype_t, typename base_t>
    constexpr __aicore__ __attribute__((always_inline)) __gm__ view_t* get_ptr(base_t base, uint64_t entry_idx = 0) const {
        static_assert(std::is_pointer_v<base_t> or std::is_same_v<base_t, uint64_t>,
                      "Buffer base must be a pointer or a uint64_t address");

        // Preserve the base type through CANN lowering
        const auto address = __builtin_bit_cast(uintptr_t, base) + offset + entry_idx * kStrideBytes;
        return __builtin_bit_cast(__gm__ view_t*, address);
    }
};

struct LinearWeights {
    __gm__ const uint8_t* data;
    __gm__ const int16_t* sf;
};

struct Weights {
    LinearWeights l1;
    LinearWeights l2;
    LinearWeights shared_l1;
    LinearWeights shared_l2;
};
static_assert(sizeof(Weights) == 8 * sizeof(uint64_t));

// GM workspace layout; all offsets are relative to the allocation base
struct Workspace {
private:
    // Host and device use the same storage layout
#ifdef __CCE__
    using fp8_t = float8_e4m3_t;
    using bf16_t = bfloat16_t;
#else
    using fp8_t = uint8_t;
    using bf16_t = uint16_t;
#endif

    // Keep capacity calculations independent of include order
    template <typename lhs_t, typename rhs_t>
    static constexpr __aicore__ auto ceil_div(lhs_t value, rhs_t divisor) -> decltype(value + divisor) {
        return (value + divisor - 1) / divisor;
    }

public:
    // Rank-barrier ABI; alternate signal slots by epoch
    struct RankBarrier {
        uint64_t epoch;
        int32_t signals[2];
    };

    struct FP8Buffer {
        Buffer<fp8_t> data;
        Buffer<int16_t> sf;

        constexpr __aicore__ FP8Buffer(uint64_t& offset, uint64_t num_rows, uint32_t num_cols):
            data{offset, num_rows * num_cols, kNumBufferAlignmentBytes},
            sf{offset, num_rows * ceil_div(num_cols, MX_SF_DIVISOR), kNumBufferAlignmentBytes} {}
    };

    Config config;

    // Reserve one packed index/weight entry per source route.
    const uint32_t num_src_slots_per_expert = config.num_max_tokens_per_rank * config.num_topk;
    const uint64_t num_src_indices = static_cast<uint64_t>(config.num_experts) * num_src_slots_per_expert;
    const uint32_t num_padded_shared_rows = aligned(config.num_max_tokens_per_rank, BLOCK_M);

    const uint32_t num_max_shared_m_blocks = config.num_shared_experts > 0 ? ceil_div(config.num_max_tokens_per_rank, BLOCK_M) : 0;
    const uint32_t num_progress_counters_per_stage = num_max_shared_m_blocks + get_num_max_m_blocks();
    const uint64_t num_full_pool_rows = get_num_max_m_blocks() * BLOCK_M;

    // Size from the registered limit, not active T
    const uint64_t num_ring_rows = static_cast<uint64_t>(get_num_ring_blocks()) * BLOCK_M;

    // Field order fixes the symmetric GM placement
    uint64_t next_offset = 0;
    struct {
        Buffer<uint32_t> grid_sync_count;
        Buffer<RankBarrier> rank_barriers;
    } sync{
        .grid_sync_count = {next_offset, 3, kNumBufferAlignmentBytes},
        .rank_barriers = {next_offset, 2, kNumBufferAlignmentBytes},
    };

    struct {
        Buffer<uint32_t> worker_expert_count;
        Buffer<uint32_t> total_expert_count;
        // Completed producer M-blocks; all ones means this rank has finished its stores
        Buffer<uint64_t, kL2SectorBytes> combine_ready_prefix;
        Buffer<uint32_t> expert_recv_count;
        Buffer<uint32_t, kNumBufferAlignmentBytes> expert_recv_count_sum;

        // Packed AIV0 arrivals/M-block count is independent of expert counters
        Buffer<uint32_t> num_m_blocks_certificate;
        Buffer<uint64_t> src_token_metadata;
        Buffer<uint32_t> staged_token_topk_idx;
    } metadata{
        .worker_expert_count = {next_offset, static_cast<uint64_t>(config.num_ai_cores) * config.num_experts, kNumBufferAlignmentBytes},
        .total_expert_count = {next_offset, config.num_experts},
        .combine_ready_prefix = {next_offset, config.num_ranks},
        .expert_recv_count = {next_offset, config.num_experts},
        .expert_recv_count_sum = {next_offset, config.num_experts / config.num_ranks},
        .num_m_blocks_certificate = {next_offset, 1, kL2SectorBytes},
        .src_token_metadata = {next_offset, num_src_indices, kNumBufferAlignmentBytes},
        .staged_token_topk_idx = {next_offset, num_src_indices, kNumBufferAlignmentBytes},
    };

    struct {
        FP8Buffer acts;
        Buffer<int64_t> topk_idx;
        Buffer<float> topk_weights;
    } input{
        .acts = {next_offset, config.num_max_tokens_per_rank, config.hidden},
        .topk_idx = {next_offset, static_cast<uint64_t>(config.num_max_tokens_per_rank) * config.num_topk, kNumBufferAlignmentBytes},
        .topk_weights = {next_offset, static_cast<uint64_t>(config.num_max_tokens_per_rank) * config.num_topk, kNumBufferAlignmentBytes},
    };

    // Clear these progress arrays together on each launch
    struct {
        Buffer<uint32_t, kL2SectorBytes> blocks;
        Buffer<uint8_t, kL2SectorBytes> num_issued_gemms;
        Buffer<uint8_t, kL2SectorBytes> metadata_ready;
    } progress{
        .blocks = {next_offset, static_cast<uint64_t>(kNumProgressStages) * num_progress_counters_per_stage},
        .num_issued_gemms = {next_offset, config.num_ai_cores},
        .metadata_ready = {next_offset, config.num_ai_cores},
    };
    const uint64_t num_progress_bytes = next_offset - progress.blocks.offset;

    struct {
        FP8Buffer l1_acts;

        // Metadata uses logical rows and never wraps
        Buffer<float> topk_weights;
        Buffer<uint32_t> src_metadata;

        // Input SF is token-major; quantized Linear2 SF is K-major within each M-block
        FP8Buffer l2_acts;
        Buffer<bf16_t> output;
    } routed{
        .l1_acts = {next_offset, num_ring_rows, config.hidden},
        .topk_weights = {next_offset, num_full_pool_rows, kNumBufferAlignmentBytes},
        .src_metadata = {next_offset, num_full_pool_rows, kNumBufferAlignmentBytes},
        .l2_acts = {next_offset, num_ring_rows, config.intermediate_hidden},
        .output = {next_offset, num_ring_rows * config.hidden, kNumBufferAlignmentBytes},
    };

    // One BF16 slot per route, plus one combined shared slot
    Buffer<bf16_t> combine_input{
        next_offset, static_cast<uint64_t>(config.num_topk + (config.num_shared_experts > 0)) * config.num_max_tokens_per_rank * config.hidden,
        kNumBufferAlignmentBytes};

    // Shared Linear2 inputs; Linear1 reads original tokens
    FP8Buffer shared_l2_acts{next_offset, config.num_shared_experts > 0 ? num_padded_shared_rows : 0, config.shared_intermediate_hidden};

    // Producer M-block for each input route; dispatch fills this before publishing readiness
    Buffer<uint32_t> route_m_blocks{
        next_offset, static_cast<uint64_t>(config.num_max_tokens_per_rank) * config.num_topk, kNumBufferAlignmentBytes};

    constexpr __aicore__ uint64_t get_num_max_m_blocks() const {
        return ceil_div(static_cast<uint64_t>(config.num_ranks) * config.num_max_tokens_per_rank * config.num_topk, BLOCK_M) + config.num_experts / config.num_ranks;
    }

    constexpr __aicore__ uint32_t get_num_ring_blocks() const {
        // Measured overlap window; the deadlock-safe minimum can still lose throughput
        const uint32_t num_windows = config.hidden / BLOCK_N <= config.num_ai_cores / 2 ? 4 : 1;
        const uint32_t num_window_blocks = num_windows * config.num_ai_cores;
        return get_num_max_m_blocks() < num_window_blocks ? static_cast<uint32_t>(get_num_max_m_blocks()) : num_window_blocks;
    }

    // Coordinates within the registered GM layout
    constexpr __aicore__ __attribute__((always_inline)) uint32_t get_progress_idx(ProgressStage stage, uint32_t progress_slot_idx) const {
        return stage * num_progress_counters_per_stage + progress_slot_idx;
    }

    constexpr __aicore__ __attribute__((always_inline)) uint64_t get_src_metadata_idx(
        uint32_t local_expert_idx, uint32_t src_rank_idx, uint32_t src_slot_idx) const {
        return (static_cast<uint64_t>(local_expert_idx) * config.num_ranks + src_rank_idx) * num_src_slots_per_expert + src_slot_idx;
    }

    constexpr __aicore__ __attribute__((always_inline)) uint64_t get_expert_recv_idx(uint32_t src_rank_idx, uint32_t local_expert_idx) const {
        return static_cast<uint64_t>(src_rank_idx) * (config.num_experts / config.num_ranks) + local_expert_idx;
    }

    constexpr __aicore__ __attribute__((always_inline)) uint64_t get_combine_m_idx(uint64_t slot_idx, uint64_t token_idx) const {
        return slot_idx * config.num_max_tokens_per_rank + token_idx;
    }

    constexpr __aicore__ uint64_t get_num_bytes() const {
        // HCCL registration uses 2 MiB alignment
        return aligned(next_offset, 2 * 1024 * 1024);
    }
};

template <uint32_t kNumRanks, uint32_t kNumExperts, uint32_t kNumSharedExperts,
          uint32_t kNumMaxTokensPerRank, uint32_t kNumTopk, uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kNumAICores>
inline constexpr Workspace kWorkspace{{
    kNumRanks, kNumExperts, kNumSharedExperts, kNumMaxTokensPerRank, kNumTopk, kHidden, kIntermediateHidden, kNumAICores}};

template <const auto& workspace>
constexpr __aicore__ uint32_t get_ring_block_idx(uint32_t pool_block_idx) {
    constexpr uint32_t kNumRingBlocks = workspace.get_num_ring_blocks();
    // 28/36-core windows are not powers of two; masking aliases live blocks.
    return kNumRingBlocks == workspace.get_num_max_m_blocks() ? pool_block_idx : pool_block_idx % kNumRingBlocks;
}

template <const auto& workspace>
constexpr __aicore__ uint64_t get_ring_m_idx(uint32_t pool_block_idx, uint32_t m_idx_in_block = 0) {
    return static_cast<uint64_t>(get_ring_block_idx<workspace>(pool_block_idx)) * BLOCK_M + m_idx_in_block;
}
} // namespace deep_gemm::mega_moe::layout
