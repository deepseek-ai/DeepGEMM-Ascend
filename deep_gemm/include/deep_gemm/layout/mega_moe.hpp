#pragma once

#include <cstdint>
#include <type_traits>

#include <deep_gemm/common.hpp>
#include <deep_gemm/layout/sym_buffer.hpp>

namespace deep_gemm::mega_moe::layout {

// Logical blocks keep the GM stride and task count independent of the MAD shape
constexpr uint32_t BLOCK_M = 256, BLOCK_N = 256;
constexpr uint32_t kNumCompletionMBlocks = 16;
constexpr uint32_t kNumProgressShards = 4;
constexpr uint8_t kMetadataPrefixesReady = 1, kMetadataComplete = 2;
constexpr uint32_t kNumMaxRanks = 256, kNumMaxExperts = 1024, kNumMaxLocalExperts = 512;

using deep_gemm::layout::SymBuffer;

// Align major GM buffer regions to 512 bytes; L2 sectors are 128 bytes
constexpr uint64_t kNumBufferAlignmentBytes = 512;

// SIMT threads per metadata VF
constexpr uint32_t kNumDispatchThreads = 512;

// Packed count: arrivals [31:24], value [23:0]
constexpr uint32_t kCountArrivalBit = 1u << 24;
constexpr uint32_t kCountValueMask = kCountArrivalBit - 1;
// Fast-metadata slots assume unique experts per token and num_tokens <= kNumFastSrcSlotsPerExpert
constexpr uint32_t kRecordArrivalBit = 1u << 31, kNumFastSrcSlotsPerExpert = kNumCompletionMBlocks * BLOCK_M;

// Routed progress counts cores or GEMM N blocks
enum ProgressStage : uint32_t {
    kDispatched,    // Linear1 inputs ready
    kLinear1Ready,  // Quantized Linear2 inputs ready
    kLinear2Ready,  // BF16 expert outputs ready
    kCombineSent,   // AIV1 MTE3 stores have completed a producer prefix
    kNumProgressStages
};

// Keep capacity calculations independent of include order
template <typename lhs_t, typename rhs_t>
static constexpr __aicore__ auto ceil_div(lhs_t value, rhs_t divisor) -> decltype(value + divisor) {
    return (value + divisor - 1) / divisor;
}

// Fixed shape and registered token capacity
struct Config {
    uint32_t num_ranks;
    uint32_t num_experts;
    uint32_t num_shared_experts;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_topk;
    uint32_t hidden;
    uint32_t intermediate_hidden;

    const uint32_t shared_intermediate_hidden = num_shared_experts * intermediate_hidden;
    const uint32_t num_input_sf_pairs = hidden / MX_SF_DIVISOR;

    // Reserve complete sectors of packed index/weight entries
    const uint32_t num_src_slots_per_expert = aligned(num_max_tokens_per_rank * num_topk, uint32_t{kL2SectorBytes / sizeof(uint64_t)});
    const uint64_t num_src_indices = static_cast<uint64_t>(num_experts) * num_src_slots_per_expert;
    const uint32_t num_padded_shared_rows = aligned(num_max_tokens_per_rank, BLOCK_M);

    const uint32_t num_max_shared_m_blocks = num_shared_experts > 0 ? ceil_div(num_max_tokens_per_rank, BLOCK_M) : 0;
    const uint64_t num_full_pool_rows = get_num_max_m_blocks() * BLOCK_M;

    // Size from the registered limit, not active T
    const uint64_t num_ring_rows = static_cast<uint64_t>(get_num_ring_blocks()) * BLOCK_M;

    // Source-prefix row strides in uint32_t elements, for UB and GM respectively
    const uint64_t src_m_end_ub_stride = aligned(num_ranks, 32 / sizeof(uint32_t));
    const uint64_t src_m_end_gm_stride = aligned(src_m_end_ub_stride, kL2SectorBytes / sizeof(uint32_t));

    // N-block counts used by all three pipelines
    constexpr __aicore__ uint32_t get_num_linear1_n_blocks() const { return intermediate_hidden / (BLOCK_N / 2); }
    constexpr __aicore__ uint32_t get_num_linear2_n_blocks() const { return hidden / BLOCK_N; }
    constexpr __aicore__ uint32_t get_num_shared_linear1_n_blocks() const { return shared_intermediate_hidden / (BLOCK_N / 2); }

    // Reserve for the larger of normal (BLOCK_M) and low-latency (BLOCK_M / 2, T <= kNumFastSrcSlotsPerExpert)
    constexpr __aicore__ uint64_t get_num_max_m_blocks() const {
        const uint64_t normal_blocks = ceil_div(static_cast<uint64_t>(num_ranks) * num_max_tokens_per_rank * num_topk, BLOCK_M);
        const uint32_t ll_tokens = num_max_tokens_per_rank < kNumFastSrcSlotsPerExpert ? num_max_tokens_per_rank : kNumFastSrcSlotsPerExpert;
        const uint64_t ll_blocks = ceil_div(static_cast<uint64_t>(num_ranks) * ll_tokens * num_topk, BLOCK_M / 2);
        return (normal_blocks > ll_blocks ? normal_blocks : ll_blocks) + num_experts / num_ranks;
    }

    template <bool kLowLatency>
    constexpr __aicore__ __attribute__((always_inline)) uint32_t get_routed_m_blocks(uint32_t num_tokens) const {
        return ceil_div(num_tokens, BLOCK_M);
    }

    constexpr __aicore__ uint32_t get_num_max_progress_blocks() const {
        return num_max_shared_m_blocks + get_num_max_m_blocks();
    }

    constexpr __aicore__ uint32_t get_num_ring_blocks() const {
        // Linear2 head, block-boundary rounding and one extra Linear1 wave of lookahead
        const uint32_t num_compute_blocks = 1 + ceil_div(kNumAICores, hidden / BLOCK_N) + ceil_div(3 * kNumAICores, intermediate_hidden / (BLOCK_N / 2));
        // Keep a filling batch and a completion-pending batch beyond the compute window
        const uint32_t num_ring_blocks = aligned(num_compute_blocks + 2 * kNumCompletionMBlocks, kNumCompletionMBlocks);
        return get_num_max_m_blocks() < num_ring_blocks ? static_cast<uint32_t>(get_num_max_m_blocks()) : num_ring_blocks;
    }

    // Coordinates within the registered GM layout
    constexpr __aicore__ __attribute__((always_inline)) uint32_t get_progress_idx(ProgressStage stage, uint32_t progress_slot_idx, uint32_t shard_idx, uint32_t num_max_progress_blocks) const {
        return (stage * num_max_progress_blocks + progress_slot_idx) * kNumProgressShards + shard_idx;
    }

    constexpr __aicore__ __attribute__((always_inline)) uint64_t get_src_metadata_idx(
        uint32_t local_expert_idx, uint32_t src_rank_idx, uint32_t src_slot_idx, uint32_t num_slots = 0) const {
        // Interleave half-sector groups across all experts for each source.
        const uint64_t src_idx = static_cast<uint64_t>(src_rank_idx) * (num_slots ? num_slots : num_src_slots_per_expert) + src_slot_idx;
        return (src_idx / 8 * (num_experts / num_ranks) + local_expert_idx) * 8 + src_idx % 8;
    }

    constexpr __aicore__ __attribute__((always_inline)) uint64_t get_expert_recv_idx(uint32_t src_rank_idx, uint32_t local_expert_idx) const {
        // Use fixed source slices within the existing count allocation.
        return static_cast<uint64_t>(src_rank_idx) * (kNumMaxExperts / num_ranks) + local_expert_idx;
    }

    constexpr __aicore__ __attribute__((always_inline)) uint64_t get_combine_m_idx(uint64_t slot_idx, uint64_t token_idx) const {
        return slot_idx * num_max_tokens_per_rank + token_idx;
    }

    constexpr __aicore__ uint32_t get_ring_block_idx(uint32_t pool_block_idx) const {
        const uint32_t num_ring_blocks = get_num_ring_blocks();
        return num_ring_blocks == get_num_max_m_blocks() ? pool_block_idx : pool_block_idx % num_ring_blocks;
    }

    constexpr __aicore__ uint64_t get_ring_m_idx(uint32_t pool_block_idx, uint32_t m_idx_in_block = 0) const {
        return static_cast<uint64_t>(get_ring_block_idx(pool_block_idx)) * BLOCK_M + m_idx_in_block;
    }
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

struct RetiredState {
    uint64_t epoch, num_experts, metadata_end, reserved;
};

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

public:
    struct FP8Buffer {
        Buffer<fp8_t> data;
        Buffer<int16_t> sf;

        constexpr __aicore__ FP8Buffer(uint64_t& offset, uint64_t num_rows, uint32_t num_cols):
            data{offset, num_rows * num_cols, kNumBufferAlignmentBytes},
            sf{offset, num_rows * ceil_div(num_cols, MX_SF_DIVISOR), kNumBufferAlignmentBytes} {}
    };

    Config config;
    uint32_t num_max_progress_blocks;

    // Control and count snapshots have fixed positions across payload layouts
    uint64_t next_offset = 0;
    Buffer<uint64_t, kL2SectorBytes> dispatch_epoch{next_offset, 1};
    Buffer<RetiredState, kL2SectorBytes> retired{next_offset, 1};
    Buffer<uint64_t, kL2SectorBytes> combine_ready_prefix{next_offset, kNumMaxRanks};
    // Zero means absent; kCountArrivalBit | count publishes a record set, including count zero
    Buffer<uint32_t> expert_recv_count{next_offset, kNumMaxExperts};
    Buffer<uint32_t> expert_recv_stats{next_offset, kNumMaxExperts};
    Buffer<uint64_t> fast_src_metadata{next_offset, static_cast<uint64_t>(kNumMaxExperts) * kNumFastSrcSlotsPerExpert, kNumBufferAlignmentBytes};
    Buffer<uint32_t> routed_m_block_count{next_offset, 1, kL2SectorBytes};

    // Reserved by the base; only active counters are reset by the kernel.
    struct {
        Buffer<uint32_t, kL2SectorBytes> blocks;
        Buffer<uint32_t, kL2SectorBytes> num_issued_gemms;
        Buffer<uint8_t, kL2SectorBytes> metadata_ready;
    } progress{
        .blocks = {next_offset, static_cast<uint64_t>(kNumProgressStages) * num_max_progress_blocks * kNumProgressShards},
        .num_issued_gemms = {next_offset, kNumAICores},
        .metadata_ready = {next_offset, kNumAICores},
    };

    Buffer<uint32_t> expert_src_m_end{next_offset, (config.num_experts / config.num_ranks) * config.src_m_end_gm_stride, kNumBufferAlignmentBytes};
    Buffer<uint32_t> expert_src_m_stride{next_offset, config.num_experts / config.num_ranks, kL2SectorBytes};
    // Per expert/source/slot: weight bits [63:32], source token-topk index [31:0]
    Buffer<uint64_t> src_token_metadata{next_offset, config.num_src_indices, kNumBufferAlignmentBytes};
    Buffer<uint32_t> expert_token_topk_idx{next_offset, config.num_src_indices, kNumBufferAlignmentBytes};

    struct {
        FP8Buffer acts;
        Buffer<int64_t> topk_idx;
        Buffer<float> topk_weights;
    } input{
        .acts = {next_offset, config.num_max_tokens_per_rank, config.hidden},
        .topk_idx = {next_offset, static_cast<uint64_t>(config.num_max_tokens_per_rank) * config.num_topk, kNumBufferAlignmentBytes},
        .topk_weights = {next_offset, static_cast<uint64_t>(config.num_max_tokens_per_rank) * config.num_topk, kNumBufferAlignmentBytes},
    };

    struct {
        FP8Buffer l1_acts;

        // Metadata uses logical rows and never wraps
        Buffer<float> topk_weights;
        Buffer<uint32_t> src_metadata;

        // Input SF is token-major; quantized Linear2 SF is K-major within each M-block
        FP8Buffer l2_acts;
        Buffer<bf16_t> output;
    } routed{
        .l1_acts = {next_offset, config.num_ring_rows, config.hidden},
        .topk_weights = {next_offset, config.num_full_pool_rows, kNumBufferAlignmentBytes},
        .src_metadata = {next_offset, config.num_full_pool_rows, kNumBufferAlignmentBytes},
        .l2_acts = {next_offset, config.num_ring_rows, config.intermediate_hidden},
        .output = {next_offset, config.num_ring_rows * config.hidden, kNumBufferAlignmentBytes},
    };

    // One BF16 slot per route, plus one combined shared slot
    Buffer<bf16_t> combine_input{
        next_offset, static_cast<uint64_t>(config.num_topk + (config.num_shared_experts > 0)) * config.num_max_tokens_per_rank * config.hidden,
        kNumBufferAlignmentBytes};

    // Shared Linear2 inputs; Linear1 reads original tokens
    FP8Buffer shared_l2_acts{next_offset, config.num_shared_experts > 0 ? config.num_padded_shared_rows : 0, config.shared_intermediate_hidden};

    // Per token/topk: rank [31:24], block [23:0]; block 0xffffff waits for the rank, UINT32_MAX masks the route
    Buffer<uint32_t> token_dst_metadata{
        next_offset, static_cast<uint64_t>(config.num_max_tokens_per_rank) * config.num_topk, kNumBufferAlignmentBytes};

    const uint64_t num_aligned_bytes = aligned(next_offset, 2 * 1024 * 1024);

    constexpr __aicore__ Workspace(Config config, uint32_t num_max_progress_blocks = 0):
        config(config), num_max_progress_blocks(num_max_progress_blocks ? num_max_progress_blocks : config.get_num_max_progress_blocks()) {}
};

} // namespace deep_gemm::mega_moe::layout
