#pragma once

#include <deep_gemm/common.hpp>

namespace deep_gemm::mega_gate::layout {

constexpr uint32_t kNumMaxTokens = 1u << 20;

struct GateArgs {
    __gm__ float* bias;
    __gm__ float* image_bias;
    __gm__ bool* image_token_mask;
    __gm__ bool* mask;
    __gm__ bool* fix_routing_mask;
    __gm__ int32_t* to_physical_map;
    __gm__ int32_t* logical_count;
    __gm__ int64_t* topk_idx;
    __gm__ int64_t* unmapped_topk_idx;
    __gm__ float* topk_weights;
    __gm__ bool* force_random;
    uint64_t unmapped_topk_idx_stride;
    uint32_t num_duplicate_experts;
    uint32_t ep_rank;
    float routed_scaling_factor;
};

// Shared host/device argument record replicated per slice by launch_replicated().
template <typename T>
struct KernelArgs {
    gm_ptr<T> gm_a{0}, gm_b{0};
    gm_ptr<float> gm_d{0};
    uint32_t shape_m;
    GateArgs gate;
};

constexpr uint32_t get_num_packed_rows(uint32_t num_rows, uint32_t num_physical_topk, bool simple_routing) {
    return num_physical_topk > 16 ? 1 : num_rows >= 8 ? 8 : num_rows >= 4 ? 4 :
           (simple_routing and num_rows >= 2 ? 2 : 1);
}

// Shared host/device UB placement; offsets and lengths are in bytes.
struct UBLayout {
    uint32_t num_experts, num_shared_experts, num_rows, num_k_splits;
    uint32_t num_cached_duplicates;
    bool has_image_bias, has_mask, has_fixed_routing;
    bool has_physical_map, has_unmapped_output, has_random_routing;
    uint32_t num_vector_bytes, num_topk;

    const uint32_t stride = aligned(num_experts, FRAC_MN);
    const uint32_t num_physical_topk = num_topk + num_shared_experts;
    const uint32_t weights_offset = aligned(num_physical_topk * sizeof(int64_t), 32u);
    const uint32_t logical_offset = weights_offset + aligned(num_physical_topk * sizeof(float), 32u);
    const uint32_t output_stride = logical_offset + (has_unmapped_output ? aligned(num_topk * sizeof(int64_t), 32u) : 0);
    const uint32_t flag_stride = aligned(num_rows, 32u);
    const uint32_t fixed_stride = aligned(num_topk * sizeof(int64_t), 32u);
    const uint64_t selected = num_rows * num_k_splits * stride * sizeof(float);
    const uint64_t ranking = selected + num_rows * output_stride;
    const bool kSimpleRouting = not (has_mask or has_random_routing or has_fixed_routing or has_image_bias);
    const uint32_t num_packed_rows = get_num_packed_rows(num_rows, num_physical_topk, kSimpleRouting);
    const uint32_t packed_width = num_vector_bytes / sizeof(float) / num_packed_rows;
    const uint32_t num_ranking_bytes = num_packed_rows == 1 ?
        aligned(num_experts * sizeof(float), num_vector_bytes) :
        (aligned(num_experts, packed_width) + packed_width) * num_packed_rows * sizeof(float);
    const uint32_t num_randoms_per_row = 2 * aligned(num_physical_topk, 8u);
    const uint64_t random = ranking + num_ranking_bytes;
    const uint64_t status = random + (has_random_routing ? num_rows * num_randoms_per_row * sizeof(uint32_t) + num_vector_bytes : 0);
    const uint64_t flags = status + aligned(num_rows * sizeof(uint32_t), 32u);
    const uint64_t row_list = flags + 4 * flag_stride;
    const uint32_t num_row_list_bytes = num_packed_rows > 1 and
                                                (has_mask or has_random_routing or has_fixed_routing or has_image_bias) ?
                                            5 * aligned(num_rows, 8u) * sizeof(uint32_t) + 32 :
                                            0;
    const uint64_t fixed = row_list + num_row_list_bytes;
    const uint64_t map = fixed + (has_fixed_routing ? num_rows * fixed_stride : 0);
    const uint64_t counts = map + aligned(uint64_t(num_experts + num_shared_experts) * num_cached_duplicates, FRAC_MN) * sizeof(int32_t);
    const uint64_t bias = counts + (has_physical_map ? aligned(num_experts + num_shared_experts, FRAC_MN) * sizeof(int32_t) : 0);
    const uint64_t args = bias + 2 * stride * sizeof(float);
    const uint64_t num_bytes = args + aligned(sizeof(GateArgs), 32u);
};

// AIC FIX staging: L0C -> UB (dual-DST M-split) -> GM, done by the two AIVs.
constexpr uint32_t kFixChunkRows = 64;
constexpr uint32_t kFixChunkHalfRows = kFixChunkRows / 2;

// Cross-subcore flag ids used by the staged store handshake.
constexpr uint8_t kChunkReadyFlag = 2;  // AIC -> AIV, +buffer parity
constexpr uint8_t kChunkAckFlag = 4;    // AIV -> AIC, +buffer parity
constexpr uint8_t kStoreDoneFlag = 6;   // AIV -> AIC, once per wave

constexpr uint64_t fix_staging_offset(uint64_t aiv_bytes) {
    return (aiv_bytes + 511u) / 512u * 512u;
}

constexpr uint64_t fix_staging_buffer_bytes(uint32_t mad_n) {
    return uint64_t(kFixChunkHalfRows) * mad_n * sizeof(float);
}

constexpr bool use_fix_staging(uint32_t block_m, uint32_t block_n, uint32_t mad_m, uint32_t mad_n,
                               uint64_t aiv_bytes, bool pipeline) {
    const bool single_tile = block_m == mad_m and block_n == mad_n;
    // The staging handshake moves whole 64-row chunks; small tiles cannot amortize it.
    const bool whole_chunks = block_m % kFixChunkRows == 0 and block_m >= 128u;
    // A 512B-aligned FIX row already stores efficiently; offloading only pays off for unaligned rows.
    const bool unaligned_row = (mad_n * sizeof(float)) % 512u != 0;
    return not pipeline and single_tile and whole_chunks and unaligned_row and
           fix_staging_offset(aiv_bytes) + 2 * fix_staging_buffer_bytes(mad_n) <= UBSizeBytes;
}

constexpr uint64_t fix_staging_bytes(uint32_t block_m, uint32_t block_n, uint32_t mad_m, uint32_t mad_n,
                                     uint64_t aiv_bytes, bool pipeline) {
    return use_fix_staging(block_m, block_n, mad_m, mad_n, aiv_bytes, pipeline) ?
               fix_staging_offset(aiv_bytes) + 2 * fix_staging_buffer_bytes(mad_n) : aiv_bytes;
}

} // namespace deep_gemm::mega_gate::layout
