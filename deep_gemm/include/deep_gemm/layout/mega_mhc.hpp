#pragma once

#include <deep_gemm/common.hpp>

namespace deep_gemm::layout::mega_mhc {

// Model geometry
constexpr uint32_t kNumRoutes = 4;
constexpr uint32_t kNumHCOutputs = kNumRoutes * (kNumRoutes + 2);
constexpr uint32_t kNumFP32toBF16Splits = 3;

// Main block dimensions
constexpr uint32_t kNumSubAIVs = 2;
constexpr uint32_t BLOCK_M = 32;
constexpr uint32_t BLOCK_N = 32;
constexpr uint32_t BLOCK_K = 512;
constexpr uint32_t MAD_K = 64;
constexpr uint32_t SUB_BLOCK_M = BLOCK_M / kNumSubAIVs;
constexpr uint32_t BLOCK_H = BLOCK_K / kNumRoutes;
constexpr uint32_t kNumKSplits = 8;

// Split-K partitions
static_assert(kNumKSplits > 0 && kNumAICores >= kNumKSplits && kNumAICores % kNumKSplits == 0);
constexpr uint32_t kNumAIVCores = kNumAICores * kNumSubAIVs;
constexpr uint32_t kNumAICoresPerSplit = kNumAICores / kNumKSplits;

// Fixed Prenorm L0C residency budget per AIC, independent of the M block size
constexpr uint32_t kNumPrenormL0CBytes = 32 * 1024;
constexpr uint32_t kNumTokensPerChunk = kNumPrenormL0CBytes / (BLOCK_N * sizeof(float)) * kNumAICoresPerSplit;

// Host/device argument packs for the Mix and Norm/Quant consumers
struct MixArgs {
    gm_ptr<float> scales;
    gm_ptr<float> bases;
    gm_ptr<float> new_prev_mix;
    gm_ptr<float> new_post_mix;
    gm_ptr<float> new_comb_res_mix;
    float hc_norm_eps;
    float hc_pre_eps;
    float hc_post_scale;
    float sinkhorn_eps;
    uint32_t num_sinkhorn_iters;
};

struct NormArgs {
    // Host storage types have the same ABI as the device BF16/FP8 types
#ifdef __CCE__
    using bf16_t = bfloat16_t;
    using fp8_t = float8_e4m3_t;
#else
    using bf16_t = uint16_t;
    using fp8_t = uint8_t;
#endif

    uint32_t shape_m;
    gm_ptr<bf16_t> weight;
    gm_ptr<bf16_t> new_residual;
    float eps;
    float scale;
    gm_ptr<bf16_t> y_bf16;
    gm_ptr<fp8_t> y_fp8;
    // SF packs two UE8M0 scales per uint16_t; stride is the non-contiguous dimension's stride in elements
    gm_ptr<uint16_t> y_sf;
    uint32_t y_sf_stride;
};

struct KernelArgs {
    // Pack all AIC inputs (residual, fn, workspace, norm_args.shape_m) into Line 0 (0..59B).
    gm_ptr<NormArgs::bf16_t> gm_residual;
    gm_ptr<float> gm_fn;
    __gm__ float* workspace;
    NormArgs norm_args;
    gm_ptr<NormArgs::bf16_t> gm_layer_input;
    gm_ptr<float> gm_shifted_prev_mix, gm_post_mix, gm_comb_res_mix;
    MixArgs mix_args;
};

// Workspace offsets and lengths are in FP32 elements
struct WorkspaceBuffer {
    uint64_t offset;
    uint64_t num_elements;

    constexpr __aicore__ __gm__ float* get_ptr(__gm__ float* base) const {
        return base + offset;
    }
};

constexpr uint32_t kWorkspaceBlockM = 64;

// Prenorm/HC workspace M order is [M block, K split, row within block]
template <uint32_t kNumKSplits, uint32_t BLOCK_M = 1>
constexpr __aicore__ uint64_t get_workspace_m_idx(uint32_t m_idx, uint32_t k_split_idx = 0) {
    static_assert(BLOCK_M > 0 && kWorkspaceBlockM % BLOCK_M == 0);
    return (uint64_t(m_idx / kWorkspaceBlockM) * kNumKSplits + k_split_idx) * kWorkspaceBlockM + m_idx % kWorkspaceBlockM;
}

template <uint32_t SHAPE_N>
constexpr __aicore__ uint32_t get_prenorm_split_stride_bytes() {
    return kWorkspaceBlockM * SHAPE_N * sizeof(float);
}

constexpr __aicore__ uint32_t get_sqr_sum_split_stride_bytes() {
    return kWorkspaceBlockM * sizeof(float);
}

template <uint32_t kNumKSplits, uint32_t M_STRIDE>
constexpr __aicore__ uint32_t get_sqr_sum_m_stride_bytes() {
    static_assert(M_STRIDE % kWorkspaceBlockM == 0);
    return M_STRIDE * kNumKSplits * sizeof(float);
}

// Prenorm [ceil(M/64), splits, 64, N], then HC [ceil(M/64), splits, 64]
// RMS reserves the same capacity as HC, packed into smaller M groups by the kernel
template <uint32_t SHAPE_N, uint32_t kNumKSplits>
struct MegaHCWorkspaceLayout {
    const uint64_t num_rows;
    const WorkspaceBuffer prenorm{0, num_rows * SHAPE_N};
    const WorkspaceBuffer hc_sqr_sum{prenorm.offset + prenorm.num_elements, num_rows};
    const WorkspaceBuffer rms_sqr_sum{hc_sqr_sum.offset + hc_sqr_sum.num_elements, num_rows};
    const uint64_t num_bytes = (rms_sqr_sum.offset + rms_sqr_sum.num_elements) * sizeof(float);

    constexpr __aicore__ explicit MegaHCWorkspaceLayout(uint32_t shape_m)
        : num_rows(((uint64_t(shape_m) + kWorkspaceBlockM - 1) / kWorkspaceBlockM) * kNumKSplits * kWorkspaceBlockM) {}
};

} // namespace deep_gemm::layout::mega_mhc
