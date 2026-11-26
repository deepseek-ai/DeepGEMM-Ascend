#pragma once

// Work assignment shared by GEMM kernels.

#include <deep_gemm/common.hpp>   // GemmType
#include <deep_gemm/ascend.hpp>   // utils

namespace deep_gemm {

// Block scheduler. Each cube core independently walks the global (m_block, n_block) grid
// via `get_next_block`, which hands out the persistent block index
// `(++current_iter) * kNumCores + core_id` and maps it to a tile.
//
//   GemmType::Normal — single flat M×N grid, GROUP_M swizzle + N-first snake for L2
//     locality; m/n tails fall back to row-major.
//
//   GemmType::MGroupedContiguousWithPsumLayout — MoE m-grouped, "contiguous" layout. A
//     [M, K] / D [M, N] are one flat buffer with each group's rows concatenated; group g's
//     valid rows occupy [align(psum[g-1], BLOCK_M), psum[g]) in the global M space (group 0
//     starts at 0), where psum = grouped_layout is the prefix-sum-of-actual-rows array of
//     length num_groups. B/SFB are per-group ([G, ...]); the group index is exposed via
//     get_group_idx() for their GM offset. FRAC_MN row groups
//     are balanced across the same number of valid m-blocks. The last valid M block
//     reports output padding rows to the epilogue for zeroing.
//
//   GemmType::KGroupedContiguousWithPsumLayout — MoE k-grouped. A [total_k, M] / B
//     [total_k, N] are single flat buffers with each group's K-elements concatenated
//     (each group's K aligned to BLOCK_K); D [G, M, N] is per-group. grouped_layout[i] =
//     cumulative end K of group i (psum convention). All groups share the same M×N block
//     grid but differ in K-loop length (current_shape_k). A/B K-indexing uses
//     get_k_idx_base() as offset; D uses get_group_idx() * shape_m * shape_n; SF uses
//     get_sf_idx_base() * shape_mn as offset.
//
//     Because each core's persistent block index is monotonically increasing, the cumsum
//     walk (current_group_idx etc.) only ever advances forward within a core — re-running
//     it from the cached state for each new (larger) block index is correct.
//
//   GemmType::Batched — standard batched matmul D[b] = A[b] @ B[b]^T. Every batch has the
//     SAME regular M×N×K, so the grid is the Normal M×N grid replicated `num_groups` (=
//     batch) times. The batch index is block_idx / num_blocks; the in-batch index reuses
//     the Normal swizzle. A/B/D and SFA/SFB all offset by the batch (in the kernel, using
//     each operand's runtime per-batch stride); get_batch_idx() exposes the batch.

// Map a flat block index to (m_block, n_block) with the GROUP_M swizzle, optional N XOR
// and super-row snake. Shared by `Scheduler` and kernels that dispatch their own groups.
__aicore__ inline void swizzled_block(uint32_t block_idx, uint32_t num_m_blocks, uint32_t num_n_blocks,
                                      uint32_t group_m, bool xor_n, bool snake,
                                      uint32_t& m_block_idx, uint32_t& n_block_idx) {
    // Small M needs no swizzle; the last <group_m rows run row-major so cores share A.
    if (num_m_blocks < group_m) {
        m_block_idx = block_idx / num_n_blocks;
        n_block_idx = block_idx % num_n_blocks;
        return;
    }
    const uint32_t tail_m_start = num_m_blocks - num_m_blocks % group_m;
    const uint32_t tail_start = tail_m_start * num_n_blocks;
    if (block_idx >= tail_start) {
        const uint32_t tail = block_idx - tail_start;
        m_block_idx = tail_m_start + tail / num_n_blocks;
        n_block_idx = tail % num_n_blocks;
        return;
    }

    const uint32_t group_n = 32 / group_m;
    const uint32_t n_groups = num_n_blocks / group_n;
    const uint32_t blocks_per_superrow = group_m * num_n_blocks;
    const uint32_t full_blocks = n_groups * 32;
    const uint32_t superrow = block_idx / blocks_per_superrow;
    const uint32_t within = block_idx - superrow * blocks_per_superrow;
    if (within < full_blocks) {
        const uint32_t group = within / 32;
        const uint32_t local = within - group * 32;
        const uint32_t m_local = local & (group_m - 1);
        uint32_t n_local = local / group_m;
        if (xor_n) n_local ^= m_local;
        const uint32_t n_group = snake && (superrow & 1) ? n_groups - 1 - group : group;
        m_block_idx = superrow * group_m + m_local;
        n_block_idx = n_group * group_n + n_local;
    } else {
        const uint32_t rem = within - full_blocks;
        m_block_idx = superrow * group_m + (rem & (group_m - 1));
        n_block_idx = n_groups * group_n + rem / group_m;
    }
}

template <
    GemmType kGemmType,
    uint32_t BLOCK_M, uint32_t BLOCK_N,
    uint32_t kNumCores,
    uint32_t kAlignment=256u,
    uint32_t kGroupM = 4, bool kXorN = true, bool kSnake = true
>
struct Scheduler {
    static_assert(kGroupM == 4 || kGroupM == 8, "kGroupM must be 4 or 8");

    int current_iter;
    uint32_t core_id;
    uint32_t shape_m;
    uint32_t shape_n;
    uint32_t num_m_blocks;
    uint32_t num_n_blocks;
    uint32_t num_blocks;       // total valid (m, n) blocks (Normal only)

    __gm__ int32_t* grouped_layout;   // prefix-sum, length num_groups
    uint32_t num_groups;

    uint32_t current_group_idx = 0;       // group the current block belongs to

    // m-grouped only
    uint32_t last_psum_m = 0;             // align(psum[group-1], BLOCK_M) = current group start row
    uint32_t current_psum_m = 0;          // psum[group] = current group's valid end row
    uint32_t current_m_block_cumsum = 0;  // # valid M tasks in previous groups
    uint32_t num_m_fractals_per_block = 0;
    uint32_t num_larger_m_blocks = 0;
    uint32_t num_m_padding = 0;           // final block's FRAC_MN padding rows

    // k-grouped only
    uint32_t current_shape_k = 0;         // per-group K length
    uint32_t last_psum_k = 0;             // align(psum[group-1], kAlignment) = current group start K
    uint32_t current_psum_k = 0;          // psum[group] = current group's valid end K
    uint32_t current_sf_k_cumsum = 0;     // cumulative SF-K offset (K/64)
    uint32_t current_k_block_cumsum = 0;  // # MxN blocks consumed by all previous K groups

    // batched only
    uint32_t current_batch_idx = 0;       // batch the current block belongs to

    // Normal constructor (no grouping).
    __aicore__ Scheduler(uint32_t core_id, uint32_t shape_m, uint32_t shape_n)
        : Scheduler(core_id, shape_m, shape_n, nullptr, 0) {}

    // Grouped constructor: grouped_layout/num_groups are ignored for Normal.
    __aicore__ Scheduler(uint32_t core_id, uint32_t shape_m, uint32_t shape_n,
                                __gm__ int32_t* grouped_layout_, uint32_t num_groups_)
        : current_iter(-1), core_id(core_id),
          shape_m(shape_m), shape_n(shape_n),
          num_m_blocks(ceil_div(shape_m, BLOCK_M)),
          num_n_blocks(ceil_div(shape_n, BLOCK_N)),
          num_blocks(ceil_div(shape_m, BLOCK_M) * ceil_div(shape_n, BLOCK_N)),
          grouped_layout(grouped_layout_), num_groups(num_groups_)
    {
        if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            // Group 0 starts at row 0; its valid end is psum[0].
            update_m_group();
        } else if constexpr (kGemmType == GemmType::KGroupedContiguousWithPsumLayout) {
            // Group 0 starts at K 0; its valid end is psum[0].
            current_psum_k = static_cast<uint32_t>(grouped_layout[0]);
            current_shape_k = current_psum_k;
        }
    }

    // Valid blocks balance FRAC_MN rows within each expert.
    __aicore__ uint32_t get_global_m_idx(uint32_t m_block_idx) const {
        if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            const uint32_t group_m_block_idx = m_block_idx - last_psum_m / BLOCK_M;
            return last_psum_m + (group_m_block_idx * num_m_fractals_per_block + min(group_m_block_idx, num_larger_m_blocks)) * FRAC_MN;
        } else {
            return m_block_idx * BLOCK_M;
        }
    }

    __aicore__ uint32_t get_global_n_idx(uint32_t n_block_idx) const {
        return n_block_idx * BLOCK_N;
    }

    // The group index of the block returned by the most recent get_next_block.
    __aicore__ uint32_t get_group_idx() const { return current_group_idx; }

    // The batch index of the block returned by the most recent get_next_block (Batched).
    __aicore__ uint32_t get_batch_idx() const { return current_batch_idx; }

    // K-grouped: cumsum offsets for A/B and SF indexing.
    __aicore__ uint32_t get_k_idx_base() const { return last_psum_k; }
    __aicore__ uint32_t get_sf_idx_base() const { return current_sf_k_cumsum; }

    // Valid rows of an m-block
    __aicore__ uint32_t get_actual_m(uint32_t m_block_idx) const {
        if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            const uint32_t group_m_block_idx = m_block_idx - last_psum_m / BLOCK_M;
            const uint32_t actual_m = (num_m_fractals_per_block + (group_m_block_idx < num_larger_m_blocks)) * FRAC_MN;
            return actual_m - (group_m_block_idx + 1 == num_m_blocks ? num_m_padding : 0);
        } else {
            return min(shape_m - m_block_idx * BLOCK_M, BLOCK_M);
        }
    }
    __aicore__ uint32_t get_num_padding_rows(uint32_t m_block_idx) const {
        if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            if (m_block_idx - last_psum_m / BLOCK_M + 1 == num_m_blocks)
                return aligned(current_psum_m, kAlignment) - current_psum_m;
        }
        return 0;
    }

    __aicore__ uint32_t get_actual_n(uint32_t n_block_idx) const {
        return min(shape_n - n_block_idx * BLOCK_N, BLOCK_N);  // caller computes the Normal n-tail from shape_n
    }

    __aicore__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        uint32_t block_idx = static_cast<uint32_t>(++current_iter) * kNumCores + core_id;

        if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            static_assert(kAlignment == BLOCK_M);
            while (true) {
                if (block_idx < (current_m_block_cumsum + num_m_blocks) * num_n_blocks)
                    break;
                if (++current_group_idx == num_groups)
                    return false;
                update_m_group();
            }
            // Map the in-group block index through the same swizzle, then shift the m-block
            // to its global position (last_psum_m is BLOCK_M-aligned so the divide is exact).
            uint32_t in_group_idx = block_idx - current_m_block_cumsum * num_n_blocks;
            get_swizzled_block(in_group_idx, num_m_blocks, m_block_idx, n_block_idx);
            m_block_idx += last_psum_m / BLOCK_M;
            return true;
        } else if constexpr (kGemmType == GemmType::KGroupedContiguousWithPsumLayout) {
            // Advance groups until block_idx lands in the current group's block range.
            // All groups share the same M×N grid, so each group spans exactly num_blocks.
            while (true) {
                if (current_shape_k > 0 and block_idx < current_k_block_cumsum + num_blocks)
                    break;
                if (++current_group_idx == num_groups)
                    return false;
                last_psum_k = aligned(current_psum_k, kAlignment);
                current_psum_k = static_cast<uint32_t>(grouped_layout[current_group_idx]);
                current_sf_k_cumsum = last_psum_k / MX_SF_DIVISOR;
                if (current_shape_k > 0) // if shape_k == 0, there are no blocks to consume, so don't increment the cumsum
                    current_k_block_cumsum += num_blocks;
                current_shape_k = current_psum_k - last_psum_k;
            }
            uint32_t in_group_idx = block_idx - current_k_block_cumsum;
            get_swizzled_block(in_group_idx, num_m_blocks, m_block_idx, n_block_idx);
            return true;
        } else if constexpr (kGemmType == GemmType::Batched) {
            // Regular batch dim: each batch spans the full num_blocks M×N grid. (num_groups
            // carries the batch count.) Map block_idx to its batch + in-batch swizzle.
            current_batch_idx = block_idx / num_blocks;
            if (current_batch_idx >= num_groups)
                return false;
            uint32_t in_batch_idx = block_idx - current_batch_idx * num_blocks;
            get_swizzled_block(in_batch_idx, num_m_blocks, m_block_idx, n_block_idx);
            return true;
        } else {
            if (block_idx >= num_blocks)
                return false;
            get_swizzled_block(block_idx, num_m_blocks, m_block_idx, n_block_idx);
            return true;
        }
    }

private:
    __aicore__ void update_m_group() {
        last_psum_m = aligned(current_psum_m, kAlignment);
        current_psum_m = static_cast<uint32_t>(grouped_layout[current_group_idx]);
        if (current_group_idx != 0)
            current_m_block_cumsum += num_m_blocks;
        const uint32_t num_tokens = current_psum_m - last_psum_m;
        num_m_blocks = ceil_div(num_tokens, BLOCK_M);
        const uint32_t num_m_fractals = ceil_div(num_tokens, FRAC_MN);
        num_m_fractals_per_block = num_m_blocks ? num_m_fractals / num_m_blocks : 0;
        num_larger_m_blocks = num_m_blocks ? num_m_fractals % num_m_blocks : 0;
        num_m_padding = num_m_fractals * FRAC_MN - num_tokens;
    }

    // Map a (group-local) block index to (m_block_idx, n_block_idx) using the GROUP_M
    // swizzle + N-first snake. `local_num_m_blocks` is the m-block count for this region
    // (the whole grid for Normal, the current group for m-grouped).
    __aicore__ void get_swizzled_block(uint32_t block_idx, uint32_t local_num_m_blocks,
                                              uint32_t& m_block_idx, uint32_t& n_block_idx) const {
        ::deep_gemm::swizzled_block(block_idx, local_num_m_blocks, num_n_blocks,
                                    kGroupM, kXorN, kSnake, m_block_idx, n_block_idx);
    }
};

} // namespace deep_gemm
