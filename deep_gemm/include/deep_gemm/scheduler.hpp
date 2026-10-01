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
//     get_group_idx() for their GM offset. The padding tail of each group's last m-block is
//     truncated by the normal actual_m mechanism (get_actual_m).
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
    uint32_t tail_m_start;     // first m_block_idx in the row-major tail region (Normal only)
    uint32_t tail_start;       // first block_idx in the row-major tail region (Normal only)

    __gm__ int32_t* grouped_layout;   // prefix-sum, length num_groups
    uint32_t num_groups;

    uint32_t current_group_idx = 0;       // group the current block belongs to

    // m-grouped only
    uint32_t last_psum_m = 0;             // align(psum[group-1], BLOCK_M) = current group start row
    uint32_t current_psum_m = 0;          // psum[group] = current group's valid end row
    uint32_t current_m_block_cumsum = 0;  // # m-blocks consumed by all previous groups

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
          // swizzle covers floor-aligned m_blocks; the remaining < kGroupM rows use row-major
          tail_m_start(num_m_blocks >= kGroupM ? (num_m_blocks - num_m_blocks % kGroupM) : 0),
          tail_start(tail_m_start * num_n_blocks),
          grouped_layout(grouped_layout_), num_groups(num_groups_)
    {
        if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            // Group 0 starts at row 0; its valid end is psum[0].
            current_psum_m = static_cast<uint32_t>(grouped_layout[0]);
            num_m_blocks = ceil_div(current_psum_m, BLOCK_M);
        } else if constexpr (kGemmType == GemmType::KGroupedContiguousWithPsumLayout) {
            // Group 0 starts at K 0; its valid end is psum[0].
            current_psum_k = static_cast<uint32_t>(grouped_layout[0]);
            current_shape_k = current_psum_k;
        }
    }

    // Global element offset for a block index (M/N axis). Group offset for B/SF is applied
    // separately in the kernel via get_group_idx(); A/D are flat (no group offset) for
    // m-grouped; for k-grouped, A/B are flat and D is per-group.
    __aicore__ uint32_t get_global_idx(uint32_t block_size, uint32_t block_idx) const {
        return block_idx * block_size;
    }

    // The group index of the block returned by the most recent get_next_block.
    __aicore__ uint32_t get_group_idx() const { return current_group_idx; }

    // The batch index of the block returned by the most recent get_next_block (Batched).
    __aicore__ uint32_t get_batch_idx() const { return current_batch_idx; }

    // K-grouped: cumsum offsets for A/B and SF indexing.
    __aicore__ uint32_t get_k_idx_base() const { return last_psum_k; }
    __aicore__ uint32_t get_sf_idx_base() const { return current_sf_k_cumsum; }

    // Effective M rows of an m-block: BLOCK_M, except the current group's last m-block,
    // which is truncated to the group's valid-row tail (psum[group] - m_block_start).
    __aicore__ uint32_t get_actual_m(uint32_t m_block_idx) const {
        if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            // TODO: remove this and add tail handling for ensure_zero_padding=True
            return BLOCK_M;
        } else {
            return min(shape_m - m_block_idx * BLOCK_M, BLOCK_M);
        }
    }
    __aicore__ uint32_t get_actual_n(uint32_t n_block_idx) const {
        return min(shape_n - n_block_idx * BLOCK_N, BLOCK_N);  // caller computes the Normal n-tail from shape_n
    }

    __aicore__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        uint32_t block_idx = static_cast<uint32_t>(++current_iter) * kNumCores + core_id;

        if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            // TODO: remove this and add tail handling for ensure_zero_padding=True
            static_assert(kAlignment == BLOCK_M);
            while (true) {
                if (block_idx < (current_m_block_cumsum + num_m_blocks) * num_n_blocks)
                    break;
                if (++current_group_idx == num_groups)
                    return false;
                last_psum_m = aligned(current_psum_m, kAlignment);
                current_psum_m = static_cast<uint32_t>(grouped_layout[current_group_idx]);
                current_m_block_cumsum += num_m_blocks;
                num_m_blocks = current_psum_m > last_psum_m ?
                    ceil_div(current_psum_m - last_psum_m, BLOCK_M) : 0;
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
                current_shape_k = current_psum_k > last_psum_k ? current_psum_k - last_psum_k : 0;
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
    // Map a (group-local) block index to (m_block_idx, n_block_idx) using the GROUP_M
    // swizzle + N-first snake. `local_num_m_blocks` is the m-block count for this region
    // (the whole grid for Normal, the current group for m-grouped).
    __aicore__ void get_swizzled_block(uint32_t block_idx, uint32_t local_num_m_blocks,
                                              uint32_t& m_block_idx, uint32_t& n_block_idx) const {
        uint32_t local_tail_m_start = local_num_m_blocks >= kGroupM
                                    ? (local_num_m_blocks - local_num_m_blocks % kGroupM) : 0;
        uint32_t local_tail_start = local_tail_m_start * num_n_blocks;

        // Fast path: small M falls back to simple row-major (no swizzle needed)
        if (local_num_m_blocks < kGroupM) {
            m_block_idx = block_idx / num_n_blocks;
            n_block_idx = block_idx % num_n_blocks;
        } else if (block_idx >= local_tail_start) {
            // Tail rows (last < kGroupM rows): row-major (n fast, m slow → reuses A across cores)
            uint32_t tail = block_idx - local_tail_start;
            m_block_idx = local_tail_m_start + tail / num_n_blocks;
            n_block_idx = tail % num_n_blocks;
        } else {
            // Swizzle path on aligned region
            constexpr uint32_t kGroupN = 32 / kGroupM;
            uint32_t n_groups = num_n_blocks / kGroupN;
            uint32_t blocks_per_superrow = kGroupM * num_n_blocks;
            uint32_t full_blocks = n_groups * 32;

            uint32_t superrow = block_idx / blocks_per_superrow;
            uint32_t within   = block_idx - superrow * blocks_per_superrow;

            if (within < full_blocks) {
                uint32_t grp   = within / 32;
                uint32_t local = within - grp * 32;
                uint32_t m_local = local & (kGroupM - 1);  // local % kGroupM (power of 2)
                uint32_t n_local = local / kGroupM;
                if constexpr (kXorN) n_local ^= m_local;

                uint32_t ng = grp;
                if (kSnake && (superrow & 1)) ng = n_groups - 1 - ng;

                m_block_idx = superrow * kGroupM + m_local;
                n_block_idx = ng * kGroupN + n_local;
            } else {
                uint32_t rem_within = within - full_blocks;
                uint32_t m_local = rem_within & (kGroupM - 1);
                uint32_t n_local = rem_within / kGroupM;

                m_block_idx = superrow * kGroupM + m_local;
                n_block_idx = n_groups * kGroupN + n_local;
            }
        }
    }
};

} // namespace deep_gemm
