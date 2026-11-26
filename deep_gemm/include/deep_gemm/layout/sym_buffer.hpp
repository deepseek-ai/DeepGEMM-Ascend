#pragma once

#include <cstdint>

#include <deep_gemm/common.hpp>

namespace deep_gemm::layout {

// MegaMoE encodes rank indices in 8 bits
constexpr uint32_t kNumMaxRanks = 1u << 8;

struct SymBuffer {
    int64_t base;
    uint32_t rank_idx;
    __gm__ int64_t* offsets;

#if defined(__CCE__) || defined(__CLION_IDE__)
    // Map through the caller's UB copy of the peer offsets
    template <typename ptr_t, typename offset_ptr_t>
    static constexpr __aicore__ __attribute__((always_inline)) ptr_t map(ptr_t ptr, uint32_t dst_rank_idx, offset_ptr_t offsets) {
        const uint64_t address = __builtin_bit_cast(uint64_t, ptr) + offsets[dst_rank_idx];
        return __builtin_bit_cast(ptr_t, address);
    }
#endif
};

} // namespace deep_gemm::layout
