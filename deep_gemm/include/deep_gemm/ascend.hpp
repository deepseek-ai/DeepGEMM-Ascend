#pragma once

#include <c_api/asc_simd.h>
#include <utils/debug/asc_assert.h>
#include <deep_gemm/common.hpp>

// Device primitives, split by abstraction layer:
//   tile_ptr — operand_ptr / result_ptr (shape-carrying device pointers) + l*_ptr aliases
//   copy     — GM/L1/L0/UB data movement, scale-factor load, mad, quant-mode inference
//   sync     — intra-block / flag / ffts synchronization
#include <deep_gemm/ascend/tile_ptr.hpp>
#include <deep_gemm/ascend/copy.hpp>
#include <deep_gemm/ascend/sync.hpp>

namespace deep_gemm {

// Read this core's 512B-aligned slice from launch_replicated() without emitting whole-struct DC_PRELOADI.
template <typename T>
__aicore__ __forceinline__ const T& get_replicated_args() {
    static_assert(sizeof(T) <= kArgCopyStrideBytes and alignof(T) <= kArgCopyStrideBytes and
                  kNumArgCopies > 0 and (kNumArgCopies & (kNumArgCopies - 1)) == 0);
    uint32_t idx = asc_get_block_idx();
    if DG_IS_AIV { idx = kNumAICores + idx * 2 + asc_get_sub_block_id(); }
    return *reinterpret_cast<const T*>(
        static_cast<uintptr_t>(get_para_base()) + (idx & (kNumArgCopies - 1)) * kArgCopyStrideBytes);
}

} // namespace deep_gemm
