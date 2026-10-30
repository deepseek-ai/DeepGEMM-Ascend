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
