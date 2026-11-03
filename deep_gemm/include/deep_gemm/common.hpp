#pragma once

// CLion uses a regular C++ frontend, so load the CCE shim before any CANN header.
// The real device compiler does not define __CLION_IDE__ and is unaffected.
#ifdef __CLION_IDE__
#define __CCE_STUB_DISABLE_ADDRESS_SPACE_QUALIFIERS__
#define __CCE_STUB_USE_LOCAL_WRAPPER__
#define __CCE_AICORE_SUPPORT_SIMT__
#define __clang__
#include <cce_stubs/cce_stubs.h>
#endif

// Shared definitions used by BOTH host (csrc/) and device (deep_gemm/include) code.
// This header is pure C++: no <kernel_operator.h>, no intrinsics, so it can be included
// from host translation units. Keep it dependency-free.

#include <cstdint>
#include <type_traits>

// The device compiler (bisheng) defines __CCE__ and provides __aicore__ / __gm__. On the
// host these are not keywords, so define them away — this lets a handful of small types
// (e.g. gm_ptr below) be shared verbatim between host and device translation units.
#ifndef __CCE__
#define __aicore__
#define __gm__
#endif

// Keep both mixed-kernel paths visible to the host indexer.
#ifdef __CLION_IDE__
#define DG_IS_AIC constexpr (true)
#define DG_IS_AIV constexpr (true)
#else
#define DG_IS_AIC ASCEND_IS_AIC
#define DG_IS_AIV ASCEND_IS_AIV
#endif

namespace deep_gemm {

#ifdef __CCE__
template <typename T1, typename T2>
__aicore__ constexpr auto ceil_div(T1 a, T2 b) -> decltype(a + b) { return (a + b - 1) / b; }

template <typename T1, typename T2>
__aicore__ constexpr auto align(T1 a, T2 b) -> decltype(a + b) { return (a + b - 1) / b * b; }
#endif

template <typename T1, typename T2>
__aicore__ constexpr auto aligned(T1 a, T2 b) -> decltype(a + b) { return (a + b - 1) / b * b; }

// Ascend 950DT core topology
inline constexpr uint32_t kNumAICores = 32;
inline constexpr uint32_t kNumAIVs = 2 * kNumAICores;

inline constexpr uint32_t L1SizeBytes = 512 * 1024;   // L1 per core
inline constexpr uint32_t L0ASizeBytes = 64 * 1024;   // L0A
inline constexpr uint32_t L0BSizeBytes = 64 * 1024;   // L0B
inline constexpr uint32_t L0CSizeBytes = 256 * 1024;  // L0C (fp32 accumulator)
inline constexpr uint32_t UBSizeBytes = 256 * 1024;   // UB
inline constexpr uint32_t kDCacheLineBytes = 64;
inline constexpr uint32_t kL2SectorBytes = 128;
inline constexpr uint32_t kNumPipeEvents = 8;

inline constexpr uint32_t FRAC_MN = 16;      // NZ fractal M/N rows
inline constexpr uint32_t MX_ADDR_DIV = 16;  // MX dst address encoding divisor
inline constexpr uint32_t MX_SF_DIVISOR = 64;  // K-elements per SF pair

#ifdef __CCE__
// only bisheng has float4_e2m1x2_t defined; host code does not
template <typename T>
__aicore__ constexpr bool is_fp4() { return std::is_same_v<T, float4_e2m1x2_t>; }

template <typename T>
__aicore__ constexpr uint32_t get_frac_k() {
    if constexpr (is_fp4<T>()) {
        return 64;  // one MX SF pair
    } else {
        return 32 / sizeof(T);  // 32 bytes per C0 / K-elements per byte
    }
}
#endif

enum class Major : uint8_t { K, MN };

__aicore__ constexpr bool is_k_major(Major m) { return m == Major::K; }
__aicore__ constexpr bool is_mn_major(Major m) { return m == Major::MN; }

enum class GemmType : uint8_t {
    Normal,
    MGroupedContiguousWithPsumLayout,
    KGroupedContiguousWithPsumLayout,
    Batched,
};

// Runtime state shared by host classes and device operators.
namespace epilogue {
struct Args {
    float alpha = 1.0f;
    uintptr_t sfd = 0;
    uint64_t sfd_stride = 0;
    uint32_t shape_n = 0;
    uintptr_t bias = 0;
};
} // namespace epilogue

__aicore__ constexpr bool is_m_grouped(GemmType t) {
    return t == GemmType::MGroupedContiguousWithPsumLayout;
}
__aicore__ constexpr bool is_k_grouped(GemmType t) {
    return t == GemmType::KGroupedContiguousWithPsumLayout;
}
__aicore__ constexpr bool is_grouped(GemmType t) {
    return is_m_grouped(t) || is_k_grouped(t);
}
__aicore__ constexpr bool is_batched(GemmType t) {
    return t == GemmType::Batched;
}

#ifndef __CCE__
// Mirror C API enum names for host JIT formatting without Bisheng headers.
enum class asc_load_l2_cache_mode : uint8_t {
    NORMAL_FIRST_VICTIM = 0,
    NORMAL_LAST_VICTIM = 1,
    NORMAL_PERSISTENT = 2,
    NOTALLOC_KEEP = 4,
    NOTALLOC_CLEAN = 5,
    NOTALLOC_DROP = 6,
};

enum class asc_store_l2_cache_mode : uint8_t {
    NORMAL_FIRST_VICTIM = 0,
    NORMAL_LAST_VICTIM = 1,
    NORMAL_PERSISTENT = 2,
    NOTALLOC_CLEAN = 4,
};
#endif

template <typename T, Major kMajor = Major::K>
struct gm_ptr {
    uintptr_t addr;
    uint64_t stride_outer;
    uint64_t stride_batch;

    using ptr_t = __gm__ T*;
    using dtype_t = T;
    static constexpr Major major = kMajor;

    constexpr __aicore__ gm_ptr(uint64_t stride_outer, uintptr_t addr = 0, uint64_t stride_batch = 0) :
        addr(addr), stride_outer(stride_outer), stride_batch(stride_batch) {}

#ifdef __CCE__
    template <typename dst_dtype_t = dtype_t>
    __aicore__ __gm__ dst_dtype_t* ptr() const {
        return reinterpret_cast<__gm__ dst_dtype_t*>(addr);
    }

    constexpr __aicore__ gm_ptr index(uint64_t mn_idx, uint64_t k_idx) const {
        constexpr uint64_t pack_size = is_fp4<dtype_t>() ? 2 : 1;
        gm_ptr r = *this;
        if constexpr (kMajor == Major::K) {
            r.addr += sizeof(dtype_t) * (mn_idx * stride_outer + k_idx / pack_size);
        } else {
            r.addr += sizeof(dtype_t) * (k_idx * stride_outer + mn_idx / pack_size);
        }
        return r;
    }

    constexpr __aicore__ gm_ptr group(uint64_t group_idx, uint64_t shape_mn, uint64_t shape_k) const {
        constexpr uint64_t pack_size = is_fp4<dtype_t>() ? 2 : 1;
        gm_ptr r = *this;
        r.addr += group_idx * shape_mn * shape_k / pack_size * sizeof(dtype_t);
        return r;
    }
#endif

    constexpr __aicore__ gm_ptr operator+(uint64_t elems) const {
        gm_ptr r = *this;
        r.addr += elems * sizeof(dtype_t);
        return r;
    }

    constexpr __aicore__ gm_ptr batch(uint64_t idx) const {
        gm_ptr r = *this;
        r.addr += idx * stride_batch * sizeof(dtype_t);
        return r;
    }
};

} // namespace deep_gemm
