#pragma once

#include <optional>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include <torch/python.h>

#include <deep_jit/utils/exception.hpp>

#include "layout.hpp"
#include "../jit_kernels/bf16_gemm.hpp"
#include "../jit_kernels/fp8_dequant_gemm.hpp"
#include "../jit_kernels/fp8_gemm.hpp"
#include "../utils/layout.hpp"

namespace deep_gemm::gemm_api {

namespace py = pybind11;

static void dispatch_fp8_fp4_gemm(const GemmDesc& desc) {
    auto ab_dtype = std::tie(desc.a_dtype, desc.b_dtype);
    if (ab_dtype == std::tie(torch::kFloat8_e4m3fn, torch::kFloat8_e4m3fn)) {
        deep_gemm::launch_fp8_gemm(desc);
    } else if (ab_dtype == std::tie(kPackedFP4, kPackedFP4)) {
        deep_gemm::launch_fp4_gemm(desc);
    } else if (ab_dtype == std::tie(torch::kFloat8_e4m3fn, kPackedFP4)) {
        deep_gemm::launch_fp8_dequant_gemm(desc);
    } else {
        DJ_HOST_ASSERT(false, "Unsupported operand dtypes");
    }
}

static uint32_t expected_k_from_ks(const std::optional<std::vector<int>>& ks_cpu, uint32_t default_k = 4096u) {
    if (ks_cpu.has_value() && !ks_cpu->empty()) {
        uint32_t expected_k = 0;
        for (auto k_i : *ks_cpu)
            expected_k += static_cast<uint32_t>(k_i);
        return std::max<uint32_t>(1, expected_k / static_cast<uint32_t>(ks_cpu->size()));
    }
    return default_k;
}

/**
 * Get the gemm desc object
 *
 *                A                 B                CD
 *
 * Normal:       [M, K]            [N, K]            [M, N]
 * MGrouped:     [M, K]         [G, N, K]            [M, N]
 * KGrouped(*):  [M, K]            [N, K]         [G, M, N]
 * Batched:   [B, M, K]         [B, N, K]         [B, M, N]
 *
 * (*) for KGrouped, the caller should call A.transpose(0, 1) and B.transpose(0, 1) before passing to this function,
 *     this is not a real transpose, but just make the shape of A, B to [M, K] and [N, K] for the purpose of getting
 *     correct major, shape and stride information, the actual data layout is not changed
 *
 */
static std::optional<GemmDesc> get_gemm_desc(
    GemmType gemm_type,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& d,
    const std::optional<torch::Tensor>& c,
    std::shared_ptr<EpilogueClass> epilogue_class,
    const std::optional<torch::Tensor>& sfa_in,
    const std::optional<torch::Tensor>& sfb_in,
    const std::optional<std::tuple<int, int, int>>& recipe_in,
    const std::optional<std::tuple<int, int>>& recipe_a_in,
    const std::optional<std::tuple<int, int>>& recipe_b_in,
    const std::optional<torch::Tensor>& grouped_layout,
    const std::string& compiled_dims,
    bool disable_ue8m0_cast = false,
    const std::optional<uint32_t>& expected_m = std::nullopt,
    const std::optional<uint32_t>& expected_k = std::nullopt
) {
    uint32_t g = 0, m = 0, n = 0, k = 0;
    auto major_a = get_major_type_ab(a);
    auto major_b = get_major_type_ab(b);
    switch (gemm_type) {
        case GemmType::Normal:
            std::tie(m, n, k) = get_shape_by_spec<"mnk">("mk"_sp = a, "nk"_sp = b, "mn"_sp = d);
            break;
        case GemmType::Batched:
            std::tie(g, m, n, k) = get_shape_by_spec<"bmnk">("bmk"_sp = a, "bnk"_sp = b, "bmn"_sp = d);
            break;
        case GemmType::MGroupedContiguousWithPsumLayout:
            std::tie(g, m, n, k) = get_shape_by_spec<"gmnk">("mk"_sp = a, "gnk"_sp = b, "mn"_sp = d, "g"_sp = grouped_layout.value());
            DJ_HOST_ASSERT(major_a == Major::K);
            break;
        case GemmType::KGroupedContiguousWithPsumLayout:
            std::tie(g, m, n, k) = get_shape_by_spec<"gmnk">("mk"_sp = a, "nk"_sp = b, "gmn"_sp = d, "g"_sp = grouped_layout.value());
            DJ_HOST_ASSERT(major_a == Major::MN);
            DJ_HOST_ASSERT(major_b == Major::MN);
            DJ_HOST_ASSERT(c.has_value(), "K-grouped GEMM requires accumulation into C");
            break;
    }

    check_major_type_cd(d);
    if (grouped_layout.has_value()) {
        DJ_HOST_ASSERT(grouped_layout->is_contiguous());
        DJ_HOST_ASSERT(grouped_layout->scalar_type() == torch::kInt);
    }
    // The epilogue accumulates in place into D, so C must be the same tensor: only the presence
    // of `c` reaches the kernel, never a pointer into it.
    if (c.has_value()) {
        DJ_HOST_ASSERT(c->data_ptr() == d.data_ptr(),
                       "c must be the same tensor as d; the kernel accumulates into d in place");
    }

    if (gemm_early_return(m, n, k, d, c, epilogue_class->get_output_sf())) {
        return std::nullopt;
    }

    std::optional<at::Tensor> sfa, sfb;

    const auto recipe = layout::get_recipe(recipe_in, recipe_a_in, recipe_b_in);
    if (sfa_in.has_value()) {
        if (gemm_type == GemmType::KGroupedContiguousWithPsumLayout) {
            sfa = layout::transform_k_grouped_sf_into_required_layout(sfa_in.value(), m, k, recipe, true, grouped_layout.value());
        } else {
            sfa = layout::transform_sf_into_required_layout(
                sfa_in.value(), m, k, recipe,
                gemm_type == GemmType::Batched ? std::make_optional(static_cast<int>(g)) : std::nullopt,
                true,
                disable_ue8m0_cast,
                gemm_type == GemmType::MGroupedContiguousWithPsumLayout ? grouped_layout : std::nullopt
            );
        }
    }
    if (sfb_in.has_value()) {
        if (gemm_type == GemmType::KGroupedContiguousWithPsumLayout) {
            sfb = layout::transform_k_grouped_sf_into_required_layout(sfb_in.value(), n, k, recipe, false, grouped_layout.value());
        } else {
            sfb = layout::transform_sf_into_required_layout(
                sfb_in.value(), n, k, recipe,
                (gemm_type == GemmType::Batched or gemm_type == GemmType::MGroupedContiguousWithPsumLayout)
                    ? std::make_optional(static_cast<int>(g)) : std::nullopt,
                false,
                disable_ue8m0_cast,
                std::nullopt
            );
        }
    }

    const auto cd = c.value_or(d);
    auto estimated_m = m, estimated_k = k;
    if (gemm_type == GemmType::MGroupedContiguousWithPsumLayout) {
        estimated_m = m / g;
    }
    if (gemm_type == GemmType::KGroupedContiguousWithPsumLayout) {
        estimated_k = k / g;
    }
    return GemmDesc {
        .a_ptr = a.data_ptr(),
        .b_ptr = b.data_ptr(),
        .d_ptr = d.data_ptr(),
        .a_dtype = a.scalar_type(),
        .b_dtype = b.scalar_type(),
        .d_dtype = cd.scalar_type(),
        .sfa_ptr = sfa.has_value() ? sfa->data_ptr() : nullptr,
        .sfb_ptr = sfb.has_value() ? sfb->data_ptr() : nullptr,
        .m = m,
        .n = n,
        .k = k,
        .num_cores = static_cast<uint32_t>(runtime->get_num_sms()),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a,
        .major_b = major_b,
        .acc = c.has_value(),
        .epilogue_class = std::move(epilogue_class),
        .compiled_dims = compiled_dims,
        .gemm_type = gemm_type,
        .grouped_layout = grouped_layout.has_value() ? grouped_layout->data_ptr() : nullptr,
        .num_groups = g,
        .expected_m = expected_m.value_or(estimated_m),
        .expected_k = expected_k.value_or(estimated_k),
        .stride_batch_a = gemm_type == GemmType::Batched ? static_cast<uint64_t>(a.stride(0)) : 0,
        .stride_batch_b = gemm_type == GemmType::Batched ? static_cast<uint64_t>(b.stride(0)) : 0,
        .stride_batch_d = gemm_type == GemmType::Batched ? static_cast<uint64_t>(d.stride(0)) : 0,
        .outer_stride_a = get_outer_stride(a, major_a),
        .outer_stride_b = get_outer_stride(b, major_b),
        .outer_stride_d = get_outer_stride(d, Major::K),
        .sfa = sfa,
        .sfb = sfb,
    };
}

// Generic dense low-precision GEMM: D = A @ B^T (+ C).
//
// Supports the **fp8xfp8**, **fp4xfp4** and **fp8xfp4** operand dtype combinations
// (**fp4xfp8** is no longer supported);
// The `nt` / `nn` / `tn` / `tt` binding suffix selects the **logical shape** of the two operands
//
// Args:
//     a: (A, sfA) tuple.
//         * **A**: [M, K] (nt/nn) or [K, M] (tn/tt), fp8_e4m3 or fp4_e2m1x2
//         * **sfA**: [ceil(M / gran_m), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
//         * **NOTE**: when sfA is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
//         will panic
//     b: (B, sfB) tuple.
//         * **B**: [N, K] (nt/tt) or [K, N] (tn/nn), fp8_e4m3 or fp4_e2m1x2
//         * **sfB**: [ceil(N / gran_n), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
//         * **NOTE**: when sfB is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
//         will panic
//     d: shape [M, N], bf16 or fp32.
//     c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
//     recipe: optional combined quantization granularity (gran_m, gran_n, gran_k);
//         mutually exclusive with recipe_a / recipe_b.
//     recipe_a: optional per-operand granularity (gran_mn, gran_k) for A.
//     recipe_b: optional per-operand granularity (gran_mn, gran_k) for B.
//     compiled_dims: which dimensions (e.g. "mn" or "nk") are baked in as compile-time constants for the kernel.
//     disable_ue8m0_cast: when true, forbid casting float32 scale factors into ue8m0.
//     alpha: optional multiplier applied to A @ B before adding C.
template <bool kTransA, bool kTransB>
static void fp8_fp4_gemm(
    const std::pair<torch::Tensor, torch::Tensor>& a_,
    const std::pair<torch::Tensor, torch::Tensor>& b_,
    const torch::Tensor& d,
    const std::optional<torch::Tensor>& c,
    std::optional<std::tuple<int, int, int>> recipe,
    std::optional<std::tuple<int, int>> recipe_a,
    std::optional<std::tuple<int, int>> recipe_b,
    const std::string& compiled_dims,
    bool disable_ue8m0_cast,
    const std::optional<float>& alpha,
    const std::shared_ptr<EpilogueClass>& epilogue_class
) {
    auto [a, sfa_in] = get_transposed<kTransA>(a_, 0, 1);
    auto [b, sfb_in] = get_transposed<kTransB>(b_, 0, 1);
    DJ_HOST_ASSERT(d.scalar_type() == at::kBFloat16 or d.scalar_type() == at::kFloat);
    auto maybe_desc = get_gemm_desc(
        GemmType::Normal,
        a, b, d, c, resolve_epilogue_class(epilogue_class, c, d, alpha), sfa_in, sfb_in,
        recipe, recipe_a, recipe_b,
        std::nullopt, compiled_dims,
        disable_ue8m0_cast
    );
    if (!maybe_desc.has_value()) return;
    dispatch_fp8_fp4_gemm(maybe_desc.value());
}

inline constexpr auto fp8_fp4_gemm_nt = fp8_fp4_gemm<false, false>;
inline constexpr auto fp8_fp4_gemm_tn = fp8_fp4_gemm<true, true>;
inline constexpr auto fp8_fp4_gemm_nn = fp8_fp4_gemm<false, true>;
inline constexpr auto fp8_fp4_gemm_tt = fp8_fp4_gemm<true, false>;

// M-grouped contiguous low-precision GEMM with D = A @ B^T.
//
// The rows of A are partitioned into G groups stored contiguously, while B is a
// per-group stack of shape [G, N, K]; group boundaries are described by
// `grouped_layout`. Supports **fp8xfp8**, **fp4xfp4** and **fp8xfp4** operand dtype
// combinations (**fp4xfp8** is no longer supported; the concrete kernel is dispatched
// from the operand dtypes).
//
// Args:
//     a: (A, sfA) tuple.
//         * **A**: [M, K] (nt/nn), fp8_e4m3 or fp4_e2m1x2
//         * **sfA**: [ceil(M / gran_m), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
//         * **NOTE**: when sfA is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
//         will panic
//     b: (B, sfB) tuple.
//         * **B**: [G, N, K] (nt) or [G, K, N] (nn), fp8_e4m3 or fp4_e2m1x2
//         * **sfB**: [G, ceil(N / gran_n), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
//         * **NOTE**: when sfB is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
//         will panic
//     d: shape [M, N], bf16 or fp32.
//     grouped_layout: [G], int32. Per-group row counts describing the contiguous partition of A.
//     recipe: optional combined quantization granularity (gran_m, gran_n, gran_k);
//         mutually exclusive with recipe_a / recipe_b.
//     recipe_a: optional per-operand granularity (gran_mn, gran_k) for A.
//     recipe_b: optional per-operand granularity (gran_mn, gran_k) for B.
//     compiled_dims: which dimensions are baked in as compile-time constants.
//     disable_ue8m0_cast: when true, forbid casting float32 scale factors into ue8m0.
//     use_psum_layout: must be true; selects the partial-sum output layout path.
//     ensure_zero_padding: when true, the output is zero-padded only if the input
//         is zero-padded (the kernel itself also accepts non-zero padding).
//     expected_m_for_psum_layout: optional expected per-group M used to size the layout.
template <bool kTransA, bool kTransB>
static void m_grouped_fp8_fp4_gemm_contiguous(
    const std::pair<torch::Tensor, torch::Tensor>& a_,
    const std::pair<torch::Tensor, torch::Tensor>& b_,
    const torch::Tensor& d,
    const torch::Tensor& grouped_layout,
    std::optional<std::tuple<int, int, int>> recipe,
    std::optional<std::tuple<int, int>> recipe_a,
    std::optional<std::tuple<int, int>> recipe_b,
    const std::string& compiled_dims,
    bool disable_ue8m0_cast,
    bool use_psum_layout,
    bool ensure_zero_padding,
    const std::optional<int>& expected_m_for_psum_layout,
    const std::shared_ptr<EpilogueClass>& epilogue_class
) {
    auto [a, sfa_in] = get_transposed<kTransA>(a_, 0, 1);
    auto [b, sfb_in] = get_transposed<kTransB>(b_, 1, 2);
    DJ_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn or a.scalar_type() == kPackedFP4);
    DJ_HOST_ASSERT(b.scalar_type() == torch::kFloat8_e4m3fn or b.scalar_type() == kPackedFP4);
    DJ_HOST_ASSERT(d.scalar_type() == at::kBFloat16 or d.scalar_type() == at::kFloat);
    DJ_HOST_ASSERT(use_psum_layout == true);
    auto maybe_desc = get_gemm_desc(
        GemmType::MGroupedContiguousWithPsumLayout,
        a, b, d, std::nullopt, resolve_epilogue_class(epilogue_class, std::nullopt, d), sfa_in, sfb_in,
        recipe, recipe_a, recipe_b,
        grouped_layout, compiled_dims,
        disable_ue8m0_cast,
        expected_m_for_psum_layout.value_or(0u)
    );
    if (!maybe_desc.has_value()) return;
    dispatch_fp8_fp4_gemm(maybe_desc.value());
}

inline constexpr auto m_grouped_fp8_fp4_gemm_nt_contiguous = m_grouped_fp8_fp4_gemm_contiguous<false, false>;
inline constexpr auto m_grouped_fp8_fp4_gemm_nn_contiguous = m_grouped_fp8_fp4_gemm_contiguous<false, true>;

// K-grouped contiguous low-precision GEMM producing D of shape [G, M, N].
//
// Contraction along K is split into G groups whose sizes are given by `ks_cpu`;
// operands are passed in the `tn` (transposed) layout. Supports **fp8xfp8**, **fp4xfp4**
// and **fp8xfp4** operand dtype combinations (**fp4xfp8** is no longer supported;
// the concrete kernel is dispatched from the operand dtypes).
//
// Args:
//     a: (A, sfA) tuple.
//         * **A**: [K, M] (tn), fp8_e4m3 or fp4_e2m1x2
//         * **sfA**: [ceil(M / gran_m), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
//         * **NOTE**: when sfA is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
//         will panic
//     b: (B, sfB) tuple.
//         * **B**: [K, N] (tn), fp8_e4m3 or fp4_e2m1x2
//         * **sfB**: [ceil(N / gran_n), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
//         * **NOTE**: when sfB is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel
//         will panic
//     d: shape [G, M, N], bf16 or fp32.
//     ks_cpu: optional [G], int (on CPU). Per-group K sizes.
//     grouped_layout: [G], int32. Per-group layout describing the contiguous K partition.
//     c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
//     recipe: combined quantization granularity (gran_m, gran_n, gran_k).
//     compiled_dims: which dimensions are baked in as compile-time constants.
//     use_psum_layout: must be true; selects the partial-sum output layout path.
static void k_grouped_fp8_gemm_tn_contiguous(
    const std::pair<torch::Tensor, torch::Tensor>& a_,
    const std::pair<torch::Tensor, torch::Tensor>& b_,
    const torch::Tensor& d,
    const std::optional<std::vector<int>>& ks_cpu,
    const torch::Tensor& grouped_layout,
    const std::optional<torch::Tensor>& c,
    const std::tuple<int, int, int>& recipe,
    const std::string& compiled_dims,
    bool use_psum_layout,
    const std::shared_ptr<EpilogueClass>& epilogue_class
) {
    // here we use transpose to make the input tensors at [m, k] shape
    // this is only for the purpose of getting the correct shape, the actual data layout is not changed
    // this will make get_gemm_desc much simpler, as we don't need to handle the transposed case separately
    auto [a, sfa_in] = get_transposed<true>(a_, 0, 1);
    auto [b, sfb_in] = get_transposed<true>(b_, 0, 1);
    auto [g, m, n, k] = get_shape_by_spec<"gmnk">("mk"_sp = a, "nk"_sp = b, "gmn"_sp = d, "g"_sp = grouped_layout);
    DJ_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn or a.scalar_type() == kPackedFP4);
    DJ_HOST_ASSERT(b.scalar_type() == torch::kFloat8_e4m3fn or b.scalar_type() == kPackedFP4);
    DJ_HOST_ASSERT(d.scalar_type() == at::kBFloat16 or d.scalar_type() == at::kFloat);
    DJ_HOST_ASSERT(use_psum_layout == true);
    auto expected_k = expected_k_from_ks(ks_cpu, k / g);
    auto maybe_desc = get_gemm_desc(
        GemmType::KGroupedContiguousWithPsumLayout,
        a, b, d, c, resolve_epilogue_class(epilogue_class, c, d), sfa_in, sfb_in,
        recipe, std::nullopt, std::nullopt, grouped_layout,
        compiled_dims,
        false,
        std::nullopt, expected_k
    );
    if (!maybe_desc.has_value()) return;
    dispatch_fp8_fp4_gemm(maybe_desc.value());
}

// Deprecated K-grouped low-precision GEMM in the `nt` layout.
//
// Takes A of shape [M, K] and B of shape [N, K], transposes them into [K, M] and
// [K, N] and forwards to `k_grouped_fp8_gemm_tn_contiguous`. Prefer calling the
// `tn` variant directly with already-transposed operands.
static void k_grouped_fp8_gemm_nt_contiguous(
    const std::pair<torch::Tensor, torch::Tensor>& a,
    const std::pair<torch::Tensor, torch::Tensor>& b,
    const torch::Tensor& d,
    const std::optional<std::vector<int>>& ks_cpu,
    const torch::Tensor& grouped_layout,
    const std::optional<torch::Tensor>& c,
    const std::tuple<int, int, int>& recipe,
    const std::string& compiled_dims,
    bool use_psum_layout,
    const std::shared_ptr<EpilogueClass>& epilogue_class
) {
    if (PyErr_WarnEx(PyExc_DeprecationWarning,
                     "Calling k_grouped_fp8_gemm_nt_contiguous is deprecated, "
                     "please transpose a, b into [k, m] and [k, n] "
                     "shape and call `k_grouped_fp8_gemm_tn_contiguous` instead",
                     1) == -1)
        throw py::error_already_set();
    k_grouped_fp8_gemm_tn_contiguous(get_transposed<true>(a, 0, 1), get_transposed<true>(b, 0, 1), d, ks_cpu,
                                     grouped_layout, c, recipe, compiled_dims, use_psum_layout, epilogue_class);
}

// Dense bfloat16 GEMM: D = A @ B^T (+ C).
//
// The `nt` / `nn` / `tn` / `tt` binding suffix selects the logical shape of the two operands.
//
// Args:
//     a: [M, K] (nt/nn) or [K, M] (tn/tt), bf16
//     b: [N, K] (nt/tt) or [K, N] (tn/nn), bf16
//     d: [M, N], bf16 or fp32.
//     c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
//     compiled_dims: which dimensions (e.g. "mn" or "nk") are baked in as compile-time constants for the kernel.
//     alpha: optional multiplier applied to A @ B before adding C.
template <bool kTransA, bool kTransB>
static void bf16_gemm(
    const torch::Tensor& a_,
    const torch::Tensor& b_,
    const torch::Tensor& d,
    const std::optional<torch::Tensor>& c,
    const std::string& compiled_dims,
    const std::optional<float>& alpha,
    const std::shared_ptr<EpilogueClass>& epilogue_class
) {
    auto a = get_transposed<kTransA>(a_, 0, 1);
    auto b = get_transposed<kTransB>(b_, 0, 1);
    DJ_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DJ_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DJ_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);

    auto maybe_desc = get_gemm_desc(
        GemmType::Normal,
        a, b, d, c, resolve_epilogue_class(epilogue_class, c, d, alpha), std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt,
        std::nullopt, compiled_dims
    );
    if (!maybe_desc.has_value()) return;
    deep_gemm::launch_bf16_gemm(maybe_desc.value());
}

inline constexpr auto bf16_gemm_nt = bf16_gemm<false, false>;
inline constexpr auto bf16_gemm_tn = bf16_gemm<true, true>;
inline constexpr auto bf16_gemm_nn = bf16_gemm<false, true>;
inline constexpr auto bf16_gemm_tt = bf16_gemm<true, false>;

// M-grouped contiguous bfloat16 GEMM with D = A @ B^T.
//
// The rows of A are partitioned into G groups stored contiguously, while B is a
// per-group stack of shape [G, N, K]; group boundaries are described by `grouped_layout`.
//
// Args:
//     a: [M, K] (nt/nn), bf16
//     b: [G, N, K] (nt) or [G, K, N] (nn), bf16
//     d: [M, N], bf16 or fp32.
//     grouped_layout: [G], int32. Per-group row counts describing the contiguous partition of A.
//     compiled_dims: which dimensions are baked in as compile-time constants.
//     use_psum_layout: must be true; selects the partial-sum output layout path.
//     ensure_zero_padding: when true, the output is zero-padded only if the input
//         is zero-padded (the kernel itself also accepts non-zero padding).
//     expected_m_for_psum_layout: optional expected per-group M used to size the layout.
template <bool kTransB>
static void m_grouped_bf16_gemm_contiguous(
    const torch::Tensor& a, const torch::Tensor& b_,
    const torch::Tensor& d,
    const torch::Tensor& grouped_layout,
    const std::string& compiled_dims,
    bool use_psum_layout, bool ensure_zero_padding,
    const std::optional<int>& expected_m_for_psum_layout,
    const std::shared_ptr<EpilogueClass>& epilogue_class
) {
    auto b = get_transposed<kTransB>(b_, 1, 2);
    DJ_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DJ_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DJ_HOST_ASSERT(d.scalar_type() == at::kBFloat16 or d.scalar_type() == at::kFloat);
    DJ_HOST_ASSERT(use_psum_layout == true);

    auto maybe_desc = get_gemm_desc(
        GemmType::MGroupedContiguousWithPsumLayout,
        a, b, d, std::nullopt, resolve_epilogue_class(epilogue_class, std::nullopt, d), std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt,
        grouped_layout, compiled_dims,
        true,
        expected_m_for_psum_layout
    );
    if (!maybe_desc.has_value()) return;
    deep_gemm::launch_bf16_gemm(maybe_desc.value());
}

inline constexpr auto m_grouped_bf16_gemm_nt_contiguous = m_grouped_bf16_gemm_contiguous<false>;
inline constexpr auto m_grouped_bf16_gemm_nn_contiguous = m_grouped_bf16_gemm_contiguous<true>;

// K-grouped contiguous bfloat16 GEMM producing D of shape [G, M, N].
//
// Contraction along K is split into G groups whose sizes are given by `ks_cpu` and grouped_layout;
// operands are passed in the `tn` (transposed) layout.
//
// Args:
//     a: [K, M] (tn), bf16
//     b: [K, N] (tn), bf16
//     d: [G, M, N], bf16 or fp32.
//     ks_cpu: optional [G], int (on CPU). Per-group K sizes.
//     grouped_layout: [G], int32. Per-group layout describing the contiguous K partition.
//     c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
//     compiled_dims: which dimensions are baked in as compile-time constants.
//     use_psum_layout: must be true; selects the partial-sum output layout path.
static void k_grouped_bf16_gemm_tn_contiguous(
    const torch::Tensor& a_,
    const torch::Tensor& b_,
    const torch::Tensor& d,
    const std::optional<std::vector<int>>& ks_cpu,
    const torch::Tensor& grouped_layout,
    const std::optional<torch::Tensor>& c,
    const std::string& compiled_dims,
    bool use_psum_layout,
    const std::shared_ptr<EpilogueClass>& epilogue_class
) {
    // here we use transpose to make the input tensors at [m, k] shape
    // this is only for the purpose of getting the correct shape, the actual data layout is not changed
    // this will make get_gemm_desc much simpler, as we don't need to handle the transposed case separately
    auto a = get_transposed<true>(a_, 0, 1);
    auto b = get_transposed<true>(b_, 0, 1);
    auto [g, m, n, k] = get_shape_by_spec<"gmnk">("mk"_sp = a, "nk"_sp = b, "gmn"_sp = d, "g"_sp = grouped_layout);
    DJ_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DJ_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DJ_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);
    DJ_HOST_ASSERT(use_psum_layout == true);

    auto expected_k = expected_k_from_ks(ks_cpu, k / g);
    auto maybe_desc = get_gemm_desc(
        GemmType::KGroupedContiguousWithPsumLayout,
        a, b, d, c, resolve_epilogue_class(epilogue_class, c, d),
        std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, std::nullopt,
        grouped_layout, compiled_dims,
        false,
        std::nullopt,
        expected_k
    );
    if (!maybe_desc.has_value()) return;
    deep_gemm::launch_bf16_gemm(maybe_desc.value());
}

static void register_apis(pybind11::module_& m) {

    m.def("fp8_fp4_gemm_nt", fp8_fp4_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("disable_ue8m0_cast") = false,
          py::arg("alpha") = std::nullopt,
          py::arg("epilogue") = nullptr);
    m.def("fp8_fp4_gemm_nn", fp8_fp4_gemm_nn,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("disable_ue8m0_cast") = false,
          py::arg("alpha") = std::nullopt,
          py::arg("epilogue") = nullptr);
    m.def("fp8_fp4_gemm_tn", fp8_fp4_gemm_tn,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "mn",
          py::arg("disable_ue8m0_cast") = false,
          py::arg("alpha") = std::nullopt,
          py::arg("epilogue") = nullptr);
    m.def("fp8_fp4_gemm_tt", fp8_fp4_gemm_tt,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "mn",
          py::arg("disable_ue8m0_cast") = false,
          py::arg("alpha") = std::nullopt,
          py::arg("epilogue") = nullptr);
    m.def("m_grouped_fp8_fp4_gemm_nt_contiguous", m_grouped_fp8_fp4_gemm_nt_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("grouped_layout"),
          py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("disable_ue8m0_cast") = false,
          py::arg("use_psum_layout") = true,
          py::arg("ensure_zero_padding") = true,
          py::arg("expected_m_for_psum_layout") = std::nullopt,
          py::arg("epilogue") = nullptr);
    m.def("m_grouped_fp8_fp4_gemm_nn_contiguous", m_grouped_fp8_fp4_gemm_nn_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("grouped_layout"),
          py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("disable_ue8m0_cast") = false,
          py::arg("use_psum_layout") = true,
          py::arg("ensure_zero_padding") = true,
          py::arg("expected_m_for_psum_layout") = std::nullopt,
          py::arg("epilogue") = nullptr);
    // m_grouped_fp8_fp4_gemm_nt_masked is not implemented in this backend.
    m.def("k_grouped_fp8_gemm_tn_contiguous", k_grouped_fp8_gemm_tn_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("ks_cpu"), py::arg("grouped_layout"),
          py::arg("c") = std::nullopt,
          py::arg("recipe") = std::make_tuple(1, 1, 32),
          py::arg("compiled_dims") = "mn",
          py::arg("use_psum_layout") = true,
          py::arg("epilogue") = nullptr);
    m.def("k_grouped_fp8_gemm_nt_contiguous", k_grouped_fp8_gemm_nt_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("ks_cpu"), py::arg("grouped_layout"),
          py::arg("c") = std::nullopt,
          py::arg("recipe") = std::make_tuple(1, 1, 32),
          py::arg("compiled_dims") = "mn",
          py::arg("use_psum_layout") = true,
          py::arg("epilogue") = nullptr);

    m.attr("fp8_gemm_nt") = m.attr("fp8_fp4_gemm_nt");
    m.attr("fp8_gemm_nn") = m.attr("fp8_fp4_gemm_nn");
    m.attr("fp8_gemm_tn") = m.attr("fp8_fp4_gemm_tn");
    m.attr("fp8_gemm_tt") = m.attr("fp8_fp4_gemm_tt");
    m.attr("m_grouped_fp8_gemm_nt_contiguous") = m.attr("m_grouped_fp8_fp4_gemm_nt_contiguous");
    m.attr("m_grouped_fp8_gemm_nn_contiguous") = m.attr("m_grouped_fp8_fp4_gemm_nn_contiguous");
    // m_grouped_fp8_gemm_nt_masked is not implemented in this backend.

    m.def("bf16_gemm_nt", bf16_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("alpha") = std::nullopt,
          py::arg("epilogue") = nullptr);
    m.def("bf16_gemm_nn", bf16_gemm_nn,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("alpha") = std::nullopt,
          py::arg("epilogue") = nullptr);
    m.def("bf16_gemm_tn", bf16_gemm_tn,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt,
          py::arg("compiled_dims") = "mn",
          py::arg("alpha") = std::nullopt,
          py::arg("epilogue") = nullptr);
    m.def("bf16_gemm_tt", bf16_gemm_tt,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt,
          py::arg("compiled_dims") = "mn",
          py::arg("alpha") = std::nullopt,
          py::arg("epilogue") = nullptr);
    m.def("m_grouped_bf16_gemm_nt_contiguous", m_grouped_bf16_gemm_nt_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("grouped_layout"),
          py::arg("compiled_dims") = "nk",
          py::arg("use_psum_layout") = true,
          py::arg("ensure_zero_padding") = true,
          py::arg("expected_m_for_psum_layout") = std::nullopt,
          py::arg("epilogue") = nullptr);
    m.def("m_grouped_bf16_gemm_nn_contiguous", m_grouped_bf16_gemm_nn_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("grouped_layout"),
          py::arg("compiled_dims") = "nk",
          py::arg("use_psum_layout") = true,
          py::arg("ensure_zero_padding") = true,
          py::arg("expected_m_for_psum_layout") = std::nullopt,
          py::arg("epilogue") = nullptr);
    m.def("k_grouped_bf16_gemm_tn_contiguous", k_grouped_bf16_gemm_tn_contiguous,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("ks_cpu"), py::arg("grouped_layout"),
          py::arg("c") = std::nullopt,
          py::arg("compiled_dims") = "mn",
          py::arg("use_psum_layout") = true,
          py::arg("epilogue") = nullptr);

}

} // namespace deep_gemm::gemm_api
