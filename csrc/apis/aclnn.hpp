#pragma once

#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_quant_matmul_v5.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_addmm.h>

#include <deep_jit/backend/ascend/driver.hpp>
#include <deep_jit/utils/exception.hpp>

#include "../utils/layout.hpp"
#include "../utils/aclnn.hpp"
#include "../jit/jit.hpp"
#include "layout.hpp"

namespace deep_gemm::aclnn_api {

namespace py = pybind11;

// Wrap an operand into an aclTensor.  FP8 goes through ConvertType, which already honours
// the tensor's real sizes/strides.  Packed FP4 is stored as int8 with 2 nibbles per byte
// along its contiguous (stride-1) axis; aclnn must see the logical FLOAT4_E2M1 [rows, cols]
// view, so derive the logical shape/strides from the actual layout (works for both K-major
// and MN-major operands, no transpose/repack needed).
static aclTensor* make_quant_operand(const torch::Tensor& t) {
    if (t.scalar_type() != kPackedFP4)
        return ConvertType(t);
    DJ_HOST_ASSERT(t.dim() == 2, "packed FP4 operand must be 2-D, got dim={}", t.dim());
    auto [d0, d1] = get_shape<2>(t);
    std::array<int64_t, 2> view_dims{}, strides{}, storage_dims{};
    if (get_major_type_ab(t) == Major::K) {
        // int8 [d0, K/2] packed along K -> logical [d0, K]
        view_dims = {d0, d1 * 2};
        strides = {t.stride(0) * 2, 1};
        storage_dims = {d0, d1 * 2};
    } else {
        // int8 [MN/2, d1] packed along M/N -> logical [MN, d1]
        view_dims = {d0 * 2, d1};
        strides = {1, t.stride(1) * 2};
        storage_dims = {d1, d0 * 2};
    }
    return aclCreateTensor(view_dims.data(), 2, ACL_FLOAT4_E2M1, strides.data(), 0,
                           ACL_FORMAT_ND, storage_dims.data(), 2, t.data_ptr());
}

// Build the ue8m0 MX scale that aclnnQuantMatmulV5 expects. aclnn aligns the scale's
// leading dims with the operand tensor it receives, so when the operand is handed over
// in [K, rows] orientation (transposed view) the scale must be [K/64, rows, 2] too;
// otherwise it is the row-major [rows, K/64, 2]. `k_major` selects the transposed form.
// The scale is ~1/32 of the operand, so this small copy avoids an O(rows*K) operand copy.
static torch::Tensor convert_sf_for_aclnn(const torch::Tensor& sf, int64_t mn, int64_t k, bool k_major) {
    DJ_HOST_ASSERT(sf.dim() == 2);
    DJ_HOST_ASSERT(sf.size(0) == mn);
    const int64_t sf_k_elems = ceil_div(k, static_cast<int64_t>(32));
    const int64_t sf_k_pairs = ceil_div(sf_k_elems, static_cast<int64_t>(2));
    const bool packed = sf.scalar_type() == at::kShort;
    DJ_HOST_ASSERT(sf.size(1) == (packed ? sf_k_pairs : sf_k_elems));

    torch::Tensor sf_bytes;
    if (packed) {
        // Each int16 stores two adjacent K scales. Native GEMM commonly uses
        // MN-major scales, so materialize the logical rows before unpacking.
        // Flatten also normalizes singleton K-pair strides before the dtype view.
        sf_bytes = sf.contiguous().flatten().view(at::kByte)
                       .reshape({mn, sf_k_pairs * 2}).slice(1, 0, sf_k_elems);
    } else if (sf.scalar_type() == at::kFloat) {
        sf_bytes = sf.view(at::kInt).floor_divide(1 << 23).to(at::kByte);
    } else if (sf.scalar_type() == at::kFloat8_e8m0fnu or sf.scalar_type() == at::kByte) {
        sf_bytes = sf.view(at::kByte);
    } else {
        DJ_PANIC("unsupported ACLNN scale factor dtype: {}", sf.scalar_type());
    }

    if (sf_k_elems % 2 != 0) {
        // pad one zero byte on the right of K so the pair dim is full.
        sf_bytes = at::pad(sf_bytes, {0, 1}, "constant", 0);
    } else if (sf_bytes.stride(1) != 1) {
        sf_bytes = sf_bytes.contiguous();
    }
    // [mn, K/32] bytes -> [mn, K/64, 2] (the inner pair runs along K).  Keep the permute
    // and the materializing copy on uint8 (AICPU Transpose has no fp8_e8m0 kernel); only
    // reinterpret as ue8m0 at the very end.
    auto out = sf_bytes.reshape(at::IntArrayRef{mn, sf_k_pairs, static_cast<int64_t>(2)});
    if (k_major)
        out = out.permute({1, 0, 2});  // [K/64, mn, 2]
    return out.contiguous().view(at::kFloat8_e8m0fnu);
}

// ACLNN-backed dense low-precision GEMM: D = A @ B^T (+ C).
//
// Computes the product through aclnnQuantMatmulV5 using MX (ue8m0) scale factors;
// supports recipe [m, n, k] = [1, 1, 32] only, gran_mn != 1 will be rejected by aclnn.
// supports the **fp8xfp8**, **fp4xfp4** and **fp8xfp4** operand dtype combinations
// (**fp4xfp8** is not supported, see the table below).
// Operands are passed in their natural orientation (aclnn honours their real
// strides), while each scale factor follows its operand's physical major. The
// `nt` / `nn` / `tn` / `tt` binding suffix selects the **logical shape** of the two
// operands (see the per-argument shapes below). Requires K to be a multiple of 64.
//
// | A dtype | A major | B dtype | B major | ok?     | reason                                            |
// |---------|---------|---------|---------|---------|---------------------------------------------------|
// | fp8     | *       | fp8     | *       | **yes** | W8A8: any major, bf16 or fp32 output              |
// | fp4     | *       | fp4     | *       | **yes** | W4A4: any major, bf16 or fp32 output              |
// | fp8     | K       | fp4     | K       | **yes** | A8W4 (NT); bf16/fp16 output only (fp32 rejected)  |
// | fp8     | K       | fp4     | MN      | **no**  | A8W4 ND only support transposeX2 is true          |
// | fp8     | MN      | fp4     | *       | **no**  | Only support transposeX1 is false                 |
// | fp4     | *       | fp8     | *       | **no**  | Invalid x1 or x2 dtype (fp4 must be x2, not x1)    |
//
// Args:
//     a: (A, sfA) tuple.
//         * **A**: [M, K] (nt/nn) or [K, M] (tn/tt), fp8_e4m3 or fp4_e2m1x2
//         * **sfA**: [ceil(M / gran_m), ceil(K / gran_k)] fp32, or [ceil(M / gran_m), ceil(K / (2 * gran_k))] short (packed ue8m0), in any major
//         * **NOTE**: when sfA is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel will panic
//     b: (B, sfB) tuple.
//         * **B**: [N, K] (nt/tt) or [K, N] (tn/nn), fp8_e4m3 or fp4_e2m1x2
//         * **sfB**: [ceil(N / gran_n), ceil(K / gran_k)] fp32, or [ceil(N / gran_n), ceil(K / (2 * gran_k))] short (packed ue8m0), in any major
//         * **NOTE**: when sfB is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel will panic
//     d: shape [M, N], bf16 or fp32.
//     c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
//     recipe: optional combined quantization granularity (gran_m, gran_n, gran_k);
//         mutually exclusive with recipe_a / recipe_b.
//     recipe_a: optional per-operand granularity (gran_mn, gran_k) for A.
//     recipe_b: optional per-operand granularity (gran_mn, gran_k) for B.
//     disable_ue8m0_cast: when true, forbid casting float32 scale factors into ue8m0.
template <bool kTransA, bool kTransB>
static void aclnn_fp8_fp4_gemm(
    const std::pair<torch::Tensor, torch::Tensor>& a_,
    const std::pair<torch::Tensor, torch::Tensor>& b_,
    const torch::Tensor& d,
    const std::optional<torch::Tensor>& c,
    std::optional<std::tuple<int, int, int>> recipe_in,
    std::optional<std::tuple<int, int>> recipe_a_in,
    std::optional<std::tuple<int, int>> recipe_b_in,
    // TODO: remove this for the whole project
    bool disable_ue8m0_cast
) {
    // normalize to [N, K] shape
    auto [a, sfa_in] = get_transposed<kTransA>(a_, 0, 1);
    auto [b, sfb_in] = get_transposed<kTransB>(b_, 0, 1);
    auto [m, n, k] = get_shape_by_spec<"mnk">("mk"_sp = a, "nk"_sp = b, "mn"_sp = d);
    if (gemm_early_return(m, n, k, d, c)) {
        return;
    }
    DJ_HOST_ASSERT(k % 64 == 0, "ACLNN requires K to be a multiple of 64, got K={}", k);

    // Pass operands in their natural orientation; aclnn honours their real strides, so the
    // logical product op(x1)=[m,k], op(x2)=[k,n] is obtained with transposeX1/X2 = false/true
    // for any physical major (no manual transpose / no contiguous copy).  The MX scale must
    // instead follow each operand's *physical* major: aclnn aligns the scale's leading dim
    // with the operand storage, so an MN-major operand needs a [K/64, rows, 2] scale.
    auto recipe = layout::get_recipe(recipe_in, recipe_a_in, recipe_b_in);
    // NOTE: aclnnQuantMatmulV5's MX (ue8m0) path only supports per-row 1x32 scaling
    // (group_sizes [1, 1, 32]); block scaling along M/N (gran_mn != 1) is rejected by the
    // op with "Unsupported groupSize". Callers must filter those recipes out beforehand.
    auto sfa = convert_sf_for_aclnn(sfa_in, m, k, /*k_major=*/get_major_type_ab(a) == Major::MN);
    auto sfb = convert_sf_for_aclnn(sfb_in, n, k, /*k_major=*/get_major_type_ab(b) == Major::MN);

    auto mm_out = c.has_value() ? at::empty_like(d) : d;

    AclTensor t_a(make_quant_operand(a)), t_b(make_quant_operand(b));
    AclTensor t_sfa(sfa), t_sfb(sfb), t_cd(mm_out);
    uint64_t group_size = (
        static_cast<uint64_t>(get<2>(recipe))
        | (static_cast<uint64_t>(get<1>(recipe)) << 16)
        | (static_cast<uint64_t>(get<0>(recipe)) << 32)
    );
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;

    DJ_ACL_CHECK(aclnnQuantMatmulV5GetWorkspaceSize(
        t_a, t_b,
        t_sfa, t_sfb, nullptr,
        nullptr, nullptr, nullptr,
        nullptr,
        /*transposeX1=*/false, /*transposeX2=*/true, group_size, t_cd,
        &ws_size, &executor
    ));

    void* ws_ptr = nullptr;
    torch::Tensor ws_tensor;
    if (ws_size > 0) {
        ws_tensor = torch::empty(
            {static_cast<int64_t>(ws_size)},
            torch::TensorOptions().dtype(torch::kUInt8).device(a.device())
        );
        ws_ptr = ws_tensor.data_ptr();
    }

    DJ_ACL_CHECK(aclnnQuantMatmulV5(ws_ptr, ws_size, executor, get_current_npu_stream()));
    if (c.has_value()) {
        d.copy_(*c);
        d.add_(mm_out);
    }
}

inline constexpr auto aclnn_fp8_fp4_gemm_nt = aclnn_fp8_fp4_gemm<false, false>;
inline constexpr auto aclnn_fp8_fp4_gemm_tn = aclnn_fp8_fp4_gemm<true, true>;
inline constexpr auto aclnn_fp8_fp4_gemm_nn = aclnn_fp8_fp4_gemm<false, true>;
inline constexpr auto aclnn_fp8_fp4_gemm_tt = aclnn_fp8_fp4_gemm<true, false>;

// ACLNN-backed dense bfloat16 GEMM: D = A @ B^T (+ C).
//
// Uses aclnnMatmul when `c` is absent, or aclnnInplaceAddmm to compute
// D = C + A @ B^T when `c` is provided. The `nt` / `nn` / `tn` / `tt` binding
// suffix selects the logical shape of the two operands (see the shapes below).
//
// Args:
//     a: [M, K] (nt/nn) or [K, M] (tn/tt), bf16
//     b: [N, K] (nt/tt) or [K, N] (tn/nn), bf16
//     d: [M, N], bf16.
//     c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
template <bool kTransA, bool kTransB>
static void aclnn_bf16_gemm(
    const torch::Tensor& a_,
    const torch::Tensor& b_,
    const torch::Tensor& d,
    const std::optional<torch::Tensor>& c
) {
    auto a = get_transposed<kTransA>(a_, 0, 1);
    auto b = get_transposed<!kTransB>(b_, 0, 1);
    auto cd = c.value_or(d);
    auto [m, n, k] = get_shape_by_spec<"mnk">("mk"_sp = a, "kn"_sp = b, "mn"_sp = d);
    if (gemm_early_return(m, n, k, d, c)) {
        return;
    }

    AclTensor t_a(a), t_b(b), t_cd(cd);
    AclScalar<float> one(1.f);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    if (c.has_value()) {
        DJ_ACL_CHECK(aclnnInplaceAddmmGetWorkspaceSize(
            t_cd, t_a, t_b,
            one, one,
            0 /* KEEP_DTYPE */,
            &ws_size, &executor
        ));
    } else {
        DJ_ACL_CHECK(aclnnMatmulGetWorkspaceSize(
            t_a, t_b, t_cd,
            0 /* KEEP_DTYPE */,
            &ws_size, &executor
        ));
    }

    void* ws_ptr = nullptr;
    torch::Tensor ws_tensor;
    if (ws_size > 0) {
        ws_tensor = torch::empty({static_cast<int64_t>(ws_size)}, torch::TensorOptions().dtype(torch::kUInt8).device(a.device()));
        ws_ptr = ws_tensor.data_ptr();
    }

    if (c.has_value()) {
        DJ_ACL_CHECK(aclnnInplaceAddmm(ws_ptr, ws_size, executor, get_current_npu_stream()));
    } else {
        DJ_ACL_CHECK(aclnnMatmul(ws_ptr, ws_size, executor, get_current_npu_stream()));
    }
}

inline constexpr auto aclnn_bf16_gemm_nt = aclnn_bf16_gemm<false, false>;
inline constexpr auto aclnn_bf16_gemm_nn = aclnn_bf16_gemm<false, true>;
inline constexpr auto aclnn_bf16_gemm_tn = aclnn_bf16_gemm<true, true>;
inline constexpr auto aclnn_bf16_gemm_tt = aclnn_bf16_gemm<true, false>;

static void register_apis(pybind11::module_& m) {

    m.def("aclnn_fp8_fp4_gemm_nt", aclnn_fp8_fp4_gemm_nt,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("disable_ue8m0_cast") = false);
    m.def("aclnn_fp8_fp4_gemm_nn", aclnn_fp8_fp4_gemm_nn,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("disable_ue8m0_cast") = false);
    m.def("aclnn_fp8_fp4_gemm_tn", aclnn_fp8_fp4_gemm_tn,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("disable_ue8m0_cast") = false);
    m.def("aclnn_fp8_fp4_gemm_tt", aclnn_fp8_fp4_gemm_tt,
          py::arg("a"), py::arg("b"), py::arg("d"),
          py::arg("c") = std::nullopt, py::arg("recipe") = std::nullopt,
          py::arg("recipe_a") = std::nullopt, py::arg("recipe_b") = std::nullopt,
          py::arg("disable_ue8m0_cast") = false);

    m.def("aclnn_bf16_gemm_nt", aclnn_bf16_gemm_nt, py::arg("a"), py::arg("b"), py::arg("d"), py::arg("c") = std::nullopt);
    m.def("aclnn_bf16_gemm_nn", aclnn_bf16_gemm_nn, py::arg("a"), py::arg("b"), py::arg("d"), py::arg("c") = std::nullopt);
    m.def("aclnn_bf16_gemm_tn", aclnn_bf16_gemm_tn, py::arg("a"), py::arg("b"), py::arg("d"), py::arg("c") = std::nullopt);
    m.def("aclnn_bf16_gemm_tt", aclnn_bf16_gemm_tt, py::arg("a"), py::arg("b"), py::arg("d"), py::arg("c") = std::nullopt);

    // Upstream cuBLASLt API aliases backed by ACLNN.
    m.attr("cublaslt_gemm_nt") = m.attr("aclnn_bf16_gemm_nt");
    m.attr("cublaslt_gemm_nn") = m.attr("aclnn_bf16_gemm_nn");
    m.attr("cublaslt_gemm_tn") = m.attr("aclnn_bf16_gemm_tn");
    m.attr("cublaslt_gemm_tt") = m.attr("aclnn_bf16_gemm_tt");
}

} // namespace deep_gemm::aclnn_api
