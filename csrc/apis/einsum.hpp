#pragma once

#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include <torch/python.h>

#include <deep_jit/utils/exception.hpp>

#include "gemm.hpp"

namespace deep_gemm::einsum {

using gemm_api::get_gemm_desc;

namespace py = pybind11;

// Batched FP8 GEMM: D[i] = A[i] @ B[i]^T (+ C[i]) for each batch i.
//
// Args:
//     a: (A, sfA) tuple.
//         * **A**: [B, M, K], fp8_e4m3
//         * **sfA**: [B, ceil(M / gran_m), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
//         * **NOTE**: when sfA is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0
//     b: (B, sfB) tuple.
//         * **B**: [B, N, K], fp8_e4m3
//         * **sfB**: [B, ceil(N / gran_n), ceil(K / gran_k)], fp32 or short (packed ue8m0), in any major
//         * **NOTE**: when sfB is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0
//     d: shape [B, M, N], bf16 or fp32, or FP8 with FP8Quantization(sfd), FP8StochasticRounding(sfd), or a (D, SFD) pair.
//     c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
//     recipe: optional combined quantization granularity (gran_m, gran_n, gran_k).
//     compiled_dims: which dimensions (e.g. "mn" or "nk") are baked in as compile-time constants for the kernel.
static void fp8_bmm(
    const std::pair<torch::Tensor, torch::Tensor>& a_,
    const std::pair<torch::Tensor, torch::Tensor>& b_,
    const std::variant<torch::Tensor, std::pair<torch::Tensor, torch::Tensor>>& d,
    const std::optional<torch::Tensor>& c,
    const std::optional<std::tuple<int, int, int>>& recipe,
    const std::string& compiled_dims,
    const std::shared_ptr<EpilogueClass>& epilogue_class
) {
    auto [a, sfa_in] = a_;
    auto [b, sfb_in] = b_;
    const auto d_fp8 = std::get_if<std::pair<torch::Tensor, torch::Tensor>>(&d);
    const auto d_tensor = d_fp8 != nullptr ? d_fp8->first : std::get<torch::Tensor>(d);
    const auto sfd = d_fp8 != nullptr ? std::make_optional(d_fp8->second) : std::nullopt;

    DJ_HOST_ASSERT(a.scalar_type() == at::kFloat8_e4m3fn);
    DJ_HOST_ASSERT(b.scalar_type() == at::kFloat8_e4m3fn);

    const auto [batch_size, m, n, k] = get_shape_by_spec<"bmnk">("bmk"_sp = a, "bnk"_sp = b, "bmn"_sp = d_tensor);
    DJ_HOST_ASSERT(a.stride(-1) == 1 or a.stride(-2) == 1);
    DJ_HOST_ASSERT(b.stride(-1) == 1 or b.stride(-2) == 1);
    DJ_HOST_ASSERT(d_tensor.stride(-1) == 1);

    if (batch_size == 0) return;
    const auto resolved = resolve_epilogue_class(epilogue_class, c, d_tensor, std::nullopt, sfd);
    if (m == 0 or n == 0 or k == 0) {
        gemm_early_return(m, n, k, d_tensor, c, resolved->get_output_sf());
        return;
    }

    auto maybe_desc = get_gemm_desc(GemmType::Batched, a, b, d_tensor, c, resolved, sfa_in, sfb_in,
                                    recipe, std::nullopt, std::nullopt, std::nullopt, compiled_dims);
    if (!maybe_desc.has_value()) return;
    deep_gemm::launch_fp8_gemm(maybe_desc.value());
}

// Batched bfloat16 GEMM: D[i] = A[i] @ B[i]^T (+ C[i]) for each batch i.
//
// Args:
//     a: A, [B, M, K], bf16
//     b: B, [B, N, K], bf16
//     d: [B, M, N], bf16 or fp32.
//     c: if present, the kernel computes C += A @ B, otherwise computes D = A @ B
//     compiled_dims: which dimensions (e.g. "mn" or "nk") are baked in as compile-time constants for the kernel.
static void bf16_bmm(
    const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
    const std::optional<torch::Tensor>& c = std::nullopt,
    const std::string& compiled_dims = "nk",
    const std::shared_ptr<EpilogueClass>& epilogue_class = nullptr
) {

    DJ_HOST_ASSERT(a.scalar_type() == at::kBFloat16);
    DJ_HOST_ASSERT(b.scalar_type() == at::kBFloat16);
    DJ_HOST_ASSERT(d.scalar_type() == at::kFloat or d.scalar_type() == at::kBFloat16);

    auto maybe_desc = get_gemm_desc(GemmType::Batched, a, b, d, c, resolve_epilogue_class(epilogue_class, c, d),
                                    std::nullopt, std::nullopt,
                                    std::nullopt, std::nullopt, std::nullopt, std::nullopt, compiled_dims);
    if (!maybe_desc.has_value()) return;
    deep_gemm::launch_bf16_gemm(maybe_desc.value());
}

// Batched bfloat16 einsum over a fixed set of contraction expressions.
//
// Permutes the operands and dispatches to a batched bf16 GEMM. Dimensions:
// b = batch, h = head, d / r = feature dims.
//
// Args:
//     expr: the einsum contraction string. Supported values (any other raises):
//         "bhr,hdr->bhd":  a [b, h, r], b [h, d, r] -> d [b, h, d]
//         "bhd,hdr->bhr":  a [b, h, d], b [h, d, r] -> d [b, h, r]
//         "bhd,bhr->hdr":  a [b, h, d], b [b, h, r] -> d [h, d, r]
//     a: bf16. Shape matches the first term of `expr`.
//     b: bf16. Shape matches the second term of `expr`.
//     d: output tensor, bf16 or fp32. Shape matches the result term of `expr`.
//     c: if present, the kernel computes C += einsum(expr, a, b), otherwise computes D = einsum(expr, a, b)
//     use_cublaslt: if true, route to the Ascend einsum kernel; not implemented yet, so it raises.
static void einsum(
    const std::string& expr,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& d,
    const std::optional<torch::Tensor>& c,
    const bool& use_cublaslt,
    const std::shared_ptr<EpilogueClass>& epilogue_class
) {
    if (use_cublaslt) {
        DJ_PANIC("einsum: use_cublaslt will call ascend einsum kernel, but is not implemented in current version");
    }
    auto perm_c = [&](std::initializer_list<int64_t> p) {
        return c.has_value() ? std::make_optional(c.value().permute(p)) : std::nullopt;
    };
    if (expr == "bhr,hdr->bhd") {
        bf16_bmm(a.permute({1, 0, 2}), b, d.permute({1, 0, 2}), perm_c({1, 0, 2}), "nk", epilogue_class);
    } else if (expr == "bhd,hdr->bhr") {
        bf16_bmm(a.permute({1, 0, 2}), b.permute({0, 2, 1}), d.permute({1, 0, 2}), perm_c({1, 0, 2}), "nk", epilogue_class);
    } else if (expr == "bhd,bhr->hdr") {
        bf16_bmm(a.permute({1, 2, 0}), b.permute({1, 2, 0}), d, c, "mn", epilogue_class);
    } else {
        DJ_HOST_ASSERT(false, "Unsupported einsum expression");
    }
}

// FP8 einsum over a fixed set of contraction expressions.
//
// Permutes the operands (and their scale factors) and dispatches to a batched fp8
// GEMM. Dimensions: b = batch, h = head, d / r = feature dims.
//
// Args:
//     expr: the einsum contraction string. Supported values (any other raises):
//         "bhr,hdr->bhd":  a [b, h, r], b [h, d, r] -> d [b, h, d]
//         "bhd,hdr->bhr":  a [b, h, d], b [h, d, r] -> d [b, h, r]
//         "bhd,bhr->hdr":  a [b, h, d], b [b, h, r] -> d [h, d, r]
//     a: (A, sfA) tuple.
//         * **A**: fp8_e4m3, shape matches the first term of `expr`.
//         * **sfA**: fp32 or short (packed ue8m0), in any major
//         * **NOTE**: when sfA is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel will panic
//     b: (B, sfB) tuple.
//         * **B**: fp8_e4m3, shape matches the second term of `expr`.
//         * **sfB**: fp32 or short (packed ue8m0), in any major
//         * **NOTE**: when sfB is fp32, only 8-bit exponential is used, 23-bit mantissa must be 0, otherwise the kernel will panic
//     d: bf16/fp32, or FP8 with FP8Quantization(sfd) or FP8StochasticRounding(sfd); forward also accepts a (D, SFD) pair.
//     c: if present, the kernel computes C += einsum(expr, a, b), otherwise computes D = einsum(expr, a, b)
//     recipe: combined quantization granularity (gran_m, gran_n, gran_k).
static void fp8_einsum(
    const std::string& expr,
    const std::pair<torch::Tensor, torch::Tensor>& a,
    const std::pair<torch::Tensor, torch::Tensor>& b,
    const std::variant<torch::Tensor, std::pair<torch::Tensor, torch::Tensor>>& d,
    const std::optional<torch::Tensor>& c,
    const std::tuple<int, int, int>& recipe,
    const std::shared_ptr<EpilogueClass>& epilogue_class
) {
    auto perm_c = [&](std::initializer_list<int64_t> p) {
        return c.has_value() ? std::make_optional(c.value().permute(p)) : std::nullopt;
    };
    if (expr == "bhr,hdr->bhd") {
        // A=[b,h,r] B=[h,d,r] D=[b,h,d] -> (B,M,N,K) = (h,b,d,r)
        auto perm_d = d;
        if (auto d_fp8 = std::get_if<std::pair<torch::Tensor, torch::Tensor>>(&perm_d))
            d_fp8->first = d_fp8->first.permute({1, 0, 2});
        else
            std::get<torch::Tensor>(perm_d) = std::get<torch::Tensor>(perm_d).permute({1, 0, 2});
        fp8_bmm({a.first.permute({1, 0, 2}), a.second.permute({1, 0, 2})},
                {b.first, b.second},
                perm_d, perm_c({1, 0, 2}), std::make_optional(recipe), "nk", epilogue_class);
    } else if (expr == "bhd,hdr->bhr") {
        // A=[b,h,d] B=[h,d,r] D=[b,h,r] -> (h,b,r,d)
        DJ_HOST_ASSERT(std::holds_alternative<torch::Tensor>(d));
        fp8_bmm({a.first.permute({1, 0, 2}), a.second.permute({1, 0, 2})},
                {b.first.permute({0, 2, 1}), b.second.permute({0, 2, 1})},
                std::get<torch::Tensor>(d).permute({1, 0, 2}), perm_c({1, 0, 2}),
                std::make_optional(recipe), "nk", epilogue_class);
    } else if (expr == "bhd,bhr->hdr") {
        // A=[b,h,d] B=[b,h,r] D=[h,d,r] -> (h,d,r,b) (batch=h, K=b reduce)
        DJ_HOST_ASSERT(std::holds_alternative<torch::Tensor>(d));
        fp8_bmm({a.first.permute({1, 2, 0}), a.second.permute({1, 2, 0})},
                {b.first.permute({1, 2, 0}), b.second.permute({1, 2, 0})},
                std::get<torch::Tensor>(d), c, std::make_optional(recipe), "mn", epilogue_class);
    } else {
        DJ_HOST_ASSERT(false, "Unsupported einsum expression");
    }
}

static void register_apis(pybind11::module_& m) {

    m.def("einsum", &einsum,
          py::arg("expr"), py::arg("a"), py::arg("b"),
          py::arg("d"), py::arg("c") = std::nullopt,
          py::arg("use_cublaslt") = false,
          py::arg("epilogue") = nullptr);
    m.def("fp8_bmm", &fp8_bmm,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("c") = std::nullopt,
          py::arg("recipe") = std::make_tuple(1, 1, 32), py::arg("compiled_dims") = "nk",
          py::arg("epilogue") = nullptr);
    m.def("bf16_bmm", &bf16_bmm,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("c") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("epilogue") = nullptr);
    m.def("fp8_einsum", &fp8_einsum,
          py::arg("expr"), py::arg("a"), py::arg("b"),
          py::arg("d"),  py::arg("c") = std::nullopt,
          py::arg("recipe") = std::make_tuple(1, 1, 32),
          py::arg("epilogue") = nullptr);
}

} // namespace deep_gemm::einsum
