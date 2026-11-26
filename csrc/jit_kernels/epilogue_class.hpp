#pragma once

#include <memory>
#include <optional>
#include <string>
#include <torch/torch.h>

#include <deep_gemm/common.hpp>
#include <deep_jit/utils/exception.hpp>

namespace deep_gemm {

// Select the device operator, carry its runtime state, and validate the output.
class EpilogueClass {
public:
    virtual ~EpilogueClass() = default;

    virtual std::string get_epilogue_operator_type() const = 0;

    virtual epilogue::Args make_epilogue_operator_args(const uint32_t& n) const { return {}; }

    virtual std::optional<torch::Tensor> get_output_sf() const { return std::nullopt; }

    virtual void check(const std::optional<torch::Tensor>& c, const torch::Tensor& d) const {
        DJ_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);
    }
};

class IdentityEpilogue final : public EpilogueClass {
public:
    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::Identity";
    }
};

class AlphaEpilogue final : public EpilogueClass {
    float alpha;

public:
    explicit AlphaEpilogue(const float& alpha) : alpha(alpha) {}

    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::ScaleByAlpha";
    }

    epilogue::Args make_epilogue_operator_args(const uint32_t& n) const override {
        return {.alpha = alpha};
    }
};

class BiasEpilogue final : public EpilogueClass {
    epilogue::Args args;

public:
    explicit BiasEpilogue(const torch::Tensor& bias) {
        args.bias = reinterpret_cast<uintptr_t>(bias.data_ptr());
    }

    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::AddBias";
    }

    epilogue::Args make_epilogue_operator_args(const uint32_t& n) const override { return args; }
};

class BF16StochasticRoundingEpilogue final : public EpilogueClass {
public:
    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::StochasticRoundToBF16";
    }

    void check(const std::optional<torch::Tensor>& c, const torch::Tensor& d) const override {
        DJ_HOST_ASSERT(not c.has_value() and d.scalar_type() == torch::kBFloat16,
                       "Stochastic rounding requires a direct BF16 output");
    }
};

class CastToFP8Epilogue : public EpilogueClass {
    torch::Tensor sfd;

public:
    explicit CastToFP8Epilogue(const torch::Tensor& sfd) : sfd(sfd) {}

    epilogue::Args make_epilogue_operator_args(const uint32_t& n) const override {
        return {.sfd = reinterpret_cast<uintptr_t>(sfd.data_ptr()),
                .sfd_stride = static_cast<uint64_t>(sfd.stride(-1)), .shape_n = n};
    }

    std::optional<torch::Tensor> get_output_sf() const override { return sfd; }

    void check(const std::optional<torch::Tensor>& c, const torch::Tensor& d) const override {
        DJ_HOST_ASSERT(not c.has_value() and d.scalar_type() == torch::kFloat8_e4m3fn,
                       "FP8 output requires a direct E4M3 output");
        DJ_HOST_ASSERT(d.dim() == 2 or d.dim() == 3);
        const auto m = d.size(-2), n = d.size(-1);
        const auto num_batches = d.dim() == 3 ? d.size(0) : 1;
        DJ_HOST_ASSERT(n % (2 * MX_SF_DIVISOR) == 0, "FP8 output requires N to be a multiple of 128");
        DJ_HOST_ASSERT(d.stride(-1) == 1 and (d.dim() == 2 or d.is_contiguous() or d.stride(0) == n));
        DJ_HOST_ASSERT(sfd.scalar_type() == torch::kShort and sfd.dim() == 2 and sfd.device() == d.device());
        DJ_HOST_ASSERT(sfd.size(0) == m and sfd.size(1) == num_batches * n / MX_SF_DIVISOR);
        DJ_HOST_ASSERT((sfd.stride(0) == 1 or m == 1) and sfd.stride(1) == m);
    }
};

class FP8QuantizationEpilogue final : public CastToFP8Epilogue {
public:
    using CastToFP8Epilogue::CastToFP8Epilogue;

    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::QuantizeToFP8";
    }
};

class FP8StochasticRoundingEpilogue final : public CastToFP8Epilogue {
public:
    using CastToFP8Epilogue::CastToFP8Epilogue;

    std::string get_epilogue_operator_type() const override {
        return "epilogue::operators::StochasticRoundToFP8";
    }
};

// Resolve the existing API inputs into one validated epilogue class.
static std::shared_ptr<EpilogueClass> resolve_epilogue_class(
    const std::shared_ptr<EpilogueClass>& epilogue_class,
    const std::optional<torch::Tensor>& c,
    const torch::Tensor& d,
    const std::optional<float>& alpha = std::nullopt,
    const std::optional<torch::Tensor>& sfd = std::nullopt
) {
    DJ_HOST_ASSERT((epilogue_class != nullptr) + alpha.has_value() + sfd.has_value() <= 1,
                   "The epilogue class, alpha and the FP8 output pair are exclusive");
    std::shared_ptr<EpilogueClass> resolved = epilogue_class;
    if (alpha.has_value())
        resolved = std::make_shared<AlphaEpilogue>(alpha.value());
    else if (sfd.has_value())
        resolved = std::make_shared<FP8QuantizationEpilogue>(sfd.value());
    else if (resolved == nullptr)
        resolved = std::make_shared<IdentityEpilogue>();
    resolved->check(c, d);
    return resolved;
}

} // namespace deep_gemm
