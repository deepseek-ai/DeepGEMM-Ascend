#include <hccl/hccl.h>  // Load SDK declarations before torch_npu's bundled HCCL header.
#include <pybind11/pybind11.h>
#include <torch/python.h>

#include <deep_jit/python_api.hpp>

#include "apis/runtime.hpp"
#include "apis/gemm.hpp"
#include "apis/epilogue_class.hpp"
#include "apis/attention.hpp"
#include "apis/einsum.hpp"
#include "apis/layout.hpp"
#include "apis/aclnn.hpp"
#include "apis/mega_mhc.hpp"
#include "apis/mega_moe.hpp"
#include "apis/mega_gate.hpp"
#include "apis/experimental.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include <torch_npu/csrc/core/npu/NPUStream.h>
#pragma GCC diagnostic pop

aclrtStream deep_gemm::get_current_npu_stream() {
    return c10_npu::getCurrentNPUStream();
}

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

// ReSharper disable once CppParameterMayBeConstPtrOrRef
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepGEMM-Ascend C++ library";

    deep_gemm::config::register_apis(m);
    deep_gemm::epilogue_class::register_apis(m);
    deep_gemm::gemm_api::register_apis(m);
    deep_gemm::attention_api::register_apis(m);
    deep_gemm::einsum::register_apis(m);
    deep_gemm::layout::register_apis(m);
    deep_gemm::aclnn_api::register_apis(m);
    deep_gemm::mega_mhc::register_apis(m);
    deep_gemm::mega_moe::register_apis(m);
    deep_gemm::mega_gate::register_apis(m);
    deep_jit::register_python_api(m, deep_gemm::jit);

    // Register experimental kernels
    deep_gemm::experimental::register_apis(m);
}
