#pragma once

#include <pybind11/pybind11.h>

#include "../experimental/v41f_qkv_lora.hpp"
#include "../experimental/v41f_wv_o_proj.hpp"

namespace deep_gemm::experimental {

static void register_apis(pybind11::module_& m) {
    auto submodule = m.def_submodule("experimental", "Experimental kernels without stability guarantees");

    submodule.def("v41f_qkv_lora", &v41f_qkv_lora, pybind11::arg("x"), pybind11::arg("qkv_weight"), pybind11::arg("q_b_weight"),
                  pybind11::arg("q_norm_weight"), pybind11::arg("kv_norm_weight"), pybind11::arg("positions"), pybind11::arg("cos_sin"),
                  pybind11::arg("slots"), pybind11::arg("q"), pybind11::arg("cache"), pybind11::arg("eps"));
    submodule.def("v41f_wv_o_proj", &v41f_wv_o_proj, pybind11::arg("x"), pybind11::arg("wv_weight"),
                  pybind11::arg("o_weight"), pybind11::arg("y"));
}

} // namespace deep_gemm::experimental
