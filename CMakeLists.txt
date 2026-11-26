# NOTES: this CMake is only for debugging/IDE indexing; for setup, please use Torch extension
cmake_minimum_required(VERSION 3.10)
project(deep_gemm_ascend LANGUAGES CXX)
set(CMAKE_VERBOSE_MAKEFILE ON)

# CXX configs
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_STANDARD 20)

# Dependency
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

# NOTES: CLion can not index linked directories
include_directories(
    csrc deep_gemm/include
    third-party/clangd-ascend/include
    third-party/deep-jit/include
    third-party/magic_enum/include
    ${ASCEND_HOME_PATH}/aarch64-linux/include
    ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS}
    ${TORCH_NPU_INSTALL_PREFIX}/include ${TORCH_NPU_INSTALL_PREFIX}/include/third_party/acl/inc
)
link_directories(
    ${ASCEND_HOME_PATH}/lib64
    ${TORCH_INSTALL_PREFIX}/lib
    ${TORCH_NPU_INSTALL_PREFIX}/lib
)

pybind11_add_module(_C csrc/python_api.cpp)
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} torch_python ascendcl hccl_fwk hccl hcomm torch_npu opapi dl)
target_link_options(_C PRIVATE "-Wl,-Bsymbolic-functions")

# CLion cannot discover device sources that are only included by generated JIT code.
# Keep a non-production target that gives those sources a real translation unit and
# points the IDE at CANN's physical include directories instead of its symlinked ones.
add_executable(indexing EXCLUDE_FROM_ALL csrc/indexing/main.cpp)
target_include_directories(indexing PRIVATE
    ${ASCEND_HOME_PATH}/aarch64-linux/include/ascendc/host_apiascendc/include/highlevel_api
    ${ASCEND_HOME_PATH}/aarch64-linux/tikcpp/tikcfw
    ${ASCEND_HOME_PATH}/aarch64-linux/tikcpp/tikcfw/lib
    ${ASCEND_HOME_PATH}/aarch64-linux/tikcpp/tikcfw/lib/matmul
    ${ASCEND_HOME_PATH}/aarch64-linux/tikcpp/tikcfw/impl
    ${ASCEND_HOME_PATH}/aarch64-linux/tikcpp/tikcfw/interface
    ${ASCEND_HOME_PATH}/aarch64-linux/asc
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/impl/adv_api
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/impl/basic_api
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/impl/c_api
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/impl/micro_api
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/impl/simt_api
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/impl/utils
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api/matmul
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/aicpu_api
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/basic_api
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/c_api
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/interface
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/micro_api
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/simt_api
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/tiling
    ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/utils
    ${ASCEND_HOME_PATH}/tools/bisheng_compiler/lib/clang/15.0.5/include
)
