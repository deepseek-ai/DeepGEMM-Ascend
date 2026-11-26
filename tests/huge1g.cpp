#include <cstdio>
#include <exception>
#include <memory>
#include <acl/acl.h>
#include <c10/core/DeviceGuard.h>
#include <torch/extension.h>
#include <torch_npu/csrc/aten/common/from_blob.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>

static at::Tensor empty_huge1g(int64_t num_bytes, int device_idx) {
    TORCH_CHECK(num_bytes > 0, "num_bytes must be positive");
    const c10::Device device(c10::DeviceType::PrivateUse1, device_idx);
    const c10::DeviceGuard guard(device);
    void* pointer;
    const auto status = aclrtMalloc(&pointer, num_bytes, ACL_MEM_MALLOC_HUGE1G_ONLY);
    TORCH_CHECK(status == ACL_SUCCESS, "HUGE1G allocation failed: ", status);
    auto owner = std::shared_ptr<void>(pointer, [device](void* pointer) noexcept {
        try {
            // Test buffers may outlive their allocating stream.
            const c10::DeviceGuard guard(device);
            c10_npu::device_synchronize();
            const auto status = aclrtFree(pointer);
            if (status != ACL_SUCCESS)
                std::fprintf(stderr, "HUGE1G release failed: %d\n", status);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "HUGE1G cleanup failed: %s\n", error.what());
        } catch (...) {
            std::fprintf(stderr, "HUGE1G cleanup failed: unknown exception\n");
        }
    });
    return at_npu::native::from_blob(pointer, {num_bytes}, [owner](void*) {},
                                    at::TensorOptions().dtype(at::kByte).device(device));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("empty_huge1g", &empty_huge1g);
}
