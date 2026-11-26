#pragma once

#include <algorithm>
#include <cstdio>
#include <limits>
#include <optional>
#include <utility>

#include <deep_gemm/layout/mega_gate.hpp>
#include <torch_npu/csrc/core/npu/NPUGuard.h>

#include "../jit/jit.hpp"
#include "heuristic_mega_gate.hpp"

namespace deep_gemm {

struct MegaGateArgs {
    const GemmConfig& config;

    void* x;
    void* weight;
    mega_gate::layout::GateArgs gate;
    uint32_t num_tokens, num_experts, hidden;
    uint32_t num_topk, num_shared_experts, num_k_splits, num_fixed_rows;
    bool has_bias, has_image_bias, has_mask, has_fixed_routing;
    bool has_physical_map, has_unmapped_output, has_random_routing;
};

static void launch_mega_gate(const MegaGateArgs& args) {
    const auto& config = args.config;
    const auto& gate = args.gate;
    const bool has_bias = args.has_bias;
    const bool has_image_bias = args.has_image_bias;
    const bool has_mask = args.has_mask;
    const bool has_fixed_routing = args.has_fixed_routing;
    const bool has_physical_map = args.has_physical_map;
    const bool has_unmapped_output = args.has_unmapped_output;
    const bool has_random_routing = args.has_random_routing;
    const uint32_t num_ai_cores = config.launch_options.num_blocks.value();
    const uint32_t num_tasks_per_row = ceil_div(args.num_experts, config.block_n) * args.num_k_splits;
    const bool pipeline = uint64_t(ceil_div(args.num_tokens, config.block_m)) * num_tasks_per_row > num_ai_cores;
    const uint32_t num_wave_tokens = std::min(args.num_tokens, ceil_div(num_ai_cores, num_tasks_per_row) * config.block_m);
    const uint32_t num_rows = args.num_fixed_rows ? args.num_fixed_rows :
        std::min(64u / args.num_k_splits, ceil_div(num_wave_tokens, 2 * num_ai_cores));

    // Prefer one wide MAD_N tile (fewer FIX output tiles) when L0B can still double-buffer it.
    // MAD_N only groups independent output columns, so per-element K accumulation stays bitwise identical.
    uint32_t mad_n = config.mad_n;
    const uint32_t wide_mad_n = config.block_n <= 192u ? config.block_n :
                                (config.block_n % 192u == 0 ? 192u : config.mad_n);
    if (wide_mad_n != config.mad_n and config.block_n % wide_mad_n == 0 and
        uint64_t(config.num_l0_stages) * wide_mad_n * config.mad_k * sizeof(uint16_t) <= L0BSizeBytes)
        mad_n = wide_mad_n;

    // Every VF is compiled with zero stack; the remaining UB can cache the physical map.
    const uint32_t num_vector_bytes = get_hw_desc().vector_reg_bytes;
    const mega_gate::layout::UBLayout cached_layout{
        args.num_experts, args.num_shared_experts, num_rows, args.num_k_splits,
        gate.num_duplicate_experts, has_image_bias, has_mask, has_fixed_routing,
        has_physical_map, has_unmapped_output, has_random_routing, num_vector_bytes, args.num_topk};
    const uint32_t num_cached_duplicates = cached_layout.num_bytes <= UBSizeBytes ? gate.num_duplicate_experts : 0;
    const mega_gate::layout::UBLayout ub_layout{
        args.num_experts, args.num_shared_experts, num_rows, args.num_k_splits,
        num_cached_duplicates, has_image_bias, has_mask, has_fixed_routing,
        has_physical_map, has_unmapped_output, has_random_routing, num_vector_bytes, args.num_topk};
    const uint64_t num_ubuf_bytes = mega_gate::layout::fix_staging_bytes(
        config.block_m, config.block_n, config.mad_m, mad_n, ub_layout.num_bytes, pipeline);
    DJ_HOST_ASSERT(num_ubuf_bytes <= UBSizeBytes);

    const auto code = std::format(R"(
#include <deep_gemm/mega_gate.hpp>

static void __instantiate() {{
    auto p = reinterpret_cast<void*>(&deep_gemm::mega_gate::mega_gate_impl<
        {}, {},
        {}, {}, {},
        {}, {}, {},
        {}, {}, {},
        {}, {},
        {}, {}, {},
        {}, {}, {}, {}, {}, {}, {},
        {}, {}, {}>);
}}

)",
        args.num_experts, args.hidden / args.num_k_splits,
        config.block_m, config.block_n, config.block_k,
        config.mad_m, mad_n, config.mad_k,
        config.num_l1_stages, config.num_l0_stages, num_ai_cores,
        config.l2_ctrl_a, config.l2_ctrl_b,
        args.num_topk, args.num_shared_experts, num_cached_duplicates,
        has_bias, has_image_bias, has_mask, has_fixed_routing,
        has_physical_map, has_unmapped_output, has_random_routing, num_rows, args.num_k_splits, pipeline);
    // Bisheng 15 VF-call loop specialization allocates a stack for the single-row gate.
    const deep_jit::Ascend::CompilerOptions compiler_options{
        .extra_bisheng_flags = {"-fno-fast-math", "-mllvm", "-cce-loop-vfcall-specialization=false"}};
    if (runtime->get_dry_run()) {
        jit->compile_without_load("mega_gate", code, compiler_options);
        return;
    }
    const auto kernel = jit->compile("mega_gate", code, compiler_options);

    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS"))
        printf("%s\n", std::format("Mega gate: M={}, N={}, K={}, block={}/{}/{}, splits={}, cores={}, UB={} bytes",
               args.num_tokens, args.num_experts, args.hidden, config.block_m, config.block_n,
               config.block_k, args.num_k_splits, num_ai_cores, ub_layout.num_bytes).c_str());

    auto options = jit->default_launch_options.override_with(config.launch_options);
    options.num_ubuf_bytes = num_ubuf_bytes;
    std::optional<c10_npu::NPUStreamGuard> stream_guard;
    if (options.stream)
        stream_guard.emplace(c10_npu::getStreamFromExternal(*options.stream, c10_npu::getCurrentNPUStream().device_index()).unwrap());
    const size_t num_scratch_bytes = size_t(args.num_tokens) * args.num_k_splits * ub_layout.stride * sizeof(float);
    const auto scratch = c10::GetAllocator(c10::DeviceType::PrivateUse1)->allocate(num_scratch_bytes);
    const mega_gate::layout::KernelArgs<uint16_t> kernel_args{
        gm_ptr<uint16_t>{args.hidden, reinterpret_cast<uintptr_t>(args.x), args.hidden / args.num_k_splits},
        gm_ptr<uint16_t>{args.hidden, reinterpret_cast<uintptr_t>(args.weight), args.hidden / args.num_k_splits},
        gm_ptr<float>{ub_layout.stride * args.num_k_splits, reinterpret_cast<uintptr_t>(scratch.get()), ub_layout.stride},
        args.num_tokens, gate};
    launch_replicated(kernel, options, kernel_args);
}

} // namespace deep_gemm
